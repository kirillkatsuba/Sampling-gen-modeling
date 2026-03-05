import os
import functools
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, HMC, NUTS
import arviz as az
import jax

# CRITICAL for MCMC / linear probability: Use 64-bit precision to delay underflow
jax.config.update("jax_enable_x64", True)

try:
    from scipy.stats import wasserstein_distance_nd
    HAS_WASSERSTEIN = True
except ImportError:
    HAS_WASSERSTEIN = False


# ==========================================
# 1. JAX Target Distributions & Exact Samplers
# ==========================================

@jax.jit
def pi_banana(x, nu=10.0):
    x = jnp.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    exponent = -(((x1 - x2**2)**2) / nu) - ((x2 - 1)**2)
    res = jnp.exp(exponent)
    return jnp.squeeze(res)

@functools.partial(jax.jit, static_argnames=['num_samples'])
def sample_exact_banana(rng_key, num_samples, nu=10.0):
    k1, k2 = jax.random.split(rng_key)
    x2_samples = jax.random.normal(k1, (num_samples,)) * jnp.sqrt(0.5) + 1.0
    x1_samples = jax.random.normal(k2, (num_samples,)) * jnp.sqrt(nu/2.0) + x2_samples**2
    return jnp.column_stack((x1_samples, x2_samples))

def get_pi_gmm_jax(weights, means, covs):
    w, m, c = jnp.array(weights), jnp.array(means), jnp.array(covs)
    
    @jax.jit
    def pdf_fn(x):
        x = jnp.atleast_2d(x)
        def comp_pdf(mean, cov):
            return jax.scipy.stats.multivariate_normal.pdf(x, mean, cov)
        probs = jax.vmap(comp_pdf)(m, c)
        res = jnp.dot(w, probs)
        return jnp.squeeze(res)
    return pdf_fn

@functools.partial(jax.jit, static_argnames=['num_samples'])
def sample_exact_gmm(rng_key, num_samples, weights, means, covs):
    k1, k2 = jax.random.split(rng_key)
    w, m, c = jnp.array(weights), jnp.array(means), jnp.array(covs)
    
    comps = jax.random.choice(k1, jnp.arange(len(w)), p=w, shape=(num_samples,))
    z = jax.random.normal(k2, (num_samples, m.shape[-1]))
    
    chol = jnp.linalg.cholesky(c)
    m_c = m[comps]
    chol_c = chol[comps]
    
    samples = m_c + jnp.einsum('nij,nj->ni', chol_c, z)
    return samples

# ==========================================
# 2. Metrics 
# ==========================================

def calculate_ess(samples):
    samples_az = np.expand_dims(np.asarray(samples), axis=0) 
    ess = az.ess(az.dict_to_dataset({"x": samples_az}))["x"].values
    return np.mean(ess)

def fast_ess(samples):
    """Computes Effective Sample Size using Fast Fourier Transform (FFT)."""
    samples = np.asarray(samples)
    N, d = samples.shape
    ess_vals = []
    
    for i in range(d):
        x = samples[:, i]
        x = x - np.mean(x)
        
        # Compute autocorrelation via FFT
        fft_x = np.fft.fft(x, n=2*N)
        acf = np.fft.ifft(fft_x * np.conj(fft_x))[:N].real
        acf /= acf[0] # Normalize
        
        # Find the first negative autocorrelation (standard cutoff heuristic)
        negative_idx = np.where(acf < 0)[0]
        cutoff = negative_idx[0] if len(negative_idx) > 0 else N
        
        tau = 1 + 2 * np.sum(acf[1:cutoff])
        ess_vals.append(N / tau)
        
    return np.mean(ess_vals)

def compute_tv_distance(samples, pdf_fn, limits, bins=100):
    samples = np.asarray(samples)
    x_edges = np.linspace(limits[0][0], limits[0][1], bins + 1)
    y_edges = np.linspace(limits[1][0], limits[1][1], bins + 1)
    
    H, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[x_edges, y_edges])
    empirical_mass = H / np.sum(H)
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    grid_points = jnp.c_[X.ravel(), Y.ravel()]
    true_pdf = np.asarray(pdf_fn(grid_points)).reshape(bins, bins)
    
    sum_true_pdf = np.sum(true_pdf)
    true_mass = true_pdf / sum_true_pdf if sum_true_pdf > 0 else np.zeros_like(true_pdf)
        
    return 0.5 * np.sum(np.abs(empirical_mass - true_mass))

def fast_tv_distance(samples, true_pdf_grid, limits, bins=100):
    """Computes TV distance using a pre-computed True PDF grid to save time."""
    samples = np.asarray(samples)
    x_edges = np.linspace(limits[0][0], limits[0][1], bins + 1)
    y_edges = np.linspace(limits[1][0], limits[1][1], bins + 1)
    
    H, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[x_edges, y_edges])
    empirical_mass = H / np.sum(H)
    
    sum_true = np.sum(true_pdf_grid)
    true_mass = true_pdf_grid / sum_true if sum_true > 0 else np.zeros_like(true_pdf_grid)
        
    return 0.5 * np.sum(np.abs(empirical_mass - true_mass))



def compute_emd(samples, true_samples, subsample_size=1500):
    if not HAS_WASSERSTEIN: return np.nan
    samples, true_samples = np.asarray(samples), np.asarray(true_samples)
    idx_s = np.random.choice(len(samples), min(len(samples), subsample_size), replace=False)
    idx_t = np.random.choice(len(true_samples), min(len(true_samples), subsample_size), replace=False)
    return wasserstein_distance_nd(samples[idx_s], true_samples[idx_t])

def strict_fast_swd(samples, true_samples, n_projections=500):
    """
    Computes a strict Sliced Wasserstein Distance (W_2 metric).
    Uses 500 projections and squares the errors to heavily punish mode collapse.
    """
    samples = np.asarray(samples)
    true_samples = np.asarray(true_samples)
    
    # 1. Generate many random 2D directions (500 instead of 50)
    theta = np.random.uniform(0, 2 * np.pi, n_projections)
    dirs = np.vstack((np.cos(theta), np.sin(theta))).T
    
    # 2. Project samples onto lines
    proj_samples = samples @ dirs.T
    proj_true = true_samples @ dirs.T
    
    # 3. Sort (This is how 1D Wasserstein is solved instantly)
    proj_samples = np.sort(proj_samples, axis=0)
    proj_true = np.sort(proj_true, axis=0)
    
    # Subsample to equal lengths for array subtraction
    min_len = min(len(samples), len(true_samples))
    idx_s = np.linspace(0, len(samples)-1, min_len, dtype=int)
    idx_t = np.linspace(0, len(true_samples)-1, min_len, dtype=int)
    
    # 4. HARSH PENALTY: Use Squared Difference (W_2) instead of Absolute (W_1)
    # This acts like Mean Squared Error, exploding when a mode is missed entirely.
    squared_diff = (proj_samples[idx_s] - proj_true[idx_t]) ** 2
    
    # Mean across points, then mean across projections, then square root
    w2_distance = np.sqrt(np.mean(squared_diff))
    
    return w2_distance

# ==========================================
# 3. Algorithms & Proposals (Refactored for JAX)
# ==========================================

def get_t_proposal_fns(scale_factor=5.0, df=3.0):
    """Returns pure JAX functions for sampling and evaluating proposal PDF."""
    loc = jnp.zeros(2)
    scale = jnp.eye(2) * scale_factor
    chol = jnp.linalg.cholesky(scale)
    
    def sample_fn(rng_key, size):
        return dist.MultivariateStudentT(df, loc=loc, scale_tril=chol).sample(rng_key, (size,))
    
    def pdf_fn(x):
        return jnp.exp(dist.MultivariateStudentT(df, loc=loc, scale_tril=chol).log_prob(x))
        
    return sample_fn, pdf_fn

# Note: Added ALL function arguments to static_argnames so JIT accepts them
@functools.partial(jax.jit, static_argnames=['target_fn', 'prop_sample_fn', 'prop_pdf_fn', 'num_samples', 'N'])
def run_isir_jax(rng_key, target_fn, prop_sample_fn, prop_pdf_fn, x0, num_samples, N=15):
    
    def isir_step(carry, key):
        x_curr = carry
        k1, k2 = jax.random.split(key)
        
        # 1. Propose
        Y_prop = prop_sample_fn(k1, N - 1)
        Y_prop = jnp.atleast_2d(Y_prop)
        Y = jnp.vstack([x_curr, Y_prop])
        
        # 2. Evaluate
        pi_Y = target_fn(Y)
        q_Y = prop_pdf_fn(Y)
        
        # 3. Weight
        w = pi_Y / (q_Y + 1e-12)
        w_sum = jnp.sum(w)
        
        # 4. Normalize Safely
        W = jax.lax.cond(
            w_sum == 0,
            lambda _: jnp.ones(N) / N,
            lambda _: w / w_sum,
            operand=None
        )
        
        # 5. Resample
        idx = jax.random.choice(k2, jnp.arange(N), p=W)
        x_next = Y[idx]
        
        return x_next, x_next

    keys = jax.random.split(rng_key, num_samples - 1)
    _, samples = jax.lax.scan(isir_step, jnp.array(x0), keys)
    
    return jnp.vstack([jnp.array(x0), samples])

def get_banana_potential_jax(nu):
    def potential(z):
        x = z['x']
        prob = jnp.exp(-(((x[0] - x[1]**2)**2) / nu) - ((x[1] - 1.0)**2))
        return -jnp.log(prob + 1e-12)
    return potential

def get_gmm_potential_jax(weights, means, covs):
    w, m, c = jnp.array(weights), jnp.array(means), jnp.array(covs)
    def potential(z):
        prob = 0.0
        for k in range(len(w)):
            prob += w[k] * jax.scipy.stats.multivariate_normal.pdf(z['x'], m[k], c[k])
        return -jnp.log(prob + 1e-12)
    return potential

def run_numpyro_sampler(rng_key, potential_fn, init_params, num_samples, sampler_type='NUTS'):
    kernel = NUTS(potential_fn=potential_fn) if sampler_type == 'NUTS' else HMC(potential_fn=potential_fn, adapt_step_size=True)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, progress_bar=False)
    mcmc.run(rng_key, init_params={'x': jnp.array(init_params)})
    return np.array(mcmc.get_samples()['x'])

# ==========================================
# 4. Plotting
# ==========================================

def plot_results(samples_dict, target_density_fn, true_samples, title, limits):
    os.makedirs("plots", exist_ok=True)

    x, y = np.linspace(limits[0][0], limits[0][1], 100), np.linspace(limits[1][0], limits[1][1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate True PDF EXACTLY ONCE for the whole figure
    grid_points = jnp.c_[X.ravel(), Y.ravel()]
    Z = np.asarray(target_density_fn(grid_points)).reshape(100, 100)
            
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16)
    
    for ax, (name, samples) in zip(axes, samples_dict.items()):
        samples = np.asarray(samples) 
        
        # ---> USING THE NEW FAST METRICS <---
        start_time = time.time()
        ess = fast_ess(samples)
        print(f'ESS is calculated in {- start_time + time.time()} sec')

        start_time = time.time()
        tv = fast_tv_distance(samples, Z, limits) # Pass Z instead of the function!
        print(f'TV is calculated in {- start_time + time.time()} sec')
        
        start_time = time.time()
        emd = strict_fast_swd(samples, true_samples)
        print(f'EMD is calculated in {- start_time + time.time()} sec')

        ax_title = f"{name}\nESS: {ess:.1f} | TV: {tv:.3f}\nSWD: {emd:.3f}"
        ax.contour(X, Y, Z, levels=15, cmap='Blues', alpha=0.8)
        ax.scatter(samples[::10, 0], samples[::10, 1], s=5, alpha=0.3, color='red')
        ax.set_title(ax_title)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        
    plt.tight_layout()
    plt.savefig(f'plots/{title}.png') 
    plt.close(fig)

# def plot_results(samples_dict, target_density_fn, true_samples, title, limits):
#     os.makedirs("plots", exist_ok=True)

#     x, y = np.linspace(limits[0][0], limits[0][1], 100), np.linspace(limits[1][0], limits[1][1], 100)
#     X, Y = np.meshgrid(x, y)
    
#     grid_points = jnp.c_[X.ravel(), Y.ravel()]
#     Z = np.asarray(target_density_fn(grid_points)).reshape(100, 100)
            
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#     fig.suptitle(title, fontsize=16)
    
#     for ax, (name, samples) in zip(axes, samples_dict.items()):
#         samples = np.asarray(samples)
#         start_time = time.time()
#         ess = fast_ess(samples)
#         print(f'ESS is calculated in {start_time - time.time()} sec')

#         start_time = time.time()
#         tv = fast_tv_distance(samples, target_density_fn, limits)
#         print(f'ESS is calculated in {start_time - time.time()} sec')

#         start_time = time.time()
#         emd = strict_fast_swd(samples, true_samples)
#         print(f'EMD (wasserstein distance) is calculated in {start_time - time.time()} sec')

#         ax_title = f"{name}\nESS: {ess:.1f} | TV: {tv:.3f}\nEMD: {emd:.3f}"
#         ax.contour(X, Y, Z, levels=15, cmap='Blues', alpha=0.8)
#         ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, color='red')
#         ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, color='red')
#         ax.set_title(ax_title)
#         ax.set_xlim(limits[0])
#         ax.set_ylim(limits[1])
        
#     plt.tight_layout()
#     plt.savefig(f'plots/{title}.png') 
#     plt.close(fig) # Closes plot so it doesn't interrupt the loop!

if __name__ == "__main__":
    NUM_SAMPLES = 10_000
    I_SIR_PARTICLES = 15
    X0 = jnp.array([0.0, 0.0])
    
    rng_key = jax.random.PRNGKey(42)
    
    print("Running Banana Distribution Ablations...")
    for nu in tqdm([0.1, 1.0, 2.0, 5.0, 12.0]):
        rng_key, k1, k2, k3, k4, k5 = jax.random.split(rng_key, 6)
        
        target_fn = lambda x: pi_banana(x, nu=nu)
        true_samples = sample_exact_banana(k1, 1_000, nu=nu)
        
        cauchy_sample, cauchy_pdf = get_t_proposal_fns(scale_factor=10.0, df=1.0)
        student_sample, student_pdf = get_t_proposal_fns(scale_factor=10.0, df=3.0)
        
        s_cauchy = run_isir_jax(k2, target_fn, cauchy_sample, cauchy_pdf, X0, NUM_SAMPLES, I_SIR_PARTICLES)
        s_student = run_isir_jax(k3, target_fn, student_sample, student_pdf, X0, NUM_SAMPLES, I_SIR_PARTICLES)
        s_hmc = run_numpyro_sampler(k4, get_banana_potential_jax(nu), X0, NUM_SAMPLES, 'HMC')
        s_nuts = run_numpyro_sampler(k5, get_banana_potential_jax(nu), X0, NUM_SAMPLES, 'NUTS')
        
        results = {"I-SIR (Cauchy)": s_cauchy, "I-SIR (Student)": s_student, "HMC": s_hmc, "NUTS": s_nuts}
        plot_results(results, target_fn, true_samples, f"Banana Target (nu={nu})", limits=([-15, 15], [-5, 5]))


    print("\nRunning Gaussian Mixture Ablations...")
    cov_base1 = np.array([[1.0,  0.6], [0.6,  1.0]])
    cov_base2 = np.array([[1.0, -0.6], [-0.6,  1.0]])
    cov_base3 = np.array([[0.8,  0.0], [0.0,  0.8]])

    configs = {
        "Balanced (Close)": {"weights": [0.33, 0.33, 0.34], "means": [[-2, -2], [2, 2], [0, 0]], "covs": [cov_base1, cov_base2, cov_base3]},
        "Imbalanced (Wide)": {"weights": [0.7, 0.2, 0.1], "means": [[-5, -5], [5, 5], [0, 0]], "covs": [cov_base1*0.5, cov_base2*2, cov_base3]}
    }
    
    for name, config in tqdm(configs.items()):
        rng_key, k1, k2, k3, k4, k5 = jax.random.split(rng_key, 6)
        
        target_fn = get_pi_gmm_jax(**config)
        true_samples = sample_exact_gmm(k1, 10_000, **config)
        
        cauchy_sample, cauchy_pdf = get_t_proposal_fns(scale_factor=15.0, df=1.0)
        student_sample, student_pdf = get_t_proposal_fns(scale_factor=15.0, df=3.0)
        
        s_cauchy = run_isir_jax(k2, target_fn, cauchy_sample, cauchy_pdf, X0, NUM_SAMPLES, I_SIR_PARTICLES)
        s_student = run_isir_jax(k3, target_fn, student_sample, student_pdf, X0, NUM_SAMPLES, I_SIR_PARTICLES)
        s_hmc = run_numpyro_sampler(k4, get_gmm_potential_jax(**config), X0, NUM_SAMPLES, 'HMC')
        s_nuts = run_numpyro_sampler(k5, get_gmm_potential_jax(**config), X0, NUM_SAMPLES, 'NUTS')
        
        results = {"I-SIR (Cauchy)": s_cauchy, "I-SIR (Student)": s_student, "HMC": s_hmc, "NUTS": s_nuts}
        plot_results(results, target_fn, true_samples, f"GMM_{name.replace(' ', '_')}", limits=([-10, 10], [-10, 10]))