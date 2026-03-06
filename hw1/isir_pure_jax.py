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
import csv 

jax.config.update("jax_enable_x64", True)

try:
    from scipy.stats import wasserstein_distance_nd
    HAS_WASSERSTEIN = True
except ImportError:
    HAS_WASSERSTEIN = False


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


def generate_gmm_config(K, mode="balanced", radius=4.0):
    """
    Dynamically generates weights, means, and covariances for K components.
    Places the means evenly in a circle around the origin.
    """
    # 1. Means (arranged in a circle)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    means = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    
    # 2. Weights
    if mode == "balanced":
        weights = np.ones(K) / K
    elif mode == "imbalanced":
        weights = np.zeros(K)
        weights[0] = 0.7
        if K > 1:
            weights[1:] = 0.3 / (K - 1)
            
    # 3. Covariances (cycle through 3 distinct shapes to make it interesting)
    cov_base1 = np.array([[1.0, 0.6], [0.6, 1.0]])   # Tilted right
    cov_base2 = np.array([[1.0, -0.6], [-0.6, 1.0]]) # Tilted left
    cov_base3 = np.array([[0.8, 0.0], [0.0, 0.8]])   # Circle
    bases = [cov_base1, cov_base2, cov_base3]
    
    covs = [bases[i % 3] for i in range(K)]
    
    return {
        "weights": weights.tolist(),
        "means": means.tolist(),
        "covs": covs
    }

# ==========================================
# 2. Metrics 
# ==========================================

def fast_ess(samples):
    samples = np.asarray(samples)
    N, d = samples.shape
    ess_vals = []
    for i in range(d):
        x = samples[:, i]
        x = x - np.mean(x)
        fft_x = np.fft.fft(x, n=2*N)
        acf = np.fft.ifft(fft_x * np.conj(fft_x))[:N].real
        acf /= acf[0]
        negative_idx = np.where(acf < 0)[0]
        cutoff = negative_idx[0] if len(negative_idx) > 0 else N
        tau = 1 + 2 * np.sum(acf[1:cutoff])
        ess_vals.append(N / max(tau, 1.0))
    return np.mean(ess_vals)

def compute_tv_distance(samples, pdf_fn, limits, bins=40):
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

def strict_fast_swd(samples, true_samples, n_projections=500):
    samples, true_samples = np.asarray(samples), np.asarray(true_samples)
    theta = np.random.uniform(0, 2 * np.pi, n_projections)
    dirs = np.vstack((np.cos(theta), np.sin(theta))).T
    
    proj_samples = np.sort(samples @ dirs.T, axis=0)
    proj_true = np.sort(true_samples @ dirs.T, axis=0)
    
    min_len = min(len(samples), len(true_samples))
    idx_s = np.linspace(0, len(samples)-1, min_len, dtype=int)
    idx_t = np.linspace(0, len(true_samples)-1, min_len, dtype=int)
    
    squared_diff = (proj_samples[idx_s] - proj_true[idx_t]) ** 2
    return np.sqrt(np.mean(squared_diff))


def get_t_proposal_fns(scale_factor=5.0, df=3.0):
    loc = jnp.zeros(2)
    scale = jnp.eye(2) * scale_factor
    chol = jnp.linalg.cholesky(scale)
    
    def sample_fn(rng_key, size):
        return dist.MultivariateStudentT(df, loc=loc, scale_tril=chol).sample(rng_key, (size,))
    def pdf_fn(x):
        return jnp.exp(dist.MultivariateStudentT(df, loc=loc, scale_tril=chol).log_prob(x))
    return sample_fn, pdf_fn

@functools.partial(jax.jit, static_argnames=['target_fn', 'prop_sample_fn', 'prop_pdf_fn', 'num_samples', 'N'])
def run_isir_jax(rng_key, target_fn, prop_sample_fn, prop_pdf_fn, x0, num_samples, N=15):
    def isir_step(carry, key):
        x_curr = carry
        k1, k2 = jax.random.split(key)
        
        Y_prop = prop_sample_fn(k1, N - 1)
        Y_prop = jnp.atleast_2d(Y_prop)
        Y = jnp.vstack([x_curr, Y_prop])
        
        pi_Y = target_fn(Y)
        q_Y = prop_pdf_fn(Y)
        w = pi_Y / (q_Y + 1e-12)
        w_sum = jnp.sum(w)
        
        W = jax.lax.cond(w_sum == 0, lambda _: jnp.ones(N) / N, lambda _: w / w_sum, operand=None)
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
        def comp_pdf(mean, cov):
            return jax.scipy.stats.multivariate_normal.pdf(z['x'], mean, cov)
        probs = jax.vmap(comp_pdf)(m, c)
        prob = jnp.dot(w, probs)
        return -jnp.log(prob + 1e-12)
    return potential

def run_numpyro_sampler(rng_key, potential_fn, init_params, num_samples, sampler_type='NUTS'):
    kernel = NUTS(potential_fn=potential_fn) if sampler_type == 'NUTS' else HMC(potential_fn=potential_fn, adapt_step_size=True)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, progress_bar=False)
    mcmc.run(rng_key, init_params={'x': jnp.array(init_params)})
    return np.array(mcmc.get_samples()['x'])


def plot_results(samples_dict, target_density_fn, true_samples, title, logs=None, target_type="", param_str=""):
    os.makedirs("plots_new", exist_ok=True)
    
    ts = np.asarray(true_samples)
    min_x, max_x = ts[:, 0].min(), ts[:, 0].max()
    min_y, max_y = ts[:, 1].min(), ts[:, 1].max()
    
    pad_x = (max_x - min_x) * 0.2
    pad_y = (max_y - min_y) * 0.2
    
    dyn_xlim = [min_x - pad_x, max_x + pad_x]
    dyn_ylim = [min_y - pad_y, max_y + pad_y]
    
    resolution = 300
    x = np.linspace(dyn_xlim[0], dyn_xlim[1], resolution)
    y = np.linspace(dyn_ylim[0], dyn_ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    grid_points = jnp.c_[X.ravel(), Y.ravel()]
    Z = np.asarray(target_density_fn(grid_points)).reshape(resolution, resolution)
            
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16)
    
    for ax, (name, samples) in zip(axes, samples_dict.items()):
        samples = np.asarray(samples) 
        
        ess = fast_ess(samples)
        tv = compute_tv_distance(samples, target_density_fn, (dyn_xlim, dyn_ylim), bins=50) 
        emd = strict_fast_swd(samples, true_samples)

        if logs is not None:
            logs.append({
                "Target": target_type,
                "Configuration": param_str,
                "Algorithm": name,
                "ESS": float(ess),
                "TV": float(tv),
                "SWD": float(emd)
            })

        ax_title = f"{name}\nESS: {ess:.1f} | TV: {tv:.3f}\nSWD: {emd:.3f}"
        
        ax.imshow(Z, 
                  extent=[dyn_xlim[0], dyn_xlim[1], dyn_ylim[0], dyn_ylim[1]], 
                  origin='lower', 
                  cmap='viridis',
                  aspect='auto', 
                  alpha=0.6)
        
        ax.scatter(samples[1::100, 0], samples[1::100, 1], s=5, c='red', alpha=0.4)
        
        ax.set_title(ax_title)
        
        ax.set_xlim(dyn_xlim)
        ax.set_ylim(dyn_ylim)
        
    plt.tight_layout()
    plt.savefig(f'plots_new/{title}.png') 
    plt.close(fig)

if __name__ == "__main__":
    NUM_SAMPLES = 10_000
    X0 = jnp.array([0.0, 0.0])
    rng_key = jax.random.PRNGKey(42)
    metric_logs = []
    
    # --- NEW: Varying I-SIR Particles ---
    particle_counts = [5, 10, 30, 50, 100]
    
    for N_particles in particle_counts:
        print(f"STARTING BENCHMARK WITH I_SIR_PARTICLES = {N_particles}")

        print(f"\nRunning Banana Ablations (N={N_particles})...")
        for nu in tqdm([0.1, 1.0, 2.0, 5.0, 12.0]):
            rng_key, k1, k2, k3, k4, k5 = jax.random.split(rng_key, 6)
            
            target_fn = lambda x: pi_banana(x, nu=nu)
            true_samples = sample_exact_banana(k1, 10_000, nu=nu)
            
            cauchy_sample, cauchy_pdf = get_t_proposal_fns(scale_factor=10.0, df=1.0)
            student_sample, student_pdf = get_t_proposal_fns(scale_factor=10.0, df=3.0)
            
            s_cauchy = run_isir_jax(k2, target_fn, cauchy_sample, cauchy_pdf, X0, NUM_SAMPLES, N_particles)
            s_student = run_isir_jax(k3, target_fn, student_sample, student_pdf, X0, NUM_SAMPLES, N_particles)
            
            s_hmc = run_numpyro_sampler(k4, get_banana_potential_jax(nu), X0, NUM_SAMPLES, 'HMC')
            s_nuts = run_numpyro_sampler(k5, get_banana_potential_jax(nu), X0, NUM_SAMPLES, 'NUTS')
            
            results = {"I-SIR (Cauchy)": s_cauchy, "I-SIR (Student)": s_student, "HMC": s_hmc, "NUTS": s_nuts}
            
            plot_results(results, target_fn, true_samples, 
                         title=f"Banana_nu={nu}_Particles={N_particles}",
                         logs=metric_logs,
                         target_type="Banana",
                         param_str=f"nu={nu}, N={N_particles}")

        print(f"\nRunning GMM Ablations (N={N_particles})...")
        k_values = [3, 5, 8, 15]
        
        for K in k_values:
            configs = {
                f"K={K}_Balanced_(Close)": generate_gmm_config(K, mode="balanced", radius=3.0),
                f"K={K}_Balanced_(Wide)": generate_gmm_config(K, mode="balanced", radius=7.0),
                f"K={K}_Balanced_(The Widest)": generate_gmm_config(K, mode="balanced", radius=15.0),
                f"K={K}_Imbalanced_(Close)": generate_gmm_config(K, mode="imbalanced", radius=3.0),
                f"K={K}_Imbalanced_(Wide)": generate_gmm_config(K, mode="imbalanced", radius=7.0),
                f"K={K}_Imbalanced_(The Widest)": generate_gmm_config(K, mode="imbalanced", radius=15.0),
            }
            
            for name, config in tqdm(configs.items()):
                print(f"--- GMM Target: {name} ---")
                rng_key, k1, k2, k3, k4, k5 = jax.random.split(rng_key, 6)
                
                target_fn = get_pi_gmm_jax(**config)
                true_samples = sample_exact_gmm(k1, 10_000, **config)
                
                cauchy_sample, cauchy_pdf = get_t_proposal_fns(scale_factor=15.0, df=1.0)
                student_sample, student_pdf = get_t_proposal_fns(scale_factor=15.0, df=3.0)
                
                s_cauchy = run_isir_jax(k2, target_fn, cauchy_sample, cauchy_pdf, X0, NUM_SAMPLES, N_particles)
                s_student = run_isir_jax(k3, target_fn, student_sample, student_pdf, X0, NUM_SAMPLES, N_particles)
                
                s_hmc = run_numpyro_sampler(k4, get_gmm_potential_jax(**config), X0, NUM_SAMPLES, 'HMC')
                s_nuts = run_numpyro_sampler(k5, get_gmm_potential_jax(**config), X0, NUM_SAMPLES, 'NUTS')
                
                results = {"I-SIR (Cauchy)": s_cauchy, "I-SIR (Student)": s_student, "HMC": s_hmc, "NUTS": s_nuts}
                
                plot_results(results, target_fn, true_samples, 
                             title=f"GMM_{name.replace(' ', '_')}_Particles={N_particles}",
                             logs=metric_logs,
                             target_type="GMM",
                             param_str=f"{name}, N={N_particles}")
            
    csv_filename = "benchmark_metrics_varied_N_big_test.csv"
    if metric_logs:
        keys = metric_logs[0].keys()
        with open(csv_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(metric_logs)
        print(f"\nAll metrics saved successfully to {csv_filename}")