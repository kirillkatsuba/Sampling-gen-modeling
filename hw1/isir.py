import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as jax_mvn
from tqdm import tqdm
import numpyro
from numpyro.infer import MCMC, HMC, NUTS
import arviz as az

try:
    from scipy.stats import wasserstein_distance_nd
    HAS_WASSERSTEIN = True
except ImportError:
    HAS_WASSERSTEIN = False


def pi_banana(x, nu=10.0):
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    exponent = -(((x1 - x2**2)**2) / nu) - ((x2 - 1)**2)
    res = np.exp(exponent)
    return res[0] if res.size == 1 else res

def pi_gmm(x, weights, means, covs):
    x = np.atleast_2d(x)
    prob = np.zeros(x.shape[0])
    for w, mu, sigma in zip(weights, means, covs):
        # scipy's pdf natively accepts 2D arrays (N_points, Dimensions)
        prob += w * stats.multivariate_normal.pdf(x, mean=mu, cov=sigma)
    return prob[0] if prob.size == 1 else prob

def sample_exact_banana(num_samples, nu=10.0):
    x2_samples = np.random.normal(loc=1.0, scale=np.sqrt(0.5), size=num_samples)
    x1_samples = np.random.normal(loc=x2_samples**2, scale=np.sqrt(nu/2.0), size=num_samples)
    return np.column_stack((x1_samples, x2_samples))

def sample_exact_gmm(num_samples, weights, means, covs):
    components = np.random.choice(len(weights), size=num_samples, p=weights)
    samples = np.zeros((num_samples, 2))
    for i in range(len(weights)):
        mask = (components == i)
        n_i = np.sum(mask)
        if n_i > 0:
            samples[mask] = np.random.multivariate_normal(means[i], covs[i], size=n_i)
    return samples

# ==========================================
# 2. Metrics (Vectorized TV Distance)
# ==========================================

def calculate_ess(samples):
    samples_az = np.expand_dims(samples, axis=0) 
    ess = az.ess(az.dict_to_dataset({"x": samples_az}))["x"].values
    return np.mean(ess)

def compute_tv_distance(samples, pdf_fn, limits, bins=100):
    x_edges = np.linspace(limits[0][0], limits[0][1], bins + 1)
    y_edges = np.linspace(limits[1][0], limits[1][1], bins + 1)
    
    H, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[x_edges, y_edges])
    empirical_mass = H / np.sum(H)
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    grid_points = np.c_[X.ravel(), Y.ravel()]
    
    # FASTER: Evaluate all grid points at once instead of using a list comprehension
    true_pdf = pdf_fn(grid_points).reshape(bins, bins)
    
    sum_true_pdf = np.sum(true_pdf)
    if sum_true_pdf > 0:
        true_mass = true_pdf / sum_true_pdf
    else:
        true_mass = np.zeros_like(true_pdf)
        
    return 0.5 * np.sum(np.abs(empirical_mass - true_mass))

def compute_emd(samples, true_samples, subsample_size=1500):
    if not HAS_WASSERSTEIN: return np.nan
    idx_s = np.random.choice(len(samples), min(len(samples), subsample_size), replace=False)
    idx_t = np.random.choice(len(true_samples), min(len(true_samples), subsample_size), replace=False)
    return wasserstein_distance_nd(samples[idx_s], true_samples[idx_t])

# ==========================================
# 3. Algorithms
# ==========================================

class MultivariateTProposal:
    def __init__(self, loc, scale, df):
        self.loc = np.array(loc)
        self.scale = np.array(scale)
        self.df = df
        self.rv = stats.multivariate_t(loc=self.loc, shape=self.scale, df=self.df)

    def sample(self, size): return self.rv.rvs(size)
    def pdf(self, x): return self.rv.pdf(x)

def get_cauchy_proposal(loc=[0.0, 0.0], scale_factor=5.0):
    return MultivariateTProposal(loc=loc, scale=np.eye(2)*scale_factor, df=1.0)
def get_student_t_proposal(loc=[0.0, 0.0], scale_factor=5.0, df=3.0):
    return MultivariateTProposal(loc=loc, scale=np.eye(2)*scale_factor, df=df)

def run_isir(target_fn, proposal, x0, num_samples, N=10):
    d = len(x0)
    samples = np.zeros((num_samples, d))
    samples[0] = x_curr = x0
    for t in range(1, num_samples):
        Y_prop = proposal.sample(size=N-1)
        if N - 1 == 1: Y_prop = Y_prop.reshape(1, d)
        Y = np.vstack([x_curr, Y_prop])
        
        # FASTER: Vectorized target evaluation. Replaces slow [target_fn(y) for y in Y]
        pi_Y = target_fn(Y)
        q_Y = proposal.pdf(Y)
        
        w = pi_Y / (q_Y + 1e-12)
        
        w_sum = np.sum(w)
        if w_sum == 0:
            W = np.ones(N) / N 
        else:
            W = w / w_sum
            
        x_curr = Y[np.random.choice(N, p=W)]
        samples[t] = x_curr
    return samples


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
            prob += w[k] * jax_mvn.pdf(z['x'], m[k], c[k])
        return -jnp.log(prob + 1e-12)
    return potential

def run_numpyro_sampler(potential_fn, init_params, num_samples, sampler_type='NUTS'):
    kernel = NUTS(potential_fn=potential_fn) if sampler_type == 'NUTS' else HMC(potential_fn=potential_fn, adapt_step_size=True)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), init_params={'x': jnp.array(init_params)})
    return np.array(mcmc.get_samples()['x'])

# ==========================================
# 4. Plotting (Vectorized)
# ==========================================

def plot_results(samples_dict, target_density_fn, true_samples, title, limits):
    os.makedirs("plots", exist_ok=True)

    x, y = np.linspace(limits[0][0], limits[0][1], 100), np.linspace(limits[1][0], limits[1][1], 100)
    X, Y = np.meshgrid(x, y)
    
    # FASTER: Removed the 10,000 iteration double for-loop!
    grid_points = np.c_[X.ravel(), Y.ravel()]
    Z = target_density_fn(grid_points).reshape(100, 100)
            
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16)
    
    for ax, (name, samples) in zip(axes, samples_dict.items()):
        ess = calculate_ess(samples)
        tv = compute_tv_distance(samples, target_density_fn, limits)
        emd = compute_emd(samples, true_samples)
        
        ax_title = f"{name}\nESS: {ess:.1f} | TV: {tv:.3f}\nEMD: {emd:.3f}"
        
        ax.contour(X, Y, Z, levels=15, cmap='Blues', alpha=0.8)
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, color='red')
        ax.scatter(true_samples[:, 0], true_samples[:, 1], s=5, alpha=0.3, color='purple')
        ax.set_title(ax_title)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        
    plt.tight_layout()
    plt.savefig(f'plots/{title}.png') 
    plt.show()

if __name__ == "__main__":
    NUM_SAMPLES = 10_000
    I_SIR_PARTICLES = 15
    X0 = np.array([0.0, 0.0])
    
    print("Running Banana Distribution Ablations...")
    for nu in tqdm([2.0, 5.0, 12.0, 20.0]):
        print(f"--- Banana Target (nu={nu}) ---")
        target_fn = lambda x: pi_banana(x, nu=nu)
        true_samples = sample_exact_banana(10000, nu=nu)
        
        s_cauchy = run_isir(target_fn, get_cauchy_proposal(scale_factor=10.0), X0, NUM_SAMPLES, I_SIR_PARTICLES)
        s_student = run_isir(target_fn, get_student_t_proposal(scale_factor=10.0, df=3.0), X0, NUM_SAMPLES, I_SIR_PARTICLES)
        s_hmc = run_numpyro_sampler(get_banana_potential_jax(nu), X0, NUM_SAMPLES, 'HMC')
        s_nuts = run_numpyro_sampler(get_banana_potential_jax(nu), X0, NUM_SAMPLES, 'NUTS')
        
        results = {"I-SIR (Cauchy)": s_cauchy, "I-SIR (Student)": s_student, "HMC": s_hmc, "NUTS": s_nuts}
        plot_results(results, target_fn, true_samples, f"Banana Target (nu={nu})", limits=([-15, 15], [-5, 5]))


    print("\nRunning Gaussian Mixture Ablations...")
    cov_base1 = np.array([
        [1.0,  0.6], 
        [0.6,  1.0]
    ])
    cov_base2 = np.array([
        [1.0, -0.6], 
        [-0.6,  1.0]
    ])
    cov_base3 = np.array([
        [0.8,  0.0], 
        [0.0,  0.8]
    ])

    configs = {
        "Balanced (Close)": {"weights": [0.33, 0.33, 0.34], "means": [[-2, -2], [2, 2], [0, 0]], "covs": [cov_base1, cov_base2, cov_base3]},
        "Imbalanced (Wide)": {"weights": [0.7, 0.2, 0.1], "means": [[-5, -5], [5, 5], [0, 0]], "covs": [cov_base1*0.5, cov_base2*2, cov_base3]}
    }
    
    for name, config in configs.items():
        print(f"--- GMM Target: {name} ---")
        target_fn = lambda x: pi_gmm(x, **config)
        true_samples = sample_exact_gmm(10000, **config)
        
        s_cauchy = run_isir(target_fn, get_cauchy_proposal(scale_factor=15.0), X0, NUM_SAMPLES, I_SIR_PARTICLES)
        s_student = run_isir(target_fn, get_student_t_proposal(scale_factor=15.0, df=3.0), X0, NUM_SAMPLES, I_SIR_PARTICLES)
        s_hmc = run_numpyro_sampler(get_gmm_potential_jax(**config), X0, NUM_SAMPLES, 'HMC')
        s_nuts = run_numpyro_sampler(get_gmm_potential_jax(**config), X0, NUM_SAMPLES, 'NUTS')
        
        results = {"I-SIR (Cauchy)": s_cauchy, "I-SIR (Student)": s_student, "HMC": s_hmc, "NUTS": s_nuts}
        plot_results(results, target_fn, true_samples, f"GMM_{name.replace(' ', '_')}", limits=([-10, 10], [-10, 10]))