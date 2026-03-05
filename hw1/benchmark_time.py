import numpy as np
import time
import arviz as az
try:
    from scipy.stats import wasserstein_distance_nd
    HAS_WASSERSTEIN = True
except ImportError:
    HAS_WASSERSTEIN = False

# ==========================================
# 0. DUMMY DATA SETUP
# ==========================================
N_SAMPLES = 10000
DIM = 2

print(f"Generating {N_SAMPLES} samples for benchmarking...")
# Create a dummy chain with some correlation (like real MCMC)
samples = np.zeros((N_SAMPLES, DIM))
samples[0] = np.random.randn(DIM)
for i in range(1, N_SAMPLES):
    samples[i] = 0.8 * samples[i-1] + 0.2 * np.random.randn(DIM)

# True independent samples
true_samples = np.random.randn(N_SAMPLES, DIM)

# Dummy Target PDF (Standard Normal)
def dummy_pdf(x):
    x = np.atleast_2d(x)
    return np.exp(-0.5 * np.sum(x**2, axis=1)) / (2 * np.pi)

# Pre-compute Grid for Fast TV
limits = ([-5, 5], [-5, 5])
bins = 100
x_edges = np.linspace(limits[0][0], limits[0][1], bins + 1)
y_edges = np.linspace(limits[1][0], limits[1][1], bins + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
grid_points = np.c_[X.ravel(), Y.ravel()]
Z_grid = dummy_pdf(grid_points).reshape(bins, bins)

# ==========================================
# 1. ORIGINAL METRICS
# ==========================================
def orig_ess(s):
    samples_az = np.expand_dims(s, axis=0) 
    return np.mean(az.ess(az.dict_to_dataset({"x": samples_az}))["x"].values)

def orig_tv(s, limits, bins=100):
    H, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=[x_edges, y_edges])
    empirical_mass = H / np.sum(H)
    true_pdf = dummy_pdf(grid_points).reshape(bins, bins) # RE-CALCULATES EVERY TIME
    true_mass = true_pdf / np.sum(true_pdf)
    return 0.5 * np.sum(np.abs(empirical_mass - true_mass))

def orig_emd(s, t_s, subsample=1500):
    if not HAS_WASSERSTEIN: return np.nan
    idx_s = np.random.choice(len(s), min(len(s), subsample), replace=False)
    idx_t = np.random.choice(len(t_s), min(len(t_s), subsample), replace=False)
    return wasserstein_distance_nd(s[idx_s], t_s[idx_t])

# ==========================================
# 2. FAST METRICS
# ==========================================
def fast_ess(s):
    N, d = s.shape
    ess_vals = []
    for i in range(d):
        x = s[:, i] - np.mean(s[:, i])
        fft_x = np.fft.fft(x, n=2*N)
        acf = np.fft.ifft(fft_x * np.conj(fft_x))[:N].real
        acf /= acf[0]
        neg_idx = np.where(acf < 0)[0]
        cutoff = neg_idx[0] if len(neg_idx) > 0 else N
        tau = 1 + 2 * np.sum(acf[1:cutoff])
        ess_vals.append(N / max(tau, 1.0))
    return np.mean(ess_vals)

def fast_tv(s, true_grid):
    H, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=[x_edges, y_edges])
    empirical_mass = H / np.sum(H)
    sum_true = np.sum(true_grid)
    true_mass = true_grid / sum_true if sum_true > 0 else np.zeros_like(true_grid)
    return 0.5 * np.sum(np.abs(empirical_mass - true_mass))

def fast_swd(s, t_s, n_projections=50):
    theta = np.random.uniform(0, 2 * np.pi, n_projections)
    dirs = np.vstack((np.cos(theta), np.sin(theta))).T
    proj_samples = np.sort(s @ dirs.T, axis=0)
    proj_true = np.sort(t_s @ dirs.T, axis=0)
    return np.mean(np.abs(proj_samples - proj_true))

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
# 3. RUN BENCHMARK
# ==========================================
def run_benchmark(name, func, *args):
    # Warmup
    _ = func(*args)
    
    # Time 10 iterations
    start = time.perf_counter()
    for _ in range(10):
        val = func(*args)
    end = time.perf_counter()
    
    avg_time = (end - start) / 10 * 1000 # Convert to milliseconds
    return val, avg_time

print("\n" + "="*50)
print(f"{'METRIC':<15} | {'ALGORITHM':<12} | {'TIME (ms)':<10} | {'VALUE'}")
print("="*50)

# 1. ESS
v_orig, t_orig = run_benchmark("ESS", orig_ess, samples)
v_fast, t_fast = run_benchmark("ESS", fast_ess, samples)
print(f"{'ESS':<15} | {'ArviZ':<12} | {t_orig:>8.2f} ms | {v_orig:.2f}")
print(f"{'ESS':<15} | {'FFT (Fast)':<12} | {t_fast:>8.2f} ms | {v_fast:.2f}")
print("-" * 50)

# 2. TV
v_orig, t_orig = run_benchmark("TV", orig_tv, samples, limits)
v_fast, t_fast = run_benchmark("TV", fast_tv, samples, Z_grid)
print(f"{'TV':<15} | {'Recalculate':<12} | {t_orig:>8.2f} ms | {v_orig:.4f}")
print(f"{'TV':<15} | {'Precomputed':<12} | {t_fast:>8.2f} ms | {v_fast:.4f}")
print("-" * 50)

# 3. EMD / SWD
# v_orig, t_orig = run_benchmark("EMD", orig_emd, samples, true_samples)
v_fast, t_fast = run_benchmark("EMD", fast_swd, samples, true_samples)
v_fast_strict, t_fast_strict = run_benchmark("EMD", strict_fast_swd, samples, true_samples)
# print(f"{'EMD':<15} | {'Exact':<12} | {t_orig:>8.2f} ms | {v_orig:.4f}")
print(f"{'EMD':<15} | {'SWD':<12} | {t_fast:>8.2f} ms | {v_fast:.4f}")
print(f"{'EMD':<15} | {'Strcit SWD':<12} | {t_fast_strict:>8.2f} ms | {v_fast_strict:.4f}")

print("="*50)

print("\nNotes:")
print("- Exact EMD had to be subsampled to 1500 points to finish. SWD runs on ALL 10,000 points.")
print("- Fast ESS (FFT) and ArviZ ESS use slightly different autocorrelation cutoffs but yield structurally identical conclusions.")