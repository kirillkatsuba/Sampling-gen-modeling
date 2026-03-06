import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the data
# Make sure your CSV file is named exactly 'benchmark_metrics_varied_N_big_test.csv'
df = pd.read_csv('benchmark_metrics_varied_N_big_test.csv')

# 2. Data Cleaning & Parsing Configuration
# We need to extract N, K, and Separation into separate columns for plotting
def parse_config(row):
    config = row['Configuration']
    
    # Extract N
    n_val = int(config.split('N=')[1].replace('"', ''))
    
    # Extract Separation (Close, Wide, The Widest)
    if "Close" in config: sep = "Close"
    elif "Wide" in config and "The Widest" not in config: sep = "Wide"
    elif "The Widest" in config: sep = "The Widest"
    else: sep = "Continuous" # For Banana
        
    # Extract K
    k_val = -1 # Default for Banana
    if "K=" in config:
        k_val = int(config.split('K=')[1].split('_')[0])
        
    return pd.Series([n_val, sep, k_val])

df[['N', 'Separation', 'K']] = df.apply(parse_config, axis=1)

# Clean up infinite values so plots don't break (replace inf with a high cap for visualization)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
SWD_CAP = 25.0 # Cap diverging SWD at 25 for the charts
df['SWD'] = df['SWD'].fillna(SWD_CAP) 

# Set professional plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# =========================================================
# PLOT 1: How Particle Count (N) affects I-SIR ESS
# =========================================================
plt.figure(figsize=(10, 6))
# Filter data: Just Banana nu=2.0 to see the pure effect of N on I-SIR
df_n_effect = df[(df['Target'] == 'Banana') & (df['Configuration'].str.contains('nu=2.0'))]

sns.lineplot(data=df_n_effect, x='N', y='ESS', hue='Algorithm', marker='o', linewidth=2, markersize=8)
plt.title('Effect of Particle Count (N) on Effective Sample Size (Banana Target)')
plt.ylabel('Effective Sample Size (ESS) - Higher is Better')
plt.xlabel('Number of Particles (N)')
plt.tight_layout()
plt.savefig('plot_effect_of_N.png', dpi=300)
plt.show()

# =========================================================
# PLOT 2: The Mode Collapse Test (SWD vs Separation)
# =========================================================
plt.figure(figsize=(10, 6))
# Filter data: Look at GMM K=5 Balanced, with N=50 to give I-SIR a fair chance
df_sep_effect = df[(df['Target'] == 'GMM') & (df['K'] == 5) & (df['N'] == 50) & (df['Configuration'].str.contains('Balanced'))]

# Sort x-axis logically
order = ['Close', 'Wide', 'The Widest']
sns.barplot(data=df_sep_effect, x='Separation', y='SWD', hue='Algorithm', order=order)

plt.title('Algorithm Robustness to Mode Separation (K=5, N=50)')
plt.ylabel('Sliced Wasserstein Distance (SWD) - Lower is Better')
plt.xlabel('Distance Between GMM Clusters')
# Add a text note about the NaN cap
plt.text(1.5, SWD_CAP - 1, '*HMC/NUTS Diverged to Infinity (Capped at 25)', color='red', fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('plot_mode_collapse.png', dpi=300)
plt.show()

# =========================================================
# PLOT 3: How the Number of Clusters (K) affects Performance
# =========================================================
plt.figure(figsize=(10, 6))
# Filter data: Wide Separation, Imbalanced, N=50
df_k_effect = df[(df['Target'] == 'GMM') & (df['Separation'] == 'Wide') & (df['N'] == 50) & (df['Configuration'].str.contains('Imbalanced'))]

sns.lineplot(data=df_k_effect, x='K', y='TV', hue='Algorithm', marker='s', linewidth=2, markersize=8)

plt.title('Effect of Target Complexity (Number of Modes K) on Accuracy')
plt.ylabel('Total Variation Distance (TV) - Lower is Better')
plt.xlabel('Number of Gaussian Components (K)')
plt.xticks([3, 5, 8, 15]) # Ensure x-axis only shows the K values we actually tested

plt.tight_layout()
plt.savefig('plot_effect_of_K.png', dpi=300)
plt.show()

print("Plots generated successfully! Saved as PNG files.")

# =========================================================
# PLOT 4 (UPDATED): Effect of K on THE WIDEST GMM
# =========================================================
plt.figure(figsize=(10, 6))

# Filter data: THE WIDEST Separation, Imbalanced, N=50
df_widest_k = df[(df['Target'] == 'GMM') & 
                 (df['Separation'] == 'The Widest') & 
                 (df['N'] == 50) & 
                 (df['Configuration'].str.contains('Imbalanced'))].copy()

# Replace NaN TV values (caused by catastrophic gradient divergence) 
# with 1.0 (maximum possible TV error) so the lines draw correctly.
df_widest_k['TV'] = df_widest_k['TV'].fillna(1.0)

# Plot the lines
sns.lineplot(data=df_widest_k, x='K', y='TV', hue='Algorithm', marker='s', linewidth=2, markersize=8)

plt.title('Effect of Target Complexity (K) on Accuracy (The Widest GMM)')
plt.ylabel('Total Variation Distance (TV) - Lower is Better')
plt.xlabel('Number of Gaussian Components (K)')
plt.xticks([3, 5, 8, 15]) # Ensure x-axis matches our discrete tests
plt.ylim(0, 1.05) # Lock y-axis from 0 to max TV

# Add a small annotation explaining the capped values
plt.text(14.5, 0.95, '*NaN values capped at TV=1.0', color='red', fontsize=10, ha='right')

plt.tight_layout()
plt.savefig('plot_effect_of_K_widest.png', dpi=300)
plt.show()