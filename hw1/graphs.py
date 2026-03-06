import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('benchmark_metrics_varied_N_big_test.csv')

def parse_config(row):
    config = row['Configuration']
    
    n_val = int(config.split('N=')[1].replace('"', ''))
    
    if "Close" in config: sep = "Close"
    elif "Wide" in config and "The Widest" not in config: sep = "Wide"
    elif "The Widest" in config: sep = "The Widest"
    else: sep = "Continuous"
        
    k_val = -1
    if "K=" in config:
        k_val = int(config.split('K=')[1].split('_')[0])
        
    return pd.Series([n_val, sep, k_val])

df[['N', 'Separation', 'K']] = df.apply(parse_config, axis=1)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
SWD_CAP = 25.0
df['SWD'] = df['SWD'].fillna(SWD_CAP) 

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


plt.figure(figsize=(10, 6))
df_n_effect = df[(df['Target'] == 'Banana') & (df['Configuration'].str.contains('nu=2.0'))]

sns.lineplot(data=df_n_effect, x='N', y='ESS', hue='Algorithm', marker='o', linewidth=2, markersize=8)
plt.title('Effect of Particle Count (N) on Effective Sample Size (Banana Target)')
plt.ylabel('Effective Sample Size (ESS) - Higher is Better')
plt.xlabel('Number of Particles (N)')
plt.tight_layout()
plt.savefig('plot_effect_of_N.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
df_sep_effect = df[(df['Target'] == 'GMM') & (df['K'] == 5) & (df['N'] == 50) & (df['Configuration'].str.contains('Balanced'))]

order = ['Close', 'Wide', 'The Widest']
sns.barplot(data=df_sep_effect, x='Separation', y='SWD', hue='Algorithm', order=order)

plt.title('Algorithm Robustness to Mode Separation (K=5, N=50)')
plt.ylabel('Sliced Wasserstein Distance (SWD) - Lower is Better')
plt.xlabel('Distance Between GMM Clusters')
plt.text(1.5, SWD_CAP - 1, '*HMC/NUTS Diverged to Infinity (Capped at 25)', color='red', fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('plot_mode_collapse.png', dpi=300)
plt.show()


plt.figure(figsize=(10, 6))
df_k_effect = df[(df['Target'] == 'GMM') & (df['Separation'] == 'Wide') & (df['N'] == 50) & (df['Configuration'].str.contains('Imbalanced'))]

sns.lineplot(data=df_k_effect, x='K', y='TV', hue='Algorithm', marker='s', linewidth=2, markersize=8)

plt.title('Effect of Target Complexity (Number of Modes K) on Accuracy')
plt.ylabel('Total Variation Distance (TV) - Lower is Better')
plt.xlabel('Number of Gaussian Components (K)')
plt.xticks([3, 5, 8, 15])

plt.tight_layout()
plt.savefig('plot_effect_of_K.png', dpi=300)
plt.show()

print("Plots generated successfully! Saved as PNG files.")

plt.figure(figsize=(10, 6))

df_widest_k = df[(df['Target'] == 'GMM') & 
                 (df['Separation'] == 'The Widest') & 
                 (df['N'] == 50) & 
                 (df['Configuration'].str.contains('Imbalanced'))].copy()


df_widest_k['TV'] = df_widest_k['TV'].fillna(1.0)

sns.lineplot(data=df_widest_k, x='K', y='TV', hue='Algorithm', marker='s', linewidth=2, markersize=8)

plt.title('Effect of Target Complexity (K) on Accuracy (The Widest GMM)')
plt.ylabel('Total Variation Distance (TV) - Lower is Better')
plt.xlabel('Number of Gaussian Components (K)')
plt.xticks([3, 5, 8, 15])
plt.ylim(0, 1.05)

plt.text(14.5, 0.95, '*NaN values capped at TV=1.0', color='red', fontsize=10, ha='right')

plt.tight_layout()
plt.savefig('plot_effect_of_K_widest.png', dpi=300)
plt.show()
