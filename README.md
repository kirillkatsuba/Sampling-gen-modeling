# Parameter ablations 
## Implemenation details 
1. All function and sampling of I-ISIR, NUTS, HMC were provided in ```hw1/isir_pure_jax.py```. This scrip provide a big variety of test wit different Number of particelse, mean and covariance for GMM, separation / scale in Banana distribution. The whole tests run approximately 25-27 minutes, for specific number of particles test run aproximately 5-6 minutes. In the code, in the ```if __name__ == "__main__":``` section you can find list ```particle_counts``` and set any number you disier. My suggestion stop between 50 and 100, because in my tests this particlese number provided the best metrics. In the next section we wil discuss this more detailed. The metrics also provided on the same graphs.

2. After running the script ```isir_pure_jax.py``` plots of target density with samples from I-SIR, NUTS, HMC is generated in ```plots``` (you can check all of my itterations, there are 252 different examples) and also file ```benchmark_metrics_varied_N_big_test.csv``` with metrics and hyperparametrs of the target densities is generated in root directory
![Example of the image](/plots/GMM_K=3_Imbalanced_(Wide)_Particles=100.png)

2. The clean structure of I-SIR algorithm were provided in the following code in the ```isir_pure_jax.py``` with JAX. This code striclty follows the logic of the algorithm provided in the task:

```python
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
```
With given target density we run one itteration of the I-SIR algorithm

3. The full code provided in the ```isir_pure_jax.py``` not that hard, but we need to explain following metrics: TV, ESS, EMD, because I use not trivial realisation.

TV is presented in the following way: We know target density and we can sample from MCMC algoritms, hence we need to approximate densities. Since we are in $\mathbf{R}^2$ we can devide each sample from MCMC algorithm in bins in $x$ and $y$, after that we build histogram and approximate density with kde. And compute TV by our basic formula: $||P, Q|| = \frac{1}{2} \int_x (P(x) - Q(x))dx$. Code provided in the function ```def compute_tv_distance(samples, pdf_fn, limits, bins=40)```


We know MCMC samples are correlated over time, hence we need to estimate the actual number of independent samples. We evaluate this for each dimension independently and average the results. To do this computationally fast, we center the samples, compute the Auto-Correlation Function using the Fast Fourier Transform, and sum these auto-correlations up to the first negative lag to find the integrated autocorrelation time $\tau$. And compute ESS by our basic formula: $\text{ESS} = \frac{N}{\tau}$, where $\tau = 1 + 2 \sum_{t=1}^{k} \rho_t$. Code provided in the function `def fast_ess(samples)`.


We want to evaluate the exact Earth Mover's Distance (EMD) to heavily penalize samplers that miss disconnected modes, but exact 2D EMD requires solving an optimal transport problem with $\mathcal{O}(N^3)$ complexity, which is computationally intractable for thousands of points. Hence we need a lightning-fast alternative. We generate random 1D lines (angles) and project both our MCMC samples and true samples onto them. After that, we simply sort the projected points. This works because *in 1D, exact EMD is mathematically proven to be exactly equal to the distance between sorted coordinates. And compute SWD by our basic formula: we calculate the Root Mean Square error between these sorted 1D arrays and average it across all random projections. While 2D SWD is technically a lower bound to true 2D EMD, relying on the 1D exact EMD equivalence allows us to approximate the spatial penalty in $\mathcal{O}(N \log N)$ time. Code provided in the function `def strict_fast_swd(samples, true_samples, n_projections=500)`.

So, we will call EMD as SWD.

### Result and parametrs analysation
We have viried a lot of parametrs and got following graphs. Let's show some intresting examples and describe why we get following results.

Let's note that you can check all the images in ```plots``` for any number for particles and any set of prametrs. We will show graphs for the particels number that equals to 50.

Starts with banana distribution:
![Example of the image](/plots/Banana_nu=0.1_Particles=50.png)
![Example of the image](/plots/Banana_nu=1.0_Particles=50.png)
![Example of the image](/plots/Banana_nu=5.0_Particles=50.png)
![Example of the image](/plots/Banana_nu=12.0_Particles=50.png)

Banana distribution isn't very complex for samples, beucause it's hava only one mode, without any non-continuos parts. Hence, almost all the  metrics for all samplers are almost the same. Some metrics are better for HMC such as ESS and TV for first two tests. NUTS outperforms all the others algorithms on the third one. So all algorithms provide very good results, maybe better samples were provided by HMC and NUTS, but I-SIR with all proposals provided almost similar samples, ESS better for I-SIR for the last two graphs.



Let's move on to the GMM, balanced data:
![Example of the image](/plots/GMM_K=3_Balanced_(Close)_Particles=5.png)
![Example of the image](/plots/GMM_K=3_Balanced_(Wide)_Particles=50.png)
![Example of the image](/plots/GMM_K=3_Balanced_(The_Widest)_Particles=50.png)
![Example of the image](/plots/GMM_K=5_Balanced_(Close)_Particles=5.png)
![Example of the image](/plots/GMM_K=5_Balanced_(Wide)_Particles=50.png)
![Example of the image](/plots/GMM_K=5_Balanced_(The_Widest)_Particles=50.png)
![Example of the image](/plots/GMM_K=15_Balanced_(Close)_Particles=5.png)
![Example of the image](/plots/GMM_K=15_Balanced_(Wide)_Particles=50.png)
![Example of the image](/plots/GMM_K=15_Balanced_(The_Widest)_Particles=50.png)

There are very interesting results for balanced data. Gaussians located in very wide range (plot 2, 3, 5, 6) we see exactly what we expect from HMC and NUTS - they fail with more thatn two modes, while I-SIR work very good, provide mauch higher metrics, for example TV distance is very good compare to HMC and NUTS. Moreover, I-SIR cover all the modes of GMM, while other approaches fails and can cover only one or two modes put of 5. HMC and NUTS can even collapse to converge, because of very wide range (plot 3, 6). For close GMM I-SIR better than HMC and NUTS, but stil very close samples in graphs.

Let's describe imbalanced data 
![Example of the image](/plots/GMM_K=3_Imbalanced_(Close)_Particles=5.png)
![Example of the image](/plots/GMM_K=3_Imbalanced_(Wide)_Particles=50.png)
![Example of the image](/plots/GMM_K=3_Imbalanced_(The_Widest)_Particles=50.png)
![Example of the image](/plots/GMM_K=5_Imbalanced_(Close)_Particles=5.png)
![Example of the image](/plots/GMM_K=5_Imbalanced_(Wide)_Particles=50.png)
![Example of the image](/plots/GMM_K=5_Imbalanced_(The_Widest)_Particles=50.png)
![Example of the image](/plots/GMM_K=15_Imbalanced_(Close)_Particles=5.png)
![Example of the image](/plots/GMM_K=15_Imbalanced_(Wide)_Particles=50.png)
![Example of the image](/plots/GMM_K=15_Imbalanced_(The_Widest)_Particles=50.png)

We witness almost the same results, but this examples even better than balanced classes. For example last image show exactly this. If in the previuos one HMC and NUTS could sample in the widest GMM with 15 Gaussians HMC and NUTS could sample in almost all modes, but here it drops significantly: HMC coveres only half of the modes, NUTS coveres only two. And metrcis here are very bad. Especialy for wide and the widest GMM some examples of NUTS shiw TV hear the one and it's very bad, it can covere wrong mode.


Let's now compare all the parametrs with metrics:
![Example of the image](/plot_mode_collapse.png)
![Example of the image](/plot_effect_of_K_widest.png)
![Example of the image](/plot_effect_of_K.png)
![Example of the image](/plot_effect_of_N.png)


First, I-SIR cover each of the modes very good. Outperform in ESS HMC and NUTS in very big number.

With bigger number of Gaussians I-SIR stay the same in context of TV, the result are very high for any number of Gaussians. While HMC and NUTS lower its values, but still can't reach the I-SIR.


## Conclusion

I-SIR is global MCMC aldorithm and with havy tailed proposals such as Cauchy and t-Student can not only provide very good samples in the regions with low probability, otherwise HMC and NUTS collapsed with tails of the distribution and with distributions with a lot of modes. 


## References

1. **Local-Global MCMC**: 
   - Samsonov, S., Lagutin, E., Gabrié, M., Durmus, A., Naumov, A., & Moulines, E. (2021). *Local-Global MCMC kernels: the best of both worlds*. arXiv preprint. Available at:[https://arxiv.org/abs/2111.02702](https://arxiv.org/abs/2111.02702)
2. **Adaptive I-SIR**:
   - Laitinen, P., & Vihola, M. (2025). *Iterated sampling importance resampling with adaptive number of proposals*. arXiv preprint. Available at:[https://arxiv.org/abs/2512.00220](https://arxiv.org/abs/2512.00220)
