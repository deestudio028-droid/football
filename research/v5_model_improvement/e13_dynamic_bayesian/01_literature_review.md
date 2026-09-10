# 01 — Dynamic Bayesian & Hierarchical Team Strength: Literature Review

## 1. Executive Summary

This literature review evaluates academic research and state-of-the-art methodologies for dynamic sports ratings, state-space latent team strength, hierarchical Bayesian shrinkage, and parameter uncertainty propagation in association football forecasting.

---

## 2. Comprehensive Methodological Audit

### 1. Dynamic State-Space Poisson Models (Fahrmeir & Tutz 1994, Rue & Salvesen 2000)
- **Methodology**: Team attacking $\alpha_i(t)$ and defensive $\delta_i(t)$ strengths are modeled as latent Gaussian random walks:
  $$\alpha_i(t) = \alpha_i(t-1) + \epsilon_{\alpha, t}, \quad \delta_i(t) = \delta_i(t-1) + \epsilon_{\delta, t}$$
  where $\epsilon \sim \mathcal{N}(0, \sigma_w^2)$.
- **Relevance**: Provides continuous time adaptation without artificial window cutoffs.
- **Risk**: Over-responsiveness to high-variance single-match goal spikes (variance leakage).

### 2. Time-Varying Dixon-Coles & Exponential Weighting (Dixon & Coles 1997)
- **Methodology**: Exponential downweighting of historical matches:
  $$\phi(t - t_k) = \exp(-\xi (t - t_k))$$
- **Finding**: Optimal decay rate $\xi \approx 0.0065$ corresponds to an effective half-life of $\sim 100\text{ days}$ ($\sim 15\text{ matches}$). Shorter half-lives increase out-of-sample forecast variance.

### 3. Bayesian Hierarchical Shrinkage (Gelman et al. 2013, Baio & Blangiardo 2010)
- **Methodology**: Team parameters are treated as exchangeable samples from common population hyperpriors:
  $$\alpha_i \sim \mathcal{N}(\mu_{\alpha}, \tau_{\alpha}^2), \quad \delta_i \sim \mathcal{N}(\mu_{\delta}, \tau_{\delta}^2)$$
- **Relevance**: Crucial for newly promoted teams with sparse sample histories ($N < 5$), shrinking noisy sample averages toward league baselines.

### 4. Season-Boundary Transition Shrinkage (Hvattum & Arntzen 2010)
- **Methodology**: Between seasons, team rosters, coaching staff, and physical baselines undergo mean-reverting shocks:
  $$R_i(s+1, 0) = \alpha_{\text{trans}} R_i(s, \text{final}) + (1 - \alpha_{\text{trans}}) \mu_0$$
  with empirical $\alpha_{\text{trans}} \in [0.75, 0.85]$.

### 5. Parameter Uncertainty Propagation (Koopman & Lit 2015)
- **Methodology**: Integrating over posterior variance $\sigma_{\lambda}^2$ via Gauss-Hermite quadrature:
  $$P(x, y) = \int \int \text{DixonColes}(x, y \mid \lambda_h, \lambda_a) \, \mathcal{N}(\ln \lambda_h \mid \mu_h, \sigma_h^2) \, \mathcal{N}(\ln \lambda_a \mid \mu_a, \sigma_a^2) \, d\lambda_h d\lambda_a$$
- **Finding**: Softens favorite probabilities when parameter uncertainty is high (e.g., promoted teams), reducing tail log-loss penalties.
