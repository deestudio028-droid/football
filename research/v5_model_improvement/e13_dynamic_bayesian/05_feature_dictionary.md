# 05 — Dynamic Bayesian & Hierarchical Feature Dictionary

## 1. Dynamic Elo Features

- `dynamic_home_elo`: Adaptive-K Home Elo rating with season-boundary regression ($\alpha = 0.85$).
- `dynamic_away_elo`: Adaptive-K Away Elo rating.
- `dynamic_elo_diff`: `(dynamic_home_elo + 100.0) - dynamic_away_elo`.
- `home_k_factor`: Current adaptive learning weight $K_h = 20 \cdot (1 + 1.5/\sqrt{n_h+1})$.
- `away_k_factor`: Current adaptive learning weight for away team.
- `dyn_elo_att_diff`: Attack-specific rating difference `(home_att_elo + 50) - away_def_elo`.
- `dyn_elo_def_diff`: Defense-specific rating difference `away_att_elo - (home_def_elo + 50)`.

---

## 2. Dynamic State-Space Attack/Defense Features

- `dyn_A_home`, `dyn_D_home`: Current latent attack and defense states for Home team.
- `dyn_A_away`, `dyn_D_away`: Current latent attack and defense states for Away team.
- `var_A_home`, `var_D_home`: Posterior parameter estimation variances for Home team.
- `var_A_away`, `var_D_away`: Posterior parameter estimation variances for Away team.
- `total_param_var_h`: `var_A_home + var_D_away` (match-level Home parameter uncertainty).
- `total_param_var_a`: `var_A_away + var_D_home` (match-level Away parameter uncertainty).
- `dyn_implied_lam_h`: State-space implied base intensity $\mu_H \cdot \exp(A_h - D_a)$.
- `dyn_implied_lam_a`: State-space implied base intensity $\mu_A \cdot \exp(A_a - D_h)$.

---

## 3. Hierarchical Shrinkage & Venue-Specific Latents

- `h_shrunk_A`: Empirical Bayes shrunk attack state $\frac{n_h}{n_h + 5} A_h + \frac{5}{n_h + 5} \cdot 0$.
- `h_shrunk_D`: Empirical Bayes shrunk defense state.
- `a_shrunk_A`, `a_shrunk_D`: Shrunk states for Away team.
- `dyn_A_venue_home`: Latent attack strength expressed strictly in home matches.
- `dyn_D_venue_home`: Latent defense strength in home matches.
- `dyn_A_venue_away`, `dyn_D_venue_away`: Away venue latent states.
