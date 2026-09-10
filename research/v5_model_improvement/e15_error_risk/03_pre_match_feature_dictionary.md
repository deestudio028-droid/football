# 03 — Pre-Match Risk Feature Dictionary

| Feature Name | Category | Mathematical Definition | Correlation with $RPS$ |
|:---|:---|:---|:---:|
| `max_prob_e10` | E10 Geometry | $\max(p_H, p_D, p_A)$ | **`-0.2683`** |
| `entropy_e10` | E10 Geometry | $-\sum p_c \ln p_c$ | **`+0.2674`** |
| `lambda_ratio` | Intensity | $\max(\lambda_H, \lambda_A) / \min(\lambda_H, \lambda_A)$ | **`-0.2661`** |
| `margin_p1_p2` | E10 Geometry | $p_{(1)} - p_{(2)}$ | **`-0.2661`** |
| `p_draw_e10` | E10 Geometry | $p_D$ | **`+0.2560`** |
| `abs_elo_diff` | Team Strength | $\|R_H + 100 - R_A\|$ | **`-0.2516`** |
| `total_lam` | Intensity | $\lambda_H + \lambda_A$ | **`-0.1888`** |
| `p_home_e10` | E10 Geometry | $p_H$ | **`-0.1597`** |
| `home_elo` | Team Strength | $R_H$ | **`-0.1407`** |
| `p_away_e10` | E10 Geometry | $p_A$ | `+0.1065` |
| `away_elo` | Team Strength | $R_A$ | `+0.0569` |
| `min_team_samples`| Maturity | $\min(n_H, n_A)$ | `+0.0273` |
| `is_promoted_sparse`| Maturity | $\mathbf{1}_{\{\min(n_H, n_A) < 5\}}$ | `-0.0174` |
| `season_match_num`| Early Season | Rank of match in season | `+0.0172` |
| `mean_abs_prob_diff`| Disagreement | $\frac{1}{3} \sum \|P_{\text{E10}} - P_{\text{E13}}\|$ | `+0.0147` |
| `is_early_3` | Early Season | $\mathbf{1}_{\{\text{season\_match\_num} \le 3\}}$ | `-0.0085` |
| `js_divergence_e10_e13`| Disagreement | $D_{\text{JS}}(P_{\text{E10}} \parallel P_{\text{E13}})$ | `-0.0040` |
| `is_early_5` | Early Season | $\mathbf{1}_{\{3 < \text{season\_match\_num} \le 5\}}$ | `+0.0010` |
