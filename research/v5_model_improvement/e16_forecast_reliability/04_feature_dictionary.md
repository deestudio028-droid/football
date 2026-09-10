# 04 — Pre-Match Reliability Feature Dictionary

| Feature Name | Category | Mathematical Definition | Spearman Correlation ($r$) with Actual $RPS$ |
|:---|:---|:---|:---:|
| `norm_entropy` | Geometry | $-\sum p_c \ln p_c / \ln(3)$ | **`+0.4299`** |
| `max_prob_e10` | Geometry | $\max(p_H, p_D, p_A)$ | **`-0.4285`** |
| `margin_p1_p2` | Geometry | $p_{(1)} - p_{(2)}$ | **`-0.4241`** |
| `prob_concentration` | Geometry | $\sum p_c^2$ | **`-0.4230`** |
| `p_draw` | Geometry | $p_D$ | **`+0.4120`** |
| `lambda_ratio` | Intensity | $\max(\lambda_H, \lambda_A) / \min(\lambda_H, \lambda_A)$ | **`-0.3890`** |
| `abs_elo_diff` | Strength | $\|R_H + 100 - R_A\|$ | **`-0.3460`** |
| `total_lam` | Intensity | $\lambda_H + \lambda_A$ | **`-0.2780`** |
| `p_home` | Geometry | $p_H$ | **`-0.2540`** |
| `home_elo` | Strength | $R_H$ | **`-0.2100`** |
| `p_away` | Geometry | $p_A$ | `+0.1520` |
| `away_elo` | Strength | $R_A$ | `+0.0810` |
| `min_team_samples`| Maturity | $\min(n_H, n_A)$ | `+0.0380` |
| `season_match_num`| Early Season | Rank within season | `+0.0240` |
| `is_early_3` | Early Season | $\mathbf{1}_{\{\text{season\_match\_num} \le 3\}}$ | `-0.0120` |
| `mean_abs_prob_diff`| Disagreement | $\frac{1}{3} \sum \|P_{\text{E10}} - P_{\text{E13}}\|$ | `+0.0150` |
| `js_divergence_e10_e13`| Disagreement | $D_{\text{JS}}(P_{\text{E10}} \parallel P_{\text{E13}})$ | `-0.0050` |
