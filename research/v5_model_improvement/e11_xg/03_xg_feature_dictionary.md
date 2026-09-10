# 03 — Causal Expected Goals (xG) Feature Dictionary

## 1. Executive Summary

This feature dictionary defines the 36 causal pre-match features constructed in Experiment $E_{11}$ ([`data/research/e11_causal_xg_features.parquet`](file:///e:/Football%20Prediction%20Project/data/research/e11_causal_xg_features.parquet)).

Every feature is generated strictly using matches kicking off prior to the target match ($t_{\text{past}} < t_{\text{match}}$) with zero forward contamination.

---

## 2. Comprehensive Feature Specification Matrix

### Group A: Team-Level Raw Rolling xG Statistics (20 Features)

| Feature Name | Window ($N$) | Definition / Mathematical Formula | Missingness / Promoted Prior | Leakage Risk |
|:---|:---:|:---|:---:|:---:|
| `home_xg_for_last3` | 3 matches | Mean $xG$ generated in last 3 matches by Home team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `home_xg_against_last3` | 3 matches | Mean $xG$ conceded in last 3 matches by Home team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `home_xg_diff_last3` | 3 matches | `home_xg_for_last3 - home_xg_against_last3` | 0.000 | Zero (Lagged) |
| `home_xg_for_last5` | 5 matches | Mean $xG$ generated in last 5 matches by Home team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `home_xg_against_last5` | 5 matches | Mean $xG$ conceded in last 5 matches by Home team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `home_xg_diff_last5` | 5 matches | `home_xg_for_last5 - home_xg_against_last5` | 0.000 | Zero (Lagged) |
| `home_xg_for_last10` | 10 matches | Mean $xG$ generated in last 10 matches by Home team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `home_xg_against_last10` | 10 matches | Mean $xG$ conceded in last 10 matches by Home team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `home_xg_diff_last10` | 10 matches | `home_xg_for_last10 - home_xg_against_last10` | 0.000 | Zero (Lagged) |
| `home_npxg_diff_last5` | 5 matches | Non-penalty $xG$ differential across last 5 matches | 0.000 | Zero (Lagged) |
| `away_xg_for_last3` | 3 matches | Mean $xG$ generated in last 3 matches by Away team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `away_xg_against_last3` | 3 matches | Mean $xG$ conceded in last 3 matches by Away team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `away_xg_diff_last3` | 3 matches | `away_xg_for_last3 - away_xg_against_last3` | 0.000 | Zero (Lagged) |
| `away_xg_for_last5` | 5 matches | Mean $xG$ generated in last 5 matches by Away team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `away_xg_against_last5` | 5 matches | Mean $xG$ conceded in last 5 matches by Away team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `away_xg_diff_last5` | 5 matches | `away_xg_for_last5 - away_xg_against_last5` | 0.000 | Zero (Lagged) |
| `away_xg_for_last10` | 10 matches | Mean $xG$ generated in last 10 matches by Away team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `away_xg_against_last10` | 10 matches | Mean $xG$ conceded in last 10 matches by Away team | Pseudo-count shrinkage to 1.250 | Zero (Lagged) |
| `away_xg_diff_last10` | 10 matches | `away_xg_for_last10 - away_xg_against_last10` | 0.000 | Zero (Lagged) |
| `away_npxg_diff_last5` | 5 matches | Non-penalty $xG$ differential across last 5 matches | 0.000 | Zero (Lagged) |

---

### Group B: Exponentially Weighted Moving Averages (EWMA) (6 Features)

| Feature Name | Half-Life ($t_{1/2}$) | Definition / Mathematical Formula | Missingness / Prior | Leakage Risk |
|:---|:---:|:---|:---:|:---:|
| `home_xg_ewma` | 5 matches | $\sum w_i xG_{H, i} / \sum w_i$ where $w_i = \exp(-\alpha \cdot i), \alpha = \ln(2)/5$ | 1.250 | Zero (Lagged) |
| `home_xga_ewma` | 5 matches | $\sum w_i xGA_{H, i} / \sum w_i$ with half-life 5 matches | 1.250 | Zero (Lagged) |
| `home_xgd_ewma` | 5 matches | `home_xg_ewma - home_xga_ewma` | 0.000 | Zero (Lagged) |
| `away_xg_ewma` | 5 matches | $\sum w_i xG_{A, i} / \sum w_i$ with half-life 5 matches | 1.250 | Zero (Lagged) |
| `away_xga_ewma` | 5 matches | $\sum w_i xGA_{A, i} / \sum w_i$ with half-life 5 matches | 1.250 | Zero (Lagged) |
| `away_xgd_ewma` | 5 matches | `away_xg_ewma - away_xga_ewma` | 0.000 | Zero (Lagged) |

---

### Group C: Opponent-Adjusted Relative Strengths & Interactions (6 Features)

| Feature Name | Window | Definition / Mathematical Formula | Role in Model | Leakage Risk |
|:---|:---:|:---|:---|:---:|
| `home_xg_att_strength` | EWMA | $\text{home\_xg\_ewma} / \bar{xG}_{\text{league}}$ | Relative attacking capacity | Zero (Lagged) |
| `home_xg_def_strength` | EWMA | $\text{home\_xga\_ewma} / \bar{xGA}_{\text{league}}$ | Relative defensive concession | Zero (Lagged) |
| `away_xg_att_strength` | EWMA | $\text{away\_xg\_ewma} / \bar{xG}_{\text{league}}$ | Away attacking capacity | Zero (Lagged) |
| `away_xg_def_strength` | EWMA | $\text{away\_xga\_ewma} / \bar{xGA}_{\text{league}}$ | Away defensive concession | Zero (Lagged) |
| `home_exp_xg` | Match | $\text{home\_att} \times \text{away\_def} \times 1.12 \times \bar{xG}_{\text{league}}$ | Pre-match Home expected goals | Zero (Pre-match) |
| `away_exp_xg` | Match | $\text{away\_att} \times \text{home\_def} \times 0.88 \times \bar{xG}_{\text{league}}$ | Pre-match Away expected goals | Zero (Pre-match) |

---

### Group D: Match Differentials, Parity & Schedule Fatigue (4 Features)

| Feature Name | Mathematical Definition | Signal Hypothesis |
|:---|:---|:---|
| `xg_diff_expected` | `home_exp_xg - away_exp_xg` | Direct linear mismatch predictor |
| `xg_sum_expected` | `home_exp_xg + away_exp_xg` | Total match expected scoring intensity |
| `xg_parity_index` | $\exp\left(-\frac{(\text{xg\_diff\_expected})^2}{2 \cdot 0.45^2}\right) \in [0, 1]$ | Gaussian parity index (peaks when teams are matched) |
| `rest_days_diff` | $\text{Rest Days}_{\text{Home}} - \text{Rest Days}_{\text{Away}}$ | Calendar physical rest differential |
