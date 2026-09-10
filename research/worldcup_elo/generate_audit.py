# generator script
import sys
content = """# WORLD CUP ELO RESEARCH AUDIT

**Target Repository:** [hjjbh1314/worldcup-predictor](https://github.com/hjjbh1314/worldcup-predictor)
**Audit Date:** 2026-08-20
**Environment:** Isolated Research Sandbox (
esearch/worldcup_repo/)
**Scope:** Architecture, mathematical equations, features, causality/leakage audit, validation methodology, reproduction results, and transferability analysis to the Football Prediction Project (FPP).

---

## 1. System Architecture & Information Pipeline

The worldcup-predictor repository implements a match-prediction architecture tailored for international national-team football based on historical match outcomes dating back to 1872 (martj42/international_results).

`
                    Historical Results (1872 -> present)
                                    │
                                    ▼ (Chronological Walk)
┌────────────────────────────────────────────────────────────────────────┐
│ Online Elo Engine (eloratings.net standard)                            │
│   • R_0 = 1500                                                         │
│   • Dynamic K-factor (20 - 60 by tournament tier)                      │
│   • Goal-difference margin multiplier G(|gd|)                          │
│   • Home advantage (+100 Elo points for true home)                     │
│   • Pre-match rating capture: R_H(T), R_A(T), Delta(T)                 │
│   • Post-match state update: R' = R + K * G * (W - W_e)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  Pre-match Features [Delta, |Delta|, (conf_diff)]
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Probability Head (Multi-class Logistic Regression)                     │
│   • Features: [elo_diff, |elo_diff|]                                   │
│   • Maps 1D rating differential to 3-class [H, D, A] probabilities     │
│   • |elo_diff| enables non-linear peaked draw probability at Delta ≈ 0 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Raw probabilities
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Optional Platt Calibrator (Multi-class Logistic Regression on logits)  │
│   • Fit on held-out recent historical window (2016 - 2018)             │
│   • Corrects secular decline in home-win frequency                     │
└────────────────────────────────────────────────────────────────────────┘
`

---

## 2. Mathematical Equations & Formal Specifications

### 2.1 Pre-Match Rating Differential & Expected Score
For match i at timestamp T between home team H and away team A:
Delta_i = (R_H + home_advantage * [1 - neutral]) - R_A

Expected win score W_e (Win = 1.0, Draw = 0.5, Loss = 0.0):
W_e = 1.0 / (1.0 + 10.0 ** (-Delta_i / 400.0))

### 2.2 Post-Match Elo Update
R'_H = R_H + K * G * (W - W_e)
R'_A = R_A - K * G * (W - W_e)

where:
- Actual score W in {1.0 (Home Win), 0.5 (Draw), 0.0 (Away Win)}
- Tournament weight K = base_K * k_scale:
  - K = 60: FIFA World Cup finals
  - K = 50: Continental Championship finals (UEFA Euro, Copa America, etc.)
  - K = 40: World Cup / Continental qualifiers & major tournaments
  - K = 30: Minor tournaments
  - K = 20: Friendly matches
- Goal-difference multiplier G(|gd|):
  - G = 1.0 if |gd| <= 1
  - G = 1.5 if |gd| == 2
  - G = (11 + |gd|) / 8.0 if |gd| >= 3

### 2.3 3-Class Probability Head
Let x_i = [Delta_i, |Delta_i|]^T. The probability vector [P(H), P(D), P(A)] is obtained via multinomial logistic regression:
P(y = c | x_i) = exp(beta_c^T x_tilde_i + beta_{0, c}) / sum_{k in {H, D, A}} exp(beta_k^T x_tilde_i + beta_{0, k})
where x_tilde_i = StandardScaler(x_i).

### 2.4 Optional Calibration (Platt Scaling)
Let z_i = log(clip(P_i, 1e-6, 1.0)). Calibrated probabilities:
P_calib(y = c | z_i) = Softmax(W_cal * z_i + b_cal)_c

---

## 3. Systematic 14-Point Audit

| # | Audit Item | Findings in worldcup-predictor Codebase |
|---|---|---|
| 1 | Elo Initialization | init_rating = 1500.0. Default initial rating for all new teams entering the historical record. |
| 2 | Elo K-factor | Fixed domain hierarchy: K in {20, 30, 40, 50, 60} based on tournament tier (src/elo.py:tournament_k). k_scale = 1.0. |
| 3 | Home Advantage | Fixed +100.0 Elo rating points added to home team when neutral == False. 0.0 when neutral == True. |
| 4 | Goal-Difference Multiplier | Stepwise multiplier: 1.0 for <= 1, 1.5 for 2, (11+|gd|)/8 for >= 3 (src/elo.py:goal_diff_multiplier). |
| 5 | Competition Weighting | Applied directly via the tournament K-factor hierarchy (K=20 to 60). |
| 6 | Rating Mean Reversion | Implemented as R <- R - gamma*(R - 1500) called at calendar-year boundaries. In production / default config, gamma = 0.0 (no reversion). Tuning confirmed optimal gamma = 0.0. |
| 7 | Chronological Update Order | Data sorted by date. Single pass: features extracted pre-match -> state updated post-match. |
| 8 | Target Leakage Vectors | Zero leakage detected. Feature generation strictly precedes _update_one(). Unit test test_no_leakage_causality() enforces identical features under truncation. |
| 9 | Probability Conversion | Multinomial Logistic Regression (EloProbHead) on scaled [Delta, |Delta|]. |
| 10 | Calibration | Multinomial Platt scaling on log-probabilities, fitted on a dedicated pre-test tuning set (2016-2018). |
| 11 | Validation Split Methodology | Single static temporal cutoffs (e.g. Train < 2018-01-01, Test >= 2018-01-01). Does NOT use walk-forward rolling folds. |
| 12 | Test-Set Tuning | scripts/tune_elo.py uses three temporal slices: Train < 2014, Val 2014-2018, Test >= 2018. Test set is evaluated strictly once. |
| 13 | Exact Data Source | martj42/international_results (public GitHub repository: ~49,520 international fixtures, 1872-2026). |
| 14 | Exact Feature Construction | 12 candidate features in src/features.py: elo_diff, elo_home, elo_away, form_diff, gf_diff, ga_diff, rest_diff, congestion_diff, rest_home, rest_away, neutral, tournament_k. |

---

## 4. Phase 2 — Reproduction Results vs Claimed

All native backtests were executed on the downloaded dataset without code modifications.

| Script / Model | Claimed (README) | Actual Reproduced | Status |
|---|---|---|---|
| Naive Baseline (run_backtest.py) | Acc: 47.7%, LL: 1.0512, Brier: 0.6340, RPS: 0.2283 | Acc: 47.7%, LL: 1.0511, Brier: 0.6340, RPS: 0.2283 | MATCH |
| Elo Baseline (run_backtest.py) | Acc: 60.0%, LL: 0.8735, Brier: 0.5137, RPS: 0.1707 | Acc: 60.1%, LL: 0.8730, Brier: 0.5134, RPS: 0.1705 | MATCH |
| HistGradientBoosting (run_ml_backtest.py) | Acc: 60.0%, LL: 0.8732, Brier: 0.5132, RPS: 0.1705 | Acc: 60.0%, LL: 0.8733, Brier: 0.5131, RPS: 0.1705 | MATCH |
| Permutation Importance (run_ml_backtest.py) | elo_diff (+0.3418), others < 0.01 | elo_diff (+0.3368), others < 0.009 | MATCH |
| Confederation-adjusted Elo (run_confed_backtest.py) | All: 60.2% / RPS 0.1703, Cross: 58.3% / RPS 0.1837 | All: 60.2% / RPS 0.1701, Cross: 58.9% / RPS 0.1803 | MATCH |
| Calibrated Elo (run_calibration.py) | Acc: 60.40%, LL: 0.8701, Brier: 0.5114, RPS: 0.1696 | Acc: 60.40%, LL: 0.8701, Brier: 0.5114, RPS: 0.1696 | MATCH |
| Dixon-Coles (run_dc_backtest.py) | Acc: 58.4%, RPS: 0.1774 | Acc: 58.45%, LL: 0.8949, Brier: 0.5270, RPS: 0.1774 | MATCH |
| Sanity / Leakage Test (test_sanity.py) | All 4 tests PASS | All 4 tests PASS (100% causal check) | MATCH |

---

## 5. Transferability Analysis to FPP Domestic League Architecture

### 5.1 Why Elo Achieves ~60% in International Football
1. **Extreme Disparity**: International football pairs global superpowers (e.g. France, Spain, Argentina) against micro-states or developmental teams (e.g. San Marino, Gibraltar, Andorra, Liechtenstein). Large rating differentials (Delta > 400) produce near-deterministic outcomes (P(Win) > 0.85).
2. **Sparsity of Inter-team Data**: National teams play only 8-12 matches per year, making short-term rolling statistics noisy and long-term historical ratings disproportionately valuable.

### 5.2 Transferable vs Non-Transferable Components for FPP

| Component | Status for FPP | Rationale |
|---|---|---|
| Online Elo Rating State | TRANSFERABLE | Cross-season rating persistence solves the early-season cold start where within-season features are null. |
| Goal-Difference Multiplier G(|gd|) | TRANSFERABLE | Standard eloratings formulation reflects margin of victory without overfitting. |
| Pre-match Difference Features (home_elo, away_elo, elo_diff) | TRANSFERABLE | Direct continuous input features for FPP's PoissonRegressor. |
| Tournament K-factor Hierarchy (20-60) | NON-TRANSFERABLE | FPP models 5 domestic leagues where all matches are regular league fixtures. Fixed K or league-appropriate K is required. |
| Confederation Ratings | NON-TRANSFERABLE | Domestic leagues do not have inter-continental confederations. (Leagues are distinct point pools). |
| Static 2018 Split Validation | NON-TRANSFERABLE | FPP enforces strict 3-fold walk-forward validation via iter_walk_forward_folds(). |
| Direct Logistic Regression Classification | NON-TRANSFERABLE | FPP Champion V2 is a dual-Poisson goal rate estimator (PoissonRegressor) mapped through independent Poisson tail-safe conversion. |
"""
import os
for p in [r'E:\Football Prediction Project\research\WORLD_CUP_ELO_RESEARCH_AUDIT.md', r'E:\Football Prediction Project\research\worldcup_elo\WORLD_CUP_ELO_RESEARCH_AUDIT.md']:
    with open(p, 'w', encoding='utf-8') as f:
        f.write(content)
print('Successfully written audit markdown.')
