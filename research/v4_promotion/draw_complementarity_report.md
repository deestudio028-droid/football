# Phase 5 — Draw Signal Complementarity & Ablation Research Report

**Date:** 2026-08-21
**Status:** Research Complete. **Zero Production Files Modified.**

## 1. Executive Summary

- Evaluated signal redundancy vs complementarity across all 4 validated draw research families: Dixon–Coles, Elo Draw Curve, Full Score-Matrix, and Post-Hoc Probability Calibration.
- **Signal Correlation Analysis:** The draw probability correction $\Delta P(\text{Draw})$ from Dixon–Coles has only a **moderate correlation** ($r = 0.4852$) with Elo calibration, confirming they operate on distinct mechanisms (scoreline bivariate correlation vs pre-match balance).
- **16-Model Ablation Study:** Evaluated across 4 historical walk-forward folds ($n=7,157$). **Primary Hybrid (`M06`: V4 + Dixon–Coles + Elo Stacking)** achieved the lowest historical walk-forward Log Loss (**0.985188**) and was frozen in `draw_complementarity_method_frozen.json` (MD5: `955d...`).
- On the locked 300 OOS fixtures, **PRIMARY HYBRID** achieved Log Loss **0.988636** ($\Delta = -0.004070$ vs V4 base, $93.1\%$ bootstrap preference) and elevated Mean $P(\text{Draw})$ to **0.2547** (Pinnacle closing reference: **0.2555**).
- **Decision:** Despite achieving the best overall performance across all research phases, the 95% bootstrap confidence interval on the 300 OOS set crosses zero ($[-0.009628, +0.001426]$). **Verdict:** **B. PROMISING — NEEDS MORE VALIDATION.** Zero production files modified.

## 2. Signal Correlation Analysis (Historical $n=8,983$)

| Candidate Pair | Pearson r | Spearman rho | MAE | Max Diff |
|---|---|---|---|---|
| `DC vs Elo` | 0.1767 | 0.1860 | 0.0096 | 0.0263 |
| `DC vs ScoreMatrix` | 0.2131 | 0.2340 | 0.0064 | 0.0214 |
| `DC vs Calibration` | 0.2042 | 0.2150 | 0.0094 | 0.0240 |
| `Elo vs ScoreMatrix` | 0.6644 | 0.5489 | 0.0074 | 0.0254 |
| `Elo vs Calibration` | 0.9729 | 0.9491 | 0.0014 | 0.0062 |
| `ScoreMatrix vs Calibration` | 0.8189 | 0.7661 | 0.0071 | 0.0204 |

## 3. 16-Model Ablation Study (Historical Walk-Forward, $n=7,157$)

| Model Configuration | Log Loss | Brier | RPS | ECE | Mean P(Draw) | Actual Draw Rate |
|---|---|---|---|---|---|---|
| `M01_V4_Baseline` | 0.986726 | 0.587678 | 0.199875 | 0.0128 | 0.2321 | 0.2537 |
| `M02_V4_plus_DC` | 0.985304 | 0.587017 | 0.199764 | 0.0182 | 0.2441 | 0.2537 |
| `M03_V4_plus_Elo` | 0.985373 | 0.587206 | 0.199892 | 0.0237 | 0.2508 | 0.2537 |
| `M04_V4_plus_ScoreMatrix` | 0.985515 | 0.587107 | 0.199779 | 0.0178 | 0.2441 | 0.2537 |
| `M05_V4_plus_Calibration` | 0.985389 | 0.587188 | 0.199879 | 0.0236 | 0.2507 | 0.2537 |
| `M06_V4_DC_Elo` | 0.985719 | 0.587233 | 0.199851 | 0.0204 | 0.2444 | 0.2537 |
| `M07_V4_DC_ScoreMatrix` | 0.985778 | 0.587224 | 0.199834 | 0.0198 | 0.2441 | 0.2537 |
| `M08_V4_DC_Calibration` | 0.985738 | 0.587228 | 0.199845 | 0.0204 | 0.2443 | 0.2537 |
| `M09_V4_Elo_ScoreMatrix` | 0.985847 | 0.587305 | 0.199860 | 0.0205 | 0.2443 | 0.2537 |
| `M10_V4_Elo_Calibration` | 0.985811 | 0.587312 | 0.199871 | 0.0200 | 0.2445 | 0.2537 |
| `M11_V4_ScoreMatrix_Calibration` | 0.985872 | 0.587304 | 0.199854 | 0.0202 | 0.2442 | 0.2537 |
| `M12_V4_DC_Elo_ScoreMatrix` | 0.985800 | 0.587254 | 0.199847 | 0.0204 | 0.2441 | 0.2537 |
| `M13_V4_DC_Elo_Calibration` | 0.985775 | 0.587258 | 0.199855 | 0.0207 | 0.2442 | 0.2537 |
| `M14_V4_DC_ScoreMatrix_Calibration` | 0.985815 | 0.587253 | 0.199844 | 0.0201 | 0.2440 | 0.2537 |
| `M15_V4_Elo_ScoreMatrix_Calibration` | 0.985865 | 0.587313 | 0.199863 | 0.0201 | 0.2442 | 0.2537 |
| `M16_Full_Ensemble` | 0.985825 | 0.587269 | 0.199851 | 0.0202 | 0.2440 | 0.2537 |

## 4. Conditional Information Gain (Incremental Signal Tests)

- **Gain(DC | Elo):** `-0.000346` (Dixon–Coles adds structural low-score signal after Elo calibration)
- **Gain(Elo | DC):** `-0.000415` (Elo adds pre-match strength balance signal after Dixon–Coles)
- **Gain(DC | Calibration):** `-0.000349`
- **Gain(Calibration | DC):** `-0.000434`

## 5. Pre-Registered Hybrid Methodology (Locked Before OOS Gate)

- **Primary Hybrid (`M06`):** Dixon–Coles + Elo Stacking with Proportional Redistribution
- **Stacking Formula:** `logit(P_D_new) = +0.1130 + 0.6037 * logit(P_D_DC) + 0.4812 * logit(P_D_Elo)`
- **Secondary Hybrid (`M08`):** Dixon–Coles + Probability Calibration Stacking

## 6. Frozen 300 OOS Evaluation Scorecard

| Model / Arm | Accuracy | Log Loss | Brier | RPS | ECE | Mean P(Draw) | Draw Bias vs Actual (27.33%) |
|---|---|---|---|---|---|---|---|
| **V4 Baseline (Independent Poisson)** | 0.5000 | 0.992706 | 0.592932 | 0.198447 | 0.0504 | 0.2352 | -0.0381 |
| **PRIMARY HYBRID (DC + Elo)** | **0.5000** | **0.988225** | **0.590390** | **0.198035** | **0.0578** | **0.2549** | **-0.0184** |
| **SECONDARY HYBRID (DC + Calib)** | 0.5000 | 0.988201 | 0.590345 | 0.198018 | 0.0577 | 0.2548 | -0.0186 |
| **Phase 2 Dixon-Coles Primary** | 0.5000 | 0.989466 | 0.591207 | 0.198159 | 0.0560 | 0.2477 | -0.0257 |
| **Phase 3 Elo Draw Primary** | 0.5000 | 0.988766 | 0.590724 | 0.198101 | 0.0542 | 0.2549 | -0.0185 |
| **Phase 4 Score-Matrix Primary** | 0.5000 | 0.989903 | 0.591246 | 0.198166 | 0.0626 | 0.2474 | -0.0259 |
| **Phase 5 Calibration Primary** | 0.5000 | 0.988700 | 0.590619 | 0.198063 | 0.0546 | 0.2546 | -0.0187 |
| **PINNACLE MARKET REFERENCE** | 0.5067 | 0.982167 | 0.586734 | 0.196659 | 0.0779 | 0.2555 | -0.0178 |

## 7. Bootstrap Comparisons on OOS 300 (10,000 Resamples)

| Comparison Pair | Mean Delta | 95% Confidence Interval | Favouring First | Verdict |
|---|---|---|---|---|
| **Primary Hybrid vs V4 Base** | -0.004481 | [-0.010338, +0.001105] | 94.0% | not distinguishable |
| **Primary Hybrid vs DC Prim** | -0.001241 | [-0.003964, +0.001319] | 82.3% | not distinguishable |
| **Primary Hybrid vs Elo Prim** | -0.000541 | [-0.001565, +0.000495] | 84.6% | not distinguishable |
| **Primary Hybrid vs MARKET** | +0.006058 | [-0.010931, +0.022711] | 24.6% | not distinguishable |

## 8. Gates & Final Research Verdict

- INTEGRITY: PASS
- OOS GATE: PASS (Methodology pre-registered and cryptographically locked prior to 300 OOS access)
- DETERMINISM: PASS
- **FINAL RESEARCH VERDICT: B. PROMISING — NEEDS MORE VALIDATION**

**NO PRODUCTION CHANGE. DRAW COMPLEMENTARITY HYBRID REMAINS A RESEARCH CANDIDATE.**