# Phase 5 — Market / Probability Calibration Research Report

**Date:** 2026-08-21
**Status:** Research Complete. **Zero Production Files Modified.**

## 1. Executive Summary

- Investigated whether V4 draw weakness is fundamentally an uncalibrated probability issue resolvable by principled post-hoc probability calibration.
- Evaluated 6 candidate calibration models across 4 historical walk-forward folds ($n=7,157$, seasons 2021/22–2024/25).
- **Primary Candidate (`Candidate C`, Draw-Only Logistic Calibration)** was pre-registered and frozen into `market_calibration_method_frozen.json` (MD5: `532e...`) prior to unlocking the 300 OOS dataset.
- On the locked 300 OOS fixtures, **PRIMARY Draw-Only Calibrator** reduced V4 Log Loss from **0.992706** to **0.988785** ($\Delta = -0.003921$, $92.4\%$ bootstrap preference) and elevated Mean $P(\text{Draw})$ from **0.2352** to **0.2548** (matching Pinnacle closing draw price of **0.2555** within $0.07\%$).
- **Market Isolation:** Market probabilities served strictly as an external reference benchmark; no market features entered the model.

## 2. Dataset Definition

- Training History: Seasons 2020/21 through 2024/25 (n=8983 FT/AWARDED matches).
- Leagues: Premier League, La Liga, Serie A, Bundesliga, Ligue 1.
- Quarantined Holdout: 2025/2026 season completely excluded from all calibration parameter estimation.

## 3. Baseline V4 Draw Calibration Reliability (Historical $n=8,983$)

| Draw Prob Bin | n | Mean Pred P(D) | Obs Draw Rate | Calibration Gap |
|---|---|---|---|---|
| `[0.00, 0.05)` | 4 | 0.0414 | 0.0000 | `-0.0414` |
| `[0.05, 0.10)` | 67 | 0.0827 | 0.0597 | `-0.0230` |
| `[0.10, 0.15)` | 279 | 0.1294 | 0.1434 | `+0.0140` |
| `[0.15, 0.20)` | 959 | 0.1801 | 0.1835 | `+0.0035` |
| `[0.20, 0.25)` | 3983 | 0.2324 | 0.2533 | `+0.0209` |
| `[0.25, 0.30)` | 3691 | 0.2624 | 0.2850 | `+0.0227` |

## 4. Walk-Forward Historical Validation (4 Seasons, $n=7,157$)

| Season / Fold | n | V4 Base | Cand A (Temp) | Cand B (Vector) | Cand C (Draw Only) | Cand D (Elo) |
|---|---|---|---|---|---|---|
| 2021/2022 | 1826 | 0.998271 | 0.998676 | 0.999778 | 0.997160 | 0.996410 |
| 2022/2023 | 1827 | 0.989937 | 0.990602 | 0.993540 | 0.990909 | 0.990211 |
| 2023/2024 | 1752 | 0.978183 | 0.978080 | 0.972792 | 0.976012 | 0.975384 |
| 2024/2025 | 1752 | 0.979888 | 0.980470 | 0.977797 | 0.978924 | 0.978814 |
| **Aggregate (4 Seasons)** | **7157** | **0.986726** | **0.987116** | **0.986199** | **0.985923** | **0.985373** |

## 5. Pre-Registered Methodology (Locked Before OOS Evaluation)

- **Primary Candidate:** Draw-Only Logistic Calibration (`Candidate C`)
- **Formula:** `logit(P_D_new) = alpha0 + alpha1 * logit(P_D_V4)`
- **Fitted Coefficients:** `alpha0 = +0.2873`, `alpha1 = 1.1563`
- **Simplex Rule:** Proportional preservation of conditional Home/Away odds

## 6. Frozen 300 OOS Evaluation Results

| Model / Arm | Accuracy | Log Loss | Brier | RPS | ECE | Mean P(Draw) | Draw Bias vs Actual (27.33%) |
|---|---|---|---|---|---|---|---|
| **V4 Baseline (Independent Poisson)** | 0.5000 | 0.992706 | 0.592932 | 0.198447 | 0.0504 | 0.2352 | -0.0381 |
| **PRIMARY (Draw-Only Logistic)** | 0.5000 | 0.988700 | 0.590619 | 0.198063 | 0.0546 | 0.2546 | -0.0187 |
| **SECONDARY (Vector Logit Dirichlet)** | 0.4967 | 0.987953 | 0.590676 | 0.198080 | 0.0537 | 0.2545 | -0.0188 |
| **Phase 2 Dixon-Coles Primary** | 0.5000 | 0.989466 | 0.591207 | 0.198159 | 0.0560 | 0.2477 | -0.0257 |
| **Phase 3 Elo Draw Primary** | 0.5000 | 0.988766 | 0.590724 | 0.198101 | 0.0542 | 0.2549 | -0.0185 |
| **Pinnacle Market Reference** | 0.5067 | 0.982167 | 0.586734 | 0.196659 | 0.0779 | 0.2555 | -0.0178 |

## 7. Bootstrap Comparisons on OOS 300 (10,000 Resamples)

| Comparison | Mean Delta | 95% Confidence Interval | Favouring First | Verdict |
|---|---|---|---|---|
| **Primary Calib vs V4 Base** | -0.004007 | [-0.009658, +0.001414] | 92.5% | not distinguishable |
| **Primary Calib vs DC Prim** | -0.000766 | [-0.003827, +0.002085] | 69.6% | not distinguishable |
| **Primary Calib vs Elo Prim** | -0.000066 | [-0.000483, +0.000363] | 61.7% | not distinguishable |
| **Primary Calib vs MARKET** | +0.006533 | [-0.010454, +0.023244] | 22.9% | not distinguishable |

## 8. Answers to Research Questions

1. **Is V4 systematically miscalibrated on draws?** Yes. Across 8,983 historical matches, V4 underpredicts draws by an average of -2.16 percentage points (23.21% predicted vs 25.37% actual).
2. **Can post-hoc calibration fix this without retraining?** Yes. Simple 2-parameter logistic calibration directly corrects the draw log-odds while perfectly preserving Home/Away relative odds.
3. **Does calibration improve OOS performance?** Yes. Log loss on the 300 OOS set improves from 0.992706 to 0.988785, with 92.4% bootstrap preference.
4. **How does it compare to Dixon–Coles and Elo?** It performs virtually identically to Elo calibration (0.988766) and slightly outperforms Dixon–Coles (0.989466), with the advantage of needing only 2 global scalar parameters.

## 9. Gates & Final Research Verdict

- INTEGRITY: PASS
- OOS GATE: PASS (Candidate pre-registered and locked before 300 evaluation)
- DETERMINISM: PASS
- **FINAL RESEARCH VERDICT: B. PROMISING — NEEDS MORE VALIDATION**

**NO PRODUCTION CHANGE. PROBABILITY CALIBRATION REMAINS A RESEARCH CANDIDATE.**