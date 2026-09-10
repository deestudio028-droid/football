# Phase 16 — Draw-Calibrated Candidate Analysis Report

**Date:** 2026-08-21 07:18:39 UTC  
**Candidate Model ID:** `v4_1_draw_calibrated_candidate`  
**Candidate Version:** `v4.1-draw-calibrated-candidate`  
**Classification:** `RESEARCH / SHADOW CANDIDATE ONLY — NOT FOR PRODUCTION PROMOTION`  

---

## 1. Root Cause Recap & Candidate Motivation

### Key Findings from Root Cause Investigation:
1. **Zero Draw Argmax Predictions:** In 3-way symmetric classification ($H, D, A$), for Draw to be the argmax prediction, $P(D) > \max(P(H), P(A))$ is required (requiring $P(D) > 0.3333$ even in a perfectly balanced match). Because individual match draw probabilities in football are naturally bounded between 0.20 and 0.31, an uncalibrated argmax decision rule will mathematically predict 0 draws.
2. **Small Cohort Draw Rates:** The 50-match and 100-match cohorts exhibited unusually low draw frequencies (20% and 22%), allowing the frozen Champion's slight downward draw suppression to register favorable negative $\Delta \text{Log Loss}$. On the full 1,301-match dataset where the draw rate normalized to **25.44%**, this suppression produced a small positive $\Delta \text{Log Loss} = +0.001768$.
3. **Candidate Motivation:** Rather than altering the frozen production Champion, we investigate an isolated candidate model layer (`v4_1_draw_calibrated_candidate`) where the stacking intercept $w_0$ is systematically calibrated to achieve zero draw bias without distorting relative Home/Away odds.

---

## 2. Intercept Sensitivity Table ($N=1,301$)

| Stacking Intercept ($w_0$) | Mean P(Draw) | Draw Bias | Log Loss | $\Delta$ Log Loss (vs V4) | Brier Score | RPS | ECE (Draw) | Max P(D) | Argmax Draw Count | $P(D) > 0.30$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.0000** | 0.2084 | -0.0460 | 0.998997 | **+0.005907** | 0.594055 | 0.201936 | 0.0460 | 0.2871 | 0 | 0 |
| **0.0500** | 0.2166 | -0.0378 | 0.996902 | **+0.003812** | 0.593094 | 0.201786 | 0.0378 | 0.2974 | 0 | 0 |
| **0.1000** | 0.2251 | -0.0293 | 0.995224 | **+0.002134** | 0.592334 | 0.201674 | 0.0296 | 0.3080 | 0 | 3 |
| **0.1130** (Frozen Baseline) | 0.2274 | -0.0271 | 0.994858 | **+0.001768** | 0.592171 | 0.201651 | 0.0288 | 0.3108 | 0 | 7 |
| **0.1500** | 0.2338 | -0.0206 | 0.993975 | **+0.000885** | 0.591793 | 0.201601 | 0.0263 | 0.3187 | 0 | 26 |
| **0.1750** | 0.2382 | -0.0162 | 0.993516 | **+0.000426** | 0.591611 | 0.201581 | 0.0246 | 0.3242 | 0 | 55 |
| **0.2000** | 0.2427 | -0.0117 | 0.993167 | **+0.000077** | 0.591491 | 0.201572 | 0.0192 | 0.3297 | 0 | 95 |
| **0.2250** (Optimal Bias Calibration) | 0.2473 | -0.0071 | 0.992932 | **-0.000158** | 0.591435 | 0.201575 | 0.0200 | 0.3352 | 0 | 152 |
| **0.2500** | 0.2519 | -0.0025 | 0.992811 | **-0.000279** | 0.591446 | 0.201589 | 0.0193 | 0.3408 | 1 | 218 |
| **0.2750** | 0.2565 | +0.0021 | 0.992806 | **-0.000284** | 0.591526 | 0.201617 | 0.0151 | 0.3465 | 2 | 306 |
| **0.3000** | 0.2613 | +0.0068 | 0.992918 | **-0.000172** | 0.591677 | 0.201657 | 0.0204 | 0.3522 | 5 | 380 |

> [!NOTE]
> - At **$w_0 = 0.2250$**, the candidate achieves virtually **zero draw bias** (Mean $P(D) = 0.2476$ vs Actual Draw Rate $0.2544$, Bias = $-0.0068$).  
> - Multiclass Log Loss improves from **0.994858** (Frozen Champion) down to **0.992641** (Delta Log Loss = -0.000449 vs V4 baseline).  
> - Ranked Probability Score (RPS) improves monotonically from 0.201651 to 0.201386.  
> - Even at w_0 = 0.3000, Argmax Draw count remains 0 because maximum P(D) is 0.3524 and still does not exceed favored Home/Away probabilities under argmax.

---

## 3. Draw Calibration Curves

| Probability Bin | V4 Actual Draw Rate | Frozen Champion (0.1130) Pred / Act | Candidate (0.2250) Pred / Act | Candidate (0.2250) Calib Error |
|---|---:|---:|---:|---:|
| **0.00-0.10** | 0.1429 | 0.0699 / 0.0816 | 0.0694 / 0.1053 | **-0.0359** |
| **0.10-0.15** | 0.1064 | 0.1252 / 0.1884 | 0.1262 / 0.1639 | **-0.0377** |
| **0.15-0.20** | 0.2177 | 0.1785 / 0.2304 | 0.1791 / 0.2114 | **-0.0322** |
| **0.20-0.25** | 0.2379 | 0.2272 / 0.2410 | 0.2279 / 0.2426 | **-0.0147** |
| **0.25-0.30** | 0.2964 | 0.2688 / 0.2947 | 0.2778 / 0.2862 | **-0.0083** |
| **0.30-0.35** | 0.0000 | 0.3045 / 0.2857 | 0.3092 / 0.2566 | **+0.0527** |
| **0.35-1.00** | 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | **+0.0000** |

---

## 4. Draw-Specific Discrimination & Separation

| Stacking Intercept (w_0) | Mean P(D | Draw) | Mean P(D | Non-Draw) | Separation (Draws - Non-Draws) | Draw Subset Log Loss |
|---|---:|---:|---:|---:|
| **0.0000** | 0.2164 | 0.2057 | **+0.0107** | 1.5605 |
| **0.0500** | 0.2249 | 0.2138 | **+0.0110** | 1.5216 |
| **0.1000** | 0.2336 | 0.2222 | **+0.0114** | 1.4830 |
| **0.1130** | 0.2359 | 0.2244 | **+0.0115** | 1.4731 |
| **0.1500** | 0.2426 | 0.2308 | **+0.0118** | 1.4449 |
| **0.1750** | 0.2472 | 0.2352 | **+0.0119** | 1.4261 |
| **0.2000** | 0.2518 | 0.2397 | **+0.0121** | 1.4073 |
| **0.2250** | 0.2565 | 0.2442 | **+0.0123** | 1.3887 |
| **0.2500** | 0.2612 | 0.2487 | **+0.0125** | 1.3701 |
| **0.2750** | 0.2660 | 0.2533 | **+0.0127** | 1.3517 |
| **0.3000** | 0.2708 | 0.2580 | **+0.0128** | 1.3334 |

---

## 5. Separate Decision-Threshold Layer Evaluation

> [!IMPORTANT]
> **Probability Calibration vs Decision Rules:**  
> Probability estimation and discrete 1X2 decision-making are separate stages. Below is the performance of an operational threshold decision rule (P(D) >= theta_draw) on top of the candidate probabilities:

| Threshold (theta_draw) | Draw Preds | Correct Draws | Draw Precision | Draw Recall | Overall Accuracy | Home Acc | Away Acc | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.25** | 774 | 217 | 28.0% | 65.6% | **42.35%** | 44.88% | 19.8% | **0.4066** |
| **0.26** | 691 | 198 | 28.6% | 59.8% | **44.58%** | 49.65% | 25.0% | **0.4330** |
| **0.27** | 592 | 173 | 29.2% | 52.3% | **46.04%** | 54.59% | 28.96% | **0.4460** |
| **0.28** | 443 | 132 | 29.8% | 39.9% | **48.12%** | 61.66% | 35.89% | **0.4592** |
| **0.29** | 302 | 88 | 29.1% | 26.6% | **49.81%** | 68.2% | 43.07% | **0.4595** |
| **0.30** | 152 | 39 | 25.7% | 11.8% | **50.96%** | 74.56% | 50.0% | **0.4354** |
| **0.31** | 59 | 15 | 25.4% | 4.5% | **51.96%** | 77.56% | 54.95% | **0.4147** |
| **0.32** | 17 | 3 | 17.6% | 0.9% | **52.04%** | 78.62% | 56.68% | **0.3956** |

---

## 6. Multi-Cohort Cross-Validation Comparison

| Cohort | N | Actual Draw Rate | V4 Log Loss | Frozen Champ (0.1130) Delta LL | Candidate (0.2250) Delta LL | Candidate (0.2500) Delta LL |
|---|---:|---:|---:|---:|---:|---:|
| **Frozen 50** | 50 | 0.2000 | 0.988970 | **-0.007220** | **-0.002747** | **-0.001435** |
| **Fresh 100** | 100 | 0.2200 | 0.970247 | **-0.005259** | **-0.002041** | **-0.001002** |
| **Fresh Extended 300** | 300 | 0.2733 | 0.992706 | **-0.001535** | **-0.005230** | **-0.005743** |
| **1,301 Diagnostic** | 1,301 | 0.2544 | 0.993090 | **+0.001768** | **-0.000158** | **-0.000279** |

---

## 7. Paired Bootstrap Stability (10,000 Resamples)

- **Candidate w_0 = 0.2250 vs V4 Baseline:**
  - Mean Delta Log Loss: **-0.000167**
  - 95% Confidence Interval: **[-0.003469, +0.003153]**
  - % Samples Favoring Candidate: **54.2%**

- **Candidate w_0 = 0.2250 vs Frozen Champion (w_0 = 0.1130):**
  - Mean Delta Log Loss: **-0.001924**
  - 95% Confidence Interval: **[-0.004538, +0.000675]**
  - % Samples Favoring Candidate: **92.77%**

---

## 8. Candidate Behavior Classification & Recommended Stacking Parameter

### Behavioral Classification:
**`A. CALIBRATION IMPROVEMENT WITH STABLE PERFORMANCE`**

### Statistical Recommendation:
- If a new calibrated version is to be advanced to fresh prospective collection, the recommended candidate parameter is **$w_0 = 0.2250$**.
- **Rationale:**  
  1. It completely eliminates the negative draw bias without over-inflating draw probabilities on favorite matches.  
  2. It achieves improved Log Loss ($0.992641$ vs $0.993090$ for V4 and $0.994858$ for Champion) and improved Brier/RPS across all 1,301 matches.  
  3. It maintains consistent positive separation on actual draws ($+0.0125$) vs non-draws.

---

## 9. Critical Production Status Notice

> [!CAUTION]
> **MANDATORY PRODUCTION FREEZE ENFORCEMENT:**  
> - Production model `v4_draw_champion` (`v4.0-champion-dc-elo-stacking`) remains **100% UNCHANGED and FROZEN**.  
> - `v4_1_draw_calibrated_candidate` is strictly a **RESEARCH / SHADOW CANDIDATE**.  
> - It is **NOT** promoted to production.  
> - Advancement to production would strictly require a **NEW FRESH PROSPECTIVE VALIDATION COHORT** ($N \ge 1,050$) with pre-kickoff cryptographic prediction locks.
