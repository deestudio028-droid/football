# Updated Draw Champion 50/100-Match Comparative Validation Report

**Date:** 2026-08-21 06:50:44 UTC  
**Classification:** `REUSED_HISTORICAL_COMPARATIVE_DIAGNOSTIC`  
**Production Model:** `v4_draw_champion`  
**Production Version:** `v4.0-champion-dc-elo-stacking`  
**Methodology Hash (MD5):** `9c396e7e5364f93f079313726c1ba499`  
**Prospective Protocol Hash (MD5):** `1311eb7fa75f51c77a1fc09c0cf4df68`  

---

## 1. Executive Summary & Diagnostic Scope

This report provides a direct empirical comparison of the **Updated Frozen Draw Champion** (`v4_draw_champion`) against the original **V4 Baseline** across the two previously frozen historical validation cohorts:
1. **Existing Frozen 50-Match Cohort** ($N=50$)
2. **Existing Frozen Fresh-100 Cohort** ($N=100$)

> [!IMPORTANT]
> **Diagnostic Classification Notice:**  
> This evaluation is classified strictly as `REUSED_HISTORICAL_COMPARATIVE_DIAGNOSTIC`. The 50-match and 100-match cohorts are reused historical datasets and **MUST NOT** be counted as fresh prospective validation data. The formal prospective confirmation gate remains strictly at $N \ge 1,050$ genuinely fresh completed fixtures.

---

## 2. Comparative Scorecard: V4 Baseline vs Draw Champion vs Market Reference

### A. 50-Match Validation Cohort ($N=50$)

- **Date Range:** 2025-08-15 to 2025-08-23
- **League Composition:** Bundesliga: 7, La Liga: 13, Ligue 1: 12, Premier League: 16, Serie A: 2
- **Outcome Distribution:** Home: 25, Draw: 10, Away: 15 (Draw Rate: 0.2000)

| Metric | V4 Baseline | Updated Draw Champion | Pinnacle Closing (Ref) | Champion vs V4 (Δ) |
|---|---|---|---|---|
| **Multiclass Log Loss** | 0.988970 | **0.981750** | 0.936019 | **-0.007220** |
| **Brier Score** | 0.587532 | **0.583470** | 0.552325 | **-0.004062** |
| **Ranked Prob Score (RPS)** | 0.214612 | **0.213704** | 0.199077 | **-0.000908** |
| **Accuracy** | 0.5400 (27/50) | **0.5400** (27/50) | 0.5400 (27/50) | **+0.0000** |
| **Expected Calib Error (ECE)** | 0.0600 | **0.0550** | 0.0913 | **-0.0050** |
| **Mean P(Draw)** | 0.2399 | **0.2299** | 0.2399 | **-0.0099** |
| **Draw Prediction Bias** | +0.0399 | **+0.0299** | +0.0399 | **-0.0099** |

### B. Fresh-100 Validation Cohort ($N=100$)

- **Date Range:** 2025-08-23 to 2025-09-13
- **League Composition:** Bundesliga: 18, La Liga: 22, Ligue 1: 17, Premier League: 22, Serie A: 21
- **Outcome Distribution:** Home: 49, Draw: 22, Away: 29 (Draw Rate: 0.2200)

| Metric | V4 Baseline | Updated Draw Champion | Pinnacle Closing (Ref) | Champion vs V4 (Δ) |
|---|---|---|---|---|
| **Multiclass Log Loss** | 0.970247 | **0.964988** | 0.939937 | **-0.005259** |
| **Brier Score** | 0.575139 | **0.572183** | 0.554034 | **-0.002956** |
| **Ranked Prob Score (RPS)** | 0.203050 | **0.202345** | 0.192421 | **-0.000705** |
| **Accuracy** | 0.5400 (54/100) | **0.5400** (54/100) | 0.5800 (58/100) | **+0.0000** |
| **Expected Calib Error (ECE)** | 0.0792 | **0.0716** | 0.1289 | **-0.0076** |
| **Mean P(Draw)** | 0.2434 | **0.2385** | 0.2511 | **-0.0049** |
| **Draw Prediction Bias** | +0.0234 | **+0.0185** | +0.0311 | **-0.0049** |

---

## 3. Dedicated Draw-Specific Analysis

### A. Draw Probability Adjustments
- **50-Match Cohort:** P(D) Increased in **16 / 50** fixtures, Decreased in **34 / 50** fixtures. Mean absolute $\Delta P(D) = 0.0160$.
- **100-Match Cohort:** P(D) Increased in **50 / 100** fixtures, Decreased in **50 / 100** fixtures. Mean absolute $\Delta P(D) = 0.0140$.

### B. Draw Probability Distribution
| Cohort | Model | Min | 25% | Median | 75% | Max |
|---|---|---|---|---|---|---|
| **50-Match** | V4 Baseline | 0.1430 | 0.2332 | 0.2460 | 0.2575 | 0.2736 |
| **50-Match** | Champion | 0.0958 | 0.2166 | 0.2409 | 0.2606 | 0.2867 |
| **100-Match** | V4 Baseline | 0.0704 | 0.2344 | 0.2526 | 0.2630 | 0.2794 |
| **100-Match** | Champion | 0.0446 | 0.2108 | 0.2539 | 0.2668 | 0.2977 |

### C. Draw Probability Strata Performance (100-Match Cohort)
| Strata Bucket | Fixtures | Actual Draw Rate | V4 Mean P(D) | Champ Mean P(D) | V4 Log Loss | Champ Log Loss | Δ Log Loss |
|---|---|---|---|---|---|---|---|
| **Low P(D) (< 0.23)** | 34 | 0.1176 | 0.2134 | 0.1893 | 0.778807 | 0.758437 | **-0.020370** |
| **Med P(D) (0.23 - 0.28)** | 54 | 0.2407 | 0.2558 | 0.2587 | 1.058260 | 1.062761 | **+0.004501** |
| **High P(D) (>= 0.28)** | 12 | 0.4167 | 0.2722 | 0.2870 | 1.116601 | 1.110239 | **-0.006362** |

---

## 4. Paired Bootstrap Statistical Uncertainty (Diagnostic Only)

- **Resamples:** 10,000 (Seed: `20260820`)
- **50-Match Cohort:** Delta Log Loss = -0.007220, 95% CI [-0.017430, +0.005454] (88.8% favoring Champion, empirical p = 0.1121)
- **100-Match Cohort:** Delta Log Loss = -0.005259, 95% CI [-0.011374, +0.001157] (94.6% favoring Champion, empirical p = 0.0538)

> [!NOTE]
> As established in Phase 7 power analyses, N=50 and N=100 cohorts have statistical power of < 20%, meaning 95% confidence intervals naturally span zero. These intervals are reported strictly for descriptive transparency.

---

## 5. Diagnostic Betting Performance Simulation

| Cohort | Model | Bets Placed (Edge $\ge 5\%$) | Win Rate | Total PnL (Units) | ROI (%) | Max Drawdown (Units) |
|---|---|---|---|---|---|---|
| **50-Match** | V4 Baseline | 34 | 20.59% | -3.34 | -9.8% | 9.47 |
| **50-Match** | Champion | 35 | 20.00% | -4.34 | -12.4% | 9.68 |
| **100-Match** | V4 Baseline | 64 | 25.00% | -2.13 | -3.3% | 22.20 |
| **100-Match** | Champion | 65 | 24.62% | -3.13 | -4.8% | 22.20 |

---

## 6. Production Invariant Verification

- **Simplex Normalization ($P(H)+P(D)+P(A)=1.0$):** Verified on all 150 fixtures (**PASS**)
- **Non-Negativity ($P \ge 0$):** Verified on all 150 fixtures (**PASS**)
- **Conditional Odds Ratio Invariance ($P(H)/P(A)$ preserved):** Max deviation = `1.78e-15` (**PASS**)
- **V4 Probability Immutability:** Input V4 probabilities completely unchanged (**PASS**)
- **Determinism:** Bit-identical repeat inference (**PASS**)
- **Protected Repository Artifacts:** All 20 pinned assets verified bit-identical (**PASS**)

---

## 7. Final Diagnostic Verdict

**`DIAGNOSTIC_VALIDATION_COMPLETE — CHAMPION REPRODUCED ON HISTORICAL 50/100 COHORTS`**

- The updated Draw Champion runs with bit-identical fidelity on both historical cohorts.
- In both cohorts, the Draw Champion reduces draw prediction bias and maintains exact conditional relative odds.
- Formal production promotion remains strictly gated by the prospective requirement of $N \ge 1,050$ genuinely fresh completed fixtures.
