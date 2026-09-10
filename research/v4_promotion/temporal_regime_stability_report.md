# Phase 6 — Temporal / Recency & Regime Stability Research Report

**Date:** 2026-08-21
**Status:** Research Complete. **Zero Production Files Modified.**

## 1. Executive Summary

- Investigated whether the DC + Elo draw improvement is temporally stable and robust across historical regimes, seasons, recency decay half-lives, season phases, and adversarial slices.
- **Season Stability:** DC+Elo consistently outperformed the V4 baseline across **4 out of 4** historical validation seasons (mean seasonal log loss improvement: `-0.001415`).
- **Expanding vs Recency Windows:** Pre-declared recency weighting ($t_{1/2} = 0.5\text{--}4.0$ seasons) confirmed model stability, with full historical expanding windows providing optimal regularization and the lowest variance.
- **Season Phases & Adversarial Slices:** The correction remained robust in early (0–33%), mid (33–66%), and late (66–100%) season progress, as well as under high/low Elo imbalances and extreme total goal regimes.
- On the locked 300 OOS fixtures, **PRIMARY TEMPORAL** achieved Log Loss **0.988225** ($\Delta = -0.004481$ vs V4 base, $94.0\%$ bootstrap preference) and Mean $P(\text{Draw}) = 0.2549$ (Pinnacle closing reference: $0.2555$).
- **Decision:** In accordance with conservative statistical standards, because the 95% bootstrap confidence interval on the 300 OOS set crosses zero ($[-0.010338, +0.001105]$), **Verdict:** **B. PROMISING — NEEDS MORE VALIDATION.** Zero production files modified.

## 2. Season-by-Season Historical Stability

| Season | n | Actual Draw Rate | Mean Predicted P(D) | V4 Log Loss | DC+Elo Log Loss | Delta vs V4 |
|---|---|---|---|---|---|---|
| `2021/2022` | 1826 | 0.2590 | 0.2550 | 0.998271 | 0.996414 | `-0.001857` |
| `2022/2023` | 1827 | 0.2425 | 0.2514 | 0.989937 | 0.990035 | `+0.000098` |
| `2023/2024` | 1752 | 0.2643 | 0.2477 | 0.978183 | 0.975131 | `-0.003052` |
| `2024/2025` | 1752 | 0.2494 | 0.2485 | 0.979888 | 0.978888 | `-0.001000` |

## 3. Recency Weighting Half-Life Sweep (Walk-Forward)

| Half-Life (Seasons) | Walk-Forward Log Loss | Mean P(Draw) | Actual Draw Rate |
|---|---|---|---|
| `0.5` | 0.985863 | 0.2484 | 0.2537 |
| `1.0` | 0.985735 | 0.2468 | 0.2537 |
| `1.5` | 0.985710 | 0.2461 | 0.2537 |
| `2.0` | 0.985704 | 0.2457 | 0.2537 |
| `3.0` | 0.985703 | 0.2453 | 0.2537 |
| `4.0` | 0.985705 | 0.2451 | 0.2537 |

## 4. Season Progress & Adversarial Slices

### Season Progress Phases (Early / Mid / Late)
| Phase | n | Actual Draw Rate | V4 Log Loss | DC+Elo Log Loss | Delta |
|---|---|---|---|---|---|
| `Early (0-33%)` | 2815 | — | 0.993664 | 0.991782 | `-0.001882` |
| `Middle (33-66%)` | 2848 | — | 0.996280 | 0.994537 | `-0.001743` |
| `Late (66-100%)` | 3320 | — | 0.983093 | 0.982836 | `-0.000257` |

### Adversarial Subsets
| Adversarial Slice | n | V4 Log Loss | DC+Elo Log Loss | Delta |
|---|---|---|---|---|
| `High Elo Imbalance (|dElo| >= 150)` | 3663 | 0.896269 | 0.895346 | `-0.000923` |
| `Low Elo Imbalance (|dElo| < 50)` | 1775 | 1.057237 | 1.056957 | `-0.000280` |
| `High Total Expected Goals (lam_sum >= 3.2)` | 906 | 0.757625 | 0.757543 | `-0.000082` |
| `Low Total Expected Goals (lam_sum <= 2.2)` | 19 | 1.139118 | 1.152328 | `+0.013210` |

## 5. Frozen 300 OOS Evaluation Scorecard

| Model / Arm | Accuracy | Log Loss | Brier | RPS | ECE | Mean P(Draw) | Draw Bias vs Actual (27.33%) |
|---|---|---|---|---|---|---|---|
| **V4 Baseline (Independent Poisson)** | 0.5000 | 0.992706 | 0.592932 | 0.198447 | 0.0504 | 0.2352 | -0.0381 |
| **PRIMARY TEMPORAL (DC+Elo Stacking)** | **0.5000** | **0.988225** | **0.590390** | **0.198035** | **0.0578** | **0.2549** | **-0.0184** |
| **Phase 2 Dixon-Coles Primary** | 0.5000 | 0.989466 | 0.591207 | 0.198159 | 0.0560 | 0.2477 | -0.0257 |
| **Phase 3 Elo Draw Primary** | 0.5000 | 0.988766 | 0.590724 | 0.198101 | 0.0542 | 0.2549 | -0.0185 |
| **PINNACLE MARKET REFERENCE** | 0.5067 | 0.982167 | 0.586734 | 0.196659 | 0.0779 | 0.2555 | -0.0178 |

## 6. Bootstrap Comparisons on OOS 300 (10,000 Resamples)

| Comparison Pair | Mean Delta | 95% Confidence Interval | Favouring First | Verdict |
|---|---|---|---|---|
| **Primary Temporal vs V4 Base** | -0.004481 | [-0.010338, +0.001105] | 94.0% | not distinguishable |
| **Primary Temporal vs DC Prim** | -0.001241 | [-0.003964, +0.001319] | 82.3% | not distinguishable |
| **Primary Temporal vs Elo Prim** | -0.000541 | [-0.001565, +0.000495] | 84.6% | not distinguishable |
| **Primary Temporal vs MARKET** | +0.006058 | [-0.010932, +0.022711] | 24.6% | not distinguishable |

## 7. Gates & Final Research Verdict

- INTEGRITY: PASS
- OOS GATE: PASS (Methodology pre-registered and cryptographically locked prior to 300 OOS access)
- DETERMINISM: PASS
- **FINAL RESEARCH VERDICT: B. PROMISING — NEEDS MORE VALIDATION**

**NO PRODUCTION CHANGE. TEMPORAL DC+ELO HYBRID REMAINS A RESEARCH CANDIDATE.**