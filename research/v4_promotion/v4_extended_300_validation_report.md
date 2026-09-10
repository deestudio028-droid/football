# V4 Fresh-Extended-300 OOS Validation Report

**Date:** 2026-08-21
**Type:** Strictly observational out-of-sample validation. **V4 is NOT promoted.**

## Sample

- Selection: `LIMIT 300 OFFSET 150` on the 2025/26 FT fixtures, chronological
- Frozen before evaluation, MD5 `0526bfd6980dd51dae59c6f6aadab2f5`
- **Zero overlap** with the frozen 20, frozen 50 or fresh 100
- Date span: 2025-09-13 to 2025-11-01
- Composition: Serie A 68, La Liga 67, Ligue 1 61, Premier League 52, Bundesliga 52
- Outcomes: H=131 D=82 A=87

## Scorecard

Direction: Accuracy ↑ · LogLoss ↓ · Brier ↓ · RPS ↓ · Draw Recall ↑ · ECE ↓

| Model | Correct | Accuracy | LogLoss | Brier | RPS | Draw Recall | ECE |
|---|---|---|---|---|---|---|---|
| V2 | 145/300 | 0.4833 | 1.003707 | 0.600782 | 0.202106 | 0.0000 | 0.027414 |
| V3 | 149/300 | 0.4967 | 0.994799 | 0.594451 | 0.199218 | 0.0000 | 0.026417 |
| V4 | 150/300 | 0.5000 | 0.992706 | 0.592932 | 0.198447 | 0.0000 | 0.033930 |
| MARKET | 152/300 | 0.5067 | 0.982167 | 0.586734 | 0.196659 | 0.0000 | 0.039793 |

## V4 Pairwise (positive = V4 better)

| Comparison | LogLoss | Brier | RPS | ECE | Accuracy | Correct |
|---|---|---|---|---|---|---|
| V4 vs V2 | +0.011001 | +0.007850 | +0.003659 | -0.006516 | +0.0167 | +5 |
| V4 vs V3 | +0.002093 | +0.001519 | +0.000771 | -0.007513 | +0.0033 | +1 |
| V4 vs MARKET | -0.010539 | -0.006198 | -0.001788 | +0.005863 | -0.0067 | -2 |

## Chronological Buckets (6 x 50)

| Bucket | n | V2 | V3 | V4 | Market |
|---|---|---|---|---|---|
| 1-50 | 50 | 26/50 (0.9466) | 29/50 (0.9265) | 29/50 (0.9229) | 31/50 (0.8999) |
| 51-100 | 50 | 21/50 (1.0128) | 19/50 (1.0096) | 19/50 (1.0080) | 21/50 (1.0018) |
| 101-150 | 50 | 26/50 (0.9721) | 27/50 (0.9584) | 27/50 (0.9546) | 29/50 (0.9484) |
| 151-200 | 50 | 24/50 (1.0346) | 25/50 (1.0293) | 25/50 (1.0286) | 23/50 (1.0095) |
| 201-250 | 50 | 27/50 (0.9991) | 28/50 (0.9959) | 28/50 (0.9941) | 25/50 (0.9877) |
| 251-300 | 50 | 21/50 (1.0571) | 21/50 (1.0492) | 22/50 (1.0480) | 23/50 (1.0457) |

Bucket-level differences are descriptive; n=50 per bucket cannot support a drift claim.

## Per-League (DESCRIPTIVE ONLY)

| League | n | V2 acc | V3 acc | V4 acc | Market acc | V4 LL | Market LL |
|---|---|---|---|---|---|---|---|
| Serie A | 68 | 0.456 | 0.441 | 0.441 | 0.456 | 1.04385 | 1.01900 |
| La Liga | 67 | 0.493 | 0.537 | 0.552 | 0.537 | 0.95756 | 0.95776 |
| Ligue 1 | 61 | 0.459 | 0.459 | 0.459 | 0.475 | 1.05433 | 1.02567 |
| Premier League | 52 | 0.519 | 0.500 | 0.500 | 0.519 | 0.97979 | 0.99117 |
| Bundesliga | 52 | 0.500 | 0.558 | 0.558 | 0.558 | 0.91173 | 0.90541 |

No per-league conclusion is drawn and no per-league selection is made from these numbers.

## Longitudinal: 50 vs 100 vs 300

| Model | 50 acc | 50 LL | 100 acc | 100 LL | 300 acc | 300 LL |
|---|---|---|---|---|---|---|
| V2 | 0.5200 | 1.0199 | 0.5100 | 0.9898 | 0.4833 | 1.003707 |
| V3 | 0.5400 | 0.9978 | 0.5500 | 0.9739 | 0.4967 | 0.994799 |
| V4 | 0.5400 | 0.9890 | 0.5400 | 0.9702 | 0.5000 | 0.992706 |
| MARKET | 0.5400 | 0.9360 | 0.5800 | 0.9399 | 0.5067 | 0.982167 |

## V4 vs V3 Probability Stability

- Mean absolute probability change: 0.00528067
- Max absolute probability change: 0.02935030
- Materially changed (>0.01): 84/300 (28.00%)
- Top-class changes: 2/300

## Bootstrap (10,000 resamples, seed 20260820)

| Comparison | Mean delta | 95% CI | Favouring V4 | Verdict |
|---|---|---|---|---|
| V4_minus_V2 | -0.011001 | [-0.018631, -0.003363] | 99.8% | V4 better |
| V4_minus_V3 | -0.002092 | [-0.004118, +0.000015] | 97.4% | not distinguishable |
| V4_minus_MARKET | +0.010539 | [-0.007429, +0.028351] | 12.7% | not distinguishable |

## Gates

- INTEGRITY: PASS
- CAUSALITY: PASS (max diff 0.000e+00 across cut points [1, 51, 101, 151, 201, 251, 300])
- DETERMINISM: PASS
- MARKET COVERAGE: PASS (300/300)
- **OVERALL: COMPLETE**

## Promotion Decision

Rule applied: do not promote merely because V4 beats V2; treat a bootstrap CI crossing zero as NOT statistically distinguishable.

- V4 significantly better than V3: **False**
- V4 significantly better than V2: **True**
- Evidence sufficient for promotion: **False**

- V4 vs V3 is NOT STATISTICALLY DISTINGUISHABLE.
- V4 vs MARKET is NOT STATISTICALLY DISTINGUISHABLE.

**V4 IS NOT PROMOTED. No production file was modified.**