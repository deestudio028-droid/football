# V4 Fresh-100 OOS Validation Report

**Date:** 2026-08-21
**Type:** Strictly observational out-of-sample validation. **V4 is NOT promoted.**

## Sample

- Selection: `LIMIT 100 OFFSET 50` on the 2025/26 FT fixtures, chronological
- Frozen before evaluation, MD5 `761ad5cc571643e6985e671bd9c3d83a`
- **Zero overlap** with the frozen 20 or frozen 50
- Date span: 2025-08-23 to 2025-09-13
- Composition: La Liga 22, Premier League 22, Serie A 21, Bundesliga 18, Ligue 1 17
- Outcomes: H=49 D=22 A=29

## Scorecard

Direction: Accuracy ↑ · LogLoss ↓ · Brier ↓ · RPS ↓ · Draw Recall ↑ · ECE ↓

| Model | Correct | Accuracy | LogLoss | Brier | RPS | Draw Recall | ECE |
|---|---|---|---|---|---|---|---|
| V2 | 51/100 | 0.5100 | 0.989843 | 0.589570 | 0.209986 | 0.0000 | 0.031239 |
| V3 | 55/100 | 0.5500 | 0.973907 | 0.577915 | 0.204401 | 0.0000 | 0.039763 |
| V4 | 54/100 | 0.5400 | 0.970247 | 0.575139 | 0.203050 | 0.0000 | 0.043411 |
| MARKET | 58/100 | 0.5800 | 0.939937 | 0.554034 | 0.192421 | 0.0000 | 0.053511 |

## V4 Pairwise (positive = V4 better)

| Comparison | LogLoss | Brier | RPS | ECE | Accuracy | Correct |
|---|---|---|---|---|---|---|
| V4 vs V2 | +0.019596 | +0.014431 | +0.006936 | -0.012172 | +0.0300 | +3 |
| V4 vs V3 | +0.003660 | +0.002776 | +0.001351 | -0.003648 | -0.0100 | -1 |
| V4 vs MARKET | -0.030310 | -0.021105 | -0.010629 | +0.010100 | -0.0400 | -4 |

## Chronological Buckets

| Bucket | n | V2 | V3 | V4 | Market |
|---|---|---|---|---|---|
| 1-20 | 20 | 11/20 (1.0197) | 11/20 (1.0199) | 10/20 (1.0214) | 10/20 (0.9969) |
| 21-40 | 20 | 13/20 (0.9347) | 14/20 (0.8938) | 14/20 (0.8831) | 14/20 (0.8200) |
| 41-60 | 20 | 6/20 (1.0330) | 8/20 (1.0070) | 8/20 (1.0029) | 9/20 (1.0082) |
| 61-80 | 20 | 9/20 (1.0326) | 10/20 (1.0236) | 10/20 (1.0210) | 12/20 (0.9710) |
| 81-100 | 20 | 12/20 (0.9292) | 12/20 (0.9253) | 12/20 (0.9228) | 13/20 (0.9035) |

## Per-League (descriptive only)

| League | n | V2 acc | V3 acc | V4 acc | Market acc | V4 LL | Market LL |
|---|---|---|---|---|---|---|---|
| La Liga | 22 | 0.500 | 0.545 | 0.545 | 0.545 | 0.98830 | 0.97975 |
| Premier League | 22 | 0.364 | 0.364 | 0.364 | 0.455 | 1.04159 | 1.02467 |
| Serie A | 21 | 0.571 | 0.667 | 0.667 | 0.619 | 0.99670 | 0.99963 |
| Bundesliga | 18 | 0.500 | 0.500 | 0.500 | 0.611 | 0.96068 | 0.88648 |
| Ligue 1 | 17 | 0.647 | 0.706 | 0.647 | 0.706 | 0.83200 | 0.76162 |

## Bootstrap (10,000 resamples, seed 20260820)

| Comparison | Mean delta | 95% CI | Favouring V4 | Verdict |
|---|---|---|---|---|
| V4_minus_V2 | -0.019596 | [-0.038155, -0.000571] | 97.7% | V4 better |
| V4_minus_V3 | -0.003660 | [-0.008578, +0.001432] | 92.1% | not distinguishable |
| V4_minus_MARKET | +0.030311 | [-0.007758, +0.067944] | 6.0% | not distinguishable |

## Gates

- INTEGRITY: PASS
- CAUSALITY: PASS (max diff 0.000e+00 across cut points [1, 21, 41, 61, 81, 100])
- DETERMINISM: PASS
- MARKET COVERAGE: PASS (100/100)
- **OVERALL: COMPLETE**

**V4 is NOT promoted. No production file was modified.**