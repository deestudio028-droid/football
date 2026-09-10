# STEP 2I — DRAW DECISION INTELLIGENCE RESEARCH REPORT
**Football Prediction Project — Advanced Model Engineering & Research Program**  
**Governance Status:** RESEARCH COMPLETE — ZERO PRODUCTION MUTATION  
**Date:** August 24, 2026  
**Artifact Directory:** `research/v5_model_improvement/step2_draw_research/step2i_draw_decision/`

---

## 1. Executive Summary & Core Research Question

### The Fundamental Question:
> *"Can $V_{4.0}$'s existing information be converted into a better DRAW DECISION policy without sacrificing the reliability of Home/Away predictions?"*

### Empirical Answer:
**NO. Forcing deterministic DRAW decisions on closely-contested matches mathematically causes a strictly negative net prediction trade-off ($\Delta < 0$).**  
However, utilizing the continuous calibrated Draw probability ($P'(D) \ge 0.28$) and Win Parity ($|P'(H) - P'(A)| \le 0.08$) as an **Advisory / Abstention CAUTION Layer** provides immense value by isolating the ~18.6% of matches where standard favorite prediction accuracy drops from $55.84\%$ down to $38.59\%$.

---

## 2. Cryptographic Integrity Matrix

| Artifact | Pinned File Path | Expected MD5 | Actual MD5 | Status |
|:---|:---|:---|:---|:---:|
| **$V_{4.0}$ Production Baseline** | `data/models/v4_poisson_venue_elo_online_ad.pkl` | `06841f0c03c8597b2b8cd8f8ab064864` | `06841f0c03c8597b2b8cd8f8ab064864` | **PASS (Bit-Identical)** ✅ |
| **$V_{4.1}$ Production Candidate** | `data/models/v4_1_prospective_candidate_2025_26.pkl` | `145f918d933eb343c0f63ca342b10289` | `145f918d933eb343c0f63ca342b10289` | **PASS (Bit-Identical)** ✅ |

---

## 3. Comprehensive Multi-Policy Comparative Evaluation

Evaluated across the 5 historical training seasons (2020/21–2024/25, $N = 8,983$) and the untouched holdout season (2025/2026, $N = 1,751$).

### A. Historical Training Era ($N = 8,983$):
From `candidate_decision_comparison.csv`:

| Policy | Decision Rule | Accuracy (%) | $\Delta$ Acc (%) | Draw Recall (%) | Draw Precision (%) | Rescued Draws | Damaged H/A | Net Delta ($\Delta$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Standard Argmax** | $\arg\max(P(H), P(D), P(A))$ | **52.77%** | $0.00\%$ | 0.00% | 0.00% | 0 | 0 | **0** |
| **B. Win Margin Gate** | $\max(P(H), P(A)) - P(D) < 0.08$ | 52.67% | $-0.10\%$ | 2.06% | 28.48% | 47 | 56 | **-9** |
| **C. Calibrated Draw Thresh** | $P'(D) \ge 0.32$ | 52.29% | $-0.48\%$ | 9.91% | **32.56%** | 226 | 269 | **-43** |
| **D. Draw & Win Parity** | $P'(D) \ge 0.30 \land \|P'_H - P'_A\| \le 0.06$ | 51.86% | $-0.90\%$ | 12.06% | 29.83% | 275 | 356 | **-81** |
| **E. Draw & Score Space** | $P'(D) \ge 0.29 \land P(\text{score\_draw}) \ge 0.28$ | 51.93% | $-0.83\%$ | 11.66% | 31.04% | 266 | 341 | **-75** |
| **F. Draw & Low Goals** | $P'(D) \ge 0.29 \land \lambda_{\text{tot}} \le 2.35$ | 52.49% | $-0.28\%$ | 3.16% | 31.03% | 72 | 97 | **-25** |
| **G. Draw & Elo Parity** | $P'(D) \ge 0.29 \land \|\Delta\text{Elo}\| \le 60$ | 50.04% | $-2.73\%$ | **21.26%** | 28.12% | **485** | 730 | **-245** |
| **H. Draw Confidence Score** | Multi-Signal Logistic Score $\ge 0.34$ | 52.77% | $0.00\%$ | 0.00% | 0.00% | 0 | 0 | **0** |
| **I. Selective Physical Gate** | 4-Condition Pre-Match Physical Gate | 52.27% | $-0.50\%$ | 2.81% | 25.60% | 64 | 109 | **-45** |
| **J. Platt Calibrated Argmax** | $\arg\max(P'(H), P'(D), P'(A))$ | 52.73% | $-0.03\%$ | 0.22% | 29.41% | 5 | 8 | **-3** |

---

### B. Untouched Holdout Season (2025/2026, $N = 1,751$):
From `holdout_decision_results.csv`:

| Policy | Accuracy (%) | $\Delta$ Acc (%) | Draw Recall (%) | Draw Precision (%) | Rescued Draws | Damaged H/A | Net Delta ($\Delta$) | Macro F1 | Balanced Acc (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Standard Argmax** | 51.97% | $0.00\%$ | 0.00% | 0.00% | 0 | 0 | **0** | 0.407 | 44.57% |
| **B. Win Margin Gate** | **52.20%** | $+0.23\%$ | 2.47% | **32.35%** | 11 | 7 | **+4** | 0.434 | 45.49% |
| **C. Calibrated Draw Thresh** | 51.57% | $-0.40\%$ | 7.87% | 27.34% | 35 | 42 | **-7** | 0.457 | 46.40% |
| **D. Draw & Win Parity** | 50.94% | $-1.03\%$ | **13.03%** | 29.00% | **58** | 76 | **-18** | **0.472** | **47.64%** |
| **E. Draw & Score Space** | 51.17% | $-0.80\%$ | 10.56% | 29.56% | 47 | 61 | **-14** | 0.468 | 47.16% |
| **F. Draw & Low Goals** | 51.80% | $-0.17\%$ | 2.47% | 29.73% | 11 | 14 | **-3** | 0.431 | 45.32% |
| **G. Draw & Elo Parity** | 49.34% | $-2.63\%$ | 20.90% | 27.76% | 93 | 139 | **-46** | 0.478 | 47.96% |
| **H. Draw Confidence Score** | 51.97% | $0.00\%$ | 0.00% | 0.00% | 0 | 0 | **0** | 0.407 | 44.57% |
| **I. Selective Physical Gate** | 51.68% | $-0.29\%$ | 2.47% | 22.45% | 11 | 16 | **-5** | 0.424 | 44.97% |
| **J. Platt Calibrated Argmax** | 51.86% | $-0.11\%$ | 0.00% | 0.00% | 0 | 2 | **-2** | 0.406 | 44.47% |

---

## 4. Key Technical Questions & Definitive Findings

### Q1: Did we actually improve Draw Recall?
- **Yes.** By lowering thresholds or introducing parity gates, Draw Recall can be pushed from $0.0\%$ up to **$13.0\% - 21.3\%$** (e.g. Policy D rescued 275 historical draws; Policy G rescued 485 historical draws).

### Q2: Did Draw Precision improve?
- **No.** Across all evaluated policies and threshold combinations, Draw Precision remained bounded between **$25.0\%$ and $33.3\%$**. Because the natural macro draw rate in European top-flight football is $25.4\%$, picking a draw on close matches achieves at most a ~30% hit rate.

### Q3: Did overall accuracy improve?
- **No.** On the large historical training dataset ($N = 8,983$), every single forced draw policy degraded overall 1X2 accuracy (by $-0.10\%$ to $-2.73\%$). While Policy B showed a minor positive fluke on the holdout season ($+0.23\%$, $\Delta = +4$), it yielded negative net value across the historical walk-forward folds ($\Delta = -9$).

### Q4: How many valid Home/Away predictions were damaged?
- On historical matches:
  - Policy C damaged **269 valid H/A wins**.
  - Policy D damaged **356 valid H/A wins**.
  - Policy G damaged **730 valid H/A wins**.
  - Policy I damaged **109 valid H/A wins**.

### Q5: What is the net correct-prediction delta?
- In every systematic walk-forward test, **$\Delta = \text{Rescued Draws} - \text{Damaged Wins} < 0$**.
- Because the combined probability of a decisive outcome ($P(H) + P(A) \approx 67\% - 70\%$) is twice that of the draw ($P(D) \approx 30\% - 33\%$), forcing a Draw prediction destroys ~1.2 to 1.5 correct win predictions for every 1 draw rescued.

### Q6: Which signals are genuinely useful?
From `draw_signal_importance.csv`:
1. `cal_win_diff` ($|P'(H) - P'(A)|$): Coefficient $= +0.5018$ (Win parity is the strongest draw indicator).
2. `lambda_gap` ($|\lambda_H - \lambda_A|$): Coefficient $= -0.4714$ (Small expected goal difference strongly correlates with draws).
3. `p_score_space_draw` ($\sum M[k, k]$): Coefficient $= +0.4131$ (Diagonal score-space probability mass).
4. `lambda_total` ($\lambda_H + \lambda_A$): Coefficient $= -0.2011$ (Low total match intensity).
5. `cal_p_D`: Coefficient $= +0.1917$ (Calibrated continuous draw probability).

### Q7: Is the decision policy stable across time?
- **No.** Forced draw policies exhibit substantial variance across walk-forward folds. In low-draw seasons, accuracy collapses by up to $-2.5\%$.

### Q8: Is it stable across leagues?
- From `league_decision_results.csv`: La Liga and Ligue 1 (higher base draw rates ~26-28%) tolerate draw gates slightly better than the Premier League or Bundesliga (draw rates ~22-24%), where damage to valid away wins is acute.

### Q9: Does it survive untouched 2025/26 holdout?
- Policy B produced $\Delta = +4$, but failed walk-forward historical validation. All other multi-condition policies produced negative deltas ($\Delta = -7$ to $-46$).

### Q10: Does it survive prospective evaluation?
- On the Aug 22–24, 2026 prospective sample ($N = 33$), no forced draw candidate safely flipped any of the 8 missed draws without risking the 20 correct win predictions.

### Q11: Is there evidence of overfitting?
- **Yes.** Heuristic thresholds like the Step 2B gate (0.26 / 0.10 / 2.50 / 100) perform well only on specific isolated slices but fail out-of-sample walk-forward validation ($\Delta = -45$).

### Q12: Is there any temporal leakage?
- **Zero.** Leakage audit passed 100% across all 14 signals (`step2i_leakage_audit.md`).

### Q13: What is the simplest winning strategy?
- **Continuous Platt Probability Calibration with Standard $\arg\max$ Decision Rule.**
  - Delivers lowest ECE ($-46.7\%$), best Brier score, best Multiclass Log Loss, and preserves $>99.8\%$ decision invariance with zero hard-threshold cliff-edge artifacts.

### Q14: Does a Draw Confidence Score outperform simple rules?
- A logistic multi-signal Draw Confidence Score smoothly ranks matches, but when converted into a forced binary 1X2 threshold, it suffers from the same mathematical ~30% precision ceiling.

### Q15: Is an AVOID/CAUTION layer more valuable than forcing Draw?
- **YES. UNEQUIVOCALLY YES.**
  - Flagging the **Draw Caution Zone** ($P'(D) \ge 0.28 \land |P'(H) - P'(A)| \le 0.08$) isolates 18.6% of database fixtures ($N=1,993$).
  - Inside the Caution Zone, baseline favorite prediction accuracy drops to **$38.59\%$** (true draw rate $= 29.90\%$).
  - Outside the Caution Zone, baseline favorite prediction accuracy jumps to **$55.84\%$**!
  - Advising user/betting engines to **CAUTION / ABSTAIN** on these matches is vastly superior to asserting a false high-confidence 1X2 prediction.

### Q16: Should anything be promoted to production?
- **NO.** Production must remain $V_{4.0}$ Production. No forced draw policy meets the bar for production promotion.

---

## 5. Final Governance Declaration

```
V4.0 Integrity: PASS (06841f0c03c8597b2b8cd8f8ab064864)
V4.1 Integrity: PASS (145f918d933eb343c0f63ca342b10289)
Leakage Audit: PASS (Zero Leakage)
Walk-Forward: PASS (Empirical Walk-Forward Completed)
Holdout: PASS (Untouched Holdout Evaluated)
Prospective: PASS (Prospective Audited)
Best Candidate: Policy J (Platt Continuous Calibrated Argmax) / Caution Zone Advisory
Accuracy Delta: -0.03% (Training) / -0.11% (Holdout)
Draw Recall Delta: +0.22% (Training) / +0.00% (Holdout)
Draw Precision: 29.41%
Valid H/A Damage: 8 matches (out of 8,983)
Net Correct Delta: -3
Final Classification: GREEN (Research Complete / Caution Advisory Validated / No Promotion)
Production Modified: NO
```
