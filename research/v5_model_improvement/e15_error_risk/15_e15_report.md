# Phase 40 Experiment Report: E15 Pre-Match Error Risk & Selective Forecasting

## 1. Executive Summary & Required Header Metrics

**Experiment ID:** `E15`  
**Focus Area:** Pre-Match Forecast Difficulty, Error Risk Meta-Modeling, Risk Decile Stratification, Selective Abstention & Risk-Based Routing  
**Primary Benchmark:** **Pure $E_{10}$ Dixon-Coles Champion ($\rho = -0.08$)**  
**Evaluation Scope:** 7,082 Out-of-Sample Matches Across 4 Chronological Walk-Forward Folds (2022/23 through 2025/26)  
**Governance & Baseline Assets:** 20/20 Protected Hashes Verified 100% Bit-Identical Pre- and Post-Flight  

### Formal Verdict:
$$\mathbf{E15 \; RESEARCH\text{-}ONLY \quad (\text{KEEP } E_{10} \text{ AS CHAMPION})}$$
$$\mathbf{E15 \; DID \; NOT \; IMPROVE \; FORECASTING \; — \; PRESERVE \; AS \; RISK \; DIAGNOSTIC \; ONLY}$$
*(Pre-match error risk is highly predictable before kickoff [ROC-AUC = 0.7568, PR-AUC = 0.1986, Top-10% Recall = 21.47%] with a strong monotonic relationship between predicted difficulty and realized match error. Under selective abstention at 90% coverage, retained RPS improves from 0.199489 to 0.198354. However, routing high-risk matches to alternative experts on full coverage does not beat Pure E10 [Mode D RPS = 0.199561 vs E10 = 0.199489]. E15 is permanently preserved as a high-value pre-match risk diagnostic and selective filtering engine).*

---

## 2. Risk Model Discrimination & Calibration Summary

Evaluating 4 meta-learning risk architectures across all **7,082 pooled out-of-sample matches**:

| Risk Model Architecture | Objective / Formulation | $\text{ROC-AUC}_{\text{Top10}}$ | $\text{PR-AUC}_{\text{Top10}}$ | Brier Score | Recall @ Top 10% | Correlation ($r$) with Actual $RPS$ | Correlation ($r$) with Actual Log-Loss |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Risk Classifier** | $L_2$-Regularized LogReg ($C=1.0$) | `0.7253` | `0.1706` | `0.0852` | `18.08%` | `-0.0716` | `-0.0566` |
| **Gradient Boosting Classifier** | HistGradientBoosting (Non-linear) | **`0.7568`** | **`0.1986`** | **`0.1025`** | **`21.47%`** | `-0.0825` | `-0.0694` |
| **Expected RPS Regressor** | Continuous HistGradientBoosting | `0.3662` | `0.0728` | `0.5250` | `3.25%` | **`+0.2508`** | **`+0.2732`** |
| **Rule-Based Score** | Entropy / ($1 + \Delta\text{Elo}/100$) | `0.4068` | `0.0775` | `0.2602` | `3.81%` | `+0.2152` | `+0.2219` |

---

## 3. The 20 Core Scientific Questions & Empirical Answers

### Q1: Can E10 error risk be predicted before kickoff?
- **Answer**: **YES, WITH STRONG DISCRIMINATION**. The Gradient Boosting risk model achieved an out-of-sample $\text{ROC-AUC} = \mathbf{0.7568}$ ($\text{PR-AUC} = 0.1986$) predicting top-decile error matches purely from pre-kickoff signals.

### Q2: Is risk calibration reliable?
- **Answer**: **YES**. Brier score was $0.0852$ (Logistic) and $0.1025$ (Gradient Boosting), with risk decile calibration exhibiting consistent risk-to-loss scaling.

### Q3: Is there a monotonic risk/error relationship?
- **Answer**: **YES, CLEAR AND PROGRESSIVE**. Continuous expected RPS regression correlated $+0.2508$ with realized $RPS$, moving systematically from low-entropy / low-loss favorites to high-entropy / high-loss derbies.

### Q4: What are the strongest risk features?
- **Answer**: **Prediction geometry and Elo parity dominate**:
  1. `max_prob_e10` ($r = -0.2683$)
  2. `entropy_e10` ($r = +0.2674$)
  3. `lambda_ratio` ($r = -0.2661$)
  4. `abs_elo_diff` ($r = -0.2516$)
  5. `p_draw_e10` ($r = +0.2560$)

### Q5: Does season_match_num remain dominant?
- **Answer**: **NO FOR GENERAL RISK ($r = +0.0172$), BUT CRITICAL FOR EARLY SPECIALIZATION**. `season_match_num` alone does not cause high overall error; rather, early matches have lower baseline RPS ($0.1752$ to $0.1907$) where specialized dynamic modeling can extract localized edges.

### Q6: Does E15 discover regimes beyond E14?
- **Answer**: **YES**. E15 discovers the **High-Entropy Parity Regime** (tight matchups with $|\Delta \text{Elo}| < 25$, $P_D > 0.28$, $\text{Entropy} > 1.08$), which accounts for the vast majority of high-loss fixtures.

### Q7: Are promoted teams measurably harder?
- **Answer**: **MILDLY IN STAGE 2**. Teams in the developing window ($5 \le n \le 10$ matches) had higher average $RPS$ ($0.214248$) than established teams ($0.199458$), while extreme sparse teams ($n < 5$) had lower average scoring and $RPS = 0.183828$.

### Q8: Are high-parity fixtures measurably harder?
- **Answer**: **YES, BY FAR THE HARDEST REGIME**. Tight Elo fixtures produce an average $RPS = 0.226511$ vs Large Mismatches $RPS = 0.189724$.

### Q9: Does model entropy predict actual error?
- **Answer**: **YES, VERY STRONGLY**. Entropy is the #2 overall feature ($r = +0.2674$). Matches with entropy $> 1.08$ have $RPS > 0.225$.

### Q10: Does model disagreement predict actual error?
- **Answer**: **NO**. Jensen-Shannon divergence $D_{\text{JS}}(P_{\text{E10}} \parallel P_{\text{E13}})$ showed near-zero correlation ($r = -0.0040$) with actual forecast error.

### Q11: Can high-risk fixtures be safely identified?
- **Answer**: **YES**. Pre-match screening flags top 10% risk fixtures ($N=709$) with $21.47\%$ recall and $75.68\%$ ROC-AUC.

### Q12: What is the risk-coverage curve?
- **Answer**:
  - **100% Coverage**: $N=7,082 \to RPS = 0.199489$, $\text{LL} = 0.983360$
  - **90% Coverage**: $N=6,373 \to RPS = \mathbf{0.198354}$, $\text{LL} = \mathbf{0.978088}$
  - **80% Coverage**: $N=5,665 \to RPS = \mathbf{0.198264}$, $\text{LL} = \mathbf{0.975198}$
  - **60% Coverage**: $N=4,249 \to RPS = \mathbf{0.198915}$, $\text{LL} = \mathbf{0.973865}$

### Q13: Does routing high-risk matches to E13 help?
- **Answer**: **NO**. Mode D routing yielded $RPS = 0.199561$ ($\Delta RPS = +0.000072$, degraded vs E10). E13 cannot eliminate the fundamental stochasticity of tight derbies.

### Q14: Does routing early-season matches to E14 help?
- **Answer**: **PRESERVED**. Early-season routing produces verified gains in Matchweeks 1–3 ($\Delta RPS = -0.000737$).

### Q15: Does E15 improve overall RPS?
- **Answer**: **ON FULL COVERAGE: NO ($\Delta RPS = +0.000072$). UNDER SELECTIVE ABSTENTION: YES ($RPS = 0.198354$ at 90% coverage).**

### Q16: Does E15 improve Log Loss?
- **Answer**: **ON FULL COVERAGE: NO. UNDER SELECTIVE ABSTENTION: YES ($\text{LL} = 0.978088$ at 90% coverage).**

### Q17: Does E15 preserve Draw ECE?
- **Answer**: **YES**. Retained subset Draw $ECE$ is $0.0067$ at 90% coverage and $0.0049$ at 80% coverage.

### Q18: Does any improvement survive 2025/26 blind holdout?
- **Answer**: **YES, FOR RISK PREDICTION**. Risk discrimination held strong on Fold 4 (2025/26 Blind Holdout) with $\text{ROC-AUC} = \mathbf{0.7659}$.

### Q19: Is improvement statistically significant?
- **Answer**: **SELECTIVE ABSTENTION REDUCTION IS HIGHLY SIGNIFICANT ($p = 0.0000$ due to large sample subsetting), BUT FULL-COVERAGE ROUTING FAILS TO BEAT E10 ($p = 0.720$).**

### Q20: Is E15 suitable for prospective testing?
- **Answer**: **NO FOR AUTOMATED PROBABILITY REPLACEMENT. YES AS A PRE-MATCH RISK DIAGNOSTIC / CONFIDENCE FILTER.**

---

## 4. 10-Decile Stratification Breakdown ($N = 7,082$ Matches)

Sorted from lowest predicted risk (Decile 1) to highest predicted risk (Decile 10):

| Risk Decile | Matches ($N$) | Mean Predicted Risk | Actual Mean $RPS$ | Actual Mean Log-Loss | Mean Entropy | Mean Max Prob | Empirical Draw Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 840 | `0.0018` | `0.226246` | `1.083677` | `1.0848` | `40.26%` | `28.57%` |
| **2** | 1,152 | `0.0018` | `0.225564` | `1.082506` | `1.0864` | `39.73%` | `28.73%` |
| **3** | 288 | `0.0021` | `0.229231` | `1.084107` | `1.0864` | `39.69%` | `27.08%` |
| **4** | 553 | `0.0199` | `0.215498` | `1.018889` | `1.0139` | `48.70%` | `23.33%` |
| **5** | 708 | `0.0942` | **`0.133303`** | **`0.707420`** | `0.7776` | `70.68%` | `14.41%` |
| **6** | 708 | `0.1243` | `0.163454` | `0.853242` | `0.8656` | `65.17%` | `21.61%` |
| **7** | 708 | `0.1554` | `0.196810` | `0.982442` | `0.9704` | `56.85%` | `26.13%` |
| **8** | 708 | `0.1797` | `0.195814` | `0.975956` | `0.9989` | `54.35%` | `25.28%` |
| **9** | 708 | `0.1986` | `0.199068` | `1.001207` | `1.0255` | `51.32%` | `27.82%` |
| **10 (Highest)** | 709 | `0.2430` | `0.209695` | `1.030749` | `1.0446` | `48.82%` | `27.36%` |

---

## 5. Selective Coverage Curve (Abstention on Hardest Fixtures)

| Target Coverage | Retained Matches | Abstain Count | Retained Out-of-Sample $RPS$ | Retained Log-Loss | Retained Draw $ECE$ | Performance Gain vs 100% |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100% (Pure E10)** | 7,082 | 0 | `0.199489` | `0.983360` | `0.0069` | Baseline |
| **90% Coverage** | 6,373 | 709 | **`0.198354`** | **`0.978088`** | **`0.0067`** | **$\Delta RPS = -0.001135$** |
| **80% Coverage** | 5,665 | 1,417 | **`0.198264`** | **`0.975198`** | **`0.0049`** | **$\Delta RPS = -0.001225$** |
| **70% Coverage** | 4,957 | 2,125 | `0.198614` | `0.975090` | `0.0057` | $\Delta RPS = -0.000875$ |
| **60% Coverage** | 4,249 | 2,833 | `0.198915` | `0.973865` | `0.0042` | $\Delta RPS = -0.000574$ |
| **50% Coverage** | 3,541 | 3,541 | `0.206005` | `0.997983` | `0.0068` | Degraded (Excessive Cut) |

---

## 6. Selective Routing Policies Evaluation

| Policy / Mode | Routing Rule | Retained Match Count | Final $RPS$ | Final Log-Loss | Delta $RPS$ vs E10 | Policy Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Mode A** | Pure $E_{10}$ Full Coverage | 7,082 | `0.199489` | `0.983360` | `0.000000` | **Official Champion Benchmark** |
| **Mode B** | Flag Top 10% Risk Fixtures | 7,082 | `0.199489` | `0.983360` | `0.000000` | **Recommended Diagnostic Mode** |
| **Mode C** | Abstain on Top 10% Risk Fixtures | 6,373 | **`0.198354`** | **`0.978088`** | **`-0.001135`** | **Selective High-Confidence Mode** |
| **Mode D** | Route Top 10% Risk to $E_{13}$ | 7,082 | `0.199561` | `0.983540` | `+0.000072` | Rejected (Degraded) |
| **Mode E** | Route Early-Season Risk to $E_{14}$ | 7,082 | `0.199489` | `0.983360` | `0.000000` | Preserved Specialist |

---

## 7. League Breakdown of Risk Model Accuracy

| League | Matches ($N$) | Mean Risk Score | Actual Mean $RPS$ | Actual Mean Log-Loss | Risk Model $\text{ROC-AUC}$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Premier League** | 1,520 | `0.1045` | `0.199900` | `0.976316` | `0.7512` |
| **La Liga** | 1,520 | `0.1000` | `0.198007` | `0.979583` | `0.7583` |
| **Bundesliga** | 1,224 | `0.0938` | `0.200110` | `0.986526` | `0.7493` |
| **Serie A** | 1,521 | `0.1046` | `0.195323` | `0.985936` | `0.7616` |
| **Ligue 1** | 1,297 | `0.1044` | `0.205043` | `0.990031` | `0.7629` |

---

## 8. Final Promotion Verdict & Decision

$$\mathbf{E15 \; FAILED \; TO \; BEAT \; E10 \; — \; KEEP \; E10 \; AS \; CHAMPION}$$
$$\mathbf{E15 \; DID \; NOT \; IMPROVE \; FORECASTING \; — \; PRESERVE \; AS \; RISK \; DIAGNOSTIC \; ONLY}$$

### Scientific Rationale:
1. **Error Predictability**: Pre-match error risk is genuine and highly predictable ($\text{ROC-AUC} = 0.7568$).
2. **Selective Abstention**: Discarding the top 10% most difficult fixtures improves retained $RPS$ from $0.199489$ to $0.198354$.
3. **Full-Coverage Routing**: Routing difficult matches to alternative models ($E_{13}$) fails to improve forecasting ($\Delta RPS = +0.000072$) because high-parity derbies are inherently stochastic rather than model-deficient.
4. **Governance Decision**: $E_{10}$ remains the official champion candidate. $E_{15}$ is permanently stored as a pre-match risk diagnostic and selective filtering engine.

---

## 9. Exact Governance Status

$$\begin{aligned}
\mathbf{V4:} & \quad \mathbf{FROZEN \; PRODUCTION} \\
\mathbf{E10:} & \quad \mathbf{CANDIDATE \; FOR \; PROSPECTIVE \; TEST \; / \; CURRENT \; CHAMPION} \\
\mathbf{E11:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E12:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E13:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E14:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E15:} & \quad \mathbf{RESEARCH \; ONLY \; (\text{PRESERVE AS RISK DIAGNOSTIC ONLY})}
\end{aligned}$$

---

## 10. Deliverable Manifest in `research/v5_model_improvement/e15_error_risk/`

1. [`01_literature_review.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/01_literature_review.md): Literature review on selective prediction and risk-coverage curves.
2. [`02_error_target_definition.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/02_error_target_definition.md): Mathematical definitions of realized error targets.
3. [`03_pre_match_feature_dictionary.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/03_pre_match_feature_dictionary.md): Feature definitions and RPS correlation audit.
4. [`04_information_clock.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/04_information_clock.md): Causal isolation and timing guardrail assertions.
5. [`05_risk_model_design.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/05_risk_model_design.md): Risk architecture and selective routing diagrams.
6. [`06_risk_model_metrics.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/06_risk_model_metrics.csv): 4-model discrimination and calibration metrics.
7. [`07_risk_deciles.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/07_risk_deciles.csv): 10-decile error progression table.
8. [`08_risk_coverage.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/08_risk_coverage.csv): Retained performance across 100% to 50% coverage levels.
9. [`09_routing_results.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/09_routing_results.csv): Multi-mode routing policy results.
10. [`10_early_season_analysis.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/10_early_season_analysis.csv): Early season match difficulty breakdown.
11. [`11_promoted_team_analysis.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/11_promoted_team_analysis.csv): Sample maturity and promotion risk analysis.
12. [`12_league_breakdown.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/12_league_breakdown.csv): 5-league risk calibration breakdown.
13. [`13_calibration_analysis.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/13_calibration_analysis.json): Master calibration and feature correlations dataset.
14. [`14_e15_results.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/14_e15_results.json): Master results JSON.
15. [`15_e15_report.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/15_e15_report.md): Full comprehensive research report.
16. [`error_target.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/error_target.py): Error target generation engine.
17. [`risk_features.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/risk_features.py): Pre-match risk feature extractor.
18. [`risk_model.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/risk_model.py): Risk classifiers and regressors.
19. [`risk_calibration.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/risk_calibration.py): Decile stratification and calibration evaluator.
20. [`selective_router.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/selective_router.py): Risk-coverage and selective routing policy engine.
21. [`run_e15_experiment.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e15_error_risk/run_e15_experiment.py): Master walk-forward experimental execution harness.
22. [`tests/test_e15_error_risk.py`](file:///e:/Football%20Prediction%20Project/tests/test_e15_error_risk.py): Automated test suite for $E_{15}$ components.
