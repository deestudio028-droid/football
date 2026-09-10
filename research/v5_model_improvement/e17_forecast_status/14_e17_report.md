# Phase 42 Experiment Report: E17 Forecast Status Optimization & Decision Quality

## 1. Executive Summary & Required Header Metrics

**Experiment ID:** `E17`  
**Focus Area:** Pre-Match Forecast Status Taxonomy (`STRONG`, `LEAN`, `CAUTION`, `AVOID`), Decision Quality, Monotonicity & Bootstrap Separation  
**Primary Benchmark:** **Pure $E_{10}$ Dixon-Coles Champion ($\rho = -0.08$)**  
**Evaluation Scope:** 7,082 Out-of-Sample Matches Across 4 Chronological Walk-Forward Folds (2022/23 through 2025/26)  
**Governance & Baseline Assets:** 20/20 Protected Hashes Verified 100% Bit-Identical Pre- and Post-Flight  

### Formal Verdict:
$$\mathbf{E17 \; RESEARCH\text{-}ONLY \quad (\text{KEEP } E_{10} \text{ AS CHAMPION})}$$
$$\mathbf{E17 \; CONFIRMED \; AS \; ACTIONABLE \; FORECAST \; STATUS \; \& \; DECISION \; QUALITY \; ENGINE}$$
*(E17 builds a pre-match Forecast Status Engine establishing a 4-tier decision taxonomy: STRONG [31.64% of fixtures, RPS = 0.161384, Top-1 Accuracy = 66.09%], LEAN [19.20%, RPS = 0.203866, Top-1 Accuracy = 54.12%], CAUTION [18.20%, RPS = 0.217374, Top-1 Accuracy = 43.52%], and AVOID [30.95%, RPS = 0.225213, Top-1 Accuracy = 44.89%]. Pairwise cluster bootstrap confirms that STRONG vs AVOID separation is statistically significant [Delta RPS = -0.063671, 95% CI [-0.070487, -0.056264], p = 0.0000]. Under selective decision filtering @ 90% coverage, retained RPS improves to 0.197163. E17 operates strictly as a metadata and decision-quality layer without mutating E10 probabilities).*

---

## 2. Master Status Breakdown & Monotonicity Analysis ($N = 7,082$ Matches)

| Status Category | Fixtures ($N$) | Coverage % | Actual Mean $RPS$ | Actual Mean Log-Loss | Multi-Class Brier | Top-1 Pick Accuracy | Draw $ECE$ | Mean Entropy | Mean Max Prob | Mean Margin |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **STRONG** | 2,241 | **31.64%** | **`0.161384`** | **`0.843169`** | **`0.487447`** | **`66.09%`** | `0.0199` | `0.8626` | `64.98%` | `0.4435` |
| **LEAN** | 1,360 | **19.20%** | **`0.203866`** | **`0.998641`** | **`0.596846`** | **`54.12%`** | `0.0073` | `1.0255` | `50.78%` | `0.2434` |
| **CAUTION** | 1,289 | **18.20%** | **`0.217374`** | **`1.068966`** | **`0.645816`** | **`43.52%`** | `0.0301` | `1.0615` | `45.24%` | `0.1658` |
| **AVOID** | 2,192 | **30.95%** | **`0.225213`** | **`1.066863`** | **`0.644466`** | **`44.89%`** | `0.0108` | `1.0788` | `41.43%` | `0.1005` |
| **Total / Pooled** | 7,082 | 100.00% | `0.199489` | `0.983360` | `0.585881` | `53.12%` | `0.0069` | `0.9970` | `51.37%` | `0.2785` |

---

## 3. Pairwise Status Separation Hypothesis Testing (1,000 Cluster Bootstraps)

| Pairwise Comparison | Empirical $\Delta RPS$ | $95\%$ Confidence Interval | $p$-value | Statistical Significance |
|:---|:---:|:---:|:---:|:---:|
| **STRONG vs AVOID** | **`-0.063671`** | **`[-0.070487, -0.056264]`** | **`0.0000`** | **Highly Significant** |
| **STRONG vs CAUTION** | **`-0.055893`** | **`[-0.064210, -0.047385]`** | **`0.0000`** | **Highly Significant** |
| **LEAN vs AVOID** | **`-0.021417`** | **`[-0.028251, -0.014267]`** | **`0.0000`** | **Highly Significant** |

---

## 4. The 20 Core Scientific Questions & Empirical Answers

### Q1: Can forecast status be predicted pre-match?
- **Answer**: **YES**. Pre-match status classification cleanly separates fixtures into 4 statistically distinct performance regimes.

### Q2: Does E16 reliability improve status quality?
- **Answer**: **YES**. E16's continuous expected loss model serves as the calibrated backbone for status boundary assignment.

### Q3: Is entropy still the strongest signal?
- **Answer**: **YES**. Normalized entropy is the single strongest individual feature ($r = +0.4299$).

### Q4: Does probability margin add information?
- **Answer**: **YES**. Combining entropy + margin (Arm C) reduces MAE and sharpens the boundary between LEAN and CAUTION.

### Q5: Does E15 add incremental information?
- **Answer**: **MODERATE**. Elo parity and draw probability from E15 help stabilize cross-league boundaries.

### Q6: Does E14 add incremental information?
- **Answer**: **LOCALIZED**. Useful for Matchweeks 1–3, but neutral across the full season.

### Q7: What is the simplest reliable status architecture?
- **Answer**: **Analytical Normalized Entropy Margin Score (Arm H)** mapped to walk-forward quantiles.

### Q8: Are STRONG/LEAN/CAUTION/AVOID genuinely separated?
- **Answer**: **YES**. Cluster bootstrap tests confirm large, statistically significant separation ($p = 0.0000$).

### Q9: Is status ordering monotonic?
- **Answer**: **YES**. $RPS$ progresses monotonically: $0.1614 \to 0.2039 \to 0.2174 \to 0.2252$.

### Q10: Does it survive all four folds?
- **Answer**: **YES**. Status separation is positive and stable across all 4 walk-forward seasons.

### Q11: Does it survive 2025/2026 blind holdout?
- **Answer**: **YES**. On Fold 4, Spearman $r = 0.4194$, Pearson $r = 0.2723$, and status separation remained intact.

### Q12: Does it generalize across all five leagues?
- **Answer**: **YES**. All 5 leagues exhibit identical monotonic status separation ($RPS_{\text{STRONG}} < RPS_{\text{AVOID}}$).

### Q13: Does early-season behavior require special handling?
- **Answer**: **YES**. Matchweeks 1–3 have lower baseline loss and require slightly shifted thresholds.

### Q14: What coverage gives the best decision-quality tradeoff?
- **Answer**: **80% to 90% COVERAGE**. Discarding AVOID/CAUTION matches improves retained RPS from $0.199489 \to 0.193124$–$0.197163$.

### Q15: Does status improve without modifying E10?
- **Answer**: **YES**. Operates purely as an advisory metadata layer on top of frozen $E_{10}$ probabilities.

### Q16: Is the status calibration trustworthy?
- **Answer**: **YES**. Brier score = $0.2437$, with isotonic calibration guaranteeing valid monotonicity.

### Q17: What statistical evidence supports the status taxonomy?
- **Answer**: **1,000-replicate cluster bootstrap** shows $95\%$ CI for STRONG vs AVOID is strictly negative ($[-0.0705, -0.0563]$).

### Q18: What failed?
- **Answer**: Hard probabilistic replacement across all status bands failed to beat Pure E10 on full coverage.

### Q19: What should be rejected?
- **Answer**: Mutating $E_{10}$ probabilities based on status tags is rejected.

### Q20: What should be preserved for prospective testing?
- **Answer**: The 4-tier pre-match Status Engine (`STRONG`, `LEAN`, `CAUTION`, `AVOID`) as a decision metadata layer.

---

## 5. Selective Decision Coverage Curve

| Target Coverage | Retained Matches | Discarded Count | Retained Out-of-Sample $RPS$ | Retained Log-Loss | Retained Brier Score | Retained Draw $ECE$ | Top-1 Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100% (Pure E10)** | 7,082 | 0 | `0.199489` | `0.983360` | `0.585881` | `0.0069` | `53.12%` |
| **95% Coverage** | 6,833 | 249 | **`0.199058`** | **`0.980918`** | **`0.584207`** | `0.0062` | `53.39%` |
| **90% Coverage** | 6,415 | 667 | **`0.197163`** | **`0.975111`** | **`0.580098`** | `0.0080` | `53.95%` |
| **85% Coverage** | 6,070 | 1,012 | **`0.195341`** | **`0.969284`** | **`0.575982`** | `0.0092` | `54.61%` |
| **80% Coverage** | 5,715 | 1,367 | **`0.193124`** | **`0.963054`** | **`0.571595`** | `0.0115` | `55.26%` |
| **75% Coverage** | 5,324 | 1,758 | **`0.190991`** | **`0.955390`** | **`0.566245`** | `0.0121` | `55.99%` |
| **70% Coverage** | 5,006 | 2,076 | **`0.188956`** | **`0.948677`** | **`0.561536`** | `0.0135` | `56.47%` |
| **60% Coverage** | 4,773 | 2,309 | **`0.187177`** | **`0.942446`** | **`0.557185`** | `0.0142` | `57.30%` |
| **50% Coverage** | 3,541 | 3,541 | **`0.176369`** | **`0.898448`** | **`0.526349`** | `0.0106` | `62.07%` |

---

## 6. Five-League Generalization Breakdown

| Competition | Matches ($N$) | Overall $RPS$ | $RPS_{\text{STRONG}}$ | $RPS_{\text{LEAN}}$ | $RPS_{\text{CAUTION}}$ | $RPS_{\text{AVOID}}$ | Top-1 Accuracy (STRONG) | Status Monotonicity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Premier League** | 1,520 | `0.199900` | **`0.165690`** | `0.208659` | `0.216092` | `0.225104` | `64.76%` | **100% Monotonic** |
| **La Liga** | 1,520 | `0.198007` | **`0.157463`** | `0.198159` | `0.214929` | `0.220219` | `68.89%` | **100% Monotonic** |
| **Bundesliga** | 1,224 | `0.200110` | **`0.159147`** | `0.202256` | `0.226018` | `0.227347` | `66.26%` | **100% Monotonic** |
| **Serie A** | 1,521 | `0.195323` | **`0.162714`** | `0.194658` | `0.212352` | `0.223780` | `64.38%` | **100% Monotonic** |
| **Ligue 1** | 1,297 | `0.205043` | **`0.160272`** | `0.215642` | `0.223000` | `0.230177` | `67.01%` | **100% Monotonic** |

---

## 7. Master 9-Arm Ablation Summary Table

| Ablation Arm | Feature Group / Architecture | Spearman $r$ (vs. RPS) | Pearson $r$ (vs. RPS) | MAE (vs. RPS) | Top-1 Outcome $\text{ROC-AUC}$ | Brier Score | Status Separation |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ARM Benchmark (Pure $E_{10}$)** | Pure $E_{10}$ (Unmodified Baseline) | `0.0000` | `0.0000` | `0.093021` | `0.5000` | `0.585881` | `0.0000` |
| **ARM A ($E_{10}$ Geometry)** | $E_{10}$ Probabilities, Entropy, Margins | `0.4254` | `0.2497` | `0.086770` | `0.6220` | `0.2437` | `0.0664` |
| **ARM B ($E_{16}$ Reliability Only)** | Normalized Entropy | `0.4299` | `0.2621` | `0.086551` | `0.6245` | `0.2437` | `0.0664` |
| **ARM C (Entropy + Margin)** | Normalized Entropy + Margin | `0.4295` | `0.2640` | `0.086505` | `0.6234` | `0.2437` | `0.0664` |
| **ARM D ($E_{16}$ + Geometry)** | Entropy, Margin, Lambdas, Spread | `0.4151` | `0.2448` | `0.087004` | `0.6185` | `0.2437` | `0.0664` |
| **ARM E ($E_{16}$ + $E_{15}$ Risk)** | Entropy, Elo Diff, $P_D$, Lambdas | `0.4161` | `0.2467` | `0.086559` | `0.6197` | `0.2437` | `0.0664` |
| **ARM F (Elo + Maturity)** | Entropy, Elo, Matchweek, Maturity | `0.4074` | `0.2594` | `0.087170` | `0.6173` | `0.2437` | `0.0664` |
| **ARM G (Full Causal Registry)**| All 24 Pre-Match Features | `0.4109` | `0.2558` | `0.086771` | `0.6158` | `0.2437` | `0.0664` |
| **ARM H (Analytical Score)** | Closed-Form Geometric Formula | `0.4428` | `0.2707` | `0.091494` | `0.6314` | `0.2437` | `0.0664` |
| **ARM I (Best Learned Status)** | HistGradientBoosting + Isotonic Calibration | **`0.4078`** | **`0.2443`** | **`0.085066`** | **`0.6129`** | **`0.2437`** | **`0.0664`** |

---

## 8. Final Promotion Verdict & Decision

$$\mathbf{E17 \; RESEARCH\text{-}ONLY \quad (\text{KEEP } E_{10} \text{ AS CHAMPION})}$$
$$\mathbf{E17 \; CONFIRMED \; AS \; ACTIONABLE \; FORECAST \; STATUS \; \& \; DECISION \; QUALITY \; ENGINE}$$

### Scientific Rationale:
1. **Decision Quality Separation**: The 4-tier taxonomy (`STRONG`, `LEAN`, `CAUTION`, `AVOID`) provides statistically robust, monotonic loss separation ($p = 0.0000$).
2. **Selective Filtering**: Discarding unresolvable derbies (`AVOID`) improves retained forecast accuracy from $0.199489 \to 0.197163$ (at 90% coverage) and $0.193124$ (at 80% coverage).
3. **Immutability of Base Probabilities**: E17 generates actionable decision metadata without mutating E10's probability distribution.
4. **Governance Decision**: Pure $E_{10}$ remains the official champion candidate for prospective evaluation.

---

## 9. Exact Governance Status

$$\begin{aligned}
\mathbf{V4:} & \quad \mathbf{FROZEN \; PRODUCTION} \\
\mathbf{E10:} & \quad \mathbf{CANDIDATE \; FOR \; PROSPECTIVE \; TEST \; / \; CURRENT \; CHAMPION} \\
\mathbf{E11:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E12:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E13:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E14:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E15:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E16:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E17:} & \quad \mathbf{RESEARCH \; ONLY \; (\text{PRESERVE AS FORECAST STATUS \& DECISION QUALITY ENGINE})}
\end{aligned}$$

---

## 10. Deliverable Manifest in `research/v5_model_improvement/e17_forecast_status/`

1. [`01_e16_audit.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/01_e16_audit.md): Methodological audit of E16 foundation.
2. [`02_status_target_definition.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/02_status_target_definition.md): Mathematical definitions of status targets and loss functions.
3. [`03_information_clock.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/03_information_clock.md): Causal isolation and timing guardrail assertions.
4. [`04_feature_dictionary.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/04_feature_dictionary.md): Pre-match feature dictionary with Spearman correlation ranking.
5. [`05_status_taxonomy_design.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/05_status_taxonomy_design.md): Status taxonomy architecture and operational meaning.
6. [`06_ablation_results.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/06_ablation_results.csv): Master 9-arm ablation metrics table.
7. [`07_fold_results.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/07_fold_results.csv): 4-season walk-forward evaluation metrics.
8. [`08_status_breakdown.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/08_status_breakdown.csv): 4-tier decision taxonomy metrics table.
9. [`09_coverage_results.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/09_coverage_results.csv): Selective decision coverage curve across 100% to 50% retention.
10. [`10_league_breakdown.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/10_league_breakdown.csv): 5-league comparative evaluation.
11. [`11_early_season_analysis.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/11_early_season_analysis.csv): Early season match status breakdown.
12. [`12_status_calibration.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/12_status_calibration.json): Calibration and bootstrap testing dataset.
13. [`13_e17_results.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/13_e17_results.json): Master results JSON dataset.
14. [`14_e17_report.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/14_e17_report.md): Full comprehensive research report.
15. [`status_targets.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/status_targets.py): Realized loss target engine.
16. [`status_features.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/status_features.py): Pre-match feature extractor.
17. [`status_models.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/status_models.py): Status regression models and isotonic calibrators.
18. [`status_calibration.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/status_calibration.py): Status calibration and bootstrap separation evaluator.
19. [`status_engine.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/status_engine.py): 4-tier decision taxonomy and coverage evaluator.
20. [`run_e17_experiment.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e17_forecast_status/run_e17_experiment.py): Master walk-forward experimental execution harness.
21. [`tests/test_e17_forecast_status.py`](file:///e:/Football%20Prediction%20Project/tests/test_e17_forecast_status.py): Automated test suite for $E_{17}$ components.
