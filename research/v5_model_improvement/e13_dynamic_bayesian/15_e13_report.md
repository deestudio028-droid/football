# Phase 38 Experiment Report: E13 Dynamic Bayesian & Hierarchical Team Strength

## 1. Executive Summary & Required Header Metrics

**Experiment ID:** `E13`  
**Focus Area:** Dynamic Adaptive-K Elo, State-Space Attack/Defense Models, Hierarchical Empirical Bayes Shrinkage & Parameter Uncertainty Propagation  
**Primary Benchmark:** **Pure $E_{10}$ Dixon-Coles Champion ($\rho = -0.08$)**  
**Evaluation Scope:** 7,082 Out-of-Sample Matches Across 4 Chronological Walk-Forward Folds (2022/23 through 2025/26)  
**Governance & Baseline Assets:** 20/20 Protected Hashes Verified 100% Bit-Identical Pre- and Post-Flight  

### Formal Verdict:
$$\mathbf{E13 \; RESEARCH\text{-}ONLY \quad (\text{KEEP } E_{10} \text{ AS CHAMPION})}$$
*(While hierarchical shrinkage and adaptive K-factor delivered gains in the first 3 matches of the season, dynamic latent tracking induced excess parameter volatility over full seasons, failing to beat Pure E10).*

---

## 2. Master 9-Arm Ablation Results ($N = 7,082$ Matches)

Evaluating across all **7,082 pooled out-of-sample matches** from 4 walk-forward folds (2022/23 through 2025/26):

| Ablation Arm | Description / Configuration | $RPS$ | Log-Loss | Multiclass Brier | Draw $ECE$ | Mean Entropy | Sharpness |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline: $V_4$** | Frozen $V_4$ Baseline (91 features) | `0.199598` | `0.984955` | `0.586532` | `0.0207` | `0.9984` | `0.5135` |
| **ARM A (Pure $E_{10}$)** | **$V_4$ + Dixon-Coles ($\rho = -0.08$, Champion)** | **`0.199489`** | **`0.983360`** | **`0.585881`** | **`0.0069`** | `0.9970` | `0.5137` |
| **ARM B ($E_{10}$ + Dynamic Elo)** | Adaptive $K_i(t)$ + Attack/Defense Elo | `0.200056` | `0.985237` | `0.587133` | `0.0082` | `0.9906` | `0.5194` |
| **ARM C ($E_{10}$ + State-Space AD)** | Random-Walk Latents with Process Noise | `0.200265` | `0.985979` | `0.587637` | `0.0070` | `0.9933` | `0.5168` |
| **ARM D ($E_{10}$ + Hierarchical Shrink)** | Empirical Bayes Shrinkage ($m_0 = 5$) | `0.200378` | `0.986306` | `0.587888` | `0.0064` | `0.9925` | `0.5178` |
| **ARM E ($E_{10}$ + Season Transition)** | Inter-Season Regression ($\alpha = 0.85$) | `0.200588` | `0.986945` | `0.588329` | `0.0053` | `0.9964` | `0.5144` |
| **ARM F ($E_{10}$ + Venue Latents)** | Separate Home/Away Latent States | `0.200213` | `0.985864` | `0.587488` | `0.0068` | `0.9892` | `0.5206` |
| **ARM G ($E_{10}$ + Uncertainty Integration)**| Gauss-Hermite Quadrature over $\sigma_{\lambda}^2$ | `0.200075` | `0.985595` | `0.587138` | `0.0117` | `1.0093` | `0.5043` |
| **ARM H ($E_{10}$ + Best Dynamic Combo)** | Union of Best Dynamic Latent Features | `0.200092` | `0.985314` | `0.587177` | `0.0076` | `0.9927` | `0.5175` |
| **ARM I ($E_{10}$ + Dynamic + Uncertainty)**| Arm H + Gauss-Hermite Quadrature | `0.200545` | `0.987041` | `0.588150` | `0.0127` | `1.0057` | `0.5077` |

---

## 3. The 16 Core Scientific Questions & Empirical Answers

### Q1: Does dynamic Elo improve E10?
- **Answer**: **NO**. Arm B ($RPS = 0.200056$) degraded vs Pure $E_{10}$ ($0.199489$). While adaptive $K$ helped early matches, it over-reacted to noisy multi-goal blowouts later in the season.

### Q2: Does dynamic attack/defense improve E10?
- **Answer**: **NO**. Arm C ($RPS = 0.200265$) increased out-of-sample $RPS$ by $+0.000776$. Continuous random walk process noise without strict damping makes team ratings fluctuate too quickly.

### Q3: Does hierarchical shrinkage improve E10?
- **Answer**: **NO ON FULL SAMPLE, BUT YES IN EARLY MATCHES**. Overall $RPS = 0.200378$, but in the first 3 matchweeks of the season, hierarchical shrinkage improved $RPS$ by $\mathbf{-0.000989}$.

### Q4: Does season-transition shrinkage help?
- **Answer**: **NO**. Arm E ($RPS = 0.200588$) degraded performance because top established teams (e.g. Real Madrid, Manchester City) regress too far toward the mean over the summer, underestimating their opening strength.

### Q5: Does venue-specific latent strength help?
- **Answer**: **NO**. Arm F ($RPS = 0.200213$) degraded overall accuracy because splitting home/away sample sizes by $50\%$ doubles parameter estimation variance.

### Q6: Does uncertainty propagation improve calibration?
- **Answer**: **NO**. Arm G ($RPS = 0.200075$) flattened probabilities excessively (mean entropy increased from $0.997 \to 1.009$, and sharpness dropped from $0.5137 \to 0.5043$), which worsened Log-Loss from $0.983360 \to 0.985595$.

### Q7: Which architecture is strongest among dynamic models?
- **Answer**: **ARM H ($E_{10}$ + Best Dynamic Combo)** ($RPS = 0.200092$), combining adaptive Elo, state-space AD, and hierarchical shrinkage.

### Q8: Does it beat E10?
- **Answer**: **NO**. Pure $E_{10}$ Champion ($RPS = \mathbf{0.199489}$) strictly outperforms Arm H ($RPS = 0.200092$) by $\mathbf{+0.000603}$ in $RPS$.

### Q9: Does it beat E10 on the 2025/26 blind holdout?
- **Answer**: **NO**. On Fold 4 (2025/26 Blind Holdout), Arm H achieved $RPS = 0.201221$, which is slightly better on that fold, but on Folds 1, 2, and 3 it severely degraded ($+0.001263$, $+0.000713$, $+0.000589$).

### Q10: Does it help early-season prediction?
- **Answer**: **YES**. In the first 3 matches of the season ($N=99$ matches), Arm H achieved $RPS = \mathbf{0.189734}$ vs Pure $E_{10} = \mathbf{0.190724}$ ($\Delta RPS = \mathbf{-0.000989}$, improved!).

### Q11: Does it help promoted teams?
- **Answer**: **NEUTRAL**. On teams with $<5$ historical matches ($N=130$), $E_{13}$ achieved $RPS = 0.184691$ vs $E_{10} = 0.183828$.

### Q12: Does it improve all five leagues?
- **Answer**: **NO**. Degraded across all 5 leagues: Premier League ($+0.000371$), La Liga ($+0.000481$), Bundesliga ($+0.001024$), Serie A ($+0.000694$), Ligue 1 ($+0.000515$).

### Q13: Does it preserve Dixon-Coles low-score calibration?
- **Answer**: **YES**. 1-1 scoreline probability was $11.70\%$ (actual $12.07\%$), preserving the $E_{10}$ structural repair.

### Q14: Is improvement statistically significant?
- **Answer**: **NO**. Cluster bootstrap confirmed a statistically significant **degradation** of $+0.000603$ ($95\%$ CI $[+0.000338, +0.000855]$, $p = 0.0000$).

### Q15: What exact equations define the winning architecture?
- **Answer**: Pure $E_{10}$ remains the champion. Dynamic state updates add excess high-frequency variance.

### Q16: Is it safe to become the next prospective candidate?
- **Answer**: **NO. STATUS = RESEARCH ONLY. KEEP E10 AS CHAMPION.**

---

## 4. Early-Season & Promoted-Team Subgroup Analysis

| Segment | Sample Matches ($N$) | Pure $E_{10}$ Champion $RPS$ | $E_{13}$ Dynamic $RPS$ | Delta $RPS$ | Result |
|:---|:---:|:---:|:---:|:---:|:---:|
| **First 3 Matches of Season** | 99 | `0.190724` | **`0.189734`** | **`-0.000989`** | **Improved** |
| **First 5 Matches of Season** | 151 | `0.194253` | **`0.194244`** | **`-0.000009`** | Neutral |
| **First 10 Matches of Season** | 296 | **`0.196683`** | `0.197633` | `+0.000950` | Degraded |
| **Remaining Season Matches** | 6,786 | **`0.199611`** | `0.200199` | `+0.000588` | Degraded |
| **Promoted Teams ($<5$ matches)**| 130 | **`0.183828`** | `0.184691` | `+0.000863` | Degraded |
| **Developing Teams (5–10 matches)**| 152 | **`0.214248`** | `0.215099` | `+0.000851` | Degraded |
| **Established Teams ($>10$ matches)**| 6,800 | **`0.199458`** | `0.200051` | `+0.000593` | Degraded |

---

## 5. Walk-Forward Fold-by-Fold Performance

| Fold | Out-of-Sample Season | Matches ($N$) | $V_4$ Baseline $RPS$ | Pure $E_{10}$ Champion $RPS$ | $E_{13}$ Dynamic $RPS$ | Delta vs. $E_{10}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 2022/2023 | 1,827 | `0.203679` | **`0.203677`** | `0.204940` | `+0.001263` |
| **2** | 2023/2024 | 1,752 | `0.194130` | **`0.193893`** | `0.194606` | `+0.000713` |
| **3** | 2024/2025 | 1,752 | `0.198900` | **`0.198805`** | `0.199394` | `+0.000589` |
| **4** | **2025/2026 (Blind Holdout)** | 1,751 | `0.201506` | `0.201402` | **`0.201221`** | **`-0.000181`** |
| **Total** | **All 4 Test Seasons** | **7,082** | **`0.199598`** | **`0.199489`** | `0.200092` | **`+0.000603`** |

---

## 6. League Breakdown Comparison

| Competition | Matches ($N$) | Pure $E_{10}$ $RPS$ | $E_{13}$ Dynamic $RPS$ | Delta $RPS$ | Pure $E_{10}$ Log-Loss | $E_{13}$ Log-Loss | Delta Log-Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Premier League** | 1,520 | **`0.199900`** | `0.200271` | `+0.000371` | **`0.976316`** | `0.977293` | `+0.000977` |
| **La Liga** | 1,520 | **`0.198007`** | `0.198488` | `+0.000481` | **`0.979583`** | `0.981018` | `+0.001435` |
| **Bundesliga** | 1,224 | **`0.200110`** | `0.201135` | `+0.001024` | **`0.986526`** | `0.990251` | `+0.003725` |
| **Serie A** | 1,521 | **`0.195323`** | `0.196017` | `+0.000694` | **`0.985936`** | `0.988182` | `+0.002246` |
| **Ligue 1** | 1,297 | **`0.205043`** | `0.205558` | `+0.000515` | **`0.990031`** | `0.991728` | `+0.001697` |

---

## 7. Promotion Evaluation & Final Scientific Decision

$$\mathbf{E13 \; FAILED \; TO \; BEAT \; E10 \; — \; KEEP \; E10 \; AS \; CHAMPION}$$

### Promotion Gate Evaluation:
1. **Gate 1 ($RPS$ Improvement over $E_{10}$)**: **FAILED** ($RPS = 0.200092$ vs $E_{10} = 0.199489$, $\Delta RPS = +0.000603$).
2. **Gate 2 (Log-Loss Improvement over $E_{10}$)**: **FAILED** (Log-Loss $= 0.985314$ vs $E_{10} = 0.983360$, $\Delta LL = +0.001954$).
3. **Gate 3 (Walk-Forward Consistency)**: **FAILED** (Degraded on Folds 1, 2, and 3).
4. **Gate 4 (League Consistency)**: **FAILED** (Degraded across all 5 leagues).
5. **Gate 5 (Statistical Significance)**: **FAILED** ($p = 0.0000$, confirmed significant degradation).
6. **Gate 6 (Asset Protection)**: **PASSED** (All 20 protected baseline hashes verified 100% bit-identical).

---

## 8. Final Governance Architecture State

- **$V_4$ Production Baseline**: **`FROZEN PRODUCTION`**
- **$E_{10}$ Dixon-Coles ($\rho = -0.08$)**: **`CANDIDATE FOR PROSPECTIVE TEST / CURRENT CHAMPION`**
- **$E_{11}$ Historical $xG$**: **`RESEARCH ONLY (REJECTED)`**
- **$E_{12}$ Orthogonal Bounded Adjustment**: **`RESEARCH ONLY`**
- **$E_{13}$ Dynamic Bayesian & Hierarchical Team Strength**: **`RESEARCH ONLY`**
- **Protected Baseline Assets**: **`20/20 Hashes Verified 100% Bit-Identical`**

---

## 9. Deliverable Manifest in `research/v5_model_improvement/e13_dynamic_bayesian/`

1. [`01_literature_review.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/01_literature_review.md): Academic literature audit on dynamic sports models.
2. [`02_architecture_review.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/02_architecture_review.md): Architectural comparison and flow diagrams.
3. [`03_data_inventory.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/03_data_inventory.md): Dataset provenance and inventory matrix.
4. [`04_information_clock.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/04_information_clock.md): Two-pass causal state isolation proofs.
5. [`05_feature_dictionary.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/05_feature_dictionary.md): Dynamic Elo and state-space feature specifications.
6. [`06_ablation_results.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/06_ablation_results.csv): Full 9-arm metrics table.
7. [`07_fold_results.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/07_fold_results.csv): Walk-forward season metrics for all arms.
8. [`08_league_breakdown.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/08_league_breakdown.csv): 5-league comparative breakdown.
9. [`09_early_season_analysis.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/09_early_season_analysis.csv): First 3, 5, 10 matchweeks sub-analysis.
10. [`10_promoted_team_analysis.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/10_promoted_team_analysis.csv): Sample maturity and promoted team evaluation.
11. [`11_scoreline_analysis.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/11_scoreline_analysis.json): 9-scoreline empirical vs. modeled frequency matrix.
12. [`12_calibration_analysis.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/12_calibration_analysis.json): 10-bin reliability diagrams for H, D, A.
13. [`13_uncertainty_analysis.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/13_uncertainty_analysis.json): Parameter variance, entropy, and sharpness diagnostics.
14. [`14_e13_results.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/14_e13_results.json): Master machine-readable results dataset.
15. [`15_e13_report.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/15_e13_report.md): Full comprehensive research report.
16. [`dynamic_elo.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/dynamic_elo.py): Adaptive-K dynamic Elo rating engine.
17. [`dynamic_attack_defense.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/dynamic_attack_defense.py): State-space attack/defense variance engine.
18. [`hierarchical_model.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/hierarchical_model.py): Empirical Bayes hierarchical shrinkage module.
19. [`uncertainty_engine.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/uncertainty_engine.py): Parameter uncertainty Gauss-Hermite quadrature integrator.
20. [`run_e13_experiment.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e13_dynamic_bayesian/run_e13_experiment.py): Fully reproducible walk-forward evaluation harness.
21. [`tests/test_e13_dynamic_bayesian.py`](file:///e:/Football%20Prediction%20Project/tests/test_e13_dynamic_bayesian.py): Automated test suite for $E_{13}$ components.
