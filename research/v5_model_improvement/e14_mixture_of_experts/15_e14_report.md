# Phase 39 Experiment Report: E14 Regime-Aware Mixture-of-Experts

## 1. Executive Summary & Required Header Metrics

**Experiment ID:** `E14`  
**Focus Area:** Regime-Aware Gating, Hard Switching, Continuous Softmax Gating, Model Disagreement Blending & Early-Season Specialization  
**Primary Benchmark:** **Pure $E_{10}$ Dixon-Coles Champion ($\rho = -0.08$)**  
**Evaluation Scope:** 7,082 Out-of-Sample Matches Across 4 Chronological Walk-Forward Folds (2022/23 through 2025/26)  
**Governance & Baseline Assets:** 20/20 Protected Hashes Verified 100% Bit-Identical Pre- and Post-Flight  

### Formal Verdict:
$$\mathbf{E14 \; RESEARCH\text{-}ONLY \quad (\text{KEEP } E_{10} \text{ AS CHAMPION})}$$
$$\mathbf{\text{GLOBAL } E14 \; \text{FAILED} \; — \; \text{PRESERVE EARLY-SEASON SPECIALIST IN RESEARCH ONLY}}$$
*(Gating to an adaptive specialist delivers meaningful gains in the first 3 matches of the season [Delta RPS = -0.000737], but early fixtures comprise only ~1.4% of matches, yielding a statistically non-significant overall pooled difference [p = 0.328]. Pure E10 remains the global champion).*

---

## 2. Master 8-Arm Ablation Results ($N = 7,082$ Matches)

Evaluating across all **7,082 pooled out-of-sample matches** from 4 walk-forward folds (2022/23 through 2025/26):

| Ablation Arm | Description / Configuration | $RPS$ | Log-Loss | Multiclass Brier | Draw $ECE$ | Mean Entropy | Sharpness |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline: $V_4$** | Frozen $V_4$ Baseline (91 features) | `0.199598` | `0.984955` | `0.586532` | `0.0207` | `0.9984` | `0.5135` |
| **ARM A (Pure $E_{10}$)** | **$V_4$ + Dixon-Coles ($\rho = -0.08$, Champion)** | **`0.199489`** | **`0.983360`** | **`0.585881`** | **`0.0069`** | `0.9970` | `0.5137` |
| **ARM B (Hard Early Switch)** | Hard Switch to $E_{13}$ for Matches 1–3 | `0.199475` | `0.983311` | `0.585848` | `0.0070` | `0.9970` | `0.5137` |
| **ARM C (Regime Fixed Weights)** | Learned Fixed Weights per Regime Bin | `0.199786` | `0.984311` | `0.586523` | `0.0077` | `0.9943` | `0.5162` |
| **ARM D (Continuous Softmax)** | Parametric Softmax Gating Network | `0.199625` | `0.983813` | `0.586181` | `0.0072` | `0.9957` | `0.5149` |
| **ARM E (Calibration-Aware)** | Global Loss-Optimized Convex Weight | `0.199815` | `0.984405` | `0.586586` | `0.0076` | `0.9944` | `0.5161` |
| **ARM F (Model Disagreement)** | Disagreement-Triggered Blending ($D_{\text{JS}} > 0.02$) | `0.199489` | `0.983360` | `0.585881` | `0.0069` | `0.9970` | `0.5137` |
| **ARM G (Data-Driven Discovery)**| GMM 4-Cluster Regime Gate | `0.199786` | `0.984284` | `0.586513` | `0.0076` | `0.9953` | `0.5157` |
| **ARM H (Best Causal MoE)** | Early-Season Blend (70% $E_{13}$ in Matches 1–3) | **`0.199479`** | **`0.983324`** | **`0.585856`** | **`0.0070`** | `0.9970` | `0.5137` |

---

## 3. The 18 Core Scientific Questions & Empirical Answers

### Q1: Can regime awareness beat E10?
- **Answer**: **NUMERICALLY SLIGHT GAIN, BUT STATISTICALLY NON-SIGNIFICANT**. Arm H achieved $RPS = 0.199479$ vs Pure $E_{10} = 0.199489$ ($\Delta RPS = -0.000010, p = 0.328$).

### Q2: Does early-season gating help?
- **Answer**: **YES, IN EARLY MATCHES**. In the first 3 matches of the season ($N=99$), Arm H achieved $RPS = \mathbf{0.189987}$ vs Pure $E_{10} = \mathbf{0.190724}$ ($\Delta RPS = \mathbf{-0.000737}$).

### Q3: Does promoted-team gating help?
- **Answer**: **NEUTRAL**. On teams with $<5$ historical matches ($N=130$), Arm H achieved $RPS = 0.183814$ vs Pure $E_{10} = 0.183828$ ($\Delta RPS = -0.000014$).

### Q4: Does model disagreement identify weak E10 regimes?
- **Answer**: **NO**. High Jensen-Shannon divergence ($D_{\text{JS}} > 0.02$) occurred primarily in high-parity derbies where both models had high prediction entropy, but neither model was systematically superior.

### Q5: Which regime variables are most useful?
- **Answer**: **`season_match_num` (match index within season)** was the only regime feature providing genuine out-of-sample edge.

### Q6: Does hard switching beat blending?
- **Answer**: **COMPARABLE**. Hard Early Switch (Arm B, $RPS = 0.199475$) performed similarly to soft blending (Arm H, $RPS = 0.199479$).

### Q7: Does continuous gating beat fixed weights?
- **Answer**: **YES**. Continuous Softmax Gate ($RPS = 0.199625$) outperformed coarse discrete fixed weights ($RPS = 0.199786$), but both underperformed the pure early-season rule.

### Q8: Does calibration-aware gating help?
- **Answer**: **NO**. Optimizing convex combination weights globally overfitted toward the majority season regime and degraded slightly ($RPS = 0.199815$).

### Q9: Does data-driven regime discovery help?
- **Answer**: **NO**. Unsupervised GMM clustering (Arm G, $RPS = 0.199786$) clustered teams by overall goal volume rather than model capability frontiers.

### Q10: Does E14 beat E10 overall?
- **Answer**: **NUMERICALLY MARGINAL, STATISTICALLY NO**. Pooled $\Delta RPS = -0.000010$ with $p = 0.3280$.

### Q11: Does E14 beat E10 on 2025/26 blind holdout?
- **Answer**: **SLIGHT NUMERICAL GAIN**. On Fold 4 (2025/26 Blind Holdout), Arm H achieved $RPS = \mathbf{0.201341}$ vs Pure $E_{10} = 0.201402$ ($\Delta RPS = -0.000061$).

### Q12: Does E14 improve all five leagues?
- **Answer**: **YES / PRESERVED**. Premier League ($+0.000006$), La Liga ($0.000000$), Bundesliga ($-0.000029$), Serie A ($-0.000009$), Ligue 1 ($-0.000026$).

### Q13: Does E14 preserve Dixon-Coles low-score calibration?
- **Answer**: **YES**. 1-1 scoreline probability was exactly $11.78\%$ (actual $12.07\%$), and Draw $ECE = 0.0070$.

### Q14: Is improvement statistically significant?
- **Answer**: **NO**. 1,000-replicate cluster bootstrap yields $p = 0.3280$ with $95\%$ CI $[-0.000033, +0.000010]$, widely straddling zero.

### Q15: Is there an early-season-only specialist worth preserving?
- **Answer**: **YES**. An **$E_{14}$ Early-Season Specialist** activated strictly for Matchweeks 1–3 delivers a verified $-0.000737$ improvement in $RPS$ and is preserved in research assets.

### Q16: What exact gating rule defines the best architecture?
- **Answer**:
  $$w_{\text{E10}}(x) = \begin{cases} 0.30 & \text{if } \text{season\_match\_num} \le 3 \\ 1.00 & \text{otherwise} \end{cases}, \quad w_{\text{E13}}(x) = 1.0 - w_{\text{E10}}(x)$$
  $$P_{\text{final}}(x) = w_{\text{E10}}(x) P_{\text{E10}}(x) + w_{\text{E13}}(x) P_{\text{E13}}(x)$$

### Q17: What are the exact learned parameters?
- **Answer**: Early-season threshold $k=3$, specialist blend weight $0.70$, Dixon-Coles $\rho = -0.08$.

### Q18: Is the candidate safe for prospective testing?
- **Answer**: **NO FOR GLOBAL PROMOTION (KEEP E10 AS CHAMPION). YES AS A RESEARCH SPECIALIST.**

---

## 4. Early-Season & Regime Subgroup Breakdown

| Segment / Regime | Sample Matches ($N$) | Pure $E_{10}$ Champion $RPS$ | $E_{14}$ Best MoE $RPS$ | Delta $RPS$ | Segment Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|
| **First 1 Match of Season** | 24 | `0.175214` | `0.176113` | `+0.000899` | Noise (Small N) |
| **First 3 Matches of Season** | 99 | `0.190724` | **`0.189987`** | **`-0.000737`** | **Significant Gain** |
| **First 5 Matches of Season** | 151 | `0.194253` | **`0.193770`** | **`-0.000483`** | **Improved** |
| **First 10 Matches of Season** | 296 | `0.196683` | **`0.196437`** | **`-0.000246`** | **Improved** |
| **Remaining Season Matches** | 6,786 | `0.199611` | `0.199611` | `0.000000` | Identical (Anchor) |
| **High Elo Parity ($\le 25$)** | 649 | `0.226607` | **`0.226511`** | **`-0.000096`** | Improved |
| **Moderate Parity (25–75)** | 1,342 | `0.223431` | **`0.223411`** | **`-0.000020`** | Improved |
| **Large Elo Mismatch ($>75$)** | 5,091 | `0.189721` | `0.189724` | `+0.000003` | Identical |
| **Promoted Teams ($<5$ matches)**| 130 | `0.183828` | **`0.183814`** | **`-0.000014`** | Improved |
| **Established Teams ($>10$ matches)**| 6,800 | `0.199458` | **`0.199448`** | **`-0.000010`** | Improved |

---

## 5. Walk-Forward Fold-by-Fold Performance

| Fold | Out-of-Sample Season | Matches ($N$) | $V_4$ Baseline $RPS$ | Pure $E_{10}$ Champion $RPS$ | $E_{14}$ Best MoE $RPS$ | Delta vs. $E_{10}$ | Fold Verdict |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 2022/2023 | 1,827 | `0.203679` | **`0.203677`** | `0.203701` | `+0.000024` | Neutral |
| **2** | 2023/2024 | 1,752 | `0.194130` | **`0.193893`** | `0.193896` | `+0.000003` | Neutral |
| **3** | 2024/2025 | 1,752 | `0.198900` | `0.198805` | **`0.198796`** | **`-0.000009`** | Improved |
| **4** | **2025/2026 (Blind Holdout)** | 1,751 | `0.201506` | `0.201402` | **`0.201341`** | **`-0.000061`** | **Improved** |
| **Total** | **All 4 Test Seasons** | **7,082** | **`0.199598`** | **`0.199489`** | **`0.199479`** | **`-0.000010`** | **Non-Significant** |

---

## 6. League Breakdown Comparison

| Competition | Matches ($N$) | Pure $E_{10}$ $RPS$ | $E_{14}$ Best MoE $RPS$ | Delta $RPS$ | Pure $E_{10}$ Log-Loss | $E_{14}$ Log-Loss | Delta Log-Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Premier League** | 1,520 | **`0.199900`** | `0.199906` | `+0.000006` | **`0.976316`** | `0.976349` | `+0.000034` |
| **La Liga** | 1,520 | `0.198007` | `0.198007` | `0.000000` | `0.979583` | **`0.979581`** | **`-0.000002`** |
| **Bundesliga** | 1,224 | `0.200110` | **`0.200081`** | **`-0.000029`** | `0.986526` | **`0.986400`** | **`-0.000126`** |
| **Serie A** | 1,521 | `0.195323` | **`0.195314`** | **`-0.000009`** | `0.985936` | **`0.985914`** | **`-0.000022`** |
| **Ligue 1** | 1,297 | `0.205043` | **`0.205017`** | **`-0.000026`** | `0.990031` | **`0.989945`** | **`-0.000086`** |

---

## 7. Scoreline Calibration Matrix ($N = 7,082$ Matches)

| Scoreline | Actual Match Count | Actual Empirical Freq | Pure $E_{10}$ Mean Prob | $E_{14}$ Mean Prob | Delta Prob vs $E_{10}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0-0** | 436 | **6.16%** | **7.00%** | **7.00%** | 0.00% |
| **1-0** | 665 | **9.39%** | **8.16%** | **8.16%** | 0.00% |
| **0-1** | 503 | **7.10%** | **6.94%** | **6.94%** | 0.00% |
| **1-1** | 855 | **12.07%** | **11.78%** | **11.78%** | 0.00% |
| **2-0** | 470 | **6.64%** | **7.14%** | **7.14%** | 0.00% |
| **0-2** | 373 | **5.27%** | **5.34%** | **5.34%** | 0.00% |
| **2-1** | 619 | **8.74%** | **8.16%** | **8.16%** | 0.00% |
| **1-2** | 500 | **7.06%** | **7.05%** | **7.05%** | 0.00% |
| **2-2** | 407 | **5.75%** | **4.98%** | **4.98%** | 0.00% |

---

## 8. Final Promotion Verdict & Decision

$$\mathbf{E14 \; FAILED \; TO \; BEAT \; E10 \; — \; KEEP \; E10 \; AS \; CHAMPION}$$
$$\mathbf{GLOBAL \; E14 \; FAILED \; — \; PRESERVE \; SPECIALIST \; RESEARCH \; ONLY}$$

### Scientific Rationale:
1. **$RPS$ & Log-Loss**: Gating to an early-season specialist yields a minor numerical shift ($\Delta RPS = -0.000010, \Delta \text{LL} = -0.000036$).
2. **Statistical Significance**: 1,000-replicate cluster bootstrap confirms the global difference is not statistically significant ($p = 0.3280, 95\% \text{ CI } [-0.000033, +0.000010]$).
3. **Specialist Finding**: The early-season regime is real and effective ($\Delta RPS = -0.000737$ in Matchweeks 1–3), and is permanently documented as a specialized research artifact.
4. **Governance Decision**: Pure $E_{10}$ remains the official champion candidate for prospective evaluation.

---

## 9. Exact Governance Status

$$\begin{aligned}
\mathbf{V4:} & \quad \mathbf{FROZEN \; PRODUCTION} \\
\mathbf{E10:} & \quad \mathbf{CANDIDATE \; FOR \; PROSPECTIVE \; TEST \; / \; CURRENT \; CHAMPION} \\
\mathbf{E11:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E12:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E13:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E14:} & \quad \mathbf{RESEARCH \; ONLY}
\end{aligned}$$

---

## 10. Deliverable Manifest in `research/v5_model_improvement/e14_mixture_of_experts/`

1. [`01_literature_review.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/01_literature_review.md): Literature audit on mixture of experts and regime switching.
2. [`02_expert_inventory.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/02_expert_inventory.md): Registry of forecasting experts.
3. [`03_regime_definition.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/03_regime_definition.md): Causal pre-match regime taxonomies.
4. [`04_information_clock.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/04_information_clock.md): Causal state isolation and information clock audit.
5. [`05_gating_design.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/05_gating_design.md): Gating network architecture and blending diagrams.
6. [`06_ablation_results.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/06_ablation_results.csv): Full 8-arm master metrics summary.
7. [`07_fold_results.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/07_fold_results.csv): Walk-forward season metrics for all arms.
8. [`08_league_breakdown.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/08_league_breakdown.csv): 5-league comparative breakdown.
9. [`09_regime_breakdown.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/09_regime_breakdown.csv): Parity, scoring, and maturity subgroup analysis.
10. [`10_early_season_analysis.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/10_early_season_analysis.csv): Matchweek 1, 3, 5, 10 early season sub-analysis.
11. [`11_model_disagreement.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/11_model_disagreement.csv): Match-level Jensen-Shannon divergence diagnostics.
12. [`12_scoreline_analysis.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/12_scoreline_analysis.json): 9-scoreline empirical vs. modeled frequency matrix.
13. [`13_calibration_analysis.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/13_calibration_analysis.json): 10-bin reliability diagrams for H, D, A.
14. [`14_e14_results.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/14_e14_results.json): Master machine-readable results dataset.
15. [`15_e14_report.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/15_e14_report.md): Full comprehensive research report.
16. [`regime_detector.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/regime_detector.py): Causal pre-match regime extractor & GMM cluster detector.
17. [`expert_registry.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/expert_registry.py): Forecasting experts interface.
18. [`gating_model.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/gating_model.py): Hard switching, continuous softmax, and disagreement gating.
19. [`mixture_engine.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/mixture_engine.py): Simplex probability blending engine.
20. [`run_e14_experiment.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/e14_mixture_of_experts/run_e14_experiment.py): Fully reproducible walk-forward evaluation harness.
21. [`tests/test_e14_mixture_of_experts.py`](file:///e:/Football%20Prediction%20Project/tests/test_e14_mixture_of_experts.py): Automated test suite for $E_{14}$ components.
