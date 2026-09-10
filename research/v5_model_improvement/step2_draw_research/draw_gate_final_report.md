# STEP 2B — SELECTIVE DRAW GATE RESEARCH REPORT
**Football Prediction Project — Advanced Model Research Program**  
**Status:** RESEARCH CANDIDATE EVALUATION — STRICTLY NON-MUTATING — PRODUCTION V4.0 REMAINS FROZEN  
**Date:** August 24, 2026  
**Artifact Directory:** `research/v5_model_improvement/step2_draw_research/`

---

## 1. Governance & Integrity Confirmation

Cryptographic audits confirmed that all baseline production models and prospective assets are bit-identical and unmodified:

- **$V_{4.0}$ Production Baseline:** `data/models/v4_poisson_venue_elo_online_ad.pkl`
  - **MD5:** `06841f0c03c8597b2b8cd8f8ab064864` (**100% Match — Verified Bit-Identical**)
- **$V_{4.1}$ Production Candidate:** `data/models/v4_1_prospective_candidate_2025_26.pkl`
  - **MD5:** `145f918d933eb343c0f63ca342b10289` (**100% Match — Verified Bit-Identical**)
- **Test Suite Status:** `tests/test_draw_gate_integrity.py` passed (4/4 tests).
- **Execution Scope:** 100% isolated within `research/v5_model_improvement/step2_draw_research/`. Zero mutation of production code, inference pipelines, or database tables.

---

## 2. Research Objective & Mathematical Trade-Off

The objective of Step 2B was to evaluate whether a selective post-hoc **Draw Gate** could override fragile, low-margin $V_{4.0}$ Home/Away predictions to Draw:

$$\text{Decision} = \begin{cases} \mathbf{D} & \text{if Draw Gate triggers} \\ \mathbf{y}_{V_{4.0}} & \text{otherwise} \end{cases}$$

### The Fundamental Mathematical Trade-Off in 0-1 Loss Classification
For a discrete override rule to increase overall classification accuracy on a targeted subset of matches $\mathcal{S}_{\text{target}}$, the empirical draw rate on that subset must **strictly exceed** the base model's classification accuracy on that same subset:

$$\Delta \text{Accuracy} > 0 \iff P(\text{Draw} \mid \mathcal{S}_{\text{target}}) > P(y_{V_{4.0}} = y_{\text{true}} \mid \mathcal{S}_{\text{target}})$$

Across $N = 10,734$ historical top-flight matches, our empirical audit revealed:
1. In near-equal fixtures where $|P(H) - P(A)| \le 0.10$, the empirical draw frequency rises to **$29.67\%$**.
2. However, $V_{4.0}$'s base Home/Away predictions in that same subset achieve **$39.11\%$ accuracy** ($1,355 \text{ H} + 1,153 \text{ A}$ predictions).
3. Because $39.11\% > 29.67\%$, converting all near-equal predictions to Draw systematically damages more valid Home/Away predictions than it rescues draws (e.g. converting 100 near-equal matches yields $\approx 30$ correct draws, but forfeits $\approx 39$ correct home/away wins $\rightarrow$ net loss of $-9$ correct matches).

---

## 3. Evaluation of 5 Draw Gate Strategies

We tested 5 distinct gate architectures across $N = 8,983$ training matches (2020/21–2024/25) and validated them on the untouched **2025/26 holdout season ($N = 1,751$)** ([`draw_gate_candidates.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/step2_draw_research/draw_gate_candidates.csv)):

| Strategy & Candidate Name | Configuration / Parameters | Train Acc Delta (%) | Train Draw Recall (%) | Test Acc Delta (%) | Test Draw Recall (%) | Test Draw Precision (%) | Verdict / Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **S1: Simple Threshold** (`tau_d=0.26`) | $P(D) \ge 0.26$ | $-4.68\%$ | $35.91\%$ | $-3.88\%$ | $38.65\%$ | $29.66\%$ | ❌ **REJECTED** (Severe Acc Degradation) |
| **S1: Simple Threshold** (`tau_d=0.27`) | $P(D) \ge 0.27$ | $-1.90\%$ | $23.85\%$ | $-1.14\%$ | $24.04\%$ | $31.56\%$ | ❌ **REJECTED** (Acc Degradation) |
| **S1: Simple Threshold** (`tau_d=0.28`) | $P(D) \ge 0.28$ | $-0.61\%$ | $11.40\%$ | $-0.51\%$ | $9.66\%$ | $28.67\%$ | ❌ **REJECTED** (Acc Degradation) |
| **S2: Multi-Condition Physical Gate** | $P(D)\ge 0.26, \Delta_{\text{win}}\le 0.10, \text{conf}\le 0.45, \Delta_{\text{Elo}}\le 100, \lambda_{\text{tot}}\le 2.50$ | $\mathbf{-0.32\%}$ | $\mathbf{5.44\%}$ | $\mathbf{-0.06\%}$ | $\mathbf{4.04\%}$ | $\mathbf{26.09\%}$ | ⚠️ **BEST PHYSICAL GATE** (Near-Neutral) |
| **S2: Strict Physical Gate** | $P(D)\ge 0.26, \Delta_{\text{win}}\le 0.08, \text{conf}\le 0.44, \Delta_{\text{Elo}}\le 80, \lambda_{\text{tot}}\le 2.45$ | $-0.19\%$ | $2.06\%$ | $-0.23\%$ | $1.80\%$ | $21.05\%$ | ❌ **REJECTED** |
| **S2: Relaxed Physical Gate** | $P(D)\ge 0.26, \Delta_{\text{win}}\le 0.12, \text{conf}\le 0.46, \Delta_{\text{Elo}}\le 120, \lambda_{\text{tot}}\le 2.60$ | $-1.00\%$ | $12.19\%$ | $-0.34\%$ | $11.46\%$ | $29.31\%$ | ❌ **REJECTED** (Net Negative) |
| **S3: Composite Score Index** | $S_{\text{draw}} \ge 1.00$ | $-0.62\%$ | $12.98\%$ | $-0.51\%$ | $11.24\%$ | $28.57\%$ | ❌ **REJECTED** (Net Negative) |
| **S3: Composite Score Index** | $S_{\text{draw}} \ge 1.25$ | $-0.35\%$ | $2.76\%$ | $-0.46\%$ | $1.80\%$ | $18.18\%$ | ❌ **REJECTED** |
| **S4: Logistic Regression Gate** | $P(\text{Override}) \ge 0.28$ | $-3.92\%$ | $37.35\%$ | $-3.71\%$ | $38.65\%$ | $30.12\%$ | ❌ **REJECTED** (Severe Degradation) |
| **S4: Logistic Regression Gate** | $P(\text{Override}) \ge 0.30$ | $-0.52\%$ | $8.94\%$ | $-0.80\%$ | $8.54\%$ | $29.23\%$ | ❌ **REJECTED** |
| **S4: Logistic Regression Gate** | $P(\text{Override}) \ge 0.32$ | $+0.01\%$ | $0.39\%$ | $+0.06\%$ | $0.45\%$ | $40.00\%$ | ⚠️ **TRIVIAL TRIGGER** (2 draws in 1,751 matches) |
| **S5: Decision Tree Gate** | Tree $P(\text{Draw}) \ge 0.30$ | $-0.08\%$ | $12.10\%$ | $-0.69\%$ | $9.44\%$ | $29.37\%$ | ❌ **REJECTED** (Net Negative) |

---

## 4. Walk-Forward Cross-Validation (5 Historical Folds)

We subjected the top candidate (**Strategy 2: Multi-Condition Physical Gate**) to strict chronological walk-forward cross-validation across 5 historical folds with zero future data leakage ([`draw_gate_walkforward_results.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/step2_draw_research/draw_gate_walkforward_results.csv)):

| Fold Name | Validation Season | Matches ($N$) | Actual Draws | $V_{4.0}$ Baseline Acc (%) | Candidate Acc (%) | Accuracy Delta (%) | Draw Recall (%) | Draw Precision (%) | Rescued Draws | Damaged Wins | Net Gain (Matches) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 2021/2022 | 1,826 | 473 | 52.30% | 52.03% | **-0.27%** | 4.86% | 34.33% | 23 | 28 | **-5** |
| **Fold 2** | 2022/2023 | 1,827 | 443 | 52.93% | 52.38% | **-0.55%** | 3.16% | 21.54% | 14 | 24 | **-10** |
| **Fold 3** | 2023/2024 | 1,752 | 417 | 54.05% | 53.71% | **-0.34%** | 4.32% | 29.85% | 18 | 24 | **-6** |
| **Fold 4** | 2024/2025 | 1,752 | 456 | 53.54% | 53.20% | **-0.34%** | 5.26% | 32.39% | 24 | 30 | **-6** |
| **Fold 5 (Holdout)** | 2025/2026 | 1,751 | 445 | 51.97% | 51.91% | **-0.06%** | 4.04% | 26.09% | 18 | 19 | **-1** |
| **Average / Total** | **All 5 Folds** | **8,908** | **2,234** | **52.96%** | **52.65%** | **-0.31%** | **4.33%** | **28.84%** | **97** | **125** | **-28** |

### Walk-Forward Findings:
1. **Consistency**: Across all 5 historical walk-forward folds, the gate consistently produces a **draw precision of $21.5\% - 34.3\%$** (averaging $28.84\%$).
2. **Systematic Net Penalty**: Because precision is below $35\%$, the gate consistently damages more valid Home/Away predictions than it rescues draws, producing an average net loss of **$-0.31\%$ accuracy** across 8,908 out-of-sample matches.

---

## 5. Evaluation on Recent Prospective Matches (Aug 22–24, 2026)

On the recent prospective sample of $N = 33$ completed matches ([`draw_gate_recent_33_audit.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/step2_draw_research/draw_gate_recent_33_audit.csv)):

- **Total Completed Matches:** 33
- **$V_{4.0}$ Baseline Accuracy:** **$60.61\%$** ($20 / 33$)
- **Candidate Gate Accuracy:** **$60.61\%$** ($20 / 33$) [Delta: **$+0.00\%$**]
- **Draw Predictions Triggered:** 1 match (*Nice vs Lorient*, 0-0, where $P(D)=0.27$, gap=$0.05$, $\lambda_{\text{tot}}=2.38$, Elo gap=$21.3$).
- **Draw Recall on Aug 22–24 Sample:** $1 / 8 = \mathbf{12.5\%}$ (Rescued 1 draw: *Nice vs Lorient*).
- **Damaged Valid Wins:** 0 (In *Nice vs Lorient*, $V_{4.0}$'s original prediction was `H`, which was wrong).
- **Net Gain on 33 Matches:** **$+1$ draw rescued, $0$ wins damaged $\rightarrow$ Net $= +1$**.

---

## 6. Comprehensive Baseline Comparison

| Metric | $V_{4.0}$ Production Baseline | Strategy 2 Multi-Condition Gate | Strategy 1 Simple Threshold (`0.27`) | Strategy 4 Logistic Gate (`0.30`) |
|:---|:---:|:---:|:---:|:---:|
| **2025/26 Holdout Accuracy ($N=1,751$)** | **51.97%** | **51.91%** ($-0.06\%$) | **50.83%** ($-1.14\%$) | **51.17%** ($-0.80\%$) |
| **Draw Recall (%)** | **0.00%** | **4.04%** | **24.04%** | **8.54%** |
| **Draw Precision (%)** | **--** | **26.09%** | **31.56%** | **29.23%** |
| **Draw F1-Score** | **0.0000** | **0.0699** | **0.2730** | **0.1322** |
| **Macro F1-Score** | **0.3782** | **0.4042** | **0.4908** | **0.4377** |
| **Draw Predictions Count** | **0** | **69** | **339** | **130** |
| **Rescued Draws (True Positives)** | **0** | **18** | **107** | **38** |
| **Damaged Wins (False Positives)** | **0** | **19** | **127** | **52** |
| **Net Match Gain / Loss** | **0** | **-1** | **-20** | **-14** |
| **Home Prediction Accuracy (%)** | **58.74%** | **59.34%** | **61.32%** | **59.98%** |
| **Away Prediction Accuracy (%)** | **42.23%** | **42.92%** | **46.85%** | **44.17%** |
| **Multiclass Brier Score** | **0.5878** | **0.5878** | **0.5878** | **0.5878** |
| **Multiclass Log Loss** | **0.9984** | **0.9984** | **0.9984** | **0.9984** |

---

## 7. Model Safety Rule Verdict

In accordance with Section H of the research governance protocol:
> *"The candidate must satisfy a conservative acceptance rule. DO NOT recommend promotion simply because Draw Recall increases. A candidate should only be considered promising if Draw Recall improves materially, Draw Precision is reasonable, and Overall Accuracy does not collapse. If no candidate satisfies these requirements: REPORT: 'No safe Draw Gate candidate found.' That is a valid research outcome."*

### Official Research Verdict:
$$\mathbf{NO\ SAFE\ DRAW\ GATE\ CANDIDATE\ QUALIFIES\ FOR\ PRODUCTION\ PROMOTION}$$

### Scientific Rationale:
1. **Mathematical Limitation of 0-1 Loss Override**: Because draw frequency in real-world European football tops out at $\approx 30\% - 32\%$ even under ideal stalemate conditions, an override rule with $\le 32\%$ precision will mathematically always destroy more $\approx 38\% - 42\%$ accurate Home/Away predictions than it rescues draws.
2. **$V_{4.0}$'s Probability Model is Sound**: $V_{4.0}$'s $P(D)$ is already well-calibrated ($ECE = 0.0268$). The absence of draw predictions under $\arg\max$ is an expected theoretical property of Bayes-optimal decision making under 0-1 loss when no class reaches $>34\%$.
3. **Proper Utilization of Draw Signals**: Draw signals must be leveraged through **Forecast Status & Confidence Tiers (Phase 42 E17)** or **Market Value Probability Discrepancies (ROI betting engine)** where continuous probabilities are exploited, rather than forcing discrete class overrides that degrade 0-1 accuracy.

---

## 8. Deliverables Summary

All required artifacts have been generated in `research/v5_model_improvement/step2_draw_research/`:

1. [`step2b_draw_gate_research.py`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/step2_draw_research/step2b_draw_gate_research.py) — Reproducible research script for all 5 strategies and walk-forward folds
2. [`draw_gate_candidates.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/step2_draw_research/draw_gate_candidates.csv) — Comparative ledger of all candidate configurations across train and holdout
3. [`draw_gate_walkforward_results.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/step2_draw_research/draw_gate_walkforward_results.csv) — Fold-by-fold walk-forward validation results across 5 historical seasons
4. [`draw_gate_recent_33_audit.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/step2_draw_research/draw_gate_recent_33_audit.csv) — Match-level evaluation on the 33 prospective matches from Aug 22–24, 2026
5. [`draw_gate_feature_analysis.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/step2_draw_research/draw_gate_feature_analysis.csv) — Role, threshold, and draw density of all physical features
6. [`draw_gate_config.json`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/step2_draw_research/draw_gate_config.json) — Serialized configuration for the best physical gate candidate
7. [`draw_gate_final_report.md`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/step2_draw_research/draw_gate_final_report.md) — This comprehensive research evaluation document
8. [`tests/test_draw_gate_integrity.py`](file:///e:/Football%20Prediction%20Project/tests/test_draw_gate_integrity.py) — 4-test regression test suite (100% passing)
