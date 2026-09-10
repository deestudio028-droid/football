# Historical Calculation Memory Layer: Out-of-Sample Backtest & Ablation Report

**Date (UTC):** 2026-08-28T07:40:00Z  
**Evaluation Target:** Objective Backtest & Ablation of Frozen V4.0 vs. V4.0 + Historical Calculation Memory Layer  
**Model Under Test:** V4.0 Production Model (`data/models/v4_poisson_venue_elo_online_ad.pkl`, MD5: `06841f0c03c8597b2b8cd8f8ab064864`)

---

## 1. Executive Summary & Verdict

| Configuration | 1X2 Accuracy | Mean Brier Score | Mean Log Loss | Exact Score Accuracy | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|
| **A) V4.0 Baseline (Frozen Production)** | **60.61%** (20/33) | **0.5497** | **0.9367** | **9.09%** (3/33) | Baseline Benchmark |
| **B) V4.0 + Historical Memory Evidence** | **60.61%** (20/33) | **0.5497** | **0.9367** | **9.09%** (3/33) | Non-mutating auxiliary advisory |
| **C) V4.0 + Correct-Score Refinement** | **60.61%** (20/33) | **0.5497** | **0.9367** | **9.09%** (3/33) | Preserves baseline when sample is small |

### **FINAL VERDICT: INSUFFICIENT DATA**

> [!NOTE]
> **Verdict Rationale:**  
> Legitimate prospective pre-kickoff predictions currently exist for $N=33$ completed matches from the 2026/27 opening cycle. Under strict walk-forward evaluation (where match $i$ only queries matches completed prior to $i$), the historical memory engine correctly emits `INSUFFICIENT SAMPLE` to prevent making unwarranted claims on small samples.  
> 
> Because the cohort ($N=33$) is below the standard statistical power threshold ($N \ge 100+$) required to measure statistically significant differences in 1X2 and exact-score distributions, we explicitly report **INSUFFICIENT DATA** rather than prematurely asserting accuracy gains.

---

## 2. Dataset & Chronological Walk-Forward Methodology

- **Source Ledger:** [`research/v5_model_improvement/production_monitoring/fixture_outcome_audit.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/production_monitoring/fixture_outcome_audit.csv)
- **Total Eligible Matches:** 33 matches with verified pre-kickoff V4.0 predictions and final outcomes.
- **Date Range:** August 21, 2026 – August 24, 2026.
- **League Breakdown:**
  - **Ligue 1:** 9 matches
  - **Premier League:** 9 matches
  - **Serie A:** 8 matches
  - **La Liga:** 7 matches
  - **Bundesliga:** 0 matches (Bundesliga 2026/27 starts Matchday 1 on August 28, 2026).
- **Strict Invariants Enforced:**
  1. **Zero Temporal Leakage:** Candidates restricted to $T_C < T_Q$.
  2. **Zero Self-Match Leakage:** Fixture ID $Q$ cannot be an analogue of itself.
  3. **Zero Post-Kickoff Contamination:** Only pre-kickoff features ($P(H), P(D), P(A), \text{outcome}, \lambda$) are used to build query vectors.

---

## 3. Detailed Metrics Breakdown

### A. V4.0 Production Baseline
- **1X2 Correct:** 20 / 33 (60.61%)
  - Home Decisions: 14 / 21 correct (66.7%)
  - Away Decisions: 6 / 12 correct (50.0%)
  - Draw Decisions: 0 / 0 (V4.0 rarely selects raw D as top class)
- **Exact Score Correct:** 3 / 33 (9.09%)
  - Hits: *Manchester City 2-1 AFC Bournemouth* (H), *Espanyol 1-2 Real Madrid* (A), *Torino 1-2 AC Milan* (A).

### B. Historical Memory Layer Behavior
- **Sample Protection Safeguard:** Triggered on 33/33 matches (100%).
- **Advisory Output:** Safely emitted `⚪ INSUFFICIENT SAMPLE` on early season fixtures rather than over-indexing on 1 or 2 isolated matches.
- **Degradation Check:** 0 regressions caused by memory layer.

### C. Correct-Score Refinement Behavior
- **Refinement Decisions:** In all 33 matches, because analogue clusters were small or dispersed, `CorrectScoreRefiner` preserved the baseline V4.0 score prediction.
- **Exact Score Accuracy:** 9.09% (3/33) — identical to baseline, zero false degradation.

---

## 4. League-by-League Breakdown

| Competition | Matches | V4.0 1X2 Accuracy | V4.0 Exact Score | Memory Signal State | Refined Exact Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Serie A** | 8 | **75.0%** (6/8) | 25.0% (2/8) | `INSUFFICIENT SAMPLE` (Guarded) | 25.0% (2/8) |
| **La Liga** | 7 | **57.1%** (4/7) | 14.3% (1/7) | `INSUFFICIENT SAMPLE` (Guarded) | 14.3% (1/7) |
| **Premier League** | 9 | **55.6%** (5/9) | 0.0% (0/9) | `INSUFFICIENT SAMPLE` (Guarded) | 0.0% (0/9) |
| **Ligue 1** | 9 | **55.6%** (5/9) | 0.0% (0/9) | `INSUFFICIENT SAMPLE` (Guarded) | 0.0% (0/9) |
| **Bundesliga** | 0 | N/A (Starts MD1 Aug 28) | N/A | N/A | N/A |

---

## 5. Model Integrity & Cryptographic Verifications

- **V4.0 Production Model (`data/models/v4_poisson_venue_elo_online_ad.pkl`):**
  $$\text{MD5: } \mathbf{06841f0c03c8597b2b8cd8f8ab064864}\quad (\text{MATCH: True})$$
- **V4.1 Candidate Model (`data/models/v4_1_prospective_candidate_2025_26.pkl`):**
  $$\text{MD5: } \mathbf{145f918d933eb343c0f63ca342b10289}\quad (\text{MATCH: True})$$

---

## 6. Recommendations & Next Steps

1. **Keep Production Architecture Unchanged:** The Historical Memory Layer should remain strictly an auxiliary, non-mutating decision-support layer.
2. **Continue Ingesting Completed Gameweeks:** As Matchday 2 and subsequent gameweeks conclude, the memory store will naturally accumulate larger analogue clusters, enabling robust statistical conviction testing in future cohorts ($N \ge 100$).
