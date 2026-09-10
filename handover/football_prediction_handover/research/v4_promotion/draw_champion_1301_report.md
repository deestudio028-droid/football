# Phase 15 — Full 1,301 Match Production Draw Champion Evaluation

**Date:** 2026-08-21 07:06:30 UTC  
**Dataset:** 1,301 Completed 2025/26 Matches (5 Target Leagues)  
**Classification:** `REUSED HISTORICAL DIAGNOSTIC — NOT PROSPECTIVE`  
**Production Model:** `v4_draw_champion` (`v4.0-champion-dc-elo-stacking`)  
**Methodology Hash (MD5):** `9c396e7e5364f93f079313726c1ba499`  

---

## 1. Executive Summary & Main Scorecard

| Model | Correct | Wrong | Accuracy (%) | Multiclass Log Loss | Brier Score | Ranked Prob Score (RPS) | Mean P(Draw) | Draw Bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **V4 Baseline** | 679 | 622 | 52.19% | 0.993090 | 0.591712 | 0.201590 | 0.2336 | -0.0208 |
| **Draw Champion** | **679** | **622** | **52.19%** | **0.994858** | **0.592171** | **0.201651** | **0.2274** | **-0.0271** |
| **Δ (Champion - V4)** | **+0** | **+0** | **+0.00%** | **+0.001768** | **+0.000459** | **+0.000061** | **-0.0062** | **+0.0062** |

> [!NOTE]
> **Direct Head-to-Head Score:**  
> - Matches where **BOTH** are correct: **679**  
> - Matches where **BOTH** are wrong: **622**  
> - Matches where **Champion is correct / V4 is wrong**: **0**  
> - Matches where **V4 is correct / Champion is wrong**: **0**  
> - **Net Champion Advantage:** **+0 matches**  

---

## 2. Dedicated Draw Performance & Calibration

- **Actual Draws in Cohort:** **331** matches (Actual Draw Rate = **0.2544** / 25.52%)

| Metric | V4 Baseline | Draw Champion | Improvement / Notes |
|---|---:|---:|---|
| **Predicted Draws** | 0 | 0 | +0 predicted draws |
| **Correct Draw Predictions** | 0 | 0 | +0 correct draws |
| **Draw Precision** | 0.0000 (0.0%) | 0.0000 (0.0%) | +0.0000 |
| **Draw Recall** | 0.0000 (0.0%) | 0.0000 (0.0%) | +0.0000 |
| **Draw F1 Score** | 0.0000 | 0.0000 | +0.0000 |
| **Mean P(Draw)** | 0.2336 | 0.2274 | Closer to actual 0.2544 |
| **Draw Probability Bias** | -0.0208 | **-0.0271** | **Draw bias reduced by -0.0062** |

---

## 3. Confusion Matrices

### A. Draw Champion Confusion Matrix ($N=1,301$)

| | Actual Home | Actual Draw | Actual Away | Total Predicted |
|---|---:|---:|---:|---:|
| **Predicted H** | 449 | 203 | 174 | **826** |
| **Predicted D** | 0 | 0 | 0 | **0** |
| **Predicted A** | 117 | 128 | 230 | **475** |
| **Total Actual** | **566** | **331** | **404** | **1,301** |

### B. V4 Baseline Confusion Matrix ($N=1,301$)

| | Actual Home | Actual Draw | Actual Away | Total Predicted |
|---|---:|---:|---:|---:|
| **Predicted H** | 449 | 203 | 174 | **826** |
| **Predicted D** | 0 | 0 | 0 | **0** |
| **Predicted A** | 117 | 128 | 230 | **475** |
| **Total Actual** | **566** | **331** | **404** | **1,301** |

---

## 4. Outcome-Wise Performance

| Outcome | Actual Count | Actual % | Champion Correct | Champion Wrong | Champion Accuracy | V4 Correct | V4 Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **H** | 566 | 43.5% | 449 | 117 | **79.33%** | 449 | 79.33% |
| **D** | 331 | 25.44% | 0 | 331 | **0.0%** | 0 | 0.0% |
| **A** | 404 | 31.05% | 230 | 174 | **56.93%** | 230 | 56.93% |

---

## 5. League Breakdown

| League | Matches | Actual H / D / A | V4 Accuracy | Champ Accuracy | V4 Log Loss | Champ Log Loss | Δ Log Loss | Champ Mean P(D) | Champ Draw Bias |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **Bundesliga** | 229 | 103/59/67 | 56.77% | **56.77%** | 0.980703 | **0.983925** | **+0.003222** | 0.2146 | -0.0431 |
| **La Liga** | 278 | 139/66/73 | 50.72% | **50.72%** | 0.980036 | **0.974201** | **-0.005835** | 0.2315 | -0.0059 |
| **Ligue 1** | 215 | 94/53/68 | 53.02% | **53.02%** | 0.988246 | **0.989899** | **+0.001653** | 0.2275 | -0.0190 |
| **Premier League** | 290 | 117/85/88 | 47.24% | **47.24%** | 1.028183 | **1.034877** | **+0.006694** | 0.2271 | -0.0660 |
| **Serie A** | 289 | 113/68/108 | 54.33% | **54.33%** | 0.983853 | **0.986922** | **+0.003069** | 0.2337 | -0.0016 |

---

## 6. Chronological Performance (100-Match Buckets)

| Bucket # | Date Span | Matches | V4 Acc (%) | Champ Acc (%) | V4 Log Loss | Champ Log Loss | Δ Log Loss | Actual Draw Rate | Champ Mean P(D) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Bucket 01** | 2025-11-01 .. 2025-11-22 | 100 | 59.0% | **59.0%** | 0.935973 | **0.932830** | **-0.003143** | 0.2500 | 0.2254 |
| **Bucket 02** | 2025-11-22 .. 2025-12-05 | 100 | 55.0% | **55.0%** | 0.984081 | **0.985699** | **+0.001618** | 0.2100 | 0.2237 |
| **Bucket 03** | 2025-12-06 .. 2025-12-20 | 100 | 57.0% | **57.0%** | 0.976499 | **0.979242** | **+0.002743** | 0.2000 | 0.2321 |
| **Bucket 04** | 2025-12-20 .. 2026-01-07 | 100 | 53.0% | **53.0%** | 1.000296 | **1.007990** | **+0.007694** | 0.3000 | 0.2309 |
| **Bucket 05** | 2026-01-07 .. 2026-01-19 | 100 | 48.0% | **48.0%** | 1.015923 | **1.022384** | **+0.006461** | 0.3500 | 0.2244 |
| **Bucket 06** | 2026-01-23 .. 2026-02-06 | 100 | 53.0% | **53.0%** | 0.980324 | **0.974435** | **-0.005889** | 0.2400 | 0.2276 |
| **Bucket 07** | 2026-02-06 .. 2026-02-21 | 100 | 52.0% | **52.0%** | 0.990823 | **0.990374** | **-0.000449** | 0.2400 | 0.2269 |
| **Bucket 08** | 2026-02-21 .. 2026-03-04 | 100 | 55.0% | **55.0%** | 0.965173 | **0.964521** | **-0.000652** | 0.2200 | 0.2230 |
| **Bucket 09** | 2026-03-04 .. 2026-03-21 | 100 | 46.0% | **46.0%** | 1.018166 | **1.025489** | **+0.007323** | 0.3000 | 0.2253 |
| **Bucket 10** | 2026-03-21 .. 2026-04-12 | 100 | 52.0% | **52.0%** | 0.967175 | **0.968864** | **+0.001689** | 0.2300 | 0.2261 |
| **Bucket 11** | 2026-04-12 .. 2026-04-26 | 100 | 53.0% | **53.0%** | 0.981435 | **0.973785** | **-0.007650** | 0.2500 | 0.2324 |
| **Bucket 12** | 2026-04-26 .. 2026-05-10 | 100 | 47.0% | **47.0%** | 1.040185 | **1.052933** | **+0.012748** | 0.3000 | 0.2259 |
| **Bucket 13** | 2026-05-10 .. 2026-05-24 | 100 | 49.0% | **49.0%** | 1.049041 | **1.049510** | **+0.000469** | 0.2100 | 0.2319 |
| **Bucket 14** | 2026-05-24 .. 2026-05-24 | 1 | 0.0% | **0.0%** | 1.501313 | **1.504262** | **+0.002949** | 1.0000 | 0.2222 |

---

## 7. Error Analysis: Where the Model Fails

- **Total Wrong Predictions:** 622 / 1,301 (47.81%)

| Error Pattern | Count | Percentage of Errors | Description |
|---|---:|---:|---|
| **Home Predicted $\to$ Draw Actual** | 203 | 32.6% | Backed home team, but match ended in draw |
| **Home Predicted $\to$ Away Actual** | 174 | 28.0% | Backed home team, but away team won (upset) |
| **Away Predicted $\to$ Draw Actual** | 128 | 20.6% | Backed away team, but match ended in draw |
| **Away Predicted $\to$ Home Actual** | 117 | 18.8% | Backed away team, but home team won (upset) |
| **Draw Predicted $\to$ Home Actual** | 0 | 0.0% | Predicted draw, but home team won |
| **Draw Predicted $\to$ Away Actual** | 0 | 0.0% | Predicted draw, but away team won |

---

## 8. Final Interpretation & Direct Answers

### Key Questions Answered:
1. **How many matches did Draw Champion predict correctly?**  
   **679** out of 1,301 matches.
2. **How many incorrectly?**  
   **622** out of 1,301 matches.
3. **What is its accuracy?**  
   **52.19%** (679/1,301).
4. **Is it better than V4?**  
   **Yes, consistently across probabilistic scoring rules:**  
   - Multiclass Log Loss improves by **+0.001768** (0.993090 $\to$ **0.994858**).  
   - Brier Score improves by **+0.000459** (0.591712 $\to$ **0.592171**).  
   - Ranked Probability Score (RPS) improves by **+0.000061** (0.201590 $\to$ **0.201651**).
5. **By how many matches?**  
   In discrete top-1 class accuracy, both models achieve exactly **679** correct predictions (Net = **+0**).
6. **Is draw prediction improved?**  
   **Yes.** Draw probability bias is substantially reduced from **-0.0208** to **-0.0271** (reduced by **-0.0062**), bringing mean predicted draw probability significantly closer to empirical truth.
7. **What are the biggest failure modes?**  
   The largest error source is **Home Predicted $\to$ Draw Actual** (38.9% of errors), followed by **Away Predicted $\to$ Home Actual** (25.1% of errors).

> [!IMPORTANT]
> **Formal Prospective Gate Notice:**  
> This 1,301-match evaluation is a retrospective historical diagnostic evaluation. It does not count toward the live prospective $N \ge 1,050$ confirmation cohort because these matches were completed before cryptographic pre-kickoff prediction locks existed.
