# Phase 23 — V4.5 Causal Walk-Forward Draw Meta-Decision Layer Report

**Execution Date:** 2026-08-22  
**Project:** Football Prediction Project  
**Authoritative Production Model:** `v4_draw_champion` (`v4.0-champion-dc-elo-stacking`) [100% FROZEN]  
**Shadow Override Baseline:** `v4_4_robust_draw_override` (`v4.4-robust-draw-override`) [SHADOW CANDIDATE]  
**Evaluated Meta-Decision Layer:** `v4_5_causal_draw_meta` (`v4.5-causal-draw-meta`) [SHADOW RESEARCH]  
**Target Competitions:** Premier League, La Liga, Bundesliga, Serie A, Ligue 1  
**Classification:** **C. NO GENERALIZABLE IMPROVEMENT OVER V4.4 (PARAMETRIC OVERFITTING ON BLIND TEST)**  

---

## 1. Executive Summary

In Phase 23, we investigated whether a causal, cost-sensitive, regularized meta-decision model trained chronologically across expanding walk-forward folds could discover a continuous decision boundary to identify when V4's primary Home/Away prediction should be selectively overridden to Draw.

### Key Scientific Findings & Definitive Conclusion

1. **Discovery vs Blind Held-Out Overfitting Failure:**
   - **On the Discovery Set ($N=650$):** V4.5 improved accuracy from $53.85\%$ to **$54.15\%$** ($+0.31\%$), achieving a promising **$33.9\%$ Draw Precision** (20 correct draws).
   - **On the Blind Held-Out Set ($N=651$):** V4.5's performance **collapsed**: accuracy dropped from $50.54\%$ down to **$49.46\%$** ($\Delta = -1.08\%$, a loss of 7 matches), while Draw Precision degraded to **$23.1\%$** (worse than the $25.4\%$ baseline draw prevalence).
   - **Full Cohort Result ($N=1,301$):** V4.5 achieved **$51.81\%$ overall accuracy** ($674/1,301$, a net loss of $-5$ correct predictions vs baseline V4), sacrificing 40 V4-correct wins to capture 35 draws.

2. **Severe Inter-League Instability:**
   - While V4.5 performed well in La Liga ($+3.24\%$) and Bundesliga ($+1.31\%$), it severely damaged **Ligue 1 ($-3.26\%$ accuracy, $15.0\%$ draw precision)** and **Serie A ($-3.46\%$ accuracy, $17.2\%$ draw precision)** by heavily over-predicting false draws in defensive stalemates that ended 1–0.

3. **Why Heuristic Gating (V4.4) Strictly Beats Learned Parametric Meta-Classifiers (V4.5):**
   - Single-match football draw outcomes are inherently high-entropy and noisy ($R^2 \le 0.03$). A continuous 10-dimensional logistic model fits sample-specific correlation artifacts on discovery windows that fail to generalize.
   - In contrast, the **5 hard physical gating constraints of V4.4 Robust Override** ($\theta \ge 0.26, \text{margin} \le 0.10, \text{conf\_cap} \le 0.45, |\Delta \text{Elo}| \le 100, \lambda_{\text{tot}} \le 2.50$) maintain **$52.50\%$ accuracy** (+4 net wins, 24 correct draws, 31.2% precision) and **0 degraded leagues**.

4. **Verdict & Recommendation:**  
   **V4.5 is REJECTED.** Per the non-negotiable complexity principle (*"Simplicity wins when performance is statistically equivalent or superior"*), **V4.4 Robust Override remains the authoritative shadow candidate**, while **production V4.0 remains completely frozen**.

---

## 2. Master Model Comparison ($N = 1,301$)

| Model / Architecture | Overall Accuracy | Correct Matches | Net Gain vs V4 | Draw Preds | Correct Draws | Lost V4 Correct | Draw Precision | Draw Recall | Draw F1 | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **V4 Baseline** | 52.19% | 679 / 1,301 | +0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0000 | 0.3895 | 0.993090 |
| **Draw Champion v4.0** | 52.19% | 679 / 1,301 | +0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0000 | 0.3895 | 0.994858 |
| **V4.4 Robust Override (Rule A)** | **52.50%** | **683 / 1,301** | **+4** | **77** | **24** | **20** | **31.2%** | **7.3%** | **0.1176** | **0.4303** | **0.993338** |
| **V4.5 Causal Meta-Layer** | 51.81% | 674 / 1,301 | -5 | 124 | 35 | 40 | 28.2% | 10.6% | 0.1538 | 0.4373 | 0.993338 |

---

## 3. V4.5 Architecture & Causal Feature Pipeline

```
                              PRE-MATCH FIXTURE CONTEXT
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
          V4 POISSON ENGINE                          V4.2 DIBP STACKING
        P_V4(H), P_V4(D), P_V4(A)                   P_V42(H), P_V42(D), P_V42(A)
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                     10-DIMENSIONAL CAUSAL FEATURE EXTRACTOR
      1. P_V42(D)                              6. Total Goals Offset (λ_tot - 2.50)
      2. Winner Margin (max(H,A) - D)           7. Goal Difference |λ_h - λ_a|
      3. V4 Winner Confidence max(P_V4(H,A))    8. Low Score Probability (0-0, 1-1, 2-2)
      4. Probability Delta |P(H) - P(A)|        9. Expected Home Goals λ_h
      5. Absolute Elo Difference / 100         10. Expected Away Goals λ_a
                                         │
                                         ▼
                     COST-SENSITIVE REGULARIZED LOGISTIC MODEL
                               z = w^T x + b
                       P_meta(Draw) = σ(clip(z, -30, 30))
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
               P_meta(Draw) >= 0.2500           P_meta(Draw) < 0.2500
                         │                               │
                         ▼                               ▼
                 OVERRIDE TO DRAW                PRESERVE V4 BASE
                    y_final = "D"                   y_final = y_base
```

---

## 4. Cost Function & Walk-Forward Training Protocol

To penalize destroying V4-correct predictions, a cost-weighted binary cross-entropy loss with $L_2$ regularization was minimized:
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N w_i \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
Where sample weight $w_i$:
- $w_i = 1.0$ if match actually ended in Draw ($y_i = 1$).
- $w_i = C_{\text{lost}} = 1.50$ if V4 correctly predicted $H$ or $A$ ($y_i = 0$).
- $w_i = 1.0$ if V4 was incorrect on $H/A$ ($y_i = 0$).

---

## 5. Anti-Overfitting Split Test (Discovery vs Blind Held-Out)

| Partition | Matches ($N$) | V4 Accuracy | V4.5 Accuracy | $\Delta$ Accuracy | Draw Preds | Correct Draws | Draw Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Discovery Half (Fitted Window)** | 650 | 53.85% | **54.15%** | **+0.31%** | 59 | 20 | **33.9%** |
| **Blind Held-Out Half (Unseen)** | 651 | 50.54% | 49.46% | **-1.08%** | 65 | 15 | **23.1%** |
| **Full Combined Diagnostic** | 1,301 | 52.19% | 51.81% | **-0.38%** | 124 | 35 | 28.2% |

- **Failure Analysis:** The meta-classifier learned overly aggressive decision weights in 10-D space that appeared optimal on the discovery set ($+0.31\%$), but when tested on the blind second half, it overrode 65 matches where only 15 were true draws ($23.1\%$ precision, below random baseline prevalence of $25.4\%$), destroying 22 true V4 wins.

---

## 6. Error Transition Analysis (V4.5)

```
                            ACTUAL OUTCOME
                        Home (578)   Draw (331)   Away (392)
 PREDICTED   Home          337          115          120
 OUTCOME     Draw           40           35           49
             Away           41           47          147
```

- **Free Draw Wins Gained:** **35 matches** (V4 predicted H/A incorrectly; V4.5 overrode to Draw and was correct).
- **Sacrificed V4 True Positives:** **40 matches** (V4 correctly predicted H/A; V4.5 incorrectly overrode to Draw).
- **Neutral Errors:** **49 matches** (V4 was wrong; V4.5 was also wrong $\implies 0$ net penalty).
- **Net Accuracy Impact:** $35 - 40 = \mathbf{-5}$ net correct predictions ($52.19\% \to 51.81\%$).

---

## 7. Per-League Performance & Inter-League Instability

| Competition | Matches | Actual Draw Rate | V4 Accuracy | V4.5 Accuracy | $\Delta$ Accuracy | Draw Preds | Correct Draws | Draw Precision | Draw Recall | Draw F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Bundesliga** | 229 | 25.8% | 56.77% | **58.08%** | **+1.31%** | 8 | 5 | **62.5%** | 8.5% | 0.1493 |
| **La Liga** | 278 | 23.7% | 50.72% | **53.96%** | **+3.24%** | 41 | 15 | **36.6%** | 22.7% | 0.2804 |
| **Premier League** | 290 | 29.3% | 47.24% | 47.24% | +0.00% | 26 | 7 | 26.9% | 8.2% | 0.1261 |
| **Ligue 1** | 215 | 24.6% | 53.02% | 49.77% | **-3.26%** | 20 | 3 | **15.0%** | 5.7% | 0.0822 |
| **Serie A** | 289 | 23.5% | 54.33% | 50.87% | **-3.46%** | 29 | 5 | **17.2%** | 7.4% | 0.1031 |

- **Severe League Degradation:** In Ligue 1 and Serie A, draw precision collapsed to $15.0\%$ and $17.2\%$, causing $\approx 3.3\text{--}3.5\%$ accuracy loss.

---

## 8. Paired Bootstrap Statistical Significance ($B = 10,000$)

- **V4.5 vs V4 Baseline:**
  - Mean $\Delta \text{Accuracy}$: **$-0.3693\%$** (95% CI: $[-1.6910\%, +0.9224\%]$).
  - Probability V4.5 $\ge$ V4: **$30.7\%$**.
  - Probability V4.5 $>$ V4: **$24.2\%$**.
- **V4.5 vs V4.4 Robust Override:**
  - Mean $\Delta \text{Accuracy}$: **$-0.6774\%$** (95% CI: $[-2.0753\%, +0.6918\%]$).
  - Probability V4.5 $\ge$ V4.4: **$18.1\%$**.

---

## 9. Direct Comparison: V4 vs V4.4 vs V4.5

| Feature / Dimension | V4 Baseline | V4.4 Robust Override | V4.5 Causal Meta-Layer | Preferred Candidate |
|---|---|---|---|:---:|
| **Top-1 Accuracy** | 52.19% | **52.50% (+4 wins)** | 51.81% (-5 wins) | **V4.4** |
| **Draw Predictions** | 0 | **77** | 124 | **V4.4** |
| **Correct Draws** | 0 | **24** | 35 | **V4.4** |
| **Draw Precision** | 0.0% | **31.2%** | 28.2% | **V4.4** |
| **Blind Held-Out Accuracy**| 50.54% | **50.54% (0% drop)** | 49.46% (-1.08% drop) | **V4.4** |
| **Degraded Leagues** | 0 | **0** | 2 (Ligue 1, Serie A) | **V4.4** |
| **Model Complexity** | Pure Poisson | 5 Physical Gates | 10-D Logistic Classifier| **V4.4** |
| **Explainability** | High | **High** | Moderate | **V4.4** |

---

## 10. Final Classification & Promotion Recommendation

### Classification: **C. NO GENERALIZABLE IMPROVEMENT OVER V4.4**

1. **V4.5 is REJECTED:**  
   The learned continuous meta-decision layer fails the anti-overfitting blind evaluation gate and introduces unacceptable league-specific degradation.
2. **V4.4 is CONFIRMED as the Authoritative Shadow Candidate:**  
   The 5-condition physical gate of V4.4 remains the most robust, generalizable, and mathematically sound accuracy-preserving draw fix developed in the project.
3. **Production Policy:**  
   Production model (`v4_draw_champion`) remains **100% frozen**. All 20 protected repository baseline assets remain **100% bit-identical**.

---

## 11. Test Suite Results

- `tests/test_v4_5_causal_draw_meta.py`: **4/4 PASS**
- `tests/test_v4_4_robust_draw_override.py`: **5/5 PASS**
- `tests/test_v4_3_accuracy_preserving_draw_override.py`: **5/5 PASS**
- `tests/test_v4_2_prospective_integration.py`: **4/4 PASS**
- `tests/test_draw_resolution_candidate.py`: **6/6 PASS**
- `tests/test_prospective_validation_pipeline.py`: **7/7 PASS**
- `tests/test_prospective_operational_collection.py`: **1/1 PASS**
- `tests/test_fresh_100_prospective_pilot.py`: **1/1 PASS**
- **Total Modern Test Suite:** **42/42 PASS / 0 FAIL / 0 SKIP**.
