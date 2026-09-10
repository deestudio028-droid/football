# 01 — Regime-Aware Mixture-of-Experts: Literature Review

## 1. Executive Summary

This literature review investigates regime-switching models, mixture-of-experts (MoE) architectures, dynamic forecast combination, and uncertainty-aware gating mechanisms in sports forecasting and probabilistic classification.

---

## 2. Comprehensive Methodological Audit

### 1. Hierarchical Mixture of Experts (Jordan & Jacobs 1994)
- **Methodology**: Divides input space into soft regimes using a parametric gating network:
  $$P(y \mid x) = \sum_{j=1}^K g_j(x; v) P_j(y \mid x; \theta_j)$$
  where $g_j(x; v) = \frac{\exp(v_j^T x)}{\sum_m \exp(v_m^T x)}$ is a softmax gating function.
- **Relevance**: Enables localized specialization where expert models focus on disjoint pre-match contexts (e.g., early-season vs established season).
- **Risk**: Overfitting gating parameters on small sample regimes.

### 2. Regime-Switching & Dynamic Model Averaging (Raftery et al. 2010)
- **Methodology**: Dynamically updates model probabilities over time based on recent predictive performance:
  $$\pi_{t \mid t-1, k} = \frac{\pi_{t-1 \mid t-1, k}^{\alpha}}{\sum_l \pi_{t-1 \mid t-1, l}^{\alpha}}$$
- **Finding**: Fast forgetting factors ($\alpha < 0.95$) induce excessive variance in low-scoring sports like association football.

### 3. Stacking for Probabilistic Classification (Smyth & Wolpert 1999)
- **Methodology**: Supervised learning of convex combination weights $w \in \Delta^{K-1}$ optimizing strictly proper scoring rules (Log-Loss or Ranked Probability Score):
  $$\min_w \sum_{i=1}^N \text{RPS}\left(\sum_{j=1}^K w_j P_j(x_i), y_i\right) \quad \text{s.t.} \quad w_j \ge 0, \sum w_j = 1$$
- **Finding**: Yields well-calibrated ensemble probabilities without destroying the marginal calibration of well-fitted constituent models.

### 4. Model Disagreement & Selective Gating (Lakshminarayanan et al. 2017)
- **Methodology**: Uses Jensen-Shannon divergence $D_{\text{JS}}(P_1 \parallel P_2)$ as an epistemological uncertainty signal to trigger conservative fallback or specialized blending.
