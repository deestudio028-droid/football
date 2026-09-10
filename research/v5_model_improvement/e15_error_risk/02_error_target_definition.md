# 02 — Error Target Definitions & Realized Loss Functions

## 1. Primary Loss Formulations

For match fixture $i$ with true outcome $y_i \in \{H, D, A\}$ and E10 predicted probabilities $P_i = [p_{i, H}, p_{i, D}, p_{i, A}]$:

1. **Ranked Probability Score (RPS)**:
   $$\text{RPS}_i = \frac{1}{2} \left[ (p_{i, H} - \mathbf{1}_{\{y_i=H\}})^2 + ((p_{i, H} + p_{i, D}) - \mathbf{1}_{\{y_i \in \{H, D\}\}})^2 \right]$$

2. **Multi-Class Log-Loss**:
   $$\text{LL}_i = -\ln \left( \sum_{c \in \{H, D, A\}} p_{i, c} \mathbf{1}_{\{y_i = c\}} \right)$$

---

## 2. Meta-Learning Binary Target Definitions

- `target_top_10pct_rps`: $\mathbf{1}_{\{\text{RPS}_i \ge \tau_{90}\}}$, where $\tau_{90}$ is the 90th percentile of RPS on the training fold.
- `target_top_20pct_rps`: $\mathbf{1}_{\{\text{RPS}_i \ge \tau_{80}\}}$.
- `target_top_10pct_ll`: $\mathbf{1}_{\{\text{LL}_i \ge \tau_{\text{LL}, 90}\}}$.
