# 02 — Reliability Target Definitions & Mathematical Formulations

## 1. Continuous Probabilistic Loss Targets

For match fixture $i$ with true outcome $y_i \in \{H, D, A\}$ and E10 predicted probabilities $P_i = [p_{i, H}, p_{i, D}, p_{i, A}]$:

1. **Ranked Probability Score (RPS)**:
   $$\text{RPS}_i = \frac{1}{2} \left[ (p_{i, H} - \mathbf{1}_{\{y_i=H\}})^2 + ((p_{i, H} + p_{i, D}) - \mathbf{1}_{\{y_i \in \{H, D\}\}})^2 \right]$$

2. **Multiclass Brier Score**:
   $$\text{Brier}_i = \sum_{c \in \{H, D, A\}} (p_{i, c} - \mathbf{1}_{\{y_i = c\}})^2$$

3. **Multi-Class Log-Loss**:
   $$\text{LL}_i = -\ln \left( \sum_{c \in \{H, D, A\}} p_{i, c} \mathbf{1}_{\{y_i = c\}} \right)$$

4. **Top-1 Correctness**:
   $$\text{Top1Correct}_i = \mathbf{1}_{\{\arg\max_c p_{i, c} = y_i\}}$$

5. **Continuous Reliability Score**:
   $$\text{Reliability}_i = \max\left(0.0, 1.0 - \frac{\hat{\text{RPS}}_i}{0.50}\right)$$
