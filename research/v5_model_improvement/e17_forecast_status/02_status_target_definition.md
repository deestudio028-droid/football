# 02 — Status Target Definitions & Mathematical Formulations

## 1. Latent Continuous Target: Expected Match Loss

For match fixture $i$ with pre-match feature context $x_i$:
$$\hat{L}(x_i) = \mathbb{E}[\text{RPS}_i \mid x_i]$$
where realized $\text{RPS}_i$ is computed from true outcome $y_i \in \{H, D, A\}$ and E10 predicted probabilities $P_i = [p_{i, H}, p_{i, D}, p_{i, A}]$:
$$\text{RPS}_i = \frac{1}{2} \left[ (p_{i, H} - \mathbf{1}_{\{y_i=H\}})^2 + ((p_{i, H} + p_{i, D}) - \mathbf{1}_{\{y_i \in \{H, D\}\}})^2 \right]$$

---

## 2. Continuous Calibrated Reliability Score

$$\text{Score}_i = \text{IsotonicTransform}\left(\max\left(0.0, 1.0 - \frac{\hat{L}(x_i)}{0.50}\right)\right) \in [0, 1]$$

---

## 3. Discrete 4-Tier Decision Taxonomy

Let $\tau_{75}, \tau_{50}, \tau_{25}$ be the 75th, 50th, and 25th percentiles of $\text{Score}$ on the training fold:
$$\text{Status}(x_i) = \begin{cases}
\text{STRONG} & \text{if } \text{Score}_i \ge \tau_{75} \\
\text{LEAN} & \text{if } \tau_{50} \le \text{Score}_i < \tau_{75} \\
\text{CAUTION} & \text{if } \tau_{25} \le \text{Score}_i < \tau_{50} \\
\text{AVOID} & \text{if } \text{Score}_i < \tau_{25}
\end{cases}$$
