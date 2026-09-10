# 02 — Expert Models Inventory

## 1. Registry of Forecasting Experts

| Expert ID | Model Name | Architecture | Strength / Target Regime | Default Status |
|:---|:---|:---|:---|:---|
| **Expert 1** | **Pure $E_{10}$** | Frozen $V_4$ Poisson GLMs + Dixon-Coles ($\rho = -0.08$) | Established season matches ($N > 5$), high-parity games, low-score joint calibration | **Default Global Anchor** |
| **Expert 2** | **$E_{13}$ Dynamic Specialist** | Adaptive-K Elo + State-Space AD + Hierarchical Empirical Bayes Shrinkage | Early season matches (matchweeks 1–3), promoted & sparse history teams ($n < 5$) | **Early-Season Specialist** |
| **Expert 3** | **Conservative Baseline** | Static league-wide historical rates + empirical prior | Fallback for extreme outlier contexts | Fallback Candidate |

---

## 2. Mathematical Contracts

For any match fixture $i$:
$$P_{\text{final}}(x_i) = \sum_{j=1}^K w_j(x_i) P_j(x_i), \quad w_j(x_i) \ge 0, \quad \sum_{j=1}^K w_j(x_i) = 1.0$$
where $P_j(x_i) \in \Delta^2$ is the 3-class probability simplex for Home, Draw, Away.
