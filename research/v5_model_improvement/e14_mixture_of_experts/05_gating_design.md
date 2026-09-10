# 05 — Gating Mechanism Architectural Design

## 1. Gating Network Architecture

```mermaid
graph TD
    subgraph "Causal Context Input"
        A[Pre-Match Regime Features] --> B{Regime Detector}
    end
    subgraph "Forecasting Experts"
        C[Expert 1: Pure E10 Anchor]
        D[Expert 2: E13 Dynamic Specialist]
    end
    subgraph "Gating Mechanism"
        B -->|Season Match <= 3| E[Early-Season Specialist Gate]
        B -->|Disagreement > tau| F[Uncertainty Blending Gate]
        E --> G[Gating Weights w_E10, w_E13]
        F --> G
    end
    subgraph "Simplex Blending Engine"
        C --> H[Convex Combination Engine]
        D --> H
        G --> H
        H --> I[Blended Probabilities P H, D, A]
    end
```

---

## 2. Mathematical Formulation

$$w_1(x) = \begin{cases} 0.30 & \text{if } \text{season\_match\_num} \le 3 \\ 1.00 & \text{otherwise} \end{cases}, \quad w_2(x) = 1.0 - w_1(x)$$

$$P_{\text{final}}(x) = w_1(x) P_{\text{E10}}(x) + w_2(x) P_{\text{E13}}(x)$$
