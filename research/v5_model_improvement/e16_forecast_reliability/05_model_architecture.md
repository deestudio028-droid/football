# 05 — Reliability Architecture & Confidence Engine Design

## 1. System Architecture

```mermaid
graph TD
    subgraph "Base Forecaster"
        A[Pre-Match Features] --> B[V4 Intensity GLMs]
        B --> C[E10 Dixon-Coles Probability Simplex]
    end
    subgraph "Reliability Layer"
        C --> D[Prediction Geometry Extractor]
        A --> E[Strength & Maturity Extractor]
        D --> F[HistGradientBoosting Expected RPS Regressor]
        E --> F
        F --> G[Isotonic Calibrator]
        G --> H[Continuous Reliability Score in 0, 1]
    end
    subgraph "Forecast Metadata Output"
        H --> I{Walk-Forward Confidence Classifier}
        I -->|Score >= tau_high| J["HIGH CONFIDENCE (30%)"]
        I -->|tau_low <= Score < tau_high| K["MODERATE CONFIDENCE (40%)"]
        I -->|Score < tau_low| L["LOW CONFIDENCE (30%)"]
    end
```

---

## 2. Confidence Bands

- **HIGH CONFIDENCE**: $N=2,136$ matches ($30.16\%$), Mean Reliability = $0.7247$, Mean RPS = $\mathbf{0.158512}$, Top-1 Accuracy = $\mathbf{66.85\%}$.
- **MODERATE CONFIDENCE**: $N=2,870$ matches ($40.53\%$), Mean Reliability = $0.5751$, Mean RPS = $\mathbf{0.211614}$, Top-1 Accuracy = $\mathbf{48.75\%}$.
- **LOW CONFIDENCE**: $N=2,076$ matches ($29.31\%$), Mean Reliability = $0.5194$, Mean RPS = $\mathbf{0.224887}$, Top-1 Accuracy = $\mathbf{45.04\%}$.
