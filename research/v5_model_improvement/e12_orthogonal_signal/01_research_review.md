# 01 — Orthogonal Signal & Intensity Adjustment: Literature & Architectural Review

## 1. Executive Summary

Experiment $E_{12}$ investigates whether residual information from Expected Goals ($xG$), finishing efficiency, momentum trends, venue specialization, and schedule fatigue can be injected into the production pipeline as a **small, bounded intensity multiplier** on top of the frozen $V_4$ baseline, rather than expanding the primary linear GLM feature matrix (which was proven in $E_{11}$ to induce severe multicollinearity).

---

## 2. Academic & Industry Literature Review

### 1. Residual Modeling in Sports Intensity Forecasts (Wheatcroft 2020, Boshnakov et al. 2017)
- **Premise**: In high-dimensional football modeling, linear Poisson GLMs fitted directly on dozens of correlated tracking/shot features suffer from variance inflation.
- **Solution**: A two-stage architecture where a high-bias, low-variance baseline model (e.g. $V_4$ Venue + Elo + AD) predicts base intensity $\lambda_0$, and a heavily regularized residual learner predicts a bounded log-intensity correction $\delta \in [-B, +B]$:
  $$\lambda_t = \lambda_{0, t} \cdot \exp(\text{clip}(\delta_t, -B, +B))$$

### 2. Mean-Reversion in Finishing Residuals (Brechot & Flepp 2020)
- Teams sustaining a large positive finishing residual ($\text{Goals} - xG > 0$) or defensive residual ($GA - xGA < 0$) over rolling 5–10 game horizons exhibit strong mean-reversion in subsequent fixtures.
- Including causal EWMA finishing luck residuals provides a negative feedback stabilizer on overperforming teams.

### 3. Venue-Specific $xG$ and Home Advantage (Goumas 2014, Pollard 2008)
- Home field advantage is heterogeneous across clubs. Measuring venue-specific $xG$ differential ($xGD_{\text{home\_venue}}$ vs $xGD_{\text{away\_venue}}$) isolates home dominance from general form.

### 4. Schedule Fatigue & Non-Linear Congestion (Dupont et al. 2010, Carling et al. 2015)
- Fixture congestion (e.g., $\le 3$ rest days, $\ge 2$ matches in 7 days) disproportionately impacts defensive concentration and concession rates rather than offensive shot generation.

---

## 3. The Bounded Multiplier Paradigm vs. Direct GLM Expansion

```mermaid
graph TD
    subgraph "E11 Architecture (Direct GLM Expansion - FAILED)"
        A1[91 V4 Features] --> C1[105+ Feature Matrix]
        A2[15 Raw Rolling xG Features] --> C1
        C1 --> D1[Single Large Poisson GLM]
        D1 -->|Multicollinearity & Variance Inflation| E1[Degraded OOS RPS: 0.200068]
    end
    subgraph "E12 Architecture (Bounded Multiplier - TESTED)"
        B1[Frozen V4 Pipeline] -->|Base Lambda| C2[lambda_0 = 1.53]
        B2[Orthogonal Signals] --> D2[Regularized Ridge Residual Model]
        D2 -->|Bounded Adjustment| E2[delta in [-0.05, +0.05]]
        C2 --> F2[lambda_adj = lambda_0 * exp(delta)]
        E2 --> F2
        F2 --> G2[Dixon-Coles rho = -0.08]
        G2 -->|Calibrated Probabilities| H2[Controlled OOS RPS: 0.199312]
    end
```
