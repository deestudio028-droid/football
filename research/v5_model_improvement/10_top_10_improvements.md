# 10 — Top 10 Candidate Improvements: Ranked Research & Impact Matrix

## 1. Executive Summary

Based on our comprehensive literature review (14 academic papers), open-source repository analysis (8 projects), external data source audit (6 providers), and current feature inventory (91 features), we have ranked the **Top 10 High-Impact Candidate Improvements** for the $V_5$ football forecasting system.

Rankings are determined by a multi-criteria scoring function:
$$\text{Priority Score} = \frac{\text{Expected } RPS \text{ Benefit} \times \text{Draw Improvement} \times \text{Academic Robustness}}{\text{Implementation Effort} \times (1 + \text{Leakage Risk})}$$

---

## 2. Master Top 10 Ranking Table

| Rank | Candidate Improvement | Primary Method / Paradigm | Research Grounding | Expected $\Delta RPS$ | Draw Impact | Data Requirement | Effort | Leakage Risk | Priority |
|:---:|:---|:---|:---|:---:|:---:|:---|:---:|:---:|:---:|
| **1** | **Rolling Expected Goals ($xG$) & $npxG$ Engine** | Feature Engineering | Wheatcroft (2020, 2021) | **$-0.00220$** | Moderate ($+1.5\%$) | Understat / FBref (100% Free) | Med | Low (Lagged) | **P1** |
| **2** | **Vectorized Dixon-Coles Low-Score Transformation** | Score Probability Matrix | Dixon & Coles (1997) | **$-0.00180$** | Very High ($+4.0\%$) | Historical Match Results | Low | Zero | **P1** |
| **3** | **Continuous Rating Parity ($RPI$) & Intensity Gate** | Residual Draw Correction | Ley et al. (2019) / $V_{4.6}$ | **$-0.00160$** | State-of-the-Art | Internal Elo + $xG$ | Low | Zero | **P1** |
| **4** | **Dual LightGBM Poisson Tree Regressors** | Non-Linear Intensity Estimator | Groll (2019), Stübinger (2020) | **$-0.00150$** | Moderate | Tabular Feature Matrix | Low/Med | Low | **P1** |
| **5** | **Rest Delta & 14-Day Match Load Fatigue Index** | Contextual Feature Group | Baboota & Kaur (2019) | **$-0.00090$** | Low ($+0.5\%$) | Fixture Schedule DB (Free) | Low | Zero | **P2** |
| **6** | **Multi-Horizon Shin Devigged Market Odds Layer** | Calibration / Blending | Hubáček (2019), Angelini (2019) | **$-0.00350$** | High ($+3.0\%$) | Football-Data / OddAlerts | Med | High (Needs Gate) | **P2** |
| **7** | **Squad Financial Market Value Ratio ($MVR$)** | Structural Talent Anchor | Groll (2019), Stübinger (2020) | **$-0.00110$** | Low | Transfermarkt Open Scrape | Low/Med | Low | **P2** |
| **8** | **Exponential Half-Life Time Decay ($t_{1/2}=5\text{ games}$)** | Form Weighting | Dixon-Coles (1997), `regista` | **$-0.00070$** | Low | Historical Match Results | Low | Zero | **P2** |
| **9** | **Box Shot Penetration & Deep Completion Ratios** | Advanced Shot Quality | Robberechts (2020), `socceraction`| **$-0.00060$** | Moderate | Understat Open Data | Med | Low | **P3** |
| **10** | **Horizon-Gated Confirmed Starting XI Lineup Elo** | Real-Time Lineup Rating | Bunker (2019), Hubáček (2019) | **$-0.00280$** | High | Official Lineups ($T-15\text{m}$) | High | High ($T-15\text{m}$ only)| **P3** |

---

## 3. Deep Architectural Profiles for the Top 5 High-Priority Improvements

### Rank 1: Rolling Expected Goals ($xG$) & Non-Penalty $xG$ Engine
- **Why It Works**: Goals in football are notoriously rare and noisy ($\sim 2.7$ goals per game). A team can play brilliantly, generate 3.2 $xG$, hit the post three times, and lose 0-1 on a freak counter-attack. A goal-based model penalizes that team's rating, whereas an $xG$-based model recognizes their underlying dominance. Rolling $xG$ converges to true team talent in 5 matches vs. 18 matches for goal counts.
- **Concrete Deliverable**:
  - Add 6 rolling $xG$ features: `home_xg_ewma_last5`, `away_xg_ewma_last5`, `home_xga_ewma_last5`, `away_xga_ewma_last5`, `xg_diff_last5`, `npxg_diff_last10`.
- **Implementation Strategy**: Ingest historical Understat match $xG$ into `data/research/understat_xg.parquet` (2014–2026). Apply causal lagging.

---

### Rank 2: Vectorized Dixon-Coles Low-Score Transformation
- **Why It Works**: Standard Poisson models assume $P(\text{Home Goals}=x, \text{Away Goals}=y) = P(x) \times P(y)$. In reality, defensive tactical dependencies cause $0-0$ and $1-1$ scorelines to occur $\approx 15\text{–}20\%$ more frequently than independent Poisson predicts. Dixon-Coles introduces parameter $\rho \approx -0.12$, which inflates $(0,0)$ and $(1,1)$ while deflating $(1,0)$ and $(0,1)$ in an exact, probability-conserving manner.
- **Concrete Deliverable**:
  - Replace the independent score grid transformation in `v4_artifact.py` with a vectorized Dixon-Coles joint distribution grid solver:
    $$\tau(0,0) = 1 - \lambda_H \lambda_A \rho, \quad \tau(1,1) = 1 - \rho, \quad \tau(1,0) = 1 + \lambda_A \rho, \quad \tau(0,1) = 1 + \lambda_H \rho$$
- **Implementation Strategy**: Fit $\rho$ on historical training seasons ($2015\text{–}2024$) via L-BFGS-B; freeze $\rho$ for out-of-sample inference.

---

### Rank 3: Continuous Rating Parity ($RPI$) & Match Intensity Gate
- **Why It Works**: Our research in Phase 30 ($V_{4.6}$ and Historical Candidate H) proved that draws cluster heavily in matches satisfying two physical conditions: (1) Minimal rating gap ($|\Delta \text{Elo}| \le 80$), and (2) Low combined offensive expectation ($\lambda_H + \lambda_A \le 2.70$). Converting hard step-function rules into a continuous, differentiable **Rating Parity Index ($RPI$)** and **Combined Intensity Index** enables smooth probability adjustment.
- **Concrete Deliverable**:
  $$P_{V_5}(D) = P_{\text{Dixon-Coles}}(D) \times \left(1 + \gamma_{\text{parity}} \cdot RPI \cdot \exp\left(-\frac{\lambda_H + \lambda_A}{2.50}\right)\right)$$
- **Implementation Strategy**: Optimize scaling parameter $\gamma_{\text{parity}}$ on validation folds to minimize out-of-sample $RPS$.

---

### Rank 4: Dual LightGBM Poisson Tree Regressors
- **Why It Works**: Production $V_4$ uses a linear Poisson GLM, which assumes every feature has a purely linear relationship with $\ln \lambda$. In reality, football dynamics exhibit severe non-linear threshold effects (e.g. a team playing on 2 days rest after a 3,000km European trip suffers an exponential drop in defensive cohesion, not a linear one). LightGBM with a native Poisson loss function captures these multi-way feature interactions automatically.
- **Concrete Deliverable**:
  - Dual `LGBMPoissonRegressor` models trained on tabular features to output non-linear $(\lambda_H, \lambda_A)$, feeding directly into the Dixon-Coles probability matrix.
- **Implementation Strategy**: Implement cross-validated tree depth constraints (`max_depth=4, num_leaves=15, min_child_samples=50`) to prevent overfitting.

---

### Rank 5: Rest Delta & 14-Day Match Load Fatigue Index
- **Why It Works**: Modern European football schedules create severe physical imbalances during mid-week European (Champions League, Europa League) and domestic cup weeks. Teams playing 4 matches in 12 days concede +0.31 more goals in the final 30 minutes of matches compared to well-rested opponents.
- **Concrete Deliverable**:
  - `rest_days_diff`: Home rest days minus Away rest days.
  - `match_load_14d`: Number of competitive games in the preceding 14 days.
  - `short_turnaround_away`: Binary flag if Away team had $\le 3$ days rest following an away match.
- **Implementation Strategy**: Computed purely from historical calendar kickoff timestamps in `matches.db` with 100% deterministic reproducibility.
