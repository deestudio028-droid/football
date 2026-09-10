# Dixon–Coles Draw Research & Mathematical Audit (Phase 1)

**Date:** 2026-08-21  
**Author:** Antigravity (Advanced Agentic Coding)  
**Status:** Strictly Observational Research & Mathematical Audit  
**Scope:** V2 / V3 / V4 Draw Modeling Investigation  

---

## 1. Executive Summary

In the recently completed **Fresh-Extended-300 Out-of-Sample (OOS)** validation, the current candidate model (V4) achieved an accuracy of 0.5000 and log loss of 0.992706, while the Pinnacle closing market reference achieved an accuracy of 0.5067 and log loss of 0.982167. Notably, both V4 and the Pinnacle reference exhibited **zero predicted draws** under discrete $\text{argmax}$ selection ($\text{Draw Recall} = 0.0000$), despite 82 of the 300 fixtures (27.33%) ending in draws.

Our investigation demonstrates that:
1. **$\text{Draw Recall} = 0.0000$ under $\text{argmax}$ is a natural structural artifact of unimodal 3-way distribution modes**, where $P(\text{Draw})$ rarely exceeds $\max(P(\text{Home}), P(\text{Away}))$.
2. **However, independent Poisson models suffer from a genuine structural draw probability deficit**:
   - V4 mean predicted $P(\text{Draw})$ on the 300 OOS fixtures was **$0.2352$**, compared to the Pinnacle market mean of **$0.2555$** and the empirical sample frequency of **$0.2733$** (a **$-3.81\%$** raw deficit).
   - In historical training data ($n=8,983$, seasons 2020/21–2024/25), independent Poisson underestimates $0\text{--}0$ scores ($+0.42\%$ deficit) and $1\text{--}1$ scores ($+0.61\%$ deficit), while overestimating $1\text{--}0$ and $0\text{--}1$ scores.
3. The **Dixon–Coles (1997) low-score dependence correction** ($\tau(x,y)$ with $\rho < 0$) mathematically resolves this structural flaw by boosting $(0,0)$ and $(1,1)$ joint probabilities while adjusting $(1,0)$ and $(0,1)$, exactly preserving the Poisson marginal rate distributions.
4. Profiling on the historical V4 training set ($n=8,983$, pre-2025/26) identifies an optimal $\rho^* = -0.1000$. Applying this fixed training parameter in a counterfactual evaluation on the 300 OOS sample improves V4 log loss from **$0.992706$** to **$0.988780$** ($\Delta = -0.003926$), increases mean $P(\text{Draw})$ to **$0.2574$** (matching market), and reduces the gap to the market benchmark by $37.2\%$.
5. However, significant **league-to-league heterogeneity** in $\rho$ exists across domestic leagues (from $\rho = -0.1138$ in Bundesliga to $\rho = +0.0152$ in the Premier League), indicating that a single global constant $\rho$ carries misspecification risks.

**Verdict:** **B. PROMISING BUT NEEDS FURTHER RESEARCH.**

---

## 2. Sources & Repositories Investigated

We surveyed the primary academic literature and four notable open-source implementations to evaluate their mathematical rigor, parameterization, and safety constraints:

| Implementation / Source | Mathematical Formulation | Optimization / Estimation | Time Decay | Likelihood & Numerical Safeguards | Known Shortcuts / Limitations |
|---|---|---|---|---|---|
| **1. Dixon & Coles (1997)**<br>*(Applied Statistics, 46(2), 265–280)* | Full joint MLE: $\log \lambda_{ij} = c + \alpha_i - \beta_j + \gamma$, $\log \mu_{ij} = c + \alpha_j - \beta_i$. Corrects $(0,0),(1,0),(0,1),(1,1)$ by $\tau(x,y)$. | Joint non-linear MLE with sum constraint $\sum \alpha_i = n$. | Continuous exponential weight $w(t) = e^{-\xi(T - t)}$. | Full bivariate log-likelihood with score adjustments. | Joint optimization is computationally demanding and assumes stationary base ratings unless time decay is tuned. |
| **2. `worldcup_repo` Internal Model**<br>*(research/worldcup_repo/src/dixon_coles.py)* | Joint MLE fitting team attack ($\alpha$), defense ($\beta$), home advantage ($\gamma$), constant ($c_0$), and dependence ($\rho$). | SciPy `minimize(L-BFGS-B)` with $L_2$ regularization on attack/defense parameters. | Exponential daily decay ($\xi = 0.0018$, half-life $\approx 1$ yr). | Clamped $\tau \ge 1\text{e}{-9}$, log-gamma factorial terms, dynamic over/under and BTTS derivation. | Assumes fixed 10-goal matrix truncation; requires continuous time timestamps. |
| **3. `penaltyblog` Python Package**<br>*(penaltyblog/models/dixon_coles.py)* | Classical Dixon–Coles joint estimation of attack, defense, home advantage, and rho. | SciPy Nelder-Mead / BFGS solver. | Optional exponential time decay. | Unconstrained optimization with constraint penalization. | Fixed grid sizing (often $K=10$ hardcoded); slow convergence on large multi-season datasets ($n > 5,000$). |
| **4. `regista` R Package (Torvaney)**<br>*(torvaney/regista)* | Extensible bivariate Poisson / Dixon–Coles model with formula interface. | Maximum likelihood via `stats::optim` (BFGS / Nelder-Mead). | Optional weight decay vector. | Penalized likelihood for identifiability constraint ($\sum \alpha = 0$). | Designed primarily as a standalone match model rather than a modular layer atop rich feature extractors. |
| **5. Modular E3 Architecture**<br>*(research/dixon_coles/dixon_coles.py)* | **Decoupled Architecture:** Takes pre-computed exogenous rates $\lambda_H, \lambda_A$ (e.g. from V3/V4) and applies isolated $\tau(x,y)$ score grid correction. | Profile log-loss search / 1D scalar optimization over $\rho$ grid on historical training data. | Inherits dynamic feature adaptation from upstream models (Elo, Online A/D). | Adaptive grid sizing ($K$ derived from tail tolerance $1\text{e}{-15}$), analytical validity bounds $\max(-1/\lambda_H, -1/\lambda_A) < \rho < \min(1, 1/(\lambda_H \lambda_A))$. | Does not backpropagate $\rho$ gradients into the linear Poisson regressors. |

---

## 3. Original Dixon–Coles Mathematics

In standard independent Poisson modeling, the number of goals scored by the home team ($X$) and the away team ($Y$) are modeled as independent Poisson random variables with rates $\lambda$ and $\mu$:
$$X \sim \text{Poisson}(\lambda), \quad Y \sim \text{Poisson}(\mu)$$
The joint probability mass function (PMF) factorizes as:
$$P(X=x, Y=y) = \frac{\lambda^x e^{-\lambda}}{x!} \cdot \frac{\mu^y e^{-\mu}}{y!}$$

Dixon & Coles (1997) observed that in actual football matches, low-scoring outcomes (especially $0\text{--}0$ and $1\text{--}1$) occur substantially more often than predicted by independence, while scores of $1\text{--}0$ and $0\text{--}1$ occur less often. 

To model this low-score dependence without abandoning the analytical tractability of Poisson marginals, Dixon and Coles introduced a modified joint probability mass function:
$$P_{DC}(X=x, Y=y) = P(X=x, Y=y) \cdot \tau_{\lambda, \mu}(x, y)$$

where $\tau_{\lambda, \mu}(x, y)$ is a bivariate adjustment function:
$$\tau_{\lambda, \mu}(x, y) = \begin{cases}
1 - \lambda \mu \rho & \text{for } (x,y) = (0,0) \\
1 + \mu \rho & \text{for } (x,y) = (1,0) \\
1 + \lambda \rho & \text{for } (x,y) = (0,1) \\
1 - \rho & \text{for } (x,y) = (1,1) \\
1 & \text{for all } x \ge 2 \text{ or } y \ge 2
\end{cases}$$

---

## 4. Tau/Rho Correction Derivation & Marginal Invariance

A crucial mathematical property of the Dixon–Coles adjustment is **marginal distribution preservation**. The correction is specifically constructed so that the marginal distributions of $X$ and $Y$ remain strictly Poisson with rates $\lambda$ and $\mu$:
$$\sum_{y=0}^{\infty} P_{DC}(X=x, Y=y) = P(X=x) = \frac{\lambda^x e^{-\lambda}}{x!}$$
$$\sum_{x=0}^{\infty} P_{DC}(X=x, Y=y) = P(Y=y) = \frac{\mu^y e^{-\mu}}{y!}$$

### Mathematical Proof of Marginal Invariance

#### 1. Home Goals Marginal for $x=0$:
$$\begin{aligned}
\sum_{y=0}^{\infty} P_{DC}(0, y) &= P_{DC}(0,0) + P_{DC}(0,1) + \sum_{y=2}^{\infty} P_{DC}(0,y) \\
&= e^{-\lambda} e^{-\mu}(1 - \lambda \mu \rho) + e^{-\lambda} (\mu e^{-\mu})(1 + \lambda \rho) + e^{-\lambda} \sum_{y=2}^{\infty} \frac{\mu^y e^{-\mu}}{y!} \\
&= e^{-\lambda} e^{-\mu} \left[ 1 - \lambda \mu \rho + \mu + \lambda \mu \rho \right] + e^{-\lambda} \sum_{y=2}^{\infty} \frac{\mu^y e^{-\mu}}{y!} \\
&= e^{-\lambda} e^{-\mu} (1 + \mu) + e^{-\lambda} \sum_{y=2}^{\infty} \frac{\mu^y e^{-\mu}}{y!} \\
&= e^{-\lambda} \sum_{y=0}^{\infty} \frac{\mu^y e^{-\mu}}{y!} = e^{-\lambda} = P(X=0)
\end{aligned}$$

#### 2. Home Goals Marginal for $x=1$:
$$\begin{aligned}
\sum_{y=0}^{\infty} P_{DC}(1, y) &= P_{DC}(1,0) + P_{DC}(1,1) + \sum_{y=2}^{\infty} P_{DC}(1,y) \\
&= (\lambda e^{-\lambda}) e^{-\mu}(1 + \mu \rho) + (\lambda e^{-\lambda}) (\mu e^{-\mu})(1 - \rho) + \lambda e^{-\lambda} \sum_{y=2}^{\infty} \frac{\mu^y e^{-\mu}}{y!} \\
&= \lambda e^{-\lambda} e^{-\mu} \left[ 1 + \mu \rho + \mu - \mu \rho \right] + \lambda e^{-\lambda} \sum_{y=2}^{\infty} \frac{\mu^y e^{-\mu}}{y!} \\
&= \lambda e^{-\lambda} e^{-\mu} (1 + \mu) + \lambda e^{-\lambda} \sum_{y=2}^{\infty} \frac{\mu^y e^{-\mu}}{y!} \\
&= \lambda e^{-\lambda} \sum_{y=0}^{\infty} \frac{\mu^y e^{-\mu}}{y!} = \lambda e^{-\lambda} = P(X=1)
\end{aligned}$$

#### 3. Home Goals Marginal for $x \ge 2$:
For any $x \ge 2$, $\tau(x, y) = 1$ for all $y \ge 0$. Hence:
$$\sum_{y=0}^{\infty} P_{DC}(x, y) = \sum_{y=0}^{\infty} P(X=x) P(Y=y) = P(X=x) \sum_{y=0}^{\infty} P(Y=y) = P(X=x)$$

By symmetry, the exact same proof holds for away marginals $\sum_{x=0}^{\infty} P_{DC}(x, y) = P(Y=y)$. 

**Why only $(0,0), (1,0), (0,1), (1,1)$ are affected:**  
The four linear perturbation terms $\pm \rho, \pm \lambda \rho, \pm \mu \rho, \mp \lambda \mu \rho$ form the unique minimal rank-1 bilinear modification on a $2 \times 2$ grid that simultaneously cancels out across both rows and columns. Extending the perturbation to higher scorelines ($x \ge 2$) would destroy this exact marginal invariance without empirical justification, since dependencies in football scores are concentrated exclusively in low-scoring draws and single-goal games.

---

## 5. Score-Matrix Derivation & 1X2 Probabilities

To obtain discrete match outcome probabilities ($P(\text{Home}), P(\text{Draw}), P(\text{Away})$), an adaptive score grid $M$ of size $(K+1) \times (K+1)$ is evaluated:

1. **Adaptive Truncation:** Choose grid size $K$ such that the upper tail truncation loss is bounded below tolerance ($1\text{e}{-15}$):
   $$K = \min \left\{ k \in \mathbb{N} : \sum_{j=0}^{k} \frac{\max(\lambda, \mu)^j e^{-\max(\lambda, \mu)}}{j!} > 1 - 10^{-15} \right\}$$
2. **Matrix Construction:**
   $$M_{i, j} = P_{DC}(X=i, Y=j) = \frac{\lambda^i e^{-\lambda}}{i!} \cdot \frac{\mu^j e^{-\mu}}{j!} \cdot \tau_{\lambda, \mu}(i, j)$$
3. **Renormalization:** Due to finite grid truncation $K$, renormalize:
   $$\tilde{M} = \frac{M}{\sum_{i=0}^K \sum_{j=0}^K M_{i, j}}$$
4. **Outcome Aggregation:**
   $$P(\text{Home}) = \sum_{i > j} \tilde{M}_{i, j}$$
   $$P(\text{Draw}) = \sum_{i = j} \tilde{M}_{i, i} = \tilde{M}_{0,0} + \tilde{M}_{1,1} + \sum_{k=2}^K \tilde{M}_{k, k}$$
   $$P(\text{Away}) = \sum_{i < j} \tilde{M}_{i, j}$$

---

## 6. Current V4 Draw Pipeline

In the current production codebase, V4 generates predictions through the following exact sequence:

```
[91 Feature Vector]
  ├─ 87 Base Features (Rolling xG, Form, Venue, Rest, E1 Causal Elo)
  └─  4 Online A/D States (A_home, D_home, A_away, D_away)
       │
       ▼
[LogisticRegressionPreprocessor.transform()]
       │
       ▼
[GLM PoissonRegressors (alpha=1.0)]
  ├─ model_home_goals.predict(E) ──> lambda_home > 0
  └─ model_away_goals.predict(E) ──> lambda_away > 0
       │
       ▼
[src/models/poisson.py :: hda_tail_safe()]
  ├─ Computes independent 1D Poisson PMF grids: ph, pa
  ├─ Cumulative vectorization:
  │    P(Home) = sum(ph[1:] * cumsum(pa)[:-1])
  │    P(Draw) = sum(ph * pa) = sum_k (ph[k] * pa[k])
  │    P(Away) = 1.0 - P(Home) - P(Draw)
  └─ Zero low-score correlation adjustment applied (tau == 1.0)
       │
       ▼
[P(Home), P(Draw), P(Away)]
```

### Key Properties of Current V4:
1. **Source of Lambdas:** Derived from separate Poisson GLMs with log-link functions fitted on historical features.
2. **Zero Low-Score Correction:** The current production pipeline (`src/models/poisson.py`) strictly assumes independent Poisson goals ($\rho \equiv 0$).
3. **P(Draw) Source:** Calculated entirely as the diagonal sum $\sum_{k=0}^K P(X=k)P(Y=k)$.
4. **Post-Processing Calibration:** None applied.

---

## 7. Current Architecture vs Dixon–Coles Proposal

```
CURRENT (V4 Baseline):
  Features ──> V4 GLM Models ──> (lambda_h, lambda_a) ──> [Independent Poisson Grid] ──> [Diagonal Sum] ──> 1X2 Probabilities

PROPOSED (Modular Dixon–Coles Layer):
  Features ──> V4 GLM Models ──> (lambda_h, lambda_a) ──> [Dixon–Coles Tau Grid] ──> [Diagonal Sum] ──> 1X2 Probabilities
                                                              ▲
                                                    rho (Fitted pre-2025/26)
```

### What Changes vs What Does NOT Change:
- **UNCHANGED:**
  - Feature extraction pipeline (`features.elo`, `features.online_attack_defense`, `features.db`).
  - Model artifacts and weights (`v4_poisson_venue_elo_online_ad.pkl`).
  - Preprocessor transformations (`StandardScaler` / preprocessing matrices).
  - Goal rate estimates $\lambda_{\text{home}}, \lambda_{\text{away}}$.
  - Data contracts (87 / 91 features).
- **CHANGED (In Isolated Research Layer Only):**
  - The probability conversion function (`hda_tail_safe` replaced by `predict_dc`).
  - Joint score matrix values for $(0,0), (1,0), (0,1), (1,1)$ before collapsing to 1X2.

---

## 8. Data Leakage & Temporal Isolation Audit

To guarantee zero lookahead bias and strict OOS validity, any parameter estimation for Dixon–Coles must satisfy strict causal boundaries:

1. **Training Set Cutoff:** All parameters (including $\rho$) must be estimated strictly on matches from seasons **2020/21 through 2024/25** ($n=8,983$).
2. **Quarantined Holdout:** The entire **2025/2026 season** (including the frozen 20, 50, 100, and 300 fixture sets) must remain 100% untouched during $\rho$ estimation.
3. **No Target Leakage:** A fixture's own kickoff time, in-play events, or final score cannot influence $\rho$.
4. **Feasibility:** Because $\rho$ represents an intrinsic structural coupling of low-scoring dynamics in professional football, estimating $\rho$ on historical data (2020/21–2024/25) is fully feasible and completely leak-free.

---

## 9. Draw-Specific Diagnostics: The Empirical Draw Deficit

### A. Historical Training Set Analysis ($n=8,983$ fixtures, 2020/21–2024/25)

Analysis of the historical training dataset shows a persistent structural draw deficit under independent Poisson modeling:

| Metric / Scoreline | Observed Historical Rate | Independent Poisson Rate | Residual ($\Delta = \text{Obs} - \text{Pois}$) |
|---|---|---|---|
| **Overall Draws ($X=Y$)** | **0.2539** (2,281 draws) | **0.2497** | **+0.0042** (+0.42% deficit) |
| **0–0** | **0.0640** (575 matches) | **0.0598** | **+0.0042** (Underestimated by Poisson) |
| **1–0** | **0.0902** (810 matches) | **0.0917** | **-0.0016** (Overestimated by Poisson) |
| **0–1** | **0.0714** (641 matches) | **0.0767** | **-0.0053** (Overestimated by Poisson) |
| **1–1** | **0.1238** (1,112 matches) | **0.1177** | **+0.0061** (Underestimated by Poisson) |

This confirms the classical **Dixon–Coles signature**: empirical football scores systematically cluster in $0\text{--}0$ and $1\text{--}1$, while independent Poisson spreads too much mass into $1\text{--}0$ and $0\text{--}1$.

### B. Fresh-Extended-300 OOS Sample Diagnostics ($n=300$)

| Evaluation Metric | V2 | V3 | V4 (Baseline) | Pinnacle Market | Empirical Sample |
|---|---|---|---|---|---|
| **Actual Draws** | 82 (27.33%) | 82 (27.33%) | 82 (27.33%) | 82 (27.33%) | 82 (27.33%) |
| **Mean $P(\text{Draw})$** | 0.2370 | 0.2354 | **0.2352** | **0.2555** | **0.2733** |
| **Median $P(\text{Draw})$** | 0.2377 | 0.2356 | **0.2356** | **0.2586** | — |
| **Max $P(\text{Draw})$** | 0.2827 | 0.2822 | **0.2895** | **0.3514** | — |
| **Min $P(\text{Draw})$** | 0.1706 | 0.1691 | **0.1695** | **0.1252** | — |
| **Draw Deficit vs Actual** | -3.63% | -3.79% | **-3.81%** | **-1.78%** | 0.00% |

The Pinnacle market reference explicitly prices in the low-scoring draw clustering (market mean $P(D) = 0.2555$, vs V4's $0.2352$).

---

## 10. Research-Only Counterfactual Analysis

Using the exact fixed $\lambda_{\text{home}}, \lambda_{\text{away}}$ vectors from the completed V4 Fresh-Extended-300 OOS evaluation, we evaluated the effect of applying the Dixon–Coles transformation across varying values of $\rho$:

| Experiment / Arm | $\rho$ Value | Accuracy | Log Loss | Brier Score | RPS | Mean $P(\text{Draw})$ | Draw Preds ($\text{argmax}$) |
|---|---|---|---|---|---|---|---|
| **V4 Baseline** | **0.000** | **0.5000** | **0.992706** | **0.592932** | **0.198447** | **0.2352** | **0** |
| **V4 + DC (Mild)** | -0.025 | 0.5000 | 0.991445 | 0.592250 | 0.198333 | 0.2408 | 0 |
| **V4 + DC (Historical MLE)** | -0.040 | 0.5000 | 0.990780 | 0.591887 | 0.198272 | 0.2441 | 0 |
| **V4 + DC (Training Optimal)** | **-0.100** | **0.5000** | **0.988780** | **0.590772** | **0.198087** | **0.2574** | **0** |
| **V4 + DC (Literature)** | -0.130 | 0.5000 | 0.988161 | 0.590419 | 0.198028 | 0.2640 | 0 |
| **Pinnacle Market Reference** | — | 0.5067 | 0.982167 | 0.586734 | 0.196659 | 0.2555 | 0 |

### Key Findings:
1. **Strict Monotonic Log Loss & Brier Improvement:** As $\rho$ shifts from $0.000$ to $-0.100$, Log Loss drops from $0.992706$ to $0.988780$ (a **$-0.003926$** improvement), Brier drops from $0.592932$ to $0.590772$ (a **$-0.002160$** improvement), and RPS drops from $0.198447$ to $0.198087$.
2. **Closing the Gap to Market:** The log loss difference between V4 and Market shrinks from $+0.010539$ down to $+0.006613$ (a **$37.2\%$ reduction in error gap**).
3. **Draw Probabilities Align with Market:** Mean $P(\text{Draw})$ under $\rho = -0.100$ reaches **$0.2574$**, almost exactly matching Pinnacle's closing draw pricing of **$0.2555$**.
4. **Argmax Classification Stability:** Even as $P(\text{Draw})$ increases to realistic levels, the modal class remains either Home or Away for all 300 fixtures, resulting in 0 discrete draw predictions.

---

## 11. $\rho$ Estimation & Sensitivity Analysis

### A. Candidate Estimation Strategies Comparison

| Strategy | Methodology | Strengths | Weaknesses / Risks |
|---|---|---|---|
| **A. Fixed Literature Value**<br>($\rho = -0.13$) | Adopts Dixon & Coles (1997) reported estimate directly. | Simple, requires zero training code. | Ignores modern tactical evolution and league differences; may overcorrect in high-scoring leagues. |
| **B. Global Historical Training Profile Log-Loss**<br>($\rho = -0.10$) | Profile search over grid minimizing log loss across all 8,983 historical training matches. | Optimizes for the primary evaluation metric directly; leak-free. | Pools all 5 leagues into a single parameter. |
| **C. League-Specific Historical MLE** | Fits distinct $\rho_c$ for each domestic league on historical data. | Captures league-specific scoring cultures and refereeing patterns. | Smaller sample per league ($n \approx 1,500\text{--}1,900$); higher variance in estimated $\rho_c$. |
| **D. Dynamic / Time-Decayed MLE** | Fits $\rho_t$ rolling over exponential window. | Adapts to secular scoring trends. | Sensitive to window hyperparameter $\xi$; complexity in production deployment. |

### B. League-Level Heterogeneity in Training Data

Fitting bivariate MLE $\rho$ per league on pre-2025/26 historical data reveals noticeable structural variation:

```
  Bundesliga       (n=1,530):  MLE rho = -0.1138  (Strong draw clustering)
  Ligue 1          (n=1,752):  MLE rho = -0.0617  (Moderate draw clustering)
  Serie A          (n=1,901):  MLE rho = -0.0359  (Mild draw clustering)
  La Liga          (n=1,900):  MLE rho = -0.0075  (Very mild draw clustering)
  Premier League   (n=1,900):  MLE rho = +0.0152  (Near zero / slight positive)
```

This divergence demonstrates why a blanket literature assumption ($\rho = -0.13$) cannot be blindly adopted without domestic league conditioning.

---

## 12. Implementation Comparison Table

| Approach | Formula | $\rho$ Source | Training Set | Leakage Risk | Draw Mechanism | Recommendation |
|---|---|---|---|---|---|---|
| **1. Current V4** | Independent Poisson ($X \perp Y$) | None ($\rho \equiv 0$) | 2020/21–2024/25 | Zero | Diagonal sum of marginals | Baseline (Structurally underestimates draws) |
| **2. Classical Dixon–Coles** | Joint MLE of $\alpha, \beta, \gamma, \rho$ | Joint MLE | Historical matches | Low if strictly pre-kickoff | Joint $\tau(x,y)$ score matrix | Reject (Lacks rich feature set of V4) |
| **3. Dixon–Coles + Time Decay** | Joint MLE with exponential weighting $e^{-\xi t}$ | Joint weighted MLE | Rolling historical window | Low if causal | Joint $\tau(x,y)$ score matrix | Reject (Incompatible with V4 gradient architecture) |
| **4. Modular V4 + Training-Fitted $\rho$** | **V4 GLMs + $\tau(x,y)$ score grid** | **Profile log loss on training set** | **2020/21–2024/25 (Fixed)** | **Zero** | **Joint $\tau(x,y)$ score matrix** | **Recommended for Next Research Phase** |
| **5. Modular V4 + League-Conditioned $\rho_c$** | **V4 GLMs + $\tau_c(x,y)$ score grid** | **Per-competition training profile** | **2020/21–2024/25 (Fixed)** | **Zero** | **League-conditioned $\tau_c(x,y)$ score matrix** | **Promising Alternative for Research Phase 2** |

---

## 13. Risks, Constraints & Mathematical Boundaries

1. **Probability Non-Negativity & Validity Bounds:**  
   The Dixon–Coles adjustment is a local linear perturbation and is only valid within strict bounds:
   $$\rho_{\min} = \max\left(-\frac{1}{\lambda_{\text{home}}}, -\frac{1}{\lambda_{\text{away}}}\right) < \rho < \min\left(1.0, \frac{1}{\lambda_{\text{home}} \lambda_{\text{away}}}\right) = \rho_{\max}$$
   If $\lambda_{\text{home}} = 3.5$ and $\lambda_{\text{away}} = 3.0$, $\rho_{\max} = \frac{1}{10.5} \approx 0.095$. Any implementation must assert batch validity bounds dynamically.
2. **Argmax Misconception:**  
   Implementing Dixon–Coles will **not** turn draw recall into a non-zero number under discrete $\text{argmax}$ mode selection unless the decision rule is adjusted (e.g., probability thresholding or utility-based betting selection). Dixon–Coles is a **probabilistic calibration improvement**, not a classification mode shifting tool.
3. **League Heterogeneity:**  
   Applying a Bundesliga-derived $\rho = -0.11$ to the Premier League ($\rho \approx 0$) would introduce artificial distortion.

---

## 14. Recommended Architecture & Implementation Design

When the project proceeds to implementation (in an isolated research branch, without modifying production), the design should adhere to:

```
[Phase 2 Design Specification]
  1. Module Location: src/models/dixon_coles.py
  2. Integration Point: Replace hda_tail_safe() call with predict_dc() in candidate evaluation harnesses.
  3. Parameter Estimation:
     - Pre-compute and freeze rho on training seasons (2020/21–2024/25).
     - Test both Global rho (-0.100) and League-conditioned rho_c.
  4. Mathematical Formulation:
     - Adaptive grid sizing (K derived from tail tolerance 1e-15).
     - Exact Dixon-Coles tau tensor calculation on (n, K+1, K+1).
     - Non-negativity clamp and sum-to-1 normalization.
  5. Safety Gates:
     - Assert rho inside analytical validity bounds for all fixture batches.
     - Require 1X2 row sums to equal 1.0 within 1e-9 tolerance.
     - Strict pre-kickoff causal isolation (no 2025/26 data in rho fitting).
```

---

## 15. Final Research Verdict

### **DIXON–COLES RESEARCH VERDICT:**  
**B. PROMISING BUT NEEDS FURTHER RESEARCH**

### Mathematical & Empirical Rationale:
1. **Mathematical Superiority over Independent Poisson:** Dixon–Coles successfully fixes the structural low-scoring draw deficit while preserving marginal Poisson goal rate consistency.
2. **Empirical Calibration Gain:** On the 300 OOS sample, V4 + DC ($\rho=-0.10$) reduces Log Loss from $0.992706$ to $0.988780$ ($\Delta = -0.003926$), improves Brier score, and elevates mean $P(\text{Draw})$ from $0.2352$ to $0.2574$ (in line with Pinnacle market odds $0.2555$).
3. **Reason for Not Selecting 'A' Immediately:** Significant league-level heterogeneity ($\rho \in [-0.1138, +0.0152]$) requires that Phase 2 investigate league-conditioned vs global $\rho$ modeling and establish formal walk-forward significance before any candidate promotion.

---

## 16. Strict Safety & Integrity Verification

Before concluding this research audit, all repository and data assets were audited to confirm zero unauthorized modifications:

- **Production Code Modified:** **NO**
- **V2 Model Artifact Modified:** **NO** (`25935b4e93fc4074f67f16e3181ed4df`)
- **V3 Model Artifact Modified:** **NO** (`a2850a7687822a5916663301f5ccc96c`)
- **V4 Model Artifact Modified:** **NO** (`06841f0c03c8597b2b8cd8f8ab064864`)
- **matches.db Modified:** **NO** (`fdeed042096fa1c851aaee6c84995247`)
- **features.db Modified:** **NO** (`e7ebe7fc07040a5927683c35b6371e63`)
- **odds_history.sqlite Modified:** **NO** (`0be31e8b59d739b72c3fb48e555d9fd8`)
- **research_dataset.sqlite Modified:** **NO** (`bdab370ffdfe5bbf8ff3a8a26e64471c`)
- **Frozen 20 / 50 / 100 / 300 Fixtures & Market DBs:** **NO** (All bit-identical)
- **Completed Evaluation Results:** **NO** (All manifests intact)
