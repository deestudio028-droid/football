# Phase 36 Experiment Report: E11 Historical xG + Causal Feature Engine

## 1. Executive Summary & Required Header Metrics

**Experiment ID:** `E11`  
**Focus Area:** Historical Expected Goals ($xG, xGA, npxG$) & Contextual Schedule Fatigue Features  
**Target Pipeline:** Linear Poisson GLMs and Non-Linear LightGBM Poisson Regressors vs. Frozen Production $V_4$  
**Evaluation Scope:** 7,082 Out-of-Sample Matches Across 4 Chronological Walk-Forward Folds (2022/23 through 2025/26)  
**Governance & Baseline Assets:** 20/20 Protected Hashes Verified 100% Bit-Identical Pre- and Post-Flight  

### Formal Verdict:
$$\mathbf{E11 \; RESEARCH\text{-}ONLY}$$
*(Rejected as an additive linear feature layer for $V_4$; preserved in research namespace for non-linear tree architectures).*

---

## 2. Master 6-Arm Ablation Results

Evaluating across all **7,082 pooled out-of-sample matches** from 4 walk-forward folds (2022/23 to 2025/26):

| Ablation Arm | Description / Configuration | $RPS$ | Log-Loss | Multiclass Brier | Draw $ECE$ | Mean $P(\text{Draw})$ | Actual Draw Rate |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm A ($V_4$ Baseline)** | Frozen $V_4$ Production Baseline (91 features) | **`0.199598`** | **`0.984955`** | **`0.586532`** | `0.0207` | `23.18%` | `25.25%` |
| **Arm B ($V_4$ + Raw xG)** | $V_4$ + Raw Rolling $xG$ (Windows 3, 5, 10) | `0.200516` | `0.988415` | `0.588866` | `0.0268` | `22.57%` | `25.25%` |
| **Arm C ($V_4$ + EWMA xG)** | $V_4$ + EWMA $xG$ ($t_{1/2} = 5$ matches) | `0.200100` | `0.986940` | `0.587909` | `0.0258` | `22.66%` | `25.25%` |
| **Arm D ($V_4$ + OppAdj xG)**| $V_4$ + Opponent-Adjusted $xG$ Strengths | `0.200078` | `0.986928` | `0.587874` | `0.0257` | `22.68%` | `25.25%` |
| **Arm E ($V_4$ + xG + Fatigue)**| $V_4$ + $xG$ + Rest Days & 14-Day Match Load | `0.200068` | `0.986888` | `0.587850` | `0.0253` | `22.72%` | `25.25%` |
| **Arm F ($V_4$ + E10 + Best xG)**| $V_4$ + Arm E + Dixon-Coles ($\rho = -0.08$) | `0.199931` | `0.985179` | `0.587029` | `0.0104` | `24.43%` | `25.25%` |
| **Benchmark: Pure $E_{10}$** | **$V_4$ Baseline + Dixon-Coles (No $xG$)** | **`0.199489`** | **`0.983360`** | **`0.585881`** | **`0.0069`** | **`24.92%`** | `25.25%` |

---

## 3. The 10 Core Scientific Questions & Empirical Answers

### Q1: Does $xG$ add information beyond $V_4$?
- **Answer**: **NO, NOT IN LINEAR GLM ARCHITECTURES**. When added on top of the existing 91 $V_4$ features, $xG$ increases $RPS$ from $0.199598 \to 0.200068$ ($\Delta RPS = \mathbf{+0.000470}$, degraded). $V_4$ already contains 36 rolling total shots and shots-on-target features, causal Elo, and Online Attack/Defense states. Adding rolling $xG$ creates severe collinearity that inflates variance in linear Poisson models.

### Q2: Which $xG$ features are useful?
- **Answer**: In tree-based feature importance, **`xg_diff_expected` (Rank #2 overall, importance 90.0)** and **`xg_sum_expected` (Rank #12, importance 28.5)** were the most informative $xG$ features. Raw rolling window features (3, 5, 10 matches) performed worse than smooth EWMA representations.

### Q3: Does $xG$ improve draws?
- **Answer**: **NO**. Without Dixon-Coles transformation, models with $xG$ actually under-predicted draws more severely ($22.57\%\text{--}22.72\%$ predicted vs $25.25\%$ actual), increasing Draw $ECE$ from $0.0207 \to 0.0253$.

### Q4: Does $xG$ improve H/D/A overall?
- **Answer**: **NO**. Overall accuracy slightly decreased from $53.12\% \to 52.88\%$, and multiclass Log-Loss degraded from $0.984955 \to 0.986888$.

### Q5: Which leagues benefit?
- **Answer**: La Liga showed marginal stability ($\Delta RPS = -0.000112$), but Premier League ($+0.000342$), Bundesliga ($+0.000546$), Serie A ($+0.000219$), and Ligue 1 ($+0.000781$) all experienced slight degradation when $xG$ was added to linear GLM models.

### Q6: How much historical data is required?
- **Answer**: At least 5 matches of rolling history are required for $xG$ features to stabilize. For newly promoted teams with $< 5$ matches, Bayesian shrinkage priors ($m = 5$ pseudo-counts to league mean $1.250$) successfully prevented wild prediction spikes.

### Q7: Does $xG$ improve $E_{10}$ further?
- **Answer**: **NO**. Pure $E_{10}$ ($V_4$ + Dixon-Coles without $xG$) achieved $RPS = \mathbf{0.199489}$, whereas $E_{10} + xG$ achieved $RPS = \mathbf{0.199931}$. Adding $xG$ harmed the pure score distribution benefits of $E_{10}$ by $+0.000442$ in $RPS$.

### Q8: Does non-linear modeling provide additional gain?
- **Answer**: Dual LightGBM Poisson Regressors with $xG$ achieved $RPS = 0.201516$ and Log-Loss = $0.990024$. While tree models naturally handled the non-linear interaction between $xG$ and fatigue, they suffered slight boundary overconfidence compared to regularized GLMs.

### Q9: Is the gain statistically credible?
- **Answer**: The 1,000-replicate matchweek cluster bootstrap for $V_4 + xG$ yielded $\Delta RPS = +0.000470$ ($95\%$ CI $[-0.000038, +0.000958]$, $p = 0.068$). There is zero statistically significant positive gain from adding $xG$ to $V_4$.

### Q10: Is $E_{11}$ production-worthy?
- **Answer**: **NO**. $E_{11}$ fails Gate 1 (No $RPS$ reduction) and Gate 2 (No Log-Loss reduction). It is rejected for production and remains research-only.

---

## 4. Chronological Walk-Forward Season-by-Season Breakdown

| Walk-Forward Fold | Season | Matches ($N$) | $V_4$ Baseline $RPS$ | Arm E ($V_4 + xG$) $RPS$ | Arm F ($V_4 + E_{10} + xG$) $RPS$ | Pure $E_{10}$ $RPS$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 2022/2023 | 1,827 | `0.203679` | `0.204266` | `0.204240` | **`0.203677`** |
| **Fold 2** | 2023/2024 | 1,752 | `0.194130` | `0.194398` | `0.194162` | **`0.193893`** |
| **Fold 3** | 2024/2025 | 1,752 | `0.198900` | `0.199566` | `0.199392` | **`0.198805`** |
| **Fold 4 (Blind Holdout)** | 2025/2026 | 1,751 | `0.201506` | `0.201864` | `0.201748` | **`0.201402`** |
| **Total / Pooled OOS** | **All 4 Folds** | **7,082** | **`0.199598`** | `0.200068` | `0.199931` | **`0.199489`** |

---

## 5. LightGBM GBDT Top 15 Feature Importance Rankings

| Rank | Feature Name | Mean Tree Split Importance | Category |
|:---:|:---|:---:|:---|
| **1** | `elo_diff` | **`128.0`** | Causal Elo Rating Difference ($E_1$) |
| **2** | `xg_diff_expected` | **`90.0`** | **Expected Goals Differential** |
| **3** | `A_home` | **`62.0`** | Online Attack State ($E_6$) |
| **4** | `D_away` | **`37.5`** | Online Defense State ($E_6$) |
| **5** | `strength_diff` | `33.5` | Historical Baseline Strength |
| **6** | `away_attack_strength_score` | `33.0` | Historical Baseline Strength |
| **7** | `away_shots_on_for_per_match_last10` | `32.5` | Rolling Shots on Target |
| **8** | `home_shots_for_per_match_season` | `31.5` | Rolling Shots Total |
| **9** | `away_shots_diff_season` | `31.5` | Rolling Shots Differential |
| **10** | `D_home` | `31.0` | Online Defense State ($E_6$) |
| **11** | `A_away` | `29.5` | Online Attack State ($E_6$) |
| **12** | `xg_sum_expected` | **`28.5`** | **Total Match $xG$ Expectation** |
| **13** | `home_shots_against_per_match_season` | `27.5` | Rolling Shots Conceded |
| **14** | `away_xg_def_strength` | **`25.5`** | **Opponent-Adjusted Defensive $xG$** |
| **15** | `home_defence_strength_score` | `25.0` | Historical Baseline Strength |

---

## 6. Root Cause: Why $xG$ Did Not Improve $V_4$ Linear Models

1. **Feature Redundancy with Existing $V_4$ Pipeline**:
   The production $V_4$ model is not a simple goal-based model; it already contains **36 rolling total shots and shots-on-target features** (`home/away_shots_on_for/against/diff_last5/10/season`), **Causal Elo ($E_1$)**, and **Online Attack/Defense gradient states ($E_6$)**.
   Since $xG$ is heavily determined by shots and shots on target ($r \approx 0.72$), adding 15+ rolling $xG$ features to a linear GLM causes severe multicollinearity without providing orthogonal signal.
2. **Structural Deficit vs. Feature Deficit**:
   As proven in Phase 35 ($E_{10}$), the primary flaw of $V_4$ was **not its team ability estimation**, but its **mathematical independence assumption in the score distribution grid**. $E_{10}$ repaired this with Dixon-Coles, reducing $RPS$ to `0.199489`. Adding $xG$ does not solve the low-score correlation issue.

---

## 7. Promotion Decision & Next Steps

### Formal Promotion Status:
$$\mathbf{E11 \; RESEARCH\text{-}ONLY}$$

### Decision Rationale:
- Fails Gate 1: Out-of-sample $RPS$ increased by $+0.000470$ (from $0.199598 \to 0.200068$).
- Fails Gate 2: Out-of-sample Log-Loss increased by $+0.001933$ (from $0.984955 \to 0.986888$).
- Fails Gate 3: Underperformed pure $E_{10}$ by $+0.000442$ in $RPS$.
- Asset Integrity: All 20 protected repository hashes verified 100% bit-identical.

**Recommendation**: Maintain production $V_4$ frozen. $E_{10}$ remains the sole candidate for prospective testing (`CANDIDATE FOR PROSPECTIVE TEST`). Preserve $E_{11}$ artifacts in the research directory for feature pruning studies.
