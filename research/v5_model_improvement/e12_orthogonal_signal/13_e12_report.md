# Phase 37 Experiment Report: E12 Orthogonal Signal & Bounded Lambda Adjustment

## 1. Executive Summary & Required Header Metrics

**Experiment ID:** `E12`  
**Focus Area:** Non-Redundant xG Residuals, Momentum Trends, Venue Specialization & Bounded Intensity Adjustment  
**Primary Benchmark:** **Pure $E_{10}$ Dixon-Coles Champion ($\rho = -0.08$)**  
**Evaluation Scope:** 7,082 Out-of-Sample Matches Across 4 Chronological Walk-Forward Folds (2022/23 through 2025/26)  
**Governance & Baseline Assets:** 20/20 Protected Hashes Verified 100% Bit-Identical Pre- and Post-Flight  

### Formal Verdict:
$$\mathbf{E12 \; RESEARCH\text{-}ONLY \quad (\text{KEEP } E_{10} \text{ AS CHAMPION})}$$
*(E12 prevented the severe collinearity degradation of E11, but failed to achieve statistically significant gains over Pure E10. Pure E10 remains the champion).*

---

## 2. Master 10-Arm Ablation Results (7,082 Out-of-Sample Matches)

| Ablation Arm | Description / Configuration | $RPS$ | Log-Loss | Multiclass Brier | Draw $ECE$ | Mean $P(\text{Draw})$ | Actual Draw Rate |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline: $V_4$ Production** | Frozen $V_4$ Baseline (91 features) | `0.199598` | `0.984955` | `0.586532` | `0.0207` | `23.18%` | `25.25%` |
| **ARM A (Pure $E_{10}$ Champion)** | **$V_4$ + Dixon-Coles ($\rho = -0.08$, No Adjustment)** | **`0.199489`** | **`0.983360`** | **`0.585881`** | **`0.0069`** | **`24.92%`** | `25.25%` |
| **ARM B ($E_{10}$ + xG Residual)** | $E_{10}$ + Finishing & Defensive Luck Residuals | `0.199849` | `0.984488` | `0.586754` | `0.0061` | `24.84%` | `25.25%` |
| **ARM C ($E_{10}$ + xG Trend)** | $E_{10}$ + Short vs Long $xG$ Momentum | `0.199539` | `0.983653` | `0.586018` | `0.0081` | `24.90%` | `25.25%` |
| **ARM D ($E_{10}$ + Venue Spec)** | $E_{10}$ + Venue-Specific Home/Away $xG$ | `0.199288` | `0.982788` | `0.585583` | `0.0104` | `24.30%` | `25.25%` |
| **ARM E ($E_{10}$ + Matchup $xG$)** | $E_{10}$ + Attack vs. Defense Matchup Interplay | `0.199344` | `0.982996` | `0.585651` | `0.0110` | `24.30%` | `25.25%` |
| **ARM F ($E_{10}$ + Fatigue)** | $E_{10}$ + Rest Days & 7-Day Match Congestion | `0.199591` | `0.983721` | `0.586124` | `0.0070` | `24.77%` | `25.25%` |
| **ARM G ($E_{10}$ + Elo-xG Orthogonal)**| $E_{10}$ + $xG$ Dominance vs Elo Expectancy | `0.199381` | `0.983053` | `0.585713` | `0.0095` | `24.30%` | `25.25%` |
| **ARM H ($E_{10}$ + Best Orth Combo)** | $E_{10}$ + Union of Top Orthogonal Signals | `0.199312` | `0.982804` | `0.585612` | `0.0108` | `24.33%` | `25.25%` |
| **ARM I ($E_{10}$ + Tight Bounded $\pm 2\%$)** | $E_{10}$ + Orthogonal Adjustment ($B = 0.02$) | `0.199328` | `0.982837` | `0.585578` | `0.0068` | `24.68%` | `25.25%` |
| **ARM J ($E_{10}$ + Bounded $\pm 5\%$)** | **$E_{10}$ + Orthogonal Adjustment ($B = 0.05$)** | **`0.199312`** | **`0.982804`** | **`0.585612`** | **`0.0108`** | **`24.33%`** | `25.25%` |

---

## 3. The 17 Core Scientific Questions & Empirical Answers

### Q1: Does xG residual add information beyond V4/E10?
- **Answer**: **NO**. Arm B ($RPS = 0.199849$) degraded vs Pure $E_{10}$ ($0.199489$). Historical finishing luck residuals revert to mean slowly over noisy short samples and add variance to intensity multipliers.

### Q2: Does xG trend add information?
- **Answer**: **MARGINAL / NEUTRAL**. Arm C ($RPS = 0.199539$) performed nearly identical to Pure $E_{10}$ ($0.199489$). Short-term form shifts are already partially captured by V4's dynamic Online Attack/Defense gradients ($E_6$).

### Q3: Does venue-specific xG help?
- **Answer**: **SLIGHT NUMERICAL GAIN**. Arm D achieved $RPS = 0.199288$ ($\Delta RPS = -0.000201$ vs $E_{10}$), indicating that teams have persistent venue-specific creation baselines not fully explained by league-wide home advantage.

### Q4: Does xG matchup help?
- **Answer**: **YES, IN ISOLATION**. Arm E achieved $RPS = 0.199344$ by projecting cross-strengths ($\text{Att}_H \times \text{Def}_A$).

### Q5: Does fatigue $\times$ xG help?
- **Answer**: **NO**. Arm F ($RPS = 0.199591$) degraded relative to Pure $E_{10}$. Schedule congestion is infrequent in domestic leagues and caused slight noise in normal 7-day turnaround matches.

### Q6: Does Elo-xG residual help?
- **Answer**: **SLIGHT NUMERICAL GAIN**. Arm G achieved $RPS = 0.199381$. Comparing $xG$ implied dominance against Elo difference identifies subtle market mispricings.

### Q7: Does bounded lambda adjustment help?
- **Answer**: **YES, ARCHITECTURALLY**. Bounding the adjustment to $\pm 5\%$ prevented the severe multi-collinear blowup seen in $E_{11}$ ($RPS = 0.200068 \to 0.199312$).

### Q8: Which feature group provides the most orthogonal information?
- **Answer**: **Venue-Specific $xGD$ (`venue_spec_xgd_diff`)** and **Matchup Delta (`xg_matchup_delta`)** provided the cleanest orthogonal signal.

### Q9: Which architecture is best?
- **Answer**: **Two-stage Regularized Ridge Residual Model with Bounded Multiplier ($B = 0.05$) on top of Dixon-Coles ($\rho = -0.08$)** (Arm J).

### Q10: Does it beat E10?
- **Answer**: **NUMERICALLY YES, BUT STATISTICALLY NO**. Arm J achieved $RPS = 0.199312$ vs $E_{10} = 0.199489$ ($\Delta RPS = \mathbf{-0.000177}$).

### Q11: Is the improvement statistically significant?
- **Answer**: **NO**. The 1,000-replicate matchweek cluster bootstrap yielded $p = 0.4220$ with $95\%$ CI $[-0.000651, +0.000237]$. The confidence interval widely straddles zero.

### Q12: Does it improve all five leagues or only some?
- **Answer**: **ONLY SOME**. La Liga ($\Delta RPS = -0.000753$), Serie A ($-0.000532$), and Premier League ($-0.000071$) improved, but Bundesliga ($+0.000486$) and Ligue 1 ($+0.000166$) degraded.

### Q13: Does it preserve E10's draw calibration?
- **Answer**: **SLIGHT DEGRADATION**. Pure $E_{10}$ had Draw $ECE = \mathbf{0.0069}$. Arm J shifted Draw $ECE$ slightly to $\mathbf{0.0108}$ because the intensity adjustment slightly tilted draws toward wins in asymmetric matches.

### Q14: Does it preserve low-score calibration?
- **Answer**: **YES**. Actual 1-1 frequency is $12.07\%$. $E_{10}$ predicted $11.78\%$, and $E_{12}$ predicted $11.50\%$ (well above $V_4$'s suppressed $10.90\%$).

### Q15: Does it generalize to the 2025/26 blind holdout?
- **Answer**: **NO**. On Fold 4 (2025/26 Blind Holdout), Arm J achieved $RPS = 0.201471$ vs Pure $E_{10} = \mathbf{0.201402}$ (degraded by $+0.000069$).

### Q16: What is the exact final E12 formula?
- **Answer**:
  $$\lambda_H^{\text{adj}} = \lambda_H^{V_4} \cdot \exp(\text{clip}(\hat{w}_H^T X_{\text{orth}}, -0.05, +0.05))$$
  $$\lambda_A^{\text{adj}} = \lambda_A^{V_4} \cdot \exp(\text{clip}(\hat{w}_A^T X_{\text{orth}}, -0.05, +0.05))$$
  $$P(x, y) = \text{DixonColes}(\lambda_H^{\text{adj}}, \lambda_A^{\text{adj}}, \rho = -0.08)$$

### Q17: Should E12 become the next prospective candidate?
- **Answer**: **NO. STATUS = RESEARCH ONLY. KEEP E10 AS CHAMPION.**

---

## 4. Chronological Walk-Forward Fold-by-Fold Breakdown

| Fold | Out-of-Sample Season | Matches ($N$) | $V_4$ Baseline $RPS$ | Pure $E_{10}$ Champion $RPS$ | $E_{12}$ Bounded ($\pm 5\%$) $RPS$ | Delta vs. $E_{10}$ | Result |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 2022/2023 | 1,827 | `0.203679` | **`0.203677`** | `0.203979` | `+0.000302` | Degraded |
| **2** | 2023/2024 | 1,752 | `0.194130` | `0.193893` | **`0.193300`** | **`-0.000593`** | Improved |
| **3** | 2024/2025 | 1,752 | `0.198900` | `0.198805` | **`0.198300`** | **`-0.000505`** | Improved |
| **4** | **2025/2026 (Blind Holdout)** | 1,751 | `0.201506` | **`0.201402`** | `0.201471` | `+0.000069` | Degraded |
| **Total** | **All 4 Test Seasons** | **7,082** | **`0.199598`** | **`0.199489`** | **`0.199312`** | **`-0.000177`** | **Inconsistent** |

---

## 5. League Breakdown Comparison

| Competition | Matches ($N$) | Pure $E_{10}$ $RPS$ | $E_{12}$ Bounded $RPS$ | Delta $RPS$ | Pure $E_{10}$ Log-Loss | $E_{12}$ Log-Loss | Delta Log-Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Premier League** | 1,520 | `0.199900` | **`0.199829`** | **`-0.000071`** | `0.976316` | `0.976144` | **`-0.000171`** |
| **La Liga** | 1,520 | `0.198007` | **`0.197254`** | **`-0.000753`** | `0.979583` | `0.977025` | **`-0.002558`** |
| **Bundesliga** | 1,224 | **`0.200110`** | `0.200596` | `+0.000486` | `0.986526` | `0.988327` | `+0.001801` |
| **Serie A** | 1,521 | `0.195323` | **`0.194791`** | **`-0.000532`** | `0.985936` | `0.984190` | **`-0.001747`** |
| **Ligue 1** | 1,297 | **`0.205043`** | `0.205209` | `+0.000166` | `0.990031` | `0.990542` | `+0.000510` |

---

## 6. Scoreline Calibration Matrix ($N = 7,082$ Matches)

| Scoreline | Actual Match Count | Actual Empirical Freq | Pure $E_{10}$ Mean Prob | $E_{12}$ Mean Prob | Delta Prob |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0-0** | 436 | **6.16%** | **7.00%** | **6.86%** | -0.14% |
| **1-0** | 665 | **9.39%** | **8.16%** | **8.04%** | -0.12% |
| **0-1** | 503 | **7.10%** | **6.94%** | **6.86%** | -0.07% |
| **1-1** | 855 | **12.07%** | **11.78%** | **11.50%** | -0.28% |
| **2-0** | 470 | **6.64%** | **7.14%** | **7.16%** | +0.02% |
| **0-2** | 373 | **5.27%** | **5.34%** | **5.39%** | +0.05% |
| **2-1** | 619 | **8.74%** | **8.16%** | **7.99%** | -0.16% |
| **1-2** | 500 | **7.06%** | **7.05%** | **6.93%** | -0.12% |
| **2-2** | 407 | **5.75%** | **4.98%** | **4.84%** | -0.14% |

---

## 7. Promotion Evaluation & Final Scientific Decision

$$\mathbf{E12 \; FAILED \; TO \; BEAT \; E10 \; CONVINCINGLY \; — \; KEEP \; E10 \; AS \; CHAMPION}$$

### Promotion Gate Evaluation:
1. **Gate 1 ($RPS$ Improvement over $E_{10}$)**: Marginal (0.199312 vs 0.199489, $\Delta RPS = -0.000177$).
2. **Gate 2 (Log Loss Improvement over $E_{10}$)**: Marginal (0.982804 vs 0.983360, $\Delta LL = -0.000556$).
3. **Gate 3 (Walk-Forward Consistency)**: **FAILED** (Folds 1 and 4 degraded; only Folds 2 and 3 improved).
4. **Gate 4 (League Consistency)**: **FAILED** (Bundesliga degraded by $+0.000486$ and Ligue 1 by $+0.000166$).
5. **Gate 5 (Statistical Significance)**: **FAILED** ($p = 0.4220$, bootstrap $95\%$ CI $[-0.000651, +0.000237]$ widely straddles zero).
6. **Gate 6 (Holdout Generalization)**: **FAILED** (Degraded on 2025/26 Blind Holdout from $0.201402 \to 0.201471$).
7. **Gate 7 (Asset Protection)**: **PASSED** (All 20 protected baseline hashes verified 100% bit-identical).

---

## 8. Final Governance Architecture State

- **$V_4$ Production**: **`FROZEN`**
- **$E_{10}$ Dixon-Coles ($\rho = -0.08$)**: **`CANDIDATE FOR PROSPECTIVE TEST (CHAMPION)`**
- **$E_{11}$ Historical $xG$**: **`RESEARCH ONLY (REJECTED)`**
- **$E_{12}$ Orthogonal Signal & Bounded Adjustment**: **`RESEARCH ONLY`**
