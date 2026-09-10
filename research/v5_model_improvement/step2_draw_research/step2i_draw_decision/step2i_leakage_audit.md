# STEP 2I — TEMPORAL & DATA LEAKAGE AUDIT
**Football Prediction Project — Advanced Model Research Program**  
**Audit Date:** August 24, 2026  
**Auditor:** Model Governance & Decision Intelligence Engine  
**Governance Status:** PASS (100% Zero Leakage Enforced)  

---

## 1. Feature Provenance & Causal Timestamp Audit

Every feature and signal evaluated within Step 2I Draw Decision Intelligence Research is audited below for strict temporal causality, pre-kickoff availability, and isolation from future match data.

| Feature Name | Source Engine | Causal Generation Point | Future Info / Post-Kickoff Risk | Audit Result |
|:---|:---|:---|:---|:---:|
| **`v4_p_H`, `v4_p_D`, `v4_p_A`** | $V_{4.0}$ Production Baseline Model | Pre-match feature matrix $X_{t-}$ | NONE (Frozen artifact from historical cutoff) | **PASS** ✅ |
| **`cal_p_H`, `cal_p_D`, `cal_p_A`** | Platt Logistic Draw Calibrator | Evaluated on $V_{4.0} \ P(D)$ with frozen parameters | NONE (Parameters $a, b$ derived from training era) | **PASS** ✅ |
| **`lambda_home`, `lambda_away`** | Pre-Match Poisson Regressor | Expected goals model using historical form | NONE (Strictly pre-match $t-$ features) | **PASS** ✅ |
| **`lambda_total`** | $\lambda_H + \lambda_A$ | Derived from pre-match $\lambda$ | NONE | **PASS** ✅ |
| **`lambda_gap`** | $|\lambda_H - \lambda_A|$ | Derived from pre-match $\lambda$ | NONE | **PASS** ✅ |
| **`home_elo`, `away_elo`** | Dynamic Dynamic Elo System | Lagged team ratings prior to kickoff | NONE (Updated strictly post-match for next fixture) | **PASS** ✅ |
| **`abs_elo_diff`** | $|\text{Elo}_H + 100 - \text{Elo}_A|$ | Computed from pre-match Elo | NONE | **PASS** ✅ |
| **`A_home`, `D_home`, `A_away`, `D_away`** | Online Attack/Defense States | Recursive decay states prior to match date | NONE (Zero lookahead across fixtures) | **PASS** ✅ |
| **`ad_net_power_gap`** | $(A_H - D_A) - (A_A - D_H)$ | Derived from pre-match AD states | NONE | **PASS** ✅ |
| **`p_00`, `p_11`, `p_22`** | Bivariate Dixon-Coles Score Matrix | Matrix PMF evaluated on pre-match $\lambda_H, \lambda_A, \rho$ | NONE | **PASS** ✅ |
| **`p_score_space_draw`** | $\sum_k M[k, k]$ (Diagonal Mass) | Bivariate Dixon-Coles score matrix | NONE | **PASS** ✅ |
| **`p_low_score_mass`** | $\sum_{i+j \le 2} M[i, j]$ | Bivariate Dixon-Coles score matrix | NONE | **PASS** ✅ |
| **`cal_win_diff`** | $|P'(H) - P'(A)|$ | Calibrated win margin | NONE | **PASS** ✅ |
| **`cal_win_to_draw_margin`** | $\max(P'(H), P'(A)) - P'(D)$ | Calibrated win-to-draw margin | NONE | **PASS** ✅ |

---

## 2. Experimental Data Splitting Audit

- **Walk-Forward Chronological Structure:**
  - Fold 1 (2021/22): Trained strictly on 2020/21 ($N=1,826$).
  - Fold 2 (2022/23): Trained strictly on 2020/21–2021/22 ($N=3,652$).
  - Fold 3 (2023/24): Trained strictly on 2020/21–2022/23 ($N=5,479$).
  - Fold 4 (2024/25): Trained strictly on 2020/21–2023/24 ($N=7,231$).
  - Full Training Era: 2020/21–2024/25 ($N=8,983$).
- **Untouched Holdout Season (2025/2026, $N=1,751$):**
  - **Zero parameter tuning or threshold optimization was performed on the 2025/2026 holdout dataset.**
  - All decision policies and threshold grids were derived exclusively on the training era.
  - Holdout evaluation was executed exactly once as a final out-of-sample test.

---

## 3. Post-Match & Target Contamination Checks

- **Target Column (`actual_result` / `is_draw_actual`):** Used solely for scoring evaluation. Zero leakage into feature transformers or inference inputs.
- **Match Status Filtering:** Only completed matches (`FT`, `AWARDED`) were evaluated, with pre-match feature vectors extracted prior to kickoff.
- **Leakage Audit Verdict:** **100% PASS — ZERO TEMPORAL OR DATA LEAKAGE DETECTED.**
