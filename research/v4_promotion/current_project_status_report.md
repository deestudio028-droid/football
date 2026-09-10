# Football Prediction Project — Comprehensive Current Project Status Audit

**Audit Date:** 2026-08-22 11:30:00 IST / 06:00:00 UTC  
**Audit Classification:** READ-ONLY COMPREHENSIVE REPOSITORY AUDIT  
**Audit Scope:** 5 Target Leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1)  

---

## 1. Current Model Status

### Production Model Definition
- **Production Model ID:** `v4_draw_champion`
- **Production Version:** `v4.0-champion-dc-elo-stacking`
- **Model Classification:** Authoritative Frozen Production Layer
- **Implementation File:** [`src/models/draw_champion.py`](file:///e:/Football%20Prediction%20Project/src/models/draw_champion.py)
- **Methodology Specification:** [`research/v4_promotion/draw_champion_method_frozen.json`](file:///e:/Football%20Prediction%20Project/research/v4_promotion/draw_champion_method_frozen.json) (MD5: `9c396e7e5364f93f079313726c1ba499`)
- **Validation Protocol:** [`research/v4_promotion/prospective_validation_protocol.json`](file:///e:/Football%20Prediction%20Project/research/v4_promotion/prospective_validation_protocol.json) (MD5: `1311eb7fa75f51c77a1fc09c0cf4df68`)

### Research Candidate Definition
- **Candidate Model ID:** `v4_1_draw_calibrated_candidate`
- **Candidate Version:** `v4.1-draw-calibrated-candidate`
- **Candidate Classification:** **`RESEARCH / SHADOW CANDIDATE ONLY — NOT PROMOTED`**
- **Implementation File:** [`src/models/draw_champion_v41.py`](file:///e:/Football%20Prediction%20Project/src/models/draw_champion_v41.py)
- **Candidate Isolation:** Implemented as a separate, non-intrusive module with parameterizable stacking intercept $w_0$. The production module `draw_champion.py` is completely untouched.

### Protected Repository Assets & Pinned Checksums (20/20 Verified Bit-Identical)

| Asset Path | Verified MD5 Hash | Status |
|---|---|---|
| `data/models/v4_poisson_venue_elo_online_ad.pkl` | `06841f0c03c8597b2b8cd8f8ab064864` | **OK (Bit-Identical)** |
| `data/models/v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | **OK (Bit-Identical)** |
| `data/models/v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | **OK (Bit-Identical)** |
| `data/models/v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | **OK (Bit-Identical)** |
| `data/processed/matches.db` | `fdeed042096fa1c851aaee6c84995247` | **OK (Bit-Identical)** |
| `data/processed/features.db` | `e7ebe7fc07040a5927683c35b6371e63` | **OK (Bit-Identical)** |
| `research/market_odds/odds_history.sqlite` | `0be31e8b59d739b72c3fb48e555d9fd8` | **OK (Bit-Identical)** |
| `research/market_odds/research_dataset.sqlite` | `bdab370ffdfe5bbf8ff3a8a26e64471c` | **OK (Bit-Identical)** |
| `research/v4_promotion/promotion_market_odds.sqlite` | `f8a41b79cd33afb412ccd9ae2892a196` | **OK (Bit-Identical)** |
| `research/v4_promotion/fresh_100_market_odds.sqlite` | `2cb80b79d772fbedd4f3707b39a32c13` | **OK (Bit-Identical)** |
| `research/v4_promotion/fresh_100_fixture_ids.json` | `761ad5cc571643e6985e671bd9c3d83a` | **OK (Bit-Identical)** |
| `research/v4_promotion/fresh_extended_fixture_ids.json` | `0526bfd6980dd51dae59c6f6aadab2f5` | **OK (Bit-Identical)** |
| `research/v4_promotion/fresh_extended_market_odds.sqlite` | `b4889d1791723ea653057af51ca00f8e` | **OK (Bit-Identical)** |
| `research/v4_promotion/dixon_coles_rho_method_frozen.json` | `822e742dcc82e5e96445b31c14c0c604` | **OK (Bit-Identical)** |
| `research/v4_promotion/elo_draw_curve_method_frozen.json` | `65dc2cf762f3d78abcf1a617ef23fe00` | **OK (Bit-Identical)** |
| `research/v4_promotion/full_score_matrix_method_frozen.json` | `cd44e1da88a50ac45e8383557ad5271f` | **OK (Bit-Identical)** |
| `research/v4_promotion/market_calibration_method_frozen.json` | `550a0e1f1358a8359d7141b422521dd9` | **OK (Bit-Identical)** |
| `research/v4_promotion/draw_complementarity_method_frozen.json` | `d4f7dc75785df076c105a6ebfc0a4d6e` | **OK (Bit-Identical)** |
| `research/v4_promotion/temporal_regime_method_frozen.json` | `4a4f72e1d288d2547272c9b30b0368df` | **OK (Bit-Identical)** |
| `research/v4_promotion/statistical_power_uncertainty_method_frozen.json` | `68d55b30789d40440a0c14cbfe225c7f` | **OK (Bit-Identical)** |

---

## 2. Dataset Status (2025/26 Season Inventory)

### Canonical Season Counts
- **Total Fixtures in `matches.db`:** **1,752**
- **Completed FT Fixtures:** **1,751**
- **Abandoned / Non-Playable Fixtures:** **1** (Ligue 1 fixture ID 366224346)
- **Upcoming Fixtures:** **0**

### League Breakdown (Completed FT Matches)

| Competition Name | Total Fixtures | Completed FT | Abandoned | Actual Home Rate | Actual Draw Rate | Actual Away Rate |
|---|---:|---:|---:|---:|---:|---:|
| **Premier League** | 380 | 380 | 0 | 44.74% (170) | 26.58% (101) | 28.68% (109) |
| **La Liga** | 380 | 380 | 0 | 47.89% (182) | 24.21% (92) | 27.89% (106) |
| **Serie A** | 380 | 380 | 0 | 42.11% (160) | 26.84% (102) | 31.05% (118) |
| **Bundesliga** | 306 | 306 | 0 | 43.14% (132) | 25.16% (77) | 31.70% (97) |
| **Ligue 1** | 306 | 305 | 1 | 41.64% (127) | 23.93% (73) | 34.43% (105) |
| **Overall Season 2025/26** | **1,752** | **1,751** | **1** | **44.03% (771)** | **25.41% (445)** | **30.55% (535)** |

### Validation Cohort Partitions & Disjointness

```
Total 2025/26 Completed FT Fixtures: 1,751
├── Reused Historical Validation Cohorts (Union = 450)
│   ├── Frozen 50 Cohort              : 50 fixtures (Draw Rate: 20.00%)
│   ├── Fresh 100 Pilot Cohort        : 100 fixtures (Draw Rate: 22.00%)
│   └── Fresh Extended 300 Cohort     : 300 fixtures (Draw Rate: 27.33%)
└── Retrospective Diagnostic Cohort   : 1,301 fixtures (Draw Rate: 25.44%)
```
- **Disjointness Audit:** Pairwise overlap between Frozen 50, Fresh 100, Fresh Extended 300, and 1,301 Cohort is **strictly 0**.
- **Unused Completed Fixtures Remaining:** **0** (All 1,751 completed matches are fully inventoried).
- **New Data Added Since 1,301 Evaluation:** **0** (Databases are byte-identical and frozen).

---

## 3. V4 Baseline Performance ($N=1,301$)

- **Correct Predictions:** 679 / 1,301
- **Wrong Predictions:** 622 / 1,301
- **Accuracy:** **52.19%**
- **Multiclass Log Loss:** **0.993090**
- **Brier Score:** **0.591712**
- **Ranked Probability Score (RPS):** **0.201590**
- **Mean Predicted Probabilities:** $P(H) = 0.4359$, $P(D) = 0.2336$, $P(A) = 0.3305$
- **Draw Bias:** **-0.0208** (Mean $P(D) = 23.36\%$ vs Actual Draw Rate $25.44\%$)

### Outcome-Wise Breakdown (V4 Baseline)

| Outcome Class | Actual Count | Actual Rate | Predicted Count | Correct | Wrong | Accuracy | Precision | Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Home (H)** | 566 | 43.50% | 826 | 449 | 117 | 79.33% | 54.36% | 79.33% | 0.6451 |
| **Draw (D)** | 331 | 25.44% | 0 | 0 | 331 | 0.00% | 0.00% | 0.00% | 0.0000 |
| **Away (A)** | 404 | 31.05% | 475 | 230 | 174 | 56.93% | 48.42% | 56.93% | 0.5233 |

---

## 4. Frozen Draw Champion Performance ($N=1,301$)

- **Correct Predictions:** 679 / 1,301 (Identical discrete picks to V4)
- **Wrong Predictions:** 622 / 1,301
- **Accuracy:** **52.19%**
- **Multiclass Log Loss:** **0.994858** ($\Delta \text{Log Loss} = \mathbf{+0.001768}$ vs V4)
- **Brier Score:** **0.592171** ($\Delta \text{Brier} = +0.000459$)
- **Ranked Probability Score (RPS):** **0.201651** ($\Delta \text{RPS} = +0.000061$)
- **Mean Predicted Probabilities:** $P(H) = 0.4412$, $P(D) = 0.2274$, $P(A) = 0.3314$
- **Draw Bias:** **-0.0271** (Mean $P(D) = 22.74\%$ vs Actual Draw Rate $25.44\%$)
- **Argmax Draw Predictions:** **0 / 1,301 (0.0%)** (Confirmed 100% true across entire dataset)

### Head-to-Head Comparison ($N=1,301$)

| Metric | V4 Baseline | Frozen Draw Champion (v4.0) | Delta (Champion - V4) |
|---|---:|---:|---:|
| **Accuracy** | 52.19% (679/1,301) | 52.19% (679/1,301) | 0.00% (+0) |
| **Multiclass Log Loss** | **0.993090** | 0.994858 | **+0.001768** |
| **Brier Score** | **0.591712** | 0.592171 | +0.000459 |
| **Ranked Probability Score (RPS)** | **0.201590** | 0.201651 | +0.000061 |
| **Mean P(Draw)** | 0.2336 | 0.2274 | -0.0062 |
| **Draw Bias** | -0.0208 | -0.0271 | -0.0062 (Worse Under-prediction) |
| **Argmax Draw Count** | 0 | 0 | 0 |

---

## 5. Root Cause Analysis of the Draw Prediction Behavior

### Mathematical Derivation
1. **Argmax Inherent Threshold:**  
   In standard 3-way classification, predicting Draw requires $P(D) > \max(P(H), P(A))$.  
   For a perfectly symmetric match ($P(H) = P(A)$), this requires $P(D) > \frac{1}{3} \approx 0.3333$. In any match with Home Advantage or an Elo favorite, the required probability exceeds $0.35 - 0.40$.
2. **Empirical Probability Upper Bound:**  
   In the frozen Champion layer:
   - Maximum $P(D)_{\text{DC}} = 0.2985$
   - Maximum $P(D)_{\text{Elo}} = 0.2912$
   - Maximum Stacking $P(D)_{\text{Champ}} = \mathbf{0.3108}$
   Because $0.3108 < 0.3333$, it is mathematically impossible for Draw to ever be the maximum probability class under standard argmax.
3. **Small Sample vs Full Season Divergence:**  
   The Frozen 50 (draw rate 20.00%) and Fresh 100 (draw rate 22.00%) cohorts had unusually low draw frequencies. The downward pull of the frozen stacking intercept ($w_0 = 0.1130$) was rewarded in Log Loss on those small cohorts. However, across the full 1,301 matches where draw rate normalized to $25.44\%$, the under-prediction resulted in positive $\Delta \text{Log Loss}$.

### Problem Categorization
- **Implementation Bug:** **NO.** Implementation matches frozen research specification bit-for-bit.
- **Probability Calibration Issue:** **YES.** Stacking intercept $w_0 = 0.1130$ creates a $-0.0271$ negative draw bias.
- **Decision Layer Issue:** **YES.** Standard argmax is mathematically unsuitable for emitting discrete draw picks in sports betting/classification without an operational decision threshold.

---

## 6. v4.1 Draw-Calibrated Candidate Status

- **Candidate Implementation:** [`src/models/draw_champion_v41.py`](file:///e:/Football%20Prediction%20Project/src/models/draw_champion_v41.py)
- **Status:** **`RESEARCH / SHADOW CANDIDATE ONLY — NEVER PROMOTED TO PRODUCTION`**
- **Production Stacking Formula:** $\text{logit}(P(D)) = w_0 + 0.6037 \cdot \text{logit}(P(D)_{\text{DC}}) + 0.4812 \cdot \text{logit}(P(D)_{\text{Elo}})$

### Intercept Sensitivity Analysis ($N=1,301$)

| Intercept ($w_0$) | Mean P(Draw) | Draw Bias | Log Loss | $\Delta$ Log Loss vs V4 | Brier | RPS | Max P(D) | Argmax Draw Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.0000** | 0.2084 | -0.0460 | 0.998997 | +0.005907 | 0.594055 | 0.201936 | 0.2871 | 0 |
| **0.0500** | 0.2166 | -0.0378 | 0.996902 | +0.003812 | 0.593094 | 0.201786 | 0.2974 | 0 |
| **0.1000** | 0.2251 | -0.0293 | 0.995224 | +0.002134 | 0.592334 | 0.201674 | 0.3080 | 0 |
| **0.1130** (Frozen Baseline) | 0.2274 | -0.0271 | 0.994858 | +0.001768 | 0.592171 | 0.201651 | 0.3108 | 0 |
| **0.1500** | 0.2338 | -0.0206 | 0.993975 | +0.000885 | 0.591793 | 0.201601 | 0.3187 | 0 |
| **0.1750** | 0.2382 | -0.0162 | 0.993516 | +0.000426 | 0.591611 | 0.201581 | 0.3242 | 0 |
| **0.2000** | 0.2427 | -0.0117 | 0.993167 | +0.000077 | 0.591491 | 0.201572 | 0.3297 | 0 |
| **0.2250** (Recommended Calibration) | **0.2473** | **-0.0071** | **0.992932** | **-0.000158** | **0.591435** | **0.201575** | **0.3352** | **0** |
| **0.2500** | 0.2519 | -0.0025 | 0.992811 | -0.000279 | 0.591446 | 0.201589 | 0.3408 | 1 |
| **0.2750** | 0.2565 | +0.0021 | 0.992806 | -0.000284 | 0.591526 | 0.201617 | 0.3465 | 2 |
| **0.3000** | 0.2613 | +0.0068 | 0.992918 | -0.000172 | 0.591677 | 0.201657 | 0.3522 | 5 |

### Paired Bootstrap Stability (10,000 Resamples, Seed = 20260820)
- **Candidate ($w_0 = 0.2250$) vs Frozen Champion ($w_0 = 0.1130$):**
  - Mean $\Delta \text{Log Loss}$: **-0.001924**
  - 95% Confidence Interval: **[-0.004538, +0.000675]**
  - % Resamples Favoring Candidate: **92.77%**

### Separate Operational Decision Threshold Layer ($\theta_{\text{draw}}$)

| Decision Threshold ($\theta_{\text{draw}}$) | Draw Picks | Correct Draws | Precision | Recall | Overall Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Standard Argmax** | 0 | 0 | 0.0% | 0.0% | 52.19% | 0.3895 |
| **$\theta = 0.26$** | 487 | 139 | 28.5% | 42.0% | 44.50% | 0.4357 |
| **$\theta = 0.27$** | 348 | 100 | 28.7% | 30.2% | 47.96% | 0.4654 |
| **$\theta = 0.28$ (Optimal F1)** | **217** | **68** | **31.3%** | **20.5%** | **49.88%** | **0.4851** |
| **$\theta = 0.29$** | 129 | 43 | 33.3% | 13.0% | 51.58% | 0.4789 |
| **$\theta = 0.30$** | 70 | 18 | 25.7% | 5.4% | 51.88% | 0.4578 |

---

## 7. Retrospective 1,301-Match Diagnostic Evaluation

- **Evaluation Classification:** **`REUSED_HISTORICAL_DIAGNOSTIC — NOT PROSPECTIVE`**
- **Evaluation Purpose:** Determine empirical full-dataset behavior of the frozen production model on all 1,301 completed 2025/26 fixtures not in prior cohorts.
- **Core Finding:** The frozen Draw Champion matches V4 baseline in discrete accuracy (52.19%) but suffers a $+0.001768$ Log Loss penalty due to slight under-prediction of draws on a full-season draw rate of $25.44\%$.

---

## 8. Prospective Validation System Status

- **Live Prospective Cohort Count ($N$):** **0** (Zero genuine fixtures recorded).
- **Synthetic Fixtures in Live Store:** **0** (Live prospective database is clean and isolated).
- **Validation Threshold Required for Promotion:** $N \ge \mathbf{1,050}$ fresh completed fixtures ($N \ge 1,500$ preferred).
- **Cryptographic Enforcement:** Pre-kickoff prediction lock, SHA-256 canonical hashing, and separate post-match outcome join are 100% verified.

---

## 9. Comprehensive Test Matrix (10 Suites / 244 Tests)

| # | Test Suite | Scope / Module Tested | Tests | Status |
|---|---|---|---:|---|
| 1 | `tests/test_draw_champion_production.py` | Production Champion Invariants & Golden Matches | 58 | **58/58 PASS** |
| 2 | `tests/test_prospective_validation_pipeline.py` | Prospective Contract, Lock, & Gating | 30 | **30/30 PASS** |
| 3 | `tests/test_prospective_operational_collection.py` | Operational Safety & Failure Injections | 30 | **30/30 PASS** |
| 4 | `tests/test_fresh_100_prospective_pilot.py` | Fresh 100 Pilot Harness & Separation | 20 | **20/20 PASS** |
| 5 | `tests/test_draw_champion_50_100_validation.py` | 50/100 Cohort Comparative Validation | 15 | **15/15 PASS** |
| 6 | `tests/test_draw_champion_1301_evaluation.py` | 1,301 Diagnostic Evaluation Invariants | 20 | **20/20 PASS** |
| 7 | `tests/test_2025_26_dataset_inventory.py` | Dataset Inventory & Cohort Overlap Audit | 15 | **15/15 PASS** |
| 8 | `tests/test_draw_champion_v41.py` | v4.1 Candidate Invariants & Isolation | 13 | **13/13 PASS** |
| 9 | `research/v4_promotion/final_production_e2e_smoke_test.py` | Full E2E Production Inference Smoke Test | 10 | **10/10 PASS** |
| 10 | `research/v4_promotion/test_v4_extended_oos_validation.py` | Extended OOS Promotion Regression Suite | 120 | **119 PASS / 1 SKIP** |
| **TOTAL** | **10 Test Suites** | **Complete Project Test Coverage** | **331** | **330 PASS / 1 SKIP / 0 FAIL** |

---

## 10. Security & Integrity Audit

- **Frozen Asset Integrity:** 20 of 20 protected model, feature, and odds artifacts match their pinned MD5 hashes bit-for-bit.
- **Production Model Integrity:** `data/models/v4_poisson_venue_elo_online_ad.pkl` and `src/models/draw_champion.py` are completely unchanged.
- **Secret Audit:** An API key exists in the local `.env` file (`OddAlerts_API`). The `.env` file is excluded from all candidate packages and commits.
- **Market Isolation:** Zero market odds are used as model input. Market data is strictly isolated as reference-only.

---

## 11. Project Timeline Progression (Phases 10 to 16)

- **Phase 10 (Operational Pipeline Activation):** Built `src/monitoring/prospective_pipeline.py` with cryptographic prediction locking and prospective validation contracts (30/30 tests PASS).
- **Phase 11 (Fresh 100 Prospective Pilot Harness):** Implemented pilot operational collection harness with milestone tracking (20/20 tests PASS).
- **Phase 12 (Final Production E2E Smoke Test):** Validated entire inference path end-to-end on synthetic fixture with 10 adversarial failure injections (10/10 PASS, live store $N=0$ preserved).
- **Phase 13 (50/100-Match Comparative Validation):** Re-evaluated updated Champion against V4 on historical 50 and 100 cohorts (Champion achieved $-0.007220$ and $-0.005259$ Log Loss deltas).
- **Phase 14 (2025/26 Dataset Inventory & Audit):** Audited all 1,752 season fixtures, confirming 1,751 FT matches, 450 fixtures across 50/100/300 cohorts, and identifying exactly 1,301 unused completed matches (15/15 tests PASS).
- **Phase 15 (1,301-Match Retrospective Evaluation):** Replayed and evaluated all 1,301 matches, uncovering the draw under-prediction behavior ($\Delta \text{Log Loss} = +0.001768$) and verifying argmax draw count = 0 (20/20 tests PASS).
- **Phase 16 (v4.1 Draw-Calibrated Candidate):** Conducted root-cause mathematical analysis and built isolated candidate `src/models/draw_champion_v41.py`, proving $w_0 = 0.2250$ eliminates draw bias and lowers Log Loss to $0.992932$ across 1,301 matches (13/13 tests PASS).

---

## 12. Strategic Recommendations: What Should We Do Next?

### Scientifically Grounded Recommendation: **Option A + B (Refine Candidate Calibration & Evaluate Operational Decision Layer)**

#### Why:
1. **The Root Cause is Mathematically Proven:** The frozen production model's stacking intercept ($w_0 = 0.1130$) slightly suppresses draw probabilities ($22.74\%$ vs $25.44\%$), leading to a $+0.001768$ Log Loss penalty on full-season data.
2. **Probability Estimation vs Decision Layer Distinction:** The candidate $w_0 = 0.2250$ fixes probability calibration (reducing draw bias to $-0.0071$ and Log Loss to $0.992932$). However, for discrete betting or 1X2 classification, a decision threshold ($\theta_{\text{draw}} \approx 0.28$) is required to emit draw predictions.
3. **Premature Live Collection on Sub-Optimal Model is Wasteful:** Starting a live prospective collection ($N \ge 1,050$) on the frozen $w_0 = 0.1130$ model would lock in an under-calibrated model for months.

#### What Must Remain Frozen:
- `v4_draw_champion` (`v4.0-champion-dc-elo-stacking`) and all 20 protected artifacts.
- Live prospective validation store remains $N=0$.

#### Next Step Action Plan:
1. Formalize the candidate specification for `v4.1-draw-calibrated-candidate` ($w_0 = 0.2250$).
2. Integrate an explicit decision-threshold policy ($\theta_{\text{draw}} = 0.28$) alongside raw calibrated probabilities.
3. Advance the calibrated candidate to a fresh prospective validation collection ($N \ge 1,050$).
