# Phase Prospective: Step 1 Training Report — V4.1 Prospective Candidate

## 1. Executive Summary & Verification Metrics

**Candidate Model Identity:** `v4_1_prospective_candidate`  
**Model Version Tag:** `v4.1-champion-dc-elo-stacking-2025-26-trained`  
**Candidate Artifact Path:** [`data/models/v4_1_prospective_candidate_2025_26.pkl`](file:///e:/Football%20Prediction%20Project/data/models/v4_1_prospective_candidate_2025_26.pkl)  
**Candidate Artifact MD5:** `145f918d933eb343c0f63ca342b10289`  
**Candidate Artifact SHA256:** `cbb00b32c3ce4dbf600564a95d11b67a0c236f6b34cd184a9ca8abcc8ab94ac8`  
**Candidate Artifact File Size:** 15,506 bytes  

---

## 2. Frozen V4 Production Integrity Status

| Attribute | Frozen Production V4 Baseline | Verification Result |
|:---|:---|:---:|
| **File Path** | `data/models/v4_poisson_venue_elo_online_ad.pkl` | Verified Immutable |
| **MD5 Hash** | `06841f0c03c8597b2b8cd8f8ab064864` | **100% Bit-Identical** |
| **SHA256 Hash** | `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5` | **100% Bit-Identical** |
| **File Size** | 13,534 bytes | **100% Match** |
| **Modification Timestamp** | `1787239544.7429636` | **Untouched** |
| **Protected Assets** | 20/20 Hashes Verified Pre- and Post-Flight | **100% Match** |

---

## 3. Dataset & Training Scope

1. **Total Historical Training Matches**: **10,734 matches**
   - 2020/2021: 1,826 matches
   - 2021/2022: 1,826 matches
   - 2022/2023: 1,827 matches
   - 2023/2024: 1,752 matches
   - 2024/2025: 1,752 matches
   - 2025/2026: **1,751 matches**
2. **Training Date Boundary**: `2020-08-21T17:00:00.000000Z` to `2026-05-24T19:45:00.000000Z`
3. **Leagues Included (5 Major European Leagues)**:
   - Premier League (CID 423): 380 matches in 2025/26 (2,280 total)
   - La Liga (CID 419): 380 matches in 2025/26 (2,280 total)
   - Serie A (CID 499): 380 matches in 2025/26 (2,281 total)
   - Bundesliga (CID 477): 306 matches in 2025/26 (1,836 total)
   - Ligue 1 (CID 200): 306 matches in 2025/26 (2,057 total)
4. **Quarantined Fixture**: `420450481` (Nantes vs Toulouse, ABANDONED — no result label).

---

## 4. Information Clock & Causal Leakage Audit

- **Historical Boundary**: No fixture beyond the 2025/26 season finale (May 24, 2026) entered training.
- **Prospective 2026 Live Fixtures**: **100% UNSEEN AND EXCLUDED**. No 2026 live matches were used for feature extraction, Elo updates, baseline goal rates, median imputation, or Poisson fitting.
- **Causality**: Features for match $i$ at time $t$ access only information generated prior to kickoff ($t_{\text{info}} \le t_{\text{kickoff}}$).

---

## 5. Training Results & Sanity Verification

- **Fitted Poisson Goal Regressors**:
  - $\lambda_{\text{home}}$: Mean = $1.5348$, Min = $0.6064$, Max = $4.5025$
  - $\lambda_{\text{away}}$: Mean = $1.2741$, Min = $0.5115$, Max = $3.6498$
- **Online Attack/Defense Baseline Goal Rates**:
  - $\mu_{\text{home}} = 1.534843$
  - $\mu_{\text{away}} = 1.273989$
  - $N_{\text{train}} = 10,734$
- **Probability Simplex Integrity**:
  - All predicted probabilities $P(H), P(D), P(A)$ are strictly positive, bounded in $[0, 1]$, and sum to $1.000000$.
  - Dixon-Coles low-score transformation ($\rho = -0.08$) properly configured.
- **Automated Test Suite**:
  - [`tests/test_v4_1_prospective_training.py`](file:///e:/Football%20Prediction%20Project/tests/test_v4_1_prospective_training.py): **3/3 PASSED (100%)**.

---

## 6. Real Prospective Validation Lock & Traceability

1. **Candidate Status**: `PROSPECTIVE_TEST_CANDIDATE` (Not yet promoted to production).
2. **Model Version**: `v4.1-champion-dc-elo-stacking-2025-26-trained`.
3. **Traceability**: All prospective predictions generated during the 2026 live season will be logged with `model_file_md5 = 145f918d933eb343c0f63ca342b10289` prior to kickoff.

---

## 7. Final Governance State

$$\begin{aligned}
\mathbf{V4.0:} & \quad \mathbf{FROZEN \; HISTORICAL \; PRODUCTION \; MODEL} \\
\mathbf{V4.1:} & \quad \mathbf{NEW \; PROSPECTIVE \; TEST \; CANDIDATE \; (TRAINED \; THROUGH \; 2025/26)} \\
\mathbf{E10:} & \quad \mathbf{CURRENT \; CHAMPION \; ARCHITECTURE \; (DIXON\text{-}COLES \; \rho = -0.08)} \\
\mathbf{E14:} & \quad \mathbf{RESEARCH \; ONLY} \\
\mathbf{E15:} & \quad \mathbf{RESEARCH \; ONLY \; / \; RISK \; DIAGNOSTIC} \\
\mathbf{E16:} & \quad \mathbf{RESEARCH \; ONLY \; / \; RELIABILITY \; LAYER} \\
\mathbf{E17:} & \quad \mathbf{RESEARCH \; ONLY \; / \; FORECAST \; STATUS \; ENGINE} \\
\mathbf{2026 \; Live \; Matches:} & \quad \mathbf{UNSEEN \; PROSPECTIVE \; TEST \; SET}
\end{aligned}$$

---

## 8. Exact Next Step for Prospective Evaluation

1. Hook `v4_1_prospective_candidate_2025_26.pkl` into the prospective prediction service alongside `v4_0_draw_champion` to run in shadow prospective mode.
2. Ingest 2026 live fixtures causally before kickoff.
3. Compute and log real-time pre-match probabilities and status tags without modifying historical weights.
4. Reconcile results post-match to track live out-of-sample RPS, Log-Loss, and Draw ECE.
