# Phase 14 Final Report: Production 2-0 / Strong Home Pathway Fix

**Status**: CERTIFIED & PRODUCTION READY  
**Scope**: Production Engineering Bug Fix & Pipeline Hardening  
**Target Model**: Frozen V4.0 Production Model (`v4_poisson_venue_elo_online_ad.pkl`)  
**Verified SHA256**: `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`  

---

## 1. Executive Summary & Objective

Following the forensic analysis in Phase 12 and prospective shadow validation in Phase 13, this engineering phase safely fixed the **Production Prediction Pathway**.

### The Problem
The frozen V4.0 model was never broken or depleted of `2-0` predictions. Instead, legitimate V4 `2-0` outputs had vanished from recorded production ledgers due to four downstream presentation and data-pipeline bugs:
1. **Simplified Batch Ledger Generator**: The script used to record `reports/upcoming_batch_2026_09_10_to_2026_09_16_ledger.jsonl` bypassed model score selection and uniformly mapped all HOME predictions to `2-1` and all AWAY predictions to `1-2`.
2. **Nearest-Integer Rounding Distortion**: In `fixture_service.py` and `app.py`, scorelines were synthesized using `int(round(lambda))`. For away rates $\lambda_a \in [0.50, 1.00)$ (e.g. $\lambda_a = 0.73$), `round(0.73) = 1`, turning pure mathematical `2-0` modes into `2-1`.
3. **Performance Monitor Rule**: In `performance_monitor_service.py`, a heuristic `base_score = 2-1 if p_h < 0.60 else 2-0` was used rather than inspecting model parameters.
4. **Missing Signal Profiles**: Production snapshots stored probabilities and expected goals but lacked typed signal classification tags (`strong_home_profile`, `is_2_0_profile`, `signal_profile`).

### The Solution
We implemented an end-to-end production fix ensuring that:
- The authoritative mathematical bivariate Poisson mode (`src/models/poisson.py:modal_scoreline`) is the sole scoreline truth.
- Zero scoreline flattening or synthetic rounding occurs anywhere in the pipeline.
- Production snapshots and dashboard predictions explicitly carry canonical scorelines and signal classification flags.
- A standardized `BatchPredictionService` produces verifiable pre-kickoff batch ledgers.
- The frozen V4 model binary remains **100% bit-identical and untouched**.

---

## 2. Invariant & Governance Compliance

| Invariant | Requirement | Status | Details |
|---|---|---|---|
| **V4 Model SHA256** | `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5` | **VERIFIED** | Bit-identical check enforced at startup and in test suites |
| **Code Isolation** | `src/models/` and `data/models/` untouched | **VERIFIED** | 0 model files modified; 0 parameters retrained |
| **Underlying Probabilities** | Pure V4 probabilities unmodified | **VERIFIED** | $P(H), P(D), P(A)$ identical to frozen inference |
| **Selection Volume** | Natural operating volume preserved | **VERIFIED** | No artificial threshold lowering or volume quotas forced |
| **Snapshot Compatibility** | Seamless backward compatibility | **VERIFIED** | `__post_init__` derives missing fields on load |

---

## 3. Benchmark Fixture Scoreline Audit

We audited six representative benchmark fixtures across three European leagues comparing the legacy flattened/distorted scorelines against the canonical restored pathway:

| Fixture ID | Matchup | League | $P(H)$ | $\lambda_h - \lambda_a$ | Legacy Batch | Legacy UI | Canonical Mode | Restored Signal Profile |
|---|---|---|---|---|---|---|---|---|
| `420637600` | **LOSC Lille vs Troyes** | Ligue 1 | 74.9% | 2.24 - 0.74 | `2-1` (Flattened) | `2-1` (Rounded) | **`2-0`** | `2-0_PROFILE` |
| `420644693` | **FC Barcelona vs Racing Santander** | La Liga | 63.8% | 2.06 - 0.92 | `2-1` (Flattened) | `2-1` (Rounded) | **`2-0`** | `2-0_PROFILE` |
| `420629290` | **Bayern München vs Union Berlin** | Bundesliga | 78.8% | 2.70 - 0.93 | `2-1` (Flattened) | `2-1` (Rounded) | **`2-0`** | `2-0_PROFILE` |
| `420629294` | **Inter vs Udinese** | Serie A | 65.6% | 2.02 - 0.94 | `2-1` (Flattened) | `2-1` (Rounded) | **`2-0`** | `2-0_PROFILE` |
| `420629288` | **RB Leipzig vs Hamburger SV** | Bundesliga | 61.8% | 2.00 - 1.02 | `2-1` | `2-1` | **`2-1`** | `STRONG_HOME_PROFILE` |
| `420629291` | **Como vs Parma** | Serie A | 65.3% | 1.89 - 0.81 | `2-1` (Flattened) | `2-1` (Rounded) | **`1-0`** | `STRONG_HOME_PROFILE` |

### Key Findings
1. **Lille vs Troyes**, **Barcelona vs Racing**, **Bayern vs Union**, and **Inter vs Udinese** are mathematically genuine `2-0` modal predictions under V4.0 ($\lfloor \lambda_h \rfloor = 2, \lfloor \lambda_a \rfloor = 0$). In the legacy ledger and dashboard, all four were erroneously displayed as `2-1`.
2. **RB Leipzig vs Hamburger SV** has $\lambda_a = 1.024 > 1.000$, making $\lfloor \lambda_a \rfloor = 1$. The canonical scoreline is honestly `2-1`, properly labeled as `STRONG_HOME_PROFILE` rather than forced into `2-0`.
3. **Como vs Parma** has $\lambda_h = 1.893 < 2.000$ and $\lambda_a = 0.812 < 1.000$. Its true bivariate mode is `1-0`, properly labeled as `STRONG_HOME_PROFILE`.

---

## 4. Architectural Summary of Code Changes

1. **`src/dashboard/prediction_snapshot_store.py`**:
   - Added `canonical_predicted_score`, `predicted_score`, `modal_scoreline_probability`, `strong_home_profile`, `is_2_0_profile`, and `signal_profile`.
   - Added `__post_init__` for automatic derivation from existing $\lambda_h, \lambda_a$ parameters.
2. **`src/dashboard/prediction_service.py`**:
   - Integrated `modal_scoreline` and `_grid_size` from `models.poisson`.
   - Enriched `SingleMatchPredictionResult` and `DashboardMatchPrediction` dataclasses.
   - Updated `predict_matchup` and `predict_dashboard_fixture` (Path A and Path B) to compute and propagate canonical modes and signal profiles.
   - Added `get_prediction_service()` singleton factory.
3. **`src/dashboard/fixture_service.py`**:
   - Eliminated nearest-integer rounding blocks.
   - Propagates canonical scores and signal profiles into evaluation records and UI tables.
4. **`src/dashboard/app.py`**:
   - Updated Streamlit presentation layer to display canonical scores and signal badges (`🟢 2-0 Signal` / `🔵 Strong Home`).
5. **`src/dashboard/performance_monitor_service.py`**:
   - Replaced static baseline heuristic with canonical scoreline inspection and modal fallback.
6. **`src/dashboard/batch_prediction_service.py`**:
   - Standardized batch generation engine with mandatory SHA256 model verification and zero scoreline flattening.

---

## 5. Verification & Test Suite

The fix is validated by a dedicated 20-test suite in `tests/test_production_2_0_pathway.py`:
- Test 1: V4 model SHA256 immutability (`1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`).
- Tests 2–5: Bivariate Poisson mode properties across home-heavy, balanced draw, away-heavy, and extreme home parameter spaces.
- Tests 6–8: Snapshot store `__post_init__` derivation and signal profile rules.
- Tests 9–11: Single match prediction contract and Lille/Barcelona benchmark reproduction.
- Tests 12–14: Dashboard fixture prediction (Path A and Path B) and batch ledger generation.
- Tests 15–17: Elimination of rounding distortion in performance monitoring, fixture service, and shadow batch validation.
- Tests 18–20: Consistency report verification, deliverable completeness, and zero secret leaks.

---

## 6. Production Recommendation & Next Steps

1. **Deploy Pathway Fix**: The updated dashboard and batch prediction services are certified for immediate production use.
2. **Future Batch Generation**: All future batch ledgers must be generated using `BatchPredictionService` to prevent legacy flattening routines from recurring.
3. **Dashboard Monitoring**: The dashboard table now cleanly distinguishes `🟢 2-0 Signal` from `🔵 Strong Home`, giving operators transparent visibility into model certainty.
