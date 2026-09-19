# Phase 15.1 — Live Score Integration Audit & Production Fix Report

**Dashboard Target:** `reports/upcoming_2026_09_18_to_2026_09_21_dashboard.html`  
**Backend Endpoint:** `https://web-production-d8a09.up.railway.app/api/live-scores`  
**Evaluation Status:** PRODUCTION VERIFIED & DEPLOYED  
**Model Invariant:** Frozen V4.0 Model (`SHA256: 1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`)  
**Static Predictions Hash:** `d179d13dd87d45d323619e3238425c41b3ec1faf69c650b295a555982d07de16` (100% Bit-Identical)

---

## 1. Root Cause Analysis: Fixture 420656757 Score Resolution

### The Issue
Fixture `420656757` (FC Bayern München vs FC Union Berlin, scheduled 2026-09-18 18:30 UTC) was completed in reality with a final score of **7-0**, but the deployed dashboard displayed:
- `Actual Score: —`
- `Status: UPCOMING`

### The Root Cause
1. **Backend Fixture ID Tracking Scope**:
   In `backend/main.py`, the backend loaded fixtures exclusively from `reports/upcoming_batch_2026_09_10_to_2026_09_16_ledger.jsonl` (the earlier 55-fixture batch from Sep 10–16) or fell back to `DEFAULT_55_FIXTURE_IDS`. The new Phase 15 upcoming batch (48 fixtures from 2026-09-18 to 2026-09-21 in `reports/upcoming_2026_09_18_to_2026_09_21_ledger.jsonl`), including fixture `420656757`, was **not in the backend's tracking list**.
2. **Upstream Request Omission**:
   When querying OddAlerts `/fixtures/multiple?ids=...`, the backend only passed the 55 old IDs. Fixture `420656757` was never requested from the provider.
3. **Response Omission & Frontend Fallback**:
   The response JSON omitted `420656757`. In the dashboard JavaScript, `liveScoresMap[fidStr]` evaluated to `undefined`, triggering the fallback:
   `status: 'UPCOMING'`, `score: '—'`.
4. **Missing Completed Matches Cache**:
   The backend had no persistent completed-scores storage. If an FT match leaves the active live feed, or is omitted during transient upstream glitches, the score was at risk of disappearing.

---

## 2. Implemented Architecture Fix

### Backend (`backend/main.py`)
1. **Multi-Batch Ledger Loading**:
   Updated `load_target_fixture_ids()` to scan `reports/` for all upcoming ledgers (`upcoming_2026_09_18_to_2026_09_21_ledger.jsonl` and `upcoming_batch_2026_09_10_to_2026_09_16_ledger.jsonl`), tracking all 103 fixtures across both batches.
2. **Persistent Completed Scores Cache (`_completed_cache`)**:
   - Maintains an in-memory and disk-backed cache (`data/completed_scores_cache.json`).
   - Pre-populates on startup from completed outcome records (`research/external_consensus/phase13/02_outcome_ledger.jsonl` and disk).
   - Once a fixture finishes (`status: FT/AET/PEN` with valid goals), it is recorded in `_completed_cache`.
   - **Critical Invariant**: A completed fixture in `_completed_cache` is **never downgraded** to `UPCOMING` or overwritten with `null`/`—` score, even if upstream feed omits it or reports `NS`.
3. **Dynamic Fixture Query Support**:
   `GET /api/live-scores?ids=...` allows client dashboards to dynamically specify fixture IDs while tracking all fixtures by default.
4. **Health Endpoints**:
   Defined both `/health` and `/api/health` returning HTTP 200, fixture tracking counts (103), and completed cache counts (59).

### Frontend Dashboard (`reports/` and `research/`)
1. **Client-Side Persistent Score Retention**:
   Added `window._persistedCompletedScores`. Once a fixture receives an `FT` score, the client retains it across all subsequent 30-second polling cycles.
2. **Defensive Score Merging**:
   Incoming polling data is merged with `window._persistedCompletedScores` so that any temporary omission from the network response never clears a finished score.
3. **Explicit Query Parameters**:
   The dashboard appends `?ids=${allFids}` when polling `/api/live-scores` as a secondary defense.

---

## 3. Production Deployment & Live Verification

### Railway Deployment
- **Repository Commit**: `4aa66dd` pushed to GitHub `origin/main`.
- **Railway Service**: `https://web-production-d8a09.up.railway.app` (single existing service, zero duplicate backends).
- **Deployment Status**: DEPLOYED & LIVE.

### Live Endpoint Verifications
- `GET /health` → **HTTP 200 OK**
  ```json
  {
    "status": "healthy",
    "service": "football-prediction-live-scores",
    "fixtures_tracked": 103,
    "completed_cached": 59,
    "cached": true
  }
  ```
- `GET /api/live-scores` → **HTTP 200 OK**
- **CORS Header**: `Access-Control-Allow-Origin: *` (Full Netlify client access permitted)
- **Target Fixture `420656757` Verified**:
  ```json
  {
    "fixture_id": 420656757,
    "status": "FT",
    "status_label": "FINISHED",
    "home_goals": 7,
    "away_goals": 0,
    "display_score": "7-0",
    "elapsed": 93,
    "time_added": 3,
    "is_finished": true
  }
  ```

---

## 4. Test Suite Summary

- **Phase 15.1 Live Integration Suite (`tests/test_phase15_live_score_integration.py`)**: 24 / 24 PASSED (100%)
- **Phase 15 Upcoming Dashboard Suite (`tests/test_phase15_upcoming_dashboard.py`)**: 20 / 20 PASSED (100%)
- **Production Regression Suite (`tests/test_production_2_0_pathway.py`, `tests/test_phase13_prospective_2_0.py`, `tests/test_phase12_2_0_forensics.py`)**: 52 / 52 PASSED (100%)
- **Total Tests Passing**: **96 / 96 PASSED**
