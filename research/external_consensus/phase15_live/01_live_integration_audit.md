# Phase 15.1 — Live Score Integration Audit & Architecture Report

**Dashboard Target:** `reports/upcoming_2026_09_18_to_2026_09_21_dashboard.html`  
**Backend Endpoint:** `https://web-production-d8a09.up.railway.app/api/live-scores`  
**Evaluation Status:** PRODUCTION VERIFIED  
**Model Invariant:** Frozen V4.0 Model (`SHA256: 1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`)  

---

## 1. Executive Summary

Phase 15.1 connects the static Phase 15 upcoming prediction dashboard (Big-5 European matchweek from 18 Sep 2026 to 21 Sep 2026) to the **existing, live-deployed Railway backend** at `https://web-production-d8a09.up.railway.app/api/live-scores` to receive automated, real-time live score updates every 30 seconds.

### Core Architectural Guarantees:
1. **Zero New Backend Services**: Reuses 100% of the deployed Railway FastAPI service (`backend/main.py`). Zero secondary deployments or duplicate infrastructure created.
2. **Strict Prediction Immutability**: All 48 fixtures' pre-kickoff predictions ($P(H), P(D), P(A)$, $\text{PRED}$, $\text{Predicted Score}$, $\text{Draw Risk}$, $\text{Signal Profile}$) remain completely immutable in the client DOM. Live scores update **only** the `Actual Score` and `Match Status` columns.
3. **Primary Key Fixture Matching**: Matches live records strictly via numeric `fixture_id`. Secondary fallback (date + home_team + away_team) is only engaged if fixture ID is omitted.
4. **Resilient Fail-Open Polling**: If the live API experiences network timeouts, transient 5xx errors, or CORS issues, the dashboard continues operating in static mode with zero visual disruption.
5. **No Secret Leaks**: No API tokens, keys, or credentials exist in the client HTML or JavaScript. OddAlerts upstream credentials remain protected within the Railway environment.

---

## 2. Upstream Backend & Endpoint Audit

### Health Endpoint Check
- **URL**: `https://web-production-d8a09.up.railway.app/api/health`
- **Response**: HTTP 200 OK
- **Payload Verified**:
```json
{
  "status": "healthy",
  "service": "football-prediction-live-scores",
  "timestamp_utc": "2026-09-19T12:16:15.583676+00:00",
  "fixtures_tracked": 55,
  "cached": true,
  "cache_age_seconds": 48.0
}
```

### Live-Scores Endpoint Check
- **URL**: `https://web-production-d8a09.up.railway.app/api/live-scores`
- **Response**: HTTP 200 OK
- **CORS Header**: `Access-Control-Allow-Origin: *` (Full browser client access permitted)
- **Top-Level Keys**: `["last_updated_utc", "active_fixtures_count", "cached", "scores"]`
- **Item Schema**:
```json
{
  "fixture_id": 420629293,
  "status": "FT",
  "status_label": "FINISHED",
  "home_goals": 1,
  "away_goals": 3,
  "display_score": "1-3",
  "elapsed": 103,
  "time_added": 12,
  "is_finished": true
}
```

---

## 3. Status Mapping & Normalization Rules

| Provider Status | Normalized Status | Dashboard Display Tag | Actual Score Display | Row CSS Class |
|---|---|---|---|---|
| `NS`, `TIMED`, `SCHEDULED` | `UPCOMING` | `🔵 UPCOMING` | `—` | *(default)* |
| `LIVE` (with elapsed) | `LIVE` | `🔴 LIVE {elapsed}'` | `{home_goals} - {away_goals}` | `row-live` |
| `LIVE` (with added time) | `LIVE` | `🔴 LIVE {elapsed}+{time_added}'` | `{home_goals} - {away_goals}` | `row-live` |
| `LIVE` (no elapsed) | `LIVE` | `🔴 LIVE` | `{home_goals} - {away_goals}` | `row-live` |
| `HT` | `HT` | `🟡 HALF TIME` | `{home_goals} - {away_goals}` | `row-ht` |
| `FT` | `FINISHED` | `⚪ FT` | `{home_goals} - {away_goals}` | `row-finished` |
| `AET` | `FINISHED` | `⚪ FT AET` | `{home_goals} - {away_goals}` | `row-finished` |
| `PEN` | `FINISHED` | `⚪ FT PEN` | `{home_goals} - {away_goals}` | `row-finished` |
| `POSTP` | `POSTP` | `POSTPONED` | `—` | *(default)* |
| `CANC` | `CANC` | `CANCELLED` | `—` | *(default)* |
| `SUSP` | `SUSP` | `SUSPENDED` | `{display_score}` | *(default)* |

---

## 4. Connection Lifecycle & Indicators

1. **Initial Load**:
   - Connection indicator displays: `🟡 RECONNECTING…` / `INITIALIZING LIVE FEED…`.
   - Initial HTTP `fetch(LIVE_SCORES_API_URL)` fires immediately on `DOMContentLoaded`.
2. **Successful Poll**:
   - Indicator updates to: `🟢 LIVE DATA CONNECTED | LAST UPDATED: HH:MM:SS UTC`.
   - `consecutiveFailures` reset to 0.
   - Live scores applied via DOM updates.
3. **Transient Failure (1 or 2 failed requests)**:
   - Indicator displays: `🟡 RECONNECTING… | LAST UPDATED: HH:MM:SS UTC`.
   - Existing prediction rows and last-known scores remain untouched.
4. **Persistent Failure (>= 3 consecutive failed requests)**:
   - Indicator displays: `🔴 LIVE DATA OFFLINE | LAST UPDATED: HH:MM:SS UTC`.
   - Static predictions continue to display without interruption.
5. **Cadence**:
   - Standard 30-second interval via `setInterval(pollLiveScores, 30000)`.

---

## 5. Security & Invariant Audit

- **Model Frozen**: `v4_poisson_venue_elo_online_ad.pkl` SHA256 verified identical (`1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`).
- **Source Code Clean**: `git status --short src/ data/models/` verified completely clean.
- **Frontend Token Isolation**: Zero occurrences of `OddAlerts_API` or `api_token` in HTML or JavaScript.
