# Market Data Audit Report

**Date:** 2026-08-20
**Phase:** 2 — Market Data Audit
**Scope:** Five European club leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1)
**Purpose:** Determine whether the existing FPP database contains sufficiently complete and causally valid bookmaker odds data for a controlled experiment

---

## 1. Executive Summary

**Odds values have never been ingested into any local database.** The project's existing data pipeline stores only a boolean `has_odds` flag per fixture (99.94% coverage across the five leagues). Actual bookmaker odds — opening, closing, and peak decimal prices — are available via the OddAlerts `GET /api/odds/history/:fixture_id` endpoint and confirmed to be permanently retained (unlike the movement endpoint's 21-day window). However, this data has never been fetched or stored locally.

Before any market-odds experiment can proceed, an odds ingestion pipeline must be built to fetch, store, and validate ~10,729 fixture odds histories from the OddAlerts API.

**Decision: USABLE WITH RESTRICTIONS** — the data source exists, is permanent, has near-universal coverage, and provides opening/closing prices suitable for a controlled experiment. But the data must first be fetched and locally stored, and the lack of per-record timestamps on the history endpoint means pre-kickoff causality must be established by definition (opening/closing are defined as pre-kickoff by OddAlerts) rather than by independent timestamp verification.

---

## 2. Five-League Scope Definition

| League | Competition ID | Country | Seasons | Fixtures |
|---|---|---|---|---|
| Premier League | 423 | England | 6 (2020/21–2025/26) | 2,280 |
| La Liga | 419 | Spain | 6 (2020/21–2025/26) | 2,280 |
| Serie A | 499 | Italy | 6 (2020/21–2025/26) | 2,281 |
| Bundesliga | 477 | Germany | 6 (2020/21–2025/26) | 1,836 |
| Ligue 1 | 200 | France | 6 (2020/21–2025/26) | 2,058 |
| **Total** | | | **30 league-seasons** | **10,735** |

All analysis in this report is restricted to these five leagues. No international, cup, or other competition data was examined.

---

## 3. Database Schema

### 3a. Odds-related columns across all databases

| Database | Table | Odds-Related Columns |
|---|---|---|
| `data/processed/matches.db` | `fixtures` | `has_odds` (INTEGER, 0/1 boolean) |
| `data/processed/features.db` | `feature_rows` | **NONE** |
| `data/processed/leagues/*.db` | `fixtures` | `has_odds` (INTEGER, 0/1 boolean) |
| `data/processed/leagues/*.db` | `feature_rows` | **NONE** |
| `data/processed/features_v1_1.db` | `feature_rows` | **NONE** |

### 3b. What is NOT stored

No database in the project contains any of the following:

- Decimal odds (home/draw/away)
- Implied probabilities
- Bookmaker identifiers
- Market type identifiers
- Opening/closing/peak prices
- Odds timestamps
- Odds movement history
- De-vigged probabilities

The `has_odds` field is the sole odds-related data point, indicating whether the OddAlerts API reported that odds existed for that fixture at ingestion time.

---

## 4. Odds Data Source

**Source:** OddAlerts API (`https://data.oddalerts.com/api`)
**Subscription:** Active (existing project API token)

### Relevant endpoints:

| Endpoint | Purpose | Retention | Status in Project |
|---|---|---|---|
| `GET /fixtures/between` | Fixture data with `has_odds` flag | Permanent | **Ingested** (flag only) |
| `GET /odds/history/:ID` | Opening/closing/peak per bookmaker per market | **Permanent** | **Never called** |
| `GET /odds/movement/:ID` | Timestamped price changes | **21 days only** | Never called |
| `GET /odds/markets` | List available market types | N/A | Never called |
| `GET /bookmakers` | List available bookmakers | N/A | Never called |

### `odds/history` record structure (from `data/audit/odds_history_example.json`):

```
fixture_id      : int     (links to fixtures table)
market_key      : str     (e.g. "ft_result" for 1X2)
market_id       : int     (e.g. 6)
outcome         : str     ("home" / "draw" / "away")
opening         : str     (decimal odds, e.g. "2.63")
closing         : str     (decimal odds, e.g. "2.90")
peak            : str     (highest price seen, e.g. "2.90")
bookmaker_id    : int     (e.g. 1 = Pinnacle, 2 = Bet365)
bookmaker_name  : str
```

Each record represents ONE outcome for ONE bookmaker for ONE market. A complete 1X2 set for one bookmaker requires 3 records (home + draw + away).

---

## 5. Coverage by League

### `has_odds` flag coverage (from matches.db):

| League | Total Fixtures | With Odds | Without Odds | Coverage |
|---|---|---|---|---|
| Bundesliga | 1,836 | 1,836 | 0 | **100.00%** |
| La Liga | 2,280 | 2,276 | 4 | **99.82%** |
| Ligue 1 | 2,058 | 2,058 | 0 | **100.00%** |
| Premier League | 2,280 | 2,280 | 0 | **100.00%** |
| Serie A | 2,281 | 2,279 | 2 | **99.91%** |
| **Pooled** | **10,735** | **10,729** | **6** | **99.94%** |

### Fixtures without odds (6 total):

| Fixture ID | Match | League | Season | Date |
|---|---|---|---|---|
| 255268 | Hellas Verona vs Roma | Serie A | 2020/21 | 2020-09-19 |
| 45213771 | Las Palmas vs Mallorca | La Liga | 2023/24 | 2023-08-12 |
| 50846769 | Girona vs Las Palmas | La Liga | 2023/24 | 2023-09-03 |
| 56297631 | Las Palmas vs Granada | La Liga | 2023/24 | 2023-09-24 |
| 97911331 | Granada vs Las Palmas | La Liga | 2023/24 | 2024-02-03 |
| 129529400 | Udinese vs Roma | Serie A | 2023/24 | 2024-04-25 |

Note: 4/6 involve Las Palmas (promoted to La Liga in 2023/24). This is a minor, non-systematic gap.

---

## 6. Coverage by Season

| League | Season | Fixtures | Has Odds | Coverage |
|---|---|---|---|---|
| Bundesliga | 2020/21 | 306 | 306 | 100.0% |
| Bundesliga | 2021/22 | 306 | 306 | 100.0% |
| Bundesliga | 2022/23 | 306 | 306 | 100.0% |
| Bundesliga | 2023/24 | 306 | 306 | 100.0% |
| Bundesliga | 2024/25 | 306 | 306 | 100.0% |
| Bundesliga | 2025/26 | 306 | 306 | 100.0% |
| La Liga | 2020/21 | 380 | 380 | 100.0% |
| La Liga | 2021/22 | 380 | 380 | 100.0% |
| La Liga | 2022/23 | 380 | 380 | 100.0% |
| La Liga | 2023/24 | 380 | 376 | 98.9% |
| La Liga | 2024/25 | 380 | 380 | 100.0% |
| La Liga | 2025/26 | 380 | 380 | 100.0% |
| Ligue 1 | 2020/21 | 380 | 380 | 100.0% |
| Ligue 1 | 2021/22 | 380 | 380 | 100.0% |
| Ligue 1 | 2022/23 | 380 | 380 | 100.0% |
| Ligue 1 | 2023/24 | 306 | 306 | 100.0% |
| Ligue 1 | 2024/25 | 306 | 306 | 100.0% |
| Ligue 1 | 2025/26 | 306 | 306 | 100.0% |
| Premier League | 2020/21 | 380 | 380 | 100.0% |
| Premier League | 2021/22 | 380 | 380 | 100.0% |
| Premier League | 2022/23 | 380 | 380 | 100.0% |
| Premier League | 2023/24 | 380 | 380 | 100.0% |
| Premier League | 2024/25 | 380 | 380 | 100.0% |
| Premier League | 2025/26 | 380 | 380 | 100.0% |
| Serie A | 2020/21 | 380 | 379 | 99.7% |
| Serie A | 2021/22 | 380 | 380 | 100.0% |
| Serie A | 2022/23 | 381 | 381 | 100.0% |
| Serie A | 2023/24 | 380 | 379 | 99.7% |
| Serie A | 2024/25 | 380 | 380 | 100.0% |
| Serie A | 2025/26 | 380 | 380 | 100.0% |

No season in any league falls below 98.9% coverage.

---

## 7. Bookmaker / Source Distribution

### Known bookmakers (from `odds_history_example.json` sample):

| Bookmaker ID | Name | Notes |
|---|---|---|
| 1 | Pinnacle | Widely considered the sharpest bookmaker |
| 2 | Bet365 | Major recreational bookmaker |
| 3 | 1xBet | High-volume international bookmaker |
| 4 | WilliamHill | Major UK bookmaker |

**Limitation:** The sample file was truncated (4 of 314 records for a single fixture). The full bookmaker roster cannot be determined without a live API call to `GET /bookmakers`. The sample confirms at least 4 bookmakers including Pinnacle (the preferred sharp line for de-vigging).

### Per-bookmaker coverage analysis:

**Cannot be performed.** Actual odds data has not been ingested. The `has_odds` flag is per-fixture, not per-bookmaker. Full per-bookmaker coverage can only be assessed after the odds ingestion pipeline is built and run.

---

## 8. Timestamp Availability

### `odds/history` endpoint:
- **No per-record timestamp field.** Records contain only `opening`, `closing`, and `peak` price values.
- `opening` = earliest recorded price for that bookmaker/market/outcome
- `closing` = final price before kickoff
- `peak` = highest price seen during the market's life
- No `created_at`, `updated_at`, `unix`, or any temporal field exists on individual records.

### `odds/movement` endpoint:
- **Has timestamps** (timestamped price changes)
- **21-day retention only** — data is automatically purged after 3 weeks
- The retention note explicitly states: *"Use the odds/history endpoint for permanent historical records (opening, closing, peak)."*
- This means timestamped odds movement data is **not available for historical fixtures** beyond the last 21 days.

### Implication:
There is no way to independently verify the exact timestamp of opening or closing prices for historical fixtures. Causality must be established by definition rather than by timestamp comparison.

---

## 9. Pre-Kickoff vs Post-Kickoff Classification

### Based on endpoint design:

| Category | Classification | Evidence |
|---|---|---|
| `opening` price | **A. Assumed pre-kickoff** | By definition: earliest recorded pre-match price |
| `closing` price | **A. Assumed pre-kickoff** | By definition: final price before kickoff |
| `peak` price | **D. Unknown timing** | Could occur at any point in the market's life |

### Assessment:
- The `odds/history` endpoint is explicitly documented as providing **pre-match** opening and closing prices
- The separate `odds/movement` endpoint (which would have exact timestamps) is only retained for 21 days, making independent timestamp verification impossible for historical fixtures
- The OddAlerts platform's own retention note directs users to `odds/history` for permanent historical data, implying these are the intended long-term records

### Risk:
We cannot independently verify that "closing" prices were recorded before kickoff vs. at kickoff vs. shortly after. We must trust OddAlerts' definition. However, this is standard industry practice — opening/closing odds from reputable data providers are universally understood to be pre-match prices.

---

## 10. Opening / Closing Snapshot Analysis

### What the API provides:

| Field | Description | Available | Suitable for Model |
|---|---|---|---|
| `opening` | Earliest recorded price | Yes | Yes (most conservative, earliest available) |
| `closing` | Final pre-kickoff price | Yes | Yes (most informed, but closest to kickoff) |
| `peak` | Highest price seen | Yes | **No** (timing unknown, could be post-lineup) |

### Snapshot structure:
- **Two snapshots per bookmaker per outcome:** opening and closing
- **No intermediate snapshots** (those would be in `odds/movement`, which is purged after 21 days)
- Opening and closing can be ordered chronologically by definition (opening comes first)

### Recommendation for future experiment:
- **Closing odds** are the standard choice for prediction models (most informative pre-match price)
- **Opening odds** provide a robustness check (less susceptible to late-information contamination such as lineup leaks)
- **Peak odds** should not be used (no temporal guarantee)

---

## 11. Missingness Analysis

### Cannot be fully performed — actual odds values have not been ingested.

**What we know from `has_odds` flag:**
- 6 of 10,735 fixtures (0.06%) have `has_odds=0`
- 4 of those 6 involve Las Palmas in La Liga 2023/24 (likely a newly promoted team with delayed bookmaker coverage at season start)
- 1 is Hellas Verona vs Roma in Serie A 2020/21 opening weekend
- 1 is Udinese vs Roma in Serie A 2023/24
- Missingness is NOT random — it clusters around specific teams/dates

**What we cannot assess until ingestion:**
- Per-bookmaker missingness (e.g., does Pinnacle cover all 10,729 fixtures?)
- Per-outcome missingness (e.g., are there fixtures with home+away odds but missing draw odds?)
- Whether any `has_odds=1` fixture actually returns empty data from the API
- Structural gaps in specific bookmaker coverage for specific leagues/seasons

---

## 12. Duplicate / Alignment Audit

### Cannot be performed — actual odds data has not been ingested.

**What we can verify from existing data:**
- No duplicate `fixture_id` values in `matches.db` (structurally enforced)
- All 10,735 fixtures have valid `fixture_id`, `home_id`, `away_id`, `unix` timestamp
- All 10,729 `has_odds=1` fixtures have valid kickoff timestamps for post-ingestion causality checks

**What must be checked after ingestion:**
- Duplicate bookmaker records for same fixture/outcome
- Conflicting odds (e.g., two different closing prices for same bookmaker/outcome)
- Impossible odds values (≤ 1.0 for decimal odds, non-positive, non-numeric)
- Fixture ID alignment between odds API response and matches.db

---

## 13. Post-Kickoff Contamination Audit

### Current state: No contamination risk — no odds data has been ingested or used.

The existing feature pipeline (`src/features/`) does not reference odds in any form. The V1, V2, and V3 models use zero odds-derived features. The `has_odds` flag is stored but never used as a model feature.

### Future risk assessment:
When odds are eventually ingested, post-kickoff contamination must be tested by comparing:
- `closing` price timestamp (not available — see Section 9)
- `unix` kickoff timestamp from `matches.db` (available for all fixtures)

Since exact closing timestamps are not available from `odds/history`, the contamination test will need to rely on:
1. OddAlerts' definition that "closing" = pre-kickoff
2. Spot-checking with `odds/movement` for any fixtures within the 21-day retention window
3. Statistical anomaly detection (e.g., closing odds that imply knowledge of the result)

---

## 14. Data Usability Decision

### Classification: **B. USABLE WITH RESTRICTIONS**

### Evidence:

**Strengths:**
- Near-universal coverage: 10,729/10,735 fixtures (99.94%) have `has_odds=1`
- Permanent retention on `odds/history` endpoint (confirmed by OddAlerts)
- Both opening and closing prices available (two pre-match snapshots)
- Multiple bookmakers available including Pinnacle (sharp line)
- The `ft_result` market (1X2) is confirmed present
- Existing OddAlerts API subscription is active
- All 10,735 fixtures have valid kickoff timestamps for causality cross-reference

**Restrictions:**
1. **Odds data has not been ingested.** An ingestion pipeline must be built before any experiment. This requires ~10,729 API calls (one per fixture) to `odds/history`.
2. **No per-record timestamps.** Pre-kickoff causality relies on OddAlerts' definition of "opening" and "closing" rather than independent timestamp verification.
3. **Per-bookmaker coverage is unknown.** Cannot guarantee Pinnacle (or any single bookmaker) covers all fixtures until data is fetched.
4. **No intermediate snapshots.** Only opening and closing are available; odds movement (timestamped) is purged after 21 days.
5. **6 fixtures have no odds at all.** These must be handled as missing values in any experiment.
6. **API rate limits unknown.** The ingestion of ~10,729 fixtures needs to respect whatever rate limits OddAlerts imposes.

### Why not READY:
The data source is confirmed and coverage is excellent, but no actual odds values exist locally. We cannot run a controlled experiment on data we haven't fetched yet. The ingestion step is a prerequisite.

### Why not INSUFFICIENT:
Coverage is 99.94%, the source is permanent, the bookmaker roster includes Pinnacle, and the required market (1X2) is confirmed. This is far above the threshold for a meaningful experiment.

### Why not REJECTED:
While independent timestamp verification is not possible, the odds/history endpoint is specifically designed to provide pre-match snapshots. This is the standard data format used across the football analytics industry. The risk of systematic post-kickoff contamination in an endpoint explicitly labeled "opening" and "closing" is very low.

---

## 15. Known Limitations

1. **No local odds data exists.** This audit assessed the data source, not data in hand.
2. **API call budget unknown.** Fetching odds for 10,729 fixtures requires understanding rate limits and pagination.
3. **Bookmaker roster incomplete.** The sample shows 4 bookmakers; the full list requires a `GET /bookmakers` call.
4. **No independent timestamp verification.** Must trust OddAlerts' opening/closing definitions.
5. **314 records per fixture (sample).** If this scales linearly, ~3.4M records would need to be processed. Filtering to `ft_result` market + Pinnacle would dramatically reduce this.
6. **`peak` field timing is ambiguous.** Should not be used as a model feature without further investigation.

---

## 16. MD5 Integrity Results

| File | Pre-Audit MD5 | Post-Audit MD5 | Status |
|---|---|---|---|
| `data/models/v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `data/models/v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `data/processed/features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `data/processed/matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

No database was modified. No model artifact was modified. No source code was modified.

---

## 17. Git / Source Changes

**Zero files changed.** This audit was entirely read-only. The only file created is this report.

---

## 18. Recommendation for Phase 3

### Prerequisite: Build Odds Ingestion Pipeline

Before any controlled experiment, the following must be completed:

1. **Extend `api_client.py`** with a method for `GET /odds/history/:fixture_id` (filtered to `market_key=ft_result`)
2. **Create an odds storage table** in `matches.db` or a new `odds.db` with columns: `fixture_id`, `bookmaker_id`, `bookmaker_name`, `outcome`, `opening`, `closing`, `peak`
3. **Fetch odds for all 10,729 fixtures** with `has_odds=1`, respecting API rate limits
4. **Run post-ingestion validation:** per-bookmaker coverage, 1X2 completeness, impossible values, fixture alignment
5. **Identify the preferred bookmaker** (Pinnacle recommended for sharpness, if coverage is sufficient)
6. **Compute de-vigged implied probabilities** from the closing 1X2 prices using the power method
7. **Re-run the Phase 2.7–2.9 analyses** (missingness, duplicates, contamination) on the actual ingested data

Only after this pipeline is built and validated should the Phase 3 controlled experiment proceed.

### Estimated scope:
- API calls: ~10,729 (one per fixture)
- Storage: ~30K–100K records (3 outcomes × N bookmakers × 10,729 fixtures, filtered to ft_result)
- New code: API endpoint wrapper, odds storage module, odds validation script
- Risk: API rate limits may require batching over multiple sessions

---

## ROADMAP PROGRESS

**3 / 14 phases complete**

**Current Phase:** Phase 2 — Market Data Audit

**Decision:** **USABLE WITH RESTRICTIONS**

The OddAlerts API provides permanently retained opening/closing 1X2 odds from multiple bookmakers (including Pinnacle) with 99.94% fixture coverage across all five target leagues and all six seasons. However, this data has never been fetched or stored locally. An odds ingestion pipeline is a required prerequisite before any controlled market-odds experiment can proceed.
