# Market Odds Ingestion & Causal Audit Report

**Date:** 2026-08-20
**Phase:** 3 — Market Odds Ingestion & Causal Audit
**Scope:** Five European club leagues, seasons 2020/21–2024/25 (2025/26 quarantined)

---

## 1. API Source

**Provider:** OddAlerts
**Base URL:** `https://data.oddalerts.com/api`
**Authentication:** API token via `OddAlerts_API` environment variable / `.env` file
**Subscription:** Active (existing project subscription)

---

## 2. Endpoint Structure

### Primary endpoint: `GET /api/odds/history/:ID`

| Parameter | Required | Description |
|---|---|---|
| `:ID` | Yes | fixture_id (same as matches.db) |
| `api_token` | Yes | Authentication |
| `markets` | No | Filter by market_id (6 = ft_result) |
| `bookmakers` | No | Filter by bookmaker_id |

**Batch variant:** `GET /api/odds/history/multiple?ids=ID1,ID2,...` (limit: 50 per request)

**Response fields per record:** `fixture_id`, `market_key`, `market_id`, `outcome`, `opening` (str), `closing` (str), `peak` (str), `bookmaker_id`, `bookmaker_name`

**No pagination** observed — all records for a fixture returned in one response.

---

## 3. Five-League Coverage

### Eligible fixtures (excluding 2025/26):

| League | Competition ID | Fixtures | has_odds=1 |
|---|---|---|---|
| Ligue 1 | 200 | 1,446 | 1,446 |
| La Liga | 419 | 1,900 | 1,896 |
| Premier League | 423 | 1,900 | 1,900 |
| Bundesliga | 477 | 1,530 | 1,530 |
| Serie A | 499 | 1,901 | 1,899 |
| **Total** | | **8,677** | **8,671** |

Note: Fixture counts shown are for seasons 2020/21–2024/25 only. Ligue 1 reduced from 20 to 18 teams in 2023/24, hence fewer fixtures.

**has_odds coverage:** 8,671 / 8,677 = **99.93%** (6 fixtures missing odds flag)

---

## 4. Season Coverage

| Season | Fixtures | has_odds=1 |
|---|---|---|
| 2020/2021 | 1,826 | 1,825 |
| 2021/2022 | 1,826 | 1,826 |
| 2022/2023 | 1,827 | 1,827 |
| 2023/2024 | 1,752 | 1,747 |
| 2024/2025 | 1,752 | 1,752 |

No season falls below 99.7% `has_odds` coverage.

---

## 5. Actual Usable 1X2 Odds Coverage

**This section requires local execution of the ingestion pipeline.**

The `has_odds` flag confirms the API *should* have odds for 8,671 fixtures. However, actual usable 1X2 coverage — meaning a fixture has valid home/draw/away closing odds from at least one bookmaker — can only be determined after running:

```bash
python research/market_odds/ingest_odds_history.py
python research/market_odds/audit_and_build_dataset.py
```

The ingestion script:
- Fetches `odds/history` for all 8,671 fixtures with `has_odds=1`
- Filters to `market_id=6` (ft_result) to minimize response size
- Stores raw records in `research/market_odds/odds_history.sqlite`
- Supports resume (safe to interrupt and re-run)
- Respects rate limits with exponential backoff

The audit script then calculates actual usable coverage per league, per season, and pooled.

---

## 6. Bookmakers Found

**From sample data (fixture 50299569):**

| ID | Name | Type |
|---|---|---|
| 1 | Pinnacle | Sharp (preferred for de-vig) |
| 2 | Bet365 | Recreational |
| 3 | 1xBet | International |
| 4 | WilliamHill | UK major |

The full bookmaker roster and per-bookmaker coverage will be reported by `audit_and_build_dataset.py` after ingestion. The Postman collection confirms a `GET /api/bookmakers` endpoint exists for the complete list.

**Bookmaker selection strategy:** Pinnacle is preferred if coverage ≥ 90%. If insufficient, the bookmaker with highest fixture coverage is selected as fallback. The audit script handles this automatically.

---

## 7. Opening / Closing Availability

**From API structure:**
- Every `odds/history` record contains both `opening` and `closing` fields
- Both are string representations of decimal odds
- Both must be parsed to float and validated (> 1.0)
- `peak` is also available but **must not be used** (unknown timing)

**Actual availability rates** (opening-only, closing-only, both, neither) will be calculated by the audit script after ingestion.

---

## 8. Timestamp Availability

### odds/history endpoint:
**No per-record timestamp.** Fields are `opening`, `closing`, `peak` only — no `unix`, `datetime`, or any temporal field.

### odds/movement endpoint:
**Has timestamps** (`unix` + `datetime` per record) but **21-day retention only.** Historical data is automatically purged. The retention advisory explicitly states: *"Use the odds/history endpoint for permanent historical records (opening, closing, peak)."*

### Conclusion:
No independent timestamp verification is possible for historical odds. This is a known, documented limitation of the OddAlerts odds/history endpoint.

---

## 9. Causality Assessment

| Field | Classification | Risk Level |
|---|---|---|
| `opening` | **ASSUMED PRE-MATCH** | LOW — earliest recorded price is structurally pre-match |
| `closing` | **ASSUMED PRE-MATCH** | MODERATE — relies on OddAlerts' definition that "closing" = final pre-kickoff price; no independent timestamp to verify |
| `peak` | **UNKNOWN TIMING** | HIGH — could occur at any point in market life; **must NOT be used** |

### Justification for ASSUMED PRE-MATCH classification:

1. OddAlerts explicitly defines "opening" as earliest and "closing" as final pre-kickoff price
2. The odds/movement endpoint (which has timestamps) tracks pre-kickoff price changes, confirming the platform models pre-match vs in-play as distinct
3. The retention note directs users to odds/history for "permanent historical records (opening, closing, peak)" — implying these are the summary snapshots of pre-match market activity
4. Industry standard across all major odds providers: opening/closing = pre-match prices

### What this does NOT guarantee:

- Edge cases where kickoff was delayed and a late price was recorded as "closing"
- Whether any bookmaker's "closing" price was actually captured at kickoff rather than strictly before it
- Whether in-play price movements could contaminate the "peak" field

### Recommendation:

Use **closing odds** as the primary feature (most informative pre-match price, standard practice). Use **opening odds** for robustness checks. **Never use peak odds.** Accept that pre-kickoff timing relies on provider definition rather than independent verification.

---

## 10. Missing-Data Pattern

**Known gaps from `has_odds` flag (6 fixtures):**

| Fixture | Match | League | Season |
|---|---|---|---|
| 255268 | Hellas Verona vs Roma | Serie A | 2020/21 |
| 45213771 | Las Palmas vs Mallorca | La Liga | 2023/24 |
| 50846769 | Girona vs Las Palmas | La Liga | 2023/24 |
| 56297631 | Las Palmas vs Granada | La Liga | 2023/24 |
| 97911331 | Granada vs Las Palmas | La Liga | 2023/24 |
| 129529400 | Udinese vs Roma | Serie A | 2023/24 |

**Pattern:** 4/6 involve Las Palmas (newly promoted in 2023/24 — likely delayed bookmaker coverage at season start). Not systematic.

Additional missing-data patterns (per-bookmaker gaps, incomplete 1X2 sets, API failures) will be identified after ingestion.

---

## 11. Power-Method Implementation

**Module:** `research/market_odds/devig.py`

**Algorithm:** Bisection root-finding on:
```
f(k) = (1/O_H)^k + (1/O_D)^k + (1/O_A)^k - 1 = 0
```

**Properties:**
- Tolerance: 1e-12
- Max iterations: 200
- Bracket expansion: automatic (up to 20 doublings)
- Final normalization: probabilities divided by their sum to eliminate residual numerical error
- Output: `DevigResult(p_home, p_draw, p_away, k, overround, ...)`

**Validation gates:**
- All odds must be > 1.0 (decimal odds)
- All odds must be finite (rejects NaN, Inf)
- All odds must be non-None
- Output probabilities must be finite, positive, sum to 1
- Deterministic: identical inputs produce identical outputs

---

## 12. De-Vig Validation

**Test suite:** `research/market_odds/test_market_odds.py`, Suite 1

| Test | Result |
|---|---|
| Fair odds → same probabilities, sum=1 | **PASS** |
| Vigged odds → sum=1 after de-vig | **PASS** |
| All probabilities > 0 | **PASS** |
| All probabilities < 1 | **PASS** |
| k > 1 for vigged odds (overround > 1) | **PASS** |
| Reject odds ≤ 1.0 | **PASS** |
| Reject None odds | **PASS** |
| Reject NaN odds | **PASS** |
| Reject Inf odds | **PASS** |
| Extreme but valid odds handled | **PASS** |
| Deterministic repeated calculation | **PASS** |
| Overround calculated correctly | **PASS** |
| Symmetric odds → equal probabilities | **PASS** |
| Heavy favorite correctly reflected | **PASS** |

**14/14 PASS.** No scipy dependency required (pure Python bisection).

---

## 13. Research Dataset Location / Schema

**Path:** `research/market_odds/research_dataset.sqlite`
**Table:** `research_odds`

| Column | Type | Description |
|---|---|---|
| fixture_id | INTEGER PK | Matches matches.db |
| competition_id | INTEGER | League identifier |
| competition_name | TEXT | League name |
| season | TEXT | Season string |
| date | TEXT | Match date |
| unix | INTEGER | Kickoff timestamp |
| home_name | TEXT | Home team |
| away_name | TEXT | Away team |
| bookmaker_id | INTEGER | Selected bookmaker |
| bookmaker_name | TEXT | Bookmaker name |
| opening_home | REAL | Raw opening home odds |
| opening_draw | REAL | Raw opening draw odds |
| opening_away | REAL | Raw opening away odds |
| closing_home | REAL | Raw closing home odds |
| closing_draw | REAL | Raw closing draw odds |
| closing_away | REAL | Raw closing away odds |
| devig_opening_home | REAL | De-vigged opening P(Home) |
| devig_opening_draw | REAL | De-vigged opening P(Draw) |
| devig_opening_away | REAL | De-vigged opening P(Away) |
| devig_opening_k | REAL | Opening de-vig k parameter |
| devig_opening_overround | REAL | Opening raw overround |
| devig_closing_home | REAL | De-vigged closing P(Home) |
| devig_closing_draw | REAL | De-vigged closing P(Draw) |
| devig_closing_away | REAL | De-vigged closing P(Away) |
| devig_closing_k | REAL | Closing de-vig k parameter |
| devig_closing_overround | REAL | Closing raw overround |
| opening_causality | TEXT | "ASSUMED PRE-MATCH" |
| closing_causality | TEXT | "ASSUMED PRE-MATCH" |

**This dataset is NOT attached to the production feature matrix.** It is isolated under `research/market_odds/`.

---

## 14. Protected-File MD5 Results

| File | Expected MD5 | Actual MD5 | Status |
|---|---|---|---|
| `data/models/v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `data/models/v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `data/processed/features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `data/processed/matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

Verified pre- and post-phase. No production file was modified.

---

## 15. Files Created / Modified

### Created (all under `research/market_odds/`):

| File | Size | Purpose |
|---|---|---|
| `ODDS_API_AUDIT.md` | ~4 KB | Phase 3A API structure documentation |
| `ingest_odds_history.py` | ~9 KB | Phase 3B ingestion script |
| `devig.py` | ~5 KB | Phase 3G power-method de-vig module |
| `audit_and_build_dataset.py` | ~12 KB | Phase 3C-F audits + Phase 3I dataset builder |
| `test_market_odds.py` | ~10 KB | Phase 3K test suite (5 suites) |
| `MARKET_ODDS_INGESTION_REPORT.md` | This file | Phase 3L final report |

### Modified:

**NONE.** Zero production files were modified. No files outside `research/market_odds/` were created or changed.

---

## 16. Known Limitations

1. **API calls cannot be made from the sandbox.** The sandbox proxy blocks outbound HTTPS to `data.oddalerts.com` (HTTP 403). All ingestion and post-ingestion audits must be run locally.

2. **No independent timestamp verification.** The `odds/history` endpoint provides no per-record timestamps. Closing odds are classified as "ASSUMED PRE-MATCH" based on OddAlerts' definition, not independent verification.

3. **Full bookmaker roster unknown.** The sample shows 4 bookmakers; the complete list requires either a live `GET /api/bookmakers` call or inspection of the full ingested dataset.

4. **Batch endpoint path unverified.** The Postman documentation mentions batch fetching via `/odds/multiple` but the exact path format could not be tested. The ingestion script falls back to single-fixture calls if batch fails.

5. **Rate limits undocumented.** Conservative 1.5-second delays between batches are used as a precaution. Actual rate limits may allow faster ingestion.

6. **Peak odds timing unknown.** The `peak` field is stored for completeness but classified as UNKNOWN TIMING and excluded from the research dataset.

7. **6 fixtures have no odds.** These are excluded from the research dataset. This is a 0.07% gap and non-systematic.

---

## 17. Recommendation for E2

### Prerequisites before E2 can begin:

1. **Run the ingestion pipeline locally:**
   ```bash
   cd "E:\Football Prediction Project"
   python research/market_odds/ingest_odds_history.py
   ```
   This fetches odds for ~8,671 fixtures. Estimated time: 3-15 minutes depending on API rate limits.

2. **Run the audit and dataset builder:**
   ```bash
   python research/market_odds/audit_and_build_dataset.py
   ```
   This produces coverage statistics, bookmaker analysis, and the clean research dataset.

3. **Run the full test suite:**
   ```bash
   python research/market_odds/test_market_odds.py
   ```
   All 5 suites must pass (Suites 2-3 and 5 will no longer be skipped after ingestion).

4. **Review audit results** in `research/market_odds/audit_results.json` and the console output from the audit script. Key thresholds:
   - Actual usable 1X2 coverage should be ≥ 95% pooled
   - Preferred bookmaker (Pinnacle) coverage should be ≥ 90%
   - De-vig failure rate should be < 1%

---

## FINAL DECISION

### SAFE TO PROCEED TO E2 MARKET EXPERIMENT

**Conditional on:**
1. Local ingestion completes successfully (Step 1 above)
2. Actual usable 1X2 coverage ≥ 95% pooled (Step 2 above)
3. All 5 test suites pass with 0 failures (Step 3 above)
4. Protected file MD5s remain identical post-ingestion

**Evidence supporting this decision:**
- API source is confirmed, permanent, and covers 99.93% of eligible fixtures
- Opening and closing odds provide two causally defensible pre-match snapshots
- Power de-vig implementation passes all 14 validation tests
- Research dataset schema is complete and isolated from production
- Zero production files modified
- All work is under `research/market_odds/` — fully reversible

**What E2 will test:**
- V3 vs V3 + de-vigged closing market probabilities
- Using the same walk-forward validation protocol as the V3 experiment
- Market weight will be determined empirically, not hardcoded

---

## ROADMAP PROGRESS

| Phase | Description | Status |
|---|---|---|
| 1 | V3 Controlled Experiment | **COMPLETE** |
| 2 | Market Data Audit | **COMPLETE** |
| 3 | Market Odds Ingestion & Causal Audit | **COMPLETE** (conditional on local execution) |
| 4 (next) | E2 — V3 vs V3 + Market | NOT STARTED |

**4 / 14 phases complete** (Phase 3 conditional on local ingestion + test pass).

**Next Phase:** E2 — V3 vs V3 + Market

**Do NOT run E2 yet. Do NOT modify V3. Do NOT modify V2. Do NOT promote anything.**
