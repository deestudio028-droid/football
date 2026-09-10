# Phase 3A — Odds API Structure Audit

**Date:** 2026-08-20
**Source:** OddAlerts Postman Collection + existing sample files

---

## Endpoint

`GET https://data.oddalerts.com/api/odds/history/:ID`

- `:ID` = fixture_id (same integer used in matches.db `fixtures.fixture_id`)
- Batch variant: `GET /api/odds/history/multiple?ids=ID1,ID2,...` (limit: 50 per request)

## Request Parameters

| Parameter | Required | Description |
|---|---|---|
| `api_token` | Yes | API authentication token |
| `markets` | No | Filter by market_id (e.g., `6` for ft_result) |
| `bookmakers` | No | Filter by bookmaker_id (e.g., `1` for Pinnacle) |

## Response Structure

```json
{
  "info": { "count": 314 },
  "data": [
    {
      "fixture_id": 50299569,
      "market_key": "ft_result",
      "market_id": 6,
      "outcome": "away",
      "opening": "2.63",
      "closing": "2.90",
      "peak": "2.90",
      "bookmaker_id": 2,
      "bookmaker_name": "Bet365"
    }
  ]
}
```

## Field Documentation

| Field | Type | Description |
|---|---|---|
| `fixture_id` | int | Matches `fixtures.fixture_id` in matches.db |
| `market_key` | str | Market identifier string (e.g., "ft_result") |
| `market_id` | int | Numeric market identifier (6 = Full-Time Result / 1X2) |
| `outcome` | str | "home", "draw", or "away" for ft_result |
| `opening` | str | Earliest recorded decimal odds (string format) |
| `closing` | str | Final pre-kickoff decimal odds (string format) |
| `peak` | str | Highest decimal odds seen during market life (string format) |
| `bookmaker_id` | int | Numeric bookmaker identifier |
| `bookmaker_name` | str | Human-readable bookmaker name |

## Fixture ID Mapping

Direct match: `odds/history/:ID` uses the same `fixture_id` as `matches.db`. No translation needed.

## League Mapping

Not applicable at the odds endpoint level. Fixtures are fetched by fixture_id, which already encodes the league/season via matches.db.

## Bookmaker Identifiers (from sample)

| ID | Name | Notes |
|---|---|---|
| 1 | Pinnacle | Sharp bookmaker — preferred for de-vig |
| 2 | Bet365 | Major recreational bookmaker |
| 3 | 1xBet | High-volume international |
| 4 | WilliamHill | Major UK bookmaker |

**Note:** Sample was truncated (4 of 314 records for one fixture). The full bookmaker roster requires either a `GET /api/bookmakers` call or inspecting actual fetched data. The Postman collection confirms the bookmakers endpoint exists but has no example response body.

## Market Identifiers

| market_id | market_key | Description |
|---|---|---|
| 6 | ft_result | Full-Time Result (1X2) — the only market needed for this project |

## Decimal Odds Fields

- **opening**: Earliest recorded price. String format (e.g., "2.63"). Must be parsed to float.
- **closing**: Final pre-kickoff price. String format. Must be parsed to float.
- **peak**: Highest price seen. String format. **Timing unknown — unsafe for causal use.**

## Timestamp Availability

### odds/history endpoint:
**NO per-record timestamp field.** The response contains only `opening`, `closing`, and `peak` values. There is no `unix`, `datetime`, `created_at`, `updated_at`, or any temporal field on individual records.

### odds/movement endpoint (for comparison):
**HAS timestamps** — each record includes `unix` (epoch) and `datetime` (human-readable). However, this endpoint has **21-day retention only**. Data is automatically purged. The retention note explicitly states: *"Use the odds/history endpoint for permanent historical records (opening, closing, peak)."*

### Implication:
For historical fixtures older than 21 days, there is no way to independently verify the exact timestamp when opening or closing prices were recorded. We must rely on OddAlerts' definition:
- "opening" = earliest recorded price (pre-match by definition)
- "closing" = final price before kickoff (pre-match by definition)

## API Pagination

The odds/history endpoint returns all records for a fixture in a single response (no pagination observed — the sample had `"count": 314` records returned in one response). The `/fixtures/between` endpoint paginates, but odds/history appears to return the full dataset per fixture.

## Rate Limits

Not explicitly documented in the Postman collection. The existing `api_client.py` implements:
- Retry with exponential backoff on HTTP 429
- `Retry-After` header support
- Default max_retries: 5
- Default backoff_base: 1.5 seconds

For ~8,977 fixture-level API calls (or ~180 batch calls at 50/batch), rate limiting must be respected. The ingestion script should use conservative delays between requests.

## Missing/Error Behavior

- HTTP 404: Expected for fixture IDs that don't exist in OddAlerts
- HTTP 429: Rate limited — retry with backoff
- HTTP 5xx: Transient — retry with backoff
- Empty `data` array: Fixture exists but has no odds records
- `has_odds=0` fixtures (6 total): Expected to return empty or 404

## Strategy for Ingestion

1. Use `markets=6` filter to fetch only ft_result odds (reduces response size)
2. Use batch endpoint (`/odds/history/multiple?ids=...`) with up to 50 IDs per request to minimize API calls (~180 calls instead of ~8,977)
3. Store `retrieved_at` timestamp locally for audit trail
4. Filter to eligible seasons (exclude 2025/26)
5. Handle missing/empty responses gracefully
