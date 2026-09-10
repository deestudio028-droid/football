# Phase 1: Historical Data Ingestion — Build & Validation Report

**Scope actually built:** data acquisition and raw/normalized storage only. No feature engineering, no attack/defence formulas, no Poisson/Monte Carlo, no ML, no predictions, no UI. `include_frozen=true` and OddAlerts' `/predictions`, `/probability`, `/correctScores` endpoints are not called anywhere in this code — they are simply not implemented, so it is structurally impossible for the pipeline to use them by accident.

**Status: full 5-league × 6-season ingestion has NOT been started.** Per the explicit instruction, only a small controlled sample was run and validated. See "6. Readiness" below for what's needed before the full run.

---

## 1. Files created

```
config/competitions.json          Known-good competition_id/season_id mapping for all 5 leagues, 2020/21-2025/26
requirements.txt                  requests, pytest (pytest unavailable in this sandbox; suite runs on stdlib unittest instead)
src/ingestion/__init__.py
src/ingestion/config.py           .env + competitions.json loading; Config object never prints/logs the token
src/ingestion/logging_setup.py    Logging setup + TokenRedactingFilter that scrubs api_token=... from every log line
src/ingestion/api_client.py       OddAlertsClient: only exposes /fixtures/between; retry/backoff; pagination
src/ingestion/season_dates.py     Season-name -> wide UTC unix bracket (a filter alongside competitions/seasons)
src/ingestion/raw_store.py        Saves raw, untouched API response pages to data/raw/<competition_id>/<season_id>/
src/ingestion/normalize.py        Raw fixture -> flat normalized record with explicit missing-field tracking
src/ingestion/db.py               SQLite schema + idempotent upsert-by-fixture_id (the dedup mechanism)
src/ingestion/checkpoint.py       Resumable per-(competition,season) progress tracking, JSON-backed
src/ingestion/ingest.py           Orchestrator + CLI entry point (`python -m ingestion.ingest`)
tests/_pathfix.py                 Shared sys.path bootstrap (no install step needed)
tests/test_normalize.py           5 tests
tests/test_db.py                  4 tests
tests/test_checkpoint.py          4 tests
tests/test_raw_store.py           4 tests
tests/test_api_client.py          11 tests (incl. 2 regression tests from bugs found during validation)
tests/test_config_and_logging.py  5 tests
data/samples/phase1_ingestion_sample_validation.json       Sample-run evidence summary (no token)
data/samples/phase1_sample_fixtures_epl_2024_25_partial.json  51 real fixture records used for validation (no token)
```

Nothing in `DATA_AUDIT.md`, `PROVIDER_RESEARCH.md`, or `docs/ODD_ALERTS_V1_DATA_SUFFICIENCY.md` was modified.

---

## 2. Architecture

**Raw layer** (`raw_store.py`): one JSON file per API page fetched, at `data/raw/<competition_id>/<season_id>/fixtures_between_page_NNNN.json`, written exactly as returned (atomic write via temp-file-then-rename, so a crash mid-write can't leave a corrupt file). This is the immutable source of truth — if a normalization bug is ever found, everything can be recomputed from these files without re-hitting the API.

**Normalized layer** (`normalize.py` + `db.py`): every fixture becomes one row in a `fixtures` table in SQLite, keyed by OddAlerts' own fixture `id`. All ~76 top-level and `stats.*` fields the audit confirmed exist on real fixtures are persisted as explicit columns (flattened, `stats.home_xg` → `stat_home_xg`). A field that the API omitted or returned `null` is stored as `NULL` **and** its name is recorded in a `missing_fields` JSON column plus a `missing_field_count`/`is_complete` flag — so "confirmed absent" is always distinguishable from "silently dropped." No derived/feature columns exist in this schema by design.

**Dedup** (requirement #11): the `fixture_id` primary key plus an `INSERT ... ON CONFLICT DO UPDATE` upsert means re-ingesting the same fixture — whether from an overlapping date window or a resumed run — updates the existing row rather than creating a duplicate. Verified directly (see §4).

**Resumability** (requirement #14): checkpointing is per `(competition_id, season_id)` window, not per-page. A season is at most ~2 pages of 250, so re-walking an incomplete window from page 1 on resume is cheap and avoids the much harder problem of resuming mid-way through an API-driven `next_page_url` chain (which can't be jumped into at an arbitrary page). A window already marked `complete` in `data/checkpoints/ingestion_state.json` is never re-fetched; raw-file writes and DB upserts are independently idempotent regardless, so an unnecessary re-fetch can't corrupt anything.

**Retry/backoff** (requirement #12): `api_client.py` retries on connection errors and HTTP 429/5xx with exponential backoff + jitter (honoring `Retry-After` when present), up to a configurable `max_retries`. Non-retryable 4xx errors (401, 403, 404, etc.) fail immediately rather than wasting retry budget. A hard `max_pages` safety cap prevents any possibility of an infinite pagination loop.

**Security** (requirements #1, security instructions): the token is read once from `.env` into `Config.api_token` and never appears in `repr(Config)`. Every logger in the project runs through a `TokenRedactingFilter` that regexes out `api_token=<value>` from every log record before it's written to a file or stdout, and `logging_setup.redact()` is applied to every string that goes into an exception message inside `api_client.py`. This was not just designed but tested against a real leak (see §5).

**Endpoint choice** (requirements #5, #15, #16): the client only implements `/fixtures/between?include=stats`. It has no generic "call any endpoint" method, so `include_frozen=true`, `/predictions`, `/probability`, and `/correctScores` cannot be reached through this codebase at all — this was a deliberate design constraint, not just a policy note.

---

## 3. Sample ingestion — what was actually run

**The bash sandbox this pipeline was developed in cannot make outbound HTTPS requests to `data.oddalerts.com` (or any external host except through the dedicated `web_fetch` tool)** — confirmed by testing `curl` against both `data.oddalerts.com` and `github.com` from bash, both of which failed at the proxy layer. This is a constraint of the development sandbox, not of the pipeline code, and is expected to work normally on the client's own machine (a normal local network connection), which is the actual deployment target per the project's "local execution" instructions.

Given that constraint, validation was split across two channels that together cover the whole pipeline:

- **The HTTP layer itself** (`api_client.py`: pagination-following, retry/backoff, 4xx-vs-5xx handling, the `max_pages` safety cap, token redaction) was validated with 11 tests against a mocked `requests.Session`, covering exactly the failure modes that matter: connection errors, HTTP 429/5xx retried-then-succeeds, HTTP 401 failing immediately without wasting retries, retries exhausted, `next_page_url` followed across 2 pages, a placeholder non-URL string not followed, and a hard cap on page count.
- **The storage/normalization/dedup layer** was validated against **real OddAlerts data**: 90 fixtures from Premier League season 2024/25 (Aug–Nov 2024 window) were fetched live via `GET /fixtures/between?competitions=423&seasons=6484&from=1723507200&to=1730419200&include=stats` (using the `web_fetch` tool, which does have external access), of which 51 were saved intact after a tool-level response-size truncation (the same truncation limit noted during the original OddAlerts audit — not an API limit). Those 51 real fixture records were run through `raw_store.save_page`, `normalize_fixture`, and `FixtureDB.upsert_fixtures` exactly as `ingest.py` would.

Evidence: `data/samples/phase1_sample_fixtures_epl_2024_25_partial.json` (the 51 real fixtures) and `data/samples/phase1_ingestion_sample_validation.json` (the validation results below, machine-readable).

---

## 4. Validation results

| Check | Result |
|---|---|
| Authentication | Confirmed working via the same live call used for the sample data (200 response with real fixture data) |
| Pagination | `info.next_page_url` correctly followed across pages in mocked tests; also discovered and now explicitly tested that `next_page_url` can be the JSON **boolean `false`** (not just `null` or a URL string) when there's only one page — a real API shape not previously documented, confirmed on this live call (`"next_page_url": false`) |
| Raw responses saved | Raw page written to `data/raw/423/6484/fixtures_between_page_0001.json`; reloaded byte-identical to what was written |
| Normalized records created | All 51 real fixtures normalized into flat records; all had at least one legitimately-null field explicitly tracked (e.g. `elapsed_seconds`, `time_added`, `winning_team` on draws, `offsides`/`red_cards` when the count is genuinely zero-and-null on OddAlerts) — nothing was silently dropped |
| Duplicates handled | Re-upserting the identical 51-record batch left the row count at 51 (no duplicates). Upserting one fixture again with a deliberately changed `home_goals` value updated the existing row in place rather than inserting a second row |
| Retries/errors handled | 9 of the 11 API-client tests specifically exercise error/retry paths (connection errors, 5xx, 429, 401, retry exhaustion) |
| API token never leaks | Confirmed absent from every raw file, the normalized DB, and all log files produced during this validation pass |

**Test suite: 33/33 passing** (`python -m unittest discover -s tests`, run from the project root). Note: `pytest` could not be installed in the sandbox (no outbound `pip` access), so the suite is written against the Python standard library's `unittest` instead — functionally equivalent, no external test-runner dependency required to run it.

---

## 5. Issues discovered during validation (both fixed, not just noted)

1. **Token leak in an exception path (found and fixed).** The very first live-style run (before the sandbox's network restriction was understood) hit a `ConnectionError` from `requests`, whose own `__str__` embeds the full request URL including `api_token=...`. The retry-exhaustion error message in `api_client.py` was interpolating that raw exception object directly instead of redacting it first, so the token appeared in one log line before the fix. This was caught immediately (the design's own redaction filter caught the earlier warning-level log lines from the same run, which is what made the one un-redacted line stand out), the source line was fixed (`redact(str(last_error))`), the leaked log/checkpoint files were deleted, a regression test (`test_exhausted_connection_error_retries_never_leak_token`) was added reproducing the exact failure shape, and the full repo was re-grepped for the literal token value to confirm nothing else was affected. This is exactly the kind of thing "run a small sample first" is meant to catch before a full run.
2. **SQLite WAL journal mode fails with `disk I/O error` on this project's mounted folder** (and on the sandbox's generic mounted output folder too — confirmed with a minimal reproduction, not assumed). WAL relies on shared-memory/byte-range locking that this mount doesn't support. Fixed by switching to the default rollback-journal mode (`PRAGMA journal_mode=DELETE`), which is slightly slower under concurrent writers but works everywhere and is sufficient since this pipeline is single-writer by design. **This is worth flagging to whoever runs the full ingestion:** if the client's own machine also hits this on their specific filesystem (unlikely on a normal local NTFS drive, but not verified from here), the same fix applies.
3. **`info.next_page_url` can be the JSON boolean `false`**, not only `null` or a URL string, when there's exactly one page of results. The existing pagination logic already handled this correctly (Python's `and` short-circuits on `False` before any `.startswith()` call), but this wasn't something the original design explicitly accounted for, so a regression test was added to lock in the behavior rather than leave it as an accidental correctness.
4. **Not yet resolved / deferred:** the 51-of-90 fixture truncation on the sample pull is a tool-side response-size limit in this environment (same class of issue noted in the original audit), not a pipeline bug — `ingest.py` itself uses the `requests` library directly against the real API and would not hit this artificial truncation once run outside this sandbox.

---

## 6. Readiness for full historical ingestion

**Not started yet, and should not start from inside this sandbox** — the sandbox's network restriction means `ingest.py` cannot reach `data.oddalerts.com` directly from here (only the separate `web_fetch` tool can, which is not how the production pipeline is meant to run). The code, tests, and sample-data validation are complete and passing; what's left is entirely about *where* it runs:

- Run `pip install -r requirements.txt` and then `python -m ingestion.ingest` from the project root **on the client's own machine** (or any environment with normal outbound internet access), where `data.oddalerts.com` is reachable directly.
- Before kicking off the full 5-league × 6-season run, run one more real end-to-end smoke test in that environment specifically (e.g. `python -m ingestion.ingest --competitions 423 --seasons "2024/2025"`) to confirm the live HTTP path (not just the mocked tests) behaves as expected there — this closes the one validation gap this sandbox couldn't reach.
- After that smoke test passes, the full run is `python -m ingestion.ingest` with no filters, which will walk all 5 competitions × up to 6 seasons each (30 windows, each a season = ~1-2 pages), respecting the checkpoint file so it can be safely interrupted and resumed.
