"""Phase 3B — Historical odds extraction from OddAlerts odds/history endpoint.

Usage:
    python research/market_odds/ingest_odds_history.py

This script:
1. Reads all eligible fixture IDs from matches.db (5 leagues, excl 2025/26)
2. Fetches odds/history for each fixture (filtered to ft_result market_id=6)
3. Stores raw records in research/market_odds/odds_history.sqlite
4. Logs progress and errors

SAFETY:
- Opens matches.db in READ-ONLY mode
- Never modifies any production database or model artifact
- All output goes to research/market_odds/odds_history.sqlite (new file)
- API token loaded from .env, never logged or printed

Run from the project root directory.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library required. Install with: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
OUTPUT_DB = Path(__file__).resolve().parent / "odds_history.sqlite"
ENV_FILE = PROJECT_ROOT / ".env"

BASE_URL = "https://data.oddalerts.com/api"
MARKET_ID_FT_RESULT = 6  # ft_result (1X2)

# Eligible seasons (2025/26 quarantined)
ELIGIBLE_SEASONS = [
    "2020/2021", "2021/2022", "2022/2023",
    "2023/2024", "2024/2025",
]

# Five target leagues
TARGET_COMPETITIONS = [423, 419, 477, 499, 200]  # PL, LaLiga, Buli, SerieA, L1

# Rate limiting
BATCH_SIZE = 50  # max IDs per batch request (API limit)
DELAY_BETWEEN_BATCHES = 1.5  # seconds
MAX_RETRIES = 5
BACKOFF_BASE = 2.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path(__file__).resolve().parent / "ingest_odds.log",
            mode="a",
        ),
    ],
)
log = logging.getLogger("odds_ingest")


def _redact_token(s: str, token: str) -> str:
    """Strip API token from any string before logging."""
    return s.replace(token, "<REDACTED>") if token else s


# ---------------------------------------------------------------------------
# API token loading
# ---------------------------------------------------------------------------

def load_api_token() -> str:
    """Load OddAlerts API token from .env file or environment."""
    # Try environment first
    token = os.environ.get("OddAlerts_API")
    if token:
        return token.strip()
    # Fall back to .env file
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "OddAlerts_API":
                return value.strip()
    raise RuntimeError(
        f"Missing API token. Set OddAlerts_API in environment or {ENV_FILE}"
    )


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def get_eligible_fixtures() -> list[dict]:
    """Load fixture IDs for eligible leagues/seasons from matches.db (read-only)."""
    uri = f"file:{MATCHES_DB.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(TARGET_COMPETITIONS))
        season_ph = ",".join("?" * len(ELIGIBLE_SEASONS))
        query = f"""
            SELECT fixture_id, competition_id, competition_name, season,
                   home_name, away_name, unix, has_odds
            FROM fixtures
            WHERE competition_id IN ({placeholders})
              AND season IN ({season_ph})
            ORDER BY unix ASC, fixture_id ASC
        """
        params = list(TARGET_COMPETITIONS) + list(ELIGIBLE_SEASONS)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def create_output_db() -> sqlite3.Connection:
    """Create the output SQLite database with schema."""
    conn = sqlite3.connect(str(OUTPUT_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER NOT NULL,
            market_key TEXT NOT NULL,
            market_id INTEGER NOT NULL,
            outcome TEXT NOT NULL,
            opening REAL,
            closing REAL,
            peak REAL,
            bookmaker_id INTEGER NOT NULL,
            bookmaker_name TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            UNIQUE(fixture_id, market_id, outcome, bookmaker_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            fixture_id INTEGER PRIMARY KEY,
            competition_id INTEGER,
            competition_name TEXT,
            season TEXT,
            has_odds INTEGER,
            status TEXT NOT NULL,
            record_count INTEGER DEFAULT 0,
            error_message TEXT,
            fetched_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingest_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    return conn


def get_already_fetched(conn: sqlite3.Connection) -> set[int]:
    """Get fixture IDs already fetched (for resume support)."""
    rows = conn.execute(
        "SELECT fixture_id FROM fetch_log WHERE status IN ('success', 'empty')"
    ).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# API fetching
# ---------------------------------------------------------------------------

def fetch_odds_batch(
    session: requests.Session,
    fixture_ids: list[int],
    api_token: str,
) -> dict[int, list[dict]]:
    """Fetch odds/history for a batch of fixture IDs.

    Uses the single-ID endpoint if batch size is 1,
    or the multiple endpoint for batches > 1.

    Returns: {fixture_id: [records]} mapping.
    """
    results: dict[int, list[dict]] = {fid: [] for fid in fixture_ids}

    if len(fixture_ids) == 1:
        url = f"{BASE_URL}/odds/history/{fixture_ids[0]}"
        params = {"api_token": api_token, "markets": MARKET_ID_FT_RESULT}
    else:
        ids_str = ",".join(str(fid) for fid in fixture_ids)
        url = f"{BASE_URL}/odds/history/multiple"
        params = {
            "api_token": api_token,
            "ids": ids_str,
            "markets": MARKET_ID_FT_RESULT,
        }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            delay = BACKOFF_BASE ** attempt + random.uniform(0, 0.5)
            log.warning(
                "Attempt %d/%d connection error: %s — retrying in %.1fs",
                attempt, MAX_RETRIES,
                _redact_token(str(exc), api_token),
                delay,
            )
            time.sleep(delay)
            continue

        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError:
                raise RuntimeError(f"Non-JSON 200 response for fixtures {fixture_ids}")

            data = body.get("data", [])
            for rec in data:
                fid = rec.get("fixture_id")
                if fid in results:
                    results[fid].append(rec)
            return results

        if resp.status_code in (429, 500, 502, 503, 504):
            last_error = RuntimeError(f"HTTP {resp.status_code}")
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = BACKOFF_BASE ** attempt
            else:
                delay = BACKOFF_BASE ** attempt + random.uniform(0, 0.5)
            log.warning(
                "Attempt %d/%d HTTP %d — retrying in %.1fs",
                attempt, MAX_RETRIES, resp.status_code, delay,
            )
            time.sleep(delay)
            continue

        # Non-retryable error
        raise RuntimeError(
            f"HTTP {resp.status_code} for fixtures {fixture_ids}: "
            f"{_redact_token(resp.text[:200], api_token)}"
        )

    raise RuntimeError(
        f"Exhausted {MAX_RETRIES} retries for fixtures {fixture_ids}: "
        f"{_redact_token(str(last_error), api_token)}"
    )


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def ingest(
    fixtures: list[dict],
    out_conn: sqlite3.Connection,
    api_token: str,
    batch_size: int = BATCH_SIZE,
) -> dict[str, int]:
    """Fetch and store odds for all fixtures. Returns summary stats."""
    already_done = get_already_fetched(out_conn)
    pending = [f for f in fixtures if f["fixture_id"] not in already_done]

    log.info(
        "Total eligible: %d, already fetched: %d, pending: %d",
        len(fixtures), len(already_done), len(pending),
    )

    stats = {
        "total": len(fixtures),
        "already_done": len(already_done),
        "success": 0,
        "empty": 0,
        "errors": 0,
        "records_stored": 0,
    }

    session = requests.Session()
    now_str = datetime.now(timezone.utc).isoformat()

    # Process in batches
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        batch_ids = [f["fixture_id"] for f in batch]
        batch_map = {f["fixture_id"]: f for f in batch}

        log.info(
            "Batch %d/%d: fetching %d fixtures (%d-%d of %d pending)",
            batch_start // batch_size + 1,
            (len(pending) + batch_size - 1) // batch_size,
            len(batch_ids),
            batch_start + 1,
            min(batch_start + batch_size, len(pending)),
            len(pending),
        )

        try:
            results = fetch_odds_batch(session, batch_ids, api_token)
        except RuntimeError as exc:
            log.error("Batch fetch failed: %s", _redact_token(str(exc), api_token))
            # Log individual failures
            for fid in batch_ids:
                fix = batch_map[fid]
                out_conn.execute(
                    """INSERT OR REPLACE INTO fetch_log
                       (fixture_id, competition_id, competition_name, season,
                        has_odds, status, error_message, fetched_at)
                       VALUES (?, ?, ?, ?, ?, 'error', ?, ?)""",
                    (fid, fix["competition_id"], fix["competition_name"],
                     fix["season"], fix["has_odds"],
                     _redact_token(str(exc), api_token), now_str),
                )
                stats["errors"] += 1
            out_conn.commit()
            time.sleep(DELAY_BETWEEN_BATCHES * 2)
            continue

        # Store results per fixture
        for fid, records in results.items():
            fix = batch_map[fid]

            if records:
                for rec in records:
                    opening = _parse_odds(rec.get("opening"))
                    closing = _parse_odds(rec.get("closing"))
                    peak = _parse_odds(rec.get("peak"))

                    try:
                        out_conn.execute(
                            """INSERT OR REPLACE INTO odds_records
                               (fixture_id, market_key, market_id, outcome,
                                opening, closing, peak,
                                bookmaker_id, bookmaker_name, retrieved_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (fid, rec.get("market_key", ""),
                             rec.get("market_id", 0),
                             rec.get("outcome", ""),
                             opening, closing, peak,
                             rec.get("bookmaker_id", 0),
                             rec.get("bookmaker_name", ""),
                             now_str),
                        )
                        stats["records_stored"] += 1
                    except sqlite3.Error as exc:
                        log.warning("Insert error for fixture %d: %s", fid, exc)

                out_conn.execute(
                    """INSERT OR REPLACE INTO fetch_log
                       (fixture_id, competition_id, competition_name, season,
                        has_odds, status, record_count, fetched_at)
                       VALUES (?, ?, ?, ?, ?, 'success', ?, ?)""",
                    (fid, fix["competition_id"], fix["competition_name"],
                     fix["season"], fix["has_odds"], len(records), now_str),
                )
                stats["success"] += 1
            else:
                out_conn.execute(
                    """INSERT OR REPLACE INTO fetch_log
                       (fixture_id, competition_id, competition_name, season,
                        has_odds, status, record_count, fetched_at)
                       VALUES (?, ?, ?, ?, ?, 'empty', 0, ?)""",
                    (fid, fix["competition_id"], fix["competition_name"],
                     fix["season"], fix["has_odds"], now_str),
                )
                stats["empty"] += 1

        out_conn.commit()

        # Rate limiting
        if batch_start + batch_size < len(pending):
            time.sleep(DELAY_BETWEEN_BATCHES)

    return stats


def _parse_odds(val: Any) -> float | None:
    """Parse odds value from string or numeric to float. Returns None if invalid."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    log.info("=" * 60)
    log.info("Phase 3B — Historical Odds Extraction")
    log.info("=" * 60)

    # Verify matches.db exists
    if not MATCHES_DB.exists():
        log.error("matches.db not found at %s", MATCHES_DB)
        return 1

    # Load API token
    try:
        api_token = load_api_token()
    except RuntimeError as exc:
        log.error(str(exc))
        return 1
    log.info("API token loaded (length=%d)", len(api_token))

    # Load eligible fixtures
    fixtures = get_eligible_fixtures()
    log.info("Loaded %d eligible fixtures from matches.db", len(fixtures))

    with_odds = sum(1 for f in fixtures if f["has_odds"])
    without_odds = len(fixtures) - with_odds
    log.info("  has_odds=1: %d, has_odds=0: %d", with_odds, without_odds)

    # Only attempt fixtures with has_odds=1
    fixtures_to_fetch = [f for f in fixtures if f["has_odds"]]
    log.info("Will attempt to fetch odds for %d fixtures", len(fixtures_to_fetch))

    # Create output database
    out_conn = create_output_db()

    # Store metadata
    out_conn.execute(
        "INSERT OR REPLACE INTO ingest_metadata (key, value) VALUES (?, ?)",
        ("start_time", datetime.now(timezone.utc).isoformat()),
    )
    out_conn.execute(
        "INSERT OR REPLACE INTO ingest_metadata (key, value) VALUES (?, ?)",
        ("eligible_fixtures", str(len(fixtures))),
    )
    out_conn.execute(
        "INSERT OR REPLACE INTO ingest_metadata (key, value) VALUES (?, ?)",
        ("target_market", "ft_result (market_id=6)"),
    )
    out_conn.execute(
        "INSERT OR REPLACE INTO ingest_metadata (key, value) VALUES (?, ?)",
        ("eligible_seasons", json.dumps(ELIGIBLE_SEASONS)),
    )
    out_conn.execute(
        "INSERT OR REPLACE INTO ingest_metadata (key, value) VALUES (?, ?)",
        ("target_competitions", json.dumps(TARGET_COMPETITIONS)),
    )
    out_conn.commit()

    # Run ingestion
    try:
        stats = ingest(fixtures_to_fetch, out_conn, api_token)
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Progress saved — re-run to resume.")
        out_conn.close()
        return 130

    # Store completion metadata
    out_conn.execute(
        "INSERT OR REPLACE INTO ingest_metadata (key, value) VALUES (?, ?)",
        ("end_time", datetime.now(timezone.utc).isoformat()),
    )
    out_conn.execute(
        "INSERT OR REPLACE INTO ingest_metadata (key, value) VALUES (?, ?)",
        ("stats", json.dumps(stats)),
    )
    out_conn.commit()
    out_conn.close()

    # Report
    log.info("=" * 60)
    log.info("INGESTION COMPLETE")
    log.info("  Total eligible:   %d", stats["total"])
    log.info("  Already done:     %d", stats["already_done"])
    log.info("  Success:          %d", stats["success"])
    log.info("  Empty response:   %d", stats["empty"])
    log.info("  Errors:           %d", stats["errors"])
    log.info("  Records stored:   %d", stats["records_stored"])
    log.info("  Output: %s", OUTPUT_DB)
    log.info("=" * 60)

    # Log has_odds=0 fixtures (not fetched, for the record)
    no_odds = [f for f in fixtures if not f["has_odds"]]
    if no_odds:
        log.info("Fixtures with has_odds=0 (not fetched):")
        for f in no_odds:
            log.info(
                "  %d: %s vs %s (%s %s)",
                f["fixture_id"], f["home_name"], f["away_name"],
                f["competition_name"], f["season"],
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
