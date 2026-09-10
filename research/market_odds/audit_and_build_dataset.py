"""Phase 3C-I — Coverage, bookmaker, causality audits + research dataset builder.

Usage:
    python research/market_odds/audit_and_build_dataset.py

Prerequisites:
    - research/market_odds/odds_history.sqlite must exist (from ingest_odds_history.py)
    - data/processed/matches.db must exist

This script:
1. Phase 3C: Coverage audit by league, season, pooled
2. Phase 3D: Bookmaker audit
3. Phase 3E: Opening vs closing availability
4. Phase 3F: Causality / temporal safety classification
5. Phase 3G/I: Build clean research dataset with de-vigged probabilities

Output:
    - research/market_odds/audit_results.json (machine-readable audit)
    - research/market_odds/research_dataset.sqlite (clean research dataset)
    - Prints human-readable audit to stdout

SAFETY:
    - Opens matches.db in READ-ONLY mode
    - Opens odds_history.sqlite in READ-ONLY mode (for auditing)
    - Creates NEW research_dataset.sqlite
    - Never modifies any production file
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devig import power_devig, DevigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
ODDS_DB = Path(__file__).resolve().parent / "odds_history.sqlite"
RESEARCH_DB = Path(__file__).resolve().parent / "research_dataset.sqlite"
AUDIT_JSON = Path(__file__).resolve().parent / "audit_results.json"

ELIGIBLE_SEASONS = [
    "2020/2021", "2021/2022", "2022/2023",
    "2023/2024", "2024/2025",
]
TARGET_COMPETITIONS = {
    200: "Ligue 1",
    419: "La Liga",
    423: "Premier League",
    477: "Bundesliga",
    499: "Serie A",
}


def load_fixture_metadata() -> dict[int, dict]:
    """Load fixture metadata from matches.db (read-only)."""
    uri = f"file:{MATCHES_DB.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(TARGET_COMPETITIONS))
    season_ph = ",".join("?" * len(ELIGIBLE_SEASONS))
    query = f"""
        SELECT fixture_id, competition_id, competition_name, season,
               home_name, away_name, unix, has_odds, date
        FROM fixtures
        WHERE competition_id IN ({placeholders})
          AND season IN ({season_ph})
        ORDER BY unix ASC
    """
    params = list(TARGET_COMPETITIONS.keys()) + list(ELIGIBLE_SEASONS)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {r["fixture_id"]: dict(r) for r in rows}


def load_odds_data() -> list[dict]:
    """Load all odds records from odds_history.sqlite (read-only)."""
    if not ODDS_DB.exists():
        print(f"ERROR: {ODDS_DB} not found. Run ingest_odds_history.py first.")
        sys.exit(1)
    uri = f"file:{ODDS_DB.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT fixture_id, market_key, market_id, outcome,
               opening, closing, peak,
               bookmaker_id, bookmaker_name, retrieved_at
        FROM odds_records
        WHERE market_id = 6
        ORDER BY fixture_id, bookmaker_id, outcome
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_fetch_log() -> dict[int, dict]:
    """Load fetch log from odds_history.sqlite."""
    uri = f"file:{ODDS_DB.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fetch_log").fetchall()
    conn.close()
    return {r["fixture_id"]: dict(r) for r in rows}


# -----------------------------------------------------------------------
# Phase 3C — Coverage Audit
# -----------------------------------------------------------------------

def coverage_audit(
    fixtures: dict[int, dict],
    odds: list[dict],
    fetch_log: dict[int, dict],
) -> dict:
    """Calculate coverage by league, season, and pooled."""
    # Group odds by fixture_id → bookmaker_id → outcomes
    fixture_odds: dict[int, dict[int, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for rec in odds:
        fid = rec["fixture_id"]
        bid = rec["bookmaker_id"]
        outcome = rec["outcome"]
        fixture_odds[fid][bid][outcome] = rec

    # Check which fixtures have complete 1X2 for at least one bookmaker
    complete_fixtures: dict[int, list[int]] = {}  # fid -> list of bookmaker_ids with complete 1X2
    for fid, bookmakers in fixture_odds.items():
        complete_bms = []
        for bid, outcomes in bookmakers.items():
            if all(o in outcomes for o in ("home", "draw", "away")):
                # Check all three have valid closing odds
                if all(
                    outcomes[o].get("closing") is not None
                    and outcomes[o]["closing"] > 0
                    for o in ("home", "draw", "away")
                ):
                    complete_bms.append(bid)
        if complete_bms:
            complete_fixtures[fid] = complete_bms

    # Coverage by league
    league_coverage = {}
    for comp_id, comp_name in TARGET_COMPETITIONS.items():
        league_fixtures = {
            fid: f for fid, f in fixtures.items()
            if f["competition_id"] == comp_id
        }
        total = len(league_fixtures)
        has_odds = sum(1 for f in league_fixtures.values() if f["has_odds"])
        fetched_success = sum(
            1 for fid in league_fixtures
            if fid in fetch_log and fetch_log[fid]["status"] == "success"
        )
        fetched_empty = sum(
            1 for fid in league_fixtures
            if fid in fetch_log and fetch_log[fid]["status"] == "empty"
        )
        fetched_error = sum(
            1 for fid in league_fixtures
            if fid in fetch_log and fetch_log[fid]["status"] == "error"
        )
        complete = sum(1 for fid in league_fixtures if fid in complete_fixtures)

        league_coverage[comp_name] = {
            "total_fixtures": total,
            "has_odds_flag": has_odds,
            "fetched_success": fetched_success,
            "fetched_empty": fetched_empty,
            "fetched_error": fetched_error,
            "complete_1x2": complete,
            "complete_pct": round(100 * complete / total, 2) if total else 0,
        }

    # Coverage by season
    season_coverage = {}
    for season in ELIGIBLE_SEASONS:
        season_fixtures = {
            fid: f for fid, f in fixtures.items()
            if f["season"] == season
        }
        total = len(season_fixtures)
        complete = sum(1 for fid in season_fixtures if fid in complete_fixtures)
        season_coverage[season] = {
            "total_fixtures": total,
            "complete_1x2": complete,
            "complete_pct": round(100 * complete / total, 2) if total else 0,
        }

    # Pooled
    total = len(fixtures)
    complete = len(complete_fixtures)
    pooled = {
        "total_fixtures": total,
        "has_odds_flag": sum(1 for f in fixtures.values() if f["has_odds"]),
        "fetched_success": sum(
            1 for fid in fixtures if fid in fetch_log
            and fetch_log[fid]["status"] == "success"
        ),
        "fetched_empty": sum(
            1 for fid in fixtures if fid in fetch_log
            and fetch_log[fid]["status"] == "empty"
        ),
        "fetched_error": sum(
            1 for fid in fixtures if fid in fetch_log
            and fetch_log[fid]["status"] == "error"
        ),
        "complete_1x2": complete,
        "complete_pct": round(100 * complete / total, 2) if total else 0,
        "incomplete_fixtures": [
            fid for fid in fixtures if fid not in complete_fixtures
        ],
    }

    return {
        "by_league": league_coverage,
        "by_season": season_coverage,
        "pooled": pooled,
        "complete_fixtures": complete_fixtures,
    }


# -----------------------------------------------------------------------
# Phase 3D — Bookmaker Audit
# -----------------------------------------------------------------------

def bookmaker_audit(odds: list[dict], fixtures: dict[int, dict]) -> dict:
    """Analyze bookmaker distribution and coverage."""
    # Group by bookmaker
    bm_data: dict[int, dict] = {}
    for rec in odds:
        bid = rec["bookmaker_id"]
        if bid not in bm_data:
            bm_data[bid] = {
                "bookmaker_id": bid,
                "bookmaker_name": rec["bookmaker_name"],
                "fixture_ids": set(),
                "opening_count": 0,
                "closing_count": 0,
                "records": 0,
            }
        bm_data[bid]["fixture_ids"].add(rec["fixture_id"])
        bm_data[bid]["records"] += 1
        if rec.get("opening") is not None and rec["opening"] > 0:
            bm_data[bid]["opening_count"] += 1
        if rec.get("closing") is not None and rec["closing"] > 0:
            bm_data[bid]["closing_count"] += 1

    total_fixtures = len(fixtures)
    result = {}
    for bid, data in sorted(bm_data.items()):
        fixture_count = len(data["fixture_ids"])
        result[data["bookmaker_name"]] = {
            "bookmaker_id": bid,
            "fixture_count": fixture_count,
            "coverage_pct": round(100 * fixture_count / total_fixtures, 2),
            "total_records": data["records"],
            "opening_available": data["opening_count"],
            "closing_available": data["closing_count"],
            "opening_pct": round(100 * data["opening_count"] / data["records"], 2) if data["records"] else 0,
            "closing_pct": round(100 * data["closing_count"] / data["records"], 2) if data["records"] else 0,
        }

    return result


# -----------------------------------------------------------------------
# Phase 3E — Opening vs Closing Audit
# -----------------------------------------------------------------------

def opening_closing_audit(odds: list[dict]) -> dict:
    """Analyze opening vs closing odds availability."""
    stats = {
        "both": 0,
        "opening_only": 0,
        "closing_only": 0,
        "neither": 0,
        "total_records": 0,
    }
    for rec in odds:
        has_opening = rec.get("opening") is not None and rec["opening"] > 0
        has_closing = rec.get("closing") is not None and rec["closing"] > 0
        stats["total_records"] += 1
        if has_opening and has_closing:
            stats["both"] += 1
        elif has_opening:
            stats["opening_only"] += 1
        elif has_closing:
            stats["closing_only"] += 1
        else:
            stats["neither"] += 1

    return stats


# -----------------------------------------------------------------------
# Phase 3F — Causality / Temporal Safety
# -----------------------------------------------------------------------

def causality_audit() -> dict:
    """Classify each odds field for causal safety."""
    return {
        "opening": {
            "classification": "ASSUMED PRE-MATCH",
            "evidence": (
                "OddAlerts defines 'opening' as the earliest recorded price. "
                "By definition this is the first price posted by a bookmaker, "
                "which occurs before the match. No independent timestamp exists "
                "on the odds/history endpoint to verify this."
            ),
            "risk": "LOW — earliest price is structurally pre-match",
        },
        "closing": {
            "classification": "ASSUMED PRE-MATCH",
            "evidence": (
                "OddAlerts defines 'closing' as the final price before kickoff. "
                "The odds/movement endpoint (which has timestamps) confirms that "
                "movement data tracks pre-kickoff changes, and the retention note "
                "directs users to odds/history for 'permanent historical records "
                "(opening, closing, peak)'. No independent timestamp exists on "
                "odds/history records."
            ),
            "risk": (
                "MODERATE — relies on OddAlerts' definition. Cannot independently "
                "verify that 'closing' was captured strictly before kickoff. "
                "Industry standard is that closing odds = final pre-match price, "
                "but edge cases (e.g., delayed kickoffs, in-play price captured "
                "as closing) cannot be ruled out without timestamps."
            ),
        },
        "peak": {
            "classification": "UNKNOWN TIMING",
            "evidence": (
                "Peak is the highest price seen during the market's lifetime. "
                "It could have occurred at any point — early pre-match, late "
                "pre-match, or even in-play if the bookmaker offers in-play "
                "markets. No temporal information is available."
            ),
            "risk": "HIGH — must NOT be used as a model feature",
        },
        "independent_timestamp_available": False,
        "movement_endpoint_retention_days": 21,
        "recommendation": (
            "Use CLOSING odds as the primary feature (most informative "
            "pre-match price, standard industry practice). Use OPENING odds "
            "as a robustness check. Do NOT use PEAK odds. Accept that "
            "pre-kickoff timing relies on OddAlerts' definition rather than "
            "independent verification."
        ),
    }


# -----------------------------------------------------------------------
# Phase 3I — Build Research Dataset
# -----------------------------------------------------------------------

def build_research_dataset(
    fixtures: dict[int, dict],
    odds: list[dict],
    complete_fixtures: dict[int, list[int]],
    preferred_bookmaker_id: int | None,
) -> int:
    """Build clean research dataset with de-vigged probabilities.

    Returns count of fixtures in the dataset.
    """
    # Group odds by fixture → bookmaker → outcome
    fixture_odds: dict[int, dict[int, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for rec in odds:
        fid = rec["fixture_id"]
        bid = rec["bookmaker_id"]
        outcome = rec["outcome"]
        fixture_odds[fid][bid][outcome] = rec

    # Create research dataset DB
    conn = sqlite3.connect(str(RESEARCH_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_odds (
            fixture_id INTEGER PRIMARY KEY,
            competition_id INTEGER NOT NULL,
            competition_name TEXT NOT NULL,
            season TEXT NOT NULL,
            date TEXT,
            unix INTEGER,
            home_name TEXT,
            away_name TEXT,
            bookmaker_id INTEGER NOT NULL,
            bookmaker_name TEXT NOT NULL,
            opening_home REAL,
            opening_draw REAL,
            opening_away REAL,
            closing_home REAL,
            closing_draw REAL,
            closing_away REAL,
            devig_opening_home REAL,
            devig_opening_draw REAL,
            devig_opening_away REAL,
            devig_opening_k REAL,
            devig_opening_overround REAL,
            devig_closing_home REAL,
            devig_closing_draw REAL,
            devig_closing_away REAL,
            devig_closing_k REAL,
            devig_closing_overround REAL,
            opening_causality TEXT NOT NULL DEFAULT 'ASSUMED PRE-MATCH',
            closing_causality TEXT NOT NULL DEFAULT 'ASSUMED PRE-MATCH'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()

    inserted = 0
    devig_failures_opening = 0
    devig_failures_closing = 0

    for fid in sorted(complete_fixtures.keys()):
        if fid not in fixtures:
            continue
        fix = fixtures[fid]
        complete_bms = complete_fixtures[fid]

        # Select bookmaker: prefer specified, else first available
        if preferred_bookmaker_id and preferred_bookmaker_id in complete_bms:
            selected_bm = preferred_bookmaker_id
        else:
            selected_bm = complete_bms[0]

        outcomes = fixture_odds[fid][selected_bm]
        bm_name = outcomes.get("home", outcomes.get("draw", {})).get(
            "bookmaker_name", ""
        )

        # Extract raw odds
        opening_h = outcomes["home"].get("opening")
        opening_d = outcomes["draw"].get("opening")
        opening_a = outcomes["away"].get("opening")
        closing_h = outcomes["home"].get("closing")
        closing_d = outcomes["draw"].get("closing")
        closing_a = outcomes["away"].get("closing")

        # De-vig opening
        dv_open = None
        if all(v is not None and v > 1.0 for v in [opening_h, opening_d, opening_a]):
            try:
                dv_open = power_devig(opening_h, opening_d, opening_a)
            except DevigError:
                devig_failures_opening += 1

        # De-vig closing
        dv_close = None
        if all(v is not None and v > 1.0 for v in [closing_h, closing_d, closing_a]):
            try:
                dv_close = power_devig(closing_h, closing_d, closing_a)
            except DevigError:
                devig_failures_closing += 1

        conn.execute(
            """INSERT OR REPLACE INTO research_odds VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fid,
                fix["competition_id"],
                fix["competition_name"],
                fix["season"],
                fix.get("date"),
                fix["unix"],
                fix["home_name"],
                fix["away_name"],
                selected_bm,
                bm_name,
                opening_h, opening_d, opening_a,
                closing_h, closing_d, closing_a,
                dv_open.p_home if dv_open else None,
                dv_open.p_draw if dv_open else None,
                dv_open.p_away if dv_open else None,
                dv_open.k if dv_open else None,
                dv_open.overround if dv_open else None,
                dv_close.p_home if dv_close else None,
                dv_close.p_draw if dv_close else None,
                dv_close.p_away if dv_close else None,
                dv_close.k if dv_close else None,
                dv_close.overround if dv_close else None,
                "ASSUMED PRE-MATCH",
                "ASSUMED PRE-MATCH",
            ),
        )
        inserted += 1

    # Metadata
    conn.execute(
        "INSERT OR REPLACE INTO dataset_metadata VALUES (?, ?)",
        ("created_at", datetime.now(timezone.utc).isoformat()),
    )
    conn.execute(
        "INSERT OR REPLACE INTO dataset_metadata VALUES (?, ?)",
        ("fixture_count", str(inserted)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO dataset_metadata VALUES (?, ?)",
        ("devig_failures_opening", str(devig_failures_opening)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO dataset_metadata VALUES (?, ?)",
        ("devig_failures_closing", str(devig_failures_closing)),
    )
    if preferred_bookmaker_id:
        conn.execute(
            "INSERT OR REPLACE INTO dataset_metadata VALUES (?, ?)",
            ("preferred_bookmaker_id", str(preferred_bookmaker_id)),
        )
    conn.commit()
    conn.close()

    return inserted


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Phase 3C-I — Odds Audit & Research Dataset Builder")
    print("=" * 60)

    # Load data
    print("\nLoading fixture metadata from matches.db...")
    fixtures = load_fixture_metadata()
    print(f"  {len(fixtures)} eligible fixtures")

    print("Loading odds data from odds_history.sqlite...")
    odds = load_odds_data()
    print(f"  {len(odds)} odds records")

    fetch_log = load_fetch_log()
    print(f"  {len(fetch_log)} fetch log entries")

    # Phase 3C — Coverage
    print("\n" + "=" * 60)
    print("PHASE 3C — COVERAGE AUDIT")
    print("=" * 60)
    cov = coverage_audit(fixtures, odds, fetch_log)

    print("\nBy League:")
    print(f"  {'League':<20} {'Total':>6} {'Odds':>6} {'Fetched':>8} "
          f"{'Complete':>9} {'%':>7}")
    print("  " + "-" * 56)
    for league, data in cov["by_league"].items():
        print(
            f"  {league:<20} {data['total_fixtures']:>6} "
            f"{data['has_odds_flag']:>6} {data['fetched_success']:>8} "
            f"{data['complete_1x2']:>9} {data['complete_pct']:>6.1f}%"
        )

    print("\nBy Season:")
    print(f"  {'Season':<12} {'Total':>6} {'Complete':>9} {'%':>7}")
    print("  " + "-" * 37)
    for season, data in cov["by_season"].items():
        print(
            f"  {season:<12} {data['total_fixtures']:>6} "
            f"{data['complete_1x2']:>9} {data['complete_pct']:>6.1f}%"
        )

    p = cov["pooled"]
    print(f"\nPooled:")
    print(f"  Total eligible:     {p['total_fixtures']}")
    print(f"  has_odds=1:         {p['has_odds_flag']}")
    print(f"  Fetched (success):  {p['fetched_success']}")
    print(f"  Fetched (empty):    {p['fetched_empty']}")
    print(f"  Fetched (error):    {p['fetched_error']}")
    print(f"  Complete 1X2:       {p['complete_1x2']}")
    print(f"  Coverage:           {p['complete_pct']:.1f}%")

    if p["incomplete_fixtures"]:
        print(f"\n  Incomplete fixtures ({len(p['incomplete_fixtures'])}):")
        for fid in p["incomplete_fixtures"][:20]:
            if fid in fixtures:
                f = fixtures[fid]
                print(f"    {fid}: {f['home_name']} vs {f['away_name']} "
                      f"({f['competition_name']} {f['season']})")
        if len(p["incomplete_fixtures"]) > 20:
            print(f"    ... and {len(p['incomplete_fixtures']) - 20} more")

    # Phase 3D — Bookmaker
    print("\n" + "=" * 60)
    print("PHASE 3D — BOOKMAKER AUDIT")
    print("=" * 60)
    bm = bookmaker_audit(odds, fixtures)

    print(f"\n  {'Bookmaker':<20} {'ID':>4} {'Fixtures':>9} {'Cov%':>6} "
          f"{'Open%':>6} {'Close%':>7}")
    print("  " + "-" * 56)
    for name, data in sorted(bm.items(), key=lambda x: -x[1]["fixture_count"]):
        print(
            f"  {name:<20} {data['bookmaker_id']:>4} "
            f"{data['fixture_count']:>9} {data['coverage_pct']:>5.1f}% "
            f"{data['opening_pct']:>5.1f}% {data['closing_pct']:>6.1f}%"
        )

    # Determine preferred bookmaker (Pinnacle if sufficient coverage)
    pinnacle_data = bm.get("Pinnacle", {})
    preferred_bm_id = None
    if pinnacle_data and pinnacle_data.get("coverage_pct", 0) >= 90:
        preferred_bm_id = pinnacle_data["bookmaker_id"]
        print(f"\n  PREFERRED BOOKMAKER: Pinnacle (id={preferred_bm_id}, "
              f"coverage={pinnacle_data['coverage_pct']:.1f}%)")
    else:
        # Pick highest coverage
        best = max(bm.items(), key=lambda x: x[1]["fixture_count"])
        preferred_bm_id = best[1]["bookmaker_id"]
        print(f"\n  Pinnacle coverage insufficient. "
              f"Using {best[0]} (id={preferred_bm_id}) as fallback.")

    # Phase 3E — Opening vs Closing
    print("\n" + "=" * 60)
    print("PHASE 3E — OPENING VS CLOSING AUDIT")
    print("=" * 60)
    oc = opening_closing_audit(odds)

    print(f"\n  Total records:  {oc['total_records']}")
    print(f"  Both available: {oc['both']} "
          f"({100*oc['both']/oc['total_records']:.1f}%)")
    print(f"  Opening only:   {oc['opening_only']}")
    print(f"  Closing only:   {oc['closing_only']}")
    print(f"  Neither:        {oc['neither']}")

    # Phase 3F — Causality
    print("\n" + "=" * 60)
    print("PHASE 3F — CAUSALITY / TEMPORAL SAFETY")
    print("=" * 60)
    causality = causality_audit()

    for field in ["opening", "closing", "peak"]:
        info = causality[field]
        print(f"\n  {field.upper()}:")
        print(f"    Classification: {info['classification']}")
        print(f"    Risk: {info['risk']}")

    print(f"\n  Independent timestamps: {causality['independent_timestamp_available']}")
    print(f"  Recommendation: {causality['recommendation']}")

    # Phase 3I — Build research dataset
    print("\n" + "=" * 60)
    print("PHASE 3I — BUILDING RESEARCH DATASET")
    print("=" * 60)
    count = build_research_dataset(
        fixtures, odds, cov["complete_fixtures"], preferred_bm_id,
    )
    print(f"\n  Research dataset: {RESEARCH_DB}")
    print(f"  Fixtures: {count}")
    print(f"  Preferred bookmaker ID: {preferred_bm_id}")

    # Save audit results
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "by_league": cov["by_league"],
            "by_season": cov["by_season"],
            "pooled": {k: v for k, v in p.items() if k != "incomplete_fixtures"},
            "incomplete_count": len(p["incomplete_fixtures"]),
        },
        "bookmakers": bm,
        "preferred_bookmaker_id": preferred_bm_id,
        "opening_closing": oc,
        "causality": {
            k: v for k, v in causality.items()
            if k not in ("evidence",)  # keep it JSON-friendly
        },
        "research_dataset": {
            "path": str(RESEARCH_DB),
            "fixture_count": count,
        },
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, default=str))
    print(f"\n  Audit results saved to: {AUDIT_JSON}")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
