"""Fresh-100 market odds ingestion — Phase 8 blocker resolution.

Usage:
    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python research/v4_promotion/ingest_fresh_100_market_odds.py

SCOPE — deliberately narrow
---------------------------
Fetches Pinnacle CLOSING 1X2 odds for EXACTLY the 100 fixture IDs already
frozen in fresh_100_fixture_ids.json. Nothing else. No other fixture, no
other bookmaker, no other market, no other price.

The frozen list is READ, never regenerated. This script contains no
fixture-selection logic at all, so it is structurally incapable of
reordering, filtering or cherry-picking the sample.

WHY A SEPARATE DATABASE
-----------------------
    research/v4_promotion/fresh_100_market_odds.sqlite   <- NEW, isolated

The existing market databases are opened READ-ONLY for verification and
are never written:

    research/v4_promotion/promotion_market_odds.sqlite   (20+50, 50 rows)
    research/market_odds/odds_history.sqlite             (E2, 50,094 rows)
    research/market_odds/research_dataset.sqlite         (E2, 4,001 rows)

METHODOLOGY — identical to the validated 20/50 ingests
------------------------------------------------------
  * bookmaker : Pinnacle only (bookmaker_id = 1), NO fallback
  * market    : ft_result (market_id = 6), i.e. 1X2
  * price     : CLOSING only — never peak, never opening
  * de-vig    : power method, imported verbatim from
                research/market_odds/devig.py
  * data_class: 'promotion-validation-only'

CAUSALITY
---------
Odds are pre-match prices. Match outcomes are never an input and are
never stored: this script issues no SQL selecting home_goals,
away_goals or label_result, and the output schema has no outcome column.

FAILURE POLICY
--------------
If ANY fixture cannot be fetched or yields incomplete/invalid odds, the
run reports the fixture ID, teams, date and exact reason, and exits
BLOCKED. Nothing is fabricated, substituted or silently dropped.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "market_odds"))

from devig import DevigError, power_devig  # noqa: E402  (E2 module, verbatim)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
ENV_FILE = PROJECT_ROOT / ".env"
FROZEN_IDS = HERE / "fresh_100_fixture_ids.json"
OUT_DB = HERE / "fresh_100_market_odds.sqlite"
MANIFEST = HERE / "fresh_100_market_ingest_manifest.json"

BASE_URL = "https://data.oddalerts.com/api"
MARKET_ID_FT_RESULT = 6
PINNACLE_ID = 1
DATA_CLASS = "promotion-validation-only"
PRICE_CLASS = "closing"
MARKET_KEY = "ft_result"

#: Existing market databases — verified read-only, never written.
EXISTING_MARKET_DBS = {
    "research/v4_promotion/promotion_market_odds.sqlite": None,   # filled at runtime
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
}
PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
#: Frozen 20/50 outputs that must not be disturbed.
FROZEN_OUTPUTS = [
    "v4_20_validation_results.json", "v4_20_validation_report.md",
    "v4_20_validation_manifest.json", "v4_50_validation_results.json",
    "v4_50_validation_report.md", "v4_50_validation_manifest.json",
]

MAX_RETRIES = 5
BACKOFF_BASE = 2.0
DELAY_BETWEEN = 1.0
PROB_TOL = 1e-9


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def blocked(msg: str, detail: str = ""):
    print(f"\n{'=' * 78}\nBLOCKED\n{'=' * 78}\n  {msg}")
    if detail:
        print(f"\n{detail}")
    raise SystemExit(1)


def load_token() -> str:
    tok = os.environ.get("OddAlerts_API")
    if tok:
        return tok.strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "OddAlerts_API":
                return v.strip()
    blocked(f"Missing OddAlerts_API in environment or {ENV_FILE}")


def _redact(s: str, tok: str) -> str:
    return s.replace(tok, "<REDACTED>") if tok else s


def load_frozen() -> tuple[list[int], dict]:
    """Read the frozen 100. This script never regenerates the selection."""
    if not FROZEN_IDS.exists():
        blocked(f"Frozen fixture list not found: {FROZEN_IDS}")
    d = json.loads(FROZEN_IDS.read_text(encoding="utf-8"))
    ids = [int(x) for x in d["fixture_ids"]]
    if len(ids) != 100:
        blocked(f"Frozen list holds {len(ids)} ids, expected 100")
    if len(set(ids)) != 100:
        blocked("Frozen list contains duplicate fixture ids")
    return ids, d


def fixture_metadata(ids: list[int]) -> dict[int, dict]:
    """Pre-match metadata only. No outcome column is selected."""
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT fixture_id, home_name, away_name, competition_name,
                   season, unix, date
            FROM fixtures WHERE fixture_id IN ({ph})""", ids).fetchall()
    conn.close()
    return {r[0]: {"fixture_id": r[0], "home_name": r[1], "away_name": r[2],
                   "competition_name": r[3], "season": r[4],
                   "unix": r[5], "date": r[6]} for r in rows}


def fetch_odds(session, fixture_id: int, token: str) -> list[dict]:
    """GET /odds/history/:ID filtered to Pinnacle + ft_result."""
    url = f"{BASE_URL}/odds/history/{fixture_id}"
    params = {"api_token": token, "markets": MARKET_ID_FT_RESULT,
              "bookmakers": PINNACLE_ID}
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=30)
        except Exception as exc:                            # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            d = BACKOFF_BASE ** attempt + random.uniform(0, 0.5)
            print(f"      attempt {attempt}/{MAX_RETRIES} connection error; "
                  f"retry in {d:.1f}s")
            time.sleep(d)
            continue
        if r.status_code == 200:
            try:
                return r.json().get("data", [])
            except ValueError as exc:
                raise RuntimeError(f"non-JSON 200 response: {exc}") from exc
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}"
            ra = r.headers.get("Retry-After")
            d = float(ra) if ra and ra.isdigit() else BACKOFF_BASE ** attempt
            print(f"      attempt {attempt}/{MAX_RETRIES} HTTP "
                  f"{r.status_code}; retry in {d:.1f}s")
            time.sleep(d)
            continue
        raise RuntimeError(
            f"HTTP {r.status_code}: {_redact(r.text[:200], token)}")
    raise RuntimeError(f"exhausted {MAX_RETRIES} retries; last error: {last}")


def _price(v):
    try:
        x = float(v)
        return x if x > 1.0 else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    print("=" * 78)
    print("FRESH 100 MARKET ODDS INGEST — Pinnacle closing 1X2, power de-vig")
    print("=" * 78)

    # ---- pre-flight integrity -------------------------------------------
    print("\n--- Pre-flight integrity ---")
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            blocked(f"protected file missing: {rel}")
        a = md5(p)
        ok = a == exp
        print(f"  [{'OK  ' if ok else 'FAIL'}] {rel}")
        if not ok:
            blocked(f"protected file changed: {rel}", f"expected {exp}, got {a}")

    old_market = PROJECT_ROOT / "research/v4_promotion/promotion_market_odds.sqlite"
    if not old_market.exists():
        blocked("existing promotion_market_odds.sqlite not found")
    old_md5_pre, old_sha_pre = md5(old_market), sha256(old_market)
    conn = sqlite3.connect(f"file:{old_market}?mode=ro", uri=True)
    old_rows_pre = conn.execute("SELECT COUNT(*) FROM promotion_market").fetchone()[0]
    conn.close()
    print(f"  [OK  ] existing market DB: {old_rows_pre} rows, md5={old_md5_pre}")

    for rel in ("research/market_odds/odds_history.sqlite",
                "research/market_odds/research_dataset.sqlite"):
        p = PROJECT_ROOT / rel
        a = md5(p)
        exp = EXISTING_MARKET_DBS[rel]
        ok = a == exp
        print(f"  [{'OK  ' if ok else 'FAIL'}] {rel}")
        if not ok:
            blocked(f"E2 database changed: {rel}")

    frozen_pre = {f: md5(HERE / f) for f in FROZEN_OUTPUTS if (HERE / f).exists()}
    print(f"  [OK  ] frozen 20/50 outputs snapshotted ({len(frozen_pre)} files)")

    # ---- frozen selection verification -----------------------------------
    print("\n--- Frozen fixture verification (read, never regenerated) ---")
    ids, frozen_doc = load_frozen()
    print(f"  frozen file md5      : {md5(FROZEN_IDS)}")
    print(f"  exactly 100          : {len(ids) == 100}")
    print(f"  zero duplicates      : {len(set(ids)) == 100}")

    meta = fixture_metadata(ids)
    missing_meta = [f for f in ids if f not in meta]
    if missing_meta:
        blocked(f"{len(missing_meta)} frozen ids absent from matches.db",
                str(missing_meta[:10]))

    u = [meta[f]["unix"] for f in ids]
    chrono = all(u[i] <= u[i + 1] for i in range(len(u) - 1))
    print(f"  chronological order  : {chrono}")
    if not chrono:
        blocked("frozen fixture list is not in chronological order")

    conn = sqlite3.connect(f"file:{old_market}?mode=ro", uri=True)
    old50 = set(r[0] for r in conn.execute("SELECT fixture_id FROM promotion_market"))
    conn.close()
    overlap = set(ids) & old50
    print(f"  overlap with old 50  : {len(overlap)}")
    if overlap:
        blocked(f"frozen 100 overlaps the existing 50: {sorted(overlap)[:10]}")

    if OUT_DB.exists():
        blocked(f"output database already exists: {OUT_DB}",
                "Refusing to overwrite. Remove it deliberately to re-ingest.")

    try:
        import requests
    except ImportError:
        blocked("'requests' is required for API access")

    token = load_token()
    print(f"  API token loaded     : length {len(token)}")

    # ---- create isolated output DB ---------------------------------------
    conn = sqlite3.connect(str(OUT_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fresh_100_market (
            fixture_id INTEGER PRIMARY KEY,
            home_name TEXT, away_name TEXT, competition_name TEXT,
            season TEXT, unix INTEGER, date TEXT,
            bookmaker_id INTEGER, bookmaker_name TEXT,
            market_key TEXT, market_id INTEGER,
            closing_home REAL, closing_draw REAL, closing_away REAL,
            p_home REAL, p_draw REAL, p_away REAL,
            devig_k REAL, overround REAL,
            price_class TEXT NOT NULL,
            data_class TEXT NOT NULL,
            source TEXT NOT NULL,
            retrieved_at TEXT NOT NULL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingest_log (
            fixture_id INTEGER PRIMARY KEY, seq INTEGER, status TEXT,
            n_records INTEGER, error TEXT, fetched_at TEXT)""")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()

    session = requests.Session()
    now = datetime.now(timezone.utc).isoformat()
    ok_n, failures = 0, []

    print(f"\n--- Fetching {len(ids)} fixtures "
          f"(Pinnacle {PINNACLE_ID}, market {MARKET_ID_FT_RESULT}, closing) ---")
    for seq, fid in enumerate(ids, 1):
        fx = meta[fid]
        label = f"{fx['home_name'][:18]} v {fx['away_name'][:18]}"
        try:
            recs = fetch_odds(session, fid, token)
        except RuntimeError as exc:
            reason = _redact(str(exc), token)
            failures.append({"seq": seq, "fixture_id": fid,
                             "home": fx["home_name"], "away": fx["away_name"],
                             "date": str(fx["date"])[:10],
                             "league": fx["competition_name"],
                             "reason": f"API failure: {reason}"})
            conn.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?,?,?)",
                         (fid, seq, "api_error", 0, reason, now))
            conn.commit()
            print(f"  {seq:>3}. {fid} FAILED  {label} — {reason[:60]}")
            time.sleep(DELAY_BETWEEN)
            continue

        by_out = {r.get("outcome"): r for r in recs
                  if r.get("market_id") == MARKET_ID_FT_RESULT
                  and r.get("bookmaker_id") == PINNACLE_ID}
        absent = [o for o in ("home", "draw", "away") if o not in by_out]
        ch = _price(by_out.get("home", {}).get("closing"))
        cd = _price(by_out.get("draw", {}).get("closing"))
        ca = _price(by_out.get("away", {}).get("closing"))

        if absent or None in (ch, cd, ca):
            reason = (f"missing Pinnacle outcomes {absent}" if absent
                      else "invalid or absent closing price")
            failures.append({"seq": seq, "fixture_id": fid,
                             "home": fx["home_name"], "away": fx["away_name"],
                             "date": str(fx["date"])[:10],
                             "league": fx["competition_name"],
                             "reason": reason})
            conn.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?,?,?)",
                         (fid, seq, "incomplete", len(recs), reason, now))
            conn.commit()
            print(f"  {seq:>3}. {fid} INCOMPLETE  {label} — {reason}")
            time.sleep(DELAY_BETWEEN)
            continue

        try:
            dv = power_devig(ch, cd, ca)
        except DevigError as exc:
            failures.append({"seq": seq, "fixture_id": fid,
                             "home": fx["home_name"], "away": fx["away_name"],
                             "date": str(fx["date"])[:10],
                             "league": fx["competition_name"],
                             "reason": f"de-vig failed: {exc}"})
            conn.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?,?,?)",
                         (fid, seq, "devig_error", len(recs), str(exc), now))
            conn.commit()
            print(f"  {seq:>3}. {fid} DEVIG FAIL  {label} — {exc}")
            time.sleep(DELAY_BETWEEN)
            continue

        conn.execute(
            "INSERT OR REPLACE INTO fresh_100_market VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fid, fx["home_name"], fx["away_name"], fx["competition_name"],
             fx["season"], fx["unix"], fx["date"],
             PINNACLE_ID, by_out["home"].get("bookmaker_name", "Pinnacle"),
             MARKET_KEY, MARKET_ID_FT_RESULT,
             ch, cd, ca, dv.p_home, dv.p_draw, dv.p_away, dv.k, dv.overround,
             PRICE_CLASS, DATA_CLASS, "OddAlerts odds/history", now))
        conn.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?,?,?)",
                     (fid, seq, "success", len(recs), None, now))
        conn.commit()
        ok_n += 1
        print(f"  {seq:>3}. {fid} OK  {label:<40} "
              f"{ch:>6.2f}/{cd:>5.2f}/{ca:>6.2f} -> "
              f"{dv.p_home:.4f}/{dv.p_draw:.4f}/{dv.p_away:.4f}")
        time.sleep(DELAY_BETWEEN)

    # ---- failure policy ---------------------------------------------------
    if failures:
        lines = [f"  {'#':>4} {'fixture_id':>11} {'date':<11} {'league':<15} "
                 f"{'match':<40} reason", "  " + "-" * 110]
        for f in failures:
            lines.append(
                f"  {f['seq']:>4} {f['fixture_id']:>11} {f['date']:<11} "
                f"{f['league'][:14]:<15} "
                f"{f['home'][:18] + ' v ' + f['away'][:18]:<40} {f['reason']}")
        conn.close()
        blocked(f"{len(failures)}/100 fixtures could not be ingested. "
                "Nothing fabricated, substituted or dropped.",
                "\n".join(lines))

    # ---- strict validation -------------------------------------------------
    print("\n--- Strict validation ---")
    rows = conn.execute(
        "SELECT fixture_id, p_home, p_draw, p_away, bookmaker_name, "
        "price_class, data_class, market_key, market_id, "
        "closing_home, closing_draw, closing_away FROM fresh_100_market"
    ).fetchall()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fresh_100_market)")]

    coverage = len(rows) == 100 and {r[0] for r in rows} == set(ids)
    pinnacle_only = {r[4] for r in rows} == {"Pinnacle"}
    closing_only = {r[5] for r in rows} == {PRICE_CLASS}
    class_ok = {r[6] for r in rows} == {DATA_CLASS}
    market_ok = ({r[7] for r in rows} == {MARKET_KEY}
                 and {r[8] for r in rows} == {MARKET_ID_FT_RESULT})
    worst = max(abs((r[1] + r[2] + r[3]) - 1.0) for r in rows) if rows else 1.0
    sums_ok = worst < PROB_TOL
    devig_ok = all(0.0 < r[1] < 1.0 and 0.0 < r[2] < 1.0 and 0.0 < r[3] < 1.0
                   and r[9] > 1.0 and r[10] > 1.0 and r[11] > 1.0 for r in rows)
    outcome_cols = ({"home_goals", "away_goals", "label_result", "result",
                     "winning_team", "score"} & set(cols))
    no_outcome = not outcome_cols

    print(f"  coverage 100/100        : {coverage} ({len(rows)} rows)")
    print(f"  Pinnacle only           : {pinnacle_only}")
    print(f"  closing only            : {closing_only}")
    print(f"  1X2 (ft_result) only    : {market_ok}")
    print(f"  data_class correct      : {class_ok}")
    print(f"  de-vig valid            : {devig_ok}")
    print(f"  probability sums        : {sums_ok} (max deviation {worst:.3e})")
    print(f"  no outcome columns      : {no_outcome} ({len(cols)} columns)")

    for k, v in [("data_class", DATA_CLASS), ("bookmaker", "Pinnacle"),
                 ("price", PRICE_CLASS), ("market", MARKET_KEY),
                 ("devig", "power method (research/market_odds/devig.py)"),
                 ("purpose", "fresh 100-match OOS validation reference"),
                 ("frozen_ids_md5", md5(FROZEN_IDS)),
                 ("existing_dbs_modified", "false"),
                 ("created_at", now)]:
        conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))
    conn.commit()
    conn.close()

    out_md5, out_sha = md5(OUT_DB), sha256(OUT_DB)
    print(f"\n  output DB: {OUT_DB.name}")
    print(f"  md5      : {out_md5}")
    print(f"  bytes    : {OUT_DB.stat().st_size}")

    # ---- post-ingest integrity --------------------------------------------
    print("\n--- Post-ingest integrity ---")
    old_md5_post = md5(old_market)
    conn = sqlite3.connect(f"file:{old_market}?mode=ro", uri=True)
    old_rows_post = conn.execute("SELECT COUNT(*) FROM promotion_market").fetchone()[0]
    conn.close()
    old_unchanged = (old_md5_post == old_md5_pre
                     and old_rows_post == old_rows_pre)
    print(f"  old 50 market DB unchanged: {old_unchanged} "
          f"({old_rows_post} rows, md5={old_md5_post})")

    protected_ok = True
    for rel, exp in PROTECTED_FILES.items():
        a = md5(PROJECT_ROOT / rel)
        if a != exp:
            protected_ok = False
            print(f"  [FAIL] {rel}")
    for rel in ("research/market_odds/odds_history.sqlite",
                "research/market_odds/research_dataset.sqlite"):
        if md5(PROJECT_ROOT / rel) != EXISTING_MARKET_DBS[rel]:
            protected_ok = False
            print(f"  [FAIL] {rel}")
    print(f"  protected files unchanged : {protected_ok}")

    frozen_post = {f: md5(HERE / f) for f in frozen_pre}
    frozen_ok = frozen_pre == frozen_post
    print(f"  frozen 20/50 outputs      : {frozen_ok}")

    overall = (coverage and pinnacle_only and closing_only and market_ok
               and class_ok and devig_ok and sums_ok and no_outcome
               and old_unchanged and protected_ok and frozen_ok)

    MANIFEST.write_text(json.dumps({
        "generated_at": now,
        "purpose": "Fresh 100-match OOS market reference (Phase 8 blocker)",
        "frozen_ids_file": FROZEN_IDS.name,
        "frozen_ids_md5": md5(FROZEN_IDS),
        "output_database": OUT_DB.name,
        "output_md5": out_md5, "output_sha256": out_sha,
        "output_bytes": OUT_DB.stat().st_size,
        "methodology": {"bookmaker": "Pinnacle", "bookmaker_id": PINNACLE_ID,
                        "market": MARKET_KEY, "market_id": MARKET_ID_FT_RESULT,
                        "price": PRICE_CLASS, "peak_used": False,
                        "opening_used": False, "fallback_bookmaker": False,
                        "devig": "power method, E2 module verbatim",
                        "data_class": DATA_CLASS},
        "results": {"fixtures": 100, "success": ok_n,
                    "failed": len(failures), "failures": failures},
        "validation": {"coverage_100_of_100": coverage,
                       "pinnacle_only": pinnacle_only,
                       "closing_only": closing_only,
                       "market_1x2_only": market_ok,
                       "data_class_correct": class_ok,
                       "devig_valid": devig_ok,
                       "probability_sums_ok": sums_ok,
                       "max_sum_deviation": worst,
                       "no_outcome_columns": no_outcome},
        "integrity": {"old_50_market_db_unchanged": old_unchanged,
                      "old_50_md5_pre": old_md5_pre,
                      "old_50_md5_post": old_md5_post,
                      "old_50_rows_pre": old_rows_pre,
                      "old_50_rows_post": old_rows_post,
                      "protected_files_unchanged": protected_ok,
                      "frozen_20_50_outputs_unchanged": frozen_ok},
        "overall_passed": overall,
    }, indent=2, default=str))
    print(f"\n  manifest -> {MANIFEST.name}")

    def mark(b): return "PASS" if b else "FAIL"
    print("\n" + "=" * 78)
    print("FRESH 100 MARKET ODDS INGEST")
    print("=" * 78)
    print(f"FIXTURES: 100")
    print(f"SUCCESS: {ok_n}/100")
    print(f"FAILED: {len(failures)}/100")
    print(f"PINNACLE ONLY: {mark(pinnacle_only)}")
    print(f"CLOSING ONLY: {mark(closing_only)}")
    print(f"1X2 ONLY: {mark(market_ok)}")
    print(f"DE-VIG: {mark(devig_ok)}")
    print(f"PROBABILITY SUMS: {mark(sums_ok)}")
    print(f"NO OUTCOME COLUMNS: {mark(no_outcome)}")
    print(f"OLD 50 MARKET DB UNCHANGED: {mark(old_unchanged)}")
    print(f"PROTECTED FILES UNCHANGED: {mark(protected_ok)}")
    print(f"OVERALL: {'PASS' if overall else 'BLOCKED'}")

    if overall:
        print("\nPhase 8 blocker RESOLVED. Market reference ready for the")
        print("fresh-100 evaluation. Harness NOT built, predictions NOT run.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
