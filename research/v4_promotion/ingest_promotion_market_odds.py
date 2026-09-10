"""V4 Step 1-2 — Promotion-validation-only market odds ingestion.

Usage:
    python research/v4_promotion/ingest_promotion_market_odds.py

SCOPE — deliberately narrow
---------------------------
Fetches Pinnacle CLOSING 1X2 odds for EXACTLY the 20 already-selected
2025/26 validation fixtures. Nothing else. No other season, no other
fixture, no other bookmaker, no other market.

WHY A SEPARATE DATABASE
-----------------------
The historical E2 research artefacts must remain reproducible and
byte-identical:

    research/market_odds/odds_history.sqlite        (E2, 50,094 rows)
    research/market_odds/research_dataset.sqlite    (E2, 4,001 rows)

This script NEVER opens either of those for writing. It creates a new,
clearly-labelled database:

    research/v4_promotion/promotion_market_odds.sqlite

Every row carries `data_class = 'promotion-validation-only'` so the
provenance can never be confused with the E2 research universe.

METHODOLOGY — identical to E2, reused not reinvented
----------------------------------------------------
  * bookmaker : Pinnacle only (bookmaker_id = 1)
  * market    : ft_result (market_id = 6)
  * price     : CLOSING only
  * de-vig    : power method, imported verbatim from
                research/market_odds/devig.py
  * NOT used  : peak odds, opening odds (primary signal), fallbacks

CAUSALITY
---------
Odds are pre-match prices. Match outcomes are NEVER an input: this
script does not read home_goals or away_goals at all. Step 2's
adversarial test rewrites all 20 outcomes and asserts the resulting
probabilities are bit-identical.
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

from devig import DevigError, power_devig  # noqa: E402  (E2's module, verbatim)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
ENV_FILE = PROJECT_ROOT / ".env"
OUT_DB = HERE / "promotion_market_odds.sqlite"
MANIFEST = HERE / "promotion_market_ingest_manifest.json"

BASE_URL = "https://data.oddalerts.com/api"
MARKET_ID_FT_RESULT = 6
PINNACLE_ID = 1
DATA_CLASS = "promotion-validation-only"

#: E2 research artefacts — read-only reference, never written by this script.
E2_FILES = {
    "research/market_odds/odds_history.sqlite":
        "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite":
        "bdab370ffdfe5bbf8ff3a8a26e64471c",
}
PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}

MAX_RETRIES = 5
BACKOFF_BASE = 2.0
DELAY_BETWEEN = 1.0


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


def check_files(label: str, spec: dict) -> dict:
    print(f"\n--- {label} ---")
    out = {}
    for rel, exp in spec.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            out[rel] = {"status": "missing"}
            print(f"  MISSING  {rel}")
            continue
        a = md5(p)
        ok = a == exp
        out[rel] = {"expected": exp, "actual": a, "sha256": sha256(p),
                    "status": "identical" if ok else "CHANGED"}
        print(f"  {'OK  ' if ok else 'FAIL'}  {rel}  {a}")
    return out


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
    raise RuntimeError(f"Missing OddAlerts_API in environment or {ENV_FILE}")


def _redact(s: str, tok: str) -> str:
    return s.replace(tok, "<REDACTED>") if tok else s


def the_twenty() -> list[dict]:
    """The exact 20 fixtures, by the documented deterministic rule.

    Selection rule from predict_blind_2025_26.py:
        season_id IN (2025/26) AND status = 'FT'
        ORDER BY unix ASC, fixture_id ASC LIMIT 20

    Outcome columns are deliberately NOT selected here.
    """
    from models.config import SEASON_NAME_TO_IDS
    ids = SEASON_NAME_TO_IDS["2025/2026"]
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    q = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT fixture_id, home_name, away_name, competition_name,
                   season, unix, date
            FROM fixtures WHERE season_id IN ({q}) AND status = 'FT'
            ORDER BY unix ASC, fixture_id ASC LIMIT 20""", list(ids)).fetchall()
    conn.close()
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 fixtures, got {len(rows)}")
    return [{"fixture_id": r[0], "home_name": r[1], "away_name": r[2],
             "competition_name": r[3], "season": r[4], "unix": r[5],
             "date": r[6]} for r in rows]


def fetch_odds(session, fixture_id: int, token: str) -> list[dict]:
    """GET /odds/history/:ID filtered to Pinnacle + ft_result."""
    url = f"{BASE_URL}/odds/history/{fixture_id}"
    params = {"api_token": token, "markets": MARKET_ID_FT_RESULT,
              "bookmakers": PINNACLE_ID}
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=30)
        except Exception as exc:                      # noqa: BLE001
            last = exc
            d = BACKOFF_BASE ** attempt + random.uniform(0, 0.5)
            print(f"    attempt {attempt}/{MAX_RETRIES} connection error; "
                  f"retry in {d:.1f}s")
            time.sleep(d)
            continue
        if r.status_code == 200:
            return r.json().get("data", [])
        if r.status_code in (429, 500, 502, 503, 504):
            last = RuntimeError(f"HTTP {r.status_code}")
            ra = r.headers.get("Retry-After")
            d = float(ra) if ra and ra.isdigit() else BACKOFF_BASE ** attempt
            print(f"    attempt {attempt}/{MAX_RETRIES} HTTP {r.status_code}; "
                  f"retry in {d:.1f}s")
            time.sleep(d)
            continue
        raise RuntimeError(
            f"HTTP {r.status_code} for fixture {fixture_id}: "
            f"{_redact(r.text[:200], token)}")
    raise RuntimeError(f"Exhausted retries for {fixture_id}: "
                       f"{_redact(str(last), token)}")


def _f(v):
    try:
        x = float(v)
        return x if x > 1.0 else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    print("=" * 76)
    print("V4 — Promotion-validation-only market odds ingestion (20 fixtures)")
    print("=" * 76)

    pre_protected = check_files("Protected files (PRE)", PROTECTED_FILES)
    pre_e2 = check_files("E2 research artefacts (PRE)", E2_FILES)
    if any(v.get("status") != "identical" for v in pre_protected.values()):
        print("\n  STOP: a protected file is not identical.")
        return 1
    if any(v.get("status") != "identical" for v in pre_e2.values()):
        print("\n  STOP: an E2 research artefact is not identical.")
        return 1

    try:
        import requests
    except ImportError:
        print("\n  STOP: 'requests' is required.")
        return 1

    token = load_token()
    print(f"\n  API token loaded (length {len(token)})")

    fixtures = the_twenty()
    print(f"  Target fixtures: {len(fixtures)} "
          f"({fixtures[0]['date'][:10]} .. {fixtures[-1]['date'][:10]})")

    conn = sqlite3.connect(str(OUT_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS promotion_market (
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
            fixture_id INTEGER PRIMARY KEY, status TEXT,
            n_records INTEGER, error TEXT, fetched_at TEXT)""")
    conn.commit()

    session = requests.Session()
    now = datetime.now(timezone.utc).isoformat()
    ok_n, fail_n = 0, 0

    print("\n--- Fetching (Pinnacle, ft_result, closing only) ---")
    for i, fx in enumerate(fixtures, 1):
        fid = fx["fixture_id"]
        try:
            recs = fetch_odds(session, fid, token)
        except RuntimeError as exc:
            print(f"  {i:>2}. {fid} FAILED: {_redact(str(exc), token)[:80]}")
            conn.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?,?)",
                         (fid, "error", 0, _redact(str(exc), token), now))
            conn.commit(); fail_n += 1
            time.sleep(DELAY_BETWEEN); continue

        by_out = {}
        for r in recs:
            if (r.get("market_id") == MARKET_ID_FT_RESULT
                    and r.get("bookmaker_id") == PINNACLE_ID):
                by_out[r.get("outcome")] = r

        missing = [o for o in ("home", "draw", "away") if o not in by_out]
        ch = _f(by_out.get("home", {}).get("closing"))
        cd = _f(by_out.get("draw", {}).get("closing"))
        ca = _f(by_out.get("away", {}).get("closing"))

        if missing or None in (ch, cd, ca):
            reason = (f"missing outcomes {missing}" if missing
                      else "invalid closing price")
            print(f"  {i:>2}. {fid} INCOMPLETE: {reason}")
            conn.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?,?)",
                         (fid, "incomplete", len(recs), reason, now))
            conn.commit(); fail_n += 1
            time.sleep(DELAY_BETWEEN); continue

        try:
            dv = power_devig(ch, cd, ca)
        except DevigError as exc:
            print(f"  {i:>2}. {fid} DEVIG FAILED: {exc}")
            conn.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?,?)",
                         (fid, "devig_error", len(recs), str(exc), now))
            conn.commit(); fail_n += 1
            time.sleep(DELAY_BETWEEN); continue

        conn.execute(
            "INSERT OR REPLACE INTO promotion_market VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fid, fx["home_name"], fx["away_name"], fx["competition_name"],
             fx["season"], fx["unix"], fx["date"],
             PINNACLE_ID, by_out["home"].get("bookmaker_name", "Pinnacle"),
             "ft_result", MARKET_ID_FT_RESULT,
             ch, cd, ca,
             dv.p_home, dv.p_draw, dv.p_away, dv.k, dv.overround,
             "closing", DATA_CLASS, "OddAlerts odds/history", now))
        conn.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?,?)",
                     (fid, "success", len(recs), None, now))
        conn.commit(); ok_n += 1
        print(f"  {i:>2}. {fid} OK  {ch:.2f}/{cd:.2f}/{ca:.2f}  ->  "
              f"{dv.p_home:.4f}/{dv.p_draw:.4f}/{dv.p_away:.4f}  "
              f"(k={dv.k:.4f}, overround={dv.overround:.4f})")
        time.sleep(DELAY_BETWEEN)

    print(f"\n  Success: {ok_n}/20   Failed: {fail_n}/20")

    # ---- verification ---------------------------------------------------
    print("\n--- Verification ---")
    rows = conn.execute(
        "SELECT fixture_id, p_home, p_draw, p_away, data_class, price_class "
        "FROM promotion_market").fetchall()
    coverage_ok = len(rows) == 20
    print(f"  20/20 coverage: {coverage_ok} ({len(rows)} rows)")

    worst = 0.0
    for r in rows:
        worst = max(worst, abs((r[1] + r[2] + r[3]) - 1.0))
    sums_ok = worst < 1e-9
    print(f"  probability sums == 1: {sums_ok} (max deviation {worst:.3e})")

    labelled = all(r[4] == DATA_CLASS for r in rows)
    closing = all(r[5] == "closing" for r in rows)
    print(f"  all rows labelled '{DATA_CLASS}': {labelled}")
    print(f"  all rows price_class 'closing': {closing}")

    # This script must never QUERY outcome columns. Inspect the SQL it
    # actually issues rather than scanning raw text -- the docstring
    # legitimately names those columns while explaining their absence.
    import ast
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    sql = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and any(k in n.value.upper()
                   for k in ("SELECT ", "FROM FIXTURES", "INSERT INTO"))]
    no_outcome = not any(("home_goals" in q or "away_goals" in q
                          or "label_result" in q) for q in sql)
    print(f"  no ingestion SQL selects an outcome column: {no_outcome} "
          f"({len(sql)} statements checked)")

    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    for k, v in [("data_class", DATA_CLASS),
                 ("purpose", "V4 promotion validation of 20 fixtures only"),
                 ("bookmaker", "Pinnacle"), ("price", "closing"),
                 ("devig", "power method (research/market_odds/devig.py)"),
                 ("e2_datasets_modified", "false"),
                 ("created_at", now)]:
        conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))
    conn.commit(); conn.close()

    post_protected = check_files("Protected files (POST)", PROTECTED_FILES)
    post_e2 = check_files("E2 research artefacts (POST)", E2_FILES)
    e2_intact = all(v.get("status") == "identical" for v in post_e2.values())
    prot_intact = all(v.get("status") == "identical"
                      for v in post_protected.values())

    all_ok = (coverage_ok and sums_ok and labelled and closing
              and no_outcome and e2_intact and prot_intact)

    MANIFEST.write_text(json.dumps({
        "generated_at": now,
        "purpose": "V4 promotion validation — 2025/26 market odds for 20 fixtures",
        "data_class": DATA_CLASS,
        "output_database": str(OUT_DB),
        "methodology": {"bookmaker": "Pinnacle", "bookmaker_id": PINNACLE_ID,
                        "market": "ft_result", "market_id": MARKET_ID_FT_RESULT,
                        "price": "closing", "peak_used": False,
                        "opening_used": False, "fallback_used": False,
                        "devig": "power method, E2 module verbatim"},
        "fixtures_targeted": 20, "fixtures_ingested": ok_n,
        "fixtures_failed": fail_n,
        "verification": {"coverage_20_of_20": coverage_ok,
                         "probability_sums_ok": sums_ok,
                         "max_sum_deviation": worst,
                         "all_rows_labelled": labelled,
                         "all_closing": closing,
                         "no_outcome_reference": no_outcome},
        "e2_artefacts_pre": pre_e2, "e2_artefacts_post": post_e2,
        "e2_artefacts_unchanged": e2_intact,
        "protected_pre": pre_protected, "protected_post": post_protected,
        "protected_unchanged": prot_intact,
        "overall_passed": all_ok,
    }, indent=2, default=str))
    print(f"\n  Manifest -> {MANIFEST}")

    print("\n" + "=" * 76)
    print(f"INGESTION: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 76)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
