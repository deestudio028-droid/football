"""50-Match Market Odds Extension — Ingest fixtures #21–#50.

SCOPE — deliberate extension of the 20-fixture ingest
------------------------------------------------------
Fetches Pinnacle CLOSING 1X2 odds for EXACTLY the 30 additional
fixtures (#21–#50) from the deterministic 50-fixture selection.

The existing 20 rows in promotion_market_odds.sqlite are NOT modified.

METHODOLOGY — identical to the validated 20-fixture ingest
----------------------------------------------------------
  * bookmaker : Pinnacle only (bookmaker_id = 1)
  * market    : ft_result (market_id = 6)
  * price     : CLOSING only
  * de-vig    : power method, imported verbatim from
                research/market_odds/devig.py
  * NOT used  : peak odds, opening odds, fallbacks
  * data_class: 'promotion-validation-only'

STRICT RULES ENFORCED
---------------------
  * DO NOT modify ingest_promotion_market_odds.py
  * DO NOT refetch or overwrite the original 20 rows
  * DO NOT modify any protected production artifact
  * If any fixture fails, STOP immediately
  * Require 50/50 coverage at exit
  * All probabilities must sum to 1

Usage:
    python research/v4_promotion/ingest_50_market_odds.py
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

BASE_URL = "https://data.oddalerts.com/api"
MARKET_ID_FT_RESULT = 6
PINNACLE_ID = 1
DATA_CLASS = "promotion-validation-only"

# Protected production files — must remain byte-identical.
PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
E2_FILES = {
    "research/market_odds/odds_history.sqlite":
        "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite":
        "bdab370ffdfe5bbf8ff3a8a26e64471c",
}

# The original 20 fixture IDs from the validated 20-match harness.
ORIGINAL_20_FIXTURE_IDS = [
    343465962, 343465735, 342254747, 343465963, 342863654,
    342863842, 342863843, 342863844, 343465733, 342864234,
    343465729, 342864255, 343465732, 342863998, 342864289,
    343465726, 343465727, 343465734, 343465842, 343465728,
]

# The 30 new fixture IDs (#21–#50) from the deterministic selection.
NEW_30_FIXTURE_IDS = [
    343465730, 343465731, 343465921, 343465941, 343465961,
    343465965, 344184009, 344184010, 345052043, 347217733,
    348751783, 347217747, 347217750, 347953543, 347953714,
    347953715, 347953716, 347953717, 347953718, 347953747,
    347953748, 347953749, 347954243, 347954246, 347954613,
    347954614, 347954615, 347954616, 347954621, 347954642,
]

# All 50 fixture IDs.
ALL_50_FIXTURE_IDS = ORIGINAL_20_FIXTURE_IDS + NEW_30_FIXTURE_IDS

# SHA-256 of the original 20 rows content (computed during pre-audit).
EXPECTED_ORIGINAL_20_SHA256 = "236c5c6591ca78df83473caeec423baa18b0ddadd270938997a1c5b1aed06b27"

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


def _content_sha256_of_rows(rows: list) -> str:
    """Compute SHA-256 hash of row contents for comparison."""
    h = hashlib.sha256()
    for r in rows:
        h.update(str(r).encode())
    return h.hexdigest()


def stop(msg: str):
    print("\n" + "!" * 78)
    print("STOP -- Integrity or validation precondition failure.")
    print(msg)
    print("!" * 78)
    sys.exit(1)


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


def _f(v):
    try:
        x = float(v)
        return x if x > 1.0 else None
    except (TypeError, ValueError):
        return None


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


def the_thirty() -> list[dict]:
    """The exact 30 new fixtures (#21–#50), by the deterministic rule.

    Selection rule: same as the 20-match harness but LIMIT 50,
    then take rows [20:50] (fixtures #21–#50).

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
            ORDER BY unix ASC, fixture_id ASC LIMIT 50""", list(ids)).fetchall()
    conn.close()
    if len(rows) < 50:
        raise RuntimeError(f"Expected >= 50 fixtures, got {len(rows)}")

    # Verify first 20 match registered IDs
    first_20 = [r[0] for r in rows[:20]]
    if first_20 != ORIGINAL_20_FIXTURE_IDS:
        raise RuntimeError(
            f"First 20 fixture IDs don't match registered list.\n"
            f"Got: {first_20}\nExpected: {ORIGINAL_20_FIXTURE_IDS}")

    # Extract #21–#50
    new_30 = rows[20:50]
    new_30_ids = [r[0] for r in new_30]
    if new_30_ids != NEW_30_FIXTURE_IDS:
        raise RuntimeError(
            f"Fixture IDs #21–#50 don't match registered list.\n"
            f"Got: {new_30_ids}\nExpected: {NEW_30_FIXTURE_IDS}")

    return [{"fixture_id": r[0], "home_name": r[1], "away_name": r[2],
             "competition_name": r[3], "season": r[4], "unix": r[5],
             "date": r[6]} for r in new_30]


def snapshot_original_20(conn: sqlite3.Connection) -> tuple[list, str]:
    """Read and hash the existing 20 rows for comparison."""
    rows = conn.execute(
        "SELECT * FROM promotion_market ORDER BY fixture_id").fetchall()
    content_hash = _content_sha256_of_rows(rows)
    return rows, content_hash


def main() -> int:
    print("=" * 78)
    print("50-MATCH MARKET ODDS EXTENSION — Ingest fixtures #21–#50")
    print("STRICT: Pinnacle closing 1X2, power de-vig, no outcome columns")
    print("=" * 78)

    # ---- PHASE 1: Protected file integrity (PRE) ----
    print("\n" + "=" * 78)
    print("PHASE 1 — PROTECTED FILE INTEGRITY (PRE)")
    print("=" * 78)
    pre_protected = check_files("Protected files (PRE)", PROTECTED_FILES)
    pre_e2 = check_files("E2 research artefacts (PRE)", E2_FILES)
    if any(v.get("status") != "identical" for v in pre_protected.values()):
        stop("A protected file is not identical.")
    if any(v.get("status") != "identical" for v in pre_e2.values()):
        stop("An E2 research artefact is not identical.")

    # ---- PHASE 2: Snapshot existing 20 rows ----
    print("\n" + "=" * 78)
    print("PHASE 2 — SNAPSHOT EXISTING 20 MARKET ROWS")
    print("=" * 78)
    if not OUT_DB.exists():
        stop(f"Market database not found: {OUT_DB}")

    conn = sqlite3.connect(str(OUT_DB))
    orig_rows, orig_hash = snapshot_original_20(conn)
    if len(orig_rows) != 20:
        stop(f"Expected 20 existing rows, got {len(orig_rows)}")

    orig_fids = sorted([r[0] for r in orig_rows])
    expected_fids = sorted(ORIGINAL_20_FIXTURE_IDS)
    if orig_fids != expected_fids:
        stop(f"Existing fixture IDs don't match registered 20.\n"
             f"Got: {orig_fids}\nExpected: {expected_fids}")

    if orig_hash != EXPECTED_ORIGINAL_20_SHA256:
        stop(f"Original 20 rows content hash mismatch.\n"
             f"Expected: {EXPECTED_ORIGINAL_20_SHA256}\n"
             f"Got:      {orig_hash}")

    print(f"  Existing rows: {len(orig_rows)}")
    print(f"  Content SHA-256: {orig_hash}")
    print(f"  Fixture IDs match registered 20: PASS")

    # Check no new fixtures already exist
    for fid in NEW_30_FIXTURE_IDS:
        existing = conn.execute(
            "SELECT COUNT(*) FROM promotion_market WHERE fixture_id=?",
            (fid,)).fetchone()[0]
        if existing > 0:
            stop(f"Fixture {fid} already exists in promotion_market!")
    print(f"  No overlap with new 30 fixture IDs: PASS")

    # ---- PHASE 3: Verify fixture list ----
    print("\n" + "=" * 78)
    print("PHASE 3 — VERIFY 30 FIXTURE IDS (#21–#50)")
    print("=" * 78)
    fixtures = the_thirty()
    print(f"  Target fixtures: {len(fixtures)}")
    for i, fx in enumerate(fixtures, 21):
        print(f"  #{i:2d}: {fx['fixture_id']}  "
              f"{str(fx['date'])[:19]}  "
              f"{fx['home_name'][:22]:22s} vs {fx['away_name'][:22]:22s}")

    # ---- PHASE 4: Fetch and ingest ----
    print("\n" + "=" * 78)
    print("PHASE 4 — FETCH PINNACLE CLOSING ODDS (30 fixtures)")
    print("=" * 78)

    try:
        import requests
    except ImportError:
        stop("'requests' is required.")

    token = load_token()
    print(f"  API token loaded (length {len(token)})")

    session = requests.Session()
    now = datetime.now(timezone.utc).isoformat()
    ok_n, fail_n = 0, 0

    # Ensure ingest_log table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingest_log (
            fixture_id INTEGER PRIMARY KEY, status TEXT,
            n_records INTEGER, error TEXT, fetched_at TEXT)""")
    conn.commit()

    print("\n--- Fetching (Pinnacle, ft_result, closing only) ---")
    for i, fx in enumerate(fixtures, 21):
        fid = fx["fixture_id"]

        # Safety: never overwrite original 20
        if fid in ORIGINAL_20_FIXTURE_IDS:
            stop(f"Fixture {fid} is in the original 20! Refusing to overwrite.")

        try:
            recs = fetch_odds(session, fid, token)
        except RuntimeError as exc:
            msg = _redact(str(exc), token)[:120]
            print(f"  #{i:2d}. {fid} FAILED: {msg}")
            stop(f"Fixture {fid} failed ingestion: {msg}")

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
            stop(f"Fixture {fid} INCOMPLETE: {reason}. "
                 f"Cannot continue with <50/50 coverage.")

        try:
            dv = power_devig(ch, cd, ca)
        except DevigError as exc:
            stop(f"Fixture {fid} DEVIG FAILED: {exc}")

        # Verify probability sum
        p_sum = dv.p_home + dv.p_draw + dv.p_away
        if abs(p_sum - 1.0) > 1e-9:
            stop(f"Fixture {fid}: probability sum {p_sum} != 1.0")

        conn.execute(
            "INSERT INTO promotion_market VALUES "
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
        conn.commit()
        ok_n += 1
        print(f"  #{i:2d}. {fid} OK  {ch:.2f}/{cd:.2f}/{ca:.2f}  ->  "
              f"{dv.p_home:.4f}/{dv.p_draw:.4f}/{dv.p_away:.4f}  "
              f"(k={dv.k:.4f}, overround={dv.overround:.4f})")
        time.sleep(DELAY_BETWEEN)

    print(f"\n  Success: {ok_n}/30   Failed: {fail_n}/30")

    if ok_n != 30:
        stop(f"Only {ok_n}/30 fixtures ingested successfully. "
             f"Cannot continue with <50/50 coverage.")

    # ---- PHASE 5: Verification ----
    print("\n" + "=" * 78)
    print("PHASE 5 — VERIFICATION")
    print("=" * 78)

    # 5a. Total coverage = 50
    all_rows = conn.execute(
        "SELECT fixture_id, p_home, p_draw, p_away, data_class, "
        "price_class, bookmaker_id FROM promotion_market "
        "ORDER BY fixture_id").fetchall()
    coverage_ok = len(all_rows) == 50
    print(f"  50/50 coverage: {coverage_ok} ({len(all_rows)} rows)")
    if not coverage_ok:
        stop(f"Expected 50 rows, got {len(all_rows)}")

    # 5b. All fixture IDs present
    db_fids = sorted([r[0] for r in all_rows])
    expected_all = sorted(ALL_50_FIXTURE_IDS)
    fid_match = db_fids == expected_all
    print(f"  All 50 fixture IDs match: {fid_match}")
    if not fid_match:
        missing_fids = set(expected_all) - set(db_fids)
        extra_fids = set(db_fids) - set(expected_all)
        stop(f"Fixture ID mismatch. Missing: {missing_fids}, Extra: {extra_fids}")

    # 5c. Probability sums = 1
    worst = 0.0
    for r in all_rows:
        worst = max(worst, abs((r[1] + r[2] + r[3]) - 1.0))
    sums_ok = worst < 1e-9
    print(f"  Probability sums == 1: {sums_ok} (max deviation {worst:.3e})")
    if not sums_ok:
        stop(f"Probability sum deviation too large: {worst}")

    # 5d. All rows labelled promotion-validation-only
    labelled = all(r[4] == DATA_CLASS for r in all_rows)
    print(f"  All rows labelled '{DATA_CLASS}': {labelled}")
    if not labelled:
        stop("Some rows have incorrect data_class")

    # 5e. All rows are closing prices
    closing = all(r[5] == "closing" for r in all_rows)
    print(f"  All rows price_class 'closing': {closing}")
    if not closing:
        stop("Some rows are not closing prices")

    # 5f. All rows are Pinnacle
    pinnacle = all(r[6] == PINNACLE_ID for r in all_rows)
    print(f"  All rows bookmaker_id = {PINNACLE_ID} (Pinnacle): {pinnacle}")
    if not pinnacle:
        stop("Some rows are not from Pinnacle")

    # 5g. Original 20 rows unchanged
    post_orig_rows, post_orig_hash = snapshot_original_20(
        sqlite3.connect(f"file:{OUT_DB}?mode=ro", uri=True))
    # Filter to just original 20
    post_orig_20 = [r for r in conn.execute(
        "SELECT * FROM promotion_market WHERE fixture_id IN ({}) ORDER BY fixture_id"
        .format(",".join("?" * 20)), ORIGINAL_20_FIXTURE_IDS).fetchall()]
    post_hash_20 = _content_sha256_of_rows(post_orig_20)

    orig_unchanged = (post_hash_20 == orig_hash)
    print(f"  Original 20 rows unchanged: {orig_unchanged}")
    print(f"    Pre-hash:  {orig_hash}")
    print(f"    Post-hash: {post_hash_20}")
    if not orig_unchanged:
        stop("Original 20 rows were modified!")

    # ---- PHASE 6: Protected file integrity (POST) ----
    print("\n" + "=" * 78)
    print("PHASE 6 — PROTECTED FILE INTEGRITY (POST)")
    print("=" * 78)
    post_protected = check_files("Protected files (POST)", PROTECTED_FILES)
    post_e2 = check_files("E2 research artefacts (POST)", E2_FILES)
    prot_intact = all(v.get("status") == "identical"
                      for v in post_protected.values())
    e2_intact = all(v.get("status") == "identical"
                    for v in post_e2.values())
    print(f"\n  Protected files intact: {prot_intact}")
    print(f"  E2 artefacts intact:   {e2_intact}")
    if not prot_intact:
        stop("Protected files were modified!")
    if not e2_intact:
        stop("E2 research artefacts were modified!")

    # ---- Update meta table ----
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                 ("purpose",
                  "V4 promotion validation of 50 fixtures (20 original + 30 extension)"))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                 ("extended_at", now))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                 ("extension_fixtures", "30"))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                 ("total_fixtures", "50"))
    conn.commit()
    conn.close()

    # ---- PHASE 7: Final Report ----
    all_ok = (coverage_ok and sums_ok and labelled and closing
              and pinnacle and orig_unchanged and prot_intact and e2_intact
              and fid_match)

    print("\n" + "=" * 78)
    print("50-MATCH MARKET ODDS INGEST REPORT")
    print("=" * 78)
    print(f"  SUCCESS:               {ok_n}/30")
    print(f"  TOTAL COVERAGE:        {len(all_rows)}/50")
    print(f"  ORIGINAL 20 UNCHANGED: {'PASS' if orig_unchanged else 'FAIL'}")
    print(f"  PROTECTED INTEGRITY:   {'PASS' if prot_intact and e2_intact else 'FAIL'}")
    print(f"  PROB SUMS == 1:        {'PASS' if sums_ok else 'FAIL'}")
    print(f"  ALL CLOSING:           {'PASS' if closing else 'FAIL'}")
    print(f"  ALL PINNACLE:          {'PASS' if pinnacle else 'FAIL'}")
    print(f"  ALL PROMO-VALID:       {'PASS' if labelled else 'FAIL'}")
    print(f"  50 FIXTURE IDS MATCH:  {'PASS' if fid_match else 'FAIL'}")
    print(f"  OVERALL:               {'PASS' if all_ok else 'FAIL'}")
    print("=" * 78)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
