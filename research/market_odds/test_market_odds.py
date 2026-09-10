"""Phase 3K — Market odds test suite.

Usage:
    python research/market_odds/test_market_odds.py

Tests are grouped into suites:
    Suite 1: De-vig correctness (no API/DB needed)
    Suite 2: Ingestion validation (requires odds_history.sqlite)
    Suite 3: Research dataset validation (requires research_dataset.sqlite)
    Suite 4: Protected file integrity
    Suite 5: Duplicate / alignment checks

Suites 2-3 are skipped if the required databases don't exist.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import sys
from pathlib import Path

# Add research/market_odds to path for devig import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from devig import power_devig, DevigError, DevigResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
ODDS_DB = Path(__file__).resolve().parent / "odds_history.sqlite"
RESEARCH_DB = Path(__file__).resolve().parent / "research_dataset.sqlite"

# Protected file checksums
PROTECTED_FILES = {
    PROJECT_ROOT / "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    PROJECT_ROOT / "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    PROJECT_ROOT / "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    PROJECT_ROOT / "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}

ELIGIBLE_SEASONS = [
    "2020/2021", "2021/2022", "2022/2023",
    "2023/2024", "2024/2025",
]


class TestResult:
    def __init__(self, suite: str):
        self.suite = suite
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.details: list[str] = []

    def ok(self, name: str, detail: str = ""):
        self.passed += 1
        msg = f"  PASS: {name}"
        if detail:
            msg += f" — {detail}"
        self.details.append(msg)

    def fail(self, name: str, detail: str = ""):
        self.failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        self.details.append(msg)

    def skip(self, name: str, reason: str = ""):
        self.skipped += 1
        msg = f"  SKIP: {name}"
        if reason:
            msg += f" — {reason}"
        self.details.append(msg)

    def report(self):
        print(f"\n{'=' * 60}")
        print(f"Suite: {self.suite}")
        print(f"{'=' * 60}")
        for d in self.details:
            print(d)
        print(f"\n  Total: {self.passed + self.failed + self.skipped}  "
              f"Pass: {self.passed}  Fail: {self.failed}  Skip: {self.skipped}")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------
# Suite 1: De-vig correctness
# -----------------------------------------------------------------------

def suite1_devig() -> TestResult:
    t = TestResult("1. De-vig Correctness")

    # 1.1 Fair odds → same probabilities
    r = power_devig(2.0, 3.0, 6.0)  # implied: 0.5 + 0.333 + 0.167 ≈ 1.0
    if abs(r.prob_sum() - 1.0) < 1e-8:
        t.ok("Fair odds sum to 1", f"sum={r.prob_sum():.10f}")
    else:
        t.fail("Fair odds sum to 1", f"sum={r.prob_sum()}")

    # 1.2 Vigged odds → probabilities sum exactly to 1
    r = power_devig(1.90, 3.40, 4.20)
    if abs(r.prob_sum() - 1.0) < 1e-8:
        t.ok("Vigged odds de-vig to sum=1", f"sum={r.prob_sum():.10f}")
    else:
        t.fail("Vigged odds de-vig to sum=1", f"sum={r.prob_sum()}")

    # 1.3 De-vigged probs are all positive
    if all(p > 0 for p in r.probabilities()):
        t.ok("All probabilities > 0")
    else:
        t.fail("All probabilities > 0", f"probs={r.probabilities()}")

    # 1.4 De-vigged probs are all < 1
    if all(p < 1 for p in r.probabilities()):
        t.ok("All probabilities < 1")
    else:
        t.fail("All probabilities < 1", f"probs={r.probabilities()}")

    # 1.5 k > 1 for vigged odds (overround > 1)
    if r.k > 1.0:
        t.ok("k > 1 for vigged odds", f"k={r.k:.6f}")
    else:
        t.fail("k > 1 for vigged odds", f"k={r.k}")

    # 1.6 Invalid odds rejected: odds <= 1.0
    try:
        power_devig(0.9, 3.0, 4.0)
        t.fail("Reject odds <= 1.0")
    except DevigError:
        t.ok("Reject odds <= 1.0")

    # 1.7 Invalid odds rejected: None
    try:
        power_devig(None, 3.0, 4.0)
        t.fail("Reject None odds")
    except DevigError:
        t.ok("Reject None odds")

    # 1.8 Invalid odds rejected: NaN
    try:
        power_devig(float("nan"), 3.0, 4.0)
        t.fail("Reject NaN odds")
    except DevigError:
        t.ok("Reject NaN odds")

    # 1.9 Invalid odds rejected: Inf
    try:
        power_devig(float("inf"), 3.0, 4.0)
        t.fail("Reject Inf odds")
    except DevigError:
        t.ok("Reject Inf odds")

    # 1.10 Extreme but valid odds handled
    r = power_devig(1.01, 50.0, 100.0)
    if abs(r.prob_sum() - 1.0) < 1e-8 and all(
        0 < p < 1 and math.isfinite(p) for p in r.probabilities()
    ):
        t.ok("Extreme odds handled", f"probs={r.probabilities()}")
    else:
        t.fail("Extreme odds handled", f"probs={r.probabilities()}")

    # 1.11 Deterministic repeated calculation
    r1 = power_devig(1.90, 3.40, 4.20)
    r2 = power_devig(1.90, 3.40, 4.20)
    if r1.p_home == r2.p_home and r1.p_draw == r2.p_draw and r1.p_away == r2.p_away:
        t.ok("Deterministic output")
    else:
        t.fail("Deterministic output")

    # 1.12 Overround reported correctly
    expected_or = 1/1.90 + 1/3.40 + 1/4.20
    if abs(r1.overround - expected_or) < 1e-10:
        t.ok("Overround correct", f"overround={r1.overround:.6f}")
    else:
        t.fail("Overround correct",
               f"expected={expected_or:.6f}, got={r1.overround:.6f}")

    # 1.13 Symmetric odds produce equal probabilities
    r = power_devig(3.0, 3.0, 3.0)
    if abs(r.p_home - r.p_draw) < 1e-10 and abs(r.p_draw - r.p_away) < 1e-10:
        t.ok("Symmetric odds → equal probs",
             f"all≈{r.p_home:.6f}")
    else:
        t.fail("Symmetric odds → equal probs", f"probs={r.probabilities()}")

    # 1.14 Heavy favorite
    r = power_devig(1.05, 20.0, 30.0)
    if r.p_home > 0.9:
        t.ok("Heavy favorite correctly reflected", f"p_home={r.p_home:.4f}")
    else:
        t.fail("Heavy favorite", f"p_home={r.p_home}")

    return t


# -----------------------------------------------------------------------
# Suite 2: Ingestion validation
# -----------------------------------------------------------------------

def suite2_ingestion() -> TestResult:
    t = TestResult("2. Ingestion Validation")

    if not ODDS_DB.exists():
        t.skip("All ingestion tests", "odds_history.sqlite not found")
        return t

    conn = sqlite3.connect(f"file:{ODDS_DB}?mode=ro", uri=True)

    # 2.1 Schema exists
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "odds_records" in tables and "fetch_log" in tables:
        t.ok("Schema: odds_records + fetch_log exist")
    else:
        t.fail("Schema tables", f"found: {tables}")

    # 2.2 Records exist
    count = conn.execute("SELECT COUNT(*) FROM odds_records").fetchone()[0]
    if count > 0:
        t.ok("Records exist", f"count={count}")
    else:
        t.fail("Records exist", "0 records")

    # 2.3 All records are ft_result (market_id=6)
    non_ft = conn.execute(
        "SELECT COUNT(*) FROM odds_records WHERE market_id != 6"
    ).fetchone()[0]
    if non_ft == 0:
        t.ok("All records are ft_result (market_id=6)")
    else:
        t.fail("Market filter", f"{non_ft} non-ft_result records")

    # 2.4 Outcomes are only home/draw/away
    outcomes = [r[0] for r in conn.execute(
        "SELECT DISTINCT outcome FROM odds_records"
    ).fetchall()]
    expected = {"home", "draw", "away"}
    if set(outcomes) == expected:
        t.ok("Outcomes correct", f"found: {outcomes}")
    else:
        t.fail("Outcomes", f"expected {expected}, got {set(outcomes)}")

    # 2.5 No duplicate records (fixture+market+outcome+bookmaker)
    dupes = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT fixture_id, market_id, outcome, bookmaker_id,
                   COUNT(*) as cnt
            FROM odds_records
            GROUP BY fixture_id, market_id, outcome, bookmaker_id
            HAVING cnt > 1
        )
    """).fetchone()[0]
    if dupes == 0:
        t.ok("No duplicate records")
    else:
        t.fail("Duplicates found", f"{dupes} duplicate groups")

    # 2.6 All odds values > 0 where present
    bad_odds = conn.execute("""
        SELECT COUNT(*) FROM odds_records
        WHERE (opening IS NOT NULL AND opening <= 0)
           OR (closing IS NOT NULL AND closing <= 0)
           OR (peak IS NOT NULL AND peak <= 0)
    """).fetchone()[0]
    if bad_odds == 0:
        t.ok("All present odds > 0")
    else:
        t.fail("Invalid odds values", f"{bad_odds} records with odds <= 0")

    # 2.7 Fixture IDs map to matches.db
    if MATCHES_DB.exists():
        mconn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
        odds_fids = set(r[0] for r in conn.execute(
            "SELECT DISTINCT fixture_id FROM odds_records"
        ).fetchall())
        matches_fids = set(r[0] for r in mconn.execute(
            "SELECT fixture_id FROM fixtures"
        ).fetchall())
        orphans = odds_fids - matches_fids
        mconn.close()
        if not orphans:
            t.ok("All odds fixture_ids exist in matches.db")
        else:
            t.fail("Orphan fixture_ids", f"{len(orphans)} not in matches.db")
    else:
        t.skip("Fixture ID mapping", "matches.db not found")

    # 2.8 No 2025/26 fixtures in odds
    if MATCHES_DB.exists():
        mconn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
        quarantined = mconn.execute(
            "SELECT fixture_id FROM fixtures WHERE season = '2025/2026'"
        ).fetchall()
        quarantined_ids = {r[0] for r in quarantined}
        mconn.close()
        leaked = odds_fids & quarantined_ids
        if not leaked:
            t.ok("No 2025/26 fixtures in odds data")
        else:
            t.fail("2025/26 leakage", f"{len(leaked)} quarantined fixtures found")

    # 2.9 Fetch log completeness
    log_count = conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0]
    success = conn.execute(
        "SELECT COUNT(*) FROM fetch_log WHERE status='success'"
    ).fetchone()[0]
    errors = conn.execute(
        "SELECT COUNT(*) FROM fetch_log WHERE status='error'"
    ).fetchone()[0]
    t.ok("Fetch log stats",
         f"total={log_count}, success={success}, errors={errors}")

    conn.close()
    return t


# -----------------------------------------------------------------------
# Suite 3: Research dataset validation
# -----------------------------------------------------------------------

def suite3_research_dataset() -> TestResult:
    t = TestResult("3. Research Dataset Validation")

    if not RESEARCH_DB.exists():
        t.skip("All dataset tests", "research_dataset.sqlite not found")
        return t

    conn = sqlite3.connect(f"file:{RESEARCH_DB}?mode=ro", uri=True)

    # 3.1 Schema exists
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "research_odds" in tables:
        t.ok("research_odds table exists")
    else:
        t.fail("research_odds missing", f"tables: {tables}")
        conn.close()
        return t

    # 3.2 Records exist
    count = conn.execute("SELECT COUNT(*) FROM research_odds").fetchone()[0]
    if count > 0:
        t.ok("Records exist", f"count={count}")
    else:
        t.fail("No records")

    # 3.3 All seasons are eligible (no 2025/26)
    seasons = [r[0] for r in conn.execute(
        "SELECT DISTINCT season FROM research_odds"
    ).fetchall()]
    if all(s in ELIGIBLE_SEASONS for s in seasons):
        t.ok("All seasons eligible", f"seasons={seasons}")
    else:
        bad = [s for s in seasons if s not in ELIGIBLE_SEASONS]
        t.fail("Ineligible seasons", f"found: {bad}")

    # 3.4 De-vigged closing probabilities sum to 1
    rows = conn.execute("""
        SELECT fixture_id, devig_closing_home, devig_closing_draw, devig_closing_away
        FROM research_odds
        WHERE devig_closing_home IS NOT NULL
    """).fetchall()
    bad_sums = []
    for fid, ph, pd, pa in rows:
        s = ph + pd + pa
        if abs(s - 1.0) > 1e-6:
            bad_sums.append((fid, s))
    if not bad_sums:
        t.ok("De-vig closing probs sum to 1",
             f"checked {len(rows)} fixtures")
    else:
        t.fail("De-vig prob sums",
               f"{len(bad_sums)} fixtures with bad sums, first: {bad_sums[0]}")

    # 3.5 De-vigged opening probabilities sum to 1
    rows = conn.execute("""
        SELECT fixture_id, devig_opening_home, devig_opening_draw, devig_opening_away
        FROM research_odds
        WHERE devig_opening_home IS NOT NULL
    """).fetchall()
    bad_sums = []
    for fid, ph, pd, pa in rows:
        s = ph + pd + pa
        if abs(s - 1.0) > 1e-6:
            bad_sums.append((fid, s))
    if not bad_sums:
        t.ok("De-vig opening probs sum to 1",
             f"checked {len(rows)} fixtures")
    else:
        t.fail("De-vig opening prob sums",
               f"{len(bad_sums)} bad sums")

    # 3.6 All probabilities in [0, 1]
    bad_range = conn.execute("""
        SELECT COUNT(*) FROM research_odds
        WHERE (devig_closing_home IS NOT NULL AND
               (devig_closing_home < 0 OR devig_closing_home > 1))
           OR (devig_closing_draw IS NOT NULL AND
               (devig_closing_draw < 0 OR devig_closing_draw > 1))
           OR (devig_closing_away IS NOT NULL AND
               (devig_closing_away < 0 OR devig_closing_away > 1))
    """).fetchone()[0]
    if bad_range == 0:
        t.ok("All probabilities in [0, 1]")
    else:
        t.fail("Out-of-range probabilities", f"{bad_range} fixtures")

    # 3.7 Causality fields populated
    causality_vals = [r[0] for r in conn.execute(
        "SELECT DISTINCT closing_causality FROM research_odds"
    ).fetchall()]
    if causality_vals == ["ASSUMED PRE-MATCH"]:
        t.ok("Causality classification present",
             f"closing={causality_vals}")
    else:
        t.fail("Causality classification", f"values={causality_vals}")

    # 3.8 No duplicate fixture_ids
    dupe_count = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT fixture_id, COUNT(*) as cnt
            FROM research_odds GROUP BY fixture_id HAVING cnt > 1
        )
    """).fetchone()[0]
    if dupe_count == 0:
        t.ok("No duplicate fixture_ids in research dataset")
    else:
        t.fail("Duplicate fixture_ids", f"{dupe_count} duplicates")

    # 3.9 Complete 1X2: all three closing odds present
    incomplete = conn.execute("""
        SELECT COUNT(*) FROM research_odds
        WHERE closing_home IS NULL OR closing_draw IS NULL OR closing_away IS NULL
    """).fetchone()[0]
    if incomplete == 0:
        t.ok("All fixtures have complete closing 1X2 odds")
    else:
        t.fail("Incomplete closing odds", f"{incomplete} fixtures")

    conn.close()
    return t


# -----------------------------------------------------------------------
# Suite 4: Protected file integrity
# -----------------------------------------------------------------------

def suite4_integrity() -> TestResult:
    t = TestResult("4. Protected File Integrity")

    for path, expected_md5 in PROTECTED_FILES.items():
        if not path.exists():
            t.skip(path.name, "file not found")
            continue
        actual = md5(path)
        if actual == expected_md5:
            t.ok(path.name, f"MD5={actual}")
        else:
            t.fail(path.name,
                   f"expected={expected_md5}, got={actual}")

    return t


# -----------------------------------------------------------------------
# Suite 5: Duplicate / alignment checks
# -----------------------------------------------------------------------

def suite5_alignment() -> TestResult:
    t = TestResult("5. Duplicate & Alignment Checks")

    if not ODDS_DB.exists():
        t.skip("All alignment tests", "odds_history.sqlite not found")
        return t

    conn = sqlite3.connect(f"file:{ODDS_DB}?mode=ro", uri=True)

    # 5.1 No impossible odds (closing <= 1.0 where present)
    impossible = conn.execute("""
        SELECT COUNT(*) FROM odds_records
        WHERE closing IS NOT NULL AND closing <= 1.0
    """).fetchone()[0]
    if impossible == 0:
        t.ok("No impossible closing odds (all > 1.0)")
    else:
        t.fail("Impossible odds", f"{impossible} records with closing <= 1.0")

    # 5.2 No impossible opening odds
    impossible_open = conn.execute("""
        SELECT COUNT(*) FROM odds_records
        WHERE opening IS NOT NULL AND opening <= 1.0
    """).fetchone()[0]
    if impossible_open == 0:
        t.ok("No impossible opening odds (all > 1.0)")
    else:
        t.fail("Impossible opening odds",
               f"{impossible_open} records with opening <= 1.0")

    # 5.3 Fixture-bookmaker alignment: each bookmaker should have 3 outcomes
    misaligned = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT fixture_id, bookmaker_id, COUNT(DISTINCT outcome) as n_outcomes
            FROM odds_records
            WHERE market_id = 6
            GROUP BY fixture_id, bookmaker_id
            HAVING n_outcomes != 3
        )
    """).fetchone()[0]
    if misaligned == 0:
        t.ok("All fixture-bookmaker groups have 3 outcomes")
    else:
        t.fail("Misaligned outcomes",
               f"{misaligned} groups without exactly 3 outcomes")

    # 5.4 Retrieved_at is populated
    missing_ts = conn.execute("""
        SELECT COUNT(*) FROM odds_records WHERE retrieved_at IS NULL
    """).fetchone()[0]
    if missing_ts == 0:
        t.ok("All records have retrieved_at timestamp")
    else:
        t.fail("Missing retrieved_at", f"{missing_ts} records")

    conn.close()
    return t


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Phase 3K — Market Odds Test Suite")
    print("=" * 60)

    suites = [
        suite1_devig(),
        suite2_ingestion(),
        suite3_research_dataset(),
        suite4_integrity(),
        suite5_alignment(),
    ]

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for s in suites:
        s.report()
        total_pass += s.passed
        total_fail += s.failed
        total_skip += s.skipped

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"  Suites: {len(suites)}")
    print(f"  Pass:   {total_pass}")
    print(f"  Fail:   {total_fail}")
    print(f"  Skip:   {total_skip}")
    print(f"  Total:  {total_pass + total_fail + total_skip}")

    if total_fail > 0:
        print("\n  VERDICT: TESTS FAILED")
        return 1
    else:
        print("\n  VERDICT: ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
