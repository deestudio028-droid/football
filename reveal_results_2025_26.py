"""Ground-truth reveal for 2025/26 fixtures (read-only, run AFTER prediction).

    python reveal_results_2025_26.py --fixture-ids 123456 234567 ...

DELIBERATELY SEPARATE FROM THE PREDICTOR. This is a different process, run
after predictions are already captured, so the outcome cannot influence
them. It imports no model, loads no artifact, and predicts nothing --
verified by the test suite, which asserts this module never imports the
artifact loader or the prediction path.

It computes NO aggregate metric. It prints the stored outcome for each
requested fixture and stops there. Scoring is a manual comparison against
the prediction output you already have; doing it inside this process
would collapse the separation the two-step design exists to create.

Read-only: matches.db is opened mode=ro, nothing is written.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

RESULT_LABEL = {"H": "HOME WIN", "D": "DRAW", "A": "AWAY WIN"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Reveal stored 2025/26 results (ground truth) after prediction.")
    ap.add_argument("--fixture-ids", nargs="+", required=True)
    args = ap.parse_args(argv)

    try:
        ids = [int(x) for x in args.fixture_ids]
    except ValueError:
        print(f"ERROR: --fixture-ids must all be integers, got {args.fixture_ids}",
              file=sys.stderr)
        return 2

    from models.config import FINAL_TEST_SEASONS, SEASON_NAME_TO_IDS
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])

    mdb = REPO / "data/processed/matches.db"
    con = sqlite3.connect(f"file:{mdb.resolve()}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = con.execute(
            f"SELECT fixture_id, home_name, away_name, home_goals, away_goals, "
            f"season_id, status FROM fixtures WHERE fixture_id IN ({placeholders})",
            ids).fetchall()
    finally:
        con.close()

    found = {r[0]: r for r in rows}
    off_season = [f for f, r in found.items() if r[5] not in test_ids]
    if off_season:
        print(f"ERROR: not 2025/26 fixtures: {off_season}", file=sys.stderr)
        return 2
    missing = [f for f in ids if f not in found]
    if missing:
        print(f"ERROR: fixture_id(s) not found in matches.db: {missing}", file=sys.stderr)
        return 2

    print("=" * 78)
    print("GROUND TRUTH -- ACTUAL STORED RESULTS (2025/26)")
    print("Revealed AFTER prediction. Not used by, and not visible to, the model.")
    print("=" * 78)
    print(f"{'fixture_id':>11}  {'home':22} {'away':22}  {'score':>7}  actual")
    print("-" * 78)
    for fid in ids:
        _, home, away, hg, ag, _, status = found[fid]
        if hg is None or ag is None:
            actual, score = f"NO RESULT ({status})", "-"
        else:
            actual = RESULT_LABEL["H" if hg > ag else "A" if hg < ag else "D"]
            score = f"{hg}-{ag}"
        print(f"{fid:>11}  {home[:22]:22} {away[:22]:22}  {score:>7}  {actual}")
    print("=" * 78)
    print(f"{len(ids)} fixture(s) revealed. No aggregate metric computed here --")
    print("compare manually against the prediction output captured earlier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
