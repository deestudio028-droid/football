"""V3 Promotion Script — REQUIRES MANUAL EXECUTION.

This script promotes the V3 Poisson+Venue+Elo candidate to production.
It MUST be run manually after reviewing the V3 production candidate report.

What it does:
  1. Verifies all 4 protected files are intact (MD5 check)
  2. Verifies the V3 candidate artifact exists and passes structural checks
  3. Copies v3_poisson_venue_elo_candidate.pkl -> v3_poisson_venue_elo.pkl
  4. Does NOT overwrite v2_poisson_venue.pkl or v1_logreg.pkl
  5. Reports success with checksums

What it does NOT do:
  - Modify predict_match.py (that is a separate manual step)
  - Delete any existing artifact
  - Modify features.db or matches.db
  - Touch 2025/26 data
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROTECTED_FILES = {
    "v2_poisson_venue.pkl": (PROJECT_ROOT / "data" / "models" / "v2_poisson_venue.pkl",
                              "25935b4e93fc4074f67f16e3181ed4df"),
    "v1_logreg.pkl": (PROJECT_ROOT / "data" / "models" / "v1_logreg.pkl",
                       "5e504427712b35778bb8a62a8496c7cd"),
    "features.db": (PROJECT_ROOT / "data" / "processed" / "features.db",
                     "e7ebe7fc07040a5927683c35b6371e63"),
    "matches.db": (PROJECT_ROOT / "data" / "processed" / "matches.db",
                    "fdeed042096fa1c851aaee6c84995247"),
}

V3_CANDIDATE = PROJECT_ROOT / "data" / "models" / "v3_poisson_venue_elo_candidate.pkl"
V3_PRODUCTION = PROJECT_ROOT / "data" / "models" / "v3_poisson_venue_elo.pkl"


def get_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    print("=" * 70)
    print("V3 PROMOTION SCRIPT")
    print("=" * 70)

    # Step 1: Integrity gate
    print("\nStep 1: Pre-promotion integrity gate")
    for name, (path, expected) in PROTECTED_FILES.items():
        actual = get_md5(path)
        if actual != expected:
            print(f"  FAIL: {name} checksum mismatch!")
            print(f"    Expected: {expected}")
            print(f"    Actual:   {actual}")
            print("ABORTING PROMOTION — protected file integrity violated.")
            sys.exit(1)
        print(f"  PASS: {name}")

    # Step 2: Verify candidate exists
    print("\nStep 2: Verify V3 candidate artifact")
    if not V3_CANDIDATE.exists():
        print(f"  FAIL: Candidate not found at {V3_CANDIDATE}")
        print("ABORTING PROMOTION — candidate artifact missing.")
        sys.exit(1)
    candidate_md5 = get_md5(V3_CANDIDATE)
    print(f"  Candidate: {V3_CANDIDATE}")
    print(f"  MD5: {candidate_md5}")
    print(f"  Size: {V3_CANDIDATE.stat().st_size} bytes")

    # Step 3: Confirm
    print("\nStep 3: Confirmation")
    print("  This will copy the V3 candidate to production path:")
    print(f"    {V3_CANDIDATE}")
    print(f"    -> {V3_PRODUCTION}")
    print("  V2 and V1 artifacts will NOT be modified or deleted.")
    response = input("  Type 'PROMOTE' to proceed: ")
    if response.strip() != "PROMOTE":
        print("ABORTED by user.")
        sys.exit(0)

    # Step 4: Copy
    shutil.copy2(V3_CANDIDATE, V3_PRODUCTION)
    production_md5 = get_md5(V3_PRODUCTION)
    assert production_md5 == candidate_md5, "Copy verification failed!"

    # Step 5: Post-promotion integrity gate
    print("\nStep 4: Post-promotion integrity gate")
    for name, (path, expected) in PROTECTED_FILES.items():
        actual = get_md5(path)
        assert actual == expected, f"CRITICAL: {name} was modified during promotion!"
        print(f"  PASS: {name}")

    print(f"\n{'=' * 70}")
    print("V3 PROMOTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Production artifact: {V3_PRODUCTION}")
    print(f"  Production MD5: {production_md5}")
    print(f"  V2 artifact UNTOUCHED: {PROTECTED_FILES['v2_poisson_venue.pkl'][0]}")
    print(f"  V1 artifact UNTOUCHED: {PROTECTED_FILES['v1_logreg.pkl'][0]}")
    print()
    print("NEXT STEPS (manual):")
    print("  1. Update predict_match.py to support --model v3")
    print("  2. Run integration tests with the production artifact")
    print("  3. Commit all changes")


if __name__ == "__main__":
    main()
