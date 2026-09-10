"""V3 Rollback Script — Restores V2 as the active production model.

This script removes the V3 production artifact and ensures V2 is intact.
It does NOT delete the V3 candidate (kept for audit trail).

What it does:
  1. Verifies V2 artifact is intact (MD5 check)
  2. Removes v3_poisson_venue_elo.pkl (production copy only)
  3. Preserves v3_poisson_venue_elo_candidate.pkl (audit trail)
  4. Verifies all protected files are intact

What it does NOT do:
  - Delete the V3 candidate artifact
  - Modify V2, V1, features.db, or matches.db
  - Modify predict_match.py (that is a separate manual step)
"""
from __future__ import annotations

import hashlib
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

V3_PRODUCTION = PROJECT_ROOT / "data" / "models" / "v3_poisson_venue_elo.pkl"
V3_CANDIDATE = PROJECT_ROOT / "data" / "models" / "v3_poisson_venue_elo_candidate.pkl"


def get_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    print("=" * 70)
    print("V3 ROLLBACK SCRIPT")
    print("=" * 70)

    # Step 1: Verify V2 is intact
    print("\nStep 1: Verify V2 champion is intact")
    v2_path, v2_expected = PROTECTED_FILES["v2_poisson_venue.pkl"]
    v2_actual = get_md5(v2_path)
    if v2_actual != v2_expected:
        print(f"  CRITICAL: V2 artifact corrupted!")
        print(f"    Expected: {v2_expected}")
        print(f"    Actual:   {v2_actual}")
        print("  Cannot rollback safely. Manual intervention required.")
        sys.exit(1)
    print(f"  PASS: V2 artifact intact (MD5: {v2_actual})")

    # Step 2: Remove V3 production artifact
    print("\nStep 2: Remove V3 production artifact")
    if V3_PRODUCTION.exists():
        response = input(f"  Remove {V3_PRODUCTION}? Type 'ROLLBACK' to proceed: ")
        if response.strip() != "ROLLBACK":
            print("ABORTED by user.")
            sys.exit(0)
        V3_PRODUCTION.unlink()
        print(f"  Removed: {V3_PRODUCTION}")
    else:
        print(f"  V3 production artifact not found (already removed or never promoted)")

    # Step 3: Verify candidate preserved
    print("\nStep 3: Verify V3 candidate preserved (audit trail)")
    if V3_CANDIDATE.exists():
        print(f"  Preserved: {V3_CANDIDATE} (MD5: {get_md5(V3_CANDIDATE)})")
    else:
        print(f"  WARNING: V3 candidate not found at {V3_CANDIDATE}")

    # Step 4: Post-rollback integrity
    print("\nStep 4: Post-rollback integrity gate")
    for name, (path, expected) in PROTECTED_FILES.items():
        actual = get_md5(path)
        status = "PASS" if actual == expected else "FAIL"
        print(f"  {name}: {actual} [{status}]")

    print(f"\n{'=' * 70}")
    print("V3 ROLLBACK COMPLETE — V2 is the active production model")
    print(f"{'=' * 70}")
    print()
    print("NEXT STEPS (manual):")
    print("  1. Ensure predict_match.py defaults to V2")
    print("  2. Verify V2 predictions work correctly")


if __name__ == "__main__":
    main()
