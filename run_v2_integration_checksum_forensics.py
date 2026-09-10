"""V2 Integration Checksum Forensics -- read-only diagnostic.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_v2_integration_checksum_forensics.py

PURPOSE: diagnose why run_v2_production_integration_test.py reports
  [PASS] ALL file checksums unchanged after test
  [FAIL] features.db unchanged
  [FAIL] matches.db unchanged
  [FAIL] v1_logreg.pkl unchanged
  [FAIL] v2_poisson_venue.pkl unchanged

This script is READ-ONLY. No training, no evaluation, no outcome access,
no file writes, no database writes, no pickle writes.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# ============================================================================
# THE FOUR FILES UNDER INVESTIGATION
# ============================================================================
FILES = {
    "features.db":         REPO / "data/processed/features.db",
    "matches.db":          REPO / "data/processed/matches.db",
    "v1_logreg.pkl":       REPO / "data/models/v1_logreg.pkl",
    "v2_poisson_venue.pkl": REPO / "data/models/v2_poisson_venue.pkl",
}

# Expected MD5s (the ground truth from the project record)
EXPECTED_MD5 = {
    "features.db":         "e7ebe7fc07040a5927683c35b6371e63",
    "matches.db":          "fdeed042096fa1c851aaee6c84995247",
    "v1_logreg.pkl":       "5e504427712b35778bb8a62a8496c7cd",
    "v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def rule(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


def main():
    print("V2 INTEGRATION CHECKSUM FORENSICS")
    print("=" * 90)
    print("Read-only diagnostic -- no writes of any kind")
    print()

    # ==================================================================
    # SECTION 1: CURRENT FILE STATE
    # ==================================================================
    rule("SECTION 1 -- CURRENT FILE STATE")

    all_match = True
    for name, path in FILES.items():
        print(f"\n  {name}:")
        if not path.exists():
            print(f"    EXISTS: NO")
            all_match = False
            continue
        print(f"    EXISTS: YES")
        current_md5 = md5(path)
        expected = EXPECTED_MD5[name]
        match = current_md5 == expected
        if not match:
            all_match = False
        print(f"    SIZE:   {path.stat().st_size} bytes")
        print(f"    MTIME:  {path.stat().st_mtime}")
        print(f"    MD5 current:  {current_md5}")
        print(f"    MD5 expected: {expected}")
        print(f"    MATCH: {match}")

        # Check for SQLite sidecars
        if name.endswith(".db"):
            for ext in ("-wal", "-shm", "-journal"):
                sidecar = path.with_name(path.name + ext)
                exists = sidecar.exists()
                size = sidecar.stat().st_size if exists else None
                print(f"    sidecar {ext}: exists={exists}" +
                      (f"  size={size}" if exists else ""))

    print(f"\n  ALL FILES MATCH EXPECTED MD5: {all_match}")

    # ==================================================================
    # SECTION 2: REPRODUCE THE BUG
    # ==================================================================
    rule("SECTION 2 -- REPRODUCE THE INTEGRATION TEST BUG")

    print("\n  Simulating snapshot_checksums() from the integration test...\n")

    # This is the EXACT code from snapshot_checksums():
    V2_ARTIFACT_PATH = REPO / "data/models/v2_poisson_venue.pkl"
    V1_ARTIFACT_PATH = REPO / "data/models/v1_logreg.pkl"
    FEATURES_DB_PATH = REPO / "data/processed/features.db"
    MATCHES_DB_PATH  = REPO / "data/processed/matches.db"

    snap = {}
    for p, _ in [(V2_ARTIFACT_PATH, "v2"), (V1_ARTIFACT_PATH, "v1"),
                  (FEATURES_DB_PATH, "fdb"), (MATCHES_DB_PATH, "mdb")]:
        if p.exists():
            snap[str(p.relative_to(REPO))] = md5(p)

    print("  Keys stored by snapshot_checksums():")
    for k in sorted(snap.keys()):
        print(f"    key: {k!r}")
        print(f"    md5: {snap[k]}")

    # Now simulate what the INDIVIDUAL checks do:
    print("\n  Keys used by INDIVIDUAL assertions (STEP 9, lines 695-702):")
    individual_keys = [
        "data/processed/features.db",
        "data/processed/matches.db",
        "data/models/v1_logreg.pkl",
        "data/models/v2_poisson_venue.pkl",
    ]
    for ik in individual_keys:
        val = snap.get(ik)
        print(f"    snap.get({ik!r}) = {val!r}")

    print("\n  Keys used by AGGREGATE loop (STEP 9, lines 682-687):")
    print("    (iterates pre_snap keys, looks up in post_snap)")
    for k in snap:
        val = snap.get(k)
        print(f"    snap.get({k!r}) = {val!r}")

    # ==================================================================
    # SECTION 3: ROOT CAUSE ANALYSIS
    # ==================================================================
    rule("SECTION 3 -- ROOT CAUSE ANALYSIS")

    # Demonstrate the mismatch
    sample_path = FEATURES_DB_PATH
    relative_str = str(sample_path.relative_to(REPO))
    forward_slash = "data/processed/features.db"
    print(f"\n  Path.relative_to(REPO) produces: {relative_str!r}")
    print(f"  os.sep on this platform:         {os.sep!r}")
    print(f"  Forward-slash key used by test:   {forward_slash!r}")
    print(f"  Keys match: {relative_str == forward_slash}")
    print()

    if os.sep == "\\":
        print("  DIAGNOSIS: Windows path separator mismatch.")
        print()
        print("  snapshot_checksums() stores keys using str(Path.relative_to(REPO)),")
        print("  which on Windows produces backslash paths like:")
        print(f"    'data\\\\processed\\\\features.db'")
        print()
        print("  The individual assertions on lines 695-702 use hardcoded")
        print("  forward-slash strings:")
        print(f"    'data/processed/features.db'")
        print()
        print("  dict.get() with a mismatched key returns None.")
        print("  None != 'e7ebe7fc07040a5927683c35b6371e63' => FAIL.")
        print()
        print("  The AGGREGATE loop iterates pre_snap's own keys (backslash)")
        print("  and looks them up in post_snap (also backslash), so")
        print("  all comparisons succeed => PASS.")
        print()
        print("  This is a TEST BOOKKEEPING BUG, not a real file modification.")
    else:
        print("  Platform uses forward slashes; the mismatch should not occur.")
        print("  Investigate further.")

    # ==================================================================
    # SECTION 4: INDEPENDENT VERIFICATION
    # ==================================================================
    rule("SECTION 4 -- INDEPENDENT FILE INTEGRITY VERIFICATION")

    print("\n  Computing fresh MD5 for each file (independent of test code):\n")
    verdicts = {}
    for name, path in FILES.items():
        current = md5(path)
        expected = EXPECTED_MD5[name]
        match = current == expected
        verdict = "UNCHANGED" if match else "CHANGED"
        verdicts[name] = verdict
        print(f"  {name}:")
        print(f"    md5 = {current}")
        print(f"    expected = {expected}")
        print(f"    verdict: {verdict}")

    # ==================================================================
    # FINAL REPORT
    # ==================================================================
    rule("CHECKSUM FORENSICS -- FINAL REPORT")

    print()
    max_len = max(len(n) for n in FILES)
    any_changed = False
    for name in FILES:
        v = verdicts[name]
        if v != "UNCHANGED":
            any_changed = True
        # Determine if test bug
        if v == "UNCHANGED":
            label = "UNCHANGED (test bookkeeping bug caused false FAIL)"
        else:
            label = "CHANGED -- INVESTIGATE"
        print(f"  {name:<{max_len}}  :  {label}")

    print()
    print("  ROOT CAUSE:")
    if not any_changed and os.sep == "\\":
        print("    Windows path separator mismatch in snapshot_checksums().")
        print("    str(Path.relative_to(REPO)) produces backslash keys on Windows.")
        print("    Individual assertions use hardcoded forward-slash keys.")
        print("    dict.get() returns None for the wrong key => false FAIL.")
        print("    The aggregate loop uses consistent keys => correct PASS.")
        print()
        print("  FIX (for future reference, not applied here):")
        print("    Replace str(p.relative_to(REPO)) with")
        print("    p.relative_to(REPO).as_posix() in snapshot_checksums().")
        print("    OR use the same key format in the individual assertions.")
    elif any_changed:
        print("    One or more files have genuinely changed. Investigate immediately.")
    else:
        print("    Unknown -- further investigation needed.")

    print()
    if any_changed:
        print("  PRODUCTION SAFE TO CONTINUE: NO")
    else:
        print("  PRODUCTION SAFE TO CONTINUE: YES")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
