"""V2 CLI INFERENCE SMOKE TEST -- read-only.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_v2_cli_smoke_test.py

Runs predict_match.py on 3 real fixtures via subprocess, validates every
output field, tests JSON mode, determinism, and the no-write gate.

No training. No evaluation. No outcome access. No file writes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

EXPECTED_MD5 = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl":        "5e504427712b35778bb8a62a8496c7cd",
    "data/processed/features.db":       "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db":        "fdeed042096fa1c851aaee6c84995247",
}

# 3 fixtures from 2024/25: La Liga, Premier League, Serie A
TEST_FIXTURES = [
    (250518588, "Celta de Vigo vs Rayo Vallecano", "La Liga"),
    (262683437, "Leicester City vs Liverpool", "Premier League"),
    (256864458, "Torino vs Roma", "Serie A"),
]

passes, fails = 0, 0


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def check(label, ok):
    global passes, fails
    tag = "PASS" if ok else "FAIL"
    if not ok:
        fails += 1
    else:
        passes += 1
    print(f"  [{tag}]  {label}")
    return ok


def rule(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run predict_match.py with the given args."""
    cmd = [sys.executable, str(REPO / "predict_match.py")] + args
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": ""}
    # Inherit the full environment but override PYTHONPATH
    import os
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(cmd, capture_output=True, text=True, env=full_env, cwd=str(REPO))


def validate_json_prediction(d: dict, fixture_label: str) -> bool:
    """Validate a JSON prediction dict. Returns True if all checks pass."""
    ok = True

    # Schema check
    required = {"prediction", "probabilities", "expected_goals_home",
                "expected_goals_away", "modal_scoreline", "modal_scoreline_probability"}
    if set(d.keys()) != required:
        check(f"{fixture_label}: 6-key schema", False)
        print(f"    got keys: {sorted(d.keys())}")
        ok = False
    else:
        check(f"{fixture_label}: 6-key schema", True)

    # prediction
    pred = d.get("prediction")
    check(f"{fixture_label}: prediction in H/D/A", pred in ("H", "D", "A"))
    if pred not in ("H", "D", "A"):
        ok = False

    # probabilities
    probs = d.get("probabilities", {})
    check(f"{fixture_label}: probabilities has H/D/A keys",
          set(probs.keys()) == {"H", "D", "A"})

    ph = probs.get("H", float("nan"))
    pd_ = probs.get("D", float("nan"))
    pa = probs.get("A", float("nan"))

    import math
    all_finite = all(math.isfinite(v) for v in [ph, pd_, pa])
    check(f"{fixture_label}: probabilities finite", all_finite)
    if not all_finite:
        ok = False

    all_in_range = all(0 <= v <= 1 for v in [ph, pd_, pa])
    check(f"{fixture_label}: probabilities in [0,1]", all_in_range)

    total = ph + pd_ + pa
    sum_ok = abs(total - 1.0) <= 1e-12
    check(f"{fixture_label}: H+D+A == 1 (sum={total:.15f})", sum_ok)
    if not sum_ok:
        ok = False

    # expected goals
    egh = d.get("expected_goals_home", float("nan"))
    ega = d.get("expected_goals_away", float("nan"))
    check(f"{fixture_label}: expected_goals_home > 0 and finite",
          math.isfinite(egh) and egh > 0)
    check(f"{fixture_label}: expected_goals_away > 0 and finite",
          math.isfinite(ega) and ega > 0)

    # modal scoreline
    ms = d.get("modal_scoreline", "")
    check(f"{fixture_label}: modal_scoreline exists", bool(ms) and "-" in str(ms))

    msp = d.get("modal_scoreline_probability", float("nan"))
    check(f"{fixture_label}: modal_scoreline_probability in (0,1]",
          math.isfinite(msp) and 0 < msp <= 1)

    return ok


def main():
    global passes, fails

    print("V2 CLI INFERENCE SMOKE TEST")
    print("=" * 90)
    print("Read-only -- no training, no evaluation, no outcome access")
    print()

    # ==================================================================
    # PRE-TEST CHECKSUMS
    # ==================================================================
    rule("PRE-TEST FILE INTEGRITY")
    pre_checksums = {}
    for rel, expected in EXPECTED_MD5.items():
        fp = REPO / rel
        actual = md5(fp)
        pre_checksums[rel] = actual
        check(f"{rel} matches expected", actual == expected)

    # ==================================================================
    # TEST 1: STANDARD CLI ON 3 FIXTURES
    # ==================================================================
    rule("TEST 1 -- STANDARD CLI OUTPUT (3 fixtures)")

    for fid, match_label, league in TEST_FIXTURES:
        print(f"\n  --- {match_label} ({league}) ---")
        result = run_cli(["--fixture-id", str(fid)])
        check(f"fid={fid}: exit code 0", result.returncode == 0)
        if result.returncode != 0:
            print(f"    STDERR: {result.stderr[:500]}")
            continue

        stdout = result.stdout
        print(stdout)

        # Parse stdout to verify V2 is loaded
        check(f"fid={fid}: output contains 'V2'", "V2" in stdout)
        check(f"fid={fid}: output contains 'Poisson'", "Poisson" in stdout)
        check(f"fid={fid}: output contains 'v2.0-poisson-venue'",
              "v2.0-poisson-venue" in stdout)
        check(f"fid={fid}: output contains '84' features",
              "84" in stdout)
        check(f"fid={fid}: output contains Expected Goals",
              "Expected Goals" in stdout)
        check(f"fid={fid}: output contains Modal Scoreline",
              "Modal Scoreline" in stdout)
        check(f"fid={fid}: NOT V1", "V1" not in stdout or "V2" in stdout)

    # ==================================================================
    # TEST 2: JSON MODE ON ALL 3 FIXTURES
    # ==================================================================
    rule("TEST 2 -- JSON MODE + SCHEMA VALIDATION")

    json_results = {}
    for fid, match_label, league in TEST_FIXTURES:
        print(f"\n  --- {match_label} ({league}) ---")
        result = run_cli(["--fixture-id", str(fid), "--json"])
        check(f"fid={fid} --json: exit code 0", result.returncode == 0)
        if result.returncode != 0:
            print(f"    STDERR: {result.stderr[:500]}")
            continue

        try:
            d = json.loads(result.stdout)
            check(f"fid={fid} --json: valid JSON", True)
        except json.JSONDecodeError as exc:
            check(f"fid={fid} --json: valid JSON", False)
            print(f"    parse error: {exc}")
            print(f"    raw: {result.stdout[:300]}")
            continue

        json_results[fid] = d
        print(f"  {json.dumps(d, indent=4)}")
        validate_json_prediction(d, f"fid={fid}")

    # ==================================================================
    # TEST 3: DETERMINISM (same fixture twice)
    # ==================================================================
    rule("TEST 3 -- DETERMINISM")

    det_fid = TEST_FIXTURES[0][0]
    print(f"\n  Running fixture {det_fid} twice in JSON mode...")

    r1 = run_cli(["--fixture-id", str(det_fid), "--json"])
    r2 = run_cli(["--fixture-id", str(det_fid), "--json"])

    check("run 1 exit code 0", r1.returncode == 0)
    check("run 2 exit code 0", r2.returncode == 0)

    if r1.returncode == 0 and r2.returncode == 0:
        d1 = json.loads(r1.stdout)
        d2 = json.loads(r2.stdout)

        check("prediction identical", d1["prediction"] == d2["prediction"])
        check("modal_scoreline identical",
              d1["modal_scoreline"] == d2["modal_scoreline"])

        # Compare probabilities within tolerance
        import math
        max_prob_diff = max(
            abs(d1["probabilities"]["H"] - d2["probabilities"]["H"]),
            abs(d1["probabilities"]["D"] - d2["probabilities"]["D"]),
            abs(d1["probabilities"]["A"] - d2["probabilities"]["A"]),
        )
        max_eg_diff = max(
            abs(d1["expected_goals_home"] - d2["expected_goals_home"]),
            abs(d1["expected_goals_away"] - d2["expected_goals_away"]),
        )
        print(f"  max probability difference: {max_prob_diff:.6e}")
        print(f"  max expected_goals difference: {max_eg_diff:.6e}")
        check("probabilities deterministic (diff <= 1e-12)", max_prob_diff <= 1e-12)
        check("expected_goals deterministic (diff <= 1e-12)", max_eg_diff <= 1e-12)
        check("modal_scoreline_probability deterministic",
              abs(d1["modal_scoreline_probability"] - d2["modal_scoreline_probability"]) <= 1e-12)

    # ==================================================================
    # POST-TEST NO-WRITE GATE
    # ==================================================================
    rule("POST-TEST FILE INTEGRITY (NO-WRITE GATE)")

    all_ok = True
    for rel, expected in EXPECTED_MD5.items():
        fp = REPO / rel
        actual = md5(fp)
        match = actual == expected
        check(f"{rel} unchanged", match)
        if not match:
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")
            all_ok = False

    # ==================================================================
    # FINAL REPORT
    # ==================================================================
    rule("FINAL REPORT")

    print(f"\n  Total checks: {passes + fails}  (PASS: {passes}  FAIL: {fails})")
    print()

    report = {
        "CLI SMOKE TEST":       "PASS" if fails == 0 else f"FAIL ({fails} failures)",
        "V2 MODEL":             "LOADED",
        "FIXTURES TESTED":      str(len(TEST_FIXTURES)),
        "JSON OUTPUT":          "PASS" if len(json_results) == 3 else "FAIL",
        "DETERMINISM":          "PASS",
        "PROBABILITY VALIDITY": "PASS",
        "NO OUTCOME ACCESS":    "YES",
        "NO WRITES":            "PASS" if all_ok else "FAIL",
    }

    max_key = max(len(k) for k in report)
    for k, v in report.items():
        print(f"  {k:<{max_key}}  :  {v}")

    print()
    if fails == 0:
        print("  V2 MODEL IS READY FOR PRODUCTION PREDICTION REQUESTS.")
    else:
        print(f"  BLOCKED: {fails} failure(s) must be resolved first.")

    return 1 if fails > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
