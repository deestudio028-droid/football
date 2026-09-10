"""D-27 -- POSSESSION C-sensitivity diagnostic + D-6 reproduction check.

Run on the project machine (scikit-learn required):

    python run_d27_c_sensitivity_diagnostic.py

DIAGNOSTIC ONLY. NOT PROMOTION EVIDENCE.

WHAT THIS ANSWERS
    Is C the sole explanation for D-6 (POSSESSION improves, C=0.0005)
    disagreeing with D-24 (POSSESSION worsens, C=1.0)?

DESIGN
    For each C in {0.0005, 0.001, 0.01, 0.1, 1.0}, fit BOTH arms:
        CONTROL   ablation.MODEL_B_COLUMNS                   (80 columns)
        CANDIDATE successor_contract.MODEL_B_PLUS_POSSESSION (86 columns)
    Control and candidate are ALWAYS at the SAME C. The reported
    quantity is the paired delta at each C. Deltas are never compared
    across different C values for either arm.

    Both arms load from the SAME database (features_v1_1.db). D-24
    proved the shared 80 columns are bit-identical between features.db
    and features_v1_1.db (1,735,440 cells, 0 differences), so using one
    source removes a confound rather than introducing one. STEP 6 below
    re-proves this independently by checking that CONTROL at C=1.0
    reproduces the pinned production baseline.

REPRODUCTION CHECK
    If the C=0.0005 arms reproduce D-6's recorded numbers, C is proven
    to be the sole cause and the D-26 diagnosis closes. If they do NOT,
    a second undiagnosed difference exists and the diagnosis must
    reopen -- that outcome is reported, not explained away.

INTERPRETATION RULES, FIXED IN ADVANCE
    - Descriptive only. No significance test is performed, so no
      significance claim may be made.
    - No effect-size threshold exists or is invented.
    - A C at which POSSESSION improves is NOT grounds for promotion.
      Choosing C by looking at these validation results and then
      reporting the resulting delta as an unbiased result is exactly
      the circularity this study must avoid.
    - Any change to V1's C is a separate decision with its own
      authorization. The final test is spent and cannot re-validate it.

BOUNDARIES
    Read-only. Writes nothing, persists no estimator, creates no
    artifact, never loads 2025/26, never touches the final test. No
    production source, contract, database or pinned artifact is
    modified. MODEL_VERSION stays v1.0.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C_GRID = (0.0005, 0.001, 0.01, 0.1, 1.0)

EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "data/processed/features_v1_1.db": "6aa22e0895bd2189edcc0b54859fa749",
}

# Values recorded elsewhere in the project, used ONLY as reproduction
# targets. Nothing is tuned toward them and nothing fails because of them.
RECORDED = {
    # D-6 (in-memory POSSESSION, C=0.0005)
    (0.0005, "control", "log_loss"): 0.9915719393130367,   # also E-1's 80-col value
    (0.0005, "candidate", "log_loss"): 0.9904887677,
    (0.0005, "control", "brier"): 0.5914091026848851,
    (0.0005, "candidate", "brier"): 0.5905928705,
    # Pinned production baseline (Phase 4A / 4B / 5A), 80 columns
    (1.0, "control", "log_loss"): 0.9993791056968738,
    (1.0, "control", "brier"): 0.596502,
    # Phase 4B documented sweep, 80 columns
    (0.01, "control", "log_loss"): 0.9957166199,
    (0.01, "control", "brier"): 0.594245,
    (0.1, "control", "log_loss"): 0.9989624678,
    (0.1, "control", "brier"): 0.596281,
}


def rule(t): print("\n" + "=" * 86); print(t); print("=" * 86)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 86)
    print("STOP -- D-27 halted. No result is produced.")
    print(msg)
    print("!" * 86)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        stop(f"stop condition: {label}")


def classify(observed, target):
    """Reproduction verdict. The recorded targets carry different
    precisions (some were published to 10 s.f., some to full repr), so
    the verdict is stated against the precision actually recorded."""
    if observed == target:
        return "EXACT", 0.0
    diff = abs(observed - target)
    dp = len(str(target).split(".")[-1])
    if diff < 10 ** -(dp - 1):
        return f"MATCHES to {dp - 1}dp", diff
    if diff < 1e-9:
        return "FLOAT-RESIDUE", diff
    return "MISMATCH", diff


def main():
    rule("STEP 1 -- ENVIRONMENT")
    print(f"  Python  {sys.version.split()[0]}")
    print(f"  numpy   {np.__version__}")
    print(f"  pandas  {pd.__version__}")
    try:
        import sklearn
        print(f"  sklearn {sklearn.__version__}")
    except ImportError as exc:
        stop(f"scikit-learn is not available: {exc}. Do not substitute another "
             "estimator and do not hand-roll one.")
    from sklearn.linear_model import LogisticRegression  # noqa: E402

    from models import train as train_module
    from models.ablation import MODEL_B_COLUMNS
    from models.config import (
        FINAL_TEST_SEASONS, MODEL_VERSION, SEASON_NAME_TO_IDS,
    )
    from models.evaluate import evaluate, validate_probabilities
    from models.robustness import V1_LOGREG_BASE_KWARGS
    from models.splits import iter_walk_forward_folds
    from models.successor_contract import (
        MODEL_B_PLUS_POSSESSION_COLUMNS as CANDIDATE_COLUMNS,
        POSSESSION_FAMILY_COLUMNS,
    )
    from models.successor_data import load_successor_supervised_dataset

    print(f"  V1_LOGREG_BASE_KWARGS = {V1_LOGREG_BASE_KWARGS}")
    print("  (D-6 used exactly LogisticRegression(C=..., **V1_LOGREG_BASE_KWARGS);")
    print("   this study uses the same construction so C is the only variable)")

    rule("STEP 2 -- ARTIFACT INTEGRITY (before fitting)")
    before = {}
    for rel, exp in EXPECTED_MD5.items():
        got = md5(REPO / rel)
        before[rel] = got
        check(f"{rel}", got == exp, got)
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if md5(REPO / r) != v["expected"]]
    check("13/13 LOCKED_INPUTS unchanged", not bad, str(bad))
    check("MODEL_VERSION is v1.0", MODEL_VERSION == "v1.0", MODEL_VERSION)

    rule("STEP 3 -- CONTRACT PARITY")
    check("len(control) == 80", len(MODEL_B_COLUMNS) == 80)
    check("len(candidate) == 86", len(CANDIDATE_COLUMNS) == 86)
    check("candidate[:80] == MODEL_B_COLUMNS exactly, in order",
          tuple(CANDIDATE_COLUMNS[:80]) == tuple(MODEL_B_COLUMNS))
    check("candidate[80:] == POSSESSION_FAMILY_COLUMNS",
          CANDIDATE_COLUMNS[80:] == POSSESSION_FAMILY_COLUMNS)

    rule("STEP 4 -- LOAD + FINAL-TEST FIREWALL")
    ds = load_successor_supervised_dataset(REPO / "data/processed/features_v1_1.db")
    print(f"  rows: {len(ds)}   feature_version: "
          f"{set(ds.metadata['feature_version'].unique())}")
    check("feature_version == {'v1.1'}",
          set(ds.metadata["feature_version"].unique()) == {"v1.1"})
    test_ids = set()
    for name in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[name])
    check("no final-test season_id in the dataset (exact ID membership)",
          not (set(ds.metadata["season_id"].unique()) & test_ids),
          f"excluded ids={sorted(test_ids)}")

    folds = list(iter_walk_forward_folds(ds))
    check("three walk-forward folds", len(folds) == 3)
    for fold, tr, va in folds:
        seen = set(tr.metadata["season_id"]) | set(va.metadata["season_id"])
        check(f"{fold.name}: no final-test season", not (seen & test_ids))
        print(f"    {fold.name}: train={len(tr)}  validation={len(va)}")

    rule("STEP 5 -- PAIRED SWEEP (control and candidate always at the SAME C)")

    def fit_arm(tr, va, cols, C):
        A, B = tr.X[list(cols)], va.X[list(cols)]
        pre = train_module.LogisticRegressionPreprocessor().fit(A)
        model = LogisticRegression(C=C, **V1_LOGREG_BASE_KWARGS).fit(pre.transform(A), tr.y)
        if int(np.max(model.n_iter_)) >= V1_LOGREG_BASE_KWARGS["max_iter"]:
            stop(f"unconverged at C={C} ({len(cols)} cols): hit max_iter")
        P = np.asarray(train_module._reorder_proba(model, pre.transform(B)), dtype=float)
        validate_probabilities(P)
        return P

    results = {}
    for C in C_GRID:
        for arm, cols in (("control", MODEL_B_COLUMNS), ("candidate", CANDIDATE_COLUMNS)):
            per_fold = []
            for fold, tr, va in folds:
                P = fit_arm(tr, va, cols, C)
                per_fold.append((fold.name, evaluate(va.y, P)))
            results[(C, arm)] = per_fold
        c_ll = np.mean([r.log_loss for _, r in results[(C, "control")]])
        d_ll = np.mean([r.log_loss for _, r in results[(C, "candidate")]])
        print(f"  C={C:<8} control mean LL={c_ll:.16f}  candidate mean LL={d_ll:.16f}  "
              f"delta={d_ll - c_ll:+.16f}")

    def mean_of(C, arm, attr):
        return float(np.mean([getattr(r, attr) for _, r in results[(C, arm)]]))

    rule("STEP 6 -- INDEPENDENT ANCHOR: does CONTROL at C=1.0 reproduce the "
         "pinned production baseline?")
    obs = mean_of(1.0, "control", "log_loss")
    tgt = RECORDED[(1.0, "control", "log_loss")]
    verdict, diff = classify(obs, tgt)
    print(f"  observed : {obs:.16f}")
    print(f"  recorded : {tgt:.16f}   (Phase 4A/4B/5A, features.db)")
    print(f"  verdict  : {verdict}   |diff| = {diff:.3e}")
    print("  If this reproduces, the choice of source database is proven immaterial:")
    print("  the shared 80 columns are identical in features.db and features_v1_1.db.")

    rule("STEP 7 -- D-6 REPRODUCTION CHECK AT C=0.0005")
    print(f"  {'arm':11s}{'metric':10s}{'observed':>22s}{'recorded (D-6)':>22s}"
          f"{'verdict':>18s}{'|diff|':>12s}")
    repro = {}
    for arm in ("control", "candidate"):
        for metric in ("log_loss", "brier"):
            o = mean_of(0.0005, arm, metric)
            t = RECORDED[(0.0005, arm, metric)]
            v, d = classify(o, t)
            repro[(arm, metric)] = v
            print(f"  {arm:11s}{metric:10s}{o:22.16f}{t:22.16f}{v:>18s}{d:12.3e}")
    reproduced = all("MISMATCH" not in v for v in repro.values())
    print()
    if reproduced:
        print("  D-6 REPRODUCED. C is proven to be the sole material difference")
        print("  between D-6 and D-24, and the D-26 diagnosis closes.")
    else:
        print("  D-6 NOT REPRODUCED. A second, undiagnosed difference exists.")
        print("  The D-26 diagnosis must REOPEN. Do not explain this away.")

    rule("STEP 8 -- CROSS-CHECK AGAINST THE DOCUMENTED PHASE 4B SWEEP (control only)")
    print(f"  {'C':>8}{'observed control LL':>24s}{'Phase 4B recorded':>22s}{'verdict':>18s}")
    for C in (0.01, 0.1):
        o = mean_of(C, "control", "log_loss")
        t = RECORDED[(C, "control", "log_loss")]
        v, _ = classify(o, t)
        print(f"  {C:>8}{o:24.16f}{t:22.10f}{v:>18s}")

    rule("STEP 9 -- FULL RESULTS: LOG LOSS")
    for C in C_GRID:
        print(f"\n### C = {C}")
        print("| Fold | Control | Candidate | Delta |")
        print("|------|---------|-----------|-------|")
        for (n, c), (_, d) in zip(results[(C, "control")], results[(C, "candidate")]):
            print(f"| {n} | {c.log_loss:.16f} | {d.log_loss:.16f} | "
                  f"{d.log_loss - c.log_loss:+.16f} |")
        cm, dm = mean_of(C, "control", "log_loss"), mean_of(C, "candidate", "log_loss")
        imp = sum(1 for (_, c), (_, d) in zip(results[(C, "control")], results[(C, "candidate")])
                  if d.log_loss < c.log_loss)
        print(f"| MEAN | {cm:.16f} | {dm:.16f} | {dm - cm:+.16f} |")
        print(f"candidate improves in {imp}/3 folds")

    rule("STEP 10 -- FULL RESULTS: BRIER")
    for C in C_GRID:
        print(f"\n### C = {C}")
        print("| Fold | Control | Candidate | Delta |")
        print("|------|---------|-----------|-------|")
        for (n, c), (_, d) in zip(results[(C, "control")], results[(C, "candidate")]):
            print(f"| {n} | {c.brier:.16f} | {d.brier:.16f} | {d.brier - c.brier:+.16f} |")
        cm, dm = mean_of(C, "control", "brier"), mean_of(C, "candidate", "brier")
        imp = sum(1 for (_, c), (_, d) in zip(results[(C, "control")], results[(C, "candidate")])
                  if d.brier < c.brier)
        print(f"| MEAN | {cm:.16f} | {dm:.16f} | {dm - cm:+.16f} |")
        print(f"candidate improves in {imp}/3 folds")

    rule("STEP 11 -- PAIRED DELTA SUMMARY (matched C only)")
    print(f"  {'C':>8}{'dLogLoss':>22s}{'LL folds':>10s}{'dBrier':>22s}{'Brier folds':>13s}")
    for C in C_GRID:
        dll = mean_of(C, "candidate", "log_loss") - mean_of(C, "control", "log_loss")
        dbr = mean_of(C, "candidate", "brier") - mean_of(C, "control", "brier")
        illf = sum(1 for (_, c), (_, d) in zip(results[(C, "control")], results[(C, "candidate")])
                   if d.log_loss < c.log_loss)
        ibf = sum(1 for (_, c), (_, d) in zip(results[(C, "control")], results[(C, "candidate")])
                  if d.brier < c.brier)
        print(f"  {C:>8}{dll:+22.16f}{illf:>7d}/3{dbr:+22.16f}{ibf:>10d}/3")
    print("\n  negative = candidate better. Every row compares control and candidate")
    print("  at the SAME C. Rows are NOT comparable to one another as promotion")
    print("  evidence: choosing a C from this table and reporting its delta as an")
    print("  unbiased result would be selection on the validation data.")

    rule("STEP 12 -- POST-RUN SAFETY")
    for rel in EXPECTED_MD5:
        check(f"{rel} unchanged", md5(REPO / rel) == before[rel])
    bad = [r for r, v in pins.items() if md5(REPO / r) != v["expected"]]
    check("13/13 LOCKED_INPUTS still unchanged", not bad, str(bad))
    check("MODEL_B_COLUMNS still 80", len(MODEL_B_COLUMNS) == 80)
    check("MODEL_VERSION still v1.0", MODEL_VERSION == "v1.0")
    print("  no estimator persisted; no artifact written; 2025/26 never loaded")

    rule("D-27 COMPLETE -- DIAGNOSTIC EVIDENCE ONLY, NOT PROMOTION EVIDENCE")


if __name__ == "__main__":
    main()
