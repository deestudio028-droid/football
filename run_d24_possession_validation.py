"""D-24 -- POSSESSION paired walk-forward validation (CONTROL vs CANDIDATE).

Run on the project machine (scikit-learn required):

    python run_d24_possession_validation.py

Read-only and additive. Writes nothing, persists no estimator, creates
no artifact, and never touches 2025/26. Every gate below is a hard fail
(SystemExit), never a warning -- if any stop condition fires, the run
aborts before or during fitting rather than producing a result that
cannot be trusted.

WHAT IS COMPARED
    CONTROL   ablation.MODEL_B_COLUMNS                     (80 columns)
              loaded from data/processed/features.db       via models.data
    CANDIDATE successor_contract.MODEL_B_PLUS_POSSESSION   (86 columns)
              loaded from data/processed/features_v1_1.db  via models.successor_data

Both arms are fitted by the SAME production function,
`train.train_logistic_regression`, so the estimator class, hyper-
parameters (max_iter=2000, C=1.0, random_state=0), imputer, scaler,
one-hot encoding and predict_proba reordering are identical by
construction rather than by configuration. The only difference the
candidate sees is six extra input columns.

NOTE ON C. The production path hardcodes C=1.0 (train.py:149), which is
the value Phase 4C's final-test evidence exists at and which
PHASE5_V2_MODEL_SPEC_DRAFT §63 records as frozen. The earlier D-6
in-memory experiment used C=0.0005, so D-24's numbers are NOT directly
comparable to D-6's and must not be read against them.

NOT ANSWERED HERE: whether POSSESSION should be promoted.
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

EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "data/processed/features_v1_1.db": "6aa22e0895bd2189edcc0b54859fa749",
}


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 78)
    print("STOP -- D-24 halted. No result is produced.")
    print(msg)
    print("!" * 78)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        stop(f"stop condition: {label}")


def exact(v):
    """Null-aware exact identity. NaN != NaN under ==, and None must
    never compare equal to 0.0, so both become sentinels."""
    if v is None:
        return ("NULL",)
    if isinstance(v, float) and v != v:
        return ("NULL",)
    try:
        if pd.isna(v):
            return ("NULL",)
    except (TypeError, ValueError):
        pass
    return (type(v).__name__, v)


def main():
    # ---------------- §2 environment ----------------
    rule("STEP 1 -- ENVIRONMENT (sklearn is a hard requirement)")
    print(f"  Python  {sys.version.split()[0]}")
    print(f"  numpy   {np.__version__}")
    print(f"  pandas  {pd.__version__}")
    try:
        import sklearn
        print(f"  sklearn {sklearn.__version__}")
    except ImportError as exc:
        stop(f"scikit-learn is not available: {exc}. Do not substitute another "
             "estimator and do not hand-roll one.")
    from models.train import train_logistic_regression  # noqa: E402
    print("  models.train.train_logistic_regression imported OK")

    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        FINAL_TEST_SEASONS, MODEL_VERSION, SEASON_NAME_TO_IDS,
    )
    from models.data import load_supervised_dataset
    from models.evaluate import evaluate
    from models.splits import iter_walk_forward_folds
    from models.successor_contract import (
        MODEL_B_PLUS_POSSESSION_COLUMNS as CANDIDATE_COLUMNS,
        POSSESSION_FAMILY_COLUMNS,
    )
    from models.successor_data import load_successor_supervised_dataset

    # ---------------- §4 database + pin integrity ----------------
    rule("STEP 2 -- DATABASE AND PIN INTEGRITY (before fitting)")
    before = {}
    for rel, exp in EXPECTED_MD5.items():
        got = md5(REPO / rel)
        before[rel] = got
        check(f"{rel} == {exp}", got == exp, got)
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if md5(REPO / r) != v["expected"]]
    check("13/13 LOCKED_INPUTS unchanged", not bad, str(bad))
    check("MODEL_VERSION is v1.0", MODEL_VERSION == "v1.0", MODEL_VERSION)

    # ---------------- §5 contract parity ----------------
    rule("STEP 3 -- CONTRACT PARITY (loaded from sources of truth)")
    check("len(control) == 80", len(MODEL_B_COLUMNS) == 80)
    check("len(candidate) == 86", len(CANDIDATE_COLUMNS) == 86)
    check("candidate[:80] == MODEL_B_COLUMNS exactly, in order",
          tuple(CANDIDATE_COLUMNS[:80]) == tuple(MODEL_B_COLUMNS))
    check("candidate[80:] == POSSESSION_FAMILY_COLUMNS",
          CANDIDATE_COLUMNS[80:] == POSSESSION_FAMILY_COLUMNS)

    # ---------------- §3 firewall + load ----------------
    rule("STEP 4 -- LOAD (V1 loader for control, successor loader for candidate)")
    control_ds = load_supervised_dataset(REPO / "data/processed/features.db")
    candidate_ds = load_successor_supervised_dataset(
        REPO / "data/processed/features_v1_1.db")
    print(f"  control rows   : {len(control_ds)}  "
          f"versions={set(control_ds.metadata['feature_version'].unique())}")
    print(f"  candidate rows : {len(candidate_ds)}  "
          f"versions={set(candidate_ds.metadata['feature_version'].unique())}")
    check("control feature_version == {'v1.0'}",
          set(control_ds.metadata["feature_version"].unique()) == {"v1.0"})
    check("candidate feature_version == {'v1.1'}",
          set(candidate_ds.metadata["feature_version"].unique()) == {"v1.1"})

    test_ids = set()
    for name in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[name])
    print(f"  final-test season_ids (exact IDs, no substring): {sorted(test_ids)}")

    # ---------------- §6 fold parity ----------------
    rule("STEP 5 -- FOLD PARITY")
    c_folds = list(iter_walk_forward_folds(control_ds))
    s_folds = list(iter_walk_forward_folds(candidate_ds))
    check("three folds in both arms", len(c_folds) == len(s_folds) == 3)
    for (f1, tr1, va1), (f2, tr2, va2) in zip(c_folds, s_folds):
        check(f"{f1.name}: same fold definition", f1.name == f2.name)
        check(f"{f1.name}: identical train fixture_ids",
              tr1.metadata["fixture_id"].tolist() == tr2.metadata["fixture_id"].tolist())
        check(f"{f1.name}: identical validation fixture_ids",
              va1.metadata["fixture_id"].tolist() == va2.metadata["fixture_id"].tolist())
        check(f"{f1.name}: identical train labels", tr1.y.tolist() == tr2.y.tolist())
        check(f"{f1.name}: identical validation labels", va1.y.tolist() == va2.y.tolist())
        seen = set(tr1.metadata["season_id"]) | set(va1.metadata["season_id"]) \
            | set(tr2.metadata["season_id"]) | set(va2.metadata["season_id"])
        check(f"{f1.name}: no final-test season_id present", not (seen & test_ids))
        print(f"    {f1.name}: train={len(tr1)}  validation={len(va1)}")

    # ---------------- §13 shared-feature parity ----------------
    rule("STEP 6 -- SHARED 80-FEATURE PARITY (null-aware exact, joined by fixture_id)")
    total = diffs = 0
    first = None
    for (f1, tr1, va1), (f2, tr2, va2) in zip(c_folds, s_folds):
        for a, b, tag in ((tr1, tr2, "train"), (va1, va2, "validation")):
            A = a.X.set_index(a.metadata["fixture_id"].values)[list(MODEL_B_COLUMNS)]
            B = b.X.set_index(b.metadata["fixture_id"].values)[list(MODEL_B_COLUMNS)]
            check(f"{f1.name}/{tag}: index alignment", A.index.equals(B.index))
            for col in MODEL_B_COLUMNS:
                for i, (x, y) in enumerate(zip(A[col].tolist(), B[col].tolist())):
                    total += 1
                    if exact(x) != exact(y):
                        diffs += 1
                        if first is None:
                            first = (f1.name, tag, col, int(A.index[i]), x, y)
    print(f"  cells compared: {total}   differing cells: {diffs}")
    if first:
        print(f"  first mismatch: {first}")
    check("every shared 80-column value is exactly identical", diffs == 0)

    # ---------------- §14 availability ----------------
    rule("STEP 7 -- POSSESSION MISSINGNESS (diagnostic, pre-imputation)")
    hdr = "  " + f"{'fold':8s}{'part':12s}" + "".join(f"{c[:26]:>28s}" for c in POSSESSION_FAMILY_COLUMNS)
    print(hdr[:200])
    for (f2, tr2, va2) in s_folds:
        for ds, tag in ((tr2, "train"), (va2, "validation")):
            pct = "".join(f"{100 * ds.X[c].isna().mean():27.2f}%" for c in POSSESSION_FAMILY_COLUMNS)
            print(f"  {f2.name:8s}{tag:12s}{pct}")
    print("  (diagnostic only -- no row is dropped and no value is imputed here;")
    print("   imputation happens inside the production preprocessor, train-fold only)")

    # ---------------- §8 paired experiment ----------------
    rule("STEP 8 -- PAIRED WALK-FORWARD FIT (identical production function, both arms)")
    print("  estimator: LogisticRegression(max_iter=2000, C=1.0, random_state=0)")
    print("  preprocessing: SimpleImputer(median) + StandardScaler + one-hot competition_id,")
    print("                 all fitted on the training fold only")
    print("  no calibration, no class weighting, no model selection, no tuning\n")

    results = {"control": [], "candidate": []}
    for (f1, tr1, va1), (f2, tr2, va2) in zip(c_folds, s_folds):
        _, _, P_ctrl = train_logistic_regression(
            tr1.X[list(MODEL_B_COLUMNS)], tr1.y, va1.X[list(MODEL_B_COLUMNS)])
        _, _, P_cand = train_logistic_regression(
            tr2.X[list(CANDIDATE_COLUMNS)], tr2.y, va2.X[list(CANDIDATE_COLUMNS)])
        r_ctrl = evaluate(va1.y, P_ctrl)
        r_cand = evaluate(va2.y, P_cand)
        results["control"].append((f1.name, r_ctrl))
        results["candidate"].append((f2.name, r_cand))
        print(f"  {f1.name}: control LL={r_ctrl.log_loss:.16f}  "
              f"candidate LL={r_cand.log_loss:.16f}")

    def recall(res, cls):
        i = CLASS_ORDER.index(cls)
        cm = res.confusion_matrix
        support = cm[i, :].sum()
        return cm[i, i] / support if support else float("nan")

    def mean(arm, attr):
        return float(np.mean([getattr(r, attr) for _, r in results[arm]]))

    # ---------------- §15 tables ----------------
    rule("STEP 9 -- RESULTS")
    for metric in ("log_loss", "brier"):
        print(f"\n### {metric}")
        print("| Fold | Control | Candidate | Delta |")
        print("|------|---------|-----------|-------|")
        for (n, c), (_, d) in zip(results["control"], results["candidate"]):
            cv, dv = getattr(c, metric), getattr(d, metric)
            print(f"| {n} | {cv:.16f} | {dv:.16f} | {dv - cv:+.16f} |")
        cm_, dm_ = mean("control", metric), mean("candidate", metric)
        improved = sum(1 for (_, c), (_, d) in zip(results["control"], results["candidate"])
                       if getattr(d, metric) < getattr(c, metric))
        print(f"| MEAN | {cm_:.16f} | {dm_:.16f} | {dm_ - cm_:+.16f} |")
        print(f"improvement count: {improved}/3")

    print("\n### Secondary metrics (sanity diagnostics only)")
    print("| Metric | Control Mean | Candidate Mean | Delta |")
    print("|--------|--------------|----------------|-------|")
    for attr, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
                        ("balanced_accuracy", "Balanced Accuracy")):
        c_, d_ = mean("control", attr), mean("candidate", attr)
        print(f"| {label} | {c_:.16f} | {d_:.16f} | {d_ - c_:+.16f} |")
    for cls, label in (("D", "Draw Recall"), ("A", "Away Recall")):
        c_ = float(np.mean([recall(r, cls) for _, r in results["control"]]))
        d_ = float(np.mean([recall(r, cls) for _, r in results["candidate"]]))
        print(f"| {label} | {c_:.16f} | {d_:.16f} | {d_ - c_:+.16f} |")

    # ---------------- §12 determinism ----------------
    rule("STEP 10 -- DETERMINISM (exact repeat)")
    ok = True
    for (f1, tr1, va1), (f2, tr2, va2), (_, r_ctrl), (_, r_cand) in zip(
            c_folds, s_folds, results["control"], results["candidate"]):
        _, _, P1 = train_logistic_regression(
            tr1.X[list(MODEL_B_COLUMNS)], tr1.y, va1.X[list(MODEL_B_COLUMNS)])
        _, _, P2 = train_logistic_regression(
            tr2.X[list(CANDIDATE_COLUMNS)], tr2.y, va2.X[list(CANDIDATE_COLUMNS)])
        a, b = evaluate(va1.y, P1), evaluate(va2.y, P2)
        same = (a.log_loss == r_ctrl.log_loss and b.log_loss == r_cand.log_loss
                and a.brier == r_ctrl.brier and b.brier == r_cand.brier)
        ok &= same
        print(f"  {f1.name}: identical on repeat = {same}")
    check("deterministic repeat reproduces every metric exactly", ok)
    print("  (deterministic by construction: random_state=0, no n_jobs, no shuffling,")
    print("   fixed fold definitions, and no sampling anywhere in the path)")

    # ---------------- §17 post-run safety ----------------
    rule("STEP 11 -- POST-RUN SAFETY")
    for rel in EXPECTED_MD5:
        check(f"{rel} unchanged", md5(REPO / rel) == before[rel])
    bad = [r for r, v in pins.items() if md5(REPO / r) != v["expected"]]
    check("13/13 LOCKED_INPUTS still unchanged", not bad, str(bad))
    check("MODEL_B_COLUMNS still 80", len(MODEL_B_COLUMNS) == 80)
    check("MODEL_VERSION still v1.0", MODEL_VERSION == "v1.0")
    print("  no estimator persisted; no artifact written; 2025/26 never loaded")

    rule("D-24 COMPLETE -- EVIDENCE ONLY, NOT A PROMOTION")


if __name__ == "__main__":
    main()
