"""E-1 -- extended-C validation experiment (pre-2025/26 folds ONLY).

PRE-REGISTERED in docs/IMPROVEMENT_EXPERIMENT_PLAN_AND_TESTSET_LOCK.md
before this script was written. The grid, the primary metric, the
all-folds requirement and the rollback conditions were fixed in advance.

WHY THIS IS A SEPARATE MODULE AND NOT AN EDIT TO robustness.py
    `robustness.build_logistic_regression` deliberately REJECTS any C
    outside the frozen Phase 4B grid (0.01 ... 100.0), stating that
    widening it "would turn a pre-declared sensitivity check into an
    unregistered hyperparameter search". That guard is correct and is
    NOT modified here. E-1 therefore declares its OWN pre-registered
    grid in its own additive module, leaving Phase 4B's frozen grid and
    its recorded results untouched.

WHAT IS REUSED, NOT REIMPLEMENTED
    train.LogisticRegressionPreprocessor  (fit on X_train only)
    train._reorder_proba                  (CLASS_ORDER column ordering)
    robustness.V1_LOGREG_BASE_KWARGS      (max_iter=2000, random_state=0)
    robustness.train_logistic_regression_with_C  (control arm, C=1.0)
    ablation.MODEL_B_COLUMNS / select_feature_columns
    splits.iter_walk_forward_folds        (the 3 approved folds)
    evaluate.evaluate / validate_probabilities
    Only `C` differs. No calibration, no class weighting, no feature
    selection, no architecture change.

2025/26
    Structurally unreachable: this script imports `iter_walk_forward_folds`
    and never `final_split`; the three approved folds span 2020/21-2024/25
    only. No 2025/26 season constant, label, outcome, metric, or error
    analysis is read.

WRITES NOTHING. No artifact, no estimator, no report file. stdout only.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_e1_extended_c_validation.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from models.ablation import MODEL_B, MODEL_B_COLUMNS, select_feature_columns
from models.baselines import CLASS_ORDER
from models.config import FINAL_TEST_SEASONS, MODEL_VERSION, WALK_FORWARD_FOLDS
from models.data import load_supervised_dataset
from models.evaluate import evaluate, validate_probabilities
from models.robustness import (
    PHASE4A_C1_MEAN_LOG_LOSS,
    V1_LOGREG_BASE_KWARGS,
    train_logistic_regression_with_C,
)
from models.splits import iter_walk_forward_folds

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"

# ---------------------------------------------------------------------
# E-1's OWN pre-registered grid. Fixed in the plan document before this
# script existed. NOT extended at runtime under any circumstance.
# ---------------------------------------------------------------------
E1_C_GRID: tuple[float, ...] = (0.0001, 0.0005, 0.001, 0.005)
CONTROL_C: float = 1.0
GRID_EDGE_C: float = 0.0001          # stop-and-report if the winner lands here

#: Phase 4B's recorded Model B mean validation log loss at C=1.0. The E-1
#: path must reproduce this EXACTLY before any new-C result is trusted.
PHASE4B_MODEL_B_C1_MEAN_LOG_LOSS: float = PHASE4A_C1_MEAN_LOG_LOSS[MODEL_B]

FROZEN_V1_SOURCES = (
    "src/models/config.py", "src/models/train.py", "src/models/splits.py",
    "src/models/evaluate.py", "src/models/data.py", "src/models/baselines.py",
    "src/models/run_experiments.py",
)


class ExperimentStop(SystemExit):
    pass


def stop(message: str) -> None:
    print("\n" + "!" * 78)
    print("STOP -- experiment halted")
    print(message)
    print("!" * 78)
    raise ExperimentStop(1)


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_e1_estimator(C: float):
    """V1's exact kwargs plus the E-1 swept C. Single choke point.

    Mirrors robustness.build_logistic_regression's discipline: grid
    membership is checked BEFORE importing scikit-learn, so an off-grid C
    is rejected in any environment.
    """
    if C not in E1_C_GRID and C != CONTROL_C:
        raise ValueError(
            f"C={C!r} is not in the pre-registered E-1 grid {E1_C_GRID} (control {CONTROL_C}). "
            "The grid was fixed in docs/IMPROVEMENT_EXPERIMENT_PLAN_AND_TESTSET_LOCK.md before "
            "this experiment ran; widening it at runtime would make E-1 an unregistered search."
        )
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, **V1_LOGREG_BASE_KWARGS)


def e1_fit_predict(X_train, y_train, X_val, C: float) -> np.ndarray:
    """V1's training path with C as the single varied parameter.

    Identical in structure to robustness.train_logistic_regression_with_C;
    only the grid guard differs. Preprocessing and CLASS_ORDER handling are
    V1's own objects, unmodified.
    """
    from models import train as train_module

    preprocessor = train_module.LogisticRegressionPreprocessor().fit(X_train)
    X_train_enc = preprocessor.transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    model = build_e1_estimator(C)
    model.fit(X_train_enc, y_train)
    return train_module._reorder_proba(model, X_val_enc)


def per_class(y_true, P: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    out = {}
    for c in CLASS_ORDER:
        tp = int(((pred == c) & (y_true == c)).sum())
        fp = int(((pred == c) & (y_true != c)).sum())
        fn = int(((pred != c) & (y_true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = dict(precision=precision, recall=recall, f1=f1,
                      support=int((y_true == c).sum()))
    return out


def run_arm(dataset, C: float, label: str, use_authorized_path: bool = False) -> dict:
    """One C across all three approved folds. Returns metrics + probabilities."""
    folds: dict[str, dict] = {}
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        X_train = select_feature_columns(train_ds.X, MODEL_B_COLUMNS, MODEL_B)
        X_val = select_feature_columns(val_ds.X, MODEL_B_COLUMNS, MODEL_B)

        if use_authorized_path:
            P = train_logistic_regression_with_C(X_train, train_ds.y, X_val, C)
        else:
            P = e1_fit_predict(X_train, train_ds.y, X_val, C)

        P = np.asarray(P, dtype=float)
        assert P.ndim == 2 and P.shape[1] == 3, f"bad probability shape {P.shape}"
        assert P.shape[0] == len(val_ds.X), "row/fixture misalignment"
        validate_probabilities(P)

        result = evaluate(val_ds.y, P)
        folds[fold.name] = dict(
            metrics=result.as_dict(),
            per_class=per_class(val_ds.y, P),
            P=P,
            n=result.n,
        )
        print(f"    [{label}] {fold.name:<8} log_loss={result.log_loss:.16f}  "
              f"brier={result.brier:.6f}  acc={result.accuracy:.6f}  n={result.n}")

    means = {
        m: statistics.fmean(f["metrics"][m] for f in folds.values())
        for m in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy")
    }
    print(f"    [{label}] MEAN log_loss={means['log_loss']:.16f}  brier={means['brier']:.16f}")
    return dict(C=C, folds=folds, means=means)


def main() -> None:
    print("E-1 -- EXTENDED-C VALIDATION EXPERIMENT (pre-2025/26 folds only)")
    print("Pre-registered. Model B only. C is the single varied parameter.")

    # -----------------------------------------------------------------
    rule("PRE-FLIGHT")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION: {MODEL_VERSION!r}")
    print(f"  MODEL_B_COLUMNS: {len(MODEL_B_COLUMNS)} | CLASS_ORDER: {CLASS_ORDER}")
    print(f"  estimator kwargs (imported): {V1_LOGREG_BASE_KWARGS} + swept C")
    print(f"  E-1 pre-registered grid: {E1_C_GRID} | control C={CONTROL_C}")

    fold_seasons = {s for f in WALK_FORWARD_FOLDS
                    for s in (f.train_seasons + f.validation_seasons)}
    overlap = fold_seasons & set(FINAL_TEST_SEASONS)
    print(f"\n  seasons reachable by the 3 folds: {sorted(fold_seasons)}")
    print(f"  FINAL_TEST_SEASONS: {FINAL_TEST_SEASONS}")
    print(f"  2025/26 reachable by this experiment: {bool(overlap)}")
    if overlap:
        stop(f"2025/26 is reachable through the walk-forward folds: {overlap}")
    print("  [PASS] 2025/26 is structurally inaccessible (no final_split import, no test season)")

    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {rel: md5(REPO / rel) for rel in pins}
    mismatches = [rel for rel, record in pins.items() if before[rel] != record["expected"]]
    print(f"  [{'PASS' if not mismatches else 'FAIL'}] 13 pinned baselines: "
          f"{len(pins)} checked, mismatches={mismatches or 'NONE'}")
    if mismatches:
        stop(f"Pinned baseline mismatch before the experiment: {mismatches}")
    for rel in FROZEN_V1_SOURCES:
        print(f"      {md5(REPO / rel)}  {rel}")
    if MODEL_VERSION != "v1.0":
        stop(f"MODEL_VERSION is {MODEL_VERSION!r}, expected 'v1.0'")
    print("  [PASS] MODEL_VERSION == v1.0")
    print("  [PASS] no artifact will be overwritten -- this script writes nothing")
    print("  [PASS] additive/read-only: robustness.py C_GRID untouched, E-1 grid is separate")

    # -----------------------------------------------------------------
    rule("CONTROL -- frozen V1 (C=1.0), via the AUTHORIZED Phase 4B path")
    dataset = load_supervised_dataset(FEATURES_DB)
    print(f"  dataset rows: {len(dataset.X)}")
    control_authorized = run_arm(dataset, CONTROL_C, "V1 C=1.0 (authorized path)",
                                 use_authorized_path=True)

    rule("HARNESS FIDELITY -- reproduce the recorded Phase 4B value at C=1.0")
    recorded = PHASE4B_MODEL_B_C1_MEAN_LOG_LOSS
    observed = control_authorized["means"]["log_loss"]
    print(f"  Phase 4B recorded Model B mean log loss @C=1.0 : {recorded:.16f}")
    print(f"  observed this run                              : {observed:.16f}")
    print(f"  absolute difference                            : {abs(observed - recorded):.3e}")
    if abs(observed - recorded) >= 1e-9:
        stop("The control did NOT reproduce Phase 4B's recorded C=1.0 result to <1e-9. "
             "No new-C result may be trusted until this is explained.")
    print("  [PASS] control reproduces the recorded reference to <1e-9")

    print("\n  Cross-check: the E-1 code path must equal the authorized path at C=1.0")
    control_e1 = run_arm(dataset, CONTROL_C, "V1 C=1.0 (E-1 path)")
    identical = all(
        control_e1["folds"][name]["P"].tobytes() == control_authorized["folds"][name]["P"].tobytes()
        for name in control_authorized["folds"]
    )
    print(f"  E-1 path byte-identical to authorized path at C=1.0: {identical}")
    if not identical:
        stop("The E-1 code path diverges from the authorized Phase 4B path at C=1.0. "
             "The experiment would not be measuring C alone.")
    print("  [PASS] E-1 path proven faithful")

    control = control_authorized

    # -----------------------------------------------------------------
    rule("CANDIDATE ARMS")
    arms: dict[float, dict] = {}
    for C in E1_C_GRID:
        print(f"\n  --- C = {C} ---")
        arms[C] = run_arm(dataset, C, f"C={C}")

    # -----------------------------------------------------------------
    rule("PER-FOLD METRICS")
    for C, arm in [(CONTROL_C, control)] + list(arms.items()):
        tag = "V1 CONTROL" if C == CONTROL_C else "candidate"
        print(f"\n  C = {C}  ({tag})")
        for name, f in arm["folds"].items():
            m = f["metrics"]
            print(f"    {name}: log_loss={m['log_loss']:.16f} brier={m['brier']:.8f} "
                  f"acc={m['accuracy']:.6f} macro_f1={m['macro_f1']:.6f} "
                  f"bal_acc={m['balanced_accuracy']:.6f} n={f['n']}")
            for c in CLASS_ORDER:
                pc = f["per_class"][c]
                print(f"        {c}: precision={pc['precision']:.6f} recall={pc['recall']:.6f} "
                      f"f1={pc['f1']:.6f} support={pc['support']}")

    rule("MEAN METRICS AND DELTAS vs FROZEN V1 (C=1.0)")
    header = f"  {'C':<10} {'mean log loss':>20} {'delta':>14} {'mean Brier':>14} {'delta':>13} {'acc':>10} {'macroF1':>10} {'balAcc':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for C, arm in [(CONTROL_C, control)] + list(arms.items()):
        m = arm["means"]
        d_ll = m["log_loss"] - control["means"]["log_loss"]
        d_br = m["brier"] - control["means"]["brier"]
        tag = "  <-- V1" if C == CONTROL_C else ""
        print(f"  {C:<10} {m['log_loss']:>20.16f} {d_ll:>14.3e} {m['brier']:>14.8f} "
              f"{d_br:>13.3e} {m['accuracy']:>10.6f} {m['macro_f1']:>10.6f} "
              f"{m['balanced_accuracy']:>10.6f}{tag}")

    # -----------------------------------------------------------------
    rule("DETERMINISM -- repeated fits of every configuration")
    determinism: dict[float, bool] = {}
    for C, arm in [(CONTROL_C, control)] + list(arms.items()):
        repeat = run_arm(dataset, C, f"repeat C={C}",
                         use_authorized_path=(C == CONTROL_C))
        same = all(
            repeat["folds"][name]["P"].tobytes() == arm["folds"][name]["P"].tobytes()
            for name in arm["folds"]
        )
        determinism[C] = same
        digest = hashlib.sha256(
            b"".join(arm["folds"][n]["P"].tobytes() for n in sorted(arm["folds"]))
        ).hexdigest()
        print(f"    C={C:<8} byte-identical repeat: {same}  sha256={digest[:32]}...")
    if not all(determinism.values()):
        stop(f"Determinism failed for: {[C for C, ok in determinism.items() if not ok]}")
    print("  [PASS] every configuration is deterministic")

    # -----------------------------------------------------------------
    rule("PRE-REGISTERED SELECTION RULE")
    print("  A candidate may replace V1 ONLY if ALL hold:")
    print("    (1) lowest mean validation log loss of all arms including V1")
    print("    (2) improves log loss in ALL THREE folds individually")
    print("    (3) mean Brier does not worsen")
    print("    (4) determinism passes")
    print("  Secondary metrics are diagnostic only and cannot override the rule.\n")

    best_C = min([CONTROL_C] + list(E1_C_GRID),
                 key=lambda c: (control if c == CONTROL_C else arms[c])["means"]["log_loss"])
    qualifying = []
    for C in E1_C_GRID:
        arm = arms[C]
        c1 = arm["means"]["log_loss"] < control["means"]["log_loss"] and C == best_C
        per_fold = {
            name: arm["folds"][name]["metrics"]["log_loss"]
                  < control["folds"][name]["metrics"]["log_loss"]
            for name in control["folds"]
        }
        c2 = all(per_fold.values())
        c3 = arm["means"]["brier"] <= control["means"]["brier"]
        c4 = determinism[C]
        passed = c1 and c2 and c3 and c4
        if passed:
            qualifying.append(C)
        print(f"  C={C}:")
        print(f"    (1) lowest mean log loss overall : {c1}  "
              f"(mean {arm['means']['log_loss']:.16f} vs V1 {control['means']['log_loss']:.16f})")
        print(f"    (2) improves in ALL three folds  : {c2}  {per_fold}")
        print(f"    (3) mean Brier not worsened      : {c3}  "
              f"({arm['means']['brier']:.10f} vs {control['means']['brier']:.10f})")
        print(f"    (4) determinism                  : {c4}")
        print(f"    => {'QUALIFIES' if passed else 'DOES NOT QUALIFY'}")

    # -----------------------------------------------------------------
    rule("OUTCOME")
    print(f"  arm with lowest mean validation log loss: C={best_C}")

    if not qualifying:
        print("\n  *** NO CHANGE WARRANTED ***")
        print("  No candidate satisfied every pre-registered condition.")
        print("  V1 (C=1.0) is RETAINED UNCHANGED. Nothing is frozen, nothing is promoted.")
    elif len(qualifying) > 1:
        stop(f"More than one candidate qualified ({qualifying}); the rule expects a unique "
             "winner by lowest mean log loss. Halting rather than choosing arbitrarily.")
    else:
        winner = qualifying[0]
        if winner == GRID_EDGE_C:
            print(f"\n  *** GRID-EDGE CONDITION ***")
            print(f"  The qualifying candidate is C={GRID_EDGE_C}, the lower edge of the "
                  "pre-registered E-1 grid.")
            print("  Per the plan, this is REPORTED, NOT ACTED UPON. The grid is NOT extended.")
            print("  No candidate is frozen; V1 remains in place pending a governance decision.")
        else:
            print(f"\n  *** CANDIDATE QUALIFIES: C={winner} ***")
            arm = arms[winner]
            print("  Frozen candidate configuration:")
            print(f"    estimator      : LogisticRegression(C={winner}, "
                  f"max_iter={V1_LOGREG_BASE_KWARGS['max_iter']}, "
                  f"random_state={V1_LOGREG_BASE_KWARGS['random_state']})")
            print(f"    features       : MODEL_B_COLUMNS ({len(MODEL_B_COLUMNS)}), contract order")
            print( "    preprocessing  : V1 LogisticRegressionPreprocessor (unchanged)")
            print( "    calibration    : none")
            print(f"    mean log loss  : {arm['means']['log_loss']:.16f} "
                  f"(delta {arm['means']['log_loss'] - control['means']['log_loss']:+.3e})")
            print(f"    mean Brier     : {arm['means']['brier']:.16f}")
            print("  This is a validation-only result. NO 2025/26 access is performed here.")
            print("  A separate explicit authorization is required for the one-time final retest.")

    print("\n  No result in this experiment is described as statistically significant "
          "(no test was performed).")
    print("  No claim of generalization beyond the validation evidence is made.")

    # -----------------------------------------------------------------
    rule("INTEGRITY / FORBIDDEN-OPERATION AUDIT")
    after = {rel: md5(REPO / rel) for rel in pins}
    drifted = [rel for rel in pins if before[rel] != after[rel]]
    print(f"  13 pinned baselines after the run: mismatches={drifted or 'NONE'}")
    if drifted:
        stop(f"The experiment altered pinned files: {drifted}")
    for rel in ("src/models/robustness.py", "src/models/ablation.py",
                "src/models/candidate_contract.py", "src/models/calibration.py"):
        print(f"    {md5(REPO / rel)}  {rel}")
    print(f"  MODEL_VERSION: {MODEL_VERSION!r}")
    serialized = [str(p.relative_to(REPO)) for pat in ("*.pkl", "*.joblib", "*.pickle", "*.onnx")
                  for p in REPO.rglob(pat) if "__pycache__" not in p.parts]
    print(f"  serialized estimators on disk: {serialized or 'NONE'}")
    for line in (
        "2025/26 accessed              : NO -- structurally unreachable, verified in pre-flight",
        "2025/26 outcomes/errors used  : NO",
        "calibration                   : NONE",
        "class weighting               : NONE",
        "feature selection             : NONE",
        "architecture change           : NONE",
        "grid extended at runtime      : NO -- E-1 grid fixed, off-grid C rejected by guard",
        "robustness.py C_GRID modified : NO",
        "model artifact persisted      : NONE",
        "artifact/database written     : NONE",
        "file created by this script   : NONE",
    ):
        print(f"  {line}")

    rule("END OF E-1 EXPERIMENT")


if __name__ == "__main__":
    main()
