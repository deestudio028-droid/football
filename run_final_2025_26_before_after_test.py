"""THE SINGLE AUTHORIZED 2025/26 BEFORE-vs-AFTER FINAL TEST.

BEFORE : frozen V1 Model B, C=1.0
AFTER  : frozen E-1 candidate Model B, C=0.0005

Both are trained ONLY on FINAL_TRAIN_SEASONS (2020/21-2024/25) and then
predict the identical 1,751 fixtures of the locked 2025/26 partition.

THIS IS A ONE-TIME TEST. The candidate was frozen BEFORE any 2025/26
access, on validation-fold evidence alone (E-1, executed on Windows,
scikit-learn 1.9.0). Nothing here tunes, selects, calibrates, or adjusts
anything. The result must not be used to choose another C or to modify
the model in any way.

CANDIDATE FREEZE (recorded before this script existed)
    estimator     : LogisticRegression(C=0.0005, max_iter=2000, random_state=0)
    features      : MODEL_B_COLUMNS (80), contract order
    preprocessing : V1 LogisticRegressionPreprocessor, unchanged
    calibration   : none
    validation    : mean log loss 0.9915719393130367 (V1 0.9993791056968738)
                    improved in all 3 folds; Brier 0.5914091026848851;
                    deterministic; not at grid edge.

WRITES NOTHING. No artifact, no estimator, no report file. stdout only.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_final_2025_26_before_after_test.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from models.ablation import MODEL_B, MODEL_B_COLUMNS, select_feature_columns
from models.baselines import CLASS_ORDER
from models.config import FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, MODEL_VERSION
from models.data import load_supervised_dataset
from models.evaluate import evaluate, validate_probabilities
from models.robustness import V1_LOGREG_BASE_KWARGS
from models.splits import final_split
from models.train import train_logistic_regression

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"

# ---------------------------------------------------------------------
# FROZEN CONSTANTS. Neither may be changed by this script or its caller.
# ---------------------------------------------------------------------
V1_C: float = 1.0
CANDIDATE_C: float = 0.0005          # frozen by E-1; no other value is permitted

#: Locked Phase 4C Model B reference. BEFORE must reproduce this exactly,
#: otherwise the harness is not faithful and no AFTER result is trusted.
PHASE4C_MODEL_B_LOG_LOSS: float = 0.9960487649063601

#: E-1 validation evidence, quoted for the record only. Never recomputed
#: here and never used to make any decision in this script.
E1_VALIDATION = {
    "v1_mean_log_loss": 0.9993791056968738,
    "candidate_mean_log_loss": 0.9915719393130367,
    "candidate_mean_brier": 0.5914091026848851,
}

FROZEN_V1_SOURCES = (
    "src/models/config.py", "src/models/train.py", "src/models/splits.py",
    "src/models/evaluate.py", "src/models/data.py", "src/models/baselines.py",
    "src/models/run_experiments.py",
)


class TestStop(SystemExit):
    pass


def stop(message: str) -> None:
    print("\n" + "!" * 78)
    print("STOP -- final test halted")
    print(message)
    print("!" * 78)
    raise TestStop(1)


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_candidate_estimator(C: float):
    """The frozen candidate estimator. Refuses any C other than the frozen one."""
    if C != CANDIDATE_C:
        raise ValueError(
            f"C={C!r} is not the frozen E-1 candidate value {CANDIDATE_C}. This script may not "
            "fit any other C: the candidate was frozen before 2025/26 was accessed, and trying "
            "another value here would be tuning on the locked test set."
        )
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, **V1_LOGREG_BASE_KWARGS)


def fit_predict_v1(train_ds, test_ds) -> np.ndarray:
    """BEFORE arm: V1's own frozen entry point, C hardcoded to 1.0 inside it."""
    X_train = select_feature_columns(train_ds.X, MODEL_B_COLUMNS, MODEL_B)
    X_test = select_feature_columns(test_ds.X, MODEL_B_COLUMNS, MODEL_B)
    _model, _preprocessor, P = train_logistic_regression(X_train, train_ds.y, X_test)
    return np.asarray(P, dtype=float)


def fit_predict_candidate(train_ds, test_ds) -> np.ndarray:
    """AFTER arm: identical path, single varied parameter C=0.0005."""
    from models import train as train_module

    X_train = select_feature_columns(train_ds.X, MODEL_B_COLUMNS, MODEL_B)
    X_test = select_feature_columns(test_ds.X, MODEL_B_COLUMNS, MODEL_B)
    preprocessor = train_module.LogisticRegressionPreprocessor().fit(X_train)
    model = build_candidate_estimator(CANDIDATE_C)
    model.fit(preprocessor.transform(X_train), train_ds.y)
    P = train_module._reorder_proba(model, preprocessor.transform(X_test))
    return np.asarray(P, dtype=float)


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
                      support=int((y_true == c).sum()), tp=tp, fp=fp, fn=fn)
    return out


def arm_report(label: str, y_true, P: np.ndarray) -> dict:
    validate_probabilities(P)
    result = evaluate(y_true, P)
    pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    correct = int((pred == np.asarray(y_true)).sum())
    return dict(
        label=label, P=P, pred=pred, result=result,
        correct=correct, incorrect=len(pred) - correct,
        per_class=per_class(y_true, P),
        distribution={c: int((pred == c).sum()) for c in CLASS_ORDER},
        mean_prob={c: float(P[:, i].mean()) for i, c in enumerate(CLASS_ORDER)},
    )


def print_arm(a: dict, y_true) -> None:
    r = a["result"]
    n = len(a["pred"])
    print(f"\n  --- {a['label']} ---")
    print(f"    total fixtures      : {n}")
    print(f"    correct / incorrect : {a['correct']} / {a['incorrect']}")
    print(f"    accuracy            : {r.accuracy:.16f}")
    print(f"    log loss            : {r.log_loss:.16f}")
    print(f"    Brier               : {r.brier:.16f}")
    print(f"    macro-F1            : {r.macro_f1:.16f}")
    print(f"    balanced accuracy   : {r.balanced_accuracy:.16f}")
    print("    per-class:")
    for c in CLASS_ORDER:
        pc = a["per_class"][c]
        print(f"      {c}: precision={pc['precision']:.6f} recall={pc['recall']:.6f} "
              f"f1={pc['f1']:.6f} support={pc['support']} (tp={pc['tp']} fp={pc['fp']} fn={pc['fn']})")
    print(f"    confusion matrix (rows=true {CLASS_ORDER}, cols=pred):")
    for row in r.confusion_matrix:
        print(f"      {row}")
    print("    predicted class distribution:")
    for c in CLASS_ORDER:
        k = a["distribution"][c]
        print(f"      {c}: {k:5d}  ({k / n * 100:.4f}%)")
    print("    mean predicted probability vs actual frequency:")
    y = np.asarray(y_true)
    for c in CLASS_ORDER:
        mp = a["mean_prob"][c]
        af = float((y == c).mean())
        print(f"      {c}: mean_pred={mp:.6f}  actual={af:.6f}  delta={mp - af:+.6f}")


def main() -> None:
    print("SINGLE AUTHORIZED 2025/26 BEFORE-vs-AFTER FINAL TEST")
    print("BEFORE: frozen V1 Model B C=1.0   |   AFTER: frozen E-1 candidate Model B C=0.0005")
    print("One-time test. No tuning before, during, or after.")

    rule("PRE-FLIGHT (before any 2025/26 outcome is read)")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION            : {MODEL_VERSION!r}")
    print(f"  FINAL_TRAIN_SEASONS      : {FINAL_TRAIN_SEASONS}")
    print(f"  FINAL_TEST_SEASONS       : {FINAL_TEST_SEASONS}")
    print(f"  MODEL_B_COLUMNS          : {len(MODEL_B_COLUMNS)} | CLASS_ORDER {CLASS_ORDER}")
    print(f"  BEFORE C                 : {V1_C}")
    print(f"  AFTER  C (frozen by E-1) : {CANDIDATE_C}")
    print(f"  estimator kwargs         : {V1_LOGREG_BASE_KWARGS} + C")
    print(f"  calibration              : none (both arms)")
    print("\n  E-1 validation evidence (quoted, not recomputed, not used to decide anything here):")
    for k, v in E1_VALIDATION.items():
        print(f"    {k}: {v}")

    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before_sums = {rel: md5(REPO / rel) for rel in pins}
    mismatches = [r for r, rec in pins.items() if before_sums[r] != rec["expected"]]
    print(f"\n  13 pinned baselines: {len(pins)} checked | mismatches={mismatches or 'NONE'}")
    if mismatches:
        stop(f"Pinned baseline mismatch before the test: {mismatches}")
    for rel in FROZEN_V1_SOURCES:
        print(f"    {md5(REPO / rel)}  {rel}")
    if MODEL_VERSION != "v1.0":
        stop(f"MODEL_VERSION is {MODEL_VERSION!r}, expected 'v1.0'")

    rule("SPLIT -- train on pre-2025/26 only, predict 2025/26")
    dataset = load_supervised_dataset(FEATURES_DB)
    train_ds, test_ds = final_split(dataset)       # verifies temporal safety internally
    print(f"  n_train = {len(train_ds.X)}   (seasons {FINAL_TRAIN_SEASONS})")
    print(f"  n_test  = {len(test_ds.X)}    (seasons {FINAL_TEST_SEASONS})")
    if len(test_ds.X) != 1751:
        stop(f"expected 1,751 test fixtures, got {len(test_ds.X)}")
    max_train_unix = train_ds.metadata["unix"].max()
    min_test_unix = test_ds.metadata["unix"].min()
    print(f"  max(train.unix)={max_train_unix} < min(test.unix)={min_test_unix}: "
          f"{max_train_unix < min_test_unix}")
    print("  [PASS] both arms will be fitted on identical training rows and scored on "
          "identical test fixtures")

    rule("FIT BOTH ARMS (no test outcome consulted during fitting)")
    P_before = fit_predict_v1(train_ds, test_ds)
    print(f"  BEFORE fitted: probabilities {P_before.shape}")
    P_after = fit_predict_candidate(train_ds, test_ds)
    print(f"  AFTER  fitted: probabilities {P_after.shape}")
    if P_before.shape != P_after.shape or P_before.shape != (1751, 3):
        stop(f"probability shape mismatch: {P_before.shape} vs {P_after.shape}")

    y_true = test_ds.y.to_numpy()
    fixture_ids = test_ds.metadata["fixture_id"].to_numpy()

    rule("HARNESS FIDELITY -- BEFORE must reproduce the locked Phase 4C reference")
    before = arm_report("BEFORE  (frozen V1, C=1.0)", y_true, P_before)
    print(f"  locked Phase 4C Model B log loss : {PHASE4C_MODEL_B_LOG_LOSS:.16f}")
    print(f"  BEFORE observed                  : {before['result'].log_loss:.16f}")
    diff = abs(before["result"].log_loss - PHASE4C_MODEL_B_LOG_LOSS)
    print(f"  absolute difference              : {diff:.3e}")
    if diff >= 1e-9:
        stop("BEFORE did not reproduce the locked Phase 4C reference to <1e-9. "
             "The AFTER result cannot be trusted until this is explained.")
    print("  [PASS] BEFORE reproduces the locked reference exactly")

    after = arm_report("AFTER   (frozen candidate, C=0.0005)", y_true, P_after)

    rule("1-11. RESULTS PER ARM")
    print_arm(before, y_true)
    print_arm(after, y_true)

    rule("12. BEFORE -> AFTER DELTAS")
    print(f"  {'Metric':<22} {'BEFORE':>22} {'AFTER':>22} {'Delta':>16}")
    print("  " + "-" * 84)
    for name, attr in (("accuracy", "accuracy"), ("log_loss", "log_loss"), ("brier", "brier"),
                       ("macro_f1", "macro_f1"), ("balanced_accuracy", "balanced_accuracy")):
        a = getattr(before["result"], attr)
        b = getattr(after["result"], attr)
        print(f"  {name:<22} {a:>22.16f} {b:>22.16f} {b - a:>+16.3e}")
    print(f"  {'correct':<22} {before['correct']:>22} {after['correct']:>22} "
          f"{after['correct'] - before['correct']:>+16}")
    print(f"  {'incorrect':<22} {before['incorrect']:>22} {after['incorrect']:>22} "
          f"{after['incorrect'] - before['incorrect']:>+16}")
    print("\n  per-class deltas:")
    for c in CLASS_ORDER:
        a, b = before["per_class"][c], after["per_class"][c]
        print(f"    {c}: precision {a['precision']:.6f} -> {b['precision']:.6f} "
              f"({b['precision'] - a['precision']:+.6f}) | "
              f"recall {a['recall']:.6f} -> {b['recall']:.6f} ({b['recall'] - a['recall']:+.6f}) | "
              f"f1 {a['f1']:.6f} -> {b['f1']:.6f} ({b['f1'] - a['f1']:+.6f})")
    print("\n  predicted distribution delta:")
    for c in CLASS_ORDER:
        print(f"    {c}: {before['distribution'][c]} -> {after['distribution'][c]} "
              f"({after['distribution'][c] - before['distribution'][c]:+d})")

    rule("13. FIXTURES WHOSE PREDICTED CLASS CHANGED")
    changed = before["pred"] != after["pred"]
    print(f"  changed: {int(changed.sum())} of {len(changed)} "
          f"({changed.mean() * 100:.4f}%)")
    if changed.any():
        transitions: dict[str, int] = {}
        for b, a in zip(before["pred"][changed], after["pred"][changed]):
            transitions[f"{b}->{a}"] = transitions.get(f"{b}->{a}", 0) + 1
        for k in sorted(transitions, key=lambda t: -transitions[t]):
            print(f"    {k}: {transitions[k]}")
        newly_right = int(((before["pred"] != y_true) & (after["pred"] == y_true)).sum())
        newly_wrong = int(((before["pred"] == y_true) & (after["pred"] != y_true)).sum())
        print(f"  newly correct : {newly_right}")
        print(f"  newly wrong   : {newly_wrong}")
        print(f"  net           : {newly_right - newly_wrong:+d}")

    rule("14. PROBABILITY MOVEMENT")
    diff_abs = np.abs(P_after - P_before)
    per_fixture_max = diff_abs.max(axis=1)
    ordered = np.sort(per_fixture_max)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    print(f"  identical probability rows      : {int((per_fixture_max == 0).sum())}")
    print(f"  non-identical                   : {int((per_fixture_max > 0).sum())}")
    print(f"  mean |delta| (all H/D/A cells)  : {diff_abs.mean():.6e}")
    print(f"  median per-fixture max |delta|  : {np.median(per_fixture_max):.6e}")
    print(f"  p95 per-fixture max |delta|     : {p95:.6e}")
    print(f"  max per-fixture max |delta|     : {per_fixture_max.max():.6e}")
    for t in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        print(f"    fixtures with max delta > {t:<7g}: {int((per_fixture_max > t).sum())}")
    print("  per-class mean signed movement (AFTER - BEFORE):")
    for i, c in enumerate(CLASS_ORDER):
        print(f"    {c}: {float((P_after[:, i] - P_before[:, i]).mean()):+.6e}")

    rule("15. DETERMINISTIC REPEAT")
    P_before_2 = fit_predict_v1(train_ds, test_ds)
    P_after_2 = fit_predict_candidate(train_ds, test_ds)
    before_same = P_before.tobytes() == P_before_2.tobytes()
    after_same = P_after.tobytes() == P_after_2.tobytes()
    print(f"  BEFORE byte-identical repeat : {before_same}  "
          f"sha256={hashlib.sha256(P_before.tobytes()).hexdigest()[:40]}...")
    print(f"  AFTER  byte-identical repeat : {after_same}  "
          f"sha256={hashlib.sha256(P_after.tobytes()).hexdigest()[:40]}...")
    print(f"  fixture_id alignment identical: "
          f"{np.array_equal(fixture_ids, test_ds.metadata['fixture_id'].to_numpy())}")
    if not (before_same and after_same):
        stop("Determinism failed on the final test.")

    rule("16. INTEGRITY / FORBIDDEN-OPERATION AUDIT")
    after_sums = {rel: md5(REPO / rel) for rel in pins}
    drifted = [rel for rel in pins if before_sums[rel] != after_sums[rel]]
    print(f"  13 pinned baselines after the run: mismatches={drifted or 'NONE'}")
    if drifted:
        stop(f"The final test altered pinned files: {drifted}")
    for rel in ("src/models/robustness.py", "src/models/ablation.py",
                "src/models/candidate_contract.py", "src/models/calibration.py"):
        print(f"    {md5(REPO / rel)}  {rel}")
    print(f"  MODEL_VERSION: {MODEL_VERSION!r}")
    serialized = [str(p.relative_to(REPO)) for pat in ("*.pkl", "*.joblib", "*.pickle", "*.onnx")
                  for p in REPO.rglob(pat) if "__pycache__" not in p.parts]
    print(f"  serialized estimators on disk: {serialized or 'NONE'}")
    for line in (
        "tuning during/after this test : NONE -- C values frozen; any other C is rejected",
        "calibration                   : NONE",
        "feature selection             : NONE",
        "architecture change           : NONE",
        "V2 created                    : NONE",
        "features.db regenerated       : NO",
        "frozen V1 source modified     : NO",
        "model artifact persisted      : NONE",
        "artifact/database written     : NONE",
        "file created by this script   : NONE",
        "2025/26 used before fitting   : NO -- outcomes read only after both arms were fitted",
    ):
        print(f"  {line}")

    rule("END OF FINAL TEST -- no further tuning is permitted on this result")


if __name__ == "__main__":
    main()
