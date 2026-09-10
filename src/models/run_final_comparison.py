"""Phase 4C: pre-registered V1 vs Model B final-test comparison.

Executes EXACTLY the protocol in docs/PHASE4C_V1_REPLACEMENT_DECISION_PROTOCOL.md
under the authorized frozen decisions:
  - Candidate C = 1.0 (Option C-1: V1's own value; the feature set is
    the only difference between the two arms).
  - No epsilon tie-threshold authorized -> this module computes deltas
    and reports their sign/magnitude but NEVER labels a delta
    "meaningful"; zone assignment defers to human review exactly as the
    protocol specifies.

This runs ONCE. It trains each arm once on FINAL_TRAIN_SEASONS and
evaluates once on FINAL_TEST_SEASONS, aligned per-fixture.

Write-once safety: every output path is refused if a file already
exists, so an authorized single execution can never silently overwrite
a prior result (protocol §12 "artifact mismatch"). It writes only to
declared Phase 4C paths and touches no existing artifact.

No V2 is created. MODEL_VERSION is not read for modification, never
written, and no production path is altered.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ablation import MODEL_B_COLUMNS, select_feature_columns
from .baselines import CLASS_ORDER
from .config import (
    APPROVED_FEATURE_COLUMNS_V1,
    FINAL_TEST_SEASONS,
    FINAL_TRAIN_SEASONS,
    KNOWN_UNLABELED_FIXTURE_IDS,
    MODEL_VERSION,
    REQUIRED_FEATURE_VERSION,
)
from .data import SupervisedDataset, load_supervised_dataset
from .evaluate import encode_labels, evaluate, validate_probabilities
from .splits import final_split, verify_temporal_safety

DEFAULT_FEATURES_DB = Path("data/processed/features.db")
DEFAULT_RESULTS_PATH = Path("data/audit/phase4c_final_comparison.json")
DEFAULT_MANIFEST_PATH = Path("data/audit/phase4c_prerun_manifest.json")
DEFAULT_PREDICTIONS_DIR = Path("data/audit/phase4c_predictions")

ARM_V1 = "v1_logistic_regression"
ARM_B = "model_b_logistic_regression"

# Authorized frozen decisions (Phase 4C authorization).
AUTHORIZED_C = 1.0
EPSILON_AUTHORIZED = None  # explicitly NOT authorized

# Locked V1 final-test metrics (Phase 3). Used ONLY for the reproduction
# check -- never to select, tune, or gate the candidate.
V1_LOCKED_FINAL_TEST = {
    "log_loss": 1.012643607907126,
    "brier": 0.6059500599589396,
    "accuracy": 0.500856653340948,
    "macro_f1": 0.3712753915810083,
    "balanced_accuracy": 0.4321797560315324,
    "n": 1751,
}

# Files whose byte-state must be unchanged before execution (protocol §11).
LOCKED_INPUTS: dict[str, str] = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "data/audit/phase3_model_comparison.json": "616279914b2730749d52eae15b5f96b9",
    "data/audit/phase3_calibration_comparison.json": "d94ed430ab13337797f0592bf82878cb",
    "data/audit/phase4a_ablation_comparison.json": "075b0686bce20bfce9f7289fa37076c0",
    "data/audit/phase4b_robustness_comparison.json": "97d0f8b8674c9bf8598b6c3d6b7c825c",
    "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
    "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
    "src/models/splits.py": "8b7991ab3739c7d4daa2bf1998163da4",
    "src/models/evaluate.py": "4e9d9313867d47a19001a383a301c2fe",
    "src/models/data.py": "b78e30eb45dbc46621c0160a188ce981",
    "src/models/baselines.py": "42e64e3a0ba8c4bf8cf264f80cdcd208",
    "src/models/run_experiments.py": "546ea0105ca8b235ab80b393a58f2831",
}


class ProtocolViolation(RuntimeError):
    """Raised on any protocol §12 stop condition. Deliberately fatal:
    a violated comparison must be discarded, never reinterpreted."""


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def verify_locked_inputs() -> dict[str, Any]:
    """Protocol §11/§12: every locked input must be byte-identical."""
    results, failures = {}, []
    for rel, expected in LOCKED_INPUTS.items():
        actual = _md5(Path(rel))
        ok = actual == expected
        results[rel] = {"expected": expected, "actual": actual, "match": ok}
        if not ok:
            failures.append(rel)
    if failures:
        raise ProtocolViolation(f"Locked input checksum mismatch (STOP): {failures}")
    return results


def _refuse_if_exists(path: Path) -> None:
    if path.exists():
        raise ProtocolViolation(
            f"Refusing to overwrite existing artifact {path}. The final test runs ONCE; "
            "a pre-existing artifact means this is a rerun, which the protocol forbids."
        )


def write_prerun_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Protocol §4 item 13 / §10: capture the locked state BEFORE the
    test set is opened, so pre-registration is evidenced."""
    import sklearn

    _refuse_if_exists(manifest_path)
    manifest = {
        "phase": "4C_final_comparison_prerun_manifest",
        "authorized_decisions": {
            "C": AUTHORIZED_C,
            "epsilon_tie_threshold": EPSILON_AUTHORIZED,
            "epsilon_authorized": False,
            "protocol": "docs/PHASE4C_V1_REPLACEMENT_DECISION_PROTOCOL.md",
        },
        "environment": {
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "frozen_spec": {
            "v1_feature_columns": list(APPROVED_FEATURE_COLUMNS_V1),
            "v1_n_features": len(APPROVED_FEATURE_COLUMNS_V1),
            "model_b_feature_columns": list(MODEL_B_COLUMNS),
            "model_b_n_features": len(MODEL_B_COLUMNS),
            "estimator": "LogisticRegression",
            "estimator_kwargs": {"max_iter": 2000, "C": AUTHORIZED_C, "random_state": 0},
            "preprocessing": "LogisticRegressionPreprocessor (median impute + standardize + one-hot competition_id), fit on FINAL_TRAIN_SEASONS only",
            "calibration": "none (raw predict_proba)",
            "class_order": list(CLASS_ORDER),
            "final_train_seasons": list(FINAL_TRAIN_SEASONS),
            "final_test_seasons": list(FINAL_TEST_SEASONS),
            "model_version_unchanged": MODEL_VERSION,
            "required_feature_version": REQUIRED_FEATURE_VERSION,
        },
        "locked_input_checksums": verify_locked_inputs(),
        "v1_locked_final_test_metrics": V1_LOCKED_FINAL_TEST,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def _predictions_frame(arm: str, metadata: pd.DataFrame, y_true, P: np.ndarray) -> pd.DataFrame:
    validate_probabilities(P)
    return pd.DataFrame({
        "arm": arm,
        "fixture_id": pd.Series(metadata["fixture_id"]).values,
        "season_id": pd.Series(metadata["season_id"]).values,
        "y_true": pd.Series(y_true).values,
        "y_pred": [CLASS_ORDER[i] for i in P.argmax(axis=1)],
        "p_home": P[:, CLASS_ORDER.index("H")],
        "p_draw": P[:, CLASS_ORDER.index("D")],
        "p_away": P[:, CLASS_ORDER.index("A")],
    })


def _per_class_metrics(y_true, P: np.ndarray) -> dict[str, Any]:
    y_idx = encode_labels(y_true, CLASS_ORDER)
    pred = P.argmax(axis=1)
    out = {}
    for i, cls in enumerate(CLASS_ORDER):
        tp = int(((pred == i) & (y_idx == i)).sum())
        fp = int(((pred == i) & (y_idx != i)).sum())
        fn = int(((pred != i) & (y_idx == i)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[cls] = {"tp": tp, "fp": fp, "fn": fn, "support": int((y_idx == i).sum()),
                    "precision": precision, "recall": recall, "f1": f1}
    return out


def run_arm(arm: str, columns, train_ds: SupervisedDataset, test_ds: SupervisedDataset,
            save_dir: Path | None) -> tuple[dict[str, Any], pd.DataFrame]:
    """Trains one arm ONCE on FINAL_TRAIN_SEASONS, evaluates on FINAL_TEST_SEASONS."""
    from . import train as train_module

    train_X = select_feature_columns(train_ds.X, tuple(columns), group_name=arm)
    test_X = select_feature_columns(test_ds.X, tuple(columns), group_name=arm)

    # Identical V1 path for BOTH arms; C=1.0 is V1's own literal, so
    # train_logistic_regression is used unmodified for each.
    _, _, P = train_module.train_logistic_regression(train_X, train_ds.y, test_X)
    validate_probabilities(P)
    result = evaluate(test_ds.y, P)

    record = {
        "arm": arm,
        "n_features": len(columns),
        "feature_columns": list(columns),
        "estimator_kwargs": {"max_iter": 2000, "C": AUTHORIZED_C, "random_state": 0},
        "calibration": "none",
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "probability_validity": "valid_3class_finite_nonnegative_sum_to_1",
        **result.as_dict(),
        "per_class": _per_class_metrics(test_ds.y, P),
    }
    frame = _predictions_frame(arm, test_ds.metadata, test_ds.y, P)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{arm}_final_test.csv"
        _refuse_if_exists(out)
        frame.to_csv(out, index=False)
    return record, frame


def run_validity_checks(train_ds, test_ds, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Protocol §5 mandatory checks. Any failure is a stop condition."""
    checks: dict[str, Any] = {}
    v1f, bf = frames[ARM_V1], frames[ARM_B]

    checks["n_test_equals_1751_both"] = bool(len(v1f) == 1751 and len(bf) == 1751)
    checks["identical_fixture_sets"] = bool(set(v1f.fixture_id) == set(bf.fixture_id))
    checks["fixture_order_aligned"] = bool((v1f.fixture_id.values == bf.fixture_id.values).all())
    checks["no_duplicate_fixtures"] = bool(
        v1f.fixture_id.duplicated().sum() == 0 and bf.fixture_id.duplicated().sum() == 0)
    checks["no_missing_rows"] = bool(set(v1f.fixture_id) == set(test_ds.metadata.fixture_id))
    checks["abandoned_fixture_absent"] = bool(
        not set(KNOWN_UNLABELED_FIXTURE_IDS) & set(v1f.fixture_id)
        and not set(KNOWN_UNLABELED_FIXTURE_IDS) & set(train_ds.metadata.fixture_id))
    checks["no_train_test_overlap"] = bool(
        set(test_ds.metadata.fixture_id).isdisjoint(set(train_ds.metadata.fixture_id)))
    checks["identical_y_true_both_arms"] = bool((v1f.y_true.values == bf.y_true.values).all())
    checks["class_order_correct"] = CLASS_ORDER == ["H", "D", "A"]

    try:
        verify_temporal_safety(train_ds, test_ds)
        checks["temporal_safety_verified"] = True
    except Exception as exc:  # pragma: no cover
        checks["temporal_safety_verified"] = f"FAILED: {exc}"

    failures = [k for k, v in checks.items() if v is not True]
    checks["all_passed"] = not failures
    checks["failures"] = failures
    return checks


def verify_v1_reproduction(v1_record: dict[str, Any], tol: float = 1e-9) -> dict[str, Any]:
    """Protocol §5: the V1 arm must reproduce the locked Phase 3 metrics.
    Failure means the harness differs from V1 -> comparison invalid."""
    checks = {}
    all_ok = True
    for k, expected in V1_LOCKED_FINAL_TEST.items():
        actual = v1_record[k] if k != "n" else v1_record["n"]
        ok = abs(actual - expected) <= tol
        checks[k] = {"locked": expected, "reproduced": actual,
                     "abs_difference": abs(actual - expected), "match": ok}
        all_ok = all_ok and ok
    return {"tolerance": tol, "all_match": all_ok, "checks": checks}


def compute_paired_deltas(v1_record: dict[str, Any], b_record: dict[str, Any]) -> dict[str, Any]:
    """B - V1 for every metric. Sign and magnitude only -- no epsilon was
    authorized, so this deliberately makes NO 'meaningful' judgment."""
    lower_better = {"log_loss", "brier"}
    out = {}
    for m in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy"):
        delta = b_record[m] - v1_record[m]
        out[m] = {
            "v1": v1_record[m], "model_b": b_record[m], "delta_b_minus_v1": delta,
            "b_numerically_better": bool(delta < 0) if m in lower_better else bool(delta > 0),
            "lower_is_better": m in lower_better,
        }
    out["_note"] = (
        "Deltas are numerical only. No epsilon tie-threshold was authorized, so no delta "
        "may be described as 'meaningful' or 'significant'; zone assignment defers to the "
        "protocol's human-review boundary."
    )
    return out


def main(features_db: Path = DEFAULT_FEATURES_DB,
         results_path: Path = DEFAULT_RESULTS_PATH,
         manifest_path: Path = DEFAULT_MANIFEST_PATH,
         predictions_dir: Path | None = DEFAULT_PREDICTIONS_DIR) -> dict[str, Any]:
    import sklearn

    _refuse_if_exists(results_path)          # write-once
    manifest = write_prerun_manifest(manifest_path)   # also re-verifies locked inputs

    dataset = load_supervised_dataset(features_db)
    train_ds, test_ds = final_split(dataset)   # FINAL_TEST_SEASONS opened HERE, after the gate

    v1_record, v1_frame = run_arm(ARM_V1, APPROVED_FEATURE_COLUMNS_V1, train_ds, test_ds, predictions_dir)
    b_record, b_frame = run_arm(ARM_B, MODEL_B_COLUMNS, train_ds, test_ds, predictions_dir)

    frames = {ARM_V1: v1_frame, ARM_B: b_frame}
    validity = run_validity_checks(train_ds, test_ds, frames)
    reproduction = verify_v1_reproduction(v1_record)
    deltas = compute_paired_deltas(v1_record, b_record)

    class_dist = test_ds.y.value_counts().to_dict()

    output = {
        "phase": "4C_v1_vs_model_b_final_comparison",
        "protocol": "docs/PHASE4C_V1_REPLACEMENT_DECISION_PROTOCOL.md",
        "authorized_decisions": manifest["authorized_decisions"],
        "environment": manifest["environment"],
        "sklearn_version": sklearn.__version__,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "test_class_distribution": {k: int(v) for k, v in class_dist.items()},
        "arms": {ARM_V1: v1_record, ARM_B: b_record},
        "paired_deltas_b_minus_v1": deltas,
        "v1_reproduction_check": reproduction,
        "validity_checks": validity,
        "decision_zone": "PENDING_HUMAN_REVIEW",
        "decision_note": (
            "No zone is assigned automatically by this runner. Zone assignment follows the "
            "pre-registered protocol §6/§8 and, with no epsilon authorized, requires explicit "
            "human review. Model B is NOT promoted, MODEL_VERSION is unchanged, no V2 created."
        ),
        "model_version_unchanged": MODEL_VERSION,
    }

    if not reproduction["all_match"]:
        output["decision_zone"] = "E_PROTOCOL_VIOLATION_V1_REPRODUCTION_FAILED"
    if not validity["all_passed"]:
        output["decision_zone"] = "E_PROTOCOL_VIOLATION_VALIDITY_CHECK_FAILED"

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    return output


if __name__ == "__main__":
    r = main()
    print(f"Phase 4C comparison written to: {DEFAULT_RESULTS_PATH}")
    print(f"V1 reproduction: {'PASS' if r['v1_reproduction_check']['all_match'] else 'FAIL'}")
    print(f"Validity checks: {'PASS' if r['validity_checks']['all_passed'] else 'FAIL'}")
    d = r["paired_deltas_b_minus_v1"]["log_loss"]
    print(f"log loss  V1={d['v1']:.12f}  B={d['model_b']:.12f}  delta={d['delta_b_minus_v1']:+.12f}")
    print(f"Decision zone: {r['decision_zone']}")
