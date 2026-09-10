"""Phase 5A: dropped-V1-feature validation experiment.

Single question, nothing else:
    Does adding `league_home_advantage_season` (the one V1 feature that
    Model B omits) to Model B improve or worsen it under the existing
    frozen walk-forward validation protocol?

This exists to give Phase 5 Condition 1 an evidence base. It is NOT a
feature search, NOT model selection, NOT final-test evaluation, and NOT
V2 creation.

Structural guarantees (each asserted in tests/test_phase5a_dropped_feature.py):
  - `splits.final_split` is never imported or called.
  - `FINAL_TEST_SEASONS` / `FINAL_TRAIN_SEASONS` are never imported or
    referenced. 2025/26 is unreachable from this module.
  - The only split source is `iter_walk_forward_folds` (folds 1-3, each
    temporally re-verified).
  - Both arms use V1's exact estimator and V1's exact preprocessing,
    unmodified: the ONLY difference between them is the one column.
  - Write-once: every output path is refused if it already exists, and
    every path is `phase5a_*`-scoped, so no prior artifact can be
    overwritten.

No hyperparameter is varied. No calibration. No class weighting. No
MODEL_VERSION change. No V2 artifact.

Usage (only when explicitly authorized):
    python -m models.run_dropped_feature
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
from .config import APPROVED_FEATURE_COLUMNS_V1, KNOWN_UNLABELED_FIXTURE_IDS
from .data import SupervisedDataset, load_supervised_dataset
from .evaluate import encode_labels, evaluate, validate_probabilities
from .splits import iter_walk_forward_folds, verify_temporal_safety

DEFAULT_FEATURES_DB = Path("data/processed/features.db")
DEFAULT_RESULTS_PATH = Path("data/audit/phase5a_dropped_feature_comparison.json")
DEFAULT_PREDICTIONS_DIR = Path("data/audit/phase5a_predictions")

# The single feature under test -- V1's, omitted by Model B.
DROPPED_FEATURE = "league_home_advantage_season"

ARM_B = "model_b"
ARM_B_PLUS = "model_b_plus_home_advantage"

MODEL_B_ARM_COLUMNS: tuple[str, ...] = tuple(MODEL_B_COLUMNS)
MODEL_B_PLUS_COLUMNS: tuple[str, ...] = MODEL_B_ARM_COLUMNS + (DROPPED_FEATURE,)

ARMS: dict[str, tuple[str, ...]] = {
    ARM_B: MODEL_B_ARM_COLUMNS,
    ARM_B_PLUS: MODEL_B_PLUS_COLUMNS,
}

METRICS = ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy")
LOWER_IS_BETTER = {"log_loss", "brier"}

# Frozen estimator -- V1's literal values. Not swept, not tuned.
ESTIMATOR_KWARGS = {"max_iter": 2000, "C": 1.0, "random_state": 0}


class ExperimentInvalid(RuntimeError):
    """Raised on any validity-check failure. The experiment must be
    marked INVALID rather than reinterpreted."""


def verify_arm_definitions() -> dict[str, Any]:
    """Mechanically proves the two arms differ by exactly the one
    feature. Raises rather than silently correcting the lists."""
    b, p = set(MODEL_B_ARM_COLUMNS), set(MODEL_B_PLUS_COLUMNS)
    problems = []
    if sorted(p - b) != [DROPPED_FEATURE]:
        problems.append(f"B_PLUS \\ B == {sorted(p - b)}, expected ['{DROPPED_FEATURE}']")
    if b - p:
        problems.append(f"B \\ B_PLUS == {sorted(b - p)}, expected empty")
    if len(b & p) != 80:
        problems.append(f"intersection == {len(b & p)}, expected 80")
    if len(MODEL_B_ARM_COLUMNS) != 80 or len(MODEL_B_PLUS_COLUMNS) != 81:
        problems.append(f"arm sizes {len(MODEL_B_ARM_COLUMNS)}/{len(MODEL_B_PLUS_COLUMNS)}, expected 80/81")
    if len(set(MODEL_B_PLUS_COLUMNS)) != len(MODEL_B_PLUS_COLUMNS):
        problems.append("duplicate column in B_PLUS")
    if DROPPED_FEATURE not in APPROVED_FEATURE_COLUMNS_V1:
        problems.append(f"{DROPPED_FEATURE} is not a V1 feature")
    if problems:
        raise ExperimentInvalid("Arm definition invalid (STOP): " + "; ".join(problems))
    return {
        "arm_b_n_columns": len(MODEL_B_ARM_COLUMNS),
        "arm_b_plus_n_columns": len(MODEL_B_PLUS_COLUMNS),
        "b_plus_minus_b": sorted(p - b),
        "b_minus_b_plus": sorted(b - p),
        "intersection_size": len(b & p),
        "verified": True,
    }


def _refuse_if_exists(path: Path) -> None:
    if path.exists():
        raise ExperimentInvalid(
            f"Refusing to overwrite existing artifact {path}. Phase 5A writes once."
        )


def _predictions_frame(arm: str, fold_name: str, metadata: pd.DataFrame, y_true, P: np.ndarray) -> pd.DataFrame:
    validate_probabilities(P)
    return pd.DataFrame({
        "arm": arm,
        "fold": fold_name,
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


def run_arm_on_fold(arm: str, fold, train_ds: SupervisedDataset, val_ds: SupervisedDataset,
                    save_dir: Path | None) -> tuple[dict[str, Any], pd.DataFrame]:
    """Trains one arm on one fold's training partition and evaluates it
    on that fold's validation partition. Uses V1's unmodified training
    path, so preprocessing is fit on the training partition only."""
    from . import train as train_module

    columns = ARMS[arm]
    train_X = select_feature_columns(train_ds.X, columns, group_name=arm)
    val_X = select_feature_columns(val_ds.X, columns, group_name=arm)

    _, _, P = train_module.train_logistic_regression(train_X, train_ds.y, val_X)
    validate_probabilities(P)
    result = evaluate(val_ds.y, P)

    record = {
        "arm": arm,
        "fold_name": fold.name,
        "train_seasons": list(fold.train_seasons),
        "validation_seasons": list(fold.validation_seasons),
        "n_features": len(columns),
        "n_train": len(train_ds),
        "n_validation": len(val_ds),
        "estimator_kwargs": dict(ESTIMATOR_KWARGS),
        "calibration": "none",
        "probability_validity": "valid_3class_finite_nonnegative_sum_to_1",
        **result.as_dict(),
        "per_class": _per_class_metrics(val_ds.y, P),
    }
    frame = _predictions_frame(arm, fold.name, val_ds.metadata, val_ds.y, P)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{arm}_{fold.name}.csv"
        _refuse_if_exists(out)
        frame.to_csv(out, index=False)
    return record, frame


def run_fold_validity_checks(fold, train_ds, val_ds, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    fb, fp = frames[ARM_B], frames[ARM_B_PLUS]
    checks = {
        "identical_validation_fixture_sets": bool(set(fb.fixture_id) == set(fp.fixture_id)),
        "fixture_order_aligned": bool((fb.fixture_id.values == fp.fixture_id.values).all()),
        "no_duplicate_fixtures": bool(fb.fixture_id.duplicated().sum() == 0 and fp.fixture_id.duplicated().sum() == 0),
        "no_missing_fixtures": bool(set(fb.fixture_id) == set(val_ds.metadata.fixture_id)),
        "identical_y_true": bool((fb.y_true.values == fp.y_true.values).all()),
        "class_order_correct": CLASS_ORDER == ["H", "D", "A"],
        "no_train_validation_overlap": bool(
            set(val_ds.metadata.fixture_id).isdisjoint(set(train_ds.metadata.fixture_id))),
        "abandoned_fixture_absent": bool(
            not set(KNOWN_UNLABELED_FIXTURE_IDS) & set(fb.fixture_id)
            and not set(KNOWN_UNLABELED_FIXTURE_IDS) & set(train_ds.metadata.fixture_id)),
        "feature_counts_80_and_81": bool(len(ARMS[ARM_B]) == 80 and len(ARMS[ARM_B_PLUS]) == 81),
        "no_2025_26_in_fold": "2025/2026" not in set(fold.train_seasons) | set(fold.validation_seasons),
    }
    try:
        verify_temporal_safety(train_ds, val_ds)
        checks["temporal_safety_verified"] = True
    except Exception as exc:  # pragma: no cover
        checks["temporal_safety_verified"] = f"FAILED: {exc}"

    failures = [k for k, v in checks.items() if v is not True]
    checks["all_passed"] = not failures
    checks["failures"] = failures
    return checks


def compute_fold_deltas(b_rec: dict[str, Any], p_rec: dict[str, Any]) -> dict[str, Any]:
    """delta = B_PLUS - B for every metric, with an explicit
    better/worse reading per metric direction."""
    out = {}
    for m in METRICS:
        delta = p_rec[m] - b_rec[m]
        out[m] = {
            "model_b": b_rec[m], "model_b_plus": p_rec[m], "delta_plus_minus_b": delta,
            "b_plus_better": bool(delta < 0) if m in LOWER_IS_BETTER else bool(delta > 0),
            "lower_is_better": m in LOWER_IS_BETTER,
        }
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean across the three folds per arm. Requires all 3 folds -- will
    not average a partial set."""
    summary = {}
    for arm in ARMS:
        cells = [r for r in records if r["arm"] == arm]
        if len(cells) != 3:
            raise ExperimentInvalid(f"{arm} has {len(cells)} fold results, expected 3")
        entry = {"n_folds": len(cells)}
        for m in METRICS:
            vals = [c[m] for c in cells]
            entry[f"mean_{m}"] = float(np.mean(vals))
            entry[f"std_{m}"] = float(np.std(vals, ddof=0))
            entry[f"by_fold_{m}"] = {c["fold_name"]: c[m] for c in cells}
        summary[arm] = entry
    return summary


def classify_result(summary: dict[str, Any]) -> dict[str, Any]:
    """Applies the pre-registered CASE A/B/C/D rule and maps it to one of
    the three permitted conclusions. Invents no epsilon and makes no
    significance or causality claim.
    """
    b, p = summary[ARM_B], summary[ARM_B_PLUS]
    mean_delta = p["mean_log_loss"] - b["mean_log_loss"]
    per_fold = {
        f: p["by_fold_log_loss"][f] - b["by_fold_log_loss"][f]
        for f in b["by_fold_log_loss"]
    }
    n_folds_plus_better = sum(1 for d in per_fold.values() if d < 0)
    consistent = n_folds_plus_better in (0, 3)

    if mean_delta < 0 and n_folds_plus_better == 3:
        case = "CASE_A_improves_and_consistent"
        conclusion = "RETAIN_OMISSION_NOT_SUPPORTED"
        rationale = ("Adding the feature improved mean validation log loss and did so in all three "
                     "folds. The evidence does not support keeping it omitted.")
    elif mean_delta > 0:
        case = "CASE_B_worsens_mean_log_loss"
        conclusion = "RETAIN_OMISSION_SUPPORTED"
        rationale = ("Adding the feature worsened mean validation log loss. The evidence supports "
                     "keeping it omitted.")
    elif mean_delta < 0 and not consistent:
        case = "CASE_C_mean_improves_but_inconsistent_across_folds"
        conclusion = "FEATURE_EFFECT_INCONCLUSIVE"
        rationale = ("Mean validation log loss improved, but the direction was not consistent "
                     "across all three folds, so the effect is not established.")
    else:
        case = "CASE_D_essentially_tied"
        conclusion = "FEATURE_EFFECT_INCONCLUSIVE"
        rationale = "No directional difference in mean validation log loss."

    secondary = {}
    for m in METRICS:
        if m == "log_loss":
            continue
        d = p[f"mean_{m}"] - b[f"mean_{m}"]
        secondary[m] = {"mean_delta": d,
                        "b_plus_better": bool(d < 0) if m in LOWER_IS_BETTER else bool(d > 0)}

    return {
        "case": case,
        "conclusion": conclusion,
        "rationale": rationale,
        "mean_log_loss_delta_plus_minus_b": mean_delta,
        "per_fold_log_loss_delta": per_fold,
        "n_folds_where_b_plus_better": n_folds_plus_better,
        "direction_consistent_across_folds": consistent,
        "secondary_metric_mean_deltas": secondary,
        "interpretation_guard": (
            "Validation-only evidence. No epsilon threshold was defined, so no delta may be "
            "called meaningful or significant. No statistical test was performed. No causal "
            "claim. This does not select a production model, does not create V2, and does not "
            "authorize any change to MODEL_VERSION."
        ),
    }


def main(features_db: Path = DEFAULT_FEATURES_DB,
         results_path: Path = DEFAULT_RESULTS_PATH,
         predictions_dir: Path | None = DEFAULT_PREDICTIONS_DIR) -> dict[str, Any]:
    import sklearn

    _refuse_if_exists(results_path)
    arm_verification = verify_arm_definitions()

    dataset = load_supervised_dataset(features_db)

    records: list[dict[str, Any]] = []
    fold_checks: dict[str, Any] = {}
    fold_deltas: dict[str, Any] = {}

    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        frames = {}
        per_arm = {}
        for arm in ARMS:
            rec, frame = run_arm_on_fold(arm, fold, train_ds, val_ds, predictions_dir)
            records.append(rec)
            frames[arm] = frame
            per_arm[arm] = rec
        checks = run_fold_validity_checks(fold, train_ds, val_ds, frames)
        fold_checks[fold.name] = checks
        if not checks["all_passed"]:
            raise ExperimentInvalid(f"{fold.name} validity checks failed: {checks['failures']}")
        fold_deltas[fold.name] = compute_fold_deltas(per_arm[ARM_B], per_arm[ARM_B_PLUS])

    summary = summarize(records)
    classification = classify_result(summary)

    output = {
        "phase": "5A_dropped_feature_validation",
        "question": (
            f"Does adding V1's omitted feature '{DROPPED_FEATURE}' to Model B improve or worsen "
            "it under the frozen walk-forward validation protocol?"
        ),
        "scope": {
            "validation_only": True,
            "final_test_used": False,
            "note": "2025/26 is never loaded, evaluated, or referenced by this experiment.",
        },
        "environment": {
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "estimator_kwargs": dict(ESTIMATOR_KWARGS),
        "calibration": "none",
        "class_order": list(CLASS_ORDER),
        "arm_verification": arm_verification,
        "arm_columns": {arm: list(cols) for arm, cols in ARMS.items()},
        "n_total_labeled_rows": len(dataset),
        "fold_records": records,
        "fold_validity_checks": fold_checks,
        "fold_deltas": fold_deltas,
        "summary_by_arm": summary,
        "classification": classification,
        "model_version_unchanged": "v1.0",
        "creates_v2": False,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    return output


if __name__ == "__main__":
    r = main()
    c = r["classification"]
    print(f"Phase 5A results written to: {DEFAULT_RESULTS_PATH}")
    print(f"  mean log loss  B      = {r['summary_by_arm'][ARM_B]['mean_log_loss']:.15f}")
    print(f"  mean log loss  B_PLUS = {r['summary_by_arm'][ARM_B_PLUS]['mean_log_loss']:.15f}")
    print(f"  delta (B_PLUS - B)    = {c['mean_log_loss_delta_plus_minus_b']:+.15f}")
    print(f"  folds where B_PLUS better: {c['n_folds_where_b_plus_better']}/3")
    print(f"  CASE: {c['case']}")
    print(f"  CONCLUSION: {c['conclusion']}")
