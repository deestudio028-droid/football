"""Phase 4A feature-ablation experiment runner (Models A/B/C/D).

Separate from `run_experiments.py` (Phase 3) by design: this writes
only to Phase 4A artifact paths and never touches, reads-for-update, or
overwrites any Phase 3 artifact, the V1 contract, or the databases.

What this runs:
  - Models A and B: all 3 approved walk-forward folds (the PRIMARY
    comparison -- directly comparable to each other on a 3-fold mean).
  - Models C and D: ONLY folds whose training partition has usable xG
    coverage (the xG FEASIBILITY comparison). Folds that fail the
    guard are recorded explicitly as INVALID_FOR_XG with the reason,
    never silently skipped.

What this deliberately does NOT do:
  - It does not compute a 3-fold mean for a model with any invalid
    fold. `summarize_model()` returns `mean_log_loss: None` plus an
    explicit `mean_not_computable_reason` in that case, so a fabricated
    C/D "3-fold average" cannot appear in any artifact or report.
  - It does not touch 2025/26. There is no final-test code path in this
    module at all -- `splits.final_split` is never imported.
  - It does not select a production model, tune hyperparameters, or
    calibrate. The LogisticRegression configuration is exactly V1's
    (`train.train_logistic_regression`), unchanged, so any metric
    difference between tiers is attributable to the feature set alone.

Usage (from project root, PYTHONPATH=src, scikit-learn installed):
    python -m models.run_ablation
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ablation import (
    ABLATION_MODELS,
    MODEL_COLUMNS,
    MODEL_GROUPS,
    XG_CORE_COLUMNS,
    XG_INCLUSIVE_MODELS,
    XgCoverageError,
    assess_xg_fold_eligibility,
    require_xg_fold_valid,
    select_feature_columns,
)
from .baselines import CLASS_ORDER
from .data import SupervisedDataset, load_supervised_dataset
from .evaluate import evaluate, validate_probabilities
from .splits import iter_walk_forward_folds

DEFAULT_FEATURES_DB = Path("data/processed/features.db")
DEFAULT_RESULTS_PATH = Path("data/audit/phase4a_ablation_comparison.json")
DEFAULT_PREDICTIONS_DIR = Path("data/audit/phase4a_predictions")

# V1's locked reference result, quoted for context in the artifact only.
# Never used to select, tune, or gate anything in this module.
V1_LOCKED_FINAL_TEST = {
    "log_loss": 1.012643607907126,
    "brier": 0.6059500599589396,
    "accuracy": 0.500856653340948,
    "note": "Phase 3 V1 locked 2025/26 final-test result, quoted for reference only.",
}


def _predictions_frame(metadata: pd.DataFrame, y_true, P: np.ndarray) -> pd.DataFrame:
    validate_probabilities(P)
    return pd.DataFrame({
        "fixture_id": pd.Series(metadata["fixture_id"]).values,
        "season_id": pd.Series(metadata["season_id"]).values,
        "y_true": pd.Series(y_true).values,
        "y_pred": [CLASS_ORDER[i] for i in P.argmax(axis=1)],
        "p_home": P[:, CLASS_ORDER.index("H")],
        "p_draw": P[:, CLASS_ORDER.index("D")],
        "p_away": P[:, CLASS_ORDER.index("A")],
    })


def run_model_on_fold(
    model_name: str,
    fold,
    train_ds: SupervisedDataset,
    val_ds: SupervisedDataset,
    save_predictions_dir: Path | None = None,
) -> dict[str, Any]:
    """Trains one ablation tier on one fold and returns a fully
    self-describing result record (including the exact feature columns
    used and the xG eligibility status), or -- for an xG-inclusive
    model on a fold with unusable xG training coverage -- an explicit
    invalid record with the reason, and no training performed.
    """
    from . import train as train_module  # lazy: scikit-learn only needed once we actually train

    columns = MODEL_COLUMNS[model_name]
    record: dict[str, Any] = {
        "model": model_name,
        "feature_groups": list(MODEL_GROUPS[model_name]),
        "feature_columns": list(columns),
        "n_feature_columns": len(columns),
        "fold_name": fold.name,
        "train_seasons": list(fold.train_seasons),
        "validation_seasons": list(fold.validation_seasons),
        "n_train": len(train_ds),
        "n_validation": len(val_ds),
    }

    train_X = select_feature_columns(train_ds.X, columns, group_name=model_name)
    val_X = select_feature_columns(val_ds.X, columns, group_name=model_name)

    is_xg_inclusive = model_name in XG_INCLUSIVE_MODELS
    record["xg_inclusive_model"] = is_xg_inclusive

    if is_xg_inclusive:
        eligibility = assess_xg_fold_eligibility(fold.name, train_X, XG_CORE_COLUMNS)
        record.update(eligibility.to_dict())
        record["fold_name"] = fold.name  # to_dict also sets this; keep it authoritative
        if not eligibility.is_valid:
            # Recorded explicitly, NOT silently skipped. No model is fit.
            record.update({
                "fold_valid": False,
                "status": "INVALID_FOR_XG",
                "log_loss": None, "brier": None, "macro_f1": None,
                "balanced_accuracy": None, "accuracy": None, "confusion_matrix": None,
                "probability_validity": "not_applicable_no_model_trained",
            })
            return record
    else:
        record.update({
            "xg_training_observed": None,
            "is_valid_for_xg": None,
            "invalid_reason": None,
        })

    # Belt-and-braces: for an xG-inclusive model this raises rather than
    # letting any fit() happen on an ineligible fold, even if the check
    # above were ever bypassed.
    if is_xg_inclusive:
        try:
            require_xg_fold_valid(fold.name, train_X, XG_CORE_COLUMNS)
        except XgCoverageError as exc:
            record.update({
                "fold_valid": False, "status": "INVALID_FOR_XG", "invalid_reason": str(exc),
                "log_loss": None, "brier": None, "macro_f1": None,
                "balanced_accuracy": None, "accuracy": None, "confusion_matrix": None,
                "probability_validity": "not_applicable_no_model_trained",
            })
            return record

    # Identical LogisticRegression configuration to V1 -- fold-scoped
    # preprocessing (median impute + standardize + one-hot competition_id),
    # all fit on this fold's training partition only.
    _, _, P_val = train_module.train_logistic_regression(train_X, train_ds.y, val_X)
    validate_probabilities(P_val)
    result = evaluate(val_ds.y, P_val)

    record.update({
        "fold_valid": True,
        "status": "OK",
        "probability_validity": "valid_3class_finite_nonnegative_sum_to_1",
        **result.as_dict(),
    })

    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)
        frame = _predictions_frame(val_ds.metadata, val_ds.y, P_val)
        frame.to_csv(save_predictions_dir / f"{model_name}_{fold.name}.csv", index=False)

    return record


def summarize_model(model_name: str, fold_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates a model's per-fold results.

    Refuses to produce a mean if ANY fold is invalid -- returning
    `None` for every mean plus an explicit reason, so no artifact or
    report can present a partial-coverage C/D score as if it were
    directly comparable to A/B's genuine 3-fold mean.
    """
    valid = [r for r in fold_records if r.get("fold_valid")]
    invalid = [r for r in fold_records if not r.get("fold_valid")]

    summary: dict[str, Any] = {
        "model": model_name,
        "feature_groups": list(MODEL_GROUPS[model_name]),
        "n_feature_columns": len(MODEL_COLUMNS[model_name]),
        "n_folds_attempted": len(fold_records),
        "n_folds_valid": len(valid),
        "n_folds_invalid": len(invalid),
        "invalid_folds": [
            {"fold_name": r["fold_name"], "reason": r.get("invalid_reason")} for r in invalid
        ],
        "valid_fold_names": [r["fold_name"] for r in valid],
    }

    metrics = ["log_loss", "brier", "macro_f1", "balanced_accuracy", "accuracy"]

    if invalid:
        summary["mean_across_all_three_folds_computable"] = False
        summary["mean_not_computable_reason"] = (
            f"{len(invalid)} of {len(fold_records)} folds are invalid for this model "
            f"({', '.join(r['fold_name'] for r in invalid)}). A 3-fold mean would be fabricated. "
            f"Per-valid-fold results are reported individually instead and must NOT be ranked "
            f"against models with a genuine 3-fold mean."
        )
        for m in metrics:
            summary[f"mean_{m}"] = None
        summary["per_valid_fold"] = {
            m: {r["fold_name"]: r[m] for r in valid} for m in metrics
        }
        return summary

    summary["mean_across_all_three_folds_computable"] = True
    summary["mean_not_computable_reason"] = None
    for m in metrics:
        values = [r[m] for r in valid]
        summary[f"mean_{m}"] = float(np.mean(values))
        summary[f"std_{m}"] = float(np.std(values, ddof=0))
        summary[f"values_by_fold_{m}"] = {r["fold_name"]: r[m] for r in valid}
    return summary


def run_ablation_experiment(
    dataset: SupervisedDataset, save_predictions_dir: Path | None = None
) -> dict[str, Any]:
    """Runs all four tiers across the 3 approved walk-forward folds.
    2025/26 never appears -- `iter_walk_forward_folds` structurally
    excludes it and re-verifies temporal safety on every fold it yields.
    """
    fold_records: list[dict[str, Any]] = []
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        for model_name in ABLATION_MODELS:
            fold_records.append(
                run_model_on_fold(model_name, fold, train_ds, val_ds, save_predictions_dir)
            )

    summaries = {
        model_name: summarize_model(
            model_name, [r for r in fold_records if r["model"] == model_name]
        )
        for model_name in ABLATION_MODELS
    }
    return {"fold_records": fold_records, "model_summaries": summaries}


def main(
    features_db: Path = DEFAULT_FEATURES_DB,
    results_path: Path = DEFAULT_RESULTS_PATH,
    predictions_dir: Path | None = DEFAULT_PREDICTIONS_DIR,
) -> dict[str, Any]:
    import sklearn  # local import: fails loudly here, before any work, if absent

    dataset = load_supervised_dataset(features_db)
    experiment = run_ablation_experiment(dataset, save_predictions_dir=predictions_dir)

    output = {
        "phase": "4A_feature_ablation",
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "n_total_labeled_rows": len(dataset),
        "model_algorithm": "LogisticRegression (identical configuration to Phase 3 V1; no tuning)",
        "primary_comparison": {
            "models": ["model_a", "model_b"],
            "basis": "mean across all 3 approved walk-forward validation folds",
        },
        "xg_feasibility_comparison": {
            "models": ["model_c", "model_d"],
            "basis": "coverage-valid folds only; NO 3-fold mean is computed or comparable",
            "note": (
                "Models C/D cannot be ranked against A/B on the 3-fold mean: folds 1-2 have "
                "zero observed xG values in their training partitions, making them structurally "
                "invalid for xG-inclusive training."
            ),
        },
        "v1_locked_reference": V1_LOCKED_FINAL_TEST,
        "fold_records": experiment["fold_records"],
        "model_summaries": experiment["model_summaries"],
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    return output


if __name__ == "__main__":
    result = main()
    print(f"Phase 4A ablation written to: {DEFAULT_RESULTS_PATH}")
    for name, s in result["model_summaries"].items():
        if s["mean_across_all_three_folds_computable"]:
            print(f"  {name}: mean log loss {s['mean_log_loss']:.6f} ({s['n_folds_valid']}/3 folds)")
        else:
            print(f"  {name}: NO 3-fold mean ({s['n_folds_valid']}/3 valid) -- {s['valid_fold_names']}")
