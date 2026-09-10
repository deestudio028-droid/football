"""Step 3 experiment runner: baselines + real candidate models across
the 3 approved walk-forward folds, then (only after fold-based model
selection is frozen) a single evaluation against the untouched final
test season (2025/26).

Module-level imports deliberately do NOT require scikit-learn -- only
`run_walk_forward_experiment`, `run_final_test_evaluation`, and `main`
touch `train.py` (imported lazily, inside the function body), so this
module can still be imported and its selection/summary logic tested in
an environment without scikit-learn installed. Actually fitting a
candidate model without scikit-learn present raises the same explicit
ImportError as `train.py` itself -- there is no fallback here either.

Usage (from project root, with PYTHONPATH=src and scikit-learn installed):
    python -m models.run_experiments
    python -m models.run_experiments --skip-final-test
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import CLASS_ORDER, FrequencyBaseline, MajorityClassBaseline
from .config import APPROVED_FEATURE_COLUMNS_V1
from .data import SupervisedDataset, load_supervised_dataset
from .evaluate import evaluate, validate_probabilities
from .splits import final_split, iter_walk_forward_folds

DEFAULT_FEATURES_DB = Path("data/processed/features.db")
DEFAULT_RESULTS_PATH = Path("data/audit/phase3_model_comparison.json")
DEFAULT_PREDICTIONS_DIR = Path("data/audit/phase3_predictions")

MODEL_FREQUENCY = "frequency_baseline"
MODEL_MAJORITY = "majority_class_baseline"
MODEL_LOGREG = "logistic_regression"
MODEL_HGB = "hist_gradient_boosting"
ALL_MODELS = (MODEL_FREQUENCY, MODEL_MAJORITY, MODEL_LOGREG, MODEL_HGB)
REAL_CANDIDATE_MODELS = (MODEL_LOGREG, MODEL_HGB)  # models eligible to be "selected"


def _missingness_report(X: pd.DataFrame) -> dict[str, int]:
    """Per-column count of NaN in a feature matrix. Computed separately
    for whatever single partition is passed in (train OR val, never
    combined) -- this is purely descriptive/audit output and never
    influences which rows or columns are used.
    """
    return {col: int(X[col].isna().sum()) for col in X.columns if X[col].isna().any()}


def _select_approved_features(X: pd.DataFrame) -> pd.DataFrame:
    """Restricts a feature matrix to exactly APPROVED_FEATURE_COLUMNS_V1
    (the 14 approved model features + competition_id), per the Step 2
    design report and the Step 3 diagnostic finding that the full
    264-column `feature_rows` table includes xG feature groups that are
    structurally all-null in early walk-forward training folds.

    Fails loudly if any approved column is missing from `X` -- this
    must never silently fall back to a smaller or different feature set.
    Never invents, renames, or substitutes a column.
    """
    missing = [c for c in APPROVED_FEATURE_COLUMNS_V1 if c not in X.columns]
    if missing:
        raise RuntimeError(
            f"APPROVED_FEATURE_COLUMNS_V1 references column(s) not present in the "
            f"feature matrix: {missing}. Refusing to silently substitute or drop them."
        )
    return X[list(APPROVED_FEATURE_COLUMNS_V1)]


@dataclass
class FoldResult:
    fold_name: str
    train_seasons: tuple[str, ...]
    validation_seasons: tuple[str, ...]
    n_train: int
    n_val: int
    train_missing_by_column: dict[str, int]
    val_missing_by_column: dict[str, int]
    model_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_name": self.fold_name,
            "train_seasons": list(self.train_seasons),
            "validation_seasons": list(self.validation_seasons),
            "n_train": self.n_train,
            "n_val": self.n_val,
            "train_missing_by_column": self.train_missing_by_column,
            "val_missing_by_column": self.val_missing_by_column,
            "model_metrics": self.model_metrics,
        }


def _baseline_predict(baseline, y_train: pd.Series, n_rows: int) -> np.ndarray:
    baseline.fit(y_train)
    return baseline.predict_proba(n_rows)


def _evaluate_and_record(y_val, P: np.ndarray, n_train: int, dropped_imputed: dict | None = None) -> dict[str, Any]:
    validate_probabilities(P)  # hard-fails here, per item 6/leakage-adjacent safety, before any metric is trusted
    result = evaluate(y_val, P)
    d = result.as_dict()
    d["n_train"] = n_train
    d["dropped_or_imputed"] = dropped_imputed or {}
    return d


def _predictions_frame(metadata: pd.DataFrame, y_true: pd.Series, probas: dict[str, np.ndarray]) -> pd.DataFrame:
    data: dict[str, Any] = {
        "fixture_id": metadata["fixture_id"].values,
        "y_true": y_true.values,
    }
    for model_name, P in probas.items():
        for i, cls in enumerate(CLASS_ORDER):
            data[f"{model_name}_P_{cls}"] = P[:, i]
    return pd.DataFrame(data)


def run_walk_forward_experiment(
    dataset: SupervisedDataset, save_predictions_dir: Path | None = None
) -> list[FoldResult]:
    """Runs Frequency baseline, Majority-class baseline, LogisticRegression,
    and HistGradientBoosting across the 3 approved walk-forward folds
    ONLY (see config.WALK_FORWARD_FOLDS / splits.iter_walk_forward_folds).
    2025/26 never appears here -- iter_walk_forward_folds structurally
    excludes it and re-verifies temporal safety on every fold it yields.
    """
    from . import train as train_module  # lazy: only required once a fold is actually run

    fold_results: list[FoldResult] = []

    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        # Scope to the approved 15-column feature set BEFORE anything
        # touches a model -- see _select_approved_features. The full
        # train_ds.X / val_ds.X (264 columns) is never passed to a
        # candidate model; only used below for the audit-facing
        # missingness report, computed on the same approved subset.
        train_X = _select_approved_features(train_ds.X)
        val_X = _select_approved_features(val_ds.X)

        train_missing = _missingness_report(train_X)
        val_missing = _missingness_report(val_X)

        fr = FoldResult(
            fold_name=fold.name,
            train_seasons=fold.train_seasons,
            validation_seasons=fold.validation_seasons,
            n_train=len(train_ds),
            n_val=len(val_ds),
            train_missing_by_column=train_missing,
            val_missing_by_column=val_missing,
        )

        freq_P = _baseline_predict(FrequencyBaseline(), train_ds.y, len(val_ds))
        fr.model_metrics[MODEL_FREQUENCY] = _evaluate_and_record(val_ds.y, freq_P, len(train_ds))

        maj_P = _baseline_predict(MajorityClassBaseline(), train_ds.y, len(val_ds))
        fr.model_metrics[MODEL_MAJORITY] = _evaluate_and_record(val_ds.y, maj_P, len(train_ds))

        # LogisticRegression: preprocessing (impute+scale+one-hot) fit
        # strictly on train_X inside train_logistic_regression -- see
        # src/models/train.py::LogisticRegressionPreprocessor.
        _, _, lr_P = train_module.train_logistic_regression(train_X, train_ds.y, val_X)
        fr.model_metrics[MODEL_LOGREG] = _evaluate_and_record(
            val_ds.y, lr_P, len(train_ds), dropped_imputed=train_missing
        )

        # HistGradientBoosting: native NaN handling, nothing imputed;
        # competition_id re-coded to a dense [0, n_categories) range by
        # train_hist_gradient_boosting's internal CompetitionCodeEncoder
        # (fit on train_X only).
        _, _, hgb_P = train_module.train_hist_gradient_boosting(train_X, train_ds.y, val_X)
        fr.model_metrics[MODEL_HGB] = _evaluate_and_record(val_ds.y, hgb_P, len(train_ds), dropped_imputed={})

        if save_predictions_dir is not None:
            save_predictions_dir.mkdir(parents=True, exist_ok=True)
            preds_df = _predictions_frame(
                val_ds.metadata, val_ds.y,
                {"frequency": freq_P, "majority": maj_P, "logreg": lr_P, "hgb": hgb_P},
            )
            preds_df.to_csv(save_predictions_dir / f"{fold.name}_predictions.csv", index=False)

        fold_results.append(fr)

    return fold_results


def summarize_across_folds(fold_results: list[FoldResult]) -> dict[str, dict[str, dict[str, Any]]]:
    """Mean/std of every scalar metric, per model, across whatever folds
    are passed in. Takes only already-computed FoldResult objects --
    has no way to reach back into raw data, so it cannot be a leakage
    vector regardless of which folds a caller passes.
    """
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    metric_names = ["log_loss", "brier", "macro_f1", "balanced_accuracy", "accuracy"]

    for model_name in ALL_MODELS:
        summary[model_name] = {}
        for metric in metric_names:
            values = [fr.model_metrics[model_name][metric] for fr in fold_results]
            summary[model_name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "values_by_fold": values,
            }
    return summary


def select_model(
    summary: dict[str, dict[str, dict[str, Any]]], log_loss_tie_margin: float = 0.005
) -> dict[str, Any]:
    """Model-selection decision. Takes ONLY the fold 1-3 summary dict --
    by construction it has no parameter through which final-test (2025/26)
    data could reach it, so this function cannot leak the final test set
    into the decision no matter what a caller does.

    PRIMARY rule: lowest mean validation log loss among the two real
    candidates. Brier score is the tie-breaker when the two candidates'
    mean log loss is within `log_loss_tie_margin` of each other.
    Baselines are reported for comparison but are never selectable.
    """
    log_losses = {m: summary[m]["log_loss"]["mean"] for m in REAL_CANDIDATE_MODELS}
    ordered = sorted(log_losses, key=log_losses.get)
    best, runner_up = ordered[0], ordered[1]

    tie = abs(log_losses[best] - log_losses[runner_up]) <= log_loss_tie_margin
    if tie:
        briers = {m: summary[m]["brier"]["mean"] for m in REAL_CANDIDATE_MODELS}
        best = min(briers, key=briers.get)

    beats_frequency_baseline = {
        m: summary[m]["log_loss"]["mean"] < summary[MODEL_FREQUENCY]["log_loss"]["mean"]
        for m in REAL_CANDIDATE_MODELS
    }

    return {
        "selected_model": best,
        "selection_rule": (
            "lowest mean validation log loss across folds 1-3; Brier score is "
            f"tie-breaker when candidates are within {log_loss_tie_margin} log loss of each other"
        ),
        "candidate_mean_log_loss": log_losses,
        "tie_break_applied": tie,
        "beats_frequency_baseline": beats_frequency_baseline,
    }


def run_final_test_evaluation(
    dataset: SupervisedDataset, save_predictions_dir: Path | None = None
) -> dict[str, Any]:
    """Runs all 4 models ONCE against the untouched final test split
    (train on 2020/21-2024/25, test on 2025/26 only). Callers must only
    invoke this AFTER model selection has already been frozen via
    run_walk_forward_experiment()/select_model() -- this function has no
    parameter or return value that could feed back into that decision,
    and `main()` below always calls it strictly after selection.
    """
    from . import train as train_module

    train_ds, test_ds = final_split(dataset)

    # Same approved 15-column scoping as the walk-forward folds -- see
    # _select_approved_features. Applied here too so the final-test
    # evaluation uses exactly the same feature set the selection
    # decision was made on.
    train_X = _select_approved_features(train_ds.X)
    test_X = _select_approved_features(test_ds.X)

    train_missing = _missingness_report(train_X)
    test_missing = _missingness_report(test_X)

    results: dict[str, Any] = {
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "train_missing_by_column": train_missing,
        "test_missing_by_column": test_missing,
        "model_metrics": {},
    }

    freq_P = _baseline_predict(FrequencyBaseline(), train_ds.y, len(test_ds))
    results["model_metrics"][MODEL_FREQUENCY] = _evaluate_and_record(test_ds.y, freq_P, len(train_ds))

    maj_P = _baseline_predict(MajorityClassBaseline(), train_ds.y, len(test_ds))
    results["model_metrics"][MODEL_MAJORITY] = _evaluate_and_record(test_ds.y, maj_P, len(train_ds))

    _, _, lr_P = train_module.train_logistic_regression(train_X, train_ds.y, test_X)
    results["model_metrics"][MODEL_LOGREG] = _evaluate_and_record(
        test_ds.y, lr_P, len(train_ds), dropped_imputed=train_missing
    )

    _, _, hgb_P = train_module.train_hist_gradient_boosting(train_X, train_ds.y, test_X)
    results["model_metrics"][MODEL_HGB] = _evaluate_and_record(test_ds.y, hgb_P, len(train_ds), dropped_imputed={})

    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)
        preds_df = _predictions_frame(
            test_ds.metadata, test_ds.y,
            {"frequency": freq_P, "majority": maj_P, "logreg": lr_P, "hgb": hgb_P},
        )
        preds_df.to_csv(save_predictions_dir / "final_test_predictions.csv", index=False)

    return results


def main(
    features_db: Path = DEFAULT_FEATURES_DB,
    results_path: Path = DEFAULT_RESULTS_PATH,
    predictions_dir: Path | None = DEFAULT_PREDICTIONS_DIR,
    run_final_test: bool = True,
) -> dict[str, Any]:
    import sklearn  # local import: fails loudly here, before any work happens, if absent

    dataset = load_supervised_dataset(features_db)

    # --- Selection phase: folds 1-3 ONLY -----------------------------
    fold_results = run_walk_forward_experiment(dataset, save_predictions_dir=predictions_dir)
    summary = summarize_across_folds(fold_results)
    selection = select_model(summary)

    output: dict[str, Any] = {
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "n_total_labeled_rows": len(dataset),
        "feature_columns": list(dataset.X.columns),
        "folds": [fr.to_dict() for fr in fold_results],
        "summary_across_folds": summary,
        "model_selection": selection,
        "final_test": None,
    }

    # --- Final-test phase: only ever reached after `selection` above is
    # already computed and written into `output`; nothing past this
    # point can change `selection`. ------------------------------------
    if run_final_test:
        output["final_test"] = run_final_test_evaluation(dataset, save_predictions_dir=predictions_dir)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument(
        "--skip-final-test", action="store_true",
        help="Only run folds 1-3 (model selection); skip the 2025/26 evaluation.",
    )
    args = parser.parse_args()

    result = main(
        features_db=args.features_db,
        results_path=args.results_path,
        predictions_dir=args.predictions_dir,
        run_final_test=not args.skip_final_test,
    )
    print(f"Selected model: {result['model_selection']['selected_model']}")
    print(f"Results written to: {args.results_path}")
