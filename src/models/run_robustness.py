"""Phase 4B regularization-sensitivity experiment runner.

STATUS: IMPLEMENTED BUT NOT EXECUTED. Phase 4B training has NOT started.
Running this module is a separate, explicit instruction.

Scope: Model A and Model B feature sets (Phase 4A frozen columns) x the
frozen C grid x the 3 approved walk-forward folds. Everything else is
V1 behaviour, unchanged.

Final-test protection (structural, not conventional):
  - `splits.final_split` is never imported and never called.
  - `config.FINAL_TEST_SEASONS` / `FINAL_TRAIN_SEASONS` are never imported.
  - The only split source is `splits.iter_walk_forward_folds`, which
    structurally yields folds 1-3 only and re-verifies temporal safety.
  - No function in this module accepts a final-test parameter.
Each of these is asserted mechanically in tests/test_model_robustness.py.

This module selects nothing for production. It answers one question:
is Model B's Phase 4A advantage robust across reasonable C values?

Usage (ONLY when explicitly instructed):
    python -m models.run_robustness
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ablation import MODEL_A, MODEL_B
from .baselines import CLASS_ORDER
from .data import SupervisedDataset, load_supervised_dataset
from .evaluate import evaluate, validate_probabilities
from .robustness import (
    C_GRID,
    METRICS,
    PHASE4A_C1_MEAN_LOG_LOSS,
    PHASE4B_MODELS,
    V1_C_VALUE,
    V1_LOGREG_BASE_KWARGS,
    select_model_features,
    train_logistic_regression_with_C,
)
from .splits import iter_walk_forward_folds

DEFAULT_FEATURES_DB = Path("data/processed/features.db")
DEFAULT_RESULTS_PATH = Path("data/audit/phase4b_robustness_comparison.json")
DEFAULT_PREDICTIONS_DIR = Path("data/audit/phase4b_predictions")


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


def run_single_cell(
    model_name: str, C: float, fold, train_ds: SupervisedDataset, val_ds: SupervisedDataset,
    save_predictions_dir: Path | None = None,
) -> dict[str, Any]:
    """One (model, C, fold) cell: fit on the fold's training partition
    only, evaluate on its validation partition only."""
    train_X = select_model_features(train_ds.X, model_name)
    val_X = select_model_features(val_ds.X, model_name)

    P_val = train_logistic_regression_with_C(train_X, train_ds.y, val_X, C)
    validate_probabilities(P_val)
    result = evaluate(val_ds.y, P_val)

    record = {
        "model": model_name,
        "C": C,
        "fold_name": fold.name,
        "train_seasons": list(fold.train_seasons),
        "validation_seasons": list(fold.validation_seasons),
        "n_train": len(train_ds),
        "n_validation": len(val_ds),
        "n_feature_columns": len(PHASE4B_MODELS[model_name]),
        "probability_validity": "valid_3class_finite_nonnegative_sum_to_1",
        **result.as_dict(),
    }

    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)
        c_tag = str(C).replace(".", "p")
        frame = _predictions_frame(val_ds.metadata, val_ds.y, P_val)
        frame.to_csv(save_predictions_dir / f"{model_name}_C{c_tag}_{fold.name}.csv", index=False)

    return record


def run_sensitivity_experiment(
    dataset: SupervisedDataset, save_predictions_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Full grid: 2 models x len(C_GRID) x 3 folds. Folds come only from
    iter_walk_forward_folds -- 2025/26 is structurally unreachable."""
    records: list[dict[str, Any]] = []
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        for model_name in (MODEL_A, MODEL_B):
            for C in C_GRID:
                records.append(
                    run_single_cell(model_name, C, fold, train_ds, val_ds, save_predictions_dir)
                )
    return records


def summarize_by_model_and_C(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean/std across the 3 folds for each (model, C). Every cell must
    have all 3 folds present -- if not, this raises rather than
    averaging a partial set (the Phase 4A "no fabricated means" rule).
    """
    summary: dict[str, dict[str, Any]] = {}
    for model_name in (MODEL_A, MODEL_B):
        summary[model_name] = {}
        for C in C_GRID:
            cells = [r for r in records if r["model"] == model_name and r["C"] == C]
            if len(cells) != 3:
                raise RuntimeError(
                    f"{model_name} @ C={C} has {len(cells)} fold results, expected 3. "
                    "Refusing to average an incomplete set."
                )
            entry: dict[str, Any] = {"n_folds": len(cells)}
            for m in METRICS:
                vals = [c[m] for c in cells]
                entry[f"mean_{m}"] = float(np.mean(vals))
                entry[f"std_{m}"] = float(np.std(vals, ddof=0))
                entry[f"by_fold_{m}"] = {c["fold_name"]: c[m] for c in cells}
            summary[model_name][str(C)] = entry
    return summary


def analyze_robustness(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Answers the 10 pre-declared robustness questions from the Phase 4B
    design. Takes ONLY the folds-1-3 summary -- there is no parameter
    through which final-test data could reach this analysis.
    """
    a_ll = {C: summary[MODEL_A][str(C)]["mean_log_loss"] for C in C_GRID}
    b_ll = {C: summary[MODEL_B][str(C)]["mean_log_loss"] for C in C_GRID}

    best_a = min(a_ll, key=a_ll.get)
    best_b = min(b_ll, key=b_ll.get)

    deltas = {C: b_ll[C] - a_ll[C] for C in C_GRID}          # negative => B better
    b_beats_a = {C: deltas[C] < 0 for C in C_GRID}
    n_b_wins = sum(b_beats_a.values())

    # Per-fold win check at each C (question 5)
    per_fold_b_beats_a = {}
    for C in C_GRID:
        a_folds = summary[MODEL_A][str(C)]["by_fold_log_loss"]
        b_folds = summary[MODEL_B][str(C)]["by_fold_log_loss"]
        per_fold_b_beats_a[C] = {f: bool(b_folds[f] < a_folds[f]) for f in a_folds}

    # Leave-one-C-out check (question 7): does the overall picture hinge
    # on any single grid point?
    dependence = {}
    for C in C_GRID:
        others = [c for c in C_GRID if c != C]
        dependence[C] = {
            "n_b_wins_excluding_this_C": sum(1 for c in others if deltas[c] < 0),
            "n_other_C_values": len(others),
        }

    # Directional consistency of secondary metrics (question 9)
    secondary = {}
    for m in ("brier", "macro_f1", "balanced_accuracy", "accuracy"):
        lower_is_better = m == "brier"
        agree = {}
        for C in C_GRID:
            av = summary[MODEL_A][str(C)][f"mean_{m}"]
            bv = summary[MODEL_B][str(C)][f"mean_{m}"]
            agree[C] = bool(bv < av) if lower_is_better else bool(bv > av)
        secondary[m] = {"b_better_at_each_C": agree, "n_C_where_b_better": sum(agree.values())}

    return {
        "q1_best_C_model_a": {"C": best_a, "mean_log_loss": a_ll[best_a]},
        "q2_best_C_model_b": {"C": best_b, "mean_log_loss": b_ll[best_b]},
        "q3_b_beats_a_at_same_C": b_beats_a,
        "q4_n_C_values_where_b_beats_a": n_b_wins,
        "q4_total_C_values": len(C_GRID),
        "q5_per_fold_b_beats_a_by_C": per_fold_b_beats_a,
        "q6_delta_range": {
            "min_delta": float(min(deltas.values())),
            "max_delta": float(max(deltas.values())),
            "deltas_by_C": deltas,
        },
        "q7_leave_one_C_out_dependence": dependence,
        "q8_delta_at_v1_C": {"C": V1_C_VALUE, "delta": deltas[V1_C_VALUE], "b_better": deltas[V1_C_VALUE] < 0},
        "q9_secondary_metric_consistency": secondary,
        "q10_edge_of_grid_flags": {
            "model_a_best_at_grid_edge": best_a in (C_GRID[0], C_GRID[-1]),
            "model_b_best_at_grid_edge": best_b in (C_GRID[0], C_GRID[-1]),
            "note": (
                "A best-C at the grid edge suggests the optimum may lie outside the frozen "
                "grid (over- or under-regularization); this is reported, not acted upon."
            ),
        },
        "interpretation_guard": (
            "Robustness is descriptive only. No statistical test was performed, so no "
            "significance claim may be made. No causal claim. This does not select a "
            "production model and does not authorize changing V1."
        ),
    }


def verify_phase4a_equivalence_at_C1(summary: dict[str, dict[str, Any]], tol: float = 1e-9) -> dict[str, Any]:
    """At C=1.0 this study must reproduce Phase 4A's recorded means. A
    mismatch means the Phase 4B path is not the V1 path and the whole
    comparison is invalid -- reported, never silently tolerated.
    """
    out: dict[str, Any] = {"tolerance": tol, "checks": {}, "all_match": True}
    for model_name, expected in PHASE4A_C1_MEAN_LOG_LOSS.items():
        actual = summary[model_name][str(V1_C_VALUE)]["mean_log_loss"]
        match = abs(actual - expected) <= tol
        out["checks"][model_name] = {
            "phase4a_recorded": expected, "phase4b_at_C1": actual,
            "abs_difference": abs(actual - expected), "match": match,
        }
        out["all_match"] = out["all_match"] and match
    return out


def main(
    features_db: Path = DEFAULT_FEATURES_DB,
    results_path: Path = DEFAULT_RESULTS_PATH,
    predictions_dir: Path | None = DEFAULT_PREDICTIONS_DIR,
) -> dict[str, Any]:
    import sklearn  # local import: fails loudly here, before any work, if absent

    dataset = load_supervised_dataset(features_db)
    records = run_sensitivity_experiment(dataset, save_predictions_dir=predictions_dir)
    summary = summarize_by_model_and_C(records)
    analysis = analyze_robustness(summary)
    equivalence = verify_phase4a_equivalence_at_C1(summary)

    output = {
        "phase": "4B_regularization_sensitivity",
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "n_total_labeled_rows": len(dataset),
        "C_grid": list(C_GRID),
        "v1_C_value": V1_C_VALUE,
        "logreg_base_kwargs": V1_LOGREG_BASE_KWARGS,
        "models_studied": {m: len(cols) for m, cols in PHASE4B_MODELS.items()},
        "feature_columns": {m: list(cols) for m, cols in PHASE4B_MODELS.items()},
        "protocol": {
            "folds": "3 approved walk-forward folds only; 2025/26 never loaded or evaluated",
            "basis": "mean validation log loss across folds 1-3",
            "purpose": "robustness check of the Phase 4A A->B result; NOT model selection",
        },
        "phase4a_equivalence_at_C1": equivalence,
        "cell_records": records,
        "summary_by_model_and_C": summary,
        "robustness_analysis": analysis,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    return output


if __name__ == "__main__":
    result = main()
    print(f"Phase 4B robustness results written to: {DEFAULT_RESULTS_PATH}")
    eq = result["phase4a_equivalence_at_C1"]
    print(f"Phase 4A equivalence at C=1.0: {'PASS' if eq['all_match'] else 'FAIL -- INVESTIGATE'}")
    an = result["robustness_analysis"]
    print(f"B beats A at {an['q4_n_C_values_where_b_beats_a']}/{an['q4_total_C_values']} C values")
    print(f"Best C  -- A: {an['q1_best_C_model_a']['C']}  B: {an['q2_best_C_model_b']['C']}")
