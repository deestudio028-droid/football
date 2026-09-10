"""Phase 3 Step 4: probability diagnostics + calibration for the
already-selected LogisticRegression model.

Frozen inputs from Step 3 (docs/PHASE3_STEP3_MODEL_COMPARISON_REPORT.md,
data/audit/phase3_model_comparison.json) that this module NEVER changes:
  - Selected model: LogisticRegression (config.APPROVED_FEATURE_COLUMNS_V1,
    train.train_logistic_regression / train.LogisticRegressionPreprocessor).
  - Selection rule: lowest mean validation log loss across walk-forward
    folds 1-3.
  - Final 2025/26 test result: log loss 1.012643607907126, Brier
    0.6059500599589396, accuracy 0.500856653340948 -- LOCKED. This
    module reports pre-/post-calibration numbers against that same
    frozen test split, once, at the very end (Step 7), and never lets
    that result influence any earlier decision.

This module does NOT retrain, retune, reselect the model, add features,
or touch xG/shots. It only:
  1. obtains out-of-sample validation predictions for each approved
     walk-forward fold from the frozen LogisticRegression configuration
     (Step 1),
  2. computes descriptive probability diagnostics and reliability data
     from those predictions (Steps 2-3),
  3. fits/compares optional calibration layers using ONLY that
     out-of-sample data, with a nested, temporally-respecting
     evaluation scheme (Steps 4-5),
  4. and applies whatever was decided, exactly once, to the frozen
     2025/26 test split (Step 7).

Calibration methods (Platt/sigmoid scaling and isotonic regression) are
implemented here in pure NumPy rather than via sklearn.calibration --
deliberately, for the same reason evaluate.py's metrics are pure NumPy:
these are small, well-defined, standard algorithms (1D Newton-Raphson
logistic regression for Platt scaling; the Pool-Adjacent-Violators
Algorithm for isotonic regression) that don't need scikit-learn's
dependency to be correct, deterministic, and independently testable.
This is NOT a workaround for the candidate models themselves -- the
model that produced the probabilities being calibrated is, and remains,
the real, sklearn-trained LogisticRegression from train.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import CLASS_ORDER
from .config import WALK_FORWARD_FOLDS
from .data import SupervisedDataset, load_supervised_dataset
from .evaluate import encode_labels, evaluate, validate_probabilities
from .run_experiments import MODEL_LOGREG, _select_approved_features
from .splits import final_split, iter_walk_forward_folds

DEFAULT_FEATURES_DB = Path("data/processed/features.db")
DEFAULT_PREDICTIONS_DIR = Path("data/audit/phase3_predictions")
DEFAULT_DIAGNOSTICS_PATH = Path("data/audit/phase3_probability_diagnostics.json")
DEFAULT_CALIBRATION_COMPARISON_PATH = Path("data/audit/phase3_calibration_comparison.json")

SELECTED_MODEL = MODEL_LOGREG  # frozen per Step 3; never changed by this module

CLASS_TO_PROB_COLUMN = {"H": "p_home", "D": "p_draw", "A": "p_away"}
_WALK_FORWARD_FOLD_NAMES = [f.name for f in WALK_FORWARD_FOLDS]


# =======================================================================
# STEP 1 -- validation prediction artifacts
# =======================================================================
def _predictions_to_frame(fold_name: str, metadata: pd.DataFrame, y_true, P: np.ndarray) -> pd.DataFrame:
    validate_probabilities(P)
    y_pred_idx = P.argmax(axis=1)
    y_pred = [CLASS_ORDER[i] for i in y_pred_idx]
    return pd.DataFrame({
        "fold_name": fold_name,
        "fixture_id": pd.Series(metadata["fixture_id"]).values,
        "season_id": pd.Series(metadata["season_id"]).values,
        "y_true": pd.Series(y_true).values,
        "y_pred": y_pred,
        "p_home": P[:, CLASS_ORDER.index("H")],
        "p_draw": P[:, CLASS_ORDER.index("D")],
        "p_away": P[:, CLASS_ORDER.index("A")],
        "max_probability": P.max(axis=1),
    })


def generate_fold_validation_predictions(
    dataset: SupervisedDataset, save_dir: Path | None = None
) -> dict[str, pd.DataFrame]:
    """The reusable mechanism (Step 1): retrains LogisticRegression on
    each approved walk-forward fold's OWN training period only (the
    frozen Step 3 configuration -- approved 15-column feature set,
    LogisticRegressionPreprocessor, unchanged), and records that fold's
    out-of-sample validation predictions. Never trains on, or predicts
    against, 2025/26. Requires scikit-learn (via train.py).
    """
    from . import train as train_module

    fold_frames: dict[str, pd.DataFrame] = {}
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        train_X = _select_approved_features(train_ds.X)
        val_X = _select_approved_features(val_ds.X)
        _, _, P = train_module.train_logistic_regression(train_X, train_ds.y, val_X)
        frame = _predictions_to_frame(fold.name, val_ds.metadata, val_ds.y, P)
        fold_frames[fold.name] = frame

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(save_dir / f"step4_{fold.name}_logreg_predictions.csv", index=False)

    return fold_frames


def load_fold_predictions_from_existing_run(
    dataset: SupervisedDataset, predictions_dir: Path = DEFAULT_PREDICTIONS_DIR
) -> dict[str, pd.DataFrame]:
    """Alternative source for Step 1 data: reshapes the LogisticRegression
    columns already present in the fold_N_predictions.csv files produced
    by run_experiments.run_walk_forward_experiment() (Step 3) into the
    exact Step 1 schema, joining season_id from the read-only feature
    dataset by fixture_id. Does not retrain anything, does not write to
    features.db (read-only lookup only). Raises loudly if a fold's CSV
    is missing or if any fixture_id fails to join a season_id -- never
    silently drops rows.
    """
    meta = dataset.metadata[["fixture_id", "season_id"]]
    fold_frames: dict[str, pd.DataFrame] = {}
    for fold_name in _WALK_FORWARD_FOLD_NAMES:
        path = predictions_dir / f"{fold_name}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- run `python -m models.run_experiments` first, "
                "or use generate_fold_validation_predictions() to retrain directly."
            )
        df = pd.read_csv(path)
        merged = df.merge(meta, on="fixture_id", how="left")
        unmatched = int(merged["season_id"].isna().sum())
        if unmatched:
            raise RuntimeError(f"{fold_name}: {unmatched} fixture_id(s) failed to join a season_id")
        P = merged[["logreg_P_H", "logreg_P_D", "logreg_P_A"]].to_numpy()
        fold_frames[fold_name] = _predictions_to_frame(fold_name, merged, merged["y_true"], P)
    return fold_frames


def load_final_test_predictions_from_existing_run(
    dataset: SupervisedDataset, predictions_dir: Path = DEFAULT_PREDICTIONS_DIR
) -> pd.DataFrame:
    """Same idea as load_fold_predictions_from_existing_run, for the
    already-produced final_test_predictions.csv from Step 3's
    run_final_test_evaluation(). This is 2025/26 data -- callers must
    only use this after calibration method selection is already frozen.
    """
    path = predictions_dir / "final_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m models.run_experiments` first.")
    meta = dataset.metadata[["fixture_id", "season_id"]]
    df = pd.read_csv(path)
    merged = df.merge(meta, on="fixture_id", how="left")
    unmatched = int(merged["season_id"].isna().sum())
    if unmatched:
        raise RuntimeError(f"final_test: {unmatched} fixture_id(s) failed to join a season_id")
    P = merged[["logreg_P_H", "logreg_P_D", "logreg_P_A"]].to_numpy()
    return _predictions_to_frame("final_test", merged, merged["y_true"], P)


def combine_fold_predictions(fold_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenates folds 1-3 validation predictions in fold order
    (fold_1, fold_2, fold_3 -- chronological), never shuffled, never
    mixed with final-test predictions.
    """
    frames = [fold_frames[name] for name in _WALK_FORWARD_FOLD_NAMES if name in fold_frames]
    return pd.concat(frames, ignore_index=True)


# =======================================================================
# STEP 2 -- probability diagnostics (+ ECE) and draw investigation
# =======================================================================
def compute_ece(y, P: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Top-label (confidence) Expected Calibration Error -- the standard
    multiclass ECE definition (Guo et al. 2017, "On Calibration of
    Modern Neural Networks"): bin predictions by their MAX predicted
    probability (the model's confidence in its own argmax choice); in
    each bin, compare mean confidence to observed top-1 accuracy
    (fraction of predictions in that bin whose argmax equals the true
    label). ECE = sum over bins of (bin_size / n) * |mean_confidence -
    accuracy|. Bins are FIXED, equal-width on [0, 1]
    (edges = np.linspace(0, 1, n_bins + 1), n_bins=10 by default --
    width 0.1) -- not equal-frequency. Fully deterministic given
    (y, P, n_bins): no randomness anywhere.
    """
    y = np.asarray(y)
    P = np.asarray(P, dtype=float)
    confidences = P.max(axis=1)
    preds = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    correct = (preds == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == n_bins - 1 else (confidences >= lo) & (confidences < hi)
        bin_n = int(mask.sum())
        if bin_n == 0:
            bins.append({"lower": float(lo), "upper": float(hi), "n": 0,
                         "mean_confidence": None, "accuracy": None, "abs_gap": None})
            continue
        mean_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(mean_conf - acc)
        ece += (bin_n / n) * gap
        bins.append({"lower": float(lo), "upper": float(hi), "n": bin_n,
                     "mean_confidence": mean_conf, "accuracy": acc, "abs_gap": gap})

    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "definition": (
            "Top-label confidence ECE (Guo et al. 2017): equal-width bins on the "
            "model's max predicted probability; per-bin |mean confidence - top-1 accuracy|, "
            "weighted by bin size."
        ),
        "bins": bins,
    }


def compute_probability_diagnostics(combined: pd.DataFrame) -> dict[str, Any]:
    """Diagnostics computed ONLY from folds 1-3 validation predictions
    (whatever is in `combined` -- callers must never pass final-test
    rows here)."""
    y = combined["y_true"].to_numpy()
    P = combined[["p_home", "p_draw", "p_away"]].to_numpy()
    n = len(combined)

    mean_predicted = {"H": float(P[:, 0].mean()), "D": float(P[:, 1].mean()), "A": float(P[:, 2].mean())}
    actual_freq = {c: float((y == c).mean()) for c in CLASS_ORDER}
    mean_max_prob = float(combined["max_probability"].mean())

    argmax_counts_raw = combined["y_pred"].value_counts().to_dict()
    argmax_counts = {c: int(argmax_counts_raw.get(c, 0)) for c in CLASS_ORDER}
    argmax_pct = {c: argmax_counts[c] / n for c in CLASS_ORDER}

    mean_pred_by_actual: dict[str, Any] = {}
    for actual_c in CLASS_ORDER:
        mask = y == actual_c
        mean_pred_by_actual[actual_c] = {
            "H": float(P[mask, 0].mean()) if mask.any() else None,
            "D": float(P[mask, 1].mean()) if mask.any() else None,
            "A": float(P[mask, 2].mean()) if mask.any() else None,
            "n": int(mask.sum()),
        }

    eval_result = evaluate(y, P)
    ece_result = compute_ece(y, P)

    return {
        "n": n,
        "mean_predicted_probability": mean_predicted,
        "actual_class_frequency": actual_freq,
        "mean_max_probability": mean_max_prob,
        "argmax_counts": argmax_counts,
        "argmax_percentage": argmax_pct,
        "mean_predicted_probability_by_actual_class": mean_pred_by_actual,
        "log_loss": eval_result.log_loss,
        "brier": eval_result.brier,
        "ece": ece_result["ece"],
        "ece_definition": ece_result["definition"],
        "ece_bin_table": ece_result["bins"],
    }


def investigate_draw_probability(combined: pd.DataFrame) -> dict[str, Any]:
    """Distinguishes (A) "merely an argmax phenomenon" from (B)
    "systematic underestimation of Draw probability", using folds 1-3
    validation predictions only. Never changes the model. The
    `top_line_heuristic_verdict` is an explicit, stated rule (not a
    black-box judgment) meant as a quick summary -- the reliability-bin
    table for Draw (Step 3) is the rigorous, per-probability-level
    answer and should be read alongside this.
    """
    y = combined["y_true"].to_numpy()
    p_draw = combined["p_draw"].to_numpy()
    P = combined[["p_home", "p_draw", "p_away"]].to_numpy()
    n = len(combined)

    empirical_draw_freq = float((y == "D").mean())
    mean_p_draw_overall = float(p_draw.mean())
    mean_p_draw_when_actual_draw = float(p_draw[y == "D"].mean()) if (y == "D").any() else None
    mean_p_draw_when_not_draw = float(p_draw[y != "D"].mean()) if (y != "D").any() else None

    argmax_draw_count = int((combined["y_pred"] == "D").sum())
    argmax_draw_pct = argmax_draw_count / n

    ranks = (-P).argsort(axis=1)  # column index of rank-0 (highest), rank-1, rank-2 per row
    draw_col = CLASS_ORDER.index("D")
    draw_rank = np.array([np.where(ranks[i] == draw_col)[0][0] for i in range(len(P))])
    pct_draw_ranked_1st = float((draw_rank == 0).mean())
    pct_draw_ranked_2nd = float((draw_rank == 1).mean())
    pct_draw_ranked_3rd = float((draw_rank == 2).mean())

    gap = mean_p_draw_overall - empirical_draw_freq
    if abs(gap) <= 0.03:
        verdict = (
            "A: mean predicted P(draw) is close to the empirical draw rate (within 0.03). "
            "Draw's near-zero argmax share looks like a structural argmax phenomenon (H or A "
            "usually edges it out), not evidence of miscalibration by this top-line check. "
            "See the Draw reliability-bin table for the rigorous per-probability-level answer."
        )
    elif gap < -0.03:
        verdict = (
            "B: mean predicted P(draw) is meaningfully BELOW the empirical draw frequency "
            "(gap < -0.03) -- evidence of systematic underestimation, not merely an argmax artifact."
        )
    else:
        verdict = (
            "Mean predicted P(draw) is meaningfully ABOVE the empirical draw frequency "
            "(gap > 0.03) -- draws are over-, not under-, estimated on this top-line check."
        )

    return {
        "n": n,
        "empirical_draw_frequency": empirical_draw_freq,
        "mean_predicted_p_draw_overall": mean_p_draw_overall,
        "mean_predicted_p_draw_when_actual_draw": mean_p_draw_when_actual_draw,
        "mean_predicted_p_draw_when_actual_not_draw": mean_p_draw_when_not_draw,
        "argmax_draw_count": argmax_draw_count,
        "argmax_draw_percentage": argmax_draw_pct,
        "pct_draw_ranked_1st_of_3": pct_draw_ranked_1st,
        "pct_draw_ranked_2nd_of_3": pct_draw_ranked_2nd,
        "pct_draw_ranked_3rd_of_3": pct_draw_ranked_3rd,
        "mean_predicted_minus_empirical_gap": gap,
        "top_line_heuristic_verdict": verdict,
        "caveat": (
            "This is a descriptive top-line heuristic, not a statistical test, and does not "
            "alter the model regardless of its verdict. Read alongside the Draw reliability-bin table."
        ),
    }


# =======================================================================
# STEP 3 -- reliability bins
# =======================================================================
def compute_reliability_bins(combined: pd.DataFrame, outcome_class: str, n_bins: int = 10) -> list[dict[str, Any]]:
    """Fixed, equal-width binning scheme: edges = np.linspace(0, 1,
    n_bins + 1) (n_bins=10 by default, bin width 0.1) on the predicted
    probability for `outcome_class` (e.g. p_draw for outcome_class="D").
    Per bin: sample count, mean predicted probability, observed
    frequency of `outcome_class` actually occurring, absolute
    calibration gap. Deterministic given (combined, outcome_class,
    n_bins). Never uses 2025/26.
    """
    prob_col = CLASS_TO_PROB_COLUMN[outcome_class]
    p = combined[prob_col].to_numpy()
    y = (combined["y_true"].to_numpy() == outcome_class).astype(float)
    n = len(combined)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        bin_n = int(mask.sum())
        if bin_n == 0:
            bins.append({"lower": float(lo), "upper": float(hi), "n": 0,
                         "mean_predicted_probability": None, "observed_frequency": None,
                         "abs_calibration_gap": None})
            continue
        mean_p = float(p[mask].mean())
        obs_freq = float(y[mask].mean())
        bins.append({
            "lower": float(lo), "upper": float(hi), "n": bin_n,
            "mean_predicted_probability": mean_p, "observed_frequency": obs_freq,
            "abs_calibration_gap": abs(mean_p - obs_freq),
        })
    return bins


# =======================================================================
# STEP 4 -- calibration methods (pure NumPy, no scikit-learn required)
# =======================================================================
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _fit_platt_1d(x: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> tuple[float, float]:
    """Fits (a, b) in sigmoid(a*x + b) to a binary target y via
    Newton-Raphson on the logistic log-likelihood (standard 1D/2-
    parameter logistic regression -- this IS Platt scaling, just
    implemented without sklearn). Deterministic: fixed zero
    initialization, fixed iteration/tolerance schedule, no randomness.
    """
    a, b = 0.0, 0.0
    for _ in range(max_iter):
        z = a * x + b
        p = np.clip(_sigmoid(z), 1e-12, 1 - 1e-12)
        grad_a = np.sum((p - y) * x)
        grad_b = np.sum(p - y)
        w = np.clip(p * (1 - p), 1e-12, None)
        h_aa = np.sum(w * x * x)
        h_ab = np.sum(w * x)
        h_bb = np.sum(w)
        H = np.array([[h_aa, h_ab], [h_ab, h_bb]]) + 1e-9 * np.eye(2)
        g = np.array([grad_a, grad_b])
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        a -= float(delta[0])
        b -= float(delta[1])
        if np.max(np.abs(delta)) < tol:
            break
    return float(a), float(b)


def _pava_weighted(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-Adjacent-Violators Algorithm (weighted): fits a non-
    decreasing step function to (x, y). Duplicate x values are first
    aggregated (mean y, weight = count) so the result is a proper
    function of x. Returns (unique_x_ascending, fitted_y) -- both
    length equal to the number of distinct x values. Deterministic:
    stable sort, no randomness.
    """
    order = np.argsort(x, kind="stable")
    x_sorted, y_sorted = x[order], y[order]

    uniq_x, inverse = np.unique(x_sorted, return_inverse=True)
    sums = np.zeros(len(uniq_x))
    counts = np.zeros(len(uniq_x))
    np.add.at(sums, inverse, y_sorted)
    np.add.at(counts, inverse, 1.0)
    means = sums / counts

    block_val: list[float] = []
    block_weight: list[float] = []
    block_size: list[int] = []
    for m, w in zip(means, counts):
        block_val.append(float(m))
        block_weight.append(float(w))
        block_size.append(1)
        while len(block_val) > 1 and block_val[-2] > block_val[-1]:
            new_w = block_weight[-2] + block_weight[-1]
            new_v = (block_val[-2] * block_weight[-2] + block_val[-1] * block_weight[-1]) / new_w
            new_s = block_size[-2] + block_size[-1]
            block_val.pop(); block_weight.pop(); block_size.pop()
            block_val[-1] = new_v
            block_weight[-1] = new_w
            block_size[-1] = new_s

    y_fit = np.concatenate([np.full(size, val) for val, size in zip(block_val, block_size)])
    return uniq_x, y_fit


@dataclass
class IdentityCalibrator:
    """No-op calibrator representing the uncalibrated baseline, exposed
    through the same fit/transform interface as the real calibrators so
    comparisons are apples-to-apples."""
    name: str = "uncalibrated"

    def fit(self, P: np.ndarray, y) -> "IdentityCalibrator":
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P, dtype=float)


@dataclass
class MulticlassPlattCalibrator:
    """One-vs-rest Platt (sigmoid) scaling per class, fit only on the
    data passed to .fit() -- callers are responsible for ensuring that
    is out-of-sample validation data, never training or final-test
    data (see evaluate_calibration_methods / run_final_frozen_evaluation
    for how that's enforced). Rows are renormalized to sum to 1 after
    the three independent per-class sigmoids are applied.
    """
    name: str = "platt_sigmoid"
    class_order: list = field(default_factory=lambda: list(CLASS_ORDER))
    _params: dict = field(default_factory=dict)
    _fitted: bool = False

    def fit(self, P: np.ndarray, y) -> "MulticlassPlattCalibrator":
        y_idx = encode_labels(y, self.class_order)
        for i, cls in enumerate(self.class_order):
            binary_y = (y_idx == i).astype(float)
            self._params[cls] = _fit_platt_1d(P[:, i], binary_y)
        self._fitted = True
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MulticlassPlattCalibrator.fit() must be called before transform().")
        out = np.zeros_like(P, dtype=float)
        for i, cls in enumerate(self.class_order):
            a, b = self._params[cls]
            out[:, i] = _sigmoid(a * P[:, i] + b)
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 1e-12, 1e-12, row_sums)
        return out / row_sums


@dataclass
class MulticlassIsotonicCalibrator:
    """One-vs-rest isotonic regression per class (via the weighted PAVA
    above), fit only on the data passed to .fit(). New probabilities are
    obtained by linear interpolation between the fitted step function's
    knots (out-of-range values clip to the nearest knot, matching
    standard isotonic-regression convention). Rows are renormalized to
    sum to 1.
    """
    name: str = "isotonic"
    class_order: list = field(default_factory=lambda: list(CLASS_ORDER))
    _knots: dict = field(default_factory=dict)
    _fitted: bool = False

    def fit(self, P: np.ndarray, y) -> "MulticlassIsotonicCalibrator":
        y_idx = encode_labels(y, self.class_order)
        for i, cls in enumerate(self.class_order):
            binary_y = (y_idx == i).astype(float)
            x_knots, y_knots = _pava_weighted(P[:, i], binary_y)
            self._knots[cls] = (x_knots, y_knots)
        self._fitted = True
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MulticlassIsotonicCalibrator.fit() must be called before transform().")
        out = np.zeros_like(P, dtype=float)
        for i, cls in enumerate(self.class_order):
            x_knots, y_knots = self._knots[cls]
            out[:, i] = np.interp(P[:, i], x_knots, y_knots)
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 1e-12, 1e-12, row_sums)
        return out / row_sums


CALIBRATION_METHODS = ("uncalibrated", "platt_sigmoid", "isotonic")


def _make_calibrator(name: str):
    if name == "uncalibrated":
        return IdentityCalibrator()
    if name == "platt_sigmoid":
        return MulticlassPlattCalibrator()
    if name == "isotonic":
        return MulticlassIsotonicCalibrator()
    raise ValueError(f"Unknown calibration method: {name!r}")


def evaluate_calibration_methods(
    fold_frames: dict[str, pd.DataFrame], min_samples_per_class_for_isotonic: int = 50
) -> dict[str, Any]:
    """Nested, expanding-window calibration comparison that respects
    temporal order and never scores a calibrator on the data it was fit
    on:
      - Assessment fold_2: fit on fold_1's validation predictions only,
        score against fold_2's validation predictions.
      - Assessment fold_3: fit on fold_1 + fold_2's validation
        predictions, score against fold_3's validation predictions.
    fold_1 is never an assessment target (nothing chronologically
    precedes it). 2025/26 never appears in this function -- it takes
    only `fold_frames` (folds 1-3), nothing else.
    """
    assessment_results: dict[str, list[dict]] = {m: [] for m in CALIBRATION_METHODS}
    isotonic_skipped_rounds: list[str] = []

    for assess_idx in range(1, len(_WALK_FORWARD_FOLD_NAMES)):
        assess_name = _WALK_FORWARD_FOLD_NAMES[assess_idx]
        fit_names = _WALK_FORWARD_FOLD_NAMES[:assess_idx]
        fit_df = pd.concat([fold_frames[n] for n in fit_names], ignore_index=True)
        assess_df = fold_frames[assess_name]

        P_fit = fit_df[["p_home", "p_draw", "p_away"]].to_numpy()
        y_fit = fit_df["y_true"].to_numpy()
        P_assess = assess_df[["p_home", "p_draw", "p_away"]].to_numpy()
        y_assess = assess_df["y_true"].to_numpy()

        class_counts = fit_df["y_true"].value_counts()
        isotonic_ok = bool((class_counts >= min_samples_per_class_for_isotonic).all())
        if not isotonic_ok:
            isotonic_skipped_rounds.append(assess_name)

        for method in CALIBRATION_METHODS:
            if method == "isotonic" and not isotonic_ok:
                continue
            calibrator = _make_calibrator(method)
            calibrator.fit(P_fit, y_fit)
            P_calibrated = calibrator.transform(P_assess)
            validate_probabilities(P_calibrated)
            result = evaluate(y_assess, P_calibrated)
            ece = compute_ece(y_assess, P_calibrated)["ece"]
            assessment_results[method].append({
                "assessment_fold": assess_name,
                "fit_folds": fit_names,
                "n_fit": len(fit_df),
                "n_assess": len(assess_df),
                "log_loss": result.log_loss,
                "brier": result.brier,
                "ece": ece,
                "macro_f1": result.macro_f1,
                "balanced_accuracy": result.balanced_accuracy,
                "accuracy": result.accuracy,
            })

    summary: dict[str, Any] = {}
    for method, rounds in assessment_results.items():
        if not rounds:
            summary[method] = None
            continue
        summary[method] = {
            "mean_log_loss": float(np.mean([r["log_loss"] for r in rounds])),
            "mean_brier": float(np.mean([r["brier"] for r in rounds])),
            "mean_ece": float(np.mean([r["ece"] for r in rounds])),
            "rounds": rounds,
        }

    return {
        "methodology": (
            "Nested expanding-window calibration comparison: for each assessment fold, "
            "calibrators are fit ONLY on validation predictions from folds strictly "
            "earlier in time, then scored on the assessment fold's own validation "
            "predictions (never the data the calibrator was fit on). fold_1 is never an "
            "assessment target. 2025/26 never appears here."
        ),
        "isotonic_skipped_rounds": isotonic_skipped_rounds,
        "isotonic_min_samples_per_class": min_samples_per_class_for_isotonic,
        "per_method_summary": summary,
    }


# =======================================================================
# STEP 5 -- calibration selection
# =======================================================================
def select_calibration_method(
    comparison: dict[str, Any], log_loss_improvement_margin: float = 0.005
) -> dict[str, Any]:
    """Selects a calibration method ONLY if it beats uncalibrated mean
    log loss (across the nested assessment rounds in `comparison`, which
    only ever covers folds 1-3) by more than `log_loss_improvement_margin`,
    AND does not worsen mean Brier by more than the same margin.
    Otherwise explicitly keeps the uncalibrated LogisticRegression
    probabilities -- calibration is never forced just because this stage
    exists. Takes only the folds-1-3 comparison dict; has no parameter
    through which 2025/26 could reach this decision.
    """
    summary = comparison["per_method_summary"]
    baseline = summary["uncalibrated"]
    candidates = {m: s for m, s in summary.items() if m != "uncalibrated" and s is not None}

    best_method = "uncalibrated"
    best_log_loss = baseline["mean_log_loss"]
    reason = "No candidate calibration method improved mean log loss beyond the margin; remaining uncalibrated."

    for method, s in candidates.items():
        improves_log_loss = (baseline["mean_log_loss"] - s["mean_log_loss"]) > log_loss_improvement_margin
        brier_not_worse = (s["mean_brier"] - baseline["mean_brier"]) <= log_loss_improvement_margin
        if improves_log_loss and brier_not_worse and s["mean_log_loss"] < best_log_loss:
            best_method = method
            best_log_loss = s["mean_log_loss"]
            reason = (
                f"{method} improved mean log loss by "
                f"{baseline['mean_log_loss'] - s['mean_log_loss']:.4f} "
                f"(> margin {log_loss_improvement_margin}) without a comparable Brier regression."
            )

    return {
        "selected_calibration_method": best_method,
        "selection_rule": (
            f"Select a calibrated method only if it improves mean log loss over uncalibrated "
            f"by more than {log_loss_improvement_margin} across the nested assessment rounds "
            f"AND does not worsen mean Brier by more than the same margin. Otherwise remain "
            f"uncalibrated. This decision never sees 2025/26 data."
        ),
        "reason": reason,
        "baseline_mean_log_loss": baseline["mean_log_loss"],
        "baseline_mean_brier": baseline["mean_brier"],
        "candidate_mean_log_loss": {m: s["mean_log_loss"] for m, s in candidates.items()},
        "candidate_mean_brier": {m: s["mean_brier"] for m, s in candidates.items()},
    }


# =======================================================================
# STEP 7 -- final frozen evaluation (2025/26), applied once
# =======================================================================
def run_final_frozen_evaluation(
    dataset: SupervisedDataset,
    fold_frames: dict[str, pd.DataFrame],
    calibration_decision: dict[str, Any],
    final_test_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Applies the FROZEN calibration decision (already made from folds
    1-3 only, before this function is ever called) to the untouched
    2025/26 final test split, exactly once. Never re-decides anything
    based on what it finds here -- there is no code path from this
    function's return value back into select_calibration_method or
    evaluate_calibration_methods.

    `final_test_predictions`, if given, must be uncalibrated
    LogisticRegression probabilities already produced by training on
    2020/21-2024/25 and applying to 2025/26 (e.g.
    load_final_test_predictions_from_existing_run(), sourced from
    run_experiments.run_final_test_evaluation()'s output). If not given,
    this function trains that model itself via train.py, requiring
    scikit-learn.
    """
    if final_test_predictions is None:
        from . import train as train_module
        train_ds, test_ds = final_split(dataset)
        train_X = _select_approved_features(train_ds.X)
        test_X = _select_approved_features(test_ds.X)
        _, _, P_test = train_module.train_logistic_regression(train_X, train_ds.y, test_X)
        final_test_predictions = _predictions_to_frame("final_test", test_ds.metadata, test_ds.y, P_test)

    P_pre = final_test_predictions[["p_home", "p_draw", "p_away"]].to_numpy()
    validate_probabilities(P_pre)
    y_test = final_test_predictions["y_true"].to_numpy()

    pre_result = evaluate(y_test, P_pre)
    pre_ece = compute_ece(y_test, P_pre)["ece"]

    method_name = calibration_decision["selected_calibration_method"]
    post_result_dict = None
    if method_name != "uncalibrated":
        # Refit the FROZEN method type on ALL of folds 1-3 (the complete
        # permitted pre-test calibration information) -- the "production"
        # fit, performed once, only after the method itself was already
        # chosen using the nested folds-1-3 comparison above.
        combined = combine_fold_predictions(fold_frames)
        P_fit_all = combined[["p_home", "p_draw", "p_away"]].to_numpy()
        y_fit_all = combined["y_true"].to_numpy()
        calibrator = _make_calibrator(method_name)
        calibrator.fit(P_fit_all, y_fit_all)
        P_post = calibrator.transform(P_pre)
        validate_probabilities(P_post)
        post_result = evaluate(y_test, P_post)
        post_ece = compute_ece(y_test, P_post)["ece"]
        post_result_dict = {**post_result.as_dict(), "ece": post_ece}

    return {
        "calibration_method_applied": method_name,
        "pre_calibration": {**pre_result.as_dict(), "ece": pre_ece},
        "post_calibration": post_result_dict,
        "note": (
            "Reported strictly AFTER calibration method selection was frozen using folds 1-3 "
            "only. This result never fed back into method selection."
        ),
    }


# =======================================================================
# Orchestration + artifacts (Step 8)
# =======================================================================
def main(
    features_db: Path = DEFAULT_FEATURES_DB,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    diagnostics_path: Path = DEFAULT_DIAGNOSTICS_PATH,
    calibration_comparison_path: Path = DEFAULT_CALIBRATION_COMPARISON_PATH,
    use_existing_run_experiments_predictions: bool = True,
) -> dict[str, Any]:
    dataset = load_supervised_dataset(features_db)

    if use_existing_run_experiments_predictions:
        fold_frames = load_fold_predictions_from_existing_run(dataset, predictions_dir)
        final_test_predictions = load_final_test_predictions_from_existing_run(dataset, predictions_dir)
    else:
        fold_frames = generate_fold_validation_predictions(dataset, save_dir=predictions_dir)
        final_test_predictions = None  # run_final_frozen_evaluation trains it itself (needs sklearn)

    combined = combine_fold_predictions(fold_frames)

    predictions_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(predictions_dir / "step4_validation_predictions_combined.csv", index=False)
    if final_test_predictions is not None:
        final_test_predictions.to_csv(predictions_dir / "step4_final_test_predictions.csv", index=False)

    diagnostics = compute_probability_diagnostics(combined)
    draw_investigation = investigate_draw_probability(combined)
    reliability = {
        "home": compute_reliability_bins(combined, "H"),
        "draw": compute_reliability_bins(combined, "D"),
        "away": compute_reliability_bins(combined, "A"),
    }

    diagnostics_output = {
        "selected_model": SELECTED_MODEL,
        "n_folds": len(fold_frames),
        "diagnostics": diagnostics,
        "draw_probability_investigation": draw_investigation,
        "reliability_bins": reliability,
    }
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics_output, indent=2, default=str), encoding="utf-8")

    comparison = evaluate_calibration_methods(fold_frames)
    selection = select_calibration_method(comparison)
    final_result = run_final_frozen_evaluation(dataset, fold_frames, selection, final_test_predictions)

    calibration_output = {
        "comparison": comparison,
        "selection": selection,
        "final_test": final_result,
    }
    calibration_comparison_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_comparison_path.write_text(json.dumps(calibration_output, indent=2, default=str), encoding="utf-8")

    return {"diagnostics": diagnostics_output, "calibration": calibration_output}


if __name__ == "__main__":
    result = main()
    print(f"Selected calibration method: {result['calibration']['selection']['selected_calibration_method']}")
    print(f"Diagnostics written to: {DEFAULT_DIAGNOSTICS_PATH}")
    print(f"Calibration comparison written to: {DEFAULT_CALIBRATION_COMPARISON_PATH}")
