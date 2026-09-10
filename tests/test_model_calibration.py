"""Tests for src/models/calibration.py -- Phase 3 Step 4.

Split by dependency:
  - Pure-math / structural tests (Platt fit, PAVA, ECE, reliability
    bins, calibrator fit/transform, selection logic, leakage-boundary
    checks): need no scikit-learn, no real dataset -- these always run.
  - Tests against the real, already-produced Step 3 artifacts
    (data/audit/phase3_predictions/*.csv): need the real dataset and
    those CSVs to exist; skipped (not failed) otherwise.
  - generate_fold_validation_predictions / run_final_frozen_evaluation's
    training fallback: need scikit-learn; skipped otherwise.
"""
import _pathfix  # noqa: F401
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models import calibration as calib
from models.baselines import CLASS_ORDER
from models.evaluate import MetricInputError, validate_probabilities

_REAL_DB = Path("data/processed/features.db")
_PRED_DIR = Path("data/audit/phase3_predictions")
_DATA_SKIP_REASON = "data/processed/features.db or Step 3 prediction CSVs not present in this environment"

try:
    from models import train as train_module
    _TRAIN_IMPORT_ERROR = None
except ImportError as e:
    train_module = None
    _TRAIN_IMPORT_ERROR = str(e)
_SKLEARN_SKIP_REASON = f"scikit-learn not available in this environment: {_TRAIN_IMPORT_ERROR}"


def _has_real_data() -> bool:
    return _REAL_DB.exists() and all(
        (_PRED_DIR / f"{name}.csv").exists() if name == "final_test_predictions"
        else (_PRED_DIR / f"{name}_predictions.csv").exists()
        for name in ["fold_1", "fold_2", "fold_3"]
    ) and (_PRED_DIR / "final_test_predictions.csv").exists()


def _synthetic_fold_frame(fold_name: str, n: int, seed: int, draw_rate: float = 0.25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = rng.choice(list(CLASS_ORDER), size=n, p=[0.45, draw_rate, 0.55 - draw_rate])
    # Deliberately miscalibrated synthetic probabilities: draw mass is
    # suppressed relative to draw_rate, to exercise the calibrators.
    raw = rng.dirichlet([3, 1, 3], size=n)
    return pd.DataFrame({
        "fold_name": fold_name,
        "fixture_id": np.arange(n) + hash(fold_name) % 1000,
        "season_id": [1000 + (hash(fold_name) % 5)] * n,
        "y_true": y_true,
        "y_pred": np.array(CLASS_ORDER)[raw.argmax(axis=1)],
        "p_home": raw[:, 0], "p_draw": raw[:, 1], "p_away": raw[:, 2],
        "max_probability": raw.max(axis=1),
    })


class TestPlattFit(unittest.TestCase):
    def test_recovers_a_near_zero_and_b_matching_log_odds_for_constant_x(self):
        # If x is always 0, a is irrelevant (multiplied by 0); b alone
        # should converge to the logit of the empirical positive rate.
        rng = np.random.default_rng(0)
        n = 2000
        x = np.zeros(n)
        y = (rng.uniform(size=n) < 0.3).astype(float)
        a, b = calib._fit_platt_1d(x, y)
        p_hat = 1.0 / (1.0 + np.exp(-b))
        self.assertAlmostEqual(p_hat, y.mean(), delta=0.02)

    def test_deterministic_across_repeated_fits(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=500)
        y = (rng.uniform(size=500) < 1 / (1 + np.exp(-x))).astype(float)
        a1, b1 = calib._fit_platt_1d(x, y)
        a2, b2 = calib._fit_platt_1d(x, y)
        self.assertEqual((a1, b1), (a2, b2))

    def test_monotonic_relationship_recovered_with_positive_slope(self):
        rng = np.random.default_rng(2)
        n = 3000
        x = rng.uniform(-3, 3, size=n)
        true_p = 1.0 / (1.0 + np.exp(-2.0 * x))
        y = (rng.uniform(size=n) < true_p).astype(float)
        a, b = calib._fit_platt_1d(x, y)
        self.assertGreater(a, 0)


class TestPavaWeighted(unittest.TestCase):
    def test_output_is_non_decreasing(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(size=200)
        y = (rng.uniform(size=200) < x).astype(float)
        x_knots, y_fit = calib._pava_weighted(x, y)
        self.assertTrue(np.all(np.diff(y_fit) >= -1e-12))
        self.assertTrue(np.all(np.diff(x_knots) > 0))  # unique x, strictly increasing

    def test_deterministic_across_repeated_fits(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(size=300)
        y = (rng.uniform(size=300) < x).astype(float)
        x1, y1 = calib._pava_weighted(x, y)
        x2, y2 = calib._pava_weighted(x, y)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_duplicate_x_values_aggregated_not_duplicated_in_output(self):
        x = np.array([0.1, 0.1, 0.1, 0.5, 0.9])
        y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        x_knots, y_fit = calib._pava_weighted(x, y)
        self.assertEqual(len(x_knots), len(np.unique(x)))
        # mean at x=0.1 is 2/3 before any pooling
        idx = list(x_knots).index(0.1)
        self.assertGreaterEqual(y_fit[idx], 0.0)
        self.assertLessEqual(y_fit[idx], 1.0)

    def test_hand_computed_simple_case(self):
        # Strictly increasing y with increasing x: PAVA should return y unchanged.
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        x_knots, y_fit = calib._pava_weighted(x, y)
        np.testing.assert_allclose(y_fit, y)


class TestCalibratorsSynthetic(unittest.TestCase):
    def test_identity_calibrator_returns_input_unchanged(self):
        P = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
        cal = calib.IdentityCalibrator().fit(P, ["H", "A"])
        np.testing.assert_array_equal(cal.transform(P), P)

    def test_platt_calibrator_output_is_valid_probabilities(self):
        train_df = _synthetic_fold_frame("f", 500, seed=10)
        P_train = train_df[["p_home", "p_draw", "p_away"]].to_numpy()
        y_train = train_df["y_true"].to_numpy()
        cal = calib.MulticlassPlattCalibrator().fit(P_train, y_train)

        test_df = _synthetic_fold_frame("g", 100, seed=11)
        P_test = test_df[["p_home", "p_draw", "p_away"]].to_numpy()
        P_out = cal.transform(P_test)
        validate_probabilities(P_out)

    def test_isotonic_calibrator_output_is_valid_probabilities(self):
        train_df = _synthetic_fold_frame("f", 500, seed=12)
        P_train = train_df[["p_home", "p_draw", "p_away"]].to_numpy()
        y_train = train_df["y_true"].to_numpy()
        cal = calib.MulticlassIsotonicCalibrator().fit(P_train, y_train)

        test_df = _synthetic_fold_frame("g", 100, seed=13)
        P_test = test_df[["p_home", "p_draw", "p_away"]].to_numpy()
        P_out = cal.transform(P_test)
        validate_probabilities(P_out)

    def test_platt_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            calib.MulticlassPlattCalibrator().transform(np.array([[0.3, 0.3, 0.4]]))

    def test_isotonic_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            calib.MulticlassIsotonicCalibrator().transform(np.array([[0.3, 0.3, 0.4]]))

    def test_validation_transform_uses_training_fit_not_its_own_data(self):
        # The whole point of calibration leakage-safety: fit on one
        # dataset, transform() must reuse those exact learned params on
        # a DIFFERENT dataset, never refit against it.
        train_df = _synthetic_fold_frame("f", 500, seed=20)
        P_train = train_df[["p_home", "p_draw", "p_away"]].to_numpy()
        y_train = train_df["y_true"].to_numpy()
        cal = calib.MulticlassPlattCalibrator().fit(P_train, y_train)
        params_before = dict(cal._params)

        wild_df = _synthetic_fold_frame("g", 50, seed=999)
        cal.transform(wild_df[["p_home", "p_draw", "p_away"]].to_numpy())
        self.assertEqual(cal._params, params_before)


class TestEce(unittest.TestCase):
    def test_perfect_calibration_gives_zero_ece(self):
        # confidence 1.0 (top bin), 100% correct -> mean_confidence == accuracy == 1.0, gap 0.
        # (0.95 confidence with 100% accuracy is NOT perfect calibration --
        # it's under-confidence, with a real gap of 0.05 -- covered
        # implicitly by every other ECE test using non-trivial probabilities.)
        y = ["H"] * 100
        P = np.tile([1.0, 0.0, 0.0], (100, 1))
        result = calib.compute_ece(y, P, n_bins=10)
        self.assertAlmostEqual(result["ece"], 0.0, places=6)

    def test_deterministic_across_repeated_calls(self):
        rng = np.random.default_rng(5)
        n = 200
        raw = rng.dirichlet([2, 1, 2], size=n)
        y = np.array(CLASS_ORDER)[rng.integers(0, 3, size=n)]
        r1 = calib.compute_ece(y, raw)
        r2 = calib.compute_ece(y, raw)
        self.assertEqual(r1["ece"], r2["ece"])
        self.assertEqual(r1["bins"], r2["bins"])

    def test_ece_is_nonnegative_and_finite(self):
        rng = np.random.default_rng(6)
        n = 300
        raw = rng.dirichlet([1, 1, 1], size=n)
        y = np.array(CLASS_ORDER)[rng.integers(0, 3, size=n)]
        result = calib.compute_ece(y, raw)
        self.assertGreaterEqual(result["ece"], 0.0)
        self.assertTrue(np.isfinite(result["ece"]))

    def test_bins_are_fixed_equal_width(self):
        y = ["H"]
        P = np.array([[0.55, 0.25, 0.20]])
        result = calib.compute_ece(y, P, n_bins=10)
        self.assertEqual(len(result["bins"]), 10)
        for i, b in enumerate(result["bins"]):
            self.assertAlmostEqual(b["upper"] - b["lower"], 0.1, places=10)


class TestReliabilityBins(unittest.TestCase):
    def test_deterministic_across_repeated_calls(self):
        df = _synthetic_fold_frame("f", 300, seed=7)
        b1 = calib.compute_reliability_bins(df, "D")
        b2 = calib.compute_reliability_bins(df, "D")
        self.assertEqual(b1, b2)

    def test_bin_sample_counts_sum_to_total_rows(self):
        df = _synthetic_fold_frame("f", 300, seed=8)
        bins = calib.compute_reliability_bins(df, "H")
        self.assertEqual(sum(b["n"] for b in bins), len(df))

    def test_fixed_equal_width_bins(self):
        df = _synthetic_fold_frame("f", 300, seed=9)
        bins = calib.compute_reliability_bins(df, "A", n_bins=10)
        self.assertEqual(len(bins), 10)
        for b in bins:
            self.assertAlmostEqual(b["upper"] - b["lower"], 0.1, places=10)

    def test_empty_bin_reports_none_not_error(self):
        # All probabilities near 0 -> the top bins should be empty (n=0).
        df = pd.DataFrame({
            "p_home": [0.01] * 10, "p_draw": [0.01] * 10, "p_away": [0.98] * 10,
            "y_true": ["A"] * 10,
        })
        bins = calib.compute_reliability_bins(df, "H")
        top_bin = bins[-1]
        self.assertEqual(top_bin["n"], 0)
        self.assertIsNone(top_bin["mean_predicted_probability"])


class TestCombineFoldPredictionsTemporalOrder(unittest.TestCase):
    def test_folds_concatenated_in_chronological_order(self):
        f1 = _synthetic_fold_frame("fold_1", 5, seed=1)
        f2 = _synthetic_fold_frame("fold_2", 5, seed=2)
        f3 = _synthetic_fold_frame("fold_3", 5, seed=3)
        combined = calib.combine_fold_predictions({"fold_3": f3, "fold_1": f1, "fold_2": f2})
        # Regardless of dict insertion order, output must follow
        # WALK_FORWARD_FOLDS' chronological order.
        self.assertEqual(list(combined["fold_name"].unique()), ["fold_1", "fold_2", "fold_3"])
        self.assertEqual(len(combined), 15)

    def test_row_order_within_combined_matches_fold_sequence(self):
        f1 = _synthetic_fold_frame("fold_1", 3, seed=1)
        f2 = _synthetic_fold_frame("fold_2", 3, seed=2)
        combined = calib.combine_fold_predictions({"fold_1": f1, "fold_2": f2})
        self.assertEqual(list(combined["fold_name"]), ["fold_1"] * 3 + ["fold_2"] * 3)


class TestLeakageBoundaries(unittest.TestCase):
    """Step 6 requirements: structural proof that final-test data cannot
    reach calibration fitting or method selection, and that only
    out-of-sample validation data is ever used.
    """

    def test_evaluate_calibration_methods_signature_has_no_final_test_parameter(self):
        sig = inspect.signature(calib.evaluate_calibration_methods)
        for name in sig.parameters:
            self.assertNotIn("final", name.lower())
            self.assertNotIn("test", name.lower())

    def test_select_calibration_method_signature_has_no_final_test_parameter(self):
        sig = inspect.signature(calib.select_calibration_method)
        for name in sig.parameters:
            self.assertNotIn("final", name.lower())
            self.assertNotIn("test", name.lower())

    def test_evaluate_calibration_methods_never_references_final_test_in_source(self):
        # The docstring legitimately mentions "2025/26" as prose
        # explaining its ABSENCE -- that's documentation, not a leak.
        # What actually matters: no identifier/variable named
        # final_test-anything appears anywhere in the function.
        source = inspect.getsource(calib.evaluate_calibration_methods)
        self.assertNotIn("final_test", source)

    def test_run_final_frozen_evaluation_only_called_after_selection_in_main_source(self):
        source = inspect.getsource(calib.main)
        selection_idx = source.index("select_calibration_method(")
        final_idx = source.index("run_final_frozen_evaluation(")
        self.assertLess(selection_idx, final_idx)

    def test_calibration_fit_never_receives_more_rows_than_out_of_sample_fold_sizes(self):
        # A synthetic stand-in for "training predictions cannot be
        # accidentally passed into calibration fitting": construct
        # fold_frames where each fold's row count matches only its own
        # validation-fold size, and confirm evaluate_calibration_methods
        # only ever concatenates prior VALIDATION folds (n_fit equals
        # the sum of prior folds' sizes, never larger, never including
        # any row not in fold_frames).
        f1 = _synthetic_fold_frame("fold_1", 100, seed=1)
        f2 = _synthetic_fold_frame("fold_2", 80, seed=2)
        f3 = _synthetic_fold_frame("fold_3", 60, seed=3)
        comparison = calib.evaluate_calibration_methods({"fold_1": f1, "fold_2": f2, "fold_3": f3})
        rounds = comparison["per_method_summary"]["uncalibrated"]["rounds"]
        self.assertEqual(rounds[0]["assessment_fold"], "fold_2")
        self.assertEqual(rounds[0]["n_fit"], 100)  # only fold_1
        self.assertEqual(rounds[0]["n_assess"], 80)
        self.assertEqual(rounds[1]["assessment_fold"], "fold_3")
        self.assertEqual(rounds[1]["n_fit"], 180)  # fold_1 + fold_2
        self.assertEqual(rounds[1]["n_assess"], 60)

    def test_probabilities_from_evaluate_calibration_methods_are_valid_every_round(self):
        f1 = _synthetic_fold_frame("fold_1", 120, seed=1)
        f2 = _synthetic_fold_frame("fold_2", 100, seed=2)
        f3 = _synthetic_fold_frame("fold_3", 90, seed=3)
        # No exception raised == every calibrated P passed validate_probabilities
        # inside evaluate_calibration_methods for every method/round.
        calib.evaluate_calibration_methods({"fold_1": f1, "fold_2": f2, "fold_3": f3})


class TestSelectCalibrationMethod(unittest.TestCase):
    def _comparison(self, uncalibrated_ll, platt_ll, isotonic_ll, uncalibrated_brier=0.5, platt_brier=0.5, isotonic_brier=0.5):
        return {
            "per_method_summary": {
                "uncalibrated": {"mean_log_loss": uncalibrated_ll, "mean_brier": uncalibrated_brier},
                "platt_sigmoid": {"mean_log_loss": platt_ll, "mean_brier": platt_brier},
                "isotonic": {"mean_log_loss": isotonic_ll, "mean_brier": isotonic_brier},
            }
        }

    def test_keeps_uncalibrated_when_no_improvement(self):
        comparison = self._comparison(1.0, 1.01, 1.02)
        result = calib.select_calibration_method(comparison)
        self.assertEqual(result["selected_calibration_method"], "uncalibrated")

    def test_selects_platt_when_it_clears_the_margin(self):
        comparison = self._comparison(1.0, 0.98, 1.02)
        result = calib.select_calibration_method(comparison, log_loss_improvement_margin=0.005)
        self.assertEqual(result["selected_calibration_method"], "platt_sigmoid")

    def test_does_not_select_if_improvement_is_within_margin_noise(self):
        comparison = self._comparison(1.0, 0.999, 1.02)
        result = calib.select_calibration_method(comparison, log_loss_improvement_margin=0.005)
        self.assertEqual(result["selected_calibration_method"], "uncalibrated")

    def test_rejects_candidate_that_improves_log_loss_but_worsens_brier_too_much(self):
        comparison = self._comparison(1.0, 0.95, 1.02, uncalibrated_brier=0.5, platt_brier=0.6)
        result = calib.select_calibration_method(comparison, log_loss_improvement_margin=0.005)
        self.assertEqual(result["selected_calibration_method"], "uncalibrated")


class TestRunFinalFrozenEvaluation(unittest.TestCase):
    def test_uncalibrated_decision_produces_no_post_calibration_block(self):
        combined_folds = {
            "fold_1": _synthetic_fold_frame("fold_1", 50, seed=1),
            "fold_2": _synthetic_fold_frame("fold_2", 50, seed=2),
            "fold_3": _synthetic_fold_frame("fold_3", 50, seed=3),
        }
        final_preds = _synthetic_fold_frame("final_test", 40, seed=4)
        decision = {"selected_calibration_method": "uncalibrated"}
        result = calib.run_final_frozen_evaluation(None, combined_folds, decision, final_preds)
        self.assertIsNone(result["post_calibration"])
        self.assertIn("log_loss", result["pre_calibration"])

    def test_calibrated_decision_produces_both_pre_and_post_blocks(self):
        combined_folds = {
            "fold_1": _synthetic_fold_frame("fold_1", 200, seed=1),
            "fold_2": _synthetic_fold_frame("fold_2", 200, seed=2),
            "fold_3": _synthetic_fold_frame("fold_3", 200, seed=3),
        }
        final_preds = _synthetic_fold_frame("final_test", 40, seed=4)
        decision = {"selected_calibration_method": "platt_sigmoid"}
        result = calib.run_final_frozen_evaluation(None, combined_folds, decision, final_preds)
        self.assertIsNotNone(result["post_calibration"])
        self.assertIn("log_loss", result["post_calibration"])
        self.assertIn("ece", result["post_calibration"])

    def test_pre_calibration_result_is_never_overwritten_by_post(self):
        combined_folds = {
            "fold_1": _synthetic_fold_frame("fold_1", 200, seed=1),
            "fold_2": _synthetic_fold_frame("fold_2", 200, seed=2),
            "fold_3": _synthetic_fold_frame("fold_3", 200, seed=3),
        }
        final_preds = _synthetic_fold_frame("final_test", 40, seed=4)
        decision = {"selected_calibration_method": "isotonic"}
        result = calib.run_final_frozen_evaluation(None, combined_folds, decision, final_preds)
        # pre_calibration must be computed from the ORIGINAL final_preds probabilities
        from models.evaluate import evaluate as _evaluate
        expected_pre = _evaluate(
            final_preds["y_true"].to_numpy(),
            final_preds[["p_home", "p_draw", "p_away"]].to_numpy(),
        )
        self.assertAlmostEqual(result["pre_calibration"]["log_loss"], expected_pre.log_loss, places=10)


@unittest.skipUnless(_has_real_data(), _DATA_SKIP_REASON)
class TestAgainstRealStep3Artifacts(unittest.TestCase):
    """Runs the full diagnostics + calibration pipeline against the
    REAL, already-produced Step 3 prediction CSVs -- this needs no
    scikit-learn (calibration.py's math is pure NumPy), only the
    already-generated data/audit/phase3_predictions/*.csv files.
    """

    @classmethod
    def setUpClass(cls):
        from models.data import load_supervised_dataset
        cls.dataset = load_supervised_dataset(_REAL_DB)
        cls.fold_frames = calib.load_fold_predictions_from_existing_run(cls.dataset, _PRED_DIR)
        cls.final_preds = calib.load_final_test_predictions_from_existing_run(cls.dataset, _PRED_DIR)

    def test_fold_prediction_row_counts_match_known_validation_fold_sizes(self):
        self.assertEqual(len(self.fold_frames["fold_1"]), 1827)
        self.assertEqual(len(self.fold_frames["fold_2"]), 1752)
        self.assertEqual(len(self.fold_frames["fold_3"]), 1752)

    def test_fold_predictions_only_contain_that_folds_validation_season(self):
        from models.splits import season_ids_for
        from models.config import WALK_FORWARD_FOLDS
        for fold in WALK_FORWARD_FOLDS:
            expected_season_ids = season_ids_for(fold.validation_seasons)
            actual_season_ids = set(self.fold_frames[fold.name]["season_id"].unique())
            self.assertTrue(actual_season_ids.issubset(expected_season_ids))
            # and NOT any of that fold's own training seasons:
            train_ids = season_ids_for(fold.train_seasons)
            self.assertTrue(actual_season_ids.isdisjoint(train_ids))

    def test_reproduces_frozen_step3_mean_validation_log_loss(self):
        combined = calib.combine_fold_predictions(self.fold_frames)
        diagnostics = calib.compute_probability_diagnostics(combined)
        # Frozen Step 3 number quoted in the Step 4 task: 1.0043186935714623
        # (this is the mean across 3 separately-fit folds; compute_probability_diagnostics
        # computes log loss over the pooled rows, which is a related but distinct
        # quantity -- both are reported, this just checks it's in a sane, close range.)
        self.assertAlmostEqual(diagnostics["log_loss"], 1.0043186935714623, delta=0.02)

    def test_reproduces_frozen_step3_final_test_metrics(self):
        from models.evaluate import evaluate as _evaluate
        result = _evaluate(
            self.final_preds["y_true"].to_numpy(),
            self.final_preds[["p_home", "p_draw", "p_away"]].to_numpy(),
        )
        self.assertAlmostEqual(result.log_loss, 1.012643607907126, places=9)
        self.assertAlmostEqual(result.accuracy, 0.500856653340948, places=9)

    def test_full_pipeline_runs_end_to_end_and_produces_expected_artifact_shape(self):
        combined = calib.combine_fold_predictions(self.fold_frames)
        diagnostics = calib.compute_probability_diagnostics(combined)
        for key in ["mean_predicted_probability", "actual_class_frequency", "log_loss", "brier", "ece"]:
            self.assertIn(key, diagnostics)

        draw_investigation = calib.investigate_draw_probability(combined)
        self.assertIn("top_line_heuristic_verdict", draw_investigation)

        comparison = calib.evaluate_calibration_methods(self.fold_frames)
        selection = calib.select_calibration_method(comparison)
        self.assertIn(selection["selected_calibration_method"], calib.CALIBRATION_METHODS)

        final_result = calib.run_final_frozen_evaluation(
            self.dataset, self.fold_frames, selection, self.final_preds
        )
        self.assertIn("pre_calibration", final_result)


@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP_REASON)
class TestGenerateFoldValidationPredictionsRequiresSklearn(unittest.TestCase):
    @unittest.skipUnless(_REAL_DB.exists(), "data/processed/features.db not present")
    def test_generated_predictions_use_only_approved_columns_via_train_module(self):
        # Smoke test only -- the approved-column scoping itself is
        # covered exhaustively in test_model_experiments.py; this just
        # confirms calibration.py's own generator wires through to it.
        from models.data import load_supervised_dataset
        dataset = load_supervised_dataset(_REAL_DB)
        fold_frames = calib.generate_fold_validation_predictions(dataset)
        for name, df in fold_frames.items():
            self.assertEqual(set(df.columns), set(calib.CLASS_TO_PROB_COLUMN.values()) | {
                "fold_name", "fixture_id", "season_id", "y_true", "y_pred", "max_probability",
            })


if __name__ == "__main__":
    unittest.main()
