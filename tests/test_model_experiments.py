"""Tests for src/models/run_experiments.py.

Split into two groups:
  - Selection/summary logic (FoldResult, summarize_across_folds,
    select_model, _missingness_report): pure functions over already-
    computed metrics, need no scikit-learn and no real dataset -- these
    always run.
  - Full experiment execution (run_walk_forward_experiment,
    run_final_test_evaluation, main): need both scikit-learn AND the
    real data/processed/features.db, so they're skipped (not failed)
    when either is unavailable in the current environment.
"""
import _pathfix  # noqa: F401
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models import run_experiments as exp
from models.evaluate import MetricInputError
from models.config import SEASON_ORDER

try:
    from models import train as train_module
    _TRAIN_IMPORT_ERROR = None
except ImportError as e:
    train_module = None
    _TRAIN_IMPORT_ERROR = str(e)

_REAL_DB = Path("data/processed/features.db")
_SKLEARN_SKIP_REASON = f"scikit-learn not available in this environment: {_TRAIN_IMPORT_ERROR}"
_DATA_SKIP_REASON = "data/processed/features.db not present in this environment"


def _fake_fold_result(name, train_seasons, val_seasons, metrics_by_model):
    return exp.FoldResult(
        fold_name=name,
        train_seasons=train_seasons,
        validation_seasons=val_seasons,
        n_train=100,
        n_val=20,
        train_missing_by_column={},
        val_missing_by_column={},
        model_metrics=metrics_by_model,
    )


def _metrics(log_loss, brier, macro_f1=0.5, balanced_accuracy=0.5, accuracy=0.5):
    return {
        "log_loss": log_loss, "brier": brier, "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy, "accuracy": accuracy,
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "n": 20, "n_train": 100,
    }


class TestSelectApprovedFeatures(unittest.TestCase):
    """Fix 1: the experiment runner must scope every model's input to
    exactly APPROVED_FEATURE_COLUMNS_V1 (14 features + competition_id),
    never the full feature_rows table. See
    docs/PHASE3_STEP3_DIAGNOSTIC_REPORT.md.
    """

    def test_returns_exactly_the_15_approved_columns_in_order(self):
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        # Build a frame with the approved columns (in a shuffled order)
        # plus several unapproved columns, including an xG one.
        data = {c: [1.0, 2.0] for c in APPROVED_FEATURE_COLUMNS_V1}
        data["home_xg_for_per_match_season"] = [0.1, 0.2]
        data["shrinkage_k"] = [5, 5]
        X = pd.DataFrame(data)[list(reversed(list(data.keys())))]  # shuffle column order

        selected = exp._select_approved_features(X)
        self.assertEqual(list(selected.columns), list(APPROVED_FEATURE_COLUMNS_V1))
        self.assertEqual(len(selected.columns), 15)

    def test_raises_loudly_if_an_approved_column_is_missing_rather_than_substituting(self):
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        data = {c: [1.0] for c in APPROVED_FEATURE_COLUMNS_V1 if c != "strength_diff"}
        X = pd.DataFrame(data)
        with self.assertRaises(RuntimeError):
            exp._select_approved_features(X)

    def test_no_xg_column_can_pass_through_even_if_present_in_input(self):
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        data = {c: [1.0, 2.0, 3.0] for c in APPROVED_FEATURE_COLUMNS_V1}
        for xg_col in ["home_xg_for_per_match_season", "away_xg_diff_last5", "home_xg_diff_form_last10"]:
            data[xg_col] = [np.nan, np.nan, np.nan]
        X = pd.DataFrame(data)
        selected = exp._select_approved_features(X)
        xg_cols_present = [c for c in selected.columns if "xg" in c.lower()]
        self.assertEqual(xg_cols_present, [])


@unittest.skipUnless(_REAL_DB.exists(), _DATA_SKIP_REASON)
class TestApprovedFeatureScopingReachesModels(unittest.TestCase):
    """Verifies the ACTUAL columns run_walk_forward_experiment /
    run_final_test_evaluation hand to each candidate model, using a fake
    `models.train` module installed directly into sys.modules. This
    needs the real dataset (to exercise the real approved-fold season
    boundaries) but deliberately does NOT need scikit-learn: the fake
    module is picked up by run_experiments.py's lazy `from . import
    train as train_module` before Python ever tries to import the real,
    sklearn-dependent module.
    """

    def setUp(self):
        import sys
        import types

        import models

        # `run_experiments.py` resolves its dependency with
        # `from . import train as train_module`. CPython resolves that by
        # first reading the ATTRIBUTE `models.train` off the package
        # object, and only falling back to `sys.modules["models.train"]`
        # when no such attribute exists. A successful `import models.train`
        # anywhere earlier in the test session (e.g. test_model_calibration.py's
        # module-level import, which succeeds wherever scikit-learn IS
        # installed) sets that package attribute to the real module.
        # Patching `sys.modules` alone therefore silently has no effect in
        # a scikit-learn-equipped environment -- the real module is used,
        # the fakes never run, and `self.recorded` stays empty. Both the
        # package attribute and sys.modules must be patched, and both
        # restored.
        self._models_pkg = models
        self._had_pkg_attr = hasattr(models, "train")
        self._real_pkg_attr = getattr(models, "train", None)
        self._real_train_module = sys.modules.get("models.train")
        self.recorded: dict[str, list[list[str]]] = {}

        fake = types.ModuleType("models.train")

        def fake_lr(X_train, y_train, X_val):
            self.recorded.setdefault("lr_train_columns", []).append(list(X_train.columns))
            self.recorded.setdefault("lr_val_columns", []).append(list(X_val.columns))
            P = np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (len(X_val), 1))
            return None, None, P

        def fake_hgb(X_train, y_train, X_val):
            self.recorded.setdefault("hgb_train_columns", []).append(list(X_train.columns))
            self.recorded.setdefault("hgb_val_columns", []).append(list(X_val.columns))
            P = np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (len(X_val), 1))
            return None, None, P

        fake.train_logistic_regression = fake_lr
        fake.train_hist_gradient_boosting = fake_hgb
        sys.modules["models.train"] = fake
        models.train = fake  # the binding `from . import train` actually reads

        from models.data import load_supervised_dataset
        self.dataset = load_supervised_dataset(_REAL_DB)

    def tearDown(self):
        import sys
        if self._real_train_module is not None:
            sys.modules["models.train"] = self._real_train_module
        else:
            sys.modules.pop("models.train", None)
        if self._had_pkg_attr:
            self._models_pkg.train = self._real_pkg_attr
        else:
            try:
                del self._models_pkg.train
            except AttributeError:
                pass

    def test_exactly_the_approved_15_columns_reach_both_models_every_fold(self):
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        exp.run_walk_forward_experiment(self.dataset)
        expected = set(APPROVED_FEATURE_COLUMNS_V1)
        for key in ("lr_train_columns", "lr_val_columns", "hgb_train_columns", "hgb_val_columns"):
            self.assertIn(key, self.recorded)
            for cols in self.recorded[key]:
                self.assertEqual(set(cols), expected)
                self.assertEqual(len(cols), 15)

    def test_no_unapproved_xg_columns_reach_either_model_in_any_fold(self):
        exp.run_walk_forward_experiment(self.dataset)
        for key in ("lr_train_columns", "lr_val_columns", "hgb_train_columns", "hgb_val_columns"):
            for cols in self.recorded[key]:
                xg_cols = [c for c in cols if "xg" in c.lower()]
                self.assertEqual(xg_cols, [], f"{key} contained xG column(s): {xg_cols}")

    def test_final_test_evaluation_also_scoped_to_approved_columns(self):
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        exp.run_final_test_evaluation(self.dataset)
        expected = set(APPROVED_FEATURE_COLUMNS_V1)
        for key in ("lr_train_columns", "lr_val_columns", "hgb_train_columns", "hgb_val_columns"):
            for cols in self.recorded[key]:
                self.assertEqual(set(cols), expected)


class TestMissingnessReport(unittest.TestCase):
    def test_reports_only_columns_with_nan(self):
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
        report = exp._missingness_report(X)
        self.assertEqual(report, {"a": 1})

    def test_empty_dict_when_no_missing_values(self):
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        self.assertEqual(exp._missingness_report(X), {})


class TestEvaluateAndRecord(unittest.TestCase):
    def test_valid_probabilities_pass_through(self):
        y = ["H", "D", "A"]
        P = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        d = exp._evaluate_and_record(y, P, n_train=50)
        self.assertEqual(d["n_train"], 50)
        self.assertEqual(d["dropped_or_imputed"], {})
        self.assertIn("log_loss", d)

    def test_invalid_probabilities_raise_not_silently_pass(self):
        y = ["H"]
        P = np.array([[0.5, 0.5, 0.5]])  # sums to 1.5
        with self.assertRaises(MetricInputError):
            exp._evaluate_and_record(y, P, n_train=10)

    def test_dropped_imputed_recorded_when_given(self):
        y = ["H"]
        P = np.array([[1.0, 0.0, 0.0]])
        d = exp._evaluate_and_record(y, P, n_train=10, dropped_imputed={"feat_a": 3})
        self.assertEqual(d["dropped_or_imputed"], {"feat_a": 3})


class TestSummarizeAcrossFolds(unittest.TestCase):
    def _two_fold_results(self):
        f1 = _fake_fold_result("fold_1", ("2020/2021",), ("2021/2022",), {
            exp.MODEL_FREQUENCY: _metrics(1.0, 0.5),
            exp.MODEL_MAJORITY: _metrics(1.2, 0.6),
            exp.MODEL_LOGREG: _metrics(0.9, 0.45),
            exp.MODEL_HGB: _metrics(0.85, 0.40),
        })
        f2 = _fake_fold_result("fold_2", ("2020/2021", "2021/2022"), ("2022/2023",), {
            exp.MODEL_FREQUENCY: _metrics(1.1, 0.55),
            exp.MODEL_MAJORITY: _metrics(1.3, 0.65),
            exp.MODEL_LOGREG: _metrics(1.1, 0.55),
            exp.MODEL_HGB: _metrics(0.95, 0.42),
        })
        return [f1, f2]

    def test_hand_computed_mean_and_std(self):
        summary = exp.summarize_across_folds(self._two_fold_results())
        # LogisticRegression log_loss: [0.9, 1.1] -> mean=1.0, std=0.1
        self.assertAlmostEqual(summary[exp.MODEL_LOGREG]["log_loss"]["mean"], 1.0, places=10)
        self.assertAlmostEqual(summary[exp.MODEL_LOGREG]["log_loss"]["std"], 0.1, places=10)

    def test_values_by_fold_preserved_in_order(self):
        summary = exp.summarize_across_folds(self._two_fold_results())
        self.assertEqual(summary[exp.MODEL_HGB]["log_loss"]["values_by_fold"], [0.85, 0.95])

    def test_all_models_and_metrics_present(self):
        summary = exp.summarize_across_folds(self._two_fold_results())
        self.assertEqual(set(summary.keys()), set(exp.ALL_MODELS))
        for model_summary in summary.values():
            self.assertEqual(
                set(model_summary.keys()),
                {"log_loss", "brier", "macro_f1", "balanced_accuracy", "accuracy"},
            )


class TestSelectModel(unittest.TestCase):
    def test_signature_takes_only_fold_summary_no_final_test_access(self):
        # select_model must have no parameter that could carry final-test
        # (2025/26) data -- this is a structural guarantee, not just a
        # convention: there is no argument here a caller could even use
        # to pass it in.
        sig = inspect.signature(exp.select_model)
        param_names = list(sig.parameters.keys())
        self.assertEqual(param_names, ["summary", "log_loss_tie_margin"])
        for name in param_names:
            self.assertNotIn("final", name.lower())
            self.assertNotIn("test", name.lower())

    def test_picks_lower_log_loss_when_not_a_tie(self):
        summary = {
            exp.MODEL_FREQUENCY: {"log_loss": {"mean": 1.05}, "brier": {"mean": 0.5}},
            exp.MODEL_LOGREG: {"log_loss": {"mean": 1.0}, "brier": {"mean": 0.5}},
            exp.MODEL_HGB: {"log_loss": {"mean": 0.8}, "brier": {"mean": 0.4}},
        }
        result = exp.select_model(summary, log_loss_tie_margin=0.005)
        self.assertEqual(result["selected_model"], exp.MODEL_HGB)
        self.assertFalse(result["tie_break_applied"])

    def test_tie_break_uses_brier_score(self):
        summary = {
            exp.MODEL_FREQUENCY: {"log_loss": {"mean": 1.2}, "brier": {"mean": 0.6}},
            exp.MODEL_LOGREG: {"log_loss": {"mean": 0.900}, "brier": {"mean": 0.42}},
            exp.MODEL_HGB: {"log_loss": {"mean": 0.902}, "brier": {"mean": 0.40}},  # within margin, lower Brier
        }
        result = exp.select_model(summary, log_loss_tie_margin=0.005)
        self.assertTrue(result["tie_break_applied"])
        self.assertEqual(result["selected_model"], exp.MODEL_HGB)

    def test_beats_frequency_baseline_flag_computed_per_candidate(self):
        summary = {
            exp.MODEL_FREQUENCY: {"log_loss": {"mean": 1.0}, "brier": {"mean": 0.5}},
            exp.MODEL_LOGREG: {"log_loss": {"mean": 1.05}, "brier": {"mean": 0.5}},  # worse than baseline
            exp.MODEL_HGB: {"log_loss": {"mean": 0.9}, "brier": {"mean": 0.4}},  # better than baseline
        }
        result = exp.select_model(summary, log_loss_tie_margin=0.005)
        self.assertFalse(result["beats_frequency_baseline"][exp.MODEL_LOGREG])
        self.assertTrue(result["beats_frequency_baseline"][exp.MODEL_HGB])


class TestNoFutureSeasonInFoldDefinition(unittest.TestCase):
    """Structural check independent of any dataset: every walk-forward
    fold's validation season(s) must be chronologically after all of
    that fold's training seasons, per SEASON_ORDER.
    """

    def test_every_fold_validation_season_after_all_training_seasons(self):
        from models.config import WALK_FORWARD_FOLDS
        for fold in WALK_FORWARD_FOLDS:
            max_train_idx = max(SEASON_ORDER.index(s) for s in fold.train_seasons)
            min_val_idx = min(SEASON_ORDER.index(s) for s in fold.validation_seasons)
            self.assertLess(
                max_train_idx, min_val_idx,
                f"{fold.name}: training season(s) {fold.train_seasons} not strictly "
                f"before validation season(s) {fold.validation_seasons}",
            )

    def test_2025_26_never_in_any_fold_train_or_validation_seasons(self):
        from models.config import WALK_FORWARD_FOLDS
        for fold in WALK_FORWARD_FOLDS:
            self.assertNotIn("2025/2026", fold.train_seasons)
            self.assertNotIn("2025/2026", fold.validation_seasons)


class TestMainOrdersSelectionBeforeFinalTest(unittest.TestCase):
    """Structural (source-level) check that main() computes fold results
    and freezes the selection decision BEFORE it ever calls
    run_final_test_evaluation -- proven by call order in the source, not
    just by convention.
    """

    def test_final_test_call_appears_after_selection_call_in_source(self):
        source = inspect.getsource(exp.main)
        selection_idx = source.index("select_model(")
        final_test_idx = source.index("run_final_test_evaluation(")
        self.assertLess(
            selection_idx, final_test_idx,
            "main() must call select_model(...) before run_final_test_evaluation(...)",
        )

    def test_output_dict_assigns_model_selection_before_final_test(self):
        source = inspect.getsource(exp.main)
        selection_assign_idx = source.index('"model_selection": selection')
        final_test_assign_idx = source.index('output["final_test"] = run_final_test_evaluation')
        self.assertLess(selection_assign_idx, final_test_assign_idx)


@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP_REASON)
class TestRunWalkForwardExperimentAgainstRealDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _REAL_DB.exists():
            raise unittest.SkipTest(_DATA_SKIP_REASON)
        from models.data import load_supervised_dataset
        cls.dataset = load_supervised_dataset(_REAL_DB)

    def test_all_four_models_evaluated_per_fold(self):
        results = exp.run_walk_forward_experiment(self.dataset)
        self.assertEqual(len(results), 3)
        for fr in results:
            self.assertEqual(set(fr.model_metrics.keys()), set(exp.ALL_MODELS))

    def test_probabilities_valid_implicitly_via_no_exception(self):
        # _evaluate_and_record raises MetricInputError on invalid P --
        # a clean run with no exception is itself the assertion that
        # every model's probabilities were valid for every fold.
        exp.run_walk_forward_experiment(self.dataset)

    def test_no_future_season_in_training_for_any_fold(self):
        results = exp.run_walk_forward_experiment(self.dataset)
        for fr in results:
            max_train_idx = max(SEASON_ORDER.index(s) for s in fr.train_seasons)
            min_val_idx = min(SEASON_ORDER.index(s) for s in fr.validation_seasons)
            self.assertLess(max_train_idx, min_val_idx)
            self.assertNotIn("2025/2026", fr.train_seasons)
            self.assertNotIn("2025/2026", fr.validation_seasons)

    def test_deterministic_across_repeated_runs(self):
        results_a = exp.run_walk_forward_experiment(self.dataset)
        results_b = exp.run_walk_forward_experiment(self.dataset)
        for fa, fb in zip(results_a, results_b):
            for model_name in exp.ALL_MODELS:
                self.assertAlmostEqual(
                    fa.model_metrics[model_name]["log_loss"],
                    fb.model_metrics[model_name]["log_loss"],
                    places=10,
                    msg=f"{fa.fold_name}/{model_name} log_loss not deterministic across runs",
                )

    def test_no_label_columns_in_feature_matrix_used_for_training(self):
        from models.splits import iter_walk_forward_folds
        for fold, train_ds, val_ds in iter_walk_forward_folds(self.dataset):
            for col in ["label_home_goals", "label_away_goals", "label_result"]:
                self.assertNotIn(col, train_ds.X.columns)
                self.assertNotIn(col, val_ds.X.columns)


@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP_REASON)
class TestFullPipelineAgainstRealDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _REAL_DB.exists():
            raise unittest.SkipTest(_DATA_SKIP_REASON)

    def test_main_writes_results_json_with_expected_top_level_keys(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "phase3_model_comparison.json"
            output = exp.main(
                features_db=_REAL_DB, results_path=results_path,
                predictions_dir=Path(tmp) / "predictions", run_final_test=True,
            )
            self.assertTrue(results_path.exists())
            for key in ["sklearn_version", "folds", "summary_across_folds", "model_selection", "final_test"]:
                self.assertIn(key, output)
            self.assertIn(output["model_selection"]["selected_model"], exp.REAL_CANDIDATE_MODELS)

    def test_selected_model_unaffected_by_skipping_final_test(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_with_final = exp.main(
                features_db=_REAL_DB, results_path=Path(tmp) / "a.json",
                predictions_dir=None, run_final_test=True,
            )
            out_without_final = exp.main(
                features_db=_REAL_DB, results_path=Path(tmp) / "b.json",
                predictions_dir=None, run_final_test=False,
            )
            self.assertEqual(
                out_with_final["model_selection"]["selected_model"],
                out_without_final["model_selection"]["selected_model"],
            )
            self.assertIsNone(out_without_final["final_test"])


if __name__ == "__main__":
    unittest.main()
