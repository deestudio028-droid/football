"""Consolidated leakage tests for Phase 3 Step 2, mapping directly to
the 10 mandatory leakage-test items from the task specification. Each
test method's docstring/name states exactly which item it proves.
Several items are already covered elsewhere (test_model_splits.py,
test_baselines.py, test_evaluation.py, test_model_training.py); those
are re-asserted here too, in one place, so this file alone is a
complete checklist against the spec.
"""
import _pathfix  # noqa: F401
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from models import train as train_module
    _IMPORT_ERROR = None
except ImportError as exc:
    train_module = None
    _IMPORT_ERROR = str(exc)

from models.baselines import CLASS_ORDER, FrequencyBaseline, class_frequencies
from models.config import FINAL_TEST_SEASONS, WALK_FORWARD_FOLDS
from models.evaluate import validate_probabilities
from models.splits import iter_walk_forward_folds, verify_temporal_safety

_SKIP_REASON = f"scikit-learn not available in this environment: {_IMPORT_ERROR}"


# --- (1) preprocessing is fitted only on training data --------------------
@unittest.skipUnless(train_module is not None, _SKIP_REASON)
class Test01PreprocessingFittedOnlyOnTrainingData(unittest.TestCase):
    def test_logistic_regression_preprocessor_statistics_come_only_from_train(self):
        X_train = pd.DataFrame({"feat_a": [10.0, 20.0, 30.0], "competition_id": [423, 423, 423]})
        pre = train_module.LogisticRegressionPreprocessor().fit(X_train)
        idx = pre.numeric_columns.index("feat_a")
        self.assertAlmostEqual(pre._scaler.mean_[idx], 20.0)  # mean of TRAIN only


# --- (2) validation data cannot change imputation statistics --------------
@unittest.skipUnless(train_module is not None, _SKIP_REASON)
class Test02ValidationCannotChangeImputationStatistics(unittest.TestCase):
    def test_transform_on_validation_does_not_mutate_imputer_statistics(self):
        X_train = pd.DataFrame({"feat_a": [1.0, 2.0, np.nan, 4.0], "competition_id": [423] * 4})
        pre = train_module.LogisticRegressionPreprocessor().fit(X_train)
        before = pre._imputer.statistics_.copy()

        X_val = pd.DataFrame({"feat_a": [999.0, -999.0, np.nan], "competition_id": [423] * 3})
        pre.transform(X_val)

        np.testing.assert_array_equal(pre._imputer.statistics_, before)


# --- (3) validation data cannot change scaling statistics ------------------
@unittest.skipUnless(train_module is not None, _SKIP_REASON)
class Test03ValidationCannotChangeScalingStatistics(unittest.TestCase):
    def test_transform_on_validation_does_not_mutate_scaler_statistics(self):
        X_train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "competition_id": [423] * 3})
        pre = train_module.LogisticRegressionPreprocessor().fit(X_train)
        mean_before = pre._scaler.mean_.copy()
        scale_before = pre._scaler.scale_.copy()

        X_val = pd.DataFrame({"feat_a": [10000.0, -10000.0], "competition_id": [423, 423]})
        pre.transform(X_val)

        np.testing.assert_array_equal(pre._scaler.mean_, mean_before)
        np.testing.assert_array_equal(pre._scaler.scale_, scale_before)


# --- (4) final test data is never used during model selection --------------
class Test04FinalTestNeverUsedDuringModelSelection(unittest.TestCase):
    def test_2025_26_absent_from_every_walk_forward_fold_definition(self):
        for fold in WALK_FORWARD_FOLDS:
            self.assertNotIn("2025/2026", fold.train_seasons)
            self.assertNotIn("2025/2026", fold.validation_seasons)

    def test_final_test_seasons_is_exactly_2025_26_and_disjoint_from_folds(self):
        self.assertEqual(FINAL_TEST_SEASONS, ("2025/2026",))
        fold_seasons = set()
        for fold in WALK_FORWARD_FOLDS:
            fold_seasons.update(fold.train_seasons)
            fold_seasons.update(fold.validation_seasons)
        self.assertTrue(set(FINAL_TEST_SEASONS).isdisjoint(fold_seasons))


# --- (5) baseline probabilities come only from training labels -------------
class Test05BaselineProbabilitiesFromTrainingLabelsOnly(unittest.TestCase):
    def test_frequency_baseline_ignores_everything_but_fit_labels(self):
        y_train = pd.Series(["H"] * 7 + ["A"] * 3)
        baseline = FrequencyBaseline().fit(y_train)
        # Requesting probabilities for very different row counts must not
        # change the per-class probabilities -- there is no code path here
        # through which validation features/labels could influence output.
        p1 = baseline.predict_proba(n_rows=1)[0]
        p999 = baseline.predict_proba(n_rows=999)[0]
        np.testing.assert_allclose(p1, p999)

    def test_class_frequencies_signature_takes_only_training_labels(self):
        import inspect
        sig = inspect.signature(class_frequencies)
        self.assertEqual(list(sig.parameters.keys()), ["y_train"])


# --- (6) labels never enter X ----------------------------------------------
class Test06LabelsNeverEnterX(unittest.TestCase):
    def test_x_excluded_columns_contains_all_label_columns(self):
        from models.config import X_EXCLUDED_COLUMNS
        from features.storage import LABEL_COLUMNS
        for col in LABEL_COLUMNS:
            self.assertIn(col, X_EXCLUDED_COLUMNS)

    def test_real_dataset_X_has_no_label_columns_if_present(self):
        real_db = Path("data/processed/features.db")
        if not real_db.exists():
            raise unittest.SkipTest("data/processed/features.db not present in this environment")
        from models.data import load_supervised_dataset
        ds = load_supervised_dataset(real_db)
        for col in ["label_home_goals", "label_away_goals", "label_result"]:
            self.assertNotIn(col, ds.X.columns)


# --- (7) probabilities sum to 1 --------------------------------------------
class Test07ProbabilitiesSumToOne(unittest.TestCase):
    def test_frequency_baseline_sums_to_one(self):
        baseline = FrequencyBaseline().fit(pd.Series(["H", "D", "D", "A"]))
        P = baseline.predict_proba(n_rows=5)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(5))

    def test_validate_probabilities_enforces_row_sum(self):
        from models.evaluate import MetricInputError
        with self.assertRaises(MetricInputError):
            validate_probabilities(np.array([[0.5, 0.5, 0.5]]))  # sums to 1.5


# --- (8) no NaN/inf probabilities are emitted -------------------------------
class Test08NoNanOrInfProbabilities(unittest.TestCase):
    def test_validate_probabilities_rejects_nan(self):
        from models.evaluate import MetricInputError
        with self.assertRaises(MetricInputError):
            validate_probabilities(np.array([[np.nan, 0.5, 0.5]]))

    def test_validate_probabilities_rejects_inf(self):
        from models.evaluate import MetricInputError
        with self.assertRaises(MetricInputError):
            validate_probabilities(np.array([[np.inf, -np.inf, 1.0]]))


# --- (9) candidate models use the approved temporal folds -------------------
@unittest.skipUnless(train_module is not None, _SKIP_REASON)
class Test09CandidateModelsUseApprovedTemporalFolds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        real_db = Path("data/processed/features.db")
        if not real_db.exists():
            raise unittest.SkipTest("data/processed/features.db not present in this environment")
        from models.data import load_supervised_dataset
        cls.dataset = load_supervised_dataset(real_db)

    def test_train_and_evaluate_only_called_with_folds_from_iter_walk_forward_folds(self):
        folds_seen = []
        for fold, train_ds, val_ds in iter_walk_forward_folds(self.dataset):
            verify_temporal_safety(train_ds, val_ds)  # re-verified, not just trusted
            folds_seen.append(fold.name)
            P_lr = train_module.train_logistic_regression(train_ds.X, train_ds.y, val_ds.X)[2]
            validate_probabilities(P_lr)
        self.assertEqual(folds_seen, [f.name for f in WALK_FORWARD_FOLDS])


# --- (10) no random split exists anywhere in the model pipeline -------------
class Test10NoRandomSplitAnywhereInPipeline(unittest.TestCase):
    def test_splits_module_has_no_random_or_shuffle_symbols(self):
        # Checks actual code usage (import statements / function calls),
        # not prose -- splits.py's own docstring says "never random" in
        # English, which would be a false positive for a naive substring
        # search. What actually matters is that no random-split machinery
        # is imported or invoked anywhere in the module.
        import ast
        import inspect
        from models import splits as splits_module

        source = inspect.getsource(splits_module)
        tree = ast.parse(source)

        forbidden_names = {"random", "shuffle", "sample", "train_test_split", "np", "numpy"}
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name in forbidden_names or "random" in name:
                        found.append(f"import of {alias.name!r}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"shuffle", "sample", "train_test_split"} or "random" in node.func.attr:
                    found.append(f"call to .{node.func.attr}(...)")

        self.assertEqual(found, [], f"splits.py must not use any random-split machinery, found: {found}")

    def test_walk_forward_folds_are_fully_deterministic_across_repeated_calls(self):
        real_db = Path("data/processed/features.db")
        if not real_db.exists():
            raise unittest.SkipTest("data/processed/features.db not present in this environment")
        from models.data import load_supervised_dataset
        dataset = load_supervised_dataset(real_db)
        run_a = [(f.name, len(tr), len(va)) for f, tr, va in iter_walk_forward_folds(dataset)]
        run_b = [(f.name, len(tr), len(va)) for f, tr, va in iter_walk_forward_folds(dataset)]
        self.assertEqual(run_a, run_b)


if __name__ == "__main__":
    unittest.main()
