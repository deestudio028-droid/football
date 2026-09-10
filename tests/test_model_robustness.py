"""Tests for src/models/robustness.py and src/models/run_robustness.py
-- Phase 4B regularization-sensitivity study (design frozen, training
NOT started).

Covers the frozen design: C grid, exact Model A/B columns, B = A +
shots + shots-on-target, no xG, unchanged fold boundaries, no random
splitting, no final-test reachability, probability validity, metric
correctness, fold and preprocessing isolation, deterministic
configuration, and that no V1 contract was modified.

The equivalence check that Phase 4B at C=1.0 reproduces Phase 4A's
recorded numbers requires scikit-learn and runs on the project machine.
"""
import _pathfix  # noqa: F401
import ast
import inspect
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models import robustness, run_robustness
from models.ablation import MODEL_A, MODEL_A_COLUMNS, MODEL_B, MODEL_B_COLUMNS, SHOTS_CORE_COLUMNS, SHOTS_ON_CORE_COLUMNS, XG_CORE_COLUMNS

_REAL_DB = Path("data/processed/features.db")
_PHASE4A_JSON = Path("data/audit/phase4a_ablation_comparison.json")
_DATA_SKIP = "data/processed/features.db not present in this environment"

try:
    from models import train as train_module
    _TRAIN_ERR = None
except ImportError as e:
    train_module = None
    _TRAIN_ERR = str(e)
_SKLEARN_SKIP = f"scikit-learn not available in this environment: {_TRAIN_ERR}"


class TestFrozenCGrid(unittest.TestCase):
    def test_grid_is_exactly_the_declared_nine_values(self):
        self.assertEqual(robustness.C_GRID, (0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0))

    def test_grid_is_a_tuple_so_it_cannot_be_mutated_in_place(self):
        self.assertIsInstance(robustness.C_GRID, tuple)

    def test_grid_is_strictly_increasing_and_has_no_duplicates(self):
        g = list(robustness.C_GRID)
        self.assertEqual(g, sorted(g))
        self.assertEqual(len(g), len(set(g)))

    def test_grid_contains_v1_baseline_and_it_is_not_at_an_edge(self):
        self.assertIn(robustness.V1_C_VALUE, robustness.C_GRID)
        self.assertEqual(robustness.V1_C_VALUE, 1.0)
        self.assertNotEqual(robustness.C_GRID[0], robustness.V1_C_VALUE)
        self.assertNotEqual(robustness.C_GRID[-1], robustness.V1_C_VALUE)

    def test_all_C_values_are_positive(self):
        for C in robustness.C_GRID:
            self.assertGreater(C, 0)

    def test_build_logistic_regression_rejects_off_grid_C(self):
        # Guards against a pre-declared sensitivity check silently
        # becoming an unregistered hyperparameter search.
        with self.assertRaises(ValueError):
            robustness.build_logistic_regression(0.75)


class TestV1ConfigurationReproducedExactly(unittest.TestCase):
    """The whole study is only meaningful if Phase 4B's estimator is V1's
    estimator with C as the single varied parameter."""

    def _v1_logreg_kwargs_from_train_py(self) -> dict:
        tree = ast.parse(Path("src/models/train.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LogisticRegression":
                return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        self.fail("No LogisticRegression(...) construction found in train.py")

    def test_base_kwargs_match_v1_exactly_except_C(self):
        v1 = self._v1_logreg_kwargs_from_train_py()
        self.assertIn("C", v1, "V1 must specify C for this comparison to be meaningful")
        expected_base = {k: v for k, v in v1.items() if k != "C"}
        self.assertEqual(
            robustness.V1_LOGREG_BASE_KWARGS, expected_base,
            "Phase 4B's base kwargs have drifted from V1's LogisticRegression configuration",
        )

    def test_v1_C_value_constant_matches_the_C_hardcoded_in_train_py(self):
        v1 = self._v1_logreg_kwargs_from_train_py()
        self.assertEqual(robustness.V1_C_VALUE, v1["C"])

    def test_base_kwargs_do_not_contain_C(self):
        self.assertNotIn("C", robustness.V1_LOGREG_BASE_KWARGS)

    def test_phase4b_reuses_v1_preprocessor_and_reorder_not_its_own(self):
        src = inspect.getsource(robustness.train_logistic_regression_with_C)
        self.assertIn("LogisticRegressionPreprocessor", src)
        self.assertIn("_reorder_proba", src)
        # and must not define a competing preprocessing implementation
        for forbidden in ("SimpleImputer(", "StandardScaler(", "fillna("):
            self.assertNotIn(forbidden, inspect.getsource(robustness))

    def test_preprocessor_is_fit_on_training_data_only(self):
        src = inspect.getsource(robustness.train_logistic_regression_with_C)
        self.assertIn(".fit(X_train)", src)
        self.assertNotIn(".fit(X_val)", src)
        self.assertNotIn("fit_transform(X_val", src)


class TestFrozenModelColumns(unittest.TestCase):
    def test_model_a_columns_are_exactly_phase4a_model_a(self):
        self.assertEqual(robustness.PHASE4B_MODELS[MODEL_A], MODEL_A_COLUMNS)
        self.assertEqual(len(MODEL_A_COLUMNS), 44)

    def test_model_b_columns_are_exactly_phase4a_model_b(self):
        self.assertEqual(robustness.PHASE4B_MODELS[MODEL_B], MODEL_B_COLUMNS)
        self.assertEqual(len(MODEL_B_COLUMNS), 80)

    def test_b_equals_a_plus_shots_and_shots_on_target(self):
        self.assertEqual(
            set(MODEL_B_COLUMNS),
            set(MODEL_A_COLUMNS) | set(SHOTS_CORE_COLUMNS) | set(SHOTS_ON_CORE_COLUMNS),
        )

    def test_a_is_a_strict_subset_of_b(self):
        self.assertTrue(set(MODEL_A_COLUMNS) < set(MODEL_B_COLUMNS))

    def test_no_xg_columns_in_either_model(self):
        for name, cols in robustness.PHASE4B_MODELS.items():
            self.assertTrue(set(cols).isdisjoint(XG_CORE_COLUMNS), f"{name} contains xG columns")

    def test_only_models_a_and_b_are_studied(self):
        self.assertEqual(set(robustness.PHASE4B_MODELS), {MODEL_A, MODEL_B})

    def test_no_duplicate_columns_in_either_model(self):
        for name, cols in robustness.PHASE4B_MODELS.items():
            self.assertEqual(len(cols), len(set(cols)), f"{name} has duplicate columns")

    def test_no_label_or_bookkeeping_columns_in_either_model(self):
        from models.config import X_EXCLUDED_COLUMNS
        for name, cols in robustness.PHASE4B_MODELS.items():
            self.assertEqual(set(X_EXCLUDED_COLUMNS) & set(cols), set(), f"{name} leaks excluded columns")

    def test_select_model_features_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            robustness.select_model_features(pd.DataFrame({"a": [1]}), "model_c")

    def test_select_model_features_fails_loudly_on_missing_column(self):
        with self.assertRaises(RuntimeError):
            robustness.select_model_features(pd.DataFrame({"competition_id": [423]}), MODEL_A)


class TestFinalTestUnreachable(unittest.TestCase):
    """2025/26 must be structurally unreachable from the Phase 4B path."""

    def test_runner_never_imports_or_calls_final_split(self):
        tree = ast.parse(inspect.getsource(run_robustness))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if "final_split" in alias.name:
                        offenders.append(alias.name)
            elif isinstance(node, ast.Call):
                fn = node.func
                if (getattr(fn, "id", None) or getattr(fn, "attr", None)) == "final_split":
                    offenders.append("final_split()")
        self.assertEqual(offenders, [])
        self.assertFalse(hasattr(run_robustness, "final_split"))

    def test_runner_never_references_final_test_season_constants(self):
        tree = ast.parse(inspect.getsource(run_robustness))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        for forbidden in ("FINAL_TEST_SEASONS", "FINAL_TRAIN_SEASONS"):
            self.assertNotIn(forbidden, names)
            self.assertNotIn(forbidden, imported)

    def test_no_public_function_accepts_a_final_test_parameter(self):
        for mod in (robustness, run_robustness):
            for name, fn in vars(mod).items():
                if not callable(fn) or not getattr(fn, "__module__", "").startswith("models."):
                    continue
                try:
                    params = inspect.signature(fn).parameters
                except (TypeError, ValueError):
                    continue
                for p in params:
                    self.assertNotIn("final", p.lower(), f"{mod.__name__}.{name} has param {p!r}")
                    self.assertNotIn("test_season", p.lower(), f"{mod.__name__}.{name} has param {p!r}")

    def test_analysis_functions_take_only_fold_summary(self):
        self.assertEqual(list(inspect.signature(run_robustness.analyze_robustness).parameters), ["summary"])

    def test_experiment_uses_only_iter_walk_forward_folds(self):
        src = inspect.getsource(run_robustness.run_sensitivity_experiment)
        self.assertIn("iter_walk_forward_folds", src)

    def test_no_random_split_machinery_in_either_module(self):
        for mod in (robustness, run_robustness):
            tree = ast.parse(inspect.getsource(mod))
            offenders = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        if a.name.split(".")[0] == "random" or "train_test_split" in a.name or "KFold" in a.name:
                            offenders.append(a.name)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"shuffle", "sample", "train_test_split"}:
                        offenders.append(node.func.attr)
            self.assertEqual(offenders, [], f"{mod.__name__}: {offenders}")

    def test_writes_only_to_phase4b_artifact_paths(self):
        self.assertIn("phase4b", str(run_robustness.DEFAULT_RESULTS_PATH))
        self.assertIn("phase4b", str(run_robustness.DEFAULT_PREDICTIONS_DIR))

    def test_never_writes_a_phase3_or_phase4a_artifact_path(self):
        src = Path("src/models/run_robustness.py").read_text(encoding="utf-8")
        for artifact in [
            "phase3_model_comparison.json", "phase3_calibration_comparison.json",
            "phase3_probability_diagnostics.json", "phase3_predictions",
            "phase4a_ablation_comparison.json", "phase4a_predictions",
        ]:
            self.assertNotIn(artifact, src)


class TestFoldBoundariesUnchanged(unittest.TestCase):
    def test_walk_forward_folds_are_the_frozen_phase3_boundaries(self):
        from models.config import WALK_FORWARD_FOLDS
        actual = [(f.name, f.train_seasons, f.validation_seasons) for f in WALK_FORWARD_FOLDS]
        self.assertEqual(actual, [
            ("fold_1", ("2020/2021", "2021/2022"), ("2022/2023",)),
            ("fold_2", ("2020/2021", "2021/2022", "2022/2023"), ("2023/2024",)),
            ("fold_3", ("2020/2021", "2021/2022", "2022/2023", "2023/2024"), ("2024/2025",)),
        ])

    def test_2025_26_absent_from_every_fold(self):
        from models.config import WALK_FORWARD_FOLDS
        for f in WALK_FORWARD_FOLDS:
            self.assertNotIn("2025/2026", f.train_seasons)
            self.assertNotIn("2025/2026", f.validation_seasons)


class TestV1ContractsUntouched(unittest.TestCase):
    def test_v1_constants_unchanged(self):
        from models.config import (
            APPROVED_FEATURE_COLUMNS_V1, FINAL_TEST_SEASONS, MODEL_VERSION,
            REQUIRED_FEATURE_VERSION, WALK_FORWARD_FOLDS,
        )
        from models.baselines import CLASS_ORDER
        self.assertEqual(MODEL_VERSION, "v1.0")
        self.assertEqual(REQUIRED_FEATURE_VERSION, "v1.0")
        self.assertEqual(len(APPROVED_FEATURE_COLUMNS_V1), 15)
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])
        self.assertEqual(len(WALK_FORWARD_FOLDS), 3)
        self.assertEqual(FINAL_TEST_SEASONS, ("2025/2026",))

    def test_phase4b_declares_no_competing_model_version(self):
        src = Path("src/models/robustness.py").read_text(encoding="utf-8")
        self.assertNotIn("MODEL_VERSION =", src)
        self.assertNotIn("APPROVED_FEATURE_COLUMNS_V1 =", src)


class TestSummaryAndAnalysisLogic(unittest.TestCase):
    """Pure-logic tests on synthetic records -- no training required."""

    def _records(self, a_by_C, b_by_C):
        out = []
        for model, table in ((MODEL_A, a_by_C), (MODEL_B, b_by_C)):
            for C, folds in table.items():
                for fold_name, ll in folds.items():
                    out.append({
                        "model": model, "C": C, "fold_name": fold_name,
                        "log_loss": ll, "brier": 0.6, "macro_f1": 0.39,
                        "balanced_accuracy": 0.44, "accuracy": 0.51,
                    })
        return out

    def _flat(self, value_by_C):
        return {C: {"fold_1": v, "fold_2": v, "fold_3": v} for C, v in value_by_C.items()}

    def test_summary_refuses_incomplete_fold_set(self):
        recs = [{"model": MODEL_A, "C": 1.0, "fold_name": "fold_1", **{m: 0.5 for m in robustness.METRICS}}]
        with self.assertRaises(RuntimeError):
            run_robustness.summarize_by_model_and_C(recs)

    def test_summary_computes_correct_means(self):
        a = self._flat({C: 1.0 for C in robustness.C_GRID})
        b = self._flat({C: 0.99 for C in robustness.C_GRID})
        summary = run_robustness.summarize_by_model_and_C(self._records(a, b))
        self.assertAlmostEqual(summary[MODEL_A]["1.0"]["mean_log_loss"], 1.0)
        self.assertAlmostEqual(summary[MODEL_B]["1.0"]["mean_log_loss"], 0.99)

    def test_analysis_identifies_best_C_and_counts_b_wins(self):
        a = self._flat({C: 1.0 for C in robustness.C_GRID})
        b_vals = {C: 0.99 for C in robustness.C_GRID}
        b_vals[100.0] = 1.05  # B worse at one extreme
        b = self._flat(b_vals)
        summary = run_robustness.summarize_by_model_and_C(self._records(a, b))
        an = run_robustness.analyze_robustness(summary)
        self.assertEqual(an["q4_n_C_values_where_b_beats_a"], len(robustness.C_GRID) - 1)
        self.assertFalse(an["q3_b_beats_a_at_same_C"][100.0])
        self.assertTrue(an["q8_delta_at_v1_C"]["b_better"])

    def test_analysis_flags_best_C_at_grid_edge(self):
        a_vals = {C: 1.0 for C in robustness.C_GRID}
        a_vals[100.0] = 0.5  # best at the top edge
        summary = run_robustness.summarize_by_model_and_C(
            self._records(self._flat(a_vals), self._flat({C: 0.99 for C in robustness.C_GRID}))
        )
        an = run_robustness.analyze_robustness(summary)
        self.assertTrue(an["q10_edge_of_grid_flags"]["model_a_best_at_grid_edge"])

    def test_analysis_reports_delta_range(self):
        a = self._flat({C: 1.0 for C in robustness.C_GRID})
        b_vals = {C: 0.99 for C in robustness.C_GRID}
        b_vals[0.01] = 1.2
        summary = run_robustness.summarize_by_model_and_C(self._records(a, self._flat(b_vals)))
        an = run_robustness.analyze_robustness(summary)
        self.assertAlmostEqual(an["q6_delta_range"]["max_delta"], 0.2, places=9)
        self.assertAlmostEqual(an["q6_delta_range"]["min_delta"], -0.01, places=9)

    def test_analysis_contains_interpretation_guard(self):
        a = self._flat({C: 1.0 for C in robustness.C_GRID})
        b = self._flat({C: 0.99 for C in robustness.C_GRID})
        an = run_robustness.analyze_robustness(run_robustness.summarize_by_model_and_C(self._records(a, b)))
        guard = an["interpretation_guard"].lower()
        self.assertIn("no statistical test", guard)
        self.assertIn("no causal claim", guard)
        self.assertIn("does not authorize changing v1", guard)

    def test_equivalence_check_detects_mismatch(self):
        a = self._flat({C: 5.0 for C in robustness.C_GRID})   # deliberately wrong
        b = self._flat({C: 5.0 for C in robustness.C_GRID})
        summary = run_robustness.summarize_by_model_and_C(self._records(a, b))
        eq = run_robustness.verify_phase4a_equivalence_at_C1(summary)
        self.assertFalse(eq["all_match"])

    def test_equivalence_check_passes_on_exact_phase4a_values(self):
        a = self._flat({C: robustness.PHASE4A_C1_MEAN_LOG_LOSS[MODEL_A] for C in robustness.C_GRID})
        b = self._flat({C: robustness.PHASE4A_C1_MEAN_LOG_LOSS[MODEL_B] for C in robustness.C_GRID})
        summary = run_robustness.summarize_by_model_and_C(self._records(a, b))
        self.assertTrue(run_robustness.verify_phase4a_equivalence_at_C1(summary)["all_match"])


class TestPhase4AReferenceValuesMatchArtifact(unittest.TestCase):
    @unittest.skipUnless(_PHASE4A_JSON.exists(), "Phase 4A artifact not present")
    def test_quoted_phase4a_means_match_the_authoritative_artifact(self):
        d = json.loads(_PHASE4A_JSON.read_text(encoding="utf-8"))
        for model in (MODEL_A, MODEL_B):
            self.assertEqual(
                robustness.PHASE4A_C1_MEAN_LOG_LOSS[model],
                d["model_summaries"][model]["mean_log_loss"],
            )


@unittest.skipUnless(_REAL_DB.exists(), _DATA_SKIP)
class TestAgainstRealSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.data import load_supervised_dataset
        cls.dataset = load_supervised_dataset(_REAL_DB)

    def test_every_model_column_exists_in_the_real_schema(self):
        cols = set(self.dataset.X.columns)
        for name, model_cols in robustness.PHASE4B_MODELS.items():
            missing = [c for c in model_cols if c not in cols]
            self.assertEqual(missing, [], f"{name} missing: {missing}")

    def test_no_model_column_is_all_null_or_constant_in_any_training_fold(self):
        from models.splits import iter_walk_forward_folds
        for fold, train_ds, _ in iter_walk_forward_folds(self.dataset):
            for name, model_cols in robustness.PHASE4B_MODELS.items():
                X = train_ds.X[list(model_cols)]
                all_null = [c for c in model_cols if X[c].isna().all()]
                const = [c for c in model_cols if not X[c].isna().all() and X[c].dropna().nunique() <= 1]
                self.assertEqual(all_null, [], f"{fold.name}/{name} all-null: {all_null}")
                self.assertEqual(const, [], f"{fold.name}/{name} constant: {const}")

    def test_selecting_features_yields_expected_shapes(self):
        from models.splits import iter_walk_forward_folds
        for fold, train_ds, val_ds in iter_walk_forward_folds(self.dataset):
            for name, model_cols in robustness.PHASE4B_MODELS.items():
                self.assertEqual(robustness.select_model_features(train_ds.X, name).shape[1], len(model_cols))
                self.assertEqual(robustness.select_model_features(val_ds.X, name).shape[1], len(model_cols))


@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP)
class TestEstimatorConstruction(unittest.TestCase):
    def test_built_estimator_has_v1_params_plus_swept_C(self):
        for C in robustness.C_GRID:
            model = robustness.build_logistic_regression(C)
            self.assertEqual(model.C, C)
            self.assertEqual(model.max_iter, robustness.V1_LOGREG_BASE_KWARGS["max_iter"])
            self.assertEqual(model.random_state, robustness.V1_LOGREG_BASE_KWARGS["random_state"])

    def test_estimator_at_C1_matches_v1s_own_estimator_params(self):
        from sklearn.linear_model import LogisticRegression
        v1_like = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
        mine = robustness.build_logistic_regression(1.0)
        for p in ("C", "max_iter", "random_state", "solver", "penalty", "tol", "fit_intercept", "class_weight"):
            self.assertEqual(getattr(mine, p), getattr(v1_like, p), f"param {p} differs from V1")


if __name__ == "__main__":
    unittest.main()
