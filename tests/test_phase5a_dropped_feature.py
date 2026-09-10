"""Tests for src/models/run_dropped_feature.py — Phase 5A.

Validates the experiment's frozen design and safety guarantees without
executing it: arm definitions, estimator freeze, fold definitions,
2025/26 unreachability, phase5a-only output paths, write-once refusal,
and the absence of any V2/production side-effect.
"""
import _pathfix  # noqa: F401
import ast
import hashlib
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models import run_dropped_feature as rdf

_SRC = Path("src/models/run_dropped_feature.py")


class TestArmDefinitions(unittest.TestCase):
    def test_exactly_80_and_81_features(self):
        self.assertEqual(len(rdf.MODEL_B_ARM_COLUMNS), 80)
        self.assertEqual(len(rdf.MODEL_B_PLUS_COLUMNS), 81)

    def test_exact_set_difference_is_the_single_feature(self):
        b, p = set(rdf.MODEL_B_ARM_COLUMNS), set(rdf.MODEL_B_PLUS_COLUMNS)
        self.assertEqual(sorted(p - b), ["league_home_advantage_season"])
        self.assertEqual(b - p, set())
        self.assertEqual(len(b & p), 80)

    def test_arm_b_matches_frozen_phase4a_model_b(self):
        from models.ablation import MODEL_B_COLUMNS
        self.assertEqual(rdf.MODEL_B_ARM_COLUMNS, tuple(MODEL_B_COLUMNS))

    def test_dropped_feature_is_a_real_v1_column(self):
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        self.assertIn(rdf.DROPPED_FEATURE, APPROVED_FEATURE_COLUMNS_V1)

    def test_no_duplicates_in_either_arm(self):
        for cols in rdf.ARMS.values():
            self.assertEqual(len(cols), len(set(cols)))

    def test_no_xg_in_either_arm(self):
        from models.ablation import XG_CORE_COLUMNS
        for cols in rdf.ARMS.values():
            self.assertTrue(set(cols).isdisjoint(XG_CORE_COLUMNS))

    def test_no_label_or_bookkeeping_columns(self):
        from models.config import X_EXCLUDED_COLUMNS
        for cols in rdf.ARMS.values():
            self.assertEqual(set(cols) & set(X_EXCLUDED_COLUMNS), set())

    def test_verify_arm_definitions_passes(self):
        result = rdf.verify_arm_definitions()
        self.assertTrue(result["verified"])
        self.assertEqual(result["intersection_size"], 80)
        self.assertEqual(result["b_plus_minus_b"], ["league_home_advantage_season"])

    def test_exactly_two_arms(self):
        self.assertEqual(set(rdf.ARMS), {rdf.ARM_B, rdf.ARM_B_PLUS})


class TestEstimatorFrozen(unittest.TestCase):
    def test_estimator_kwargs_are_v1_literals(self):
        self.assertEqual(rdf.ESTIMATOR_KWARGS, {"max_iter": 2000, "C": 1.0, "random_state": 0})

    def test_C_is_1_point_0(self):
        self.assertEqual(rdf.ESTIMATOR_KWARGS["C"], 1.0)

    def test_matches_v1_estimator_in_train_py(self):
        tree = ast.parse(Path("src/models/train.py").read_text(encoding="utf-8"))
        kwargs = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LogisticRegression":
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        self.assertEqual(rdf.ESTIMATOR_KWARGS, kwargs)

    def test_no_C_grid_or_sweep_in_module(self):
        src = _SRC.read_text(encoding="utf-8")
        for forbidden in ("C_GRID", "for C in", "GridSearch", "RandomizedSearch"):
            self.assertNotIn(forbidden, src)

    def test_calibration_is_none(self):
        src = _SRC.read_text(encoding="utf-8")
        self.assertIn('"calibration": "none"', src)
        for forbidden in ("CalibratedClassifierCV", "Platt", "isotonic"):
            self.assertNotIn(forbidden, src)

    def test_uses_v1_training_path_unmodified(self):
        src = inspect.getsource(rdf.run_arm_on_fold)
        self.assertIn("train_module.train_logistic_regression", src)
        for forbidden in ("SimpleImputer(", "StandardScaler(", "fillna("):
            self.assertNotIn(forbidden, src)


class TestFoldDefinitionsFrozen(unittest.TestCase):
    def test_walk_forward_folds_unchanged(self):
        from models.config import WALK_FORWARD_FOLDS
        actual = [(f.name, f.train_seasons, f.validation_seasons) for f in WALK_FORWARD_FOLDS]
        self.assertEqual(actual, [
            ("fold_1", ("2020/2021", "2021/2022"), ("2022/2023",)),
            ("fold_2", ("2020/2021", "2021/2022", "2022/2023"), ("2023/2024",)),
            ("fold_3", ("2020/2021", "2021/2022", "2022/2023", "2023/2024"), ("2024/2025",)),
        ])

    def test_experiment_uses_only_walk_forward_folds(self):
        src = inspect.getsource(rdf.main)
        self.assertIn("iter_walk_forward_folds", src)


class TestFinalTestUnreachable(unittest.TestCase):
    def test_never_imports_or_calls_final_split(self):
        tree = ast.parse(_SRC.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if "final_split" in a.name:
                        offenders.append(a.name)
            elif isinstance(node, ast.Call):
                fn = node.func
                if (getattr(fn, "id", None) or getattr(fn, "attr", None)) == "final_split":
                    offenders.append("final_split()")
        self.assertEqual(offenders, [])
        self.assertFalse(hasattr(rdf, "final_split"))

    def test_never_references_final_test_season_constants(self):
        tree = ast.parse(_SRC.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        for forbidden in ("FINAL_TEST_SEASONS", "FINAL_TRAIN_SEASONS"):
            self.assertNotIn(forbidden, names)
            self.assertNotIn(forbidden, imported)

    def test_no_public_function_takes_a_final_test_parameter(self):
        for name, fn in vars(rdf).items():
            if not callable(fn) or not getattr(fn, "__module__", "").startswith("models."):
                continue
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                continue
            for p in params:
                self.assertNotIn("final", p.lower())
                self.assertNotIn("test_season", p.lower())

    def test_no_random_split_machinery(self):
        tree = ast.parse(_SRC.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if a.name.split(".")[0] == "random" or "train_test_split" in a.name or "KFold" in a.name:
                        offenders.append(a.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"shuffle", "train_test_split"}:
                    offenders.append(node.func.attr)
        self.assertEqual(offenders, [])


class TestOutputPathsAndWriteOnce(unittest.TestCase):
    def test_all_output_paths_are_phase5a_scoped(self):
        self.assertIn("phase5a", str(rdf.DEFAULT_RESULTS_PATH))
        self.assertIn("phase5a", str(rdf.DEFAULT_PREDICTIONS_DIR))

    def test_never_writes_to_earlier_phase_paths(self):
        src = _SRC.read_text(encoding="utf-8")
        for line in src.splitlines():
            if "write_text" in line or "to_csv" in line:
                for forbidden in ("phase3_", "phase4a_", "phase4b_", "phase4c_", "PHASE5_"):
                    self.assertNotIn(forbidden, line)

    def test_refuse_if_exists_blocks_overwrite(self):
        with self.assertRaises(rdf.ExperimentInvalid):
            rdf._refuse_if_exists(Path("data/audit/phase3_model_comparison.json"))

    def test_refuse_if_exists_allows_new_path(self):
        rdf._refuse_if_exists(Path("data/audit/__phase5a_not_present__.json"))

    def test_main_refuses_when_results_exist(self):
        self.assertIn("_refuse_if_exists(results_path)", inspect.getsource(rdf.main))


class TestSummaryAndClassificationLogic(unittest.TestCase):
    def _summary(self, b_by_fold, p_by_fold):
        def entry(vals):
            e = {"n_folds": 3}
            for m in rdf.METRICS:
                e[f"mean_{m}"] = float(np.mean(list(vals.values())))
                e[f"by_fold_{m}"] = dict(vals)
            return e
        return {rdf.ARM_B: entry(b_by_fold), rdf.ARM_B_PLUS: entry(p_by_fold)}

    def test_case_a_improves_and_consistent(self):
        s = self._summary({"fold_1": 1.0, "fold_2": 1.0, "fold_3": 1.0},
                          {"fold_1": 0.99, "fold_2": 0.98, "fold_3": 0.97})
        c = rdf.classify_result(s)
        self.assertEqual(c["case"], "CASE_A_improves_and_consistent")
        self.assertEqual(c["conclusion"], "RETAIN_OMISSION_NOT_SUPPORTED")

    def test_case_b_worsens(self):
        s = self._summary({"fold_1": 1.0, "fold_2": 1.0, "fold_3": 1.0},
                          {"fold_1": 1.01, "fold_2": 1.02, "fold_3": 1.03})
        c = rdf.classify_result(s)
        self.assertEqual(c["case"], "CASE_B_worsens_mean_log_loss")
        self.assertEqual(c["conclusion"], "RETAIN_OMISSION_SUPPORTED")

    def test_case_c_mean_improves_but_inconsistent(self):
        s = self._summary({"fold_1": 1.0, "fold_2": 1.0, "fold_3": 1.0},
                          {"fold_1": 0.90, "fold_2": 1.02, "fold_3": 1.02})
        c = rdf.classify_result(s)
        self.assertEqual(c["case"], "CASE_C_mean_improves_but_inconsistent_across_folds")
        self.assertEqual(c["conclusion"], "FEATURE_EFFECT_INCONCLUSIVE")

    def test_only_permitted_conclusions_are_possible(self):
        allowed = {"RETAIN_OMISSION_SUPPORTED", "RETAIN_OMISSION_NOT_SUPPORTED", "FEATURE_EFFECT_INCONCLUSIVE"}
        src = _SRC.read_text(encoding="utf-8")
        import re
        found = set(re.findall(r'"(RETAIN_OMISSION_\w+|FEATURE_EFFECT_\w+)"', src))
        self.assertTrue(found.issubset(allowed), f"unexpected conclusion label: {found - allowed}")

    def test_classification_carries_interpretation_guard(self):
        s = self._summary({"fold_1": 1.0, "fold_2": 1.0, "fold_3": 1.0},
                          {"fold_1": 0.99, "fold_2": 0.99, "fold_3": 0.99})
        guard = rdf.classify_result(s)["interpretation_guard"].lower()
        self.assertIn("no statistical test", guard)
        self.assertIn("no causal claim", guard)
        self.assertIn("does not create v2", guard)

    def test_summary_refuses_incomplete_fold_set(self):
        recs = [{"arm": rdf.ARM_B, "fold_name": "fold_1", **{m: 1.0 for m in rdf.METRICS}}]
        with self.assertRaises(rdf.ExperimentInvalid):
            rdf.summarize(recs)

    def test_delta_direction_respects_metric_polarity(self):
        b = {m: 1.0 for m in rdf.METRICS}
        p = {"log_loss": 0.9, "brier": 0.9, "accuracy": 1.1, "macro_f1": 1.1, "balanced_accuracy": 1.1}
        d = rdf.compute_fold_deltas(b, p)
        for m in rdf.METRICS:
            self.assertTrue(d[m]["b_plus_better"], f"{m} polarity wrong")


class TestNoProductionSideEffects(unittest.TestCase):
    def test_model_version_unchanged(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_module_never_writes_model_version(self):
        src = _SRC.read_text(encoding="utf-8")
        self.assertNotIn("MODEL_VERSION =", src)

    def test_no_v2_module_or_artifact(self):
        for p in Path("src/models").glob("*.py"):
            self.assertNotIn("v2", p.stem.lower())
        for p in Path("data/audit").glob("*"):
            self.assertNotIn("v2", p.name.lower())

    def test_locked_artifacts_unchanged(self):
        expected = {
            "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
            "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
            "data/audit/phase3_model_comparison.json": "616279914b2730749d52eae15b5f96b9",
            "data/audit/phase4a_ablation_comparison.json": "075b0686bce20bfce9f7289fa37076c0",
            "data/audit/phase4b_robustness_comparison.json": "97d0f8b8674c9bf8598b6c3d6b7c825c",
            "data/audit/phase4c_final_comparison.json": "effd9e54130b2bc5aaf51cd643962396",
            "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
            "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
            "src/models/ablation.py": "9bf8bf4a1f332f8bb47d640a72384ed7",
        }
        for rel, exp in expected.items():
            actual = hashlib.md5(Path(rel).read_bytes()).hexdigest()
            self.assertEqual(actual, exp, f"locked artifact modified: {rel}")


if __name__ == "__main__":
    unittest.main()
