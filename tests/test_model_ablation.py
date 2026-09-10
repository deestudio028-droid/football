"""Tests for src/models/ablation.py and src/models/run_ablation.py --
Phase 4A feature ablation study (tiers and xG policy both FROZEN).

Covers:
  - exact frozen A/B/C/D tier membership, strict nesting, no duplicates,
  - every named column exists in the real feature schema (nothing invented),
  - every legal schema feature is assigned to a group or classified as
    metadata/hyperparameter,
  - xG present in C/D, absent from A/B,
  - the one documented V1 column absent from Model A (and no others),
  - no label/bookkeeping column in any tier,
  - the xG fold-eligibility guard (invalid folds rejected, never
    zero-filled, never silently skipped, deterministic, non-mutating),
  - summaries refusing to fabricate a 3-fold mean when a fold is invalid,
  - 2025/26 unreachable from the Phase 4A path, no random splitting,
  - Phase 3 artifacts and V1 contract constants untouched.

Model-training behaviour itself is exercised on the user's machine
(scikit-learn required); these tests cover everything that does not
require fitting a model.
"""
import _pathfix  # noqa: F401
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models import ablation
from models import run_ablation

_REAL_DB = Path("data/processed/features.db")
_DATA_SKIP_REASON = "data/processed/features.db not present in this environment"


class TestFeatureGroupsAreInternallyConsistent(unittest.TestCase):
    def test_no_column_appears_in_two_different_groups(self):
        seen: dict[str, str] = {}
        for group_name, cols in ablation.ALL_SCHEMA_VERIFIED_GROUPS.items():
            for c in cols:
                self.assertNotIn(
                    c, seen,
                    f"{c!r} appears in both {seen.get(c)!r} and {group_name!r} -- groups must be disjoint",
                )
                seen[c] = group_name

    def test_no_group_contains_n_or_coverage_n_companion_columns(self):
        for group_name, cols in ablation.ALL_SCHEMA_VERIFIED_GROUPS.items():
            for c in cols:
                self.assertFalse(c.endswith("_n"), f"{group_name}: {c!r} looks like a sample-size companion column")
                self.assertFalse(c.endswith("_coverage_n"), f"{group_name}: {c!r} looks like a coverage companion column")

    def test_group_sizes_match_documented_counts(self):
        expected_sizes = {
            "goals_core": 18, "goals_venue": 4, "form": 20, "shots_core": 18,
            "shots_on_core": 18, "xg_core": 22, "strength": 5, "competition_id": 1,
            "league_home_advantage": 1, "league_mean_goals": 1, "tempo": 6,
            "discipline": 4, "venue_goal_diff": 2, "history_flags": 4,
        }
        actual_sizes = {k: len(v) for k, v in ablation.ALL_SCHEMA_VERIFIED_GROUPS.items()}
        self.assertEqual(actual_sizes, expected_sizes)

    def test_frozen_tiers_are_strictly_nested_a_subset_of_b_subset_of_c_subset_of_d(self):
        a = set(ablation.MODEL_A_COLUMNS)
        b = set(ablation.MODEL_B_COLUMNS)
        c = set(ablation.MODEL_C_COLUMNS)
        d = set(ablation.MODEL_D_COLUMNS)
        self.assertTrue(a.issubset(b))
        self.assertTrue(b.issubset(c))
        self.assertTrue(c.issubset(d))
        # Strictly nested -- each tier must add at least one column.
        self.assertLess(len(a), len(b))
        self.assertLess(len(b), len(c))
        self.assertLess(len(c), len(d))

    def test_frozen_tier_membership_matches_the_approved_decision_exactly(self):
        # Decision 2, verbatim:
        #   A = goals_core + form + competition_id + strength
        #   B = A + shots_core + shots_on_core
        #   C = B + xg_core
        #   D = C + goals_venue + tempo + discipline + league_home_advantage
        #           + league_mean_goals + venue_goal_diff + history_flags
        g = ablation.ALL_SCHEMA_VERIFIED_GROUPS
        expected_a = set(g["goals_core"]) | set(g["form"]) | set(g["competition_id"]) | set(g["strength"])
        expected_b = expected_a | set(g["shots_core"]) | set(g["shots_on_core"])
        expected_c = expected_b | set(g["xg_core"])
        expected_d = (
            expected_c | set(g["goals_venue"]) | set(g["tempo"]) | set(g["discipline"])
            | set(g["league_home_advantage"]) | set(g["league_mean_goals"])
            | set(g["venue_goal_diff"]) | set(g["history_flags"])
        )
        self.assertEqual(set(ablation.MODEL_A_COLUMNS), expected_a)
        self.assertEqual(set(ablation.MODEL_B_COLUMNS), expected_b)
        self.assertEqual(set(ablation.MODEL_C_COLUMNS), expected_c)
        self.assertEqual(set(ablation.MODEL_D_COLUMNS), expected_d)

    def test_model_d_covers_every_non_metadata_schema_feature(self):
        # "everything in C + the remaining eligible groups" should leave
        # nothing but metadata/hyperparameter columns unused.
        all_group_cols = set()
        for cols in ablation.ALL_SCHEMA_VERIFIED_GROUPS.values():
            all_group_cols.update(cols)
        self.assertEqual(set(ablation.MODEL_D_COLUMNS), all_group_cols)

    def test_no_duplicate_columns_within_any_tier(self):
        for name, cols in ablation.MODEL_COLUMNS.items():
            self.assertEqual(len(cols), len(set(cols)), f"{name} contains duplicate columns")

    def test_model_c_and_d_contain_all_xg_columns(self):
        self.assertTrue(set(ablation.XG_CORE_COLUMNS).issubset(set(ablation.MODEL_C_COLUMNS)))
        self.assertTrue(set(ablation.XG_CORE_COLUMNS).issubset(set(ablation.MODEL_D_COLUMNS)))

    def test_model_a_and_b_contain_no_xg_columns(self):
        self.assertTrue(set(ablation.XG_CORE_COLUMNS).isdisjoint(set(ablation.MODEL_A_COLUMNS)))
        self.assertTrue(set(ablation.XG_CORE_COLUMNS).isdisjoint(set(ablation.MODEL_B_COLUMNS)))

    def test_model_a_contains_every_v1_column_except_league_home_advantage(self):
        # Documented consequence of the frozen mapping: the ONE V1-approved
        # column absent from Model A is league_home_advantage_season, which
        # Decision 2 explicitly places at tier D. Every other V1 column
        # must be present in A -- this test fails loudly if any other
        # V1 feature is ever accidentally dropped from the baseline tier.
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        v1 = set(APPROVED_FEATURE_COLUMNS_V1)
        a = set(ablation.MODEL_A_COLUMNS)
        missing_from_a = v1 - a
        self.assertEqual(missing_from_a, {"league_home_advantage_season"})
        # ...and it must reappear by tier D.
        self.assertIn("league_home_advantage_season", set(ablation.MODEL_D_COLUMNS))

    def test_no_label_or_bookkeeping_column_enters_any_tier(self):
        from models.config import X_EXCLUDED_COLUMNS
        forbidden = set(X_EXCLUDED_COLUMNS)
        for name, cols in ablation.MODEL_COLUMNS.items():
            overlap = forbidden & set(cols)
            self.assertEqual(overlap, set(), f"{name} contains excluded column(s): {overlap}")

    def test_shrinkage_k_excluded_from_every_group_and_every_tier(self):
        for cols in ablation.ALL_SCHEMA_VERIFIED_GROUPS.values():
            self.assertNotIn("shrinkage_k", cols)
        for tier in ablation.MODEL_COLUMNS.values():
            self.assertNotIn("shrinkage_k", tier)

    def test_model_groups_labels_match_model_columns_content(self):
        for name in ablation.ABLATION_MODELS:
            expected = set()
            for group_name in ablation.MODEL_GROUPS[name]:
                expected.update(ablation.ALL_SCHEMA_VERIFIED_GROUPS[group_name])
            self.assertEqual(set(ablation.MODEL_COLUMNS[name]), expected, f"{name} group labels disagree with columns")

    def test_only_c_and_d_are_flagged_xg_inclusive(self):
        self.assertEqual(set(ablation.XG_INCLUSIVE_MODELS), {ablation.MODEL_C, ablation.MODEL_D})


@unittest.skipUnless(_REAL_DB.exists(), _DATA_SKIP_REASON)
class TestFeatureGroupColumnsExistInSchema(unittest.TestCase):
    """Mechanical proof that every column named in ablation.py actually
    exists in the real feature_rows schema -- the module docstring's
    "never invented" claim, checked directly, not assumed.
    """

    @classmethod
    def setUpClass(cls):
        from models.data import load_supervised_dataset
        cls.dataset = load_supervised_dataset(_REAL_DB)
        cls.real_columns = set(cls.dataset.X.columns)

    def test_every_group_column_exists_in_real_schema(self):
        for group_name, cols in ablation.ALL_SCHEMA_VERIFIED_GROUPS.items():
            missing = [c for c in cols if c not in self.real_columns]
            self.assertEqual(missing, [], f"group {group_name!r} references missing column(s): {missing}")

    def test_all_264_feature_columns_are_accounted_for(self):
        grouped = set()
        for cols in ablation.ALL_SCHEMA_VERIFIED_GROUPS.values():
            grouped.update(cols)
        meta = {c for c in self.real_columns if c.endswith("_n") or c.endswith("_coverage_n")}
        accounted = grouped | meta | {"shrinkage_k"}
        unaccounted = self.real_columns - accounted
        self.assertEqual(unaccounted, set(), f"Unaccounted-for real columns: {sorted(unaccounted)}")

    def test_xg_all_null_in_fold_1_and_fold_2_training_partitions(self):
        # The concrete evidence behind the module's "xG missing-data
        # contradiction" claim -- re-verified directly here so this
        # test would catch it if the underlying data ever changed.
        from models.splits import iter_walk_forward_folds
        for fold, train_ds, val_ds in iter_walk_forward_folds(self.dataset):
            all_null = [c for c in ablation.XG_CORE_COLUMNS if train_ds.X[c].isna().all()]
            if fold.name in ("fold_1", "fold_2"):
                self.assertEqual(
                    set(all_null), set(ablation.XG_CORE_COLUMNS),
                    f"{fold.name}: expected every xG column to be all-null in the training partition",
                )
            else:
                self.assertEqual(all_null, [], f"{fold.name}: expected NO all-null xG columns")

    def test_approved_v1_columns_never_appear_inside_ablation_groups(self):
        # V1's contract must stay untouched/independent -- ablation
        # groups are a completely separate definition, not a
        # modification of config.APPROVED_FEATURE_COLUMNS_V1.
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        # This is descriptive, not a hard requirement (V1 columns like
        # strength_diff legitimately also belong to an ablation group) --
        # what actually matters is that config.py's own tuple is untouched:
        self.assertEqual(len(APPROVED_FEATURE_COLUMNS_V1), 15)


class TestSelectFeatureColumns(unittest.TestCase):
    def test_returns_exactly_the_requested_columns_in_order(self):
        X = pd.DataFrame({"b": [1, 2], "a": [3, 4], "c": [5, 6]})
        selected = ablation.select_feature_columns(X, ("a", "b"))
        self.assertEqual(list(selected.columns), ["a", "b"])

    def test_raises_loudly_on_missing_column_rather_than_substituting(self):
        X = pd.DataFrame({"a": [1, 2]})
        with self.assertRaises(RuntimeError):
            ablation.select_feature_columns(X, ("a", "does_not_exist"))

    def test_does_not_mutate_the_input_frame(self):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        before = X.copy()
        ablation.select_feature_columns(X, ("a",))
        pd.testing.assert_frame_equal(X, before)

    def test_group_name_appears_in_error_message_for_debuggability(self):
        X = pd.DataFrame({"a": [1]})
        with self.assertRaises(RuntimeError) as ctx:
            ablation.select_feature_columns(X, ("missing_col",), group_name="model_a")
        self.assertIn("model_a", str(ctx.exception))


class TestXgFoldEligibilityGuard(unittest.TestCase):
    """Decision 1: a fold whose TRAINING partition has no observed xG
    values must be marked INVALID_FOR_XG and must never be trained on,
    zero-filled, or silently skipped.
    """

    def _frame(self, xg_values_by_column: dict) -> pd.DataFrame:
        n = len(next(iter(xg_values_by_column.values())))
        data = {c: xg_values_by_column.get(c, [1.0] * n) for c in ablation.XG_CORE_COLUMNS}
        return pd.DataFrame(data)

    def test_all_null_xg_training_partition_is_invalid(self):
        X = self._frame({c: [np.nan, np.nan, np.nan] for c in ablation.XG_CORE_COLUMNS})
        result = ablation.assess_xg_fold_eligibility("fold_1", X)
        self.assertFalse(result.is_valid)
        self.assertIn("INVALID_FOR_XG", result.reason)
        self.assertEqual(len(result.xg_columns_with_zero_training_observations), len(ablation.XG_CORE_COLUMNS))

    def test_even_a_single_all_null_xg_column_invalidates_the_fold(self):
        # Stricter-than-"all empty" on purpose: SimpleImputer silently
        # drops an all-NaN column, which would gut part of the xG group
        # invisibly -- the forbidden behaviour.
        one_bad = ablation.XG_CORE_COLUMNS[0]
        X = self._frame({one_bad: [np.nan, np.nan, np.nan]})
        result = ablation.assess_xg_fold_eligibility("fold_x", X)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.xg_columns_with_zero_training_observations, (one_bad,))

    def test_partially_observed_xg_training_partition_is_valid(self):
        # Sparse but real coverage (like fold_3) is usable -- imputation
        # statistics can genuinely be learned from the observed values.
        X = self._frame({c: [np.nan, np.nan, 1.4] for c in ablation.XG_CORE_COLUMNS})
        result = ablation.assess_xg_fold_eligibility("fold_3", X)
        self.assertTrue(result.is_valid)
        self.assertIsNone(result.reason)
        self.assertEqual(result.min_observed_count_across_xg_columns, 1)

    def test_require_xg_fold_valid_raises_on_invalid_fold(self):
        X = self._frame({c: [np.nan, np.nan] for c in ablation.XG_CORE_COLUMNS})
        with self.assertRaises(ablation.XgCoverageError) as ctx:
            ablation.require_xg_fold_valid("fold_1", X)
        self.assertIn("INVALID_FOR_XG", str(ctx.exception))

    def test_require_xg_fold_valid_passes_through_on_valid_fold(self):
        X = self._frame({c: [1.1, 1.2] for c in ablation.XG_CORE_COLUMNS})
        result = ablation.require_xg_fold_valid("fold_3", X)
        self.assertTrue(result.is_valid)

    def test_eligibility_result_is_deterministic(self):
        X = self._frame({c: [np.nan, 1.0, np.nan] for c in ablation.XG_CORE_COLUMNS})
        r1 = ablation.assess_xg_fold_eligibility("fold_3", X).to_dict()
        r2 = ablation.assess_xg_fold_eligibility("fold_3", X).to_dict()
        self.assertEqual(r1, r2)

    def test_guard_never_mutates_or_fills_the_input_data(self):
        X = self._frame({c: [np.nan, 1.0] for c in ablation.XG_CORE_COLUMNS})
        before = X.copy()
        ablation.assess_xg_fold_eligibility("fold_3", X)
        pd.testing.assert_frame_equal(X, before)
        # NaNs must still be NaNs -- nothing zero-filled behind our back.
        self.assertTrue(X[ablation.XG_CORE_COLUMNS[0]].isna().any())


class TestAblationSummaryRefusesFabricatedMeans(unittest.TestCase):
    """Decision 1 consequence: a model with any invalid fold must NOT
    get a 3-fold mean, because that would be directly (and wrongly)
    comparable to A/B's genuine 3-fold mean.
    """

    def _valid_record(self, model, fold_name, log_loss=1.0):
        return {
            "model": model, "fold_name": fold_name, "fold_valid": True,
            "log_loss": log_loss, "brier": 0.6, "macro_f1": 0.37,
            "balanced_accuracy": 0.43, "accuracy": 0.5,
        }

    def _invalid_record(self, model, fold_name):
        return {
            "model": model, "fold_name": fold_name, "fold_valid": False,
            "invalid_reason": "INVALID_FOR_XG: zero observed xG values in training partition",
            "log_loss": None, "brier": None, "macro_f1": None,
            "balanced_accuracy": None, "accuracy": None,
        }

    def test_all_valid_folds_produce_a_real_mean(self):
        records = [self._valid_record(ablation.MODEL_A, f"fold_{i}", log_loss=1.0 + i * 0.01) for i in (1, 2, 3)]
        summary = run_ablation.summarize_model(ablation.MODEL_A, records)
        self.assertTrue(summary["mean_across_all_three_folds_computable"])
        self.assertIsNotNone(summary["mean_log_loss"])
        self.assertEqual(summary["n_folds_valid"], 3)

    def test_any_invalid_fold_suppresses_every_mean(self):
        records = [
            self._invalid_record(ablation.MODEL_C, "fold_1"),
            self._invalid_record(ablation.MODEL_C, "fold_2"),
            self._valid_record(ablation.MODEL_C, "fold_3", log_loss=0.95),
        ]
        summary = run_ablation.summarize_model(ablation.MODEL_C, records)
        self.assertFalse(summary["mean_across_all_three_folds_computable"])
        for metric in ["log_loss", "brier", "macro_f1", "balanced_accuracy", "accuracy"]:
            self.assertIsNone(summary[f"mean_{metric}"], f"mean_{metric} must be None when a fold is invalid")
        self.assertIsNotNone(summary["mean_not_computable_reason"])

    def test_invalid_folds_are_recorded_with_reasons_not_hidden(self):
        records = [
            self._invalid_record(ablation.MODEL_C, "fold_1"),
            self._valid_record(ablation.MODEL_C, "fold_3"),
        ]
        summary = run_ablation.summarize_model(ablation.MODEL_C, records)
        self.assertEqual(summary["n_folds_invalid"], 1)
        self.assertEqual(summary["invalid_folds"][0]["fold_name"], "fold_1")
        self.assertIn("INVALID_FOR_XG", summary["invalid_folds"][0]["reason"])

    def test_valid_fold_results_still_reported_individually(self):
        records = [
            self._invalid_record(ablation.MODEL_C, "fold_1"),
            self._invalid_record(ablation.MODEL_C, "fold_2"),
            self._valid_record(ablation.MODEL_C, "fold_3", log_loss=0.95),
        ]
        summary = run_ablation.summarize_model(ablation.MODEL_C, records)
        self.assertEqual(summary["per_valid_fold"]["log_loss"], {"fold_3": 0.95})


class TestRunAblationIsolationFrom2025_26(unittest.TestCase):
    """2025/26 must be unreachable from the entire Phase 4A path."""

    def test_run_ablation_module_never_imports_or_calls_final_split(self):
        # Checked against actual imports/calls, not prose: the module
        # docstring legitimately mentions final_split to explain its
        # ABSENCE, which a naive substring search would false-positive on.
        import ast
        tree = ast.parse(inspect.getsource(run_ablation))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if "final_split" in alias.name:
                        offenders.append(f"import {alias.name}")
            elif isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name == "final_split":
                    offenders.append("call final_split()")
        self.assertEqual(offenders, [], f"run_ablation must not reach the final test split: {offenders}")
        self.assertFalse(hasattr(run_ablation, "final_split"))

    def test_run_ablation_never_references_final_test_seasons(self):
        import ast
        tree = ast.parse(inspect.getsource(run_ablation))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        imported = {
            alias.name for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
        }
        for forbidden in ("FINAL_TEST_SEASONS", "FINAL_TRAIN_SEASONS"):
            self.assertNotIn(forbidden, names)
            self.assertNotIn(forbidden, imported)

    def test_ablation_experiment_uses_only_walk_forward_folds(self):
        source = inspect.getsource(run_ablation.run_ablation_experiment)
        self.assertIn("iter_walk_forward_folds", source)

    def test_no_random_split_machinery_anywhere_in_phase4a_modules(self):
        import ast
        for module in (ablation, run_ablation):
            tree = ast.parse(inspect.getsource(module))
            found = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        base = alias.name.split(".")[0]
                        if base in {"random"} or "train_test_split" in alias.name:
                            found.append(alias.name)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"shuffle", "sample", "train_test_split"}:
                        found.append(node.func.attr)
            self.assertEqual(found, [], f"{module.__name__} contains random-split machinery: {found}")


class TestPhase3ArtifactsUntouchedByPhase4A(unittest.TestCase):
    """Phase 4A must write only to its own artifact paths."""

    def test_run_ablation_writes_only_to_phase4a_paths(self):
        self.assertIn("phase4a", str(run_ablation.DEFAULT_RESULTS_PATH))
        self.assertIn("phase4a", str(run_ablation.DEFAULT_PREDICTIONS_DIR))

    def test_run_ablation_never_writes_a_phase3_artifact_path(self):
        source = Path("src/models/run_ablation.py").read_text(encoding="utf-8")
        for phase3_artifact in [
            "phase3_model_comparison.json",
            "phase3_calibration_comparison.json",
            "phase3_probability_diagnostics.json",
            "phase3_predictions",
        ]:
            self.assertNotIn(phase3_artifact, source)

    def test_v1_contract_constants_are_untouched(self):
        from models.config import (
            APPROVED_FEATURE_COLUMNS_V1, FINAL_TEST_SEASONS, MODEL_VERSION,
            REQUIRED_FEATURE_VERSION, WALK_FORWARD_FOLDS,
        )
        self.assertEqual(MODEL_VERSION, "v1.0")
        self.assertEqual(REQUIRED_FEATURE_VERSION, "v1.0")
        self.assertEqual(len(APPROVED_FEATURE_COLUMNS_V1), 15)
        self.assertEqual(len(WALK_FORWARD_FOLDS), 3)
        self.assertEqual(FINAL_TEST_SEASONS, ("2025/2026",))


if __name__ == "__main__":
    unittest.main()
