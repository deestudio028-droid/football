"""Gate 1 — V2 candidate feature contract tests.

Verifies Gate 1's acceptance condition mechanically ("exact set and order
match; both conditions closed in writing") and the governance invariants
that must survive this stage. Trains nothing, loads no 2025/26 data,
creates no V2 artifact.
"""
import _pathfix  # noqa: F401
import ast
import hashlib
import inspect
import json
import re
import unittest
from pathlib import Path

import pandas as pd

from models import candidate_contract as cc
from models.ablation import MODEL_B_COLUMNS, XG_CORE_COLUMNS
from models.config import APPROVED_FEATURE_COLUMNS_V1, X_EXCLUDED_COLUMNS

_SRC = Path("src/models/candidate_contract.py")
_REVIEW5 = Path("docs/PHASE5_V2_CANDIDATE_REVIEW.md")
_REVIEW5B = Path("docs/PHASE5B_DRAW_LIMITATION_REVIEW.md")
_P5A = Path("data/audit/phase5a_dropped_feature_comparison.json")


class TestGate1ExactContract(unittest.TestCase):
    """Gate 1 pass condition: exact set AND order match."""

    def test_contract_has_exactly_80_columns(self):
        self.assertEqual(len(cc.CANDIDATE_FEATURE_COLUMNS), 80)
        self.assertEqual(cc.CANDIDATE_N_FEATURES, 80)

    def test_contract_order_matches_model_b_exactly(self):
        # Order is part of the contract, so this is tuple equality, not
        # set equality.
        self.assertEqual(tuple(cc.CANDIDATE_FEATURE_COLUMNS), tuple(MODEL_B_COLUMNS))

    def test_contract_set_matches_model_b_exactly(self):
        self.assertEqual(set(cc.CANDIDATE_FEATURE_COLUMNS), set(MODEL_B_COLUMNS))

    def test_contract_is_an_immutable_tuple(self):
        self.assertIsInstance(cc.CANDIDATE_FEATURE_COLUMNS, tuple)

    def test_no_duplicate_columns(self):
        self.assertEqual(len(cc.CANDIDATE_FEATURE_COLUMNS), len(set(cc.CANDIDATE_FEATURE_COLUMNS)))

    def test_verify_feature_contract_passes(self):
        r = cc.verify_feature_contract()
        self.assertTrue(r["verified"])
        self.assertTrue(r["matches_model_b_exactly"])
        self.assertTrue(r["order_preserved"])
        self.assertEqual(r["n_features"], 80)

    def test_family_counts_match_spec(self):
        self.assertEqual(cc.CANDIDATE_FAMILY_COUNTS,
                         {"goals_core": 18, "form": 20, "strength": 5,
                          "competition_id": 1, "shots_core": 18, "shots_on_core": 18})
        self.assertEqual(sum(cc.CANDIDATE_FAMILY_COUNTS.values()), 80)

    def test_contract_does_not_restate_columns_as_a_second_literal_list(self):
        # Two sources of truth could silently diverge; the contract must
        # derive from ablation.MODEL_B_COLUMNS.
        src = _SRC.read_text(encoding="utf-8")
        self.assertIn("tuple(MODEL_B_COLUMNS)", src)


class TestGate1StopConditions(unittest.TestCase):
    """Gate 1 stop condition: any column added/removed/reordered."""

    def _restore(self, original):
        cc.CANDIDATE_FEATURE_COLUMNS = original

    def test_reordering_is_detected(self):
        original = cc.CANDIDATE_FEATURE_COLUMNS
        try:
            cc.CANDIDATE_FEATURE_COLUMNS = tuple(reversed(original))
            with self.assertRaises(cc.FeatureContractViolation) as ctx:
                cc.verify_feature_contract()
            self.assertIn("ORDER differs", str(ctx.exception))
        finally:
            self._restore(original)

    def test_added_column_is_detected(self):
        original = cc.CANDIDATE_FEATURE_COLUMNS
        try:
            cc.CANDIDATE_FEATURE_COLUMNS = original + ("some_extra_column",)
            with self.assertRaises(cc.FeatureContractViolation):
                cc.verify_feature_contract()
        finally:
            self._restore(original)

    def test_removed_column_is_detected(self):
        original = cc.CANDIDATE_FEATURE_COLUMNS
        try:
            cc.CANDIDATE_FEATURE_COLUMNS = original[:-1]
            with self.assertRaises(cc.FeatureContractViolation):
                cc.verify_feature_contract()
        finally:
            self._restore(original)

    def test_reintroducing_the_omitted_v1_feature_is_detected(self):
        # Condition 1 (RETAIN_OMISSION_SUPPORTED) must be enforced by
        # code, not just documented.
        original = cc.CANDIDATE_FEATURE_COLUMNS
        try:
            cc.CANDIDATE_FEATURE_COLUMNS = original + (cc.OMITTED_V1_FEATURE,)
            with self.assertRaises(cc.FeatureContractViolation) as ctx:
                cc.verify_feature_contract()
            self.assertIn("Condition 1", str(ctx.exception))
        finally:
            self._restore(original)


class TestGate1ExclusionsHold(unittest.TestCase):
    def test_omitted_v1_feature_is_absent(self):
        self.assertNotIn(cc.OMITTED_V1_FEATURE, cc.CANDIDATE_FEATURE_COLUMNS)
        self.assertEqual(cc.OMITTED_V1_FEATURE, "league_home_advantage_season")

    def test_no_xg_columns(self):
        self.assertTrue(set(cc.CANDIDATE_FEATURE_COLUMNS).isdisjoint(XG_CORE_COLUMNS))

    def test_no_label_or_bookkeeping_columns(self):
        self.assertEqual(set(cc.CANDIDATE_FEATURE_COLUMNS) & set(X_EXCLUDED_COLUMNS), set())

    def test_candidate_is_not_a_strict_superset_of_v1(self):
        v1, cand = set(APPROVED_FEATURE_COLUMNS_V1), set(cc.CANDIDATE_FEATURE_COLUMNS)
        self.assertFalse(v1 < cand)
        self.assertEqual(v1 - cand, {"league_home_advantage_season"})
        self.assertEqual(len(v1 & cand), 14)

    def test_v1_contract_itself_is_untouched(self):
        self.assertEqual(len(APPROVED_FEATURE_COLUMNS_V1), 15)
        self.assertIn("league_home_advantage_season", APPROVED_FEATURE_COLUMNS_V1)


class TestGate1ConditionsClosedInWriting(unittest.TestCase):
    """Gate 1 also requires both conditions closed in writing."""

    def test_condition_closures_recorded_in_code(self):
        self.assertEqual(cc.CONDITION_CLOSURES["condition_1"], "RETAIN_OMISSION_SUPPORTED")
        self.assertEqual(cc.CONDITION_CLOSURES["condition_2"], "DRAW_LIMITATION_ACCEPTED")

    @unittest.skipUnless(_REVIEW5.exists(), "Phase 5 review absent")
    def test_condition_1_closed_in_writing(self):
        text = _REVIEW5.read_text(encoding="utf-8")
        self.assertIn("Condition 1 — RESOLVED", text)
        self.assertIn("RETAIN_OMISSION_SUPPORTED", text)

    @unittest.skipUnless(_REVIEW5B.exists(), "Phase 5B review absent")
    def test_condition_2_closed_in_writing(self):
        text = _REVIEW5B.read_text(encoding="utf-8")
        self.assertIn("DRAW_LIMITATION_ACCEPTED", text)

    @unittest.skipUnless(_P5A.exists(), "Phase 5A artifact absent")
    def test_code_closures_match_the_authoritative_phase5a_artifact(self):
        d = json.loads(_P5A.read_text(encoding="utf-8"))
        self.assertEqual(cc.CONDITION_CLOSURES["condition_1"], d["classification"]["conclusion"])


class TestGate1SelectionBehaviour(unittest.TestCase):
    def test_selection_preserves_contract_order(self):
        frame = pd.DataFrame({c: [1.0] for c in reversed(cc.CANDIDATE_FEATURE_COLUMNS)})
        out = cc.select_candidate_features(frame)
        self.assertEqual(list(out.columns), list(cc.CANDIDATE_FEATURE_COLUMNS))

    def test_selection_fails_loudly_on_missing_column(self):
        frame = pd.DataFrame({c: [1.0] for c in cc.CANDIDATE_FEATURE_COLUMNS[:-1]})
        with self.assertRaises(RuntimeError):
            cc.select_candidate_features(frame)

    def test_selection_drops_extra_columns_not_in_contract(self):
        data = {c: [1.0] for c in cc.CANDIDATE_FEATURE_COLUMNS}
        data["league_home_advantage_season"] = [1.0]
        data["home_xg_for_per_match_season"] = [1.0]
        out = cc.select_candidate_features(pd.DataFrame(data))
        self.assertEqual(len(out.columns), 80)
        self.assertNotIn("league_home_advantage_season", out.columns)
        self.assertNotIn("home_xg_for_per_match_season", out.columns)

    def test_selection_does_not_mutate_input(self):
        frame = pd.DataFrame({c: [1.0] for c in cc.CANDIDATE_FEATURE_COLUMNS})
        before = frame.copy()
        cc.select_candidate_features(frame)
        pd.testing.assert_frame_equal(frame, before)


class TestGate1IsStructuralOnly(unittest.TestCase):
    """Gate 1 must not train, evaluate, calibrate, tune, or touch 2025/26."""

    def test_module_never_imports_or_instantiates_an_estimator(self):
        # AST-based. Originally a substring search, which now
        # false-positives on prose such as
        # "LogisticRegressionPreprocessor" in the Gate 8 docstring. The
        # invariant that still matters: this module imports no ML library
        # and constructs no estimator.
        tree = ast.parse(_SRC.read_text(encoding="utf-8"))
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                imported.add(n.module or "")
                imported.update(a.name for a in n.names)
        for forbidden in ("sklearn", "sklearn.linear_model", "LogisticRegression",
                          "HistGradientBoostingClassifier"):
            self.assertNotIn(forbidden, imported)
        constructed = [
            getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)
        ]
        for forbidden in ("LogisticRegression", "HistGradientBoostingClassifier",
                          "SimpleImputer", "StandardScaler"):
            self.assertNotIn(forbidden, constructed)

    def test_module_never_fits_anything(self):
        # Premise update: Gate 8 added an authorized *inference* boundary
        # (`predict_candidate`), so "never predicts" no longer holds and
        # was superseded by explicit authorization. What must still hold
        # -- and is enforced here -- is that this module never FITS:
        # it may only apply already-fitted objects via .transform().
        tree = ast.parse(_SRC.read_text(encoding="utf-8"))
        fit_calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) in {"fit", "fit_transform", "partial_fit"}
        ]
        self.assertEqual(fit_calls, [], "candidate_contract must never fit an estimator or preprocessor")

    def test_module_has_no_calibration_or_tuning(self):
        src = _SRC.read_text(encoding="utf-8")
        for forbidden in ("Calibrated", "isotonic", "Platt", "GridSearch", "C_GRID", "max_iter", "random_state"):
            self.assertNotIn(forbidden, src)

    def test_module_cannot_reach_the_final_test_split(self):
        tree = ast.parse(_SRC.read_text(encoding="utf-8"))
        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for forbidden in ("final_split", "FINAL_TEST_SEASONS", "FINAL_TRAIN_SEASONS"):
            self.assertNotIn(forbidden, imported)
            self.assertNotIn(forbidden, names)
        self.assertFalse(hasattr(cc, "final_split"))

    def test_module_writes_no_artifact(self):
        src = _SRC.read_text(encoding="utf-8")
        for forbidden in ("write_text", "to_csv", "to_json", "open("):
            self.assertNotIn(forbidden, src)

    def test_module_never_writes_model_version(self):
        src = _SRC.read_text(encoding="utf-8")
        self.assertNotIn("MODEL_VERSION =", src)


class TestGate1GovernanceInvariants(unittest.TestCase):
    def test_model_version_still_v1(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_no_v2_named_module_created(self):
        for p in Path("src/models").glob("*.py"):
            self.assertNotIn("v2", p.stem.lower())

    def test_no_v2_data_artifact_created(self):
        for p in Path("data/audit").glob("*"):
            self.assertNotIn("v2", p.name.lower())

    def test_no_gate1_result_artifact_created(self):
        # Gate 1 is structural: it produces code and tests, not data.
        for p in Path("data/audit").rglob("*"):
            self.assertNotIn("gate1", p.name.lower())

    def test_locked_v1_and_phase_artifacts_unchanged(self):
        expected = {
            "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
            "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
            "data/audit/phase3_model_comparison.json": "616279914b2730749d52eae15b5f96b9",
            "data/audit/phase4a_ablation_comparison.json": "075b0686bce20bfce9f7289fa37076c0",
            "data/audit/phase4b_robustness_comparison.json": "97d0f8b8674c9bf8598b6c3d6b7c825c",
            "data/audit/phase4c_final_comparison.json": "effd9e54130b2bc5aaf51cd643962396",
            "data/audit/phase5a_dropped_feature_comparison.json": "3ceb7090f9e19c106d88eaa7a8848818",
            "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
            "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
            "src/models/ablation.py": "9bf8bf4a1f332f8bb47d640a72384ed7",
            "src/models/data.py": "b78e30eb45dbc46621c0160a188ce981",
            "src/models/splits.py": "8b7991ab3739c7d4daa2bf1998163da4",
            "src/models/evaluate.py": "4e9d9313867d47a19001a383a301c2fe",
        }
        for rel, exp in expected.items():
            actual = hashlib.md5(Path(rel).read_bytes()).hexdigest()
            self.assertEqual(actual, exp, f"locked artifact modified: {rel}")


if __name__ == "__main__":
    unittest.main()
