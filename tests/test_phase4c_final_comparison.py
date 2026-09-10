"""Tests for src/models/run_final_comparison.py -- Phase 4C.

These validate the runner's protocol safeguards WITHOUT executing the
final-test comparison: locked-checksum verification, write-once refusal,
frozen spec constants, arm definitions, delta computation, and the
absence of any automatic zone assignment or V2 promotion.

No test here trains a model or evaluates 2025/26.
"""
import _pathfix  # noqa: F401
import ast
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models import run_final_comparison as rfc

_SRC = Path("src/models/run_final_comparison.py")


class TestAuthorizedFrozenDecisions(unittest.TestCase):
    def test_C_is_1_point_0_option_c1(self):
        self.assertEqual(rfc.AUTHORIZED_C, 1.0)

    def test_no_epsilon_threshold_authorized(self):
        self.assertIsNone(rfc.EPSILON_AUTHORIZED)

    def test_locked_v1_metrics_match_phase3(self):
        import json
        d = json.loads(Path("data/audit/phase3_model_comparison.json").read_text(encoding="utf-8"))
        m = d["final_test"]["model_metrics"]["logistic_regression"]
        self.assertEqual(rfc.V1_LOCKED_FINAL_TEST["log_loss"], m["log_loss"])
        self.assertEqual(rfc.V1_LOCKED_FINAL_TEST["brier"], m["brier"])
        self.assertEqual(rfc.V1_LOCKED_FINAL_TEST["accuracy"], m["accuracy"])
        self.assertEqual(rfc.V1_LOCKED_FINAL_TEST["n"], d["final_test"]["n_test"])

    def test_two_arms_use_the_frozen_feature_sets(self):
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        from models.ablation import MODEL_B_COLUMNS
        self.assertEqual(len(APPROVED_FEATURE_COLUMNS_V1), 15)
        self.assertEqual(len(MODEL_B_COLUMNS), 80)


class TestLockedInputVerification(unittest.TestCase):
    def test_all_thirteen_locked_inputs_declared(self):
        self.assertEqual(len(rfc.LOCKED_INPUTS), 13)
        for p in rfc.LOCKED_INPUTS:
            self.assertTrue(Path(p).exists(), f"declared locked input missing: {p}")

    def test_verify_locked_inputs_passes_on_current_repo(self):
        result = rfc.verify_locked_inputs()
        self.assertTrue(all(v["match"] for v in result.values()))

    def test_verify_locked_inputs_raises_on_mismatch(self):
        original = dict(rfc.LOCKED_INPUTS)
        try:
            rfc.LOCKED_INPUTS["src/models/config.py"] = "0" * 32
            with self.assertRaises(rfc.ProtocolViolation):
                rfc.verify_locked_inputs()
        finally:
            rfc.LOCKED_INPUTS.clear()
            rfc.LOCKED_INPUTS.update(original)


class TestWriteOnceSafety(unittest.TestCase):
    def test_refuse_if_exists_raises_for_existing_file(self):
        with self.assertRaises(rfc.ProtocolViolation):
            rfc._refuse_if_exists(Path("data/audit/phase3_model_comparison.json"))

    def test_refuse_if_exists_allows_absent_path(self):
        rfc._refuse_if_exists(Path("data/audit/__definitely_not_present__.json"))

    def test_all_output_paths_are_phase4c_scoped(self):
        for p in (rfc.DEFAULT_RESULTS_PATH, rfc.DEFAULT_MANIFEST_PATH, rfc.DEFAULT_PREDICTIONS_DIR):
            self.assertIn("phase4c", str(p))

    def test_runner_never_writes_to_phase3_4a_4b_paths(self):
        src = _SRC.read_text(encoding="utf-8")
        for line in src.splitlines():
            if "write_text" in line or "to_csv" in line:
                for forbidden in ("phase3_", "phase4a_", "phase4b_"):
                    self.assertNotIn(forbidden, line)

    def test_main_refuses_when_results_artifact_already_exists(self):
        src = inspect.getsource(rfc.main)
        self.assertIn("_refuse_if_exists(results_path)", src)


class TestNoAutomaticPromotion(unittest.TestCase):
    def test_runner_never_writes_model_version(self):
        src = _SRC.read_text(encoding="utf-8")
        self.assertNotIn("MODEL_VERSION =", src)
        self.assertNotIn('MODEL_VERSION="', src)

    def test_default_decision_zone_is_pending_human_review(self):
        src = inspect.getsource(rfc.main)
        self.assertIn("PENDING_HUMAN_REVIEW", src)

    def test_deltas_carry_no_meaningfulness_judgment(self):
        v1 = {"log_loss": 1.0, "brier": 0.6, "accuracy": 0.5, "macro_f1": 0.37, "balanced_accuracy": 0.43}
        b = {"log_loss": 0.99, "brier": 0.59, "accuracy": 0.51, "macro_f1": 0.38, "balanced_accuracy": 0.44}
        out = rfc.compute_paired_deltas(v1, b)
        self.assertAlmostEqual(out["log_loss"]["delta_b_minus_v1"], -0.01)
        self.assertTrue(out["log_loss"]["b_numerically_better"])
        # must expose only numerical framing
        for key in out:
            if key.startswith("_"):
                continue
            self.assertNotIn("meaningful", str(out[key]).lower())
            self.assertNotIn("significant", str(out[key]).lower())
        self.assertIn("no epsilon", out["_note"].lower())

    def test_delta_direction_respects_lower_is_better(self):
        v1 = {"log_loss": 1.0, "brier": 0.6, "accuracy": 0.5, "macro_f1": 0.37, "balanced_accuracy": 0.43}
        b = {"log_loss": 1.1, "brier": 0.7, "accuracy": 0.45, "macro_f1": 0.30, "balanced_accuracy": 0.40}
        out = rfc.compute_paired_deltas(v1, b)
        self.assertFalse(out["log_loss"]["b_numerically_better"])
        self.assertFalse(out["accuracy"]["b_numerically_better"])


class TestV1ReproductionCheck(unittest.TestCase):
    def test_passes_on_exact_locked_metrics(self):
        rec = dict(rfc.V1_LOCKED_FINAL_TEST)
        self.assertTrue(rfc.verify_v1_reproduction(rec)["all_match"])

    def test_fails_on_perturbed_metric(self):
        rec = dict(rfc.V1_LOCKED_FINAL_TEST)
        rec["log_loss"] = rec["log_loss"] + 1e-6
        self.assertFalse(rfc.verify_v1_reproduction(rec)["all_match"])

    def test_tolerance_is_1e_minus_9(self):
        self.assertEqual(inspect.signature(rfc.verify_v1_reproduction).parameters["tol"].default, 1e-9)


class TestValidityCheckLogic(unittest.TestCase):
    def _frame(self, arm, fixtures, y="H"):
        n = len(fixtures)
        return pd.DataFrame({
            "arm": arm, "fixture_id": fixtures, "season_id": [9999] * n,
            "y_true": [y] * n, "y_pred": [y] * n,
            "p_home": [0.5] * n, "p_draw": [0.3] * n, "p_away": [0.2] * n,
        })

    def test_detects_fixture_set_mismatch(self):
        class DS:
            def __init__(self, ids):
                self.metadata = pd.DataFrame({"fixture_id": ids, "unix": range(len(ids))})
        frames = {rfc.ARM_V1: self._frame(rfc.ARM_V1, [1, 2, 3]),
                  rfc.ARM_B: self._frame(rfc.ARM_B, [1, 2, 4])}
        checks = rfc.run_validity_checks(DS([10, 11]), DS([1, 2, 3]), frames)
        self.assertFalse(checks["identical_fixture_sets"])
        self.assertFalse(checks["all_passed"])

    def test_detects_wrong_n_test(self):
        class DS:
            def __init__(self, ids):
                self.metadata = pd.DataFrame({"fixture_id": ids, "unix": range(len(ids))})
        frames = {rfc.ARM_V1: self._frame(rfc.ARM_V1, [1, 2]),
                  rfc.ARM_B: self._frame(rfc.ARM_B, [1, 2])}
        checks = rfc.run_validity_checks(DS([9]), DS([1, 2]), frames)
        self.assertFalse(checks["n_test_equals_1751_both"])


class TestFinalSplitUsedNotWalkForward(unittest.TestCase):
    def test_runner_uses_final_split_as_authorized(self):
        # Phase 4C is the ONE authorized place final_split may be called.
        src = _SRC.read_text(encoding="utf-8")
        self.assertIn("final_split", src)

    def test_runner_has_no_random_split_machinery(self):
        tree = ast.parse(_SRC.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if a.name.split(".")[0] == "random" or "train_test_split" in a.name:
                        offenders.append(a.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"shuffle", "train_test_split"}:
                    offenders.append(node.func.attr)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
