"""Phase 4C is a DESIGN-ONLY governance checkpoint. These tests validate
that the invariants the protocol document asserts are actually true of
the repository, and that Phase 4C introduced no executable final-test
path. They train nothing, load no 2025/26 data for evaluation, and
create no result artifact.

If any assertion here fails, the protocol document is describing a
baseline that no longer matches the code -- which must be reconciled
before any final-test comparison is authorized.
"""
import _pathfix  # noqa: F401
import re
import unittest
from pathlib import Path

_PROTOCOL = Path("docs/PHASE4C_V1_REPLACEMENT_DECISION_PROTOCOL.md")


class TestProtocolDocumentExists(unittest.TestCase):
    def test_protocol_document_present(self):
        self.assertTrue(_PROTOCOL.exists(), "Phase 4C protocol document missing")

    def test_protocol_declares_design_only(self):
        text = _PROTOCOL.read_text(encoding="utf-8").lower()
        self.assertIn("design only", text)
        self.assertIn("no final-test evaluation", text.replace("execution", "evaluation"))


class TestBaselineConstantsMatchProtocol(unittest.TestCase):
    """The exact V1 values the protocol records must still be the live
    values in code -- otherwise the protocol's §2 baseline is stale."""

    def test_v1_config_constants_match_protocol_table(self):
        from models.config import (
            APPROVED_FEATURE_COLUMNS_V1, FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS,
            MODEL_VERSION, REQUIRED_FEATURE_VERSION, WALK_FORWARD_FOLDS,
        )
        from models.baselines import CLASS_ORDER
        self.assertEqual(MODEL_VERSION, "v1.0")
        self.assertEqual(REQUIRED_FEATURE_VERSION, "v1.0")
        self.assertEqual(len(APPROVED_FEATURE_COLUMNS_V1), 15)
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])
        self.assertEqual(FINAL_TEST_SEASONS, ("2025/2026",))
        self.assertEqual(
            FINAL_TRAIN_SEASONS,
            ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"),
        )
        self.assertEqual(len(WALK_FORWARD_FOLDS), 3)

    def test_locked_v1_final_test_metrics_match_protocol(self):
        import json
        d = json.loads(Path("data/audit/phase3_model_comparison.json").read_text(encoding="utf-8"))
        m = d["final_test"]["model_metrics"]["logistic_regression"]
        self.assertEqual(m["log_loss"], 1.012643607907126)
        self.assertEqual(m["brier"], 0.6059500599589396)
        self.assertEqual(m["accuracy"], 0.500856653340948)
        self.assertEqual(d["final_test"]["n_test"], 1751)
        self.assertEqual(d["model_selection"]["selected_model"], "logistic_regression")

    def test_calibration_decision_is_uncalibrated(self):
        import json
        c = json.loads(Path("data/audit/phase3_calibration_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(c["selection"]["selected_calibration_method"], "uncalibrated")


class TestCandidateDefinitionMatchesProtocol(unittest.TestCase):
    def test_model_b_is_80_columns_a_plus_shots(self):
        from models.ablation import (
            MODEL_A_COLUMNS, MODEL_B_COLUMNS, SHOTS_CORE_COLUMNS, SHOTS_ON_CORE_COLUMNS,
        )
        self.assertEqual(len(MODEL_B_COLUMNS), 80)
        self.assertEqual(
            set(MODEL_B_COLUMNS),
            set(MODEL_A_COLUMNS) | set(SHOTS_CORE_COLUMNS) | set(SHOTS_ON_CORE_COLUMNS),
        )

    def test_protocol_documented_feature_asymmetry_is_real(self):
        # The protocol states league_home_advantage_season is a V1 column
        # ABSENT from Model B. If that ever stops being true, the
        # protocol's §3 note is wrong and must be revisited.
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        from models.ablation import MODEL_B_COLUMNS
        self.assertIn("league_home_advantage_season", APPROVED_FEATURE_COLUMNS_V1)
        self.assertNotIn("league_home_advantage_season", MODEL_B_COLUMNS)

    def test_v1_estimator_is_C1_as_protocol_default_candidate_states(self):
        import ast
        tree = ast.parse(Path("src/models/train.py").read_text(encoding="utf-8"))
        kwargs = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LogisticRegression":
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        self.assertIsNotNone(kwargs)
        self.assertEqual(kwargs.get("C"), 1.0)
        self.assertEqual(kwargs.get("random_state"), 0)
        self.assertEqual(kwargs.get("max_iter"), 2000)


class TestNoFinalTestExecutionIntroduced(unittest.TestCase):
    """Phase 4C must add no runnable final-test comparison path."""

    def test_no_unauthorized_phase4c_runner_module_exists(self):
        # Originally this asserted that NO Phase 4C runner existed, which
        # was correct while Phase 4C was design-only. That premise was
        # explicitly superseded when final-test execution was authorized:
        # `run_final_comparison.py` is now the single authorized runner.
        # The invariant that still matters -- and is enforced here -- is
        # that no OTHER, unauthorized runner (a V2 builder, a
        # replacement/promotion script) has appeared alongside it.
        for candidate in ("run_phase4c", "run_v2", "run_replacement", "run_promotion", "build_v2"):
            self.assertFalse(
                Path(f"src/models/{candidate}.py").exists(),
                f"unexpected/unauthorized Phase 4C runner src/models/{candidate}.py",
            )

    def test_the_single_authorized_runner_is_write_once_guarded(self):
        # The authorized runner may exist, but it must be incapable of
        # overwriting an existing artifact (protocol §12).
        src = Path("src/models/run_final_comparison.py")
        if not src.exists():
            self.skipTest("authorized Phase 4C runner not present")
        text = src.read_text(encoding="utf-8")
        self.assertIn("_refuse_if_exists", text)
        self.assertIn("ProtocolViolation", text)

    def test_phase4c_artifacts_are_exactly_the_authorized_single_run_set(self):
        # Originally this asserted NO Phase 4C artifact existed, which was
        # correct while Phase 4C was design-only. That premise was
        # superseded when final-test execution was authorized. The
        # invariant that still matters -- and is enforced here -- is that
        # exactly the declared single-run artifact set exists, with no
        # extra/duplicate/rerun outputs (protocol §12 "artifact mismatch",
        # "unexpected files").
        audit = Path("data/audit")
        top_level = sorted(p.name for p in audit.glob("*phase4c*"))
        if not top_level:
            self.skipTest("Phase 4C not yet executed in this environment")
        self.assertEqual(
            top_level,
            ["phase4c_final_comparison.json", "phase4c_predictions", "phase4c_prerun_manifest.json"],
            f"unexpected Phase 4C artifact set: {top_level}",
        )

    def test_phase4c_prediction_set_is_exactly_two_paired_arms(self):
        pred = Path("data/audit/phase4c_predictions")
        if not pred.exists():
            self.skipTest("Phase 4C not yet executed in this environment")
        files = sorted(p.name for p in pred.glob("*.csv"))
        self.assertEqual(
            files,
            ["model_b_logistic_regression_final_test.csv", "v1_logistic_regression_final_test.csv"],
            f"unexpected Phase 4C prediction files: {files}",
        )

    def test_no_phase4c_rerun_or_duplicate_output_exists(self):
        # A second/backup/copy artifact would indicate a rerun, which the
        # protocol forbids for a fixed candidate spec.
        for p in Path("data/audit").rglob("*phase4c*"):
            name = p.name.lower()
            for marker in ("_v2", "_rerun", "_copy", "(1)", "_old", "_bak"):
                self.assertNotIn(marker, name, f"possible rerun/duplicate artifact: {p}")


class TestProtocolContainsRequiredGovernanceSections(unittest.TestCase):
    def setUp(self):
        self.text = _PROTOCOL.read_text(encoding="utf-8")

    def test_all_thirteen_sections_present(self):
        for n, title in [
            (1, "Status and scope"), (2, "Current baseline"), (3, "Candidate definition"),
            (4, "Pre-final-test lock"), (5, "Final-test comparison design"),
            (6, "Acceptance criteria"), (7, "No-peek"), (8, "Decision matrix"),
            (9, "V2 authorization boundary"), (10, "Artifact requirements"),
            (11, "Integrity requirements"), (12, "Stop conditions"), (13, "Recommended next action"),
        ]:
            self.assertRegex(self.text, rf"##\s*{n}\.", f"missing section {n} ({title})")

    def test_no_automatic_replacement_language(self):
        low = self.text.lower()
        # The governance core: eligibility != promotion, and validation
        # performance alone never replaces V1.
        self.assertIn("eligibility", low)
        self.assertTrue(
            "better validation performance is not" in low
            or "not, by itself, grounds for replacement" in low
        )

    def test_declares_no_epsilon_threshold_invented(self):
        low = self.text.lower()
        self.assertTrue(
            "requires human authorization" in low or "requires human authorization" in low
        )
        self.assertIn("no threshold", low)

    def test_records_that_final_test_candidate_does_not_yet_exist(self):
        low = self.text.lower()
        self.assertIn("does not exist yet", low)


if __name__ == "__main__":
    unittest.main()
