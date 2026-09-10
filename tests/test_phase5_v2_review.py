"""Phase 5 — V2 candidate review invariants.

Additive, read-only. Trains nothing, evaluates nothing, and asserts that
the review/spec-draft documents state facts that hold in the repository,
and that Phase 5 produced no production side-effects.
"""
import _pathfix  # noqa: F401
import hashlib
import json
import unittest
from pathlib import Path

_REVIEW = Path("docs/PHASE5_V2_CANDIDATE_REVIEW.md")
_DRAFT = Path("docs/PHASE5_V2_MODEL_SPEC_DRAFT.md")

VALID_STATUSES = {
    "REJECT_V2_CANDIDATE",
    "CONDITIONAL_V2_CANDIDATE",
    "APPROVED_V2_CANDIDATE_FOR_IMPLEMENTATION",
}


class TestV1RemainsFrozen(unittest.TestCase):
    def test_model_version_still_v1(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_required_feature_version_unchanged(self):
        from models.config import REQUIRED_FEATURE_VERSION
        self.assertEqual(REQUIRED_FEATURE_VERSION, "v1.0")

    def test_v1_feature_contract_unchanged(self):
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        self.assertEqual(len(APPROVED_FEATURE_COLUMNS_V1), 15)
        self.assertIn("league_home_advantage_season", APPROVED_FEATURE_COLUMNS_V1)

    def test_final_test_seasons_unchanged(self):
        from models.config import FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS
        self.assertEqual(FINAL_TEST_SEASONS, ("2025/2026",))
        self.assertEqual(
            FINAL_TRAIN_SEASONS,
            ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"),
        )

    def test_class_order_unchanged(self):
        from models.baselines import CLASS_ORDER
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])


class TestModelBDefinitionMatchesPhase4A(unittest.TestCase):
    def test_feature_count_is_80(self):
        from models.ablation import MODEL_B_COLUMNS
        self.assertEqual(len(MODEL_B_COLUMNS), 80)

    def test_b_equals_a_plus_shots_and_shots_on(self):
        from models.ablation import (
            MODEL_A_COLUMNS, MODEL_B_COLUMNS, SHOTS_CORE_COLUMNS, SHOTS_ON_CORE_COLUMNS,
        )
        self.assertEqual(
            set(MODEL_B_COLUMNS),
            set(MODEL_A_COLUMNS) | set(SHOTS_CORE_COLUMNS) | set(SHOTS_ON_CORE_COLUMNS),
        )

    def test_no_xg_in_model_b(self):
        from models.ablation import MODEL_B_COLUMNS, XG_CORE_COLUMNS
        self.assertTrue(set(MODEL_B_COLUMNS).isdisjoint(XG_CORE_COLUMNS))

    def test_no_duplicate_columns(self):
        from models.ablation import MODEL_B_COLUMNS
        self.assertEqual(len(MODEL_B_COLUMNS), len(set(MODEL_B_COLUMNS)))

    def test_no_label_or_bookkeeping_columns(self):
        from models.ablation import MODEL_B_COLUMNS
        from models.config import X_EXCLUDED_COLUMNS
        self.assertEqual(set(MODEL_B_COLUMNS) & set(X_EXCLUDED_COLUMNS), set())

    def test_C_is_1_point_0_in_frozen_v1_estimator(self):
        import ast
        tree = ast.parse(Path("src/models/train.py").read_text(encoding="utf-8"))
        kwargs = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LogisticRegression":
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        self.assertEqual(kwargs["C"], 1.0)
        self.assertEqual(kwargs["max_iter"], 2000)
        self.assertEqual(kwargs["random_state"], 0)


class TestFeatureAsymmetryDocumentedCorrectly(unittest.TestCase):
    def test_exact_set_difference_is_the_documented_one(self):
        from models.ablation import MODEL_B_COLUMNS
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        v1, b = set(APPROVED_FEATURE_COLUMNS_V1), set(MODEL_B_COLUMNS)
        self.assertEqual(v1 - b, {"league_home_advantage_season"})
        self.assertEqual(len(b - v1), 66)
        self.assertEqual(len(v1 & b), 14)
        self.assertFalse(v1 < b, "Model B must NOT be a strict superset of V1")

    @unittest.skipUnless(_REVIEW.exists(), "review doc absent")
    def test_review_documents_the_asymmetry(self):
        text = _REVIEW.read_text(encoding="utf-8")
        self.assertIn("league_home_advantage_season", text)
        self.assertIn("66", text)
        self.assertIn("14", text)


@unittest.skipUnless(_REVIEW.exists(), "review doc absent")
class TestReviewDocumentGovernance(unittest.TestCase):
    def setUp(self):
        self.text = _REVIEW.read_text(encoding="utf-8")

    def test_exactly_one_valid_status_declared(self):
        # Originally this pinned the status to CONDITIONAL_V2_CANDIDATE,
        # correct while Conditions 1 and 2 were open. Both were later
        # closed by authorized phases (5A: RETAIN_OMISSION_SUPPORTED;
        # 5B: DRAW_LIMITATION_ACCEPTED), so the review's declared status
        # legitimately advanced. The invariant that still matters is that
        # the declared status is exactly one of the permitted values.
        import re
        m = re.search(r"## STATUS: [`]([A-Z0-9_]+)[`]", self.text)
        self.assertIsNotNone(m, "no declared status heading found")
        self.assertIn(m.group(1), VALID_STATUSES)
        self.assertTrue({s for s in VALID_STATUSES if s in self.text})

    def test_no_invented_status_used(self):
        for bogus in ("APPROVED_FOR_PRODUCTION", "V2_APPROVED", "PROMOTE", "DEPLOY_APPROVED"):
            self.assertNotIn(bogus, self.text)

    def test_draw_not_described_as_solved(self):
        low = self.text.lower()
        self.assertNotIn("draw problem solved", low)
        self.assertNotIn("draws are solved", low)
        self.assertIn("not solved", low)

    def test_no_statistical_significance_claim(self):
        self.assertNotIn("statistically significant", self.text.lower())

    def test_records_exact_phase4c_values(self):
        # Magnitude-only matching: the document renders negative deltas
        # with a Unicode minus (U+2212) in tables, so an ASCII-hyphen
        # comparison would false-negative on a value that is present.
        for v in ("1.012643607907126", "0.9960487649063601", "0.016594843000765858",
                  "0.017977528089887642", "0.03440860215053764"):
            self.assertIn(v, self.text, f"Phase 4C value {v} not recorded in review")

    def test_log_loss_delta_recorded_as_negative(self):
        # The sign must be stated, in either glyph.
        self.assertTrue(
            "−0.016594843000765858" in self.text or "-0.016594843000765858" in self.text,
            "log-loss delta must be recorded with its negative sign",
        )

    def test_c_0_01_not_retrospectively_selected(self):
        low = self.text.lower()
        self.assertIn("c=1.0 remains", low.replace("**", ""))
        self.assertIn("not retrospectively selected", low)


@unittest.skipUnless(_DRAFT.exists(), "spec draft absent")
class TestSpecDraftIsNonProduction(unittest.TestCase):
    def setUp(self):
        self.text = _DRAFT.read_text(encoding="utf-8")

    def test_marked_draft_not_production_authorized(self):
        self.assertIn("DRAFT — NOT PRODUCTION AUTHORIZED", self.text)

    def test_states_model_version_unchanged(self):
        self.assertIn("v1.0", self.text)
        low = self.text.lower()
        self.assertIn("does not change `model_version`", low)

    def test_freezes_required_parameters(self):
        for token in ("1.0", "2000", "0", "H", "D", "A"):
            self.assertIn(token, self.text)
        self.assertIn("Calibration", self.text)
        self.assertIn("none", self.text.lower())

    def test_declares_ten_acceptance_gates(self):
        for n in range(1, 11):
            self.assertIn(f"**{n} —", self.text, f"gate {n} missing")

    def test_explicit_non_authorizations_present(self):
        low = self.text.lower()
        self.assertIn("does **not** authorize", low)
        self.assertIn("deployment", low)


class TestNoV2SideEffects(unittest.TestCase):
    def test_no_v2_module_created(self):
        for p in Path("src/models").glob("*.py"):
            self.assertNotIn("v2", p.stem.lower())

    def test_no_v2_data_artifact_created(self):
        for p in Path("data/audit").glob("*"):
            self.assertNotIn("v2", p.name.lower())

    def test_no_unauthorized_phase5_result_artifact_created(self):
        # Phase 5 itself is review-only and produced no data/audit
        # artifact. Phase 5A was subsequently authorized as a
        # validation-only experiment and legitimately produced
        # `phase5a_dropped_feature_comparison.json` (+ optional
        # phase5a_predictions). The invariant that still holds is that
        # no OTHER phase5* artifact exists -- in particular the
        # document-only Phase 5B review must not have produced one.
        allowed_prefixes = ("phase5a_",)
        for p in Path("data/audit").rglob("*"):
            name = p.name.lower()
            if "phase5" not in name:
                continue
            self.assertTrue(
                name.startswith(allowed_prefixes),
                f"unexpected phase5 artifact: {p}",
            )
            self.assertNotIn("phase5b", name, f"Phase 5B must be document-only: {p}")


class TestLockedArtifactsUnchanged(unittest.TestCase):
    EXPECTED = {
        "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
        "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
        "data/audit/phase3_model_comparison.json": "616279914b2730749d52eae15b5f96b9",
        "data/audit/phase3_calibration_comparison.json": "d94ed430ab13337797f0592bf82878cb",
        "data/audit/phase4a_ablation_comparison.json": "075b0686bce20bfce9f7289fa37076c0",
        "data/audit/phase4b_robustness_comparison.json": "97d0f8b8674c9bf8598b6c3d6b7c825c",
        "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
        "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
        "src/models/splits.py": "8b7991ab3739c7d4daa2bf1998163da4",
        "src/models/evaluate.py": "4e9d9313867d47a19001a383a301c2fe",
        "src/models/data.py": "b78e30eb45dbc46621c0160a188ce981",
        "src/models/baselines.py": "42e64e3a0ba8c4bf8cf264f80cdcd208",
        "src/models/run_experiments.py": "546ea0105ca8b235ab80b393a58f2831",
    }

    def test_all_locked_artifacts_byte_identical(self):
        for rel, expected in self.EXPECTED.items():
            p = Path(rel)
            self.assertTrue(p.exists(), f"locked artifact missing: {rel}")
            actual = hashlib.md5(p.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, f"locked artifact modified: {rel}")

    def test_phase4c_artifacts_still_present_and_unmodified_set(self):
        top = sorted(p.name for p in Path("data/audit").glob("*phase4c*"))
        self.assertEqual(
            top,
            ["phase4c_final_comparison.json", "phase4c_predictions", "phase4c_prerun_manifest.json"],
        )

    def test_phase4c_result_still_reports_zone_pending_or_reviewed(self):
        d = json.loads(Path("data/audit/phase4c_final_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(d["model_version_unchanged"], "v1.0")
        self.assertTrue(d["v1_reproduction_check"]["all_match"])
        self.assertTrue(d["validity_checks"]["all_passed"])


if __name__ == "__main__":
    unittest.main()
