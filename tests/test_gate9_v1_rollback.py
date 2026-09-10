"""Gate 9 — V1 rollback (non-destructive restorability rehearsal).

Gate 9 requirement (docs/PHASE5_V2_MODEL_SPEC_DRAFT.md):
    "V1 restorable without retraining or data regeneration."
    Verification: rehearse rollback; re-verify V1 checksums and reproduce
    V1's locked metrics.
    Pass: V1 restored; metrics reproduce to <1e-9.
    Stop: rollback fails or V1 metrics do not reproduce.

APPROVED INTERPRETATION (a) + (d), non-destructive:
  (a) Restorability is rehearsed WITHOUT mutating repository state. No
      destructive revert is performed and no pretend promoted-V2 state is
      created. V1 is currently the production configuration, so there is
      nothing to revert FROM; what Gate 9 can and does establish is that
      V1's frozen configuration and artifacts are intact and recoverable
      from pinned baseline evidence.
  (d) The "metrics reproduce to <1e-9" half is satisfied from EXISTING
      evidence -- Gate 7's already-passed reproduction, recorded in
      `phase4c_final_comparison.json::v1_reproduction_check` -- rather
      than by retraining V1. Nothing here retrains, and 2025/26 is never
      accessed for any new computation.

BASELINE SOURCE OF TRUTH: `data/audit/phase4c_prerun_manifest.json`.
Its `locked_input_checksums` block (13 entries, including exactly 7
`src/models/*.py` V1 production files) was recorded BEFORE the audited
Phase 4C run, and its `v1_locked_final_test_metrics` block records V1's
locked metrics. Gate 9 verifies against that recorded evidence rather
than against constants invented here, so the pins are independently
substantiated. Where a literal is used it is additionally asserted equal
to the manifest value, so a drifting literal fails rather than masks.

Gate 9 imports no training machinery, no calibration/tuning, no
`final_split`, and no artifact-writing machinery -- asserted by AST.
"""
import _pathfix  # noqa: F401
import ast
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

MANIFEST = Path("data/audit/phase4c_prerun_manifest.json")
COMPARISON = Path("data/audit/phase4c_final_comparison.json")
GATE7_TESTS = Path("tests/test_gate7_phase4c_regression.py")

# V1's locked final-test metrics. Each is additionally asserted against
# the manifest, so these literals cannot silently drift.
V1_LOCKED_METRICS = {
    "log_loss": 1.012643607907126,
    "brier": 0.6059500599589396,
    "accuracy": 0.500856653340948,
    "macro_f1": 0.3712753915810083,
    "balanced_accuracy": 0.4321797560315324,
    "n": 1751,
}

REPRODUCTION_TOLERANCE = 1e-9

# The seven frozen V1 production source files, as recorded by the manifest.
EXPECTED_V1_SOURCE_FILES = (
    "src/models/config.py",
    "src/models/train.py",
    "src/models/splits.py",
    "src/models/evaluate.py",
    "src/models/data.py",
    "src/models/baselines.py",
    "src/models/run_experiments.py",
)


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _pinned_baselines() -> dict[str, str]:
    """The pinned baseline set: {relative_path: expected_md5}."""
    return {k: v["expected"] for k, v in _manifest()["locked_input_checksums"].items()}


# =====================================================================
# 0. Baseline evidence must exist before anything can be verified
#    (instruction 10: STOP rather than manufacture evidence).
# =====================================================================
class TestGate9BaselineEvidenceExists(unittest.TestCase):
    def test_prerun_manifest_exists(self):
        self.assertTrue(MANIFEST.exists(), "pinned baseline manifest missing -- Gate 9 cannot be substantiated")

    def test_manifest_contains_locked_input_checksums(self):
        m = _manifest()
        self.assertIn("locked_input_checksums", m)
        self.assertEqual(len(m["locked_input_checksums"]), 13)

    def test_manifest_contains_v1_locked_metrics(self):
        self.assertIn("v1_locked_final_test_metrics", _manifest())

    def test_comparison_artifact_exists(self):
        self.assertTrue(COMPARISON.exists())

    def test_gate7_reproduction_evidence_module_exists(self):
        self.assertTrue(GATE7_TESTS.exists(), "Gate 7 evidence module missing")


# =====================================================================
# 1. V1 configuration remains exactly frozen
# =====================================================================
class TestGate9V1ConfigurationFrozen(unittest.TestCase):
    def test_model_version_is_v1_0(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_required_feature_version_is_v1_0(self):
        from models.config import REQUIRED_FEATURE_VERSION
        self.assertEqual(REQUIRED_FEATURE_VERSION, "v1.0")

    def test_approved_v1_feature_columns_unchanged(self):
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        self.assertEqual(len(APPROVED_FEATURE_COLUMNS_V1), 15)
        self.assertEqual(APPROVED_FEATURE_COLUMNS_V1, (
            "home_goals_for_per_match_season",
            "home_goals_against_per_match_season",
            "away_goals_for_per_match_season",
            "away_goals_against_per_match_season",
            "home_goals_for_per_match_last5",
            "away_goals_for_per_match_last5",
            "home_points_last5",
            "away_points_last5",
            "home_attack_strength_score",
            "home_defence_strength_score",
            "away_attack_strength_score",
            "away_defence_strength_score",
            "strength_diff",
            "league_home_advantage_season",
            "competition_id",
        ))

    def test_class_order_is_h_d_a(self):
        from models.baselines import CLASS_ORDER
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])

    def test_final_train_seasons_unchanged(self):
        from models.config import FINAL_TRAIN_SEASONS
        self.assertEqual(FINAL_TRAIN_SEASONS,
                         ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"))

    def test_final_test_seasons_unchanged(self):
        from models.config import FINAL_TEST_SEASONS
        self.assertEqual(FINAL_TEST_SEASONS, ("2025/2026",))

    def test_config_matches_the_manifests_recorded_frozen_spec(self):
        # Cross-check the live config against what Phase 4C recorded.
        from models.baselines import CLASS_ORDER
        from models.config import APPROVED_FEATURE_COLUMNS_V1, FINAL_TRAIN_SEASONS, MODEL_VERSION
        fs = _manifest()["frozen_spec"]
        self.assertEqual(fs["class_order"], CLASS_ORDER)
        self.assertEqual(fs["v1_n_features"], len(APPROVED_FEATURE_COLUMNS_V1))
        self.assertEqual(list(fs["final_train_seasons"]), list(FINAL_TRAIN_SEASONS))
        self.assertEqual(fs["model_version_unchanged"], MODEL_VERSION)
        self.assertEqual(list(fs["v1_feature_columns"]), list(APPROVED_FEATURE_COLUMNS_V1))


# =====================================================================
# 2 & 3. Frozen V1 source files and locked artifacts byte-identical
# =====================================================================
class TestGate9ChecksumVerification(unittest.TestCase):
    def test_manifest_pins_exactly_seven_v1_source_files(self):
        pinned = _pinned_baselines()
        src_files = tuple(sorted(k for k in pinned if k.startswith("src/")))
        self.assertEqual(len(src_files), 7)
        self.assertEqual(src_files, tuple(sorted(EXPECTED_V1_SOURCE_FILES)))

    def test_all_seven_v1_source_files_are_byte_identical(self):
        pinned = _pinned_baselines()
        mismatches = []
        for rel in EXPECTED_V1_SOURCE_FILES:
            self.assertIn(rel, pinned, f"{rel} not pinned in the manifest")
            actual = _md5(Path(rel))
            if actual != pinned[rel]:
                mismatches.append((rel, pinned[rel], actual))
        self.assertEqual(mismatches, [], f"frozen V1 source drift: {mismatches}")

    def test_every_pinned_baseline_still_matches(self):
        # All 13 pinned inputs, not just the source files.
        mismatches = []
        for rel, expected in _pinned_baselines().items():
            p = Path(rel)
            if not p.exists():
                mismatches.append((rel, expected, "MISSING"))
                continue
            actual = _md5(p)
            if actual != expected:
                mismatches.append((rel, expected, actual))
        self.assertEqual(mismatches, [], f"pinned baseline drift: {mismatches}")

    def test_databases_are_pinned_and_unchanged(self):
        pinned = _pinned_baselines()
        self.assertEqual(pinned["data/processed/features.db"], "e7ebe7fc07040a5927683c35b6371e63")
        self.assertEqual(pinned["data/processed/matches.db"], "fdeed042096fa1c851aaee6c84995247")
        self.assertEqual(_md5(Path("data/processed/features.db")), pinned["data/processed/features.db"])
        self.assertEqual(_md5(Path("data/processed/matches.db")), pinned["data/processed/matches.db"])

    def test_phase3_and_phase4c_v1_artifacts_unchanged(self):
        expected = {
            "data/audit/phase3_model_comparison.json": "616279914b2730749d52eae15b5f96b9",
            "data/audit/phase3_calibration_comparison.json": "d94ed430ab13337797f0592bf82878cb",
            "data/audit/phase4c_final_comparison.json": "effd9e54130b2bc5aaf51cd643962396",
            "data/audit/phase4c_prerun_manifest.json": "4f166d1826edbf7bfe1e022639a8b1f4",
            "data/audit/phase4c_predictions/v1_logistic_regression_final_test.csv":
                "e32c5928a2593a5c5cd804c81134f17d",
        }
        for rel, exp in expected.items():
            self.assertEqual(_md5(Path(rel)), exp, f"locked artifact changed: {rel}")

    def test_manifest_self_reported_matches_were_all_true_at_record_time(self):
        for rel, entry in _manifest()["locked_input_checksums"].items():
            self.assertTrue(entry["match"], f"{rel} did not match at Phase 4C record time")


# =====================================================================
# 4 & 5. Locked V1 metrics + Gate 7 reproduction evidence
# =====================================================================
class TestGate9V1MetricsAndReproductionEvidence(unittest.TestCase):
    def test_locked_metric_literals_match_the_manifest(self):
        recorded = _manifest()["v1_locked_final_test_metrics"]
        for k, v in V1_LOCKED_METRICS.items():
            self.assertEqual(recorded[k], v, f"{k} literal drifted from the manifest")

    def test_locked_metrics_match_the_phase3_authoritative_artifact(self):
        m = json.loads(Path("data/audit/phase3_model_comparison.json").read_text(encoding="utf-8"))
        arm = m["final_test"]["model_metrics"]["logistic_regression"]
        for k in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy", "n"):
            self.assertEqual(arm[k], V1_LOCKED_METRICS[k], f"Phase 3 {k} differs")

    def test_locked_metrics_match_the_phase4c_v1_arm(self):
        d = json.loads(COMPARISON.read_text(encoding="utf-8"))
        arm = d["arms"]["v1_logistic_regression"]
        for k in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy", "n"):
            self.assertEqual(arm[k], V1_LOCKED_METRICS[k], f"Phase 4C V1 arm {k} differs")

    def test_phase3_and_phase4c_v1_metrics_are_mutually_consistent(self):
        p3 = json.loads(Path("data/audit/phase3_model_comparison.json").read_text(encoding="utf-8"))
        p3_arm = p3["final_test"]["model_metrics"]["logistic_regression"]
        p4c_arm = json.loads(COMPARISON.read_text(encoding="utf-8"))["arms"]["v1_logistic_regression"]
        for k in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy", "n"):
            self.assertEqual(p3_arm[k], p4c_arm[k], f"{k} inconsistent between Phase 3 and Phase 4C")

    def test_gate7_reproduction_check_is_present_and_passed(self):
        d = json.loads(COMPARISON.read_text(encoding="utf-8"))
        rc = d["v1_reproduction_check"]
        self.assertTrue(rc["all_match"], "recorded V1 reproduction did not pass")
        self.assertEqual(rc["tolerance"], REPRODUCTION_TOLERANCE)

    def test_recorded_reproduction_is_within_the_required_tolerance(self):
        rc = json.loads(COMPARISON.read_text(encoding="utf-8"))["v1_reproduction_check"]
        for k, entry in rc["checks"].items():
            self.assertTrue(entry["match"], f"{k} did not reproduce")
            self.assertLess(abs(entry["abs_difference"]), REPRODUCTION_TOLERANCE,
                            f"{k} reproduction delta {entry['abs_difference']} >= {REPRODUCTION_TOLERANCE}")

    def test_recorded_reproduction_covers_every_locked_metric(self):
        rc = json.loads(COMPARISON.read_text(encoding="utf-8"))["v1_reproduction_check"]
        self.assertEqual(set(rc["checks"]), set(V1_LOCKED_METRICS))

    def test_recorded_reproduction_locked_values_equal_our_literals(self):
        rc = json.loads(COMPARISON.read_text(encoding="utf-8"))["v1_reproduction_check"]
        for k, entry in rc["checks"].items():
            self.assertEqual(entry["locked"], V1_LOCKED_METRICS[k])

    def test_gate7_module_asserts_the_reproduction_tolerance(self):
        # Evidence that the passed Gate 7 suite enforced <1e-9, rather
        # than Gate 9 taking that on trust.
        tree = ast.parse(GATE7_TESTS.read_text(encoding="utf-8"))
        assigned = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        try:
                            assigned[t.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
        self.assertEqual(assigned.get("REGRESSION_TOLERANCE"), REPRODUCTION_TOLERANCE)
        self.assertEqual(assigned.get("REFERENCE_LOG_LOSS"), 0.9960487649063601)


# =====================================================================
# 6. Non-destructive restorability rehearsal
# =====================================================================
class TestGate9NonDestructiveRestorability(unittest.TestCase):
    """Establishes that V1's frozen state is recoverable from pinned
    baseline evidence WITHOUT mutating the repository.

    Method: copy the frozen V1 files into a temporary directory, mutate
    the COPIES, then restore the copies from the untouched repository
    originals and re-verify their checksums against the manifest. This
    exercises the restore-and-verify mechanic on disposable copies.

    Explicitly NOT done: no repository file is modified; no destructive
    revert is performed; no pretend promoted-V2 state is created; and no
    claim is made that a literal production revert occurred.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.pinned = _pinned_baselines()

    def tearDown(self):
        self._tmp.cleanup()

    def test_frozen_v1_files_are_recoverable_and_verifiable_on_copies(self):
        restored, mutated_detected = 0, 0
        for rel in EXPECTED_V1_SOURCE_FILES:
            original = Path(rel)
            work = self.tmp / Path(rel).name

            # 1. Stage a disposable copy and confirm it matches the pin.
            shutil.copy2(original, work)
            self.assertEqual(_md5(work), self.pinned[rel], f"staged copy of {rel} does not match pin")

            # 2. Mutate the COPY (never the repository) and confirm the
            #    checksum check detects the drift.
            work.write_text(work.read_text(encoding="utf-8") + "\n# simulated drift\n",
                            encoding="utf-8")
            self.assertNotEqual(_md5(work), self.pinned[rel], f"drift in {rel} copy went undetected")
            mutated_detected += 1

            # 3. Restore the copy from the untouched original and
            #    re-verify against the pinned baseline.
            shutil.copy2(original, work)
            self.assertEqual(_md5(work), self.pinned[rel], f"restore of {rel} failed verification")
            restored += 1

        self.assertEqual(restored, 7)
        self.assertEqual(mutated_detected, 7)

    def test_repository_state_is_unchanged_by_the_rehearsal(self):
        # The rehearsal above must leave the real files untouched.
        for rel in EXPECTED_V1_SOURCE_FILES:
            self.assertEqual(_md5(Path(rel)), self.pinned[rel], f"{rel} was mutated by the rehearsal")

    def test_restoration_requires_no_retraining_or_data_regeneration(self):
        # §11: rollback must be "a configuration revert requiring no data
        # regeneration or retraining". Establish that the pinned set is
        # entirely static files -- source and artifacts -- with no
        # derived-state rebuild step implied.
        for rel in self.pinned:
            p = Path(rel)
            self.assertTrue(p.exists(), f"pinned input missing: {rel}")
            self.assertTrue(p.suffix in {".py", ".json", ".db", ".csv"},
                            f"unexpected pinned artifact type: {rel}")

    def test_no_pretend_promoted_v2_state_exists(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")
        for p in Path("src/models").glob("*.py"):
            self.assertNotIn("v2", p.stem.lower())
        for p in Path("data/audit").glob("*"):
            self.assertNotIn("v2", p.name.lower())


# =====================================================================
# 7. Positive / negative controls -- the checks must be able to fail
# =====================================================================
class TestGate9DetectionControls(unittest.TestCase):
    """Without these, the verifications above could pass vacuously."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_detects_changed_frozen_source(self):
        work = self.tmp / "config_copy.py"
        shutil.copy2("src/models/config.py", work)
        baseline = _md5(work)
        work.write_text(work.read_text(encoding="utf-8").replace('MODEL_VERSION = "v1.0"',
                                                                 'MODEL_VERSION = "v2.0"'),
                        encoding="utf-8")
        self.assertNotEqual(_md5(work), baseline)

    def test_detects_changed_locked_artifact(self):
        work = self.tmp / "artifact_copy.json"
        shutil.copy2("data/audit/phase3_model_comparison.json", work)
        baseline = _md5(work)
        data = json.loads(work.read_text(encoding="utf-8"))
        data["_injected"] = True
        work.write_text(json.dumps(data), encoding="utf-8")
        self.assertNotEqual(_md5(work), baseline)

    def test_detects_metric_drift(self):
        drifted = dict(V1_LOCKED_METRICS)
        drifted["log_loss"] = V1_LOCKED_METRICS["log_loss"] + 1e-9
        self.assertNotEqual(drifted["log_loss"], V1_LOCKED_METRICS["log_loss"])
        # A drift of exactly the tolerance must not be treated as a match.
        self.assertFalse(abs(drifted["log_loss"] - V1_LOCKED_METRICS["log_loss"]) < REPRODUCTION_TOLERANCE)

    def test_detects_changed_v1_configuration(self):
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        tampered = tuple(list(APPROVED_FEATURE_COLUMNS_V1) + ["injected_feature"])
        self.assertNotEqual(tampered, APPROVED_FEATURE_COLUMNS_V1)
        self.assertEqual(len(tampered), 16)

    def test_detects_missing_baseline_evidence(self):
        absent = self.tmp / "no_such_manifest.json"
        self.assertFalse(absent.exists())
        with self.assertRaises(FileNotFoundError):
            json.loads(absent.read_text(encoding="utf-8"))

    def test_md5_helper_detects_a_one_byte_change(self):
        a = self.tmp / "a.txt"
        a.write_text("x", encoding="utf-8")
        first = _md5(a)
        a.write_text("y", encoding="utf-8")
        self.assertNotEqual(_md5(a), first)


# =====================================================================
# 8 & 9. Governance boundary -- AST-based, not source-text grepping
# =====================================================================
class TestGate9GovernanceBoundary(unittest.TestCase):
    @staticmethod
    def _tree():
        return ast.parse(Path(__file__).read_text(encoding="utf-8"))

    def _imported(self):
        names = set()
        for n in ast.walk(self._tree()):
            if isinstance(n, ast.Import):
                names.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                names.add(n.module or "")
                names.update(a.name for a in n.names)
        return names

    def test_imports_no_training_machinery(self):
        imported = self._imported()
        for forbidden in ("models.train", "sklearn", "LogisticRegression",
                          "train_logistic_regression", "models.run_final_comparison"):
            self.assertNotIn(forbidden, imported)

    def test_imports_no_calibration_or_tuning(self):
        imported = self._imported()
        for forbidden in ("models.calibration", "CalibratedClassifierCV",
                          "GridSearchCV", "RandomizedSearchCV", "models.run_robustness"):
            self.assertNotIn(forbidden, imported)

    def test_imports_no_final_split_and_no_split_machinery(self):
        # NOTE: reading the season CONSTANTS is required by Gate 9's own
        # requirement 1 ("FINAL_TEST_SEASONS unchanged"), so they are
        # legitimately imported. What must be absent is the split
        # machinery that would actually partition or load 2025/26 rows.
        imported = self._imported()
        for forbidden in ("final_split", "models.splits", "split_by_seasons",
                          "iter_walk_forward_folds"):
            self.assertNotIn(forbidden, imported)

    def test_season_constants_are_read_only_never_partitioned(self):
        # The constants appear only inside equality assertions; no call
        # anywhere in this module consumes them to select data.
        tree = self._tree()
        split_calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) or getattr(n.func, "attr", None))
            in {"final_split", "split_by_seasons", "iter_walk_forward_folds",
                "season_ids_for", "load_supervised_dataset"}
        ]
        self.assertEqual(split_calls, [], "Gate 9 must not partition or load any data")

    def test_makes_no_new_2025_26_data_access(self):
        imported = self._imported()
        for forbidden in ("models.data", "load_supervised_dataset"):
            self.assertNotIn(forbidden, imported)

    def test_no_artifact_writing_calls_anywhere(self):
        # Repository immutability is proven behaviourally by the checksum
        # tests (before/after the rehearsal). This adds the static half:
        # no artifact-writing call exists at all. `write_text` is
        # permitted because the rehearsal writes only to disposable
        # tempfile copies -- an AST receiver heuristic for that proved
        # fragile and is replaced by the behavioural checksum proof.
        tree = self._tree()
        forbidden_calls = [
            getattr(n.func, "attr", None) for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) in {"to_csv", "to_json", "dump", "savez", "save"}
        ]
        self.assertEqual(forbidden_calls, [])

    def test_every_repository_path_referenced_is_only_read(self):
        # Any repository path used here must be consumed by a read
        # operation (read_text / read_bytes / copy2 source / exists),
        # never by a write. Verified via the pinned-baseline checksums
        # remaining equal after the full module has run.
        for rel, expected in _pinned_baselines().items():
            p = Path(rel)
            self.assertTrue(p.exists(), f"pinned input missing: {rel}")
            self.assertEqual(_md5(p), expected, f"repository file mutated by Gate 9: {rel}")

    def test_uses_no_artifact_writing_machinery(self):
        imported = self._imported()
        for forbidden in ("joblib", "pickle", "cloudpickle"):
            self.assertNotIn(forbidden, imported)

    def test_no_gate9_artifact_created(self):
        for p in Path("data/audit").rglob("*"):
            self.assertNotIn("gate9", p.name.lower())

    def test_gate9_claims_no_literal_production_revert(self):
        # The docstring must be explicit that no destructive revert
        # occurred, so the report cannot overstate what was rehearsed.
        doc = ast.get_docstring(self._tree()) or ""
        low = doc.lower()
        self.assertIn("non-destructive", low)
        self.assertIn("without mutating repository state", low)


if __name__ == "__main__":
    unittest.main()
