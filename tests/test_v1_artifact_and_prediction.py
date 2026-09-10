"""D-29 -- V1 model artifact and fixture-ID predictor.

Split deliberately into two groups:

  STATIC tests run everywhere, including environments without
  scikit-learn. They prove by AST/source inspection that the predictor
  cannot use labels, cannot fit, and cannot write -- the leakage-relevant
  properties, which should never depend on a trained artifact existing.

  ARTIFACT tests need a trained model and skip cleanly when none is
  present, so the suite stays green before the first training run.
"""
import _pathfix  # noqa: F401
import ast
import hashlib
import inspect
import json
import subprocess
import sys
import unittest
from pathlib import Path

from models import artifact as artifact_mod
from models.ablation import MODEL_B_COLUMNS
from models.baselines import CLASS_ORDER
from models.config import MODEL_VERSION, REQUIRED_FEATURE_VERSION

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / artifact_mod.DEFAULT_ARTIFACT_PATH
PREDICT_CLI = REPO / "predict_match.py"
TRAIN_CLI = REPO / "train_v1_model.py"
LABEL_COLUMNS = ("label_result", "label_home_goals", "label_away_goals")


def _tree(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _strings(tree):
    """Every string constant that is NOT a docstring -- docstrings
    legitimately discuss the labels they promise never to read, and a
    plain grep would flag that discussion as a violation."""
    docstrings = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                docstrings.add(id(n.body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings]


# =====================================================================
# STATIC -- no artifact, no sklearn required
# =====================================================================
class TestPredictorCannotLeak(unittest.TestCase):
    """Tests 7, 8, 9."""

    def setUp(self):
        self.tree = _tree(PREDICT_CLI)
        self.consts = _strings(self.tree)

    def test_7_8_9_predictor_never_references_a_label_column(self):
        for label in LABEL_COLUMNS:
            self.assertNotIn(label, self.consts,
                             f"predictor must never reference {label}")
        # Control: the detector must be able to see a label string if one
        # were present, or these assertions prove nothing.
        self.assertIn("label_result", _strings(ast.parse("x = 'label_result'")))

    def test_predictor_selects_exactly_the_frozen_contract(self):
        src = PREDICT_CLI.read_text(encoding="utf-8")
        self.assertIn("art.feature_columns", src)
        self.assertIn("MODEL_B_COLUMNS", src)

    def test_predictor_fits_nothing(self):
        called = {n.func.attr for n in ast.walk(self.tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for banned in ("fit", "fit_transform", "partial_fit"):
            self.assertNotIn(banned, called)

    def test_predictor_opens_no_writable_database(self):
        src = PREDICT_CLI.read_text(encoding="utf-8")
        for conn in [n for n in ast.walk(self.tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "connect"]:
            self.assertTrue(any("mode=ro" in s for s in _strings(ast.parse(ast.unparse(conn)))),
                            "every sqlite connection in the predictor must be mode=ro")
        for banned in ("INSERT", "UPDATE", "DELETE FROM", "DROP "):
            self.assertNotIn(banned, src)

    def test_predictor_derives_no_expected_goals(self):
        src = PREDICT_CLI.read_text(encoding="utf-8")
        self.assertIn("NOT AVAILABLE", src)
        for banned in ("expected_goals", "xg_pred", "poisson", "Poisson"):
            self.assertNotIn(banned, src)


class TestArtifactModuleContract(unittest.TestCase):
    def test_3_4_5_contract_version_and_class_order_are_taken_from_frozen_sources(self):
        tree = _tree(Path(inspect.getfile(artifact_mod)))
        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom) for a in n.names}
        for name in ("MODEL_B_COLUMNS", "MODEL_VERSION", "CLASS_ORDER"):
            self.assertIn(name, imported)
        self.assertEqual(len(MODEL_B_COLUMNS), 80)
        self.assertEqual(MODEL_VERSION, "v1.0")
        self.assertEqual(REQUIRED_FEATURE_VERSION, "v1.0")
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])

    def test_save_refuses_a_mismatched_contract(self):
        with self.assertRaises(artifact_mod.ArtifactError):
            artifact_mod._verify(("a", "b"), "v1.0", CLASS_ORDER, "test")
        with self.assertRaises(artifact_mod.ArtifactError):
            artifact_mod._verify(MODEL_B_COLUMNS, "v9.9", CLASS_ORDER, "test")
        with self.assertRaises(artifact_mod.ArtifactError):
            artifact_mod._verify(MODEL_B_COLUMNS, "v1.0", ["A", "D", "H"], "test")

    def test_reordered_contract_is_rejected_with_a_specific_message(self):
        swapped = (MODEL_B_COLUMNS[1], MODEL_B_COLUMNS[0]) + tuple(MODEL_B_COLUMNS[2:])
        with self.assertRaises(artifact_mod.ArtifactError) as cm:
            artifact_mod._verify(swapped, "v1.0", CLASS_ORDER, "test")
        self.assertIn("ORDER", str(cm.exception))

    def test_1_missing_artifact_fails_cleanly(self):
        with self.assertRaises(artifact_mod.ArtifactError) as cm:
            artifact_mod.load(REPO / "data/models/does_not_exist.pkl")
        self.assertIn("train one first", str(cm.exception).lower())

    def test_artifact_path_contains_no_v2(self):
        self.assertNotIn("v2", str(artifact_mod.DEFAULT_ARTIFACT_PATH).lower())


class TestFrozenArtifactsUntouched(unittest.TestCase):
    def test_pinned_inputs_unchanged(self):
        pins = json.loads(
            (REPO / "data/audit/phase4c_prerun_manifest.json").read_text()
        )["locked_input_checksums"]
        for rel, v in pins.items():
            self.assertEqual(hashlib.md5((REPO / rel).read_bytes()).hexdigest(),
                             v["expected"], f"locked artifact modified: {rel}")

    def test_databases_unchanged(self):
        for rel, exp in (("data/processed/features.db", "e7ebe7fc07040a5927683c35b6371e63"),
                         ("data/processed/matches.db", "fdeed042096fa1c851aaee6c84995247")):
            self.assertEqual(hashlib.md5((REPO / rel).read_bytes()).hexdigest(), exp, rel)

    def test_no_possession_feature_reachable_from_the_v1_path(self):
        for cli in (TRAIN_CLI, PREDICT_CLI):
            src = cli.read_text(encoding="utf-8")
            for banned in ("POSSESSION_FAMILY_COLUMNS", "MODEL_B_PLUS_POSSESSION_COLUMNS",
                           "features_v1_1", "successor_data"):
                self.assertNotIn(banned, src, f"{cli.name} must not use {banned}")

    def test_trainer_uses_the_production_estimator_only(self):
        tree = _tree(TRAIN_CLI)
        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom) for a in n.names}
        self.assertIn("train_logistic_regression", imported)
        self.assertIn("load_supervised_dataset", imported)
        # The estimator must never be CONSTRUCTED here -- checked via AST,
        # because the module docstring legitimately names the frozen
        # configuration and a substring search would flag that prose.
        constructed = [n for n in ast.walk(tree)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id in ("LogisticRegression",
                                         "HistGradientBoostingClassifier",
                                         "SimpleImputer", "StandardScaler")]
        self.assertEqual(constructed, [],
                         "trainer must reuse the production path, not rebuild it")
        # Control: the detector must still catch an actual construction.
        self.assertEqual(
            len([n for n in ast.walk(ast.parse("m = LogisticRegression(C=1.0)"))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "LogisticRegression"]), 1)


# =====================================================================
# ARTIFACT -- require a trained model; skip cleanly when absent
# =====================================================================
@unittest.skipUnless(ARTIFACT.exists(), "no trained artifact yet: run train_v1_model.py")
class TestTrainedArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("scikit-learn not installed")
        cls.art = artifact_mod.load(ARTIFACT)

    def test_1_2_artifact_exists_and_reloads(self):
        self.assertTrue(ARTIFACT.exists())
        self.assertIsNotNone(self.art.model)
        self.assertIsNotNone(self.art.preprocessor)

    def test_3_artifact_carries_the_80_column_contract_in_order(self):
        self.assertEqual(tuple(self.art.feature_columns), tuple(MODEL_B_COLUMNS))
        self.assertEqual(len(self.art.feature_columns), 80)

    def test_4_5_version_and_class_order(self):
        self.assertEqual(self.art.model_version, "v1.0")
        self.assertEqual(self.art.class_order, ["H", "D", "A"])

    def test_estimator_configuration_is_the_frozen_one(self):
        self.assertEqual(self.art.model.C, 1.0)
        self.assertEqual(self.art.model.max_iter, 2000)
        self.assertEqual(self.art.model.random_state, 0)

    def test_2_reloads_in_a_fresh_process(self):
        code = (
            "import sys; sys.path.insert(0, r'%s')\n"
            "from models.artifact import load\n"
            "a = load(r'%s')\n"
            "print(len(a.feature_columns), a.model_version, '/'.join(a.class_order))\n"
        ) % (REPO / "src", ARTIFACT)
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), "80 v1.0 H/D/A")

    def _predict(self, fixture_id):
        res = subprocess.run(
            [sys.executable, str(PREDICT_CLI), "--fixture-id", str(fixture_id)],
            capture_output=True, text=True, cwd=str(REPO))
        return res

    def _a_real_fixture_id(self):
        from models.data import load_supervised_dataset
        ds = load_supervised_dataset(REPO / "data/processed/features.db")
        return int(ds.metadata["fixture_id"].iloc[0])

    def test_6_fixture_lookup_works(self):
        res = self._predict(self._a_real_fixture_id())
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("MATCH PREDICTION", res.stdout)
        self.assertIn("Predicted Result :", res.stdout)
        self.assertIn("NOT AVAILABLE", res.stdout)

    def test_10_repeated_prediction_is_exactly_deterministic(self):
        fid = self._a_real_fixture_id()
        a, b = self._predict(fid), self._predict(fid)
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(a.stdout, b.stdout)      # exact, not tolerance-based

    def test_11_missing_fixture_id_fails_cleanly(self):
        res = self._predict(999999999)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("no feature row", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_12_malformed_fixture_id_fails_cleanly(self):
        res = self._predict("not-a-number")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("must be an integer", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_probabilities_are_valid_and_in_class_order(self):
        from models.candidate_contract import predict_candidate
        from models.data import load_supervised_dataset
        ds = load_supervised_dataset(REPO / "data/processed/features.db")
        X = ds.X.head(5)[list(self.art.feature_columns)]
        out = predict_candidate(self.art.model, self.art.preprocessor, X,
                                ds.metadata["fixture_id"].head(5).tolist())
        self.assertEqual(out.class_order, ["H", "D", "A"])
        self.assertEqual(out.probabilities.shape, (5, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
