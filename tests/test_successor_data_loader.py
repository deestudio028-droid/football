"""D-23 -- successor (v1.1) data loader.

Two things must hold together:

  1. V1 IS UNTOUCHED. `models.data` and `models.config` are byte-
     identical, REQUIRED_FEATURE_VERSION is still "v1.0", and the V1
     loader still accepts v1.0 and rejects everything else exactly as
     before.

  2. THE SUCCESSOR LOADER IS A PARITY PATH, NOT A BYPASS. It reuses the
     production read-only connection, label whitelist, unlabeled-fixture
     check and feature-column rules; the only behavioural difference is
     which feature_version it demands.

The parity proof is deliberately doubled: a behavioural table covering
all four version scenarios for both loaders, and an AST comparison of
the two version-check functions with the constant name normalised. The
behavioural table proves they behave alike today; the AST comparison
proves one cannot quietly drift from the other tomorrow.
"""
import _pathfix  # noqa: F401
import ast
import hashlib
import inspect
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from features.config import SUCCESSOR_FEATURE_VERSION
from models import data as v1_data
from models import successor_data as sd
from models.ablation import MODEL_B_COLUMNS
from models.config import REQUIRED_FEATURE_VERSION
from models.data import FeatureVersionError, ModelDataError, UnexpectedLabelError
from models.successor_contract import MODEL_B_PLUS_POSSESSION_COLUMNS

REPO = Path(__file__).resolve().parents[1]

METADATA = ["fixture_id", "competition_id", "season_id", "unix",
            "feature_version", "generated_at", "home_id", "away_id"]
LABELS = ["label_home_goals", "label_away_goals", "label_result"]
FEATURES = ["home_goals_for_per_match_season", "away_goals_for_per_match_season"]


def make_db(tmp: Path, rows: list[dict], name="f.db") -> Path:
    """Minimal feature_rows table shaped like the production schema."""
    cols = METADATA + LABELS + FEATURES
    quoted = ", ".join('"%s"' % c for c in cols)          # 3.10-compatible
    marks = ", ".join("?" for _ in cols)
    p = tmp / name
    con = sqlite3.connect(str(p))
    con.execute("CREATE TABLE feature_rows (%s);" % quoted)
    con.executemany(
        "INSERT INTO feature_rows (%s) VALUES (%s);" % (quoted, marks),
        [tuple(r.get(c) for c in cols) for r in rows])
    con.commit()
    con.close()
    return p


def row(fid, version, label="H", **over):
    r = {"fixture_id": fid, "competition_id": 1, "season_id": 100, "unix": 1000 * fid,
         "feature_version": version, "generated_at": "2026-01-01T00:00:00+00:00",
         "home_id": 1, "away_id": 2,
         "label_home_goals": 2, "label_away_goals": 1, "label_result": label,
         "home_goals_for_per_match_season": 1.5,
         "away_goals_for_per_match_season": None}
    r.update(over)
    return r


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()


# =====================================================================
# A-D, M -- version handling
# =====================================================================
class TestVersionHandling(_Tmp):
    def test_a_v1_1_db_loads_successfully(self):
        db = make_db(self.tmp, [row(1, "v1.1"), row(2, "v1.1", label="D")])
        ds = sd.load_successor_supervised_dataset(db)
        self.assertEqual(len(ds), 2)
        self.assertEqual(sorted(ds.y.tolist()), ["D", "H"])

    def test_b_v1_0_db_is_rejected_by_successor_loader(self):
        db = make_db(self.tmp, [row(1, "v1.0"), row(2, "v1.0")])
        with self.assertRaises(FeatureVersionError) as cm:
            sd.load_successor_supervised_dataset(db)
        self.assertIn("v1.0", str(cm.exception))
        self.assertIn("v1.1", str(cm.exception))

    def test_b_inverse_v1_1_db_is_rejected_by_the_v1_loader(self):
        """The isolation must cut both ways, or the two paths are not
        actually separated."""
        db = make_db(self.tmp, [row(1, "v1.1")])
        with self.assertRaises(FeatureVersionError):
            v1_data.load_supervised_dataset(db)

    def test_c_mixed_versions_are_rejected(self):
        db = make_db(self.tmp, [row(1, "v1.0"), row(2, "v1.1")])
        for loader in (sd.load_successor_supervised_dataset,
                       v1_data.load_supervised_dataset):
            with self.assertRaises(FeatureVersionError) as cm:
                loader(db)
            self.assertIn("multiple feature_version values", str(cm.exception))

    def test_d_zero_version_rows_are_rejected(self):
        db = make_db(self.tmp, [row(1, None), row(2, None)])
        for loader in (sd.load_successor_supervised_dataset,
                       v1_data.load_supervised_dataset):
            with self.assertRaises(FeatureVersionError) as cm:
                loader(db)
            self.assertIn("no rows with a feature_version set", str(cm.exception))

    def test_d_unexpected_version_is_rejected(self):
        db = make_db(self.tmp, [row(1, "v9.9")])
        with self.assertRaises(FeatureVersionError):
            sd.load_successor_supervised_dataset(db)

    def test_m_successor_output_has_exactly_one_feature_version(self):
        db = make_db(self.tmp, [row(i, "v1.1") for i in range(1, 6)])
        ds = sd.load_successor_supervised_dataset(db)
        self.assertEqual(set(ds.metadata["feature_version"].unique()), {"v1.1"})

    def test_successor_constant_matches_what_the_builder_stamps(self):
        self.assertEqual(sd.SUCCESSOR_REQUIRED_FEATURE_VERSION, "v1.1")
        self.assertEqual(sd.SUCCESSOR_REQUIRED_FEATURE_VERSION, SUCCESSOR_FEATURE_VERSION)
        self.assertNotEqual(sd.SUCCESSOR_REQUIRED_FEATURE_VERSION, REQUIRED_FEATURE_VERSION)

    def test_load_successor_feature_rows_is_unfiltered(self):
        db = make_db(self.tmp, [row(1, "v1.1"), row(2, "v1.1", label=None)])
        self.assertEqual(len(sd.load_successor_feature_rows(db)), 2)
        self.assertEqual(len(sd.load_successor_supervised_dataset(db)), 1)


# =====================================================================
# E, F -- the non-version checks must behave identically
# =====================================================================
class TestNonVersionChecksPreserved(_Tmp):
    def test_e_invalid_label_rejected_exactly_as_v1_rejects_it(self):
        s_db = make_db(self.tmp, [row(1, "v1.1", label="X")], "s.db")
        v_db = make_db(self.tmp, [row(1, "v1.0", label="X")], "v.db")
        with self.assertRaises(UnexpectedLabelError) as s_err:
            sd.load_successor_supervised_dataset(s_db)
        with self.assertRaises(UnexpectedLabelError) as v_err:
            v1_data.load_supervised_dataset(v_db)
        self.assertEqual(str(s_err.exception), str(v_err.exception))

    def test_e_known_unlabeled_fixture_check_is_active(self):
        from models.config import KNOWN_UNLABELED_FIXTURE_IDS
        bad = sorted(KNOWN_UNLABELED_FIXTURE_IDS)[0]
        db = make_db(self.tmp, [row(bad, "v1.1")])
        with self.assertRaises(ModelDataError) as cm:
            sd.load_successor_supervised_dataset(db)
        self.assertIn("ABANDONED", str(cm.exception))

    def test_f_missing_required_column_is_rejected(self):
        p = self.tmp / "bad.db"
        con = sqlite3.connect(str(p))
        con.execute('CREATE TABLE feature_rows ("fixture_id", "label_result");')
        con.execute("INSERT INTO feature_rows VALUES (1, 'H');")
        con.commit(); con.close()
        with self.assertRaises(Exception):
            sd.load_successor_supervised_dataset(p)

    def test_no_blanket_dropna_null_features_survive(self):
        db = make_db(self.tmp, [row(1, "v1.1")])
        ds = sd.load_successor_supervised_dataset(db)
        self.assertEqual(len(ds), 1)
        self.assertTrue(pd.isna(ds.X["away_goals_for_per_match_season"].iloc[0]))

    def test_column_selection_rules_are_the_production_ones(self):
        db = make_db(self.tmp, [row(1, "v1.1")])
        ds = sd.load_successor_supervised_dataset(db)
        for excluded in ("season_id", "unix", "home_id", "away_id", "fixture_id",
                         "generated_at", "feature_version", *LABELS):
            self.assertNotIn(excluded, ds.X.columns)
        self.assertIn("competition_id", ds.X.columns)   # deliberately kept
        self.assertEqual(list(ds.metadata.columns),
                         ["fixture_id", "competition_id", "season_id", "unix",
                          "feature_version"])


# =====================================================================
# 5 -- NO BYPASS: the successor path uses the production machinery
# =====================================================================
class TestNoBypass(unittest.TestCase):
    def _tree(self):
        return ast.parse(inspect.getsource(sd))

    def test_reuses_production_helpers_rather_than_reimplementing(self):
        imported = {a.name for n in ast.walk(self._tree())
                    if isinstance(n, ast.ImportFrom) for a in n.names}
        for name in ("_read_only_connection", "_verify_allowed_labels",
                     "get_feature_columns", "SupervisedDataset",
                     "FeatureVersionError", "ModelDataError"):
            self.assertIn(name, imported, f"{name} must be reused, not redefined")

    def test_does_not_open_its_own_database_connection(self):
        """A raw sqlite3.connect here would mean the successor arm was
        loading through a different path from the control arm."""
        src = inspect.getsource(sd)
        tree = ast.parse(src)
        modules = {a.name.split(".")[0] for n in ast.walk(tree)
                   if isinstance(n, ast.Import) for a in n.names}
        self.assertNotIn("sqlite3", modules)
        calls = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertNotIn("connect", calls)

    def test_defines_no_duplicate_of_the_reused_helpers(self):
        defined = {n.name for n in ast.walk(self._tree())
                   if isinstance(n, ast.FunctionDef)}
        for banned in ("_read_only_connection", "_verify_allowed_labels",
                       "get_feature_columns"):
            self.assertNotIn(banned, defined)

    def test_imports_no_estimator_and_no_final_test_constant(self):
        src = inspect.getsource(sd)
        for banned in ("sklearn", "LogisticRegression", "FINAL_TEST_SEASONS",
                       "2025/2026", "MODEL_VERSION"):
            self.assertNotIn(banned, src.split('"""')[2])

    def test_s_module_name_contains_no_v2(self):
        self.assertNotIn("v2", Path(sd.__file__).stem.lower())


# =====================================================================
# 8 -- VALIDATION PARITY
# =====================================================================
class TestValidationParity(_Tmp):
    SCENARIOS = ("none", "multiple", "wrong", "correct")

    def _version_outcome(self, loader, db):
        try:
            loader(db)
            return "OK"
        except FeatureVersionError as e:
            return str(e).split(":")[0].split(".")[0][:40]

    def test_behavioural_parity_across_all_version_scenarios(self):
        """Same four scenarios, same outcomes -- with 'correct' meaning
        v1.0 for V1 and v1.1 for the successor."""
        results = {}
        for label, own, other in (("v1", "v1.0", "v1.1"), ("succ", "v1.1", "v1.0")):
            loader = (v1_data.load_supervised_dataset if label == "v1"
                      else sd.load_successor_supervised_dataset)
            cases = {
                "none": [row(1, None)],
                "multiple": [row(1, own), row(2, other)],
                "wrong": [row(1, other)],
                "correct": [row(1, own)],
            }
            results[label] = {
                s: self._version_outcome(loader, make_db(self.tmp, cases[s], f"{label}_{s}.db"))
                for s in self.SCENARIOS
            }
        self.assertEqual(results["v1"], results["succ"])
        self.assertEqual(results["v1"]["correct"], "OK")
        for s in ("none", "multiple", "wrong"):
            self.assertNotEqual(results["v1"][s], "OK")

    def test_structural_parity_of_the_two_version_checks(self):
        """The successor check must be branch-for-branch identical to the
        V1 original once the constant name is normalised -- so neither
        can gain or lose a guard without this failing."""
        def normalise(fn, const):
            tree = ast.parse(inspect.getsource(fn).strip())
            for n in ast.walk(tree):
                if isinstance(n, ast.Name) and n.id == const:
                    n.id = "REQUIRED"
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    n.name = "check"
                    if (n.body and isinstance(n.body[0], ast.Expr)
                            and isinstance(n.body[0].value, ast.Constant)):
                        n.body = n.body[1:]          # drop docstring
                if isinstance(n, ast.JoinedStr):     # normalise message f-strings
                    n.values = [v for v in n.values if not isinstance(v, ast.FormattedValue)]
            return ast.dump(tree)

        self.assertEqual(
            normalise(v1_data._verify_single_feature_version, "REQUIRED_FEATURE_VERSION"),
            normalise(sd._verify_single_successor_feature_version,
                      "SUCCESSOR_REQUIRED_FEATURE_VERSION"),
        )


# =====================================================================
# G-L -- contracts and frozen artifacts
# =====================================================================
class TestContractsAndArtifacts(unittest.TestCase):
    def test_g_control_contract_is_exactly_80(self):
        self.assertEqual(len(MODEL_B_COLUMNS), 80)

    def test_h_successor_contract_is_exactly_86(self):
        self.assertEqual(len(MODEL_B_PLUS_POSSESSION_COLUMNS), 86)
        self.assertEqual(tuple(MODEL_B_PLUS_POSSESSION_COLUMNS[:80]), tuple(MODEL_B_COLUMNS))

    def test_i_locked_inputs_unchanged(self):
        pins = json.loads(
            (REPO / "data/audit/phase4c_prerun_manifest.json").read_text()
        )["locked_input_checksums"]
        self.assertEqual(len(pins), 13)
        for rel, v in pins.items():
            got = hashlib.md5((REPO / rel).read_bytes()).hexdigest()
            self.assertEqual(got, v["expected"], f"locked artifact modified: {rel}")

    def test_j_k_databases_unchanged(self):
        for rel, exp in (("data/processed/features.db", "e7ebe7fc07040a5927683c35b6371e63"),
                         ("data/processed/matches.db", "fdeed042096fa1c851aaee6c84995247")):
            self.assertEqual(
                hashlib.md5((REPO / rel).read_bytes()).hexdigest(), exp, rel)

    def test_v1_constants_unchanged(self):
        from models.config import MODEL_VERSION
        self.assertEqual(REQUIRED_FEATURE_VERSION, "v1.0")
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_l_no_final_test_data_introduced(self):
        """The successor database this loader targets was built with
        2025/26 excluded (D-21); confirm it still contains none."""
        from models.config import FINAL_TEST_SEASONS, SEASON_NAME_TO_IDS
        ids = set()
        for s in FINAL_TEST_SEASONS:
            ids.update(SEASON_NAME_TO_IDS[s])
        db = REPO / "data/processed/features_v1_1.db"
        if not db.exists():
            self.skipTest("successor database not present")
        import shutil
        with tempfile.TemporaryDirectory() as t:
            local = Path(t) / db.name
            shutil.copyfile(db, local)
            con = sqlite3.connect(f"file:{local}?mode=ro", uri=True)
            try:
                q = ",".join(str(i) for i in sorted(ids))
                n = con.execute(
                    f"SELECT COUNT(*) FROM feature_rows WHERE season_id IN ({q})"
                ).fetchone()[0]
                versions = {r[0] for r in con.execute(
                    "SELECT DISTINCT feature_version FROM feature_rows")}
            finally:
                con.close()
        self.assertEqual(n, 0)
        self.assertEqual(versions, {"v1.1"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
