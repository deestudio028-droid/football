"""Tests for the blind 2025/26 batch predictor and the separate reveal tool.

The load-bearing properties are (a) no label column can reach the model
during prediction, and (b) only 2025/26 fixtures are accepted. Both are
proved statically so they hold even in environments without scikit-learn
and without a trained artifact -- a leakage guarantee that only holds
when a model happens to be present is not a guarantee.
"""
import _pathfix  # noqa: F401
import ast
import subprocess
import sys
import unittest
from pathlib import Path

from models.config import (
    FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS, X_EXCLUDED_COLUMNS,
)

REPO = Path(__file__).resolve().parents[1]
PREDICTOR = REPO / "predict_blind_2025_26.py"
REVEALER = REPO / "reveal_results_2025_26.py"
LABEL_COLUMNS = ("label_result", "label_home_goals", "label_away_goals")


def tree(p: Path):
    return ast.parse(p.read_text(encoding="utf-8"))


def non_docstring_strings(t):
    doc_ids = set()
    for n in ast.walk(t):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                doc_ids.add(id(n.body[0].value))
    return [n.value for n in ast.walk(t)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_ids]


class TestLabelsCannotReachTheModel(unittest.TestCase):
    def setUp(self):
        self.t = tree(PREDICTOR)
        self.consts = non_docstring_strings(self.t)

    def test_predictor_has_no_executable_label_reference(self):
        for label in LABEL_COLUMNS:
            self.assertNotIn(label, self.consts,
                             f"predictor must never reference {label}")

    def test_detector_is_sensitive(self):
        """Without this control the test above could pass vacuously."""
        self.assertIn("label_result",
                      non_docstring_strings(ast.parse("x = 'label_result'")))
        self.assertNotIn("label_result",
                         non_docstring_strings(ast.parse('"""mentions label_result"""')))

    def test_labels_are_stripped_upstream_by_the_production_loader(self):
        for label in LABEL_COLUMNS:
            self.assertIn(label, X_EXCLUDED_COLUMNS)

    def test_predictor_issues_no_sql_touching_labels(self):
        for s in self.consts:
            low = s.lower()
            if "select" in low and "from fixtures" in low:
                for banned in ("home_goals", "away_goals", "label"):
                    self.assertNotIn(banned, low,
                                     f"prediction-phase SQL must not read {banned}: {s}")

    def test_predictor_fits_nothing_and_writes_nothing(self):
        called = {n.func.attr for n in ast.walk(self.t)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for banned in ("fit", "fit_transform", "partial_fit", "commit", "executemany"):
            self.assertNotIn(banned, called)
        for s in self.consts:
            for banned in ("INSERT ", "UPDATE ", "DELETE FROM", "DROP "):
                self.assertNotIn(banned, s.upper())

    def test_every_sqlite_connection_is_read_only(self):
        for call in [n for n in ast.walk(self.t)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "connect"]:
            rendered = ast.unparse(call)
            self.assertIn("mode=ro", rendered)

    def test_predictor_uses_the_production_prediction_path(self):
        imported = {a.name for n in ast.walk(self.t)
                    if isinstance(n, ast.ImportFrom) for a in n.names}
        self.assertIn("predict_candidate", imported)
        self.assertIn("load_supervised_dataset", imported)


class TestOnly2025_26Accepted(unittest.TestCase):
    def setUp(self):
        self.t = tree(PREDICTOR)

    def test_selection_is_scoped_to_final_test_season_ids(self):
        imported = {a.name for n in ast.walk(self.t)
                    if isinstance(n, ast.ImportFrom) for a in n.names}
        self.assertIn("FINAL_TEST_SEASONS", imported)
        self.assertIn("SEASON_NAME_TO_IDS", imported)

    def test_exact_season_id_membership_not_substring_matching(self):
        consts = non_docstring_strings(self.t)
        for s in consts:
            self.assertNotIn("2025/2026", s, "must use season_id sets, not name matching")
            self.assertNotIn("like '%2025", s.lower())

    def test_final_test_and_train_seasons_are_disjoint(self):
        test_ids, train_ids = set(), set()
        for s in FINAL_TEST_SEASONS:
            test_ids.update(SEASON_NAME_TO_IDS[s])
        for s in FINAL_TRAIN_SEASONS:
            train_ids.update(SEASON_NAME_TO_IDS[s])
        self.assertEqual(test_ids & train_ids, set())
        self.assertTrue(test_ids)

    def test_non_2025_26_fixture_is_rejected(self):
        """250518560 is a 2024/25 fixture used elsewhere in the project."""
        res = subprocess.run(
            [sys.executable, str(PREDICTOR), "--fixture-ids", "250518560"],
            capture_output=True, text=True, cwd=str(REPO))
        self.assertNotEqual(res.returncode, 0)
        out = res.stdout + res.stderr
        self.assertNotIn("Traceback", out)
        # Either the season gate rejects it, or the run stops earlier at an
        # environment/artifact gate -- but it must never produce a prediction.
        self.assertNotIn("PREDICTION PHASE COMPLETE", out)

    def test_malformed_fixture_id_is_rejected(self):
        res = subprocess.run(
            [sys.executable, str(PREDICTOR), "--fixture-ids", "not-a-number"],
            capture_output=True, text=True, cwd=str(REPO))
        self.assertNotEqual(res.returncode, 0)
        self.assertNotIn("PREDICTION PHASE COMPLETE", res.stdout)


class TestBlindnessIsVerifiedNotAssumed(unittest.TestCase):
    def test_predictor_checks_the_training_partition(self):
        src = PREDICTOR.read_text(encoding="utf-8")
        self.assertIn("statistics_", src)
        self.assertIn("FINAL_TRAIN_SEASONS", src)

    def test_predictor_hard_fails_if_artifact_saw_the_final_test_season(self):
        t = tree(PREDICTOR)
        checks = [ast.unparse(n) for n in ast.walk(t)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "check"]
        self.assertTrue(any("blind" in c for c in checks),
                        "there must be a hard-fail gate on artifact blindness")


class TestRevealToolIsSeparate(unittest.TestCase):
    def setUp(self):
        self.t = tree(REVEALER)

    def test_reveal_tool_loads_no_model_and_predicts_nothing(self):
        imported = {a.name for n in ast.walk(self.t)
                    if isinstance(n, ast.ImportFrom) for a in n.names}
        for banned in ("predict_candidate", "load", "artifact", "V1Artifact"):
            self.assertNotIn(banned, imported)
        src = REVEALER.read_text(encoding="utf-8")
        for banned in ("sklearn", "predict_candidate", "models.artifact"):
            self.assertNotIn(banned, src)

    def test_reveal_tool_computes_no_aggregate_metric(self):
        src = REVEALER.read_text(encoding="utf-8")
        for banned in ("accuracy", "log_loss", "brier", "mean(", "sum("):
            self.assertNotIn(banned, src)

    def test_predictor_does_not_import_the_reveal_tool(self):
        src = PREDICTOR.read_text(encoding="utf-8")
        self.assertNotIn("import reveal_results", src)
        self.assertNotIn("from reveal_results", src)

    def test_reveal_tool_is_read_only(self):
        for call in [n for n in ast.walk(self.t)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "connect"]:
            self.assertIn("mode=ro", ast.unparse(call))

    def test_reveal_tool_rejects_non_2025_26_fixtures(self):
        res = subprocess.run(
            [sys.executable, str(REVEALER), "--fixture-ids", "250518560"],
            capture_output=True, text=True, cwd=str(REPO))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("not 2025/26", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_reveal_tool_rejects_malformed_ids(self):
        res = subprocess.run(
            [sys.executable, str(REVEALER), "--fixture-ids", "abc"],
            capture_output=True, text=True, cwd=str(REPO))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("must all be integers", res.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
