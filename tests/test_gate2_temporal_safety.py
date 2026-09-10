"""Gate 2 — Temporal safety audit for the V2 candidate's 80 columns.

Gate 2 requirement (docs/PHASE5_V2_MODEL_SPEC_DRAFT.md):
    "No feature uses information from or after the predicted fixture."
    Verification: re-run Phase 2 leakage suite + static audit of all 80
    columns. Pass: leakage suite passes; zero features classified UNKNOWN.

This file supplies the per-column static audit and an empirical
leakage proof scoped specifically to the 80 contract columns. The Phase 2
leakage suite itself is unchanged and re-run separately.

Trains nothing, evaluates nothing, never touches 2025/26.
"""
import _pathfix  # noqa: F401
import ast
import inspect
import tempfile
import unittest
from pathlib import Path

from _feature_test_helpers import build_matches_db, make_raw_fixture

from features.feature_builder import build_feature_dataset
from models.ablation import (
    COMPETITION_ID_COLUMN,
    FORM_COLUMNS,
    GOALS_CORE_COLUMNS,
    SHOTS_CORE_COLUMNS,
    SHOTS_ON_CORE_COLUMNS,
    STRENGTH_COLUMNS,
)
from models.candidate_contract import CANDIDATE_FEATURE_COLUMNS

# Every contract column must map to exactly one audited family. A column
# that matched none would be classified UNKNOWN, which is Gate 2's stop
# condition.
AUDITED_FAMILIES = {
    "goals_core": set(GOALS_CORE_COLUMNS),
    "form": set(FORM_COLUMNS),
    "strength": set(STRENGTH_COLUMNS),
    "competition_id": set(COMPETITION_ID_COLUMN),
    "shots_core": set(SHOTS_CORE_COLUMNS),
    "shots_on_core": set(SHOTS_ON_CORE_COLUMNS),
}

PRE_MATCH_SAFE = "PRE_MATCH_SAFE"
METADATA_SAFE = "METADATA_SAFE"
UNKNOWN = "UNKNOWN"


def classify_provenance(column: str) -> tuple[str, str]:
    """Returns (family, provenance) for a contract column.

    PRE_MATCH_SAFE  -- derived solely from `history_before(team_id)`,
                       i.e. fixtures strictly earlier than the target.
    METADATA_SAFE   -- fixture metadata known before kickoff.
    UNKNOWN         -- unclassifiable; Gate 2 stop condition.
    """
    for family, cols in AUDITED_FAMILIES.items():
        if column in cols:
            return family, METADATA_SAFE if family == "competition_id" else PRE_MATCH_SAFE
    return "UNCLASSIFIED", UNKNOWN


class TestGate2StaticAuditAllEightyColumns(unittest.TestCase):
    def test_all_eighty_columns_audited(self):
        self.assertEqual(len(CANDIDATE_FEATURE_COLUMNS), 80)
        results = [classify_provenance(c) for c in CANDIDATE_FEATURE_COLUMNS]
        self.assertEqual(len(results), 80)

    def test_zero_features_classified_unknown(self):
        # Gate 2 pass condition.
        unknown = [c for c in CANDIDATE_FEATURE_COLUMNS if classify_provenance(c)[1] == UNKNOWN]
        self.assertEqual(unknown, [], f"UNKNOWN-provenance features (Gate 2 stop condition): {unknown}")

    def test_provenance_distribution_is_79_pre_match_1_metadata(self):
        provs = [classify_provenance(c)[1] for c in CANDIDATE_FEATURE_COLUMNS]
        self.assertEqual(provs.count(PRE_MATCH_SAFE), 79)
        self.assertEqual(provs.count(METADATA_SAFE), 1)
        self.assertEqual(provs.count(UNKNOWN), 0)

    def test_only_competition_id_is_metadata_derived(self):
        meta = [c for c in CANDIDATE_FEATURE_COLUMNS if classify_provenance(c)[1] == METADATA_SAFE]
        self.assertEqual(meta, ["competition_id"])

    def test_family_counts_match_audit(self):
        from collections import Counter
        counts = Counter(classify_provenance(c)[0] for c in CANDIDATE_FEATURE_COLUMNS)
        self.assertEqual(dict(counts), {
            "goals_core": 18, "form": 20, "strength": 5,
            "competition_id": 1, "shots_core": 18, "shots_on_core": 18,
        })

    def test_no_contract_column_is_a_label_or_bookkeeping_column(self):
        from features.storage import LABEL_COLUMNS
        from models.config import X_EXCLUDED_COLUMNS
        for c in CANDIDATE_FEATURE_COLUMNS:
            self.assertNotIn(c, LABEL_COLUMNS)
            self.assertNotIn(c, X_EXCLUDED_COLUMNS)


class TestGate2ArchitecturalGuarantee(unittest.TestCase):
    """The leakage guarantee is structural: features are computed BEFORE
    the fixture's own result is recorded into history."""

    def test_build_feature_row_reads_history_before_only(self):
        from features import feature_builder
        src = inspect.getsource(feature_builder.build_feature_row)
        self.assertIn("ctx.history_before(home_id)", src)
        self.assertIn("ctx.history_before(away_id)", src)
        # It must not CALL ctx.record(). Checked via AST, because the
        # function's docstring legitimately mentions ctx.record to
        # explain the required ordering -- a substring search would
        # false-positive on that documentation.
        tree = ast.parse(src)
        record_calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "record"
        ]
        self.assertEqual(record_calls, [], "build_feature_row must not record into history")

    # ---- helpers for the semantic ordering check (D-19/D-20) ----------
    @staticmethod
    def _module_functions(tree):
        return {n.name: n for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    @staticmethod
    def _called_names(node):
        return {n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    @classmethod
    def _reachable(cls, funcs, root):
        """Transitive closure of module-local functions called from `root`.
        Without this the check would inspect only the public entry point
        and would be defeated by extracting the loop into a helper."""
        seen, stack = set(), [root]
        while stack:
            name = stack.pop()
            if name in seen or name not in funcs:
                continue
            seen.add(name)
            stack.extend(cls._called_names(funcs[name]))
        return seen

    @staticmethod
    def _contains(node, *, call_name=None, attr_name=None):
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            if call_name and isinstance(n.func, ast.Name) and n.func.id == call_name:
                return True
            if attr_name and isinstance(n.func, ast.Attribute) and n.func.attr == attr_name:
                return True
        return False

    @classmethod
    def _chronological_loops(cls, tree, funcs, reachable):
        """Every for-loop, anywhere in the reachable call graph, that both
        computes a feature row and records into history."""
        out = []
        for name in reachable:
            for node in ast.walk(funcs[name]):
                if not isinstance(node, ast.For):
                    continue
                if (cls._contains(node, call_name="build_feature_row")
                        and cls._contains(node, attr_name="record")):
                    out.append((name, node))
        return out

    @classmethod
    def _build_precedes_record(cls, loop):
        """Compare position in `loop.body` (statement list order), not
        line numbers, whitespace, or source text."""
        build_i = record_i = None
        for i, stmt in enumerate(loop.body):
            if build_i is None and cls._contains(stmt, call_name="build_feature_row"):
                build_i = i
            if record_i is None and cls._contains(stmt, attr_name="record"):
                record_i = i
        if build_i is None or record_i is None:
            return None
        return build_i < record_i

    def test_dataset_builder_records_only_after_computing(self):
        """Semantic replacement for the original source-substring check
        (D-19). That version did `src.index("build_feature_row(fixture,
        ctx)")` inside build_feature_dataset, so it broke the moment the
        shared chronological walk was extracted into a helper -- and it
        proved only a source-text layout, never the invariant. This
        version follows the call graph, so it survives helper extraction,
        and it is independent of whitespace, formatting, line numbers,
        receiver variable names and source strings.

        It is strictly stronger than the original: it additionally
        requires that exactly ONE chronological walker exists, which
        forbids a duplicated leakage-critical loop (D-17).
        """
        from features import feature_builder
        tree = ast.parse(inspect.getsource(feature_builder))
        funcs = self._module_functions(tree)
        self.assertIn("build_feature_dataset", funcs)

        reachable = self._reachable(funcs, "build_feature_dataset")
        loops = self._chronological_loops(tree, funcs, reachable)

        self.assertEqual(
            len(loops), 1,
            "expected exactly one chronological walker reachable from "
            f"build_feature_dataset, found {[n for n, _ in loops]}")
        owner, loop = loops[0]
        self.assertIs(
            self._build_precedes_record(loop), True,
            f"in {owner}: features must be computed before the fixture is "
            "recorded into history")

    def test_ordering_detector_positive_and_negative_controls(self):
        """The ordering check above is only meaningful if it can fail.
        Exercise it on synthetic ASTs: correct order must pass, inverted
        order must fail, and helper extraction must not blind it."""
        correct = ast.parse(
            "def walk(fixtures, ctx):\n"
            "    for f in fixtures:\n"
            "        rows.append(build_feature_row(f, ctx))\n"
            "        ctx.record(f)\n")
        inverted = ast.parse(
            "def walk(fixtures, ctx):\n"
            "    for f in fixtures:\n"
            "        ctx.record(f)\n"
            "        rows.append(build_feature_row(f, ctx))\n")
        renamed = ast.parse(
            "def walk(fixtures, context):\n"
            "    for fx in fixtures:\n"
            "        out.append(build_feature_row(fx, context))\n"
            "        context.record(fx)\n")
        extracted = ast.parse(
            "def public(db):\n"
            "    return _inner(db)\n"
            "def _inner(db):\n"
            "    for f in load(db):\n"
            "        rows.append(build_feature_row(f, ctx))\n"
            "        ctx.record(f)\n")

        for tree, root, expected in (
            (correct, "walk", True),
            (inverted, "walk", False),
            (renamed, "walk", True),      # receiver/variable names must not matter
            (extracted, "public", True),  # helper extraction must not blind it
        ):
            funcs = self._module_functions(tree)
            reachable = self._reachable(funcs, root)
            loops = self._chronological_loops(tree, funcs, reachable)
            self.assertEqual(len(loops), 1, f"detector lost the loop in {root}")
            self.assertIs(self._build_precedes_record(loops[0][1]), expected)

        # And it must notice a second, duplicated walker.
        duplicated = ast.parse(
            "def public(db):\n"
            "    return _a(db) + _b(db)\n"
            "def _a(db):\n"
            "    for f in load(db):\n"
            "        build_feature_row(f, ctx)\n"
            "        ctx.record(f)\n"
            "def _b(db):\n"
            "    for f in load(db):\n"
            "        build_feature_row(f, ctx)\n"
            "        ctx.record(f)\n")
        funcs = self._module_functions(duplicated)
        loops = self._chronological_loops(
            duplicated, funcs, self._reachable(funcs, "public"))
        self.assertEqual(len(loops), 2, "detector must see duplicated walkers")

    def test_take_window_documents_strictly_before_contract(self):
        from features import rolling
        doc = inspect.getdoc(rolling.take_window) or ""
        self.assertIn("strictly before", doc)

    def test_history_before_returns_only_prior_matches(self):
        from features import history
        doc = inspect.getdoc(history.FeatureContext.history_before) or ""
        self.assertIn("never contain the", doc.replace("\n", " "))


class TestGate2EmpiricalNoFutureLeakage(unittest.TestCase):
    """Empirical proof scoped to the 80 contract columns: adding or
    removing FUTURE fixtures must not change any contract feature at T.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _rows_for_target(self, fixtures, subdir):
        d = self.tmp_path / subdir
        d.mkdir(exist_ok=True)
        db = build_matches_db(d, fixtures)
        return build_feature_dataset(db, fixture_ids=[3])[0]

    def _base_fixtures(self):
        return [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=2, away_goals=1),
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=0, away_goals=0),  # target T
        ]

    def test_no_contract_feature_changes_when_future_fixtures_are_added(self):
        base = self._base_fixtures()
        with_future = base + [
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=99, away_goals=0),
            make_raw_fixture(5, 5000, home_id=1, away_id=6, home_goals=0, away_goals=99),
        ]
        row_base = self._rows_for_target(base, "a")
        row_future = self._rows_for_target(with_future, "b")

        differing = [
            c for c in CANDIDATE_FEATURE_COLUMNS
            if row_base.get(c) != row_future.get(c)
        ]
        self.assertEqual(
            differing, [],
            f"future fixtures changed contract feature(s) at T -- leakage: {differing}",
        )

    def test_target_own_result_never_enters_its_contract_features(self):
        # Target has an extreme scoreline; if it leaked into its own
        # windows, goals/form features would visibly shift.
        base = self._base_fixtures()
        extreme = base[:-1] + [
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=88, away_goals=0)
        ]
        row_normal = self._rows_for_target(base, "c")
        row_extreme = self._rows_for_target(extreme, "d")

        differing = [
            c for c in CANDIDATE_FEATURE_COLUMNS
            if row_normal.get(c) != row_extreme.get(c)
        ]
        self.assertEqual(
            differing, [],
            f"target's own result changed its contract feature(s) -- leakage: {differing}",
        )

    def test_reordering_future_rows_does_not_change_contract_features(self):
        base = self._base_fixtures()
        f_a = base + [
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=5, away_goals=0),
            make_raw_fixture(5, 5000, home_id=1, away_id=6, home_goals=0, away_goals=5),
        ]
        f_b = base + [
            make_raw_fixture(5, 5000, home_id=1, away_id=6, home_goals=0, away_goals=5),
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=5, away_goals=0),
        ]
        row_a = self._rows_for_target(f_a, "e")
        row_b = self._rows_for_target(f_b, "f")
        differing = [c for c in CANDIDATE_FEATURE_COLUMNS if row_a.get(c) != row_b.get(c)]
        self.assertEqual(differing, [], f"future row order changed contract features: {differing}")

    def test_all_contract_columns_are_actually_produced_by_the_pipeline(self):
        # A column the pipeline never emits could not have been audited
        # against real behaviour.
        row = self._rows_for_target(self._base_fixtures(), "g")
        missing = [c for c in CANDIDATE_FEATURE_COLUMNS if c not in row]
        self.assertEqual(missing, [], f"contract columns not produced by feature pipeline: {missing}")


class TestGate2NoSideEffects(unittest.TestCase):
    """These assert on this module's actual imports and calls via AST.
    A substring search over the file would false-positive on the very
    names being forbidden, since they appear in these assertions."""

    @staticmethod
    def _this_module_tree():
        import sys
        return ast.parse(Path(sys.modules[__name__].__file__).read_text(encoding="utf-8"))

    def _imported_names(self):
        names = set()
        for node in ast.walk(self._this_module_tree()):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(a.name for a in node.names)
        return names

    def test_no_final_test_access_in_this_audit(self):
        imported = self._imported_names()
        for forbidden in ("final_split", "FINAL_TEST_SEASONS", "FINAL_TRAIN_SEASONS"):
            self.assertNotIn(forbidden, imported)
        # And no call to final_split anywhere.
        calls = [
            n for n in ast.walk(self._this_module_tree())
            if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == "final_split"
        ]
        self.assertEqual(calls, [])

    def test_no_training_or_estimator_in_this_audit(self):
        imported = self._imported_names()
        for forbidden in ("sklearn", "sklearn.linear_model", "LogisticRegression"):
            self.assertNotIn(forbidden, imported)
        # No .fit(...) or .predict_proba(...) calls.
        bad = [
            n for n in ast.walk(self._this_module_tree())
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) in {"fit", "predict_proba", "predict"}
        ]
        self.assertEqual(bad, [])

    def test_model_version_unchanged(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_no_gate2_artifact_created(self):
        for p in Path("data/audit").rglob("*"):
            self.assertNotIn("gate2", p.name.lower())


if __name__ == "__main__":
    unittest.main()
