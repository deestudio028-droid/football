"""D-18 -- v1.1 successor feature architecture (D-17 Option B).

Proves two things that must both hold:

  1. V1 IS UNCHANGED. The frozen registry still holds exactly its five
     entries, the public V1 builder still selects them, MODEL_B_COLUMNS
     is still 80 columns in its original order, and FEATURE_VERSION /
     MODEL_VERSION / REQUIRED_FEATURE_VERSION are all still "v1.0".

  2. THE SUCCESSOR IS ISOLATED AND CORRECT. The v1.1 registry is derived
     from V1's (so the shared entries cannot drift), only the successor
     entry point selects it, and the six new columns are produced by the
     SAME production machinery D-5 used -- gated_mean / take_window /
     windowed_mean / WINDOW_MIN_COVERAGE -- with identical NULL,
     coverage, season-boundary and own-side semantics.

On the D-5 equivalence question: this module deliberately does NOT
import run_d5_new_feature_construction_audit.py. That script is a
root-level audit harness which reads matches.db, features.db and the
Phase 4C manifest at module scope; importing it into the production test
suite would couple these tests to pinned data files and to the models
package. Its algorithm is NOT copied either -- copying it would prove
only that two copies agree.

Instead the equivalence is established structurally, which is stronger:
D-5 called production's `gated_mean(history, "season", season_id,
extractor)` directly, and TestSameProductionMachineryAsD5 proves by AST
that the successor emission path calls that same production helper, via
the same single emitter (`_season_only_family_features`), with no second
rolling/statistics implementation anywhere in the features package. The
behavioural tests below then pin the observable semantics D-5 audited.
"""
import _pathfix  # noqa: F401
import ast
import inspect
import sqlite3
import tempfile
import unittest
from pathlib import Path

from _feature_test_helpers import build_matches_db

from features import feature_builder as fb
from features.config import (
    FEATURE_VERSION,
    SUCCESSOR_FEATURE_VERSION,
    WINDOW_MIN_COVERAGE,
)
from features.history import team_match_record
from features.missingness import EXTRACTORS
from models.ablation import MODEL_B_COLUMNS
from models.successor_contract import (
    MODEL_B_PLUS_POSSESSION_COLUMNS,
    POSSESSION_FAMILY_COLUMNS,
    SUCCESSOR_N_FEATURES,
    V1_N_FEATURES,
)
from models.config import MODEL_VERSION, REQUIRED_FEATURE_VERSION

V1_REGISTRY_KEYS = ("attacks", "dang_attacks", "pressure", "fouls", "yellow_cards")
NEW_REGISTRY_KEYS = ("possession", "corners", "red_cards")
FEATURES_PKG = Path(fb.__file__).parent


def raw_fixture(
    fixture_id, unix, home_id, away_id, home_goals=1, away_goals=0,
    competition_id=1, season_id=100, status="FT",
    home_possession=None, away_possession=None,
    home_corners=None, away_corners=None,
    home_red_cards=None, away_red_cards=None,
):
    """Local factory. tests/_feature_test_helpers.make_raw_fixture is an
    existing V1 test file and must not be modified (D-18 requirement T),
    so the successor stats are supplied here instead. build_matches_db
    and ingestion.normalize_fixture are reused unchanged -- they already
    persist stat_{side}_{possession,corners,red_cards}.
    """
    return {
        "id": fixture_id, "unix": unix, "home_id": home_id, "away_id": away_id,
        "home_name": f"Team{home_id}", "away_name": f"Team{away_id}",
        "home_goals": home_goals, "away_goals": away_goals,
        "competition_id": competition_id, "season_id": season_id, "status": status,
        "stats": {
            "home_possession": home_possession, "away_possession": away_possession,
            "home_corners": home_corners, "away_corners": away_corners,
            "home_red_cards": home_red_cards, "away_red_cards": away_red_cards,
        },
    }


class _TmpMixin(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


# =====================================================================
# 1. V1 IS UNCHANGED
# =====================================================================
class TestV1Frozen(unittest.TestCase):
    def test_a_model_b_columns_is_exactly_80(self):
        self.assertEqual(len(MODEL_B_COLUMNS), 80)

    def test_b_model_b_columns_order_unchanged(self):
        # The successor is built by concatenation, so slicing it back is a
        # genuine order check, not a tautology about set membership.
        self.assertEqual(tuple(MODEL_B_PLUS_POSSESSION_COLUMNS[:80]), tuple(MODEL_B_COLUMNS))
        self.assertEqual(len(set(MODEL_B_COLUMNS)), 80)

    def test_c_v1_registry_contains_exactly_its_previous_entries(self):
        self.assertEqual(tuple(fb._SEASON_ONLY_EXTRACTOR.keys()), V1_REGISTRY_KEYS)
        self.assertEqual(len(fb._SEASON_ONLY_EXTRACTOR), 5)
        for family in NEW_REGISTRY_KEYS:
            self.assertNotIn(family, fb._SEASON_ONLY_EXTRACTOR)

    def test_g_feature_version_still_v1_0(self):
        self.assertEqual(FEATURE_VERSION, "v1.0")

    def test_v1_model_constants_untouched(self):
        self.assertEqual(MODEL_VERSION, "v1.0")
        self.assertEqual(REQUIRED_FEATURE_VERSION, "v1.0")

    def test_o_minimum_coverage_still_two(self):
        self.assertEqual(WINDOW_MIN_COVERAGE["season"], 2)

    def test_r_v1_registry_cannot_be_mutated(self):
        # MappingProxyType turns "silently corrupted V1" into a raised
        # exception -- the D-17 hazard, closed mechanically rather than
        # by convention.
        with self.assertRaises(TypeError):
            fb._SEASON_ONLY_EXTRACTOR["possession"] = "possession_for"
        with self.assertRaises(TypeError):
            fb._SEASON_ONLY_EXTRACTOR_V1_1["anything"] = "x"

    def test_k_existing_extractors_unchanged(self):
        # Additive only: every V1 extractor key still present, and the
        # V1 registry's targets all still resolve.
        for key in ("goals_for", "goals_against", "xg_for", "xg_against",
                    "shots_for", "shots_against", "shots_on_for", "shots_on_against",
                    "attacks_for", "dang_attacks_for", "pressure_for",
                    "fouls_for", "yellow_cards_for"):
            self.assertIn(key, EXTRACTORS)
        for extractor_key in fb._SEASON_ONLY_EXTRACTOR.values():
            self.assertIn(extractor_key, EXTRACTORS)


class TestV1BuilderStillSelectsV1(_TmpMixin):
    def test_i_v1_builder_emits_v1_columns_and_v1_0_stamp(self):
        fixtures = [
            raw_fixture(1, 1000, 1, 2, home_possession=60, home_corners=7, home_red_cards=1),
            raw_fixture(2, 2000, 1, 3, home_possession=55, home_corners=5, home_red_cards=0),
            raw_fixture(3, 3000, 1, 4, home_possession=50, home_corners=3, home_red_cards=0),
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        rows = fb.build_feature_dataset(db_path)
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["feature_version"], "v1.0")
            for family in NEW_REGISTRY_KEYS:
                for prefix in ("home", "away"):
                    self.assertNotIn(f"{prefix}_{family}_per_match_season", row)
            # V1's own season-only families are still emitted.
            self.assertIn("home_fouls_per_match_season", row)

    def test_v1_default_call_is_byte_identical_to_explicit_v1_call(self):
        fixtures = [
            raw_fixture(1, 1000, 1, 2, home_possession=60),
            raw_fixture(2, 2000, 1, 3, home_possession=55),
            raw_fixture(3, 3000, 1, 4, home_possession=50),
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        default_rows = fb.build_feature_dataset(db_path)
        explicit_rows = fb._walk_and_build(
            db_path, None,
            season_only_extractor=fb._SEASON_ONLY_EXTRACTOR,
            feature_version=FEATURE_VERSION,
        )
        self.assertEqual(len(default_rows), len(explicit_rows))
        for a, b in zip(default_rows, explicit_rows):
            a = {k: v for k, v in a.items() if k != "generated_at"}
            b = {k: v for k, v in b.items() if k != "generated_at"}
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assertEqual(_null_safe(a[key]), _null_safe(b[key]), msg=key)


def _null_safe(v):
    """Exact comparison that is not defeated by NaN != NaN and does not
    collapse None into 0.0."""
    if v is None:
        return ("NULL",)
    if isinstance(v, float) and v != v:
        return ("NAN",)
    return (type(v).__name__, v)


# =====================================================================
# 2. SUCCESSOR CONTRACT
# =====================================================================
class TestSuccessorContract(unittest.TestCase):
    def test_e_successor_contract_is_86_columns(self):
        self.assertEqual(len(MODEL_B_PLUS_POSSESSION_COLUMNS), 86)
        self.assertEqual(len(set(MODEL_B_PLUS_POSSESSION_COLUMNS)), 86)

    def test_f_first_80_equal_model_b_exactly(self):
        for i, (a, b) in enumerate(zip(MODEL_B_PLUS_POSSESSION_COLUMNS, MODEL_B_COLUMNS)):
            self.assertEqual(a, b, msg=f"contract position {i}")
        self.assertEqual(MODEL_B_PLUS_POSSESSION_COLUMNS[80:], POSSESSION_FAMILY_COLUMNS)

    def test_possession_family_is_the_six_expected_names(self):
        self.assertEqual(POSSESSION_FAMILY_COLUMNS, (
            "home_possession_per_match_season", "away_possession_per_match_season",
            "home_corners_per_match_season", "away_corners_per_match_season",
            "home_red_cards_per_match_season", "away_red_cards_per_match_season",
        ))

    def test_new_columns_absent_from_v1_contract(self):
        self.assertEqual(set(POSSESSION_FAMILY_COLUMNS) & set(MODEL_B_COLUMNS), set())

    def test_h_successor_feature_version(self):
        self.assertEqual(SUCCESSOR_FEATURE_VERSION, "v1.1")
        self.assertNotEqual(SUCCESSOR_FEATURE_VERSION, FEATURE_VERSION)

    def test_d_successor_registry_is_v1_then_new(self):
        keys = tuple(fb._SEASON_ONLY_EXTRACTOR_V1_1.keys())
        self.assertEqual(keys[:5], V1_REGISTRY_KEYS)
        self.assertEqual(keys[5:], NEW_REGISTRY_KEYS)
        self.assertEqual(len(keys), 8)
        # Shared entries must be identical values, not merely same keys.
        for k in V1_REGISTRY_KEYS:
            self.assertEqual(fb._SEASON_ONLY_EXTRACTOR_V1_1[k], fb._SEASON_ONLY_EXTRACTOR[k])

    def test_d_successor_registry_is_derived_not_restated(self):
        # Proves drift is structurally impossible: the source must unpack
        # the V1 registry rather than repeat its five entries.
        src = inspect.getsource(fb)
        tree = ast.parse(src)
        assign = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "_SEASON_ONLY_EXTRACTOR_V1_1" for t in n.targets)
        )
        unpacked = [
            k for d in ast.walk(assign) if isinstance(d, ast.Dict)
            for k in d.keys if k is None
        ]
        self.assertTrue(unpacked, "successor registry must unpack the V1 registry")
        names = {n.id for n in ast.walk(assign) if isinstance(n, ast.Name)}
        self.assertIn("_SEASON_ONLY_EXTRACTOR", names)


# =====================================================================
# 3. SUCCESSOR ISOLATION
# =====================================================================
class TestSuccessorIsolation(unittest.TestCase):
    def _defaults(self, fn):
        sig = inspect.signature(fn)
        return {n: p.default for n, p in sig.parameters.items()
                if p.default is not inspect.Parameter.empty}

    def test_i_public_v1_api_signature_unchanged(self):
        params = list(inspect.signature(fb.build_feature_dataset).parameters)
        self.assertEqual(params, ["db_path", "fixture_ids"])

    def test_i_v1_defaults_resolve_to_frozen_registry_and_v1_0(self):
        d = self._defaults(fb.build_feature_row)
        self.assertIs(d["_season_only_extractor"], fb._SEASON_ONLY_EXTRACTOR)
        self.assertEqual(d["_feature_version"], "v1.0")
        self.assertIs(
            self._defaults(fb._season_only_family_features)["extractor_registry"],
            fb._SEASON_ONLY_EXTRACTOR,
        )

    def test_j_only_successor_entry_point_references_v1_1_registry(self):
        tree = ast.parse(inspect.getsource(fb))
        users = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Name) and n.id == "_SEASON_ONLY_EXTRACTOR_V1_1":
                        users.add(fn.name)
        self.assertEqual(users, {"build_successor_feature_dataset"})

    def test_j_only_successor_entry_point_stamps_v1_1(self):
        tree = ast.parse(inspect.getsource(fb))
        users = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Name) and n.id == "SUCCESSOR_FEATURE_VERSION":
                        users.add(fn.name)
        self.assertEqual(users, {"build_successor_feature_dataset"})

    def test_q_successor_cannot_default_to_the_pinned_v1_database(self):
        for fn in (fb.build_successor_feature_dataset, fb._walk_and_build, fb.build_feature_row):
            for name, default in self._defaults(fn).items():
                self.assertNotIsInstance(default, Path, msg=f"{fn.__name__}.{name}")
                if isinstance(default, str):
                    self.assertNotIn("features.db", default)
        # No output/destination parameter exists at all, so there is no
        # path -- default or otherwise -- for this function to write to.
        params = list(inspect.signature(fb.build_successor_feature_dataset).parameters)
        self.assertEqual(params, ["db_path", "fixture_ids"])

    def test_q_no_write_capability_in_the_features_builder(self):
        """The builder module cannot persist anything, so no caller can
        make it write to the pinned V1 database. Checked structurally via
        AST -- a source-text grep would flag this module's own docstring
        mentioning FeatureDB, which is prose, not capability."""
        tree = ast.parse(inspect.getsource(fb))
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                imported.add((n.module or "").split(".")[0])
                imported.update(a.name for a in n.names)
        for banned in ("sqlite3", "storage", "FeatureDB"):
            self.assertNotIn(banned, imported)
        called = {
            n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        for banned in ("commit", "execute", "executemany", "executescript", "upsert_fixtures"):
            self.assertNotIn(banned, called)

    def test_s_no_v2_named_module_or_artifact_introduced(self):
        for p in FEATURES_PKG.rglob("*.py"):
            self.assertNotIn("v2", p.name.lower())
        for p in (FEATURES_PKG.parent / "models").rglob("*.py"):
            self.assertNotIn("v2", p.name.lower())
        self.assertNotIn("v2", Path(__file__).name.lower())


# =====================================================================
# 4. SAME PRODUCTION MACHINERY AS D-5 (no second implementation)
# =====================================================================
class TestSameProductionMachineryAsD5(unittest.TestCase):
    def test_l_single_emitter_used_for_all_season_only_families(self):
        """Exactly one function in the builder emits `*_per_match_season`
        column names, so both versions necessarily go through the same
        construction. Matches on the literal f-string fragment, not on
        "contains an f-string" -- `_rolling_family_features` builds
        `_per_match_{window}` and `_venue_features` builds
        `_{venue_label}_season`, and neither must be caught here."""
        tree = ast.parse(inspect.getsource(fb))
        emitters = [
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and any(
                isinstance(part, ast.Constant) and isinstance(part.value, str)
                and "_per_match_season" in part.value
                for js in ast.walk(n) if isinstance(js, ast.JoinedStr)
                for part in js.values
            )
        ]
        self.assertEqual(emitters, ["_season_only_family_features"])
        # Control: the predicate must still see the other two emitters
        # under their own templates, or it proves nothing.
        def _fragments(fn_name, needle):
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == fn_name)
            return any(
                isinstance(p, ast.Constant) and isinstance(p.value, str) and needle in p.value
                for js in ast.walk(fn) if isinstance(js, ast.JoinedStr)
                for p in js.values
            )
        self.assertTrue(_fragments("_rolling_family_features", "_per_match_"))
        self.assertTrue(_fragments("_venue_features", "_season"))

    def test_l_emitter_uses_production_gated_mean(self):
        tree = ast.parse(inspect.getsource(fb._season_only_family_features))
        called = {
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        self.assertIn("gated_mean", called)
        self.assertIs(fb.gated_mean.__module__, fb.gated_mean.__module__)
        self.assertEqual(fb.gated_mean.__module__, "features.rolling")

    def test_l_no_second_rolling_implementation_in_features_package(self):
        for p in FEATURES_PKG.rglob("*.py"):
            if p.name == "rolling.py":
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            for banned in ("gated_mean", "windowed_mean", "take_window"):
                self.assertNotIn(banned, defined, msg=f"{p.name} redefines {banned}")

    def test_history_record_supplies_own_side_stats_only(self):
        fixture = {
            "fixture_id": 1, "unix": 1, "competition_id": 1, "season_id": 100,
            "home_goals": 1, "away_goals": 0,
            "stat_home_possession": 60, "stat_away_possession": 40,
            "stat_home_corners": 7, "stat_away_corners": 2,
            "stat_home_red_cards": 1, "stat_away_red_cards": 0,
        }
        home = team_match_record(fixture, "home")
        away = team_match_record(fixture, "away")
        self.assertEqual((home["possession_for"], home["corners_for"], home["red_cards_for"]), (60, 7, 1))
        self.assertEqual((away["possession_for"], away["corners_for"], away["red_cards_for"]), (40, 2, 0))

    def test_m_current_fixture_excluded_before_recording(self):
        tree = ast.parse(inspect.getsource(fb._walk_and_build))
        order = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name) and n.func.id == "build_feature_row":
                    order.append(("build", n.lineno))
                if isinstance(n.func, ast.Attribute) and n.func.attr == "record":
                    order.append(("record", n.lineno))
        self.assertEqual([k for k, _ in sorted(order, key=lambda t: t[1])], ["build", "record"])


# =====================================================================
# 5. SUCCESSOR BEHAVIOUR (the semantics D-5 audited)
# =====================================================================
class TestSuccessorConstruction(_TmpMixin):
    def _rows(self, fixtures):
        db_path = build_matches_db(self.tmp_path, fixtures)
        return {r["fixture_id"]: r for r in fb.build_successor_feature_dataset(db_path)}

    def test_successor_emits_all_six_columns_and_companions(self):
        rows = self._rows([raw_fixture(1, 1000, 1, 2, home_possession=60)])
        for col in POSSESSION_FAMILY_COLUMNS:
            self.assertIn(col, rows[1])
            self.assertIn(f"{col}_n", rows[1])
            self.assertIn(f"{col}_coverage_n", rows[1])

    def test_i_successor_rows_stamp_v1_1(self):
        rows = self._rows([raw_fixture(1, 1000, 1, 2), raw_fixture(2, 2000, 3, 4)])
        self.assertEqual({r["feature_version"] for r in rows.values()}, {"v1.1"})

    def test_o_and_p_below_min_coverage_is_null_not_zero(self):
        # Team 1 plays m1, m2, m3. At m1 it has 0 prior; at m2 it has 1
        # (< WINDOW_MIN_COVERAGE["season"] == 2) -> NULL; at m3 it has 2.
        rows = self._rows([
            raw_fixture(1, 1000, 1, 2, home_possession=60),
            raw_fixture(2, 2000, 1, 3, home_possession=50),
            raw_fixture(3, 3000, 1, 4, home_possession=40),
        ])
        self.assertIsNone(rows[1]["home_possession_per_match_season"])
        self.assertEqual(rows[1]["home_possession_per_match_season_coverage_n"], 0)
        self.assertIsNone(rows[2]["home_possession_per_match_season"])
        self.assertEqual(rows[2]["home_possession_per_match_season_coverage_n"], 1)
        self.assertEqual(rows[3]["home_possession_per_match_season"], 55.0)  # (60+50)/2
        self.assertEqual(rows[3]["home_possession_per_match_season_coverage_n"], 2)

    def test_p_missing_raw_stat_is_skipped_not_zero_filled(self):
        # m2's possession is missing entirely; the mean over m1..m3 must
        # be (60+40)/2 == 50.0, not (60+0+40)/3.
        rows = self._rows([
            raw_fixture(1, 1000, 1, 2, home_possession=60),
            raw_fixture(2, 2000, 1, 3, home_possession=None),
            raw_fixture(3, 3000, 1, 4, home_possession=40),
            raw_fixture(4, 4000, 1, 5, home_possession=99),
        ])
        self.assertEqual(rows[4]["home_possession_per_match_season"], 50.0)
        self.assertEqual(rows[4]["home_possession_per_match_season_n"], 3)
        self.assertEqual(rows[4]["home_possession_per_match_season_coverage_n"], 2)

    def test_m_current_fixture_never_enters_its_own_feature(self):
        # If m3's own value leaked in, the mean would be (60+50+40)/3=50.0.
        rows = self._rows([
            raw_fixture(1, 1000, 1, 2, home_possession=60),
            raw_fixture(2, 2000, 1, 3, home_possession=50),
            raw_fixture(3, 3000, 1, 4, home_possession=40),
        ])
        self.assertEqual(rows[3]["home_possession_per_match_season"], 55.0)

    def test_n_season_boundary_resets(self):
        rows = self._rows([
            raw_fixture(1, 1000, 1, 2, season_id=100, home_possession=60),
            raw_fixture(2, 2000, 1, 3, season_id=100, home_possession=50),
            raw_fixture(3, 3000, 1, 4, season_id=101, home_possession=10),
            raw_fixture(4, 4000, 1, 5, season_id=101, home_possession=20),
            raw_fixture(5, 5000, 1, 6, season_id=101, home_possession=30),
        ])
        # First fixture of the new season: two prior matches exist, but
        # both are last season -> excluded -> NULL.
        self.assertIsNone(rows[3]["home_possession_per_match_season"])
        self.assertEqual(rows[3]["home_possession_per_match_season_coverage_n"], 0)
        self.assertEqual(rows[5]["home_possession_per_match_season"], 15.0)  # (10+20)/2

    def test_own_side_semantics_home_vs_away(self):
        # Team 1 is home in m1 (poss 60) and away in m2 (poss 30).
        # Team 2 is away in m1 (poss 40) and home in m3 (poss 70).
        rows = self._rows([
            raw_fixture(1, 1000, 1, 2, home_possession=60, away_possession=40),
            raw_fixture(2, 2000, 5, 1, home_possession=70, away_possession=30),
            raw_fixture(3, 3000, 2, 6, home_possession=70, away_possession=10),
            raw_fixture(4, 4000, 1, 2, home_possession=1, away_possession=2),
        ])
        self.assertEqual(rows[4]["home_possession_per_match_season"], 45.0)  # team1: (60+30)/2
        self.assertEqual(rows[4]["away_possession_per_match_season"], 55.0)  # team2: (40+70)/2

    def test_corners_and_red_cards_follow_the_same_rules(self):
        rows = self._rows([
            raw_fixture(1, 1000, 1, 2, home_corners=8, home_red_cards=1),
            raw_fixture(2, 2000, 1, 3, home_corners=4, home_red_cards=0),
            raw_fixture(3, 3000, 1, 4, home_corners=0, home_red_cards=0),
        ])
        self.assertEqual(rows[3]["home_corners_per_match_season"], 6.0)
        self.assertEqual(rows[3]["home_red_cards_per_match_season"], 0.5)

    def test_unplayed_fixtures_never_enter_history(self):
        rows = self._rows([
            raw_fixture(1, 1000, 1, 2, home_possession=60),
            raw_fixture(2, 2000, 1, 3, home_possession=50, status="CANC"),
            raw_fixture(3, 3000, 1, 4, home_possession=40),
            raw_fixture(4, 4000, 1, 5, home_possession=99),
        ])
        # Only m1 and m3 count -> coverage 2, mean (60+40)/2.
        self.assertEqual(rows[4]["home_possession_per_match_season_coverage_n"], 2)
        self.assertEqual(rows[4]["home_possession_per_match_season"], 50.0)

    def test_successor_shares_v1_values_for_the_v1_columns(self):
        """The additive change must disturb nothing: for identical input,
        every V1 column must be exactly equal between the two builders."""
        fixtures = [
            raw_fixture(1, 1000, 1, 2, home_goals=2, away_goals=1, home_possession=60, home_corners=8),
            raw_fixture(2, 2000, 1, 3, home_goals=0, away_goals=0, home_possession=50, home_corners=4),
            raw_fixture(3, 3000, 2, 3, home_goals=1, away_goals=3, home_possession=45, home_corners=6),
            raw_fixture(4, 4000, 1, 2, home_goals=1, away_goals=1, home_possession=55, home_corners=5),
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        v1 = {r["fixture_id"]: r for r in fb.build_feature_dataset(db_path)}
        succ = {r["fixture_id"]: r for r in fb.build_successor_feature_dataset(db_path)}
        self.assertEqual(v1.keys(), succ.keys())
        skip = {"feature_version", "generated_at"}
        for fid, v1_row in v1.items():
            for col, val in v1_row.items():
                if col in skip:
                    continue
                self.assertIn(col, succ[fid])
                self.assertEqual(_null_safe(val), _null_safe(succ[fid][col]),
                                 msg=f"fixture {fid} column {col}")
        # And the successor adds exactly the expected 18 columns.
        extra = set(succ[1]) - set(v1[1])
        self.assertEqual(len(extra), 18)
        self.assertTrue(set(POSSESSION_FAMILY_COLUMNS).issubset(extra))

    def test_l_fixture_ordering_identical_between_builders(self):
        fixtures = [
            raw_fixture(3, 3000, 1, 4), raw_fixture(1, 1000, 1, 2), raw_fixture(2, 2000, 1, 3),
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        v1_order = [r["fixture_id"] for r in fb.build_feature_dataset(db_path)]
        succ_order = [r["fixture_id"] for r in fb.build_successor_feature_dataset(db_path)]
        self.assertEqual(v1_order, succ_order)
        self.assertEqual(v1_order, [1, 2, 3])


# =====================================================================
# 6. PINNED V1 DATA UNTOUCHED
# =====================================================================
class TestPinnedArtifactsUntouched(unittest.TestCase):
    REPO = Path(__file__).resolve().parents[1]

    def test_a_b_pinned_databases_unchanged(self):
        import hashlib
        expected = {
            "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
            "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
        }
        for rel, want in expected.items():
            got = hashlib.md5((self.REPO / rel).read_bytes()).hexdigest()
            self.assertEqual(got, want, msg=rel)

    def test_six_new_columns_absent_from_pinned_features_db(self):
        con = sqlite3.connect(f"file:{self.REPO / 'data/processed/features.db'}?mode=ro", uri=True)
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(feature_rows)")}
        finally:
            con.close()
        for col in POSSESSION_FAMILY_COLUMNS:
            self.assertNotIn(col, cols)
        # The pinned V1 database still declares only v1.0 rows.
        con = sqlite3.connect(f"file:{self.REPO / 'data/processed/features.db'}?mode=ro", uri=True)
        try:
            versions = {r[0] for r in con.execute("SELECT DISTINCT feature_version FROM feature_rows")}
        finally:
            con.close()
        self.assertEqual(versions, {"v1.0"})


# =====================================================================
# 7. SUCCESSOR CONTRACT MODULE HYGIENE + FINAL-TEST BOUNDARY
# =====================================================================
class TestSuccessorContractModule(unittest.TestCase):
    def test_sizes_are_derived_not_hardcoded_by_callers(self):
        self.assertEqual(V1_N_FEATURES, 80)
        self.assertEqual(SUCCESSOR_N_FEATURES, 86)

    def test_module_imports_model_b_rather_than_restating_it(self):
        from models import successor_contract as sc
        tree = ast.parse(inspect.getsource(sc))
        imported = {
            a.name for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) for a in n.names
        }
        self.assertIn("MODEL_B_COLUMNS", imported)
        # The 80 V1 names must appear nowhere as literals in this module.
        literals = {
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        self.assertEqual(literals & set(MODEL_B_COLUMNS), set())

    def test_module_has_no_training_or_generation_logic_and_no_side_effects(self):
        from models import successor_contract as sc
        tree = ast.parse(inspect.getsource(sc))
        self.assertEqual(
            [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)], [])
        self.assertEqual(
            [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)], [])
        called = {
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        self.assertTrue(called <= {"len"}, f"unexpected module-level calls: {called}")
        for banned in ("sqlite3", "pandas", "numpy", "sklearn"):
            self.assertNotIn(banned, inspect.getsource(sc).split('"""')[2])

    def test_s_module_name_contains_no_v2(self):
        from models import successor_contract as sc
        self.assertNotIn("v2", Path(sc.__file__).stem.lower())


class TestFinalTestBoundary(_TmpMixin):
    """Row-level season exclusion is the CALLER's job (as in D-5's own
    harness, which filtered FINAL_TEST_SEASONS before calling the
    builder). What the features package must guarantee is weaker but
    checkable: it knows nothing about the final test and cannot reach
    production data on its own."""

    def test_h_features_package_has_no_final_test_knowledge(self):
        for p in FEATURES_PKG.rglob("*.py"):
            src = p.read_text(encoding="utf-8")
            self.assertNotIn("FINAL_TEST_SEASONS", src, msg=p.name)
            self.assertNotIn("2025/26", src, msg=p.name)
            self.assertNotIn("2025/2026", src, msg=p.name)

    def test_h_builders_cannot_reach_production_databases_by_default(self):
        for fn in (fb.build_feature_dataset, fb.build_successor_feature_dataset):
            sig = inspect.signature(fn)
            self.assertIs(sig.parameters["db_path"].default, inspect.Parameter.empty)
            self.assertIsNone(sig.parameters["fixture_ids"].default)

    def test_h_caller_side_season_filtering_yields_only_retained_seasons(self):
        fixtures = [
            raw_fixture(1, 1000, 1, 2, season_id=100, home_possession=60),
            raw_fixture(2, 2000, 1, 3, season_id=100, home_possession=50),
            raw_fixture(3, 3000, 1, 4, season_id=999, home_possession=40),  # "final test"
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        keep = [r["fixture_id"] for r in fb.build_successor_feature_dataset(db_path)
                if r["season_id"] != 999]
        self.assertEqual(keep, [1, 2])

    def test_g_each_builder_emits_exactly_one_feature_version(self):
        fixtures = [raw_fixture(i, i * 1000, 1, i + 1) for i in range(1, 5)]
        db_path = build_matches_db(self.tmp_path, fixtures)
        self.assertEqual(
            {r["feature_version"] for r in fb.build_feature_dataset(db_path)}, {"v1.0"})
        self.assertEqual(
            {r["feature_version"] for r in fb.build_successor_feature_dataset(db_path)}, {"v1.1"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
