"""D-30 Step 1 -- derived per-league database partitions.

Verifies the partitions are exact, isolated subsets of the pinned
sources. Skips cleanly when the partitions are absent, so the suite stays
green in a checkout that has not run the partitioner.

Nothing here trains, evaluates, imputes, or reads a 2025/26 outcome.
"""
import _pathfix  # noqa: F401
import hashlib
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from models.ablation import MODEL_B_COLUMNS
from models.config import (
    FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, KNOWN_UNLABELED_FIXTURE_IDS,
    SEASON_NAME_TO_IDS,
)

REPO = Path(__file__).resolve().parents[1]
LEAGUE_DIR = REPO / "data/processed/leagues"

LEAGUES = {
    423: ("Premier League", "premier_league.db", 2280, 1900, 380),
    477: ("Bundesliga", "bundesliga.db", 1836, 1530, 306),
    419: ("La Liga", "la_liga.db", 2280, 1900, 380),
    200: ("Ligue 1", "ligue_1.db", 2058, 1752, 306),
    499: ("Serie A", "serie_a.db", 2281, 1901, 380),
}

TRAIN_IDS, TEST_IDS = set(), set()
for _s in FINAL_TRAIN_SEASONS:
    TRAIN_IDS.update(SEASON_NAME_TO_IDS[_s])
for _s in FINAL_TEST_SEASONS:
    TEST_IDS.update(SEASON_NAME_TO_IDS[_s])


def _present():
    return LEAGUE_DIR.exists() and all(
        (LEAGUE_DIR / f).exists() for _, f, *_ in LEAGUES.values())


@unittest.skipUnless(_present(), "league partitions not built")
class TestLeaguePartitions(unittest.TestCase):
    """SQLite may be unable to query the mounted repository directly, so
    every database is copied bit-for-bit to a temp dir and md5 equality is
    asserted -- the assertions therefore apply to the installed files."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        for src in list(LEAGUE_DIR.glob("*.db")) + [
                REPO / "data/processed/features.db", REPO / "data/processed/matches.db"]:
            dst = cls.tmp / src.name
            shutil.copyfile(src, dst)
            assert hashlib.md5(src.read_bytes()).hexdigest() == \
                hashlib.md5(dst.read_bytes()).hexdigest(), src.name

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def con(self, name):
        return sqlite3.connect(f"file:{self.tmp / name}?mode=ro", uri=True)

    def test_cross_league_isolation(self):
        for cid, (name, fname, *_ ) in LEAGUES.items():
            c = self.con(fname)
            try:
                for table in ("feature_rows", "fixtures"):
                    ids = {r[0] for r in c.execute(
                        f"SELECT DISTINCT competition_id FROM {table}")}
                    self.assertEqual(ids, {cid}, f"{name}/{table}")
            finally:
                c.close()

    def test_row_counts_match_the_audited_inventory(self):
        for cid, (name, fname, total, train, test) in LEAGUES.items():
            c = self.con(fname)
            try:
                self.assertEqual(c.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0],
                                 total, name)
                q = ",".join(str(i) for i in sorted(TRAIN_IDS))
                t = ",".join(str(i) for i in sorted(TEST_IDS))
                self.assertEqual(c.execute(
                    f"SELECT COUNT(*) FROM fixtures WHERE season_id IN ({q})").fetchone()[0],
                    train, f"{name} training")
                self.assertEqual(c.execute(
                    f"SELECT COUNT(*) FROM fixtures WHERE season_id IN ({t})").fetchone()[0],
                    test, f"{name} 2025/26")
            finally:
                c.close()

    def test_partitions_sum_to_the_sources(self):
        src = self.con("matches.db")
        srcf = self.con("features.db")
        try:
            total_fx = src.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
            total_ft = srcf.execute("SELECT COUNT(*) FROM feature_rows").fetchone()[0]
        finally:
            src.close(); srcf.close()
        fx = ft = 0
        for _, (name, fname, *_ ) in LEAGUES.items():
            c = self.con(fname)
            try:
                fx += c.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
                ft += c.execute("SELECT COUNT(*) FROM feature_rows").fetchone()[0]
            finally:
                c.close()
        self.assertEqual(fx, total_fx, "fixtures lost or duplicated across partitions")
        self.assertEqual(ft, total_ft, "feature rows lost or duplicated across partitions")

    def test_all_80_contract_columns_present_with_source_ordering(self):
        src = self.con("features.db")
        try:
            src_cols = [r[1] for r in src.execute("PRAGMA table_info(feature_rows)")]
        finally:
            src.close()
        for _, (name, fname, *_ ) in LEAGUES.items():
            c = self.con(fname)
            try:
                cols = [r[1] for r in c.execute("PRAGMA table_info(feature_rows)")]
            finally:
                c.close()
            self.assertEqual(cols, src_cols, f"{name}: schema/order drifted")
            for col in MODEL_B_COLUMNS:
                self.assertIn(col, cols, f"{name}: missing {col}")

    def test_feature_version_preserved(self):
        for _, (name, fname, *_ ) in LEAGUES.items():
            c = self.con(fname)
            try:
                v = {r[0] for r in c.execute(
                    "SELECT DISTINCT feature_version FROM feature_rows")}
            finally:
                c.close()
            self.assertEqual(v, {"v1.0"}, name)

    def test_fixture_id_joinability(self):
        for _, (name, fname, *_ ) in LEAGUES.items():
            c = self.con(fname)
            try:
                n = c.execute("SELECT COUNT(*) FROM feature_rows").fetchone()[0]
                j = c.execute("SELECT COUNT(*) FROM feature_rows f "
                              "JOIN fixtures x ON f.fixture_id = x.fixture_id").fetchone()[0]
            finally:
                c.close()
            self.assertEqual(j, n, f"{name}: feature rows that do not join")

    def test_values_are_untransformed(self):
        """Byte-for-byte row equality against the source for a sample."""
        src = self.con("features.db")
        try:
            for cid, (name, fname, *_ ) in LEAGUES.items():
                c = self.con(fname)
                try:
                    ids = [r[0] for r in c.execute(
                        "SELECT fixture_id FROM feature_rows ORDER BY fixture_id LIMIT 25")]
                    marks = ",".join("?" for _ in ids)
                    got = {r[0]: r for r in c.execute(
                        f"SELECT * FROM feature_rows WHERE fixture_id IN ({marks})", ids)}
                finally:
                    c.close()
                want = {r[0]: r for r in src.execute(
                    f"SELECT * FROM feature_rows WHERE fixture_id IN ({marks})", ids)}
                self.assertEqual(got, want, f"{name}: values differ from source")
        finally:
            src.close()

    def test_known_unlabelled_fixture_stays_unlabelled_in_ligue_1(self):
        c = self.con("ligue_1.db")
        try:
            row = c.execute(
                "SELECT label_result FROM feature_rows WHERE fixture_id = ?",
                (sorted(KNOWN_UNLABELED_FIXTURE_IDS)[0],)).fetchone()
            labelled = c.execute(
                "SELECT COUNT(*) FROM feature_rows WHERE label_result IS NOT NULL"
            ).fetchone()[0]
        finally:
            c.close()
        self.assertIsNotNone(row, "the ABANDONED fixture should still be present")
        self.assertIsNone(row[0], "it must remain unlabelled, not coerced")
        self.assertEqual(labelled, 2057)

    def test_no_v2_in_any_partition_name(self):
        for p in LEAGUE_DIR.glob("*.db"):
            self.assertNotIn("v2", p.name.lower())


class TestSourcesUnchanged(unittest.TestCase):
    def test_source_databases_byte_identical(self):
        for rel, exp in (("data/processed/features.db", "e7ebe7fc07040a5927683c35b6371e63"),
                         ("data/processed/matches.db", "fdeed042096fa1c851aaee6c84995247")):
            self.assertEqual(hashlib.md5((REPO / rel).read_bytes()).hexdigest(), exp, rel)

    def test_locked_inputs_unchanged(self):
        pins = json.loads(
            (REPO / "data/audit/phase4c_prerun_manifest.json").read_text()
        )["locked_input_checksums"]
        for rel, v in pins.items():
            self.assertEqual(hashlib.md5((REPO / rel).read_bytes()).hexdigest(),
                             v["expected"], f"locked artifact modified: {rel}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
