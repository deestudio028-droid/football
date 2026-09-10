import _pathfix  # noqa: F401
import tempfile
import unittest
from pathlib import Path

from ingestion.checkpoint import CheckpointStore


class TestCheckpointStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_not_started_by_default(self):
        cp = CheckpointStore(self.tmp_path / "state.json")
        self.assertFalse(cp.is_complete(423, 6484))
        self.assertEqual(cp.get_window(423, 6484)["status"], "not_started")

    def test_mark_page_then_complete(self):
        cp = CheckpointStore(self.tmp_path / "state.json")
        cp.mark_page_fetched(423, 6484, page=1, fixtures_in_page=250)
        cp.mark_page_fetched(423, 6484, page=2, fixtures_in_page=130)
        window = cp.get_window(423, 6484)
        self.assertEqual(window["status"], "in_progress")
        self.assertEqual(window["pages_fetched"], 2)
        self.assertEqual(window["fixture_count"], 380)

        cp.mark_window_complete(423, 6484)
        self.assertTrue(cp.is_complete(423, 6484))

    def test_state_persists_across_instances(self):
        path = self.tmp_path / "state.json"
        cp1 = CheckpointStore(path)
        cp1.mark_window_complete(423, 6484)

        cp2 = CheckpointStore(path)
        self.assertTrue(cp2.is_complete(423, 6484))

    def test_failed_window_is_not_complete(self):
        cp = CheckpointStore(self.tmp_path / "state.json")
        cp.mark_window_failed(423, 6484, "boom")
        self.assertFalse(cp.is_complete(423, 6484))
        self.assertEqual(cp.get_window(423, 6484)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
