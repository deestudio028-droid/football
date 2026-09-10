import _pathfix  # noqa: F401
import tempfile
import unittest
from pathlib import Path

from ingestion.raw_store import RawStore


class TestRawStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_load_roundtrip(self):
        store = RawStore(self.tmp_path)
        body = {"info": {"page": 1}, "data": [{"id": 1}]}
        store.save_page(423, 6484, 1, body)
        self.assertTrue(store.exists(423, 6484, 1))
        loaded = store.load_page(423, 6484, 1)
        self.assertEqual(loaded, body)

    def test_iter_saved_pages_sorted(self):
        store = RawStore(self.tmp_path)
        store.save_page(423, 6484, 2, {"data": []})
        store.save_page(423, 6484, 1, {"data": []})
        paths = list(store.iter_saved_pages(423, 6484))
        self.assertEqual(
            [p.name for p in paths],
            ["fixtures_between_page_0001.json", "fixtures_between_page_0002.json"],
        )

    def test_raw_file_is_untouched_original_shape(self):
        store = RawStore(self.tmp_path)
        body = {"info": {"page": 1, "weird_extra_field": "kept as-is"},
                 "data": [{"id": 1, "nested": {"a": 1}}]}
        store.save_page(423, 6484, 1, body)
        loaded = store.load_page(423, 6484, 1)
        self.assertEqual(loaded, body)

    def test_no_saved_pages_returns_empty(self):
        store = RawStore(self.tmp_path)
        self.assertEqual(list(store.iter_saved_pages(999, 999)), [])

    # -- token sanitization regression tests (Phase 1.1) --------------

    def test_next_page_url_with_token_is_sanitized_before_disk_write(self):
        store = RawStore(self.tmp_path)
        body = {
            "info": {
                "page": 1, "per_page": 250, "count": 250, "total": 380, "total_pages": 2,
                "next_page_url": "https://data.oddalerts.com/api/fixtures/between?"
                                  "api_token=SECRET_TOKEN_VALUE&competitions=423&page=2",
            },
            "data": [{"id": 1}],
        }
        store.save_page(423, 6484, 1, body)
        loaded = store.load_page(423, 6484, 1)
        self.assertIn("api_token=REDACTED", loaded["info"]["next_page_url"])
        self.assertNotIn("SECRET_TOKEN_VALUE", loaded["info"]["next_page_url"])

    def test_actual_token_never_appears_anywhere_in_written_file(self):
        store = RawStore(self.tmp_path)
        body = {
            "info": {
                "page": 1,
                "next_page_url": "https://data.oddalerts.com/api/fixtures/between?"
                                  "api_token=SECRET_TOKEN_VALUE&page=2",
            },
            "data": [{"id": 1, "home_name": "A"}],
        }
        target = store.save_page(423, 6484, 1, body)
        raw_text = target.read_text(encoding="utf-8")
        self.assertNotIn("SECRET_TOKEN_VALUE", raw_text)

    def test_next_page_url_without_token_is_unchanged(self):
        store = RawStore(self.tmp_path)
        body = {
            "info": {"page": 1, "next_page_url": "https://data.oddalerts.com/api/fixtures/between?page=2"},
            "data": [{"id": 1}],
        }
        store.save_page(423, 6484, 1, body)
        loaded = store.load_page(423, 6484, 1)
        self.assertEqual(
            loaded["info"]["next_page_url"],
            "https://data.oddalerts.com/api/fixtures/between?page=2",
        )

    def test_next_page_url_false_is_unchanged(self):
        # Confirmed real API shape: no-next-page is the JSON boolean false,
        # not a URL string. Must pass through untouched (not stringified).
        store = RawStore(self.tmp_path)
        body = {"info": {"page": 2, "next_page_url": False}, "data": [{"id": 1}]}
        store.save_page(423, 6484, 2, body)
        loaded = store.load_page(423, 6484, 2)
        self.assertIs(loaded["info"]["next_page_url"], False)

    def test_fixture_data_untouched_by_sanitization(self):
        store = RawStore(self.tmp_path)
        body = {
            "info": {"page": 1, "next_page_url": "https://x/y?api_token=SECRET&page=2"},
            "data": [{
                "id": 152908783, "home_name": "Manchester United", "away_name": "Fulham",
                "home_goals": 1, "away_goals": 0, "unix": 1723834800,
                "stats": {"home_xg": 1.7564, "home_shots": 24},
            }],
        }
        store.save_page(423, 6484, 1, body)
        loaded = store.load_page(423, 6484, 1)
        self.assertEqual(loaded["data"], body["data"])

    def test_caller_supplied_dict_is_not_mutated_by_save(self):
        # The in-memory response (and its real next_page_url) must remain
        # usable for actual pagination after being handed to save_page --
        # sanitization must operate on a copy, not the caller's object.
        store = RawStore(self.tmp_path)
        original_url = "https://x/y?api_token=SECRET_TOKEN_VALUE&page=2"
        body = {"info": {"page": 1, "next_page_url": original_url}, "data": []}
        store.save_page(423, 6484, 1, body)
        self.assertEqual(body["info"]["next_page_url"], original_url)

    def test_other_info_fields_preserved_alongside_sanitized_url(self):
        store = RawStore(self.tmp_path)
        body = {
            "info": {
                "page": 1, "per_page": 250, "count": 250, "total": 380, "total_pages": 2,
                "next_page_url": "https://x/y?api_token=SECRET&page=2",
                "warning": "Date range exceeds 365 days. Results have been limited to 365 days from the start date.",
            },
            "data": [],
        }
        store.save_page(423, 6484, 1, body)
        loaded = store.load_page(423, 6484, 1)
        self.assertEqual(loaded["info"]["total"], 380)
        self.assertEqual(loaded["info"]["total_pages"], 2)
        self.assertEqual(
            loaded["info"]["warning"],
            "Date range exceeds 365 days. Results have been limited to 365 days from the start date.",
        )


if __name__ == "__main__":
    unittest.main()
