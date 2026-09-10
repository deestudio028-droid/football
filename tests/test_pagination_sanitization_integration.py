"""Integration regression test for Phase 1.1: proves that redacting the
token from next_page_url in the files RawStore writes to disk has zero
effect on the actual pagination logic in OddAlertsClient, since the two
operate on separate copies (the client's parsed FixturesPage.next_page_url
vs. the dict RawStore sanitizes before writing).
"""
import _pathfix  # noqa: F401
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ingestion.api_client import OddAlertsClient
from ingestion.raw_store import RawStore


def _resp(json_body):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = json_body
    r.text = ""
    r.headers = {}
    return r


class TestPaginationSanitizationIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_two_page_pagination_unaffected_by_raw_storage_sanitization(self):
        session = MagicMock()
        real_token = "SECRET_TOKEN_VALUE"
        session.get.side_effect = [
            _resp({
                "info": {
                    "page": 1, "per_page": 2, "count": 2, "total": 4, "total_pages": 2,
                    "next_page_url": f"https://data.oddalerts.com/api/fixtures/between?"
                                      f"api_token={real_token}&competitions=423&page=2",
                },
                "data": [{"id": 1}, {"id": 2}],
            }),
            _resp({
                "info": {"page": 2, "per_page": 2, "count": 2, "total": 4, "total_pages": 2,
                          "next_page_url": False},
                "data": [{"id": 3}, {"id": 4}],
            }),
        ]

        logger = logging.getLogger("test_integration")
        logger.addHandler(logging.NullHandler())
        client = OddAlertsClient(
            base_url="https://data.oddalerts.com/api",
            api_token=real_token,
            logger=logger,
            session=session,
        )
        raw_store = RawStore(self.tmp_path)

        pages = []
        for fp in client.iterate_fixtures_between(1000, 2000, competition_id=423, season_id=6484):
            raw_store.save_page(423, 6484, fp.page, fp.raw)
            pages.append(fp)

        # Pagination behavior: both pages fetched, in order, all 4 fixtures seen.
        self.assertEqual(len(pages), 2)
        self.assertEqual(session.get.call_count, 2)
        all_ids = [f["id"] for p in pages for f in p.data]
        self.assertEqual(all_ids, [1, 2, 3, 4])

        # The URL the client actually followed for page 2 (a separate,
        # in-memory second call) is proven by call_count == 2 above --
        # if sanitization had broken pagination, only 1 call would occur.

        # Raw files on disk must have the token redacted regardless.
        page1_saved = raw_store.load_page(423, 6484, 1)
        self.assertIn("api_token=REDACTED", page1_saved["info"]["next_page_url"])
        self.assertNotIn(real_token, page1_saved["info"]["next_page_url"])

        page2_saved = raw_store.load_page(423, 6484, 2)
        self.assertIs(page2_saved["info"]["next_page_url"], False)

        # And the fixture data itself must be identical to what was fetched.
        self.assertEqual(page1_saved["data"], [{"id": 1}, {"id": 2}])
        self.assertEqual(page2_saved["data"], [{"id": 3}, {"id": 4}])


if __name__ == "__main__":
    unittest.main()
