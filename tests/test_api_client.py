import _pathfix  # noqa: F401
import logging
import unittest
from unittest.mock import MagicMock

import requests

from ingestion.api_client import ApiError, OddAlertsClient


def _resp(status=200, json_body=None, text="", headers=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_body if json_body is not None else {}
    r.text = text
    r.headers = headers or {}
    return r


def make_client(session, max_retries=3, backoff_base_seconds=0.01):
    logger = logging.getLogger("test_api_client")
    logger.addHandler(logging.NullHandler())
    return OddAlertsClient(
        base_url="https://data.oddalerts.com/api",
        api_token="SECRET_TOKEN_VALUE",
        logger=logger,
        timeout_seconds=5,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        session=session,
    )


class TestOddAlertsClient(unittest.TestCase):
    def test_single_page_no_next_url(self):
        session = MagicMock()
        session.get.return_value = _resp(200, {
            "info": {"page": 1, "per_page": 250, "count": 2, "next_page_url": None},
            "data": [{"id": 1}, {"id": 2}],
        })
        client = make_client(session)
        pages = list(client.iterate_fixtures_between(1000, 2000, competition_id=423, season_id=6484))
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].count, 2)
        self.assertEqual(len(pages[0].data), 2)

    def test_follows_next_page_url_until_none(self):
        session = MagicMock()
        session.get.side_effect = [
            _resp(200, {
                "info": {"page": 1, "per_page": 2, "count": 4,
                          "next_page_url": "https://data.oddalerts.com/api/fixtures/between?page=2&api_token=SECRET_TOKEN_VALUE"},
                "data": [{"id": 1}, {"id": 2}],
            }),
            _resp(200, {
                "info": {"page": 2, "per_page": 2, "count": 4, "next_page_url": None},
                "data": [{"id": 3}, {"id": 4}],
            }),
        ]
        client = make_client(session)
        pages = list(client.iterate_fixtures_between(1000, 2000))
        self.assertEqual(len(pages), 2)
        self.assertEqual(sum(len(p.data) for p in pages), 4)

    def test_placeholder_next_page_url_string_is_not_followed(self):
        session = MagicMock()
        session.get.return_value = _resp(200, {
            "info": {"page": 1, "per_page": 250, "count": 1,
                      "next_page_url": "NEXT PAGE URL WILL APPEAR HERE"},
            "data": [{"id": 1}],
        })
        client = make_client(session)
        pages = list(client.iterate_fixtures_between(1000, 2000))
        self.assertEqual(len(pages), 1)
        self.assertEqual(session.get.call_count, 1)

    def test_next_page_url_as_boolean_false_is_handled(self):
        # Confirmed real API shape (seen on a live /fixtures/between call
        # during sample-ingestion validation): next_page_url can be the
        # JSON boolean `false`, not just `null` or a URL string. This must
        # not raise (e.g. from calling .startswith on a bool).
        session = MagicMock()
        session.get.return_value = _resp(200, {
            "info": {"page": 1, "per_page": 250, "total": 90, "total_pages": 1,
                      "count": 90, "next_page_url": False},
            "data": [{"id": i} for i in range(90)],
        })
        client = make_client(session)
        pages = list(client.iterate_fixtures_between(1000, 2000, competition_id=423, season_id=6484))
        self.assertEqual(len(pages), 1)
        self.assertEqual(session.get.call_count, 1)

    def test_retries_on_5xx_then_succeeds(self):
        session = MagicMock()
        session.get.side_effect = [
            _resp(503, text="temporarily unavailable"),
            _resp(200, {"info": {"page": 1, "per_page": 250, "count": 1, "next_page_url": None},
                         "data": [{"id": 1}]}),
        ]
        client = make_client(session)
        pages = list(client.iterate_fixtures_between(1000, 2000))
        self.assertEqual(len(pages), 1)
        self.assertEqual(session.get.call_count, 2)

    def test_retries_on_connection_error_then_succeeds(self):
        session = MagicMock()
        session.get.side_effect = [
            requests.exceptions.ConnectionError("boom"),
            _resp(200, {"info": {"page": 1, "per_page": 250, "count": 1, "next_page_url": None},
                         "data": [{"id": 1}]}),
        ]
        client = make_client(session)
        pages = list(client.iterate_fixtures_between(1000, 2000))
        self.assertEqual(len(pages), 1)

    def test_exhausts_retries_and_raises_api_error(self):
        session = MagicMock()
        session.get.return_value = _resp(500, text="server error")
        client = make_client(session, max_retries=2)
        with self.assertRaises(ApiError):
            list(client.iterate_fixtures_between(1000, 2000))
        self.assertEqual(session.get.call_count, 2)

    def test_non_retryable_4xx_raises_immediately(self):
        session = MagicMock()
        session.get.return_value = _resp(401, text="unauthorized")
        client = make_client(session, max_retries=5)
        with self.assertRaises(ApiError):
            list(client.iterate_fixtures_between(1000, 2000))
        self.assertEqual(session.get.call_count, 1)

    def test_max_pages_safety_cap_prevents_infinite_loop(self):
        session = MagicMock()

        def infinite_next_page(*args, **kwargs):
            return _resp(200, {
                "info": {"page": 1, "per_page": 1, "count": 1000,
                          "next_page_url": "https://data.oddalerts.com/api/fixtures/between?page=999"},
                "data": [{"id": 1}],
            })
        session.get.side_effect = infinite_next_page
        client = make_client(session)
        pages = list(client.iterate_fixtures_between(1000, 2000, max_pages=3))
        self.assertEqual(len(pages), 3)

    def test_exhausted_connection_error_retries_never_leak_token(self):
        # Regression test for a real bug found during sample-ingestion
        # validation: requests/urllib3 connection-error objects embed the
        # full request URL (including api_token=...) in their own __str__,
        # and the final "exhausted retries" ApiError was interpolating
        # that raw exception directly instead of redacting it first.
        session = MagicMock()
        leaky_url = (
            "https://data.oddalerts.com/api/fixtures/between?"
            "api_token=SECRET_TOKEN_VALUE&from=1&to=2&competitions=423"
        )
        session.get.side_effect = requests.exceptions.ConnectionError(
            f"HTTPSConnectionPool(host='data.oddalerts.com', port=443): "
            f"Max retries exceeded with url: {leaky_url} "
            f"(Caused by ProxyError('Unable to connect to proxy'))"
        )
        client = make_client(session, max_retries=2)
        with self.assertRaises(ApiError) as ctx:
            list(client.iterate_fixtures_between(1000, 2000))
        self.assertNotIn("SECRET_TOKEN_VALUE", str(ctx.exception))
        self.assertIn("<redacted>", str(ctx.exception))

    def test_error_messages_never_contain_raw_token(self):
        session = MagicMock()
        session.get.return_value = _resp(400, text="bad request for api_token=SECRET_TOKEN_VALUE")
        client = make_client(session, max_retries=3)
        with self.assertRaises(ApiError) as ctx:
            list(client.iterate_fixtures_between(1000, 2000))
        self.assertNotIn("SECRET_TOKEN_VALUE", str(ctx.exception))
        self.assertIn("<redacted>", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
