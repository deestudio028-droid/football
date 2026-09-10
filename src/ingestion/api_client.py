"""Thin, deliberately narrow OddAlerts API client.

Design choices, all intentional:

- Only exposes the endpoints this project is allowed to use for feature
  construction: `/fixtures/between` (with `include=stats`). There is no
  generic "call any endpoint" method, so it is structurally impossible
  for the ingestion pipeline to accidentally call `/predictions`,
  `/probability`, `/correctScores`, or use `include_frozen=true` --
  those are simply not implemented here.
- Retries with exponential backoff + jitter on transient failures
  (connection errors, timeouts, HTTP 429, HTTP 5xx). Does not retry on
  4xx errors other than 429, since those indicate a request problem
  that retrying will not fix.
- Every log line and every raised exception message is passed through
  `logging_setup.redact` before it can reach a log file, so the token
  can never appear in logs even if something goes wrong mid-request.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

from .logging_setup import redact

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ApiError(RuntimeError):
    """Raised for non-retryable or exhausted-retry API failures."""


@dataclass
class FixturesPage:
    page: int
    per_page: int
    count: int
    next_page_url: str | None
    data: list[dict[str, Any]]
    raw: dict[str, Any]


class OddAlertsClient:
    def __init__(
        self,
        base_url: str,
        api_token: str,
        logger,
        timeout_seconds: int = 30,
        max_retries: int = 5,
        backoff_base_seconds: float = 1.5,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._logger = logger
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._backoff_base = backoff_base_seconds
        self._session = session or requests.Session()

    # -- internal request machinery -----------------------------------

    def _request(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET with retry/backoff. `url` may be a full URL (e.g. a
        next_page_url returned by the API) or a path relative to base_url.
        """
        if not url.startswith("http"):
            url = f"{self._base_url}/{url.lstrip('/')}"

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                self._logger.warning(
                    "Request attempt %d/%d failed with connection error: %s",
                    attempt, self._max_retries, redact(str(exc)),
                )
                self._sleep_backoff(attempt)
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise ApiError(
                        f"Non-JSON 200 response from {redact(url)}: {exc}"
                    ) from exc

            if resp.status_code in RETRYABLE_STATUS_CODES:
                last_error = ApiError(
                    f"HTTP {resp.status_code} from {redact(url)}"
                )
                self._logger.warning(
                    "Request attempt %d/%d got retryable status %d",
                    attempt, self._max_retries, resp.status_code,
                )
                self._sleep_backoff(attempt, retry_after=resp.headers.get("Retry-After"))
                continue

            # Non-retryable client error (401, 403, 404, etc.)
            raise ApiError(
                f"Non-retryable HTTP {resp.status_code} from {redact(url)}: "
                f"{redact(resp.text[:500])}"
            )

        # last_error may be a raw requests/urllib3 exception whose own
        # __str__ embeds the full request URL -- including query params,
        # i.e. the token -- regardless of what we passed as `url` above.
        # It MUST be redacted here too, not just the `url` variable.
        raise ApiError(
            f"Exhausted {self._max_retries} retries against {redact(url)}: "
            f"{redact(str(last_error))}"
        )

    def _sleep_backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                delay = float(retry_after)
                time.sleep(delay)
                return
            except ValueError:
                pass
        delay = self._backoff_base ** attempt + random.uniform(0, 0.5)
        time.sleep(delay)

    # -- public, narrow surface ----------------------------------------

    def get_fixtures_between_page(
        self,
        from_unix: int,
        to_unix: int,
        competition_id: int | None = None,
        season_id: int | None = None,
        include: str = "stats",
        page: int = 1,
    ) -> FixturesPage:
        """Fetch a single page of /fixtures/between. Callers should use
        `iterate_fixtures_between` for full pagination.
        """
        params: dict[str, Any] = {
            "api_token": self._api_token,
            "from": from_unix,
            "to": to_unix,
            "include": include,
            "page": page,
        }
        if competition_id is not None:
            params["competitions"] = competition_id
        if season_id is not None:
            params["seasons"] = season_id

        body = self._request("fixtures/between", params=params)
        info = body.get("info", {})
        return FixturesPage(
            page=info.get("page", page),
            per_page=info.get("per_page", 0),
            count=info.get("count", 0),
            next_page_url=info.get("next_page_url"),
            data=body.get("data", []),
            raw=body,
        )

    def iterate_fixtures_between(
        self,
        from_unix: int,
        to_unix: int,
        competition_id: int | None = None,
        season_id: int | None = None,
        include: str = "stats",
        max_pages: int = 200,
    ) -> Iterator[FixturesPage]:
        """Yield every page for a /fixtures/between query, following
        `info.next_page_url` when present. `max_pages` is a hard safety
        cap to guarantee this can never loop forever on an unexpected
        API response shape.
        """
        page_num = 1
        first = self.get_fixtures_between_page(
            from_unix, to_unix, competition_id, season_id, include, page=page_num
        )
        yield first

        next_url = first.next_page_url
        pages_fetched = 1
        while next_url and self._looks_like_real_url(next_url) and pages_fetched < max_pages:
            body = self._request(next_url)
            info = body.get("info", {})
            page_num += 1
            fp = FixturesPage(
                page=info.get("page", page_num),
                per_page=info.get("per_page", 0),
                count=info.get("count", 0),
                next_page_url=info.get("next_page_url"),
                data=body.get("data", []),
                raw=body,
            )
            yield fp
            next_url = fp.next_page_url
            pages_fetched += 1

        if pages_fetched >= max_pages and next_url:
            self._logger.warning(
                "Hit max_pages=%d safety cap while paginating; more data may remain.",
                max_pages,
            )

    @staticmethod
    def _looks_like_real_url(url: str) -> bool:
        # The Postman-documented example response contains a placeholder
        # string ("NEXT PAGE URL WILL APPEAR HERE") instead of a real URL
        # when there genuinely is no next page in some sample payloads.
        # Guard against ever treating that literal string as a URL.
        return url.startswith("http")
