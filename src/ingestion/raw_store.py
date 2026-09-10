"""Persists raw, unmodified API response bodies to disk.

Raw storage is intentionally dumb: it writes exactly what the API
returned (the full `{info, data}` body), one file per page fetched,
under `data/raw/<competition_id>/<season_id>/`. It never mutates,
filters, or reshapes the response -- with exactly one, narrowly-scoped
exception described below. This is the immutable source of truth that
the normalized layer (`normalize.py` + `db.py`) is built from -- if a
normalization bug is ever found, the raw files let us recompute
without re-hitting the API.

Pagination-URL credential sanitization
---------------------------------------
OddAlerts embeds the live API token directly in `info.next_page_url`
(and, if present, any other pagination-URL field in `info`) on every
paginated response -- this was discovered empirically during the EPL
2024/25 smoke-test validation: `data/raw/423/6484/fixtures_between_page_0001.json`
contained the real token inside `info.next_page_url` before this fix.

Persisting that field verbatim would mean every raw page file becomes
a place a live credential lives at rest, for the life of the project.
That is unacceptable regardless of how "untouched" raw storage is
supposed to be, so this module applies exactly one transformation
before writing to disk: any `api_token=<value>` substring inside a
pagination-URL-shaped field under `info` (currently just
`next_page_url`, but the check is written to catch any sibling field
ending in `_url` too, in case OddAlerts adds `previous_page_url` or
similar later) is replaced with `api_token=REDACTED`.

Nothing else is touched:
  - `data` (the actual fixtures -- scores, stats, IDs, timestamps,
    everything) is never modified.
  - Every other field of `info` (`page`, `per_page`, `count`, `total`,
    `total_pages`, any `warning` text, etc.) is preserved exactly.
  - This sanitization happens only on the copy being written to disk.
    The in-memory response used for actual pagination
    (`api_client.OddAlertsClient.iterate_fixtures_between`) already
    extracted `next_page_url` into `FixturesPage.next_page_url` before
    `RawStore.save_page` is ever called, so pagination behavior is
    completely unaffected by this -- it operates on a separate,
    unsanitized copy that is never written anywhere.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

_TOKEN_IN_URL_RE = re.compile(r"(api_token=)[^&\s]+", re.IGNORECASE)


def _sanitize_pagination_urls_for_storage(body: dict) -> dict:
    """Return a deep copy of `body` with any `api_token=...` value inside
    a pagination-URL field under `info` replaced with a placeholder.
    Does not touch `data` or any other part of `info`.
    """
    sanitized = copy.deepcopy(body)
    info = sanitized.get("info")
    if not isinstance(info, dict):
        return sanitized

    for key, value in info.items():
        if not key.endswith("_url"):
            continue
        if isinstance(value, str) and "api_token=" in value:
            info[key] = _TOKEN_IN_URL_RE.sub(r"\1REDACTED", value)
        # Non-string values (e.g. the boolean `false` OddAlerts returns
        # when there is no next page) are left exactly as-is.

    return sanitized


class RawStore:
    def __init__(self, raw_data_dir: Path) -> None:
        self._root = raw_data_dir

    def path_for_page(self, competition_id: int, season_id: int, page: int) -> Path:
        season_dir = self._root / str(competition_id) / str(season_id)
        return season_dir / f"fixtures_between_page_{page:04d}.json"

    def exists(self, competition_id: int, season_id: int, page: int) -> bool:
        return self.path_for_page(competition_id, season_id, page).exists()

    def save_page(
        self, competition_id: int, season_id: int, page: int, body: dict
    ) -> Path:
        target = self.path_for_page(competition_id, season_id, page)
        target.parent.mkdir(parents=True, exist_ok=True)
        body_to_write = _sanitize_pagination_urls_for_storage(body)
        # Write to a temp file then atomically rename, so a crash mid-write
        # can never leave a corrupt/partial raw file that later gets
        # mistaken for a successfully-saved page during a resume.
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(body_to_write, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        return target

    def load_page(self, competition_id: int, season_id: int, page: int) -> dict:
        target = self.path_for_page(competition_id, season_id, page)
        return json.loads(target.read_text(encoding="utf-8"))

    def iter_saved_pages(self, competition_id: int, season_id: int):
        season_dir = self._root / str(competition_id) / str(season_id)
        if not season_dir.exists():
            return
        for path in sorted(season_dir.glob("fixtures_between_page_*.json")):
            yield path
