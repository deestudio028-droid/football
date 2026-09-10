"""Resumable ingestion progress tracking.

One JSON file records, per (competition_id, season_id) window, whether
ingestion has completed and how many pages/fixtures were pulled. On
restart, `ingest.py` skips any window already marked "complete" and
re-verifies (rather than blindly trusting) "in_progress" windows by
checking which raw pages already exist on disk. This means an
interrupted run (crash, network outage, manual stop) can be resumed
without re-downloading data that was already safely persisted.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _window_key(competition_id: int, season_id: int) -> str:
    return f"{competition_id}:{season_id}"


class CheckpointStore:
    def __init__(self, checkpoint_path: Path) -> None:
        self._path = checkpoint_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return {"windows": {}}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def get_window(self, competition_id: int, season_id: int) -> dict[str, Any]:
        return self._state["windows"].get(
            _window_key(competition_id, season_id),
            {"status": "not_started", "pages_fetched": 0, "fixture_count": 0},
        )

    def is_complete(self, competition_id: int, season_id: int) -> bool:
        return self.get_window(competition_id, season_id).get("status") == "complete"

    def mark_page_fetched(
        self, competition_id: int, season_id: int, page: int, fixtures_in_page: int
    ) -> None:
        key = _window_key(competition_id, season_id)
        window = self._state["windows"].setdefault(
            key, {"status": "in_progress", "pages_fetched": 0, "fixture_count": 0}
        )
        window["status"] = "in_progress"
        window["pages_fetched"] = max(window.get("pages_fetched", 0), page)
        window["fixture_count"] = window.get("fixture_count", 0) + fixtures_in_page
        window["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def mark_window_complete(self, competition_id: int, season_id: int) -> None:
        key = _window_key(competition_id, season_id)
        window = self._state["windows"].setdefault(key, {"pages_fetched": 0, "fixture_count": 0})
        window["status"] = "complete"
        window["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def mark_window_failed(self, competition_id: int, season_id: int, error: str) -> None:
        key = _window_key(competition_id, season_id)
        window = self._state["windows"].setdefault(key, {"pages_fetched": 0, "fixture_count": 0})
        window["status"] = "failed"
        window["last_error"] = error
        window["failed_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def summary(self) -> dict[str, Any]:
        return self._state["windows"]
