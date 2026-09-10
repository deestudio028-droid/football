"""Durable local monitoring store (append-only JSONL).

WHY JSONL RATHER THAN SQLITE
    Both were authorized ("prefer SQLite or JSONL, whichever integrates
    cleanly"). JSONL was chosen because:
      - it adds no dependency and no schema migration;
      - it is append-only, so an emitted observation cannot be silently
        rewritten;
      - the project's existing SQLite files (features.db, matches.db) are
        pinned baselines, and adding monitoring tables to them -- or
        introducing a parallel schema to maintain -- would touch or
        shadow data the governance rules protect.

No table is added to any existing database. The store writes only to the
path the caller supplies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .events import MonitoringEvent, SchemaViolation, validate_event


class JsonlEventStore:
    """Append-only JSONL event store.

    Deliberately minimal: append, iterate, count. No update, no delete --
    monitoring history is evidence, not mutable state.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # -- writing --------------------------------------------------------
    def append(self, event: MonitoringEvent) -> MonitoringEvent:
        payload = event.to_dict()
        validate_event(payload)  # defence in depth; also validated at construction
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        return event

    def append_all(self, events: Iterable[MonitoringEvent]) -> list[MonitoringEvent]:
        return [self.append(e) for e in events]

    # -- reading --------------------------------------------------------
    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SchemaViolation(f"corrupt monitoring record at line {line_no}") from exc
        return iter(out)

    def read_all(self) -> list[dict[str, Any]]:
        return list(iter(self))

    def count(self) -> int:
        return len(self.read_all())

    def signal_ids(self) -> set[str]:
        return {rec["signal_id"] for rec in self.read_all()}


__all__ = ["JsonlEventStore"]
