"""Ingestion orchestrator: walks competitions x seasons, pulls
/fixtures/between, saves raw pages, normalizes, upserts into SQLite,
and checkpoints progress for safe resumption.

Resume granularity is per (competition_id, season_id) window, not
per-page. This is a deliberate simplification: a season is at most a
couple of pages (~380 fixtures / 250 per page), so re-walking an
incomplete window from page 1 on resume is cheap, and it avoids the
much harder problem of resuming mid-way through an API-driven
`next_page_url` pagination chain that cannot be jumped into at an
arbitrary page. A window already marked "complete" in the checkpoint
file is never re-fetched. Raw-file writes and DB upserts are also
independently idempotent (same page file gets overwritten with
identical content; same fixture_id upserts to the same row), so even
an unnecessary re-fetch cannot create duplicate or inconsistent data.

Explicitly out of scope in this module, per project instructions:
feature engineering, attack/defence strength, Poisson/Monte Carlo
simulation, ML, predictions, `include_frozen`, and OddAlerts'
`/predictions`, `/probability`, `/correctScores` outputs.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api_client import ApiError, OddAlertsClient
from .checkpoint import CheckpointStore
from .config import Competition, Config, Season, load_config
from .db import FixtureDB
from .logging_setup import setup_logging
from .normalize import NormalizationError, normalize_fixture
from .raw_store import RawStore
from .season_dates import season_unix_bracket


@dataclass
class WindowResult:
    competition_id: int
    season_id: int
    season_name: str
    pages_fetched: int
    fixtures_seen: int
    fixtures_stored: int
    normalization_errors: int
    status: str


def ingest_window(
    client: OddAlertsClient,
    raw_store: RawStore,
    db: FixtureDB,
    checkpoint: CheckpointStore,
    logger,
    competition: Competition,
    season: Season,
    max_pages: int,
) -> WindowResult:
    if checkpoint.is_complete(competition.competition_id, season.season_id):
        logger.info(
            "SKIP (already complete): %s season_id=%d",
            competition.name, season.season_id,
        )
        prior = checkpoint.get_window(competition.competition_id, season.season_id)
        return WindowResult(
            competition.competition_id, season.season_id, season.season_name,
            pages_fetched=prior.get("pages_fetched", 0),
            fixtures_seen=prior.get("fixture_count", 0),
            fixtures_stored=0, normalization_errors=0, status="skipped_already_complete",
        )

    from_unix, to_unix = season_unix_bracket(season.season_name)
    pages_fetched = 0
    fixtures_seen = 0
    fixtures_stored = 0
    normalization_errors = 0

    try:
        for fp in client.iterate_fixtures_between(
            from_unix, to_unix,
            competition_id=competition.competition_id,
            season_id=season.season_id,
            include="stats",
            max_pages=max_pages,
        ):
            raw_store.save_page(competition.competition_id, season.season_id, fp.page, fp.raw)
            pages_fetched += 1
            fixtures_seen += len(fp.data)

            normalized_records = []
            for raw_fixture in fp.data:
                try:
                    normalized_records.append(normalize_fixture(raw_fixture))
                except NormalizationError as exc:
                    normalization_errors += 1
                    logger.error(
                        "Failed to normalize a fixture in %s season_id=%d page=%d: %s",
                        competition.name, season.season_id, fp.page, exc,
                    )
            if normalized_records:
                db.upsert_fixtures(normalized_records)
                fixtures_stored += len(normalized_records)

            checkpoint.mark_page_fetched(
                competition.competition_id, season.season_id, fp.page, len(fp.data)
            )
            logger.info(
                "%s season_id=%d page=%d: %d fixtures fetched, %d stored, count=%d",
                competition.name, season.season_id, fp.page, len(fp.data),
                len(normalized_records), fp.count,
            )

        checkpoint.mark_window_complete(competition.competition_id, season.season_id)
        status = "complete"
    except ApiError as exc:
        checkpoint.mark_window_failed(competition.competition_id, season.season_id, str(exc))
        logger.error(
            "Window failed for %s season_id=%d: %s", competition.name, season.season_id, exc
        )
        status = "failed"

    return WindowResult(
        competition.competition_id, season.season_id, season.season_name,
        pages_fetched, fixtures_seen, fixtures_stored, normalization_errors, status,
    )


def run(
    config: Config,
    competition_ids: list[int] | None = None,
    season_names: list[str] | None = None,
    max_pages: int = 50,
) -> list[WindowResult]:
    logger = setup_logging(config.log_dir)
    raw_store = RawStore(config.raw_data_dir)
    client = OddAlertsClient(
        base_url=config.base_url,
        api_token=config.api_token,
        logger=logger,
        timeout_seconds=config.request_timeout_seconds,
        max_retries=config.max_retries,
        backoff_base_seconds=config.backoff_base_seconds,
    )
    checkpoint = CheckpointStore(config.checkpoint_path)
    results: list[WindowResult] = []

    with FixtureDB(config.db_path) as db:
        for competition in config.competitions:
            if competition_ids and competition.competition_id not in competition_ids:
                continue
            for season in competition.seasons:
                if season_names and season.season_name not in season_names:
                    continue
                logger.info(
                    "Starting window: %s (%d) season %s (season_id=%d)",
                    competition.name, competition.competition_id,
                    season.season_name, season.season_id,
                )
                result = ingest_window(
                    client, raw_store, db, checkpoint, logger, competition, season, max_pages
                )
                results.append(result)

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OddAlerts historical fixture ingestion.")
    parser.add_argument("--project-root", default=".", help="Project root containing .env/config/data.")
    parser.add_argument(
        "--competitions", default=None,
        help="Comma-separated competition_ids to restrict to (default: all 5 configured leagues).",
    )
    parser.add_argument(
        "--seasons", default=None,
        help="Comma-separated season names (e.g. '2024/2025') to restrict to (default: all configured seasons).",
    )
    parser.add_argument(
        "--max-pages", type=int, default=50,
        help="Safety cap on pages followed per window (default 50; a season is normally 1-2 pages).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    competition_ids = (
        [int(c) for c in args.competitions.split(",")] if args.competitions else None
    )
    season_names = args.seasons.split(",") if args.seasons else None

    config = load_config(Path(args.project_root))
    results = run(config, competition_ids, season_names, args.max_pages)

    total_stored = sum(r.fixtures_stored for r in results)
    total_errors = sum(r.normalization_errors for r in results)
    failed = [r for r in results if r.status == "failed"]

    print(f"Windows processed: {len(results)}")
    print(f"Fixtures stored: {total_stored}")
    print(f"Normalization errors: {total_errors}")
    print(f"Failed windows: {len(failed)}")
    for r in failed:
        print(f"  FAILED competition_id={r.competition_id} season_id={r.season_id}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
