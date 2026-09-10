"""Operational Runner for V4.6 Live Prospective Fixture Collection with Real Feed Integration.

Usage:
    python src/monitoring/run_v46_live_collection.py [--db-path PATH] [--safety-buffer-minutes MINS] [--dry-run] [--provider {default,oddalerts,local}]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "dixon_coles"))

from data.providers.football_fixture_provider import (
    BaseFixtureProvider,
    FixtureProviderError,
    LocalDatabaseFixtureProvider,
    NoProviderCredentialsError,
    OddAlertsFixtureProvider,
    UpcomingFixture,
    get_default_fixture_provider,
)
from features.elo import ELO_COLUMNS, load_elo_features
from features.online_attack_defense import AD_COLUMNS, compute_ad_states, fit_baseline_rates
from models.baselines import CLASS_ORDER
from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
from models.data import load_supervised_dataset
from models.poisson import predict_poisson
from models.v4_artifact import load_v4_artifact
from models.v4_2_draw_resolution_candidate import (
    DrawResolutionConfig,
    compute_dibp_scoreline_matrix,
    predict_draw_resolution,
)
from models.v4_6_physical_draw_gate import (
    V46PhysicalGateConfig,
    predict_v4_6_physical_gate,
)
from monitoring.v46_live_prospective_collector import (
    DEFAULT_LIVE_DB,
    DuplicateFixtureError,
    HistoricalFixtureRejectionError,
    PreKickoffViolationError,
    V46LiveProspectiveCollector,
    init_live_prospective_db,
    load_known_historical_fixture_ids,
)

logger = logging.getLogger("run_v46_live_collection")

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
V4_ARTIFACT_PATH = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"


def run_collection(
    db_path: Path = DEFAULT_LIVE_DB,
    safety_buffer_minutes: int = 15,
    dry_run: bool = False,
    provider: Optional[BaseFixtureProvider] = None,
    provider_name: str = "default",
) -> Dict[str, Any]:
    """Execute live prospective fixture collection cycle."""
    init_live_prospective_db(db_path)
    collector = V46LiveProspectiveCollector(db_path=db_path)
    historical_ids = collector.historical_ids

    # Query already locked fixtures in live prospective store
    conn_live = sqlite3.connect(str(db_path))
    cur = conn_live.cursor()
    cur.execute("SELECT fixture_id FROM live_predictions")
    locked_ids = set(int(row[0]) for row in cur.fetchall())
    conn_live.close()

    # Determine Fixture Provider
    fixture_provider = provider
    if fixture_provider is None:
        if provider_name == "oddalerts":
            fixture_provider = OddAlertsFixtureProvider()
        elif provider_name == "local":
            fixture_provider = LocalDatabaseFixtureProvider()
        else:
            fixture_provider = get_default_fixture_provider()

    provider_id_name = fixture_provider.__class__.__name__

    # 1. Fetch Upcoming Fixtures from Provider
    try:
        upcoming_fixtures: List[UpcomingFixture] = fixture_provider.get_upcoming_fixtures(days_ahead=14)
    except NoProviderCredentialsError:
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider_id_name,
            "discovered_fixtures": 0,
            "eligible_fixtures": 0,
            "new_predictions_locked": 0,
            "duplicates_rejected": 0,
            "historical_rejected": 0,
            "late_fixtures_rejected": 0,
            "missing_features_rejected": 0,
            "total_live_store_locked": len(locked_ids),
            "dry_run": dry_run,
            "status": "NO LIVE PROVIDER CONFIGURED (WAITING SAFELY FOR CREDENTIALS/FEED)",
        }
        _print_summary(summary)
        return summary
    except Exception as exc:
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider_id_name,
            "discovered_fixtures": 0,
            "eligible_fixtures": 0,
            "new_predictions_locked": 0,
            "duplicates_rejected": 0,
            "historical_rejected": 0,
            "late_fixtures_rejected": 0,
            "missing_features_rejected": 0,
            "total_live_store_locked": len(locked_ids),
            "dry_run": dry_run,
            "status": f"PROVIDER_ERROR: {exc}",
        }
        _print_summary(summary)
        return summary

    discovered_cnt = len(upcoming_fixtures)
    now_utc = datetime.now(timezone.utc)
    safety_delta = timedelta(minutes=safety_buffer_minutes)

    eligible_cnt = 0
    new_locked_cnt = 0
    duplicate_cnt = 0
    historical_cnt = 0
    late_cnt = 0
    missing_features_cnt = 0

    if discovered_cnt == 0:
        summary = {
            "timestamp": now_utc.isoformat(),
            "provider": provider_id_name,
            "discovered_fixtures": 0,
            "eligible_fixtures": 0,
            "new_predictions_locked": 0,
            "duplicates_rejected": 0,
            "historical_rejected": 0,
            "late_fixtures_rejected": 0,
            "missing_features_rejected": 0,
            "total_live_store_locked": len(locked_ids),
            "dry_run": dry_run,
            "status": "IDLE — NO UPCOMING FIXTURES IN FEED (WAITING FOR SCHEDULED MATCHES)",
        }
        _print_summary(summary)
        return summary

    # Load Model Artifact & Feature Stores if eligible fixtures exist
    v4 = load_v4_artifact(V4_ARTIFACT_PATH)
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)

    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])

    cfg_v42 = DrawResolutionConfig(stacking_intercept=0.2450, use_dibp=True, dibp_inflation_p=0.0500)

    for f in upcoming_fixtures:
        fid = f.fixture_id

        # Check duplicate
        if fid in locked_ids:
            duplicate_cnt += 1
            continue

        # Check historical
        if fid in historical_ids:
            historical_cnt += 1
            continue

        # Check kickoff boundary
        kickoff_dt = datetime.fromisoformat(f.scheduled_kickoff.replace("Z", "+00:00"))
        if now_utc > (kickoff_dt - safety_delta):
            late_cnt += 1
            continue

        # Check causal feature availability
        if fid not in meta["fixture_id"].values:
            missing_features_cnt += 1
            continue

        eligible_cnt += 1

        if dry_run:
            continue

        # Compute Pre-Match Features
        m_idx = int(np.where(meta.fixture_id == fid)[0][0])
        X_sub = X.iloc[[m_idx]].reset_index(drop=True)
        E = v4.preprocessor.transform(X_sub[list(v4.feature_columns)])
        lh = float(v4.model_home_goals.predict(E)[0])
        la = float(v4.model_away_goals.predict(E)[0])
        pr_v4 = predict_poisson(np.array([lh]), np.array([la]), list(v4.class_order))[0]
        pv4 = np.array([pr_v4.probabilities["H"], pr_v4.probabilities["D"], pr_v4.probabilities["A"]])

        abs_elo = float(abs((elo.loc[fid]["home_elo"] + 100.0) - elo.loc[fid]["away_elo"]))
        grid = compute_dibp_scoreline_matrix(lh, la, -0.0560, 0.05)
        p_res = predict_draw_resolution(lh, la, pv4, abs_elo, f.league_name, cfg_v42)
        pv42 = np.array([p_res.probabilities["H"], p_res.probabilities["D"], p_res.probabilities["A"]])
        low_score = float(grid[0, 0] + grid[1, 1] + grid[2, 2])

        # Lock Prediction
        collector.collect_and_lock_prediction(
            fixture_id=fid,
            competition_id=f.league_id,
            competition_name=f.league_name,
            home_team=f.home_team,
            away_team=f.away_team,
            scheduled_kickoff=kickoff_dt,
            p_v4=pv4,
            p_v42=pv42,
            abs_elo_diff=abs_elo,
            lambda_home=lh,
            lambda_away=la,
            low_score_prob=low_score,
            prediction_timestamp=now_utc,
        )
        new_locked_cnt += 1
        locked_ids.add(fid)

    status_str = "DRY_RUN_COMPLETED (NO MUTATION)" if dry_run else "COLLECTION_CYCLE_COMPLETE"

    summary = {
        "timestamp": now_utc.isoformat(),
        "provider": provider_id_name,
        "discovered_fixtures": discovered_cnt,
        "eligible_fixtures": eligible_cnt,
        "new_predictions_locked": new_locked_cnt,
        "duplicates_rejected": duplicate_cnt,
        "historical_rejected": historical_cnt,
        "late_fixtures_rejected": late_cnt,
        "missing_features_rejected": missing_features_cnt,
        "total_live_store_locked": len(locked_ids),
        "dry_run": dry_run,
        "status": status_str,
    }
    _print_summary(summary)
    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 80)
    print("REAL FIXTURE FEED & LIVE V4.6 PROSPECTIVE COLLECTION")
    print("=" * 80)
    print(f"Timestamp:                 {summary['timestamp']}")
    print(f"Provider:                  {summary['provider']}")
    print(f"Discovered fixtures:       {summary['discovered_fixtures']}")
    print(f"Eligible fixtures:         {summary['eligible_fixtures']}")
    print(f"New predictions locked:    {summary['new_predictions_locked']}")
    print(f"Duplicates rejected:       {summary['duplicates_rejected']}")
    print(f"Historical rejected:       {summary['historical_rejected']}")
    print(f"Late fixtures rejected:    {summary['late_fixtures_rejected']}")
    print(f"Missing features rejected: {summary['missing_features_rejected']}")
    print(f"Total Live Store Locked:   {summary['total_live_store_locked']}")
    print(f"Dry Run Mode:              {summary['dry_run']}")
    print(f"Status:                    {summary['status']}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4.6 Live Prospective Collection Runner")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_LIVE_DB, help="Path to live prospective SQLite database")
    parser.add_argument("--safety-buffer-minutes", type=int, default=15, help="Pre-kickoff safety buffer in minutes")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode: validate without database mutation")
    parser.add_argument("--provider", type=str, default="default", choices=["default", "oddalerts", "local"], help="Fixture provider to use")
    args = parser.parse_args()
    run_collection(
        db_path=args.db_path,
        safety_buffer_minutes=args.safety_buffer_minutes,
        dry_run=args.dry_run,
        provider_name=args.provider,
    )
