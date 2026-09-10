"""Operational Runner for V4.6 Prospective Outcome Reconciliation.

Usage:
    python src/monitoring/run_v46_outcome_reconciliation.py [--db-path PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from monitoring.v46_live_metrics import (
    compute_live_metrics,
    format_monitoring_dashboard,
)
from monitoring.v46_live_prospective_collector import (
    DEFAULT_LIVE_DB,
    init_live_prospective_db,
)
from monitoring.v46_outcome_reconciler import (
    HashMismatchTamperError,
    UnlockedFixtureOutcomeError,
    V46OutcomeReconciler,
)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"


def run_reconciliation(
    db_path: Path = DEFAULT_LIVE_DB,
    outcomes_source_db: Optional[Path] = MATCHES_DB,
) -> Dict[str, Any]:
    """Execute live prospective outcome reconciliation cycle."""
    init_live_prospective_db(db_path)
    reconciler = V46OutcomeReconciler(db_path=db_path)

    # 1. Query pending locked fixtures
    conn_live = sqlite3.connect(str(db_path))
    conn_live.row_factory = sqlite3.Row
    cur = conn_live.cursor()
    cur.execute("""
        SELECT p.fixture_id, p.scheduled_kickoff, p.home_team, p.away_team
        FROM live_predictions p
        LEFT JOIN match_outcomes o ON p.fixture_id = o.fixture_id
        WHERE o.fixture_id IS NULL
    """)
    pending_rows = cur.fetchall()
    conn_live.close()

    pending_cnt = len(pending_rows)
    reconciled_cnt = 0
    not_finished_cnt = 0

    if pending_cnt == 0:
        metrics = compute_live_metrics(db_path)
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pending_fixtures_checked": 0,
            "outcomes_reconciled": 0,
            "uncompleted_fixtures": 0,
            "total_completed_N": metrics.completed_reconciled,
            "status": "IDLE — NO PENDING FIXTURES TO RECONCILE",
        }
        _print_reconciliation_summary(summary, metrics)
        return summary

    # 2. Check outcomes from source DB
    conn_src = sqlite3.connect(f"file:{outcomes_source_db}?mode=ro", uri=True)
    p_fids = [int(r["fixture_id"]) for r in pending_rows]
    q_marks = ",".join("?" for _ in p_fids)
    df_src = pd.read_sql_query(f"""
        SELECT fixture_id, status, home_goals, away_goals
        FROM fixtures
        WHERE fixture_id IN ({q_marks}) AND status = 'FT'
    """, conn_src, params=p_fids)
    conn_src.close()

    finished_map = {int(r["fixture_id"]): (int(r["home_goals"]), int(r["away_goals"])) for _, r in df_src.iterrows()}

    for r in pending_rows:
        fid = int(r["fixture_id"])
        if fid in finished_map:
            hg, ag = finished_map[fid]
            reconciler.reconcile_fixture_outcome(fixture_id=fid, home_goals=hg, away_goals=ag, status="FT")
            reconciled_cnt += 1
        else:
            not_finished_cnt += 1

    metrics = compute_live_metrics(db_path)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pending_fixtures_checked": pending_cnt,
        "outcomes_reconciled": reconciled_cnt,
        "uncompleted_fixtures": not_finished_cnt,
        "total_completed_N": metrics.completed_reconciled,
        "status": "RECONCILIATION_CYCLE_COMPLETE",
    }
    _print_reconciliation_summary(summary, metrics)
    return summary


def _print_reconciliation_summary(summary: Dict[str, Any], metrics: Any) -> None:
    print("=" * 80)
    print("LIVE V4.6 PROSPECTIVE OUTCOME RECONCILIATION SUMMARY")
    print("=" * 80)
    print(f"Timestamp:                {summary['timestamp']}")
    print(f"Pending fixtures checked: {summary['pending_fixtures_checked']}")
    print(f"Outcomes reconciled:      {summary['outcomes_reconciled']}")
    print(f"Uncompleted fixtures:     {summary['uncompleted_fixtures']}")
    print(f"Total Completed N:        {summary['total_completed_N']}")
    print(f"Status:                   {summary['status']}")
    print("=" * 80)
    print("\n" + format_monitoring_dashboard(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4.6 Live Prospective Outcome Reconciler")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_LIVE_DB, help="Path to live prospective SQLite database")
    args = parser.parse_args()
    run_reconciliation(db_path=args.db_path)
