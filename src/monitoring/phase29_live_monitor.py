"""Phase 29 Live Prospective Monitoring & Dashboard Script.

Usage:
    python src/monitoring/phase29_live_monitor.py [--db-path PATH]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from monitoring.v46_live_metrics import (
    compute_live_metrics,
    format_monitoring_dashboard,
)
from monitoring.v46_live_prospective_collector import DEFAULT_LIVE_DB


def display_phase29_monitor(db_path: Path = DEFAULT_LIVE_DB) -> Dict[str, Any]:
    """Calculate and display Phase 29 live prospective validation status."""
    metrics = compute_live_metrics(db_path)
    
    print("\n" + format_monitoring_dashboard(metrics) + "\n")
    
    return {
        "fresh_total_locked": metrics.fresh_total_locked,
        "completed_reconciled": metrics.completed_reconciled,
        "pending_fixtures": metrics.pending_fixtures,
        "remaining_to_power_gate": metrics.remaining_to_power_gate,
        "power_gate_progress_pct": metrics.power_gate_progress_pct,
        "status": metrics.current_status,
        "v4_accuracy_pct": metrics.v4_accuracy_pct,
        "v4_6_accuracy_pct": metrics.v4_6_accuracy_pct,
        "delta_accuracy_pct": metrics.delta_accuracy_pct,
        "draw_predictions": metrics.draw_predictions,
        "correct_draws": metrics.correct_draws,
        "draw_precision_pct": metrics.draw_precision_pct,
        "draw_recall_pct": metrics.draw_recall_pct,
        "draw_f1": metrics.draw_f1,
        "macro_f1": metrics.macro_f1,
        "good_draw_overrides": metrics.good_draw_overrides,
        "bad_draw_overrides": metrics.bad_draw_overrides,
        "neutral_overrides": metrics.neutral_overrides,
        "net_transition_gain": metrics.net_transition_gain,
        "active_milestone": metrics.active_milestone,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 29 Live Prospective Monitor")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_LIVE_DB, help="Path to live prospective SQLite database")
    args = parser.parse_args()
    display_phase29_monitor(db_path=args.db_path)
