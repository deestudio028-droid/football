"""V4.6 Live Prospective Metrics & Milestone Engine.

Calculates continuous live validation metrics and milestone progress exclusively from
the isolated `research/v4_promotion/live_v46_prospective.sqlite` store.

INVARIANTS:
1. Zero Historical Contamination: Evaluates only records in the live prospective store.
2. Read-Only Metric Computation: Never modifies prediction or outcome records.
3. Milestone Tracking: Automatically detects progressive milestones (N=100 to N=1,301).
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LIVE_DB = PROJECT_ROOT / "research" / "v4_promotion" / "live_v46_prospective.sqlite"

POWER_GATE_TARGET = 1050
FULL_COHORT_TARGET = 1301
MILESTONE_THRESHOLDS = [100, 300, 450, 600, 750, 900, 1050, 1301]


@dataclass
class LiveProspectiveMetrics:
    """Container for live prospective metrics."""
    fresh_total_locked: int
    completed_reconciled: int
    pending_fixtures: int
    remaining_to_power_gate: int
    power_gate_progress_pct: float
    current_status: str
    v4_accuracy_pct: float
    v4_correct: int
    v4_6_accuracy_pct: float
    v4_6_correct: int
    delta_accuracy_pct: float
    draw_predictions: int
    correct_draws: int
    draw_precision_pct: float
    draw_recall_pct: float
    draw_f1: float
    macro_f1: float
    good_draw_overrides: int
    bad_draw_overrides: int
    neutral_overrides: int
    net_transition_gain: int
    active_milestone: Optional[int]


def compute_live_metrics(db_path: Path = DEFAULT_LIVE_DB) -> LiveProspectiveMetrics:
    """Compute live prospective validation metrics from the isolated database."""
    if not db_path.exists():
        return LiveProspectiveMetrics(
            fresh_total_locked=0,
            completed_reconciled=0,
            pending_fixtures=0,
            remaining_to_power_gate=POWER_GATE_TARGET,
            power_gate_progress_pct=0.0,
            current_status="COLLECTING (N=0 / 1,050)",
            v4_accuracy_pct=0.0,
            v4_correct=0,
            v4_6_accuracy_pct=0.0,
            v4_6_correct=0,
            delta_accuracy_pct=0.0,
            draw_predictions=0,
            correct_draws=0,
            draw_precision_pct=0.0,
            draw_recall_pct=0.0,
            draw_f1=0.0,
            macro_f1=0.0,
            good_draw_overrides=0,
            bad_draw_overrides=0,
            neutral_overrides=0,
            net_transition_gain=0,
            active_milestone=None,
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Total locked predictions
    cur.execute("SELECT COUNT(*) FROM live_predictions")
    total_locked = int(cur.fetchone()[0])

    # Reconciled matches with final outcomes
    cur.execute("""
        SELECT p.*, o.home_goals, o.away_goals, o.actual_outcome
        FROM live_predictions p
        JOIN match_outcomes o ON p.fixture_id = o.fixture_id
        ORDER BY p.scheduled_kickoff ASC
    """)
    rows = cur.fetchall()
    conn.close()

    n_completed = len(rows)
    n_pending = total_locked - n_completed
    remaining_power = max(0, POWER_GATE_TARGET - n_completed)
    power_pct = round(min(100.0, (n_completed / POWER_GATE_TARGET) * 100.0), 1)

    if n_completed == 0:
        return LiveProspectiveMetrics(
            fresh_total_locked=total_locked,
            completed_reconciled=0,
            pending_fixtures=n_pending,
            remaining_to_power_gate=remaining_power,
            power_gate_progress_pct=power_pct,
            current_status=f"COLLECTING (N=0 / {POWER_GATE_TARGET})",
            v4_accuracy_pct=0.0,
            v4_correct=0,
            v4_6_accuracy_pct=0.0,
            v4_6_correct=0,
            delta_accuracy_pct=0.0,
            draw_predictions=0,
            correct_draws=0,
            draw_precision_pct=0.0,
            draw_recall_pct=0.0,
            draw_f1=0.0,
            macro_f1=0.0,
            good_draw_overrides=0,
            bad_draw_overrides=0,
            neutral_overrides=0,
            net_transition_gain=0,
            active_milestone=None,
        )

    y_true = [r["actual_outcome"] for r in rows]
    p_v4 = [r["v4_base_decision"] for r in rows]
    p_v46 = [r["v4_6_final_decision"] for r in rows]

    corr_v4 = sum(1 for yt, yp in zip(y_true, p_v4) if yt == yp)
    corr_v46 = sum(1 for yt, yp in zip(y_true, p_v46) if yt == yp)

    d_preds = sum(1 for p in p_v46 if p == "D")
    c_draws = sum(1 for yt, yp in zip(y_true, p_v46) if yt == "D" and yp == "D")

    lost_v4 = sum(1 for i in range(n_completed) if p_v46[i] == "D" and p_v4[i] == y_true[i] and y_true[i] != "D")
    gained_draws = sum(1 for i in range(n_completed) if p_v46[i] == "D" and p_v4[i] != y_true[i] and y_true[i] == "D")
    neutral_err = sum(1 for i in range(n_completed) if p_v46[i] == "D" and p_v4[i] != y_true[i] and y_true[i] != "D")

    draw_prec = (c_draws / d_preds * 100.0) if d_preds > 0 else 0.0
    act_draws = y_true.count("D")
    draw_rec = (c_draws / act_draws * 100.0) if act_draws > 0 else 0.0
    draw_f1 = (2 * (draw_prec / 100.0) * (draw_rec / 100.0)) / ((draw_prec / 100.0) + (draw_rec / 100.0)) if (draw_prec + draw_rec) > 0 else 0.0

    # Macro F1
    f1_list = []
    for c in ["H", "D", "A"]:
        tp = sum(1 for yt, yp in zip(y_true, p_v46) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, p_v46) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, p_v46) if yt == c and yp != c)
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0.0
        f1_list.append(f)
    macro_f1 = float(np.mean(f1_list))

    # Determine Active Milestone
    active_m = None
    for m in reversed(MILESTONE_THRESHOLDS):
        if n_completed >= m:
            active_m = m
            break

    status = (
        "POWER GATE REACHED (READY FOR FORMAL STATISTICAL AUDIT)"
        if n_completed >= POWER_GATE_TARGET
        else (f"MILESTONE N={active_m} REACHED" if active_m else f"COLLECTING (N={n_completed} / {POWER_GATE_TARGET})")
    )

    return LiveProspectiveMetrics(
        fresh_total_locked=total_locked,
        completed_reconciled=n_completed,
        pending_fixtures=n_pending,
        remaining_to_power_gate=remaining_power,
        power_gate_progress_pct=power_pct,
        current_status=status,
        v4_accuracy_pct=round(corr_v4 / n_completed * 100.0, 2),
        v4_correct=corr_v4,
        v4_6_accuracy_pct=round(corr_v46 / n_completed * 100.0, 2),
        v4_6_correct=corr_v46,
        delta_accuracy_pct=round((corr_v46 - corr_v4) / n_completed * 100.0, 2),
        draw_predictions=d_preds,
        correct_draws=c_draws,
        draw_precision_pct=round(draw_prec, 1),
        draw_recall_pct=round(draw_rec, 1),
        draw_f1=round(draw_f1, 4),
        macro_f1=round(macro_f1, 4),
        good_draw_overrides=gained_draws,
        bad_draw_overrides=lost_v4,
        neutral_overrides=neutral_err,
        net_transition_gain=gained_draws - lost_v4,
        active_milestone=active_m,
    )


def format_monitoring_dashboard(metrics: LiveProspectiveMetrics) -> str:
    """Render structured monitoring dashboard text."""
    lines = [
        "=" * 96,
        "V4.6 LIVE PROSPECTIVE MONITORING DASHBOARD",
        "=" * 96,
        f"Cohort Progress:          N = {metrics.completed_reconciled} / {POWER_GATE_TARGET} ({metrics.power_gate_progress_pct:.1f}% toward Statistical Power Gate)",
        f"Fresh Total Locked:       {metrics.fresh_total_locked} fixtures",
        f"Completed / Reconciled:   {metrics.completed_reconciled} fixtures",
        f"Pending Kickoff/Outcome:  {metrics.pending_fixtures} fixtures",
        f"Remaining to N=1,050:     {metrics.remaining_to_power_gate} fixtures",
        f"Current Status:           {metrics.current_status}",
        "",
        "ACCURACY & SCORECARD:",
        f"  V4 Baseline Accuracy:   {metrics.v4_accuracy_pct:6.2f}% ({metrics.v4_correct} / {metrics.completed_reconciled})",
        f"  V4.6 Candidate Accuracy:{metrics.v4_6_accuracy_pct:6.2f}% ({metrics.v4_6_correct} / {metrics.completed_reconciled})",
        f"  Delta Accuracy:         {metrics.delta_accuracy_pct:+6.2f}%",
        f"  Macro F1 Score:         {metrics.macro_f1:.4f}",
        "",
        "DRAW METRICS:",
        f"  Draw Predictions:       {metrics.draw_predictions}",
        f"  Correct Draws:          {metrics.correct_draws}",
        f"  Draw Precision:         {metrics.draw_precision_pct:6.1f}%",
        f"  Draw Recall:            {metrics.draw_recall_pct:6.1f}%",
        f"  Draw F1 Score:          {metrics.draw_f1:.4f}",
        "",
        "ERROR ECONOMICS:",
        f"  Good Draw Overrides:    {metrics.good_draw_overrides} (Free Draw Wins)",
        f"  Bad Draw Overrides:     {metrics.bad_draw_overrides} (Sacrificed V4 True Positives)",
        f"  Neutral Overrides:      {metrics.neutral_overrides} (Neutral Error Shift)",
        f"  Net Transition Gain:    {metrics.net_transition_gain:+d} net correct predictions",
        "=" * 96,
    ]
    return "\n".join(lines)
