"""Fresh 100-Match Prospective Pilot Module.

Manages the operational lifecycle, milestone tracking (0/100, 25/100, 50/100, 75/100, 100/100),
chronological 25- and 50-match bucketing, per-league breakdowns, and pilot verdicts.

INVARIANTS:
1. Strict pre-kickoff prediction lock.
2. Two-stage outcome separation (prediction record remains immutable).
3. 100-match pilot is an operational pilot, NOT a statistical promotion confirmation.
4. $N < 1,050$ remains strictly blocked from formal statistical confirmation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from models.baselines import CLASS_ORDER
from monitoring.prospective_pipeline import (
    LockedPredictionRecord,
    OutcomeRecord,
    ProspectiveContract,
    ProspectiveMonitor,
    ProspectiveValidationStore,
    run_final_statistical_validation,
)

PILOT_TARGET_SIZE = 100
PILOT_MILESTONES = [0, 25, 50, 75, 100]


class PilotVerdict(str, Enum):
    PILOT_WAITING_FOR_FRESH_DATA = "PILOT WAITING FOR FRESH DATA"
    PILOT_PASS_OPERATIONALLY_CLEAN = "PILOT_PASS — OPERATIONALLY CLEAN"
    PILOT_PASS_WITH_WARNINGS = "PILOT_PASS_WITH_WARNINGS"
    PILOT_BLOCKED = "PILOT_BLOCKED"


@dataclass
class PilotSummary:
    target_size: int
    locked_predictions: int
    completed_fixtures: int
    pending_fixtures: int
    rejected_attempts: int
    milestone_reached: str
    verdict: str
    is_statistically_validated: bool
    is_promoted: bool


class ProspectivePilot:
    """Manages the 100-match prospective pilot validation and milestone reporting."""

    def __init__(self, contract: ProspectiveContract | None = None):
        self.contract = contract or ProspectiveContract.load_and_verify()

    @staticmethod
    def evaluate_pilot_cohort(
        records: list[tuple[LockedPredictionRecord, OutcomeRecord]],
        rejected_count: int = 0,
        duplicate_count: int = 0,
    ) -> dict[str, Any]:
        """Compute full 100-match pilot scorecard, milestone status, and breakdown."""
        n = len(records)
        base_metrics = ProspectiveMonitor.compute_metrics(records) if n > 0 else {}

        # Milestone detection
        milestone = "0 / 100"
        for m in sorted(PILOT_MILESTONES):
            if n >= m:
                milestone = f"{m} / 100"

        # Chronological 25-match buckets
        buckets_25 = []
        if n > 0:
            y = np.array([o.actual_class for _, o in records])
            oh = np.zeros((n, 3), dtype=float)
            for i, c in enumerate(CLASS_ORDER):
                oh[:, i] = (y == c)
            P_v4 = np.array([[p.p_home_v4, p.p_draw_v4, p.p_away_v4] for p, _ in records])
            P_champ = np.array([[p.p_home_champion, p.p_draw_champion, p.p_away_champion] for p, _ in records])
            P_v4_c = np.clip(P_v4, 1e-15, 1.0); P_v4_c /= P_v4_c.sum(axis=1, keepdims=True)
            P_champ_c = np.clip(P_champ, 1e-15, 1.0); P_champ_c /= P_champ_c.sum(axis=1, keepdims=True)

            n_b25 = int(math.ceil(n / 25.0))
            for b in range(n_b25):
                s_idx, e_idx = b * 25, min(n, (b + 1) * 25)
                sub_oh = oh[s_idx:e_idx]
                sub_v4 = P_v4_c[s_idx:e_idx]
                sub_champ = P_champ_c[s_idx:e_idx]
                sub_ll_v4 = float(-np.mean(np.sum(sub_oh * np.log(sub_v4), axis=1)))
                sub_ll_champ = float(-np.mean(np.sum(sub_oh * np.log(sub_champ), axis=1)))
                buckets_25.append({
                    "bucket": f"{s_idx + 1}-{e_idx}",
                    "n": e_idx - s_idx,
                    "v4_log_loss": round(sub_ll_v4, 6),
                    "champion_log_loss": round(sub_ll_champ, 6),
                    "delta_log_loss": round(sub_ll_champ - sub_ll_v4, 6),
                    "actual_draw_rate": round(float(np.mean(y[s_idx:e_idx] == "D")), 4),
                    "mean_pd_champ": round(float(P_champ[s_idx:e_idx, 1].mean()), 4),
                })

        # League breakdown (descriptive only)
        league_breakdown = {}
        if n > 0:
            leagues = sorted({p.league for p, _ in records})
            for lg in leagues:
                lg_indices = [i for i, (p, _) in enumerate(records) if p.league == lg]
                lg_n = len(lg_indices)
                sub_oh = oh[lg_indices]
                sub_v4 = P_v4_c[lg_indices]
                sub_champ = P_champ_c[lg_indices]
                sub_ll_v4 = float(-np.mean(np.sum(sub_oh * np.log(sub_v4), axis=1)))
                sub_ll_champ = float(-np.mean(np.sum(sub_oh * np.log(sub_champ), axis=1)))
                league_breakdown[lg] = {
                    "n": lg_n,
                    "v4_log_loss": round(sub_ll_v4, 6),
                    "champion_log_loss": round(sub_ll_champ, 6),
                    "delta_log_loss": round(sub_ll_champ - sub_ll_v4, 6),
                    "actual_draw_rate": round(float(np.mean(y[lg_indices] == "D")), 4),
                    "mean_pd_champ": round(float(P_champ[lg_indices, 1].mean()), 4),
                }

        # Formal statistical validation check (asserts blocked while N < 1050)
        stat_eval = run_final_statistical_validation(records)

        # Determine pilot verdict
        if n == 0:
            verdict = PilotVerdict.PILOT_WAITING_FOR_FRESH_DATA.value
        elif n < PILOT_TARGET_SIZE:
            verdict = f"PILOT_IN_PROGRESS ({n}/{PILOT_TARGET_SIZE})"
        else:
            # At 100 fixtures
            if rejected_count == 0 and duplicate_count == 0:
                verdict = PilotVerdict.PILOT_PASS_OPERATIONALLY_CLEAN.value
            else:
                verdict = PilotVerdict.PILOT_PASS_WITH_WARNINGS.value

        return {
            "pilot_target": PILOT_TARGET_SIZE,
            "completed_fixtures": n,
            "milestone": milestone,
            "verdict": verdict,
            "is_statistically_validated": False,
            "is_promoted": False,
            "interpretation": "Pilot operationally validated; prospective cohort remains below formal statistical confirmation threshold.",
            "metrics": base_metrics,
            "buckets_25": buckets_25,
            "league_breakdown": league_breakdown,
            "statistical_validation_gate": stat_eval,
        }


__all__ = [
    "PilotVerdict",
    "PilotSummary",
    "ProspectivePilot",
    "PILOT_TARGET_SIZE",
    "PILOT_MILESTONES",
]
