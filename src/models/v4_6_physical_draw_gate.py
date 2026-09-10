"""V4.6 Physical Draw Gate Model.

Architecture:
- Base Layer: Frozen V4 Poisson Venue Elo Online A/D Model (default decision = argmax(P_v4)).
- Secondary Signal: V4.2 DIBP + Calibrated Stacking probability layer.
- Physical Gate Layer: Evaluates interpretable pre-kickoff physical constraints grounded in
  Poisson scoreline mechanics to selectively override fragile Home/Away predictions to Draw.

Key Invariants:
1. Zero Continuous ML: No logistic regression, trees, neural nets, or learned meta-classifiers.
2. Strict Pre-Kickoff Causality: Uses only features available prior to match kickoff.
3. Simplex Conservation: Underlying probability distribution remains normalized on the 3-simplex.
4. Production Isolation: Does not modify v4_draw_champion, v4_4, or frozen production assets.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CLASS_ORDER = ["H", "D", "A"]


@dataclass(frozen=True)
class V46PhysicalGateConfig:
    """Configuration for V4.6 Physical Draw Gate Decision Layer."""
    name: str = "robust_optimal_gate"
    version: str = "v4.6-physical-draw-gate"
    draw_prob_threshold: float = 0.2600
    winner_margin_cap: float = 0.1000
    v4_winner_conf_cap: float = 0.4500
    abs_elo_cap: float = 100.0
    tot_expected_goals_cap: float = 2.5000
    low_score_prob_floor: float = 0.0000

    @classmethod
    def robust_optimal_gate(cls) -> V46PhysicalGateConfig:
        """Top robust physical gate (52.50% full accuracy, 50.69% blind accuracy, +4 net wins, 24 correct draws, 0 degraded leagues)."""
        return cls(
            name="robust_optimal_gate",
            draw_prob_threshold=0.2600,
            winner_margin_cap=0.1000,
            v4_winner_conf_cap=0.4500,
            abs_elo_cap=100.0,
            tot_expected_goals_cap=2.5000,
            low_score_prob_floor=0.0000,
        )

    @classmethod
    def high_precision_gate(cls) -> V46PhysicalGateConfig:
        """High-precision conservative gate (52.50% full accuracy, 66 draw preds, 21 correct draws, 31.8% precision, only 17 sacrificed V4 points)."""
        return cls(
            name="high_precision_gate",
            draw_prob_threshold=0.2600,
            winner_margin_cap=0.1000,
            v4_winner_conf_cap=0.4500,
            abs_elo_cap=80.0,
            tot_expected_goals_cap=2.5000,
            low_score_prob_floor=0.0000,
        )

    @classmethod
    def strict_parity_gate(cls) -> V46PhysicalGateConfig:
        """Strict parity gate with low-score concentration constraint."""
        return cls(
            name="strict_parity_gate",
            draw_prob_threshold=0.2600,
            winner_margin_cap=0.0800,
            v4_winner_conf_cap=0.4400,
            abs_elo_cap=75.0,
            tot_expected_goals_cap=2.4500,
            low_score_prob_floor=0.2400,
        )


@dataclass
class V46PhysicalGatePrediction:
    """Output container for V4.6 prediction."""
    fixture_id: Optional[int]
    probabilities: Dict[str, float]
    v4_base_decision: str
    v4_6_final_decision: str
    override_applied: bool
    gate_name: str
    metadata: Dict[str, Any]


def evaluate_physical_draw_gate(
    p_v4: List[float] | np.ndarray,
    p_v42: List[float] | np.ndarray,
    abs_elo_diff: float,
    lambda_home: float,
    lambda_away: float,
    low_score_prob: float = 0.25,
    config: Optional[V46PhysicalGateConfig] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Determine whether match conditions satisfy physical draw gating constraints."""
    if config is None:
        config = V46PhysicalGateConfig.robust_optimal_gate()

    pv4 = np.asarray(p_v4, dtype=float)
    pv42 = np.asarray(p_v42, dtype=float)

    v4_base_dec = CLASS_ORDER[int(np.argmax(pv4))]
    v4_winner_conf = float(max(pv4[0], pv4[2]))

    v42_winner_conf = float(max(pv42[0], pv42[2]))
    v42_draw_prob = float(pv42[1])
    winner_margin = float(v42_winner_conf - v42_draw_prob)

    tot_goals = float(lambda_home + lambda_away)
    goal_diff = float(abs(lambda_home - lambda_away))

    # Evaluate physical gating conditions
    cond_draw_prob = v42_draw_prob >= config.draw_prob_threshold
    cond_margin = winner_margin <= config.winner_margin_cap
    cond_v4_conf = v4_winner_conf <= config.v4_winner_conf_cap
    cond_elo = abs_elo_diff <= config.abs_elo_cap
    cond_tot_goals = tot_goals <= config.tot_expected_goals_cap
    cond_low_score = low_score_prob >= config.low_score_prob_floor

    do_override = bool(
        cond_draw_prob
        and cond_margin
        and cond_v4_conf
        and cond_elo
        and cond_tot_goals
        and cond_low_score
    )

    final_decision = "D" if do_override else v4_base_dec

    metadata = {
        "v4_winner_conf": round(v4_winner_conf, 4),
        "v42_draw_prob": round(v42_draw_prob, 4),
        "winner_margin": round(winner_margin, 4),
        "tot_expected_goals": round(tot_goals, 4),
        "abs_elo_diff": round(abs_elo_diff, 2),
        "low_score_prob": round(low_score_prob, 4),
        "cond_draw_prob": cond_draw_prob,
        "cond_margin": cond_margin,
        "cond_v4_conf": cond_v4_conf,
        "cond_elo": cond_elo,
        "cond_tot_goals": cond_tot_goals,
        "cond_low_score": cond_low_score,
    }

    return do_override, final_decision, metadata


def predict_v4_6_physical_gate(
    lambda_home: float,
    lambda_away: float,
    p_v4: List[float] | np.ndarray,
    p_v42: List[float] | np.ndarray,
    abs_elo_diff: float,
    low_score_prob: float = 0.25,
    config: Optional[V46PhysicalGateConfig] = None,
    fixture_id: Optional[int] = None,
) -> V46PhysicalGatePrediction:
    """Generate V4.6 physical gate draw prediction."""
    if config is None:
        config = V46PhysicalGateConfig.robust_optimal_gate()

    pv4 = np.asarray(p_v4, dtype=float)
    pv42 = np.asarray(p_v42, dtype=float)
    v4_dec = CLASS_ORDER[int(np.argmax(pv4))]

    do_override, final_dec, meta = evaluate_physical_draw_gate(
        p_v4=pv4,
        p_v42=pv42,
        abs_elo_diff=abs_elo_diff,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        low_score_prob=low_score_prob,
        config=config,
    )

    prob_dict = {
        "H": float(pv42[0]),
        "D": float(pv42[1]),
        "A": float(pv42[2]),
    }

    return V46PhysicalGatePrediction(
        fixture_id=fixture_id,
        probabilities=prob_dict,
        v4_base_decision=v4_dec,
        v4_6_final_decision=final_dec,
        override_applied=do_override,
        gate_name=config.name,
        metadata=meta,
    )
