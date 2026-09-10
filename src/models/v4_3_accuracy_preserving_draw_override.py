"""V4.3 Accuracy-Preserving Selective Draw Override Model.

Architecture:
- Base Layer: Frozen V4 Poisson Venue Elo Online A/D Model (default decision = argmax(P_v4)).
- Secondary Evidence: V4.2 DIBP + Calibrated Stacking probability layer.
- Selective Override Decision Layer: Evaluates positive-value Draw evidence to selectively
  override fragile, uncertain H/A predictions to Draw WITHOUT reducing overall accuracy below the 52.19% floor.

Invariants:
1. Production Isolation: Does not modify v4_draw_champion, v4_2_draw_resolution_candidate, or frozen artifacts.
2. Simplex Conservation: Underlying probability distribution remains normalized on the 3-simplex.
3. Selective Gate: Override only occurs when Draw probability exceeds confidence cap, team balance,
   margin constraints, and low-score expectancy.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CLASS_ORDER = ["H", "D", "A"]


@dataclass(frozen=True)
class V43DrawOverrideConfig:
    """Configuration for V4.3 Accuracy-Preserving Selective Draw Override."""
    name: str = "v4_3_accuracy_maximizer"
    version: str = "v4.3-accuracy-preserving-draw-override"
    draw_prob_threshold: float = 0.2600
    winner_margin_cap: float = 0.1000
    v4_winner_conf_cap: float = 0.4500
    abs_elo_cap: float = 100.0
    tot_expected_goals_cap: float = 2.5000
    stacking_intercept: float = 0.2450
    dibp_inflation_p: float = 0.0500
    use_dibp: bool = True

    @classmethod
    def accuracy_maximizer(cls) -> V43DrawOverrideConfig:
        """Top accuracy configuration (52.50% accuracy, +4 net wins vs V4, 77 draw preds)."""
        return cls(
            name="accuracy_maximizer",
            draw_prob_threshold=0.2600,
            winner_margin_cap=0.1000,
            v4_winner_conf_cap=0.4500,
            abs_elo_cap=100.0,
            tot_expected_goals_cap=2.5000,
        )

    @classmethod
    def balanced_draw_preserver(cls) -> V43DrawOverrideConfig:
        """Balanced configuration (52.19% baseline accuracy, 103 draw preds, 33 correct draws, 32.0% prec)."""
        return cls(
            name="balanced_draw_preserver",
            draw_prob_threshold=0.2600,
            winner_margin_cap=0.1000,
            v4_winner_conf_cap=0.4500,
            abs_elo_cap=150.0,
            tot_expected_goals_cap=2.5000,
        )

    @classmethod
    def high_precision_conservative(cls) -> V43DrawOverrideConfig:
        """Conservative configuration (52.50% accuracy, 66 draw preds, 21 correct draws, 31.8% prec)."""
        return cls(
            name="high_precision_conservative",
            draw_prob_threshold=0.2600,
            winner_margin_cap=0.1000,
            v4_winner_conf_cap=0.4500,
            abs_elo_cap=80.0,
            tot_expected_goals_cap=2.5000,
        )


@dataclass
class V43OverridePrediction:
    """Output container for V4.3 prediction."""
    fixture_id: Optional[int]
    probabilities: Dict[str, float]
    v4_base_decision: str
    v4_3_final_decision: str
    override_applied: bool
    override_rule_name: str
    draw_evidence_score: float
    metadata: Dict[str, Any]


def evaluate_selective_override(
    p_v4: List[float] | np.ndarray,
    p_v42: List[float] | np.ndarray,
    abs_elo_diff: float,
    lambda_home: float,
    lambda_away: float,
    config: V43DrawOverrideConfig,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Determine whether to apply a selective Draw override on top of V4."""
    pv4 = np.asarray(p_v4, dtype=float)
    pv42 = np.asarray(p_v42, dtype=float)

    # V4 base decision
    v4_base_dec = CLASS_ORDER[int(np.argmax(pv4))]
    v4_winner_conf = float(max(pv4[0], pv4[2]))

    # V4.2 candidate metrics
    v42_winner_conf = float(max(pv42[0], pv42[2]))
    v42_draw_prob = float(pv42[1])
    winner_margin = float(v42_winner_conf - v42_draw_prob)

    # Match intensity metrics
    tot_goals = float(lambda_home + lambda_away)
    goal_diff = float(abs(lambda_home - lambda_away))

    # Evaluate conditions
    cond_draw_prob = v42_draw_prob >= config.draw_prob_threshold
    cond_margin = winner_margin <= config.winner_margin_cap
    cond_v4_conf = v4_winner_conf <= config.v4_winner_conf_cap
    cond_elo = abs_elo_diff <= config.abs_elo_cap
    cond_tot_goals = tot_goals <= config.tot_expected_goals_cap

    do_override = bool(
        cond_draw_prob
        and cond_margin
        and cond_v4_conf
        and cond_elo
        and cond_tot_goals
    )

    final_decision = "D" if do_override else v4_base_dec

    metadata = {
        "v4_winner_conf": round(v4_winner_conf, 4),
        "v42_draw_prob": round(v42_draw_prob, 4),
        "winner_margin": round(winner_margin, 4),
        "tot_expected_goals": round(tot_goals, 4),
        "abs_elo_diff": round(abs_elo_diff, 2),
        "cond_draw_prob": cond_draw_prob,
        "cond_margin": cond_margin,
        "cond_v4_conf": cond_v4_conf,
        "cond_elo": cond_elo,
        "cond_tot_goals": cond_tot_goals,
    }

    return do_override, final_decision, metadata


def predict_v4_3_selective_override(
    lambda_home: float,
    lambda_away: float,
    p_v4: List[float] | np.ndarray,
    p_v42: List[float] | np.ndarray,
    abs_elo_diff: float,
    config: Optional[V43DrawOverrideConfig] = None,
    fixture_id: Optional[int] = None,
) -> V43OverridePrediction:
    """Generate V4.3 selective draw override prediction."""
    if config is None:
        config = V43DrawOverrideConfig.accuracy_maximizer()

    pv42 = np.asarray(p_v42, dtype=float)
    pv4 = np.asarray(p_v4, dtype=float)
    v4_dec = CLASS_ORDER[int(np.argmax(pv4))]

    do_override, final_dec, meta = evaluate_selective_override(
        p_v4=pv4,
        p_v42=pv42,
        abs_elo_diff=abs_elo_diff,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        config=config,
    )

    prob_dict = {
        "H": float(pv42[0]),
        "D": float(pv42[1]),
        "A": float(pv42[2]),
    }

    return V43OverridePrediction(
        fixture_id=fixture_id,
        probabilities=prob_dict,
        v4_base_decision=v4_dec,
        v4_3_final_decision=final_dec,
        override_applied=do_override,
        override_rule_name=config.name,
        draw_evidence_score=float(pv42[1]),
        metadata=meta,
    )
