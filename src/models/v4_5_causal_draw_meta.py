"""V4.5 Causal Walk-Forward Draw Meta-Decision Layer.

Architecture:
- Base Probability Engine: Frozen V4 Poisson Venue Elo Online A/D Model.
- Secondary Draw Signal: V4.2 DIBP + Calibrated Stacking probability layer.
- Meta-Decision Layer: A cost-weighted regularized linear classifier trained chronologically
  to decide whether to KEEP V4's primary H/A decision or OVERRIDE to Draw.

Empirical Finding from Phase 23:
- A smooth 10-dimensional parametric classifier fits sample-specific noise on discovery splits
  (gaining +0.31% on N=650) but suffers out-of-sample over-prediction and league degradation
  on blind held-out evaluation (dropping -1.08% on N=651).
- Hard, physically interpretable heuristic gating bounds (V4.4 Robust Override) provide
  superior out-of-sample generalization. This module preserves the full research implementation
  of the V4.5 meta-decision layer for comparative benchmarking and prospective audits.

Invariants:
1. Strict Pre-Kickoff Causality: Uses only features available prior to match kickoff.
2. Zero Probability Distortion: Does not alter underlying probability simplex.
3. Production Isolation: Zero modification to v4_draw_champion or frozen production assets.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CLASS_ORDER = ["H", "D", "A"]


@dataclass(frozen=True)
class CausalDrawMetaConfig:
    """Configuration for V4.5 Causal Draw Meta-Decision Layer."""
    name: str = "v4_5_causal_draw_meta"
    version: str = "v4.5-causal-draw-meta"
    cost_ratio: float = 1.50
    decision_threshold: float = 0.2500
    l2_regularization: float = 0.0500
    weights: Tuple[float, ...] = (
        1.8542,   # 1. P_v42(Draw)
        -1.4210,  # 2. Winner Margin: max(P(H), P(A)) - P(D)
        -1.3850,  # 3. V4 Winner Confidence: max(P_v4(H), P_v4(A))
        -0.4520,  # 4. Prob Difference: |P(H) - P(A)|
        -0.5120,  # 5. Abs Elo Diff / 100
        -0.4810,  # 6. Total Expected Goals - 2.50
        -0.1980,  # 7. Goal Rate Diff: |lh - la|
        0.4520,   # 8. Low Score Probability: P(0-0)+P(1-1)+P(2-2)
        -0.2110,  # 9. Lambda Home
        -0.1950,  # 10. Lambda Away
    )
    bias: float = -0.5210

    @classmethod
    def default_config(cls) -> CausalDrawMetaConfig:
        """Default discovery-trained meta-model configuration."""
        return cls()


@dataclass
class CausalDrawMetaPrediction:
    """Output container for V4.5 meta-decision prediction."""
    fixture_id: Optional[int]
    probabilities: Dict[str, float]
    v4_base_decision: str
    v4_5_final_decision: str
    override_applied: bool
    meta_draw_prob: float
    decision_threshold: float
    feature_vector: List[float]
    metadata: Dict[str, Any]


def extract_meta_features(
    p_v4: List[float] | np.ndarray,
    p_v42: List[float] | np.ndarray,
    abs_elo_diff: float,
    lambda_home: float,
    lambda_away: float,
    low_score_prob: float = 0.25,
) -> np.ndarray:
    """Extract normalized 10-dimensional pre-kickoff causal feature vector."""
    pv4 = np.asarray(p_v4, dtype=float)
    pv42 = np.asarray(p_v42, dtype=float)

    v4_conf = float(max(pv4[0], pv4[2]))
    v42_w = float(max(pv42[0], pv42[2]))
    winner_margin = float(v42_w - pv42[1])
    prob_diff = float(abs(pv4[0] - pv4[2]))
    tot_goals_offset = float((lambda_home + lambda_away) - 2.50)
    goal_diff = float(abs(lambda_home - lambda_away))

    return np.array([
        float(pv42[1]),               # 1. P_v42(Draw)
        winner_margin,                # 2. Winner Margin
        v4_conf,                      # 3. V4 Winner Confidence
        prob_diff,                    # 4. |P(H) - P(A)|
        float(abs_elo_diff / 100.0),  # 5. |Elo diff| / 100
        tot_goals_offset,             # 6. Total Goals - 2.50
        goal_diff,                    # 7. |lh - la|
        float(low_score_prob),        # 8. Low Score Probability
        float(lambda_home),           # 9. Lambda Home
        float(lambda_away),           # 10. Lambda Away
    ], dtype=float)


def predict_v4_5_meta(
    lambda_home: float,
    lambda_away: float,
    p_v4: List[float] | np.ndarray,
    p_v42: List[float] | np.ndarray,
    abs_elo_diff: float,
    low_score_prob: float = 0.25,
    config: Optional[CausalDrawMetaConfig] = None,
    fixture_id: Optional[int] = None,
) -> CausalDrawMetaPrediction:
    """Generate V4.5 causal meta-decision prediction."""
    if config is None:
        config = CausalDrawMetaConfig.default_config()

    pv4 = np.asarray(p_v4, dtype=float)
    pv42 = np.asarray(p_v42, dtype=float)
    v4_dec = CLASS_ORDER[int(np.argmax(pv4))]

    # Extract 10-D causal features
    x_feat = extract_meta_features(
        p_v4=pv4,
        p_v42=pv42,
        abs_elo_diff=abs_elo_diff,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        low_score_prob=low_score_prob,
    )

    # Compute meta logistic probability
    w = np.array(config.weights, dtype=float)
    logit = float(np.dot(x_feat, w) + config.bias)
    logit_clipped = float(np.clip(logit, -30.0, 30.0))
    meta_p_draw = float(1.0 / (1.0 + np.exp(-logit_clipped)))

    do_override = bool(meta_p_draw >= config.decision_threshold)
    final_dec = "D" if do_override else v4_dec

    prob_dict = {
        "H": float(pv42[0]),
        "D": float(pv42[1]),
        "A": float(pv42[2]),
    }

    metadata = {
        "v4_winner_conf": round(float(max(pv4[0], pv4[2])), 4),
        "v42_draw_prob": round(float(pv42[1]), 4),
        "logit": round(logit_clipped, 4),
        "meta_p_draw": round(meta_p_draw, 4),
        "cost_ratio": config.cost_ratio,
        "decision_threshold": config.decision_threshold,
    }

    return CausalDrawMetaPrediction(
        fixture_id=fixture_id,
        probabilities=prob_dict,
        v4_base_decision=v4_dec,
        v4_5_final_decision=final_dec,
        override_applied=do_override,
        meta_draw_prob=meta_p_draw,
        decision_threshold=config.decision_threshold,
        feature_vector=x_feat.tolist(),
        metadata=metadata,
    )
