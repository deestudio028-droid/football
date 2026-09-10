"""Production-Safe Draw Risk & Caution Advisory Component.

Provides non-mutating decision-support metrics and vulnerability intelligence
for V4.0 match predictions without altering underlying 1X2 probabilities or decisions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("draw_risk_advisor")

DEFAULT_WEIGHTS = {
    "calibrated_draw_prob": 0.35,
    "win_parity_margin": 0.30,
    "score_space_draw_mass": 0.20,
    "expected_goal_parity": 0.15,
}

DEFAULT_THRESHOLDS = {
    "medium": 0.40,
    "high": 0.65,
    "critical": 0.80,
}

DEFAULT_LABELS = {
    "LOW": "Low draw vulnerability",
    "MEDIUM": "Moderate draw vulnerability",
    "HIGH": "High draw vulnerability",
    "CRITICAL": "Critical draw vulnerability",
}

DEFAULT_BADGES = {
    "LOW": "🟢 LOW",
    "MEDIUM": "🟡 MEDIUM",
    "HIGH": "🟠 HIGH",
    "CRITICAL": "🔴 CRITICAL",
}


class DrawRiskAdvisor:
    """Production-safe component that calculates Draw Risk without mutating predictions."""

    def __init__(self, config_path: Optional[Path] = None):
        self.weights = DEFAULT_WEIGHTS.copy()
        self.thresholds = DEFAULT_THRESHOLDS.copy()

        if config_path and config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.weights = cfg.get("risk_weights", self.weights)
                    self.thresholds = cfg.get("tier_thresholds", self.thresholds)
                logger.info(f"Loaded DrawRiskAdvisor config from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}, using defaults: {e}")

    def compute_risk_score(
        self,
        cal_p_d: float,
        cal_win_diff: float,
        score_space_draw: float,
        lambda_gap: float,
    ) -> float:
        """Compute the continuous Draw Risk Score (0.0 to 1.0)."""
        w_d = self.weights.get("calibrated_draw_prob", 0.35)
        w_win = self.weights.get("win_parity_margin", 0.30)
        w_score = self.weights.get("score_space_draw_mass", 0.20)
        w_gap = self.weights.get("expected_goal_parity", 0.15)

        term_d = np.clip((cal_p_d - 0.22) / 0.12, 0.0, 1.0)
        term_win = 1.0 - np.clip(cal_win_diff / 0.20, 0.0, 1.0)
        term_score = np.clip((score_space_draw - 0.22) / 0.10, 0.0, 1.0)
        term_gap = 1.0 - np.clip(lambda_gap / 0.60, 0.0, 1.0)

        score = w_d * term_d + w_win * term_win + w_score * term_score + w_gap * term_gap
        return float(np.clip(score, 0.0, 1.0))

    def classify_tier(self, score: float) -> str:
        """Assign deterministic categorical tier from continuous score."""
        if score >= self.thresholds.get("critical", 0.80):
            return "CRITICAL"
        elif score >= self.thresholds.get("high", 0.65):
            return "HIGH"
        elif score >= self.thresholds.get("medium", 0.40):
            return "MEDIUM"
        else:
            return "LOW"

    def generate_reasons(
        self,
        tier: str,
        cal_p_d: float,
        cal_win_diff: float,
        score_space_draw: float,
        lambda_gap: float,
        tot_goals: float,
        elo_gap: float,
    ) -> List[str]:
        """Generate deterministic human-readable explanation bullet points."""
        reasons = []
        p_d_pct = cal_p_d * 100.0
        win_gap_pct = cal_win_diff * 100.0
        score_draw_pct = score_space_draw * 100.0

        if tier in ("HIGH", "CRITICAL"):
            if p_d_pct >= 28.0:
                reasons.append(f"Elevated calibrated draw probability ({p_d_pct:.1f}%)")
            if win_gap_pct <= 6.0:
                reasons.append(f"Near-parity Home/Away win margin (|H-A| = {win_gap_pct:.1f}%)")
            elif win_gap_pct <= 12.0:
                reasons.append(f"Closely matched Home/Away probabilities (|H-A| = {win_gap_pct:.1f}%)")
            if score_draw_pct >= 26.0:
                reasons.append(f"Supporting score-space draw mass ({score_draw_pct:.1f}%)")
            if tot_goals <= 2.40:
                reasons.append(f"Low expected-goal environment (xG = {tot_goals:.2f})")
            if elo_gap <= 50.0:
                reasons.append(f"Balanced team ratings (Elo gap = {elo_gap:.0f})")
        elif tier == "MEDIUM":
            reasons.append(f"Moderate draw probability ({p_d_pct:.1f}%)")
            reasons.append(f"Moderate favorite separation (|H-A| = {win_gap_pct:.1f}%)")
        else:
            reasons.append(f"Decisive favorite separation (|H-A| = {win_gap_pct:.1f}%)")
            reasons.append(f"Low baseline draw exposure ({p_d_pct:.1f}%)")

        if not reasons:
            reasons.append("Standard pre-match probability profile.")

        return reasons

    def evaluate_match_risk(
        self,
        cal_p_d: float,
        cal_win_diff: float,
        score_space_draw: float,
        lambda_gap: float,
        tot_goals: float = 2.50,
        elo_gap: float = 50.0,
    ) -> Dict[str, Any]:
        """Perform complete risk evaluation for a fixture."""
        score = self.compute_risk_score(cal_p_d, cal_win_diff, score_space_draw, lambda_gap)
        tier = self.classify_tier(score)
        label = DEFAULT_LABELS.get(tier, "Moderate draw vulnerability")
        badge = DEFAULT_BADGES.get(tier, "🟡 MEDIUM")
        reasons = self.generate_reasons(
            tier=tier,
            cal_p_d=cal_p_d,
            cal_win_diff=cal_win_diff,
            score_space_draw=score_space_draw,
            lambda_gap=lambda_gap,
            tot_goals=tot_goals,
            elo_gap=elo_gap,
        )
        is_vulnerable = bool(tier in ("HIGH", "CRITICAL"))

        return {
            "draw_risk_score": round(score, 4),
            "draw_risk_tier": tier,
            "draw_risk_label": label,
            "draw_risk_badge": badge,
            "draw_risk_reasons": reasons,
            "is_vulnerable": is_vulnerable,
        }
