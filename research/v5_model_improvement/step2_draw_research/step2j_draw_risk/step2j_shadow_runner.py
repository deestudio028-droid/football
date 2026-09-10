"""Step 2J Isolated Shadow Draw Risk & Caution Intelligence Runner.

Executes standalone pre-match inference to produce:
1. V4.0 Production Prediction (HOME / AWAY) — 100% UNCHANGED.
2. V4.0 Raw & Calibrated Probabilities P(H), P(D), P(A).
3. Continuous Draw Risk Score (0.00 to 1.00).
4. Categorical Draw Risk Tier (LOW / MEDIUM / HIGH / CRITICAL).
5. Deterministic Human-Readable Advisory Reason.

ZERO PRODUCTION MUTATION — STRICTLY ISOLATED RESEARCH ARTIFACT.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys_paths = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"),
]
import sys
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from draw_probability_calibrator import DrawProbabilityCalibrator
from models.baselines import CLASS_ORDER
from models.draw_champion import (
    DrawChampionConfig,
    compute_dc_draw_probability,
    compute_elo_draw_probability,
    redistribute_proportional_odds,
    stable_logit,
    stable_sigmoid,
)
from models.poisson import predict_poisson
from models.v4_artifact import load_v4_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("step2j_shadow_runner")

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
FROZEN_CHAMPION_PATH = PROJECT_ROOT / "research/v4_promotion/draw_champion_method_frozen.json"
CALIBRATOR_CONFIG_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration/draw_calibrator_config.json"
STEP2J_CONFIG_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2j_draw_risk/step2j_candidate_config.json"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


@dataclass
class ShadowDrawRiskPrediction:
    fixture_id: int
    kickoff: str
    competition_name: str
    home_team: str
    away_team: str
    # 1. Unchanged V4.0 Production Prediction
    v4_decision: str
    v4_p_H: float
    v4_p_D: float
    v4_p_A: float
    # 2. Calibrated Continuous Probabilities
    cal_p_H: float
    cal_p_D: float
    cal_p_A: float
    # 3. Draw Risk Intelligence
    draw_risk_score: float
    draw_risk_tier: str
    draw_risk_reason: str
    is_vulnerable_fixture: bool


class Step2JDrawRiskEngine:
    def __init__(self):
        act_v4 = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
        assert act_v4 == EXP_V4_MD5, f"V4.0 MD5 mutated! Expected {EXP_V4_MD5}, got {act_v4}"
        self.v4 = load_v4_artifact(V4_PATH)
        self.cfg_champ = DrawChampionConfig.from_frozen_json(FROZEN_CHAMPION_PATH)
        self.calibrator = DrawProbabilityCalibrator.from_config_file(CALIBRATOR_CONFIG_PATH)

        with open(STEP2J_CONFIG_PATH, "r") as f:
            self.cfg_2j = json.load(f)

        self.thresholds = self.cfg_2j["tier_thresholds"]

    def compute_composite_risk_score(
        self,
        cal_p_d: float,
        cal_win_diff: float,
        score_space_draw: float,
        lambda_gap: float,
    ) -> float:
        score = (
            0.35 * np.clip((cal_p_d - 0.22) / 0.12, 0.0, 1.0)
            + 0.30 * (1.0 - np.clip(cal_win_diff / 0.20, 0.0, 1.0))
            + 0.20 * np.clip((score_space_draw - 0.22) / 0.10, 0.0, 1.0)
            + 0.15 * (1.0 - np.clip(lambda_gap / 0.60, 0.0, 1.0))
        )
        return float(np.clip(score, 0.0, 1.0))

    def assign_tier(self, score: float) -> str:
        if score >= self.thresholds["critical"]:
            return "CRITICAL"
        elif score >= self.thresholds["high"]:
            return "HIGH"
        elif score >= self.thresholds["medium"]:
            return "MEDIUM"
        else:
            return "LOW"

    def generate_reason(
        self,
        tier: str,
        cal_p_d: float,
        cal_win_diff: float,
        tot_goals: float,
        elo_gap: float,
        score_draw: float,
    ) -> str:
        p_d_pct = cal_p_d * 100.0
        win_gap_pct = cal_win_diff * 100.0
        score_draw_pct = score_draw * 100.0

        if tier == "CRITICAL":
            return f"Extreme draw risk: tight win parity (|H-A|={win_gap_pct:.1f}%), elevated draw prob ({p_d_pct:.1f}%), low goal environment (xG={tot_goals:.2f}), score mass={score_draw_pct:.1f}%."
        elif tier == "HIGH":
            return f"Elevated draw risk: close win margin (|H-A|={win_gap_pct:.1f}%), draw probability={p_d_pct:.1f}%, balanced team ratings (Elo gap={elo_gap:.0f})."
        elif tier == "MEDIUM":
            return f"Moderate contest: draw prob={p_d_pct:.1f}%, moderate win separation (|H-A|={win_gap_pct:.1f}%)."
        else:
            return f"Decisive win margin (|H-A|={win_gap_pct:.1f}%) favored by baseline model; low draw exposure ({p_d_pct:.1f}%)."

    def predict_match_risk(
        self,
        fixture_id: int,
        kickoff: str,
        competition_name: str,
        home_team: str,
        away_team: str,
        lh: float,
        la: float,
        abs_elo: float,
    ) -> ShadowDrawRiskPrediction:
        rho = self.cfg_champ.league_rhos.get(competition_name, self.cfg_champ.global_fallback_rho)

        # 1. Poisson probabilities
        pr_poiss = predict_poisson(np.array([lh]), np.array([la]), list(self.v4.class_order))[0]
        pv4 = np.array([pr_poiss.probabilities["H"], pr_poiss.probabilities["D"], pr_poiss.probabilities["A"]])

        # 2. Dixon-Coles & Elo draw stacking (V4.0 Production)
        p_dc = compute_dc_draw_probability(np.array([lh]), np.array([la]), np.array([rho]))[0]
        p_elo = compute_elo_draw_probability(np.array([pv4[1]]), np.array([abs_elo]), self.cfg_champ)[0]

        z = self.cfg_champ.stacking_intercept + self.cfg_champ.stacking_weight_dc * stable_logit(p_dc) + self.cfg_champ.stacking_weight_elo * stable_logit(p_elo)
        p_d_ch = stable_sigmoid(z)
        p_ch = redistribute_proportional_odds(pv4.reshape(1, 3), np.array([p_d_ch]))[0]
        v4_dec = CLASS_ORDER[int(np.argmax(p_ch))]

        # 3. Platt continuous calibration
        p_cal = self.calibrator.calibrate_array(p_ch.reshape(1, 3))[0]

        # 4. Score-space signals
        p_h = np.array([poisson.pmf(i, lh) for i in range(11)])
        p_a = np.array([poisson.pmf(j, la) for j in range(11)])
        mat = np.outer(p_h, p_a)
        if lh > 0 and la > 0:
            mat[0, 0] = max(1e-15, mat[0, 0] * (1.0 - lh * la * rho))
            mat[1, 0] = max(1e-15, mat[1, 0] * (1.0 + la * rho))
            mat[0, 1] = max(1e-15, mat[0, 1] * (1.0 + lh * rho))
            mat[1, 1] = max(1e-15, mat[1, 1] * (1.0 - rho))
        s = np.sum(mat)
        mat = mat / s if s > 0 else mat
        score_draw = sum(mat[k, k] for k in range(mat.shape[0]))

        # 5. Composite Risk Score & Tier
        win_diff = abs(p_cal[0] - p_cal[2])
        lambda_gap = abs(lh - la)
        tot_goals = lh + la

        risk_score = self.compute_composite_risk_score(
            cal_p_d=float(p_cal[1]),
            cal_win_diff=float(win_diff),
            score_space_draw=float(score_draw),
            lambda_gap=float(lambda_gap),
        )
        tier = self.assign_tier(risk_score)
        reason = self.generate_reason(
            tier=tier,
            cal_p_d=float(p_cal[1]),
            cal_win_diff=float(win_diff),
            tot_goals=float(tot_goals),
            elo_gap=float(abs_elo),
            score_draw=float(score_draw),
        )
        is_vulnerable = bool(tier in ("HIGH", "CRITICAL"))

        return ShadowDrawRiskPrediction(
            fixture_id=fixture_id,
            kickoff=kickoff,
            competition_name=competition_name,
            home_team=home_team,
            away_team=away_team,
            v4_decision=v4_dec,  # 100% UNCHANGED V4.0 DECISION
            v4_p_H=round(float(p_ch[0]), 4),
            v4_p_D=round(float(p_ch[1]), 4),
            v4_p_A=round(float(p_ch[2]), 4),
            cal_p_H=round(float(p_cal[0]), 4),
            cal_p_D=round(float(p_cal[1]), 4),
            cal_p_A=round(float(p_cal[2]), 4),
            draw_risk_score=round(risk_score, 4),
            draw_risk_tier=tier,
            draw_risk_reason=reason,
            is_vulnerable_fixture=is_vulnerable,
        )


if __name__ == "__main__":
    engine = Step2JDrawRiskEngine()
    # Test 1: Fulham vs Chelsea (Competitive Derby)
    p1 = engine.predict_match_risk(
        fixture_id=1001,
        kickoff="2026-08-30T14:00:00+00:00",
        competition_name="Premier League",
        home_team="Fulham",
        away_team="Chelsea",
        lh=1.35,
        la=1.40,
        abs_elo=40.0,
    )
    print("Example 1 (Fulham vs Chelsea):")
    print(json.dumps(asdict(p1), indent=2))

    # Test 2: Man City vs Luton (Clear Favorite)
    p2 = engine.predict_match_risk(
        fixture_id=1002,
        kickoff="2026-08-30T16:30:00+00:00",
        competition_name="Premier League",
        home_team="Manchester City",
        away_team="Luton Town",
        lh=2.70,
        la=0.55,
        abs_elo=195.0,
    )
    print("\nExample 2 (Man City vs Luton):")
    print(json.dumps(asdict(p2), indent=2))
