"""Step 2I Isolated Shadow Prediction & Caution Layer Runner.

Generates isolated comparative predictions without modifying production.
Evaluates V4.0 baseline, Platt calibrated probabilities, draw confidence scores,
and flags high-uncertainty Draw Caution Zone fixtures.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys_paths = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from draw_probability_calibrator import DrawProbabilityCalibrator
from features.elo import ELO_COLUMNS, load_elo_features
from features.online_attack_defense import AD_COLUMNS, compute_ad_states, fit_baseline_rates
from models.baselines import CLASS_ORDER
from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
from models.data import load_supervised_dataset
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
logger = logging.getLogger("step2i_shadow_runner")

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
FROZEN_CHAMPION_PATH = PROJECT_ROOT / "research/v4_promotion/draw_champion_method_frozen.json"
CALIBRATOR_CONFIG_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration/draw_calibrator_config.json"
STEP2I_CONFIG_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2i_draw_decision/step2i_candidate_config.json"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


@dataclass
class ShadowDecisionPrediction:
    fixture_id: int
    kickoff: str
    league: str
    home_team: str
    away_team: str
    # V4.0 Baseline
    v4_p_H: float
    v4_p_D: float
    v4_p_A: float
    v4_decision: str
    # Calibrated Shadow Candidate
    cal_p_H: float
    cal_p_D: float
    cal_p_A: float
    cal_decision: str
    # Decision Intelligence & Advisory
    draw_score: float
    draw_confidence: str
    decision_reason: str
    is_caution_zone: bool
    advisory_status: str


class Step2IShadowEngine:
    def __init__(self):
        act_v4 = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
        assert act_v4 == EXP_V4_MD5, f"V4.0 MD5 mutated! Expected {EXP_V4_MD5}, got {act_v4}"
        self.v4 = load_v4_artifact(V4_PATH)
        self.cfg_champ = DrawChampionConfig.from_frozen_json(FROZEN_CHAMPION_PATH)
        self.calibrator = DrawProbabilityCalibrator.from_config_file(CALIBRATOR_CONFIG_PATH)

        with open(STEP2I_CONFIG_PATH, "r") as f:
            self.cfg_2i = json.load(f)

        self.fcols = self.cfg_2i["draw_confidence_features"]
        self.intercept = self.cfg_2i["draw_confidence_intercept"]
        self.coefs = self.cfg_2i["draw_confidence_coefficients"]

    def compute_draw_confidence_score(self, feat_dict: Dict[str, float]) -> float:
        z = self.intercept
        for col in self.fcols:
            z += self.coefs.get(col, 0.0) * feat_dict.get(col, 0.0)
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0))))

    def generate_shadow_prediction(
        self,
        fixture_id: int,
        kickoff: str,
        league: str,
        home_team: str,
        away_team: str,
        lh: float,
        la: float,
        abs_elo: float,
        X_df: pd.DataFrame,
    ) -> ShadowDecisionPrediction:
        rho = self.cfg_champ.league_rhos.get(league, self.cfg_champ.global_fallback_rho)

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
        cal_dec = CLASS_ORDER[int(np.argmax(p_cal))]

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

        p_00 = mat[0, 0]
        p_11 = mat[1, 1]
        p_low = mat[0, 0] + mat[1, 0] + mat[0, 1] + mat[1, 1] + mat[2, 0] + mat[0, 2]
        p_score_draw = sum(mat[k, k] for k in range(mat.shape[0]))

        # 5. Draw Confidence Score & Caution Flag
        feat_dict = {
            "cal_p_D": float(p_cal[1]),
            "cal_win_diff": float(abs(p_cal[0] - p_cal[2])),
            "cal_win_to_draw_margin": float(max(p_cal[0], p_cal[2]) - p_cal[1]),
            "lambda_total": float(lh + la),
            "lambda_gap": float(abs(lh - la)),
            "abs_elo_diff": float(abs_elo),
            "p_score_space_draw": float(p_score_draw),
            "p_00": float(p_00),
            "p_11": float(p_11),
            "p_low_score_mass": float(p_low),
            "ad_net_power_gap": float(abs((X_df.get("A_home", 0.0) - X_df.get("D_away", 0.0)) - (X_df.get("A_away", 0.0) - X_df.get("D_home", 0.0)))),
        }
        draw_score = self.compute_draw_confidence_score(feat_dict)

        # Caution zone condition: Elevated Draw + Close Win Margin
        is_caution = bool(p_cal[1] >= 0.28 and abs(p_cal[0] - p_cal[2]) <= 0.08)

        if is_caution:
            conf_str = "HIGH_UNCERTAINTY"
            reason = f"Draw probability elevated ({p_cal[1]*100:.1f}%) and win parity tight (|H-A|={abs(p_cal[0]-p_cal[2])*100:.1f}%)."
            adv_status = "CAUTION_DRAW_ZONE"
        elif p_cal[1] >= 0.30:
            conf_str = "ELEVATED_DRAW_POTENTIAL"
            reason = f"Draw probability elevated ({p_cal[1]*100:.1f}%) with favorable low-score profile."
            adv_status = "ELEVATED_DRAW"
        else:
            conf_str = "STANDARD_DECISIVE"
            reason = f"Decisive win margin favored by baseline Poisson + DC model."
            adv_status = "STANDARD_WIN"

        return ShadowDecisionPrediction(
            fixture_id=fixture_id,
            kickoff=kickoff,
            league=league,
            home_team=home_team,
            away_team=away_team,
            v4_p_H=round(float(p_ch[0]), 4),
            v4_p_D=round(float(p_ch[1]), 4),
            v4_p_A=round(float(p_ch[2]), 4),
            v4_decision=v4_dec,
            cal_p_H=round(float(p_cal[0]), 4),
            cal_p_D=round(float(p_cal[1]), 4),
            cal_p_A=round(float(p_cal[2]), 4),
            cal_decision=cal_dec,
            draw_score=round(draw_score, 4),
            draw_confidence=conf_str,
            decision_reason=reason,
            is_caution_zone=is_caution,
            advisory_status=adv_status,
        )


if __name__ == "__main__":
    engine = Step2IShadowEngine()
    pred = engine.generate_shadow_prediction(
        fixture_id=420582254,
        kickoff="2026-08-24T18:45:00+00:00",
        league="Serie A",
        home_team="Udinese",
        away_team="Como",
        lh=1.25,
        la=1.20,
        abs_elo=35.0,
        X_df={"A_home": 1.05, "D_home": 1.02, "A_away": 0.98, "D_away": 1.04},
    )
    print("Sample Shadow Decision Prediction:")
    print(json.dumps(asdict(pred), indent=2))
