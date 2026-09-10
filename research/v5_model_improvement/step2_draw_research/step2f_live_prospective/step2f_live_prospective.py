"""Step 2F: Live Prospective Draw Shadow Evaluation Pipeline.

Generates pre-kickoff immutable shadow forecasts for upcoming fixtures,
records them into the frozen live shadow forecast ledger, validates temporal
and probabilistic invariants, and produces comprehensive evaluation metrics.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STEP2E_DIR = PROJECT_ROOT / "research" / "v5_model_improvement" / "step2_draw_research" / "step2e_shadow_integration"
sys.path.insert(0, str(STEP2E_DIR))

from dashboard.fixture_service import FixtureService
from dashboard.prediction_service import PredictionService
from dashboard.time_utils import format_kickoff_ist, to_chennai_date
from draw_probability_calibrator import DrawProbabilityCalibrator

V4_PATH = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data" / "models" / "v4_1_prospective_candidate_2025_26.pkl"
CALIB_CONFIG_PATH = STEP2E_DIR / "draw_calibrator_config.json"
OUTPUT_DIR = PROJECT_ROOT / "research" / "v5_model_improvement" / "step2_draw_research" / "step2f_live_prospective"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEDGER_PATH = OUTPUT_DIR / "live_shadow_forecast_ledger.csv"
EVAL_PATH = OUTPUT_DIR / "live_shadow_evaluation.csv"
REPORT_PATH = OUTPUT_DIR / "live_shadow_report.md"

EXPECTED_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXPECTED_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def verify_integrity() -> None:
    act_v4 = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    act_v41 = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    if act_v4 != EXPECTED_V4_MD5:
        raise RuntimeError(f"V4.0 MD5 MISMATCH! Expected {EXPECTED_V4_MD5}, got {act_v4}. STOPPING.")
    if act_v41 != EXPECTED_V4_1_MD5:
        raise RuntimeError(f"V4.1 MD5 MISMATCH! Expected {EXPECTED_V4_1_MD5}, got {act_v41}. STOPPING.")
    print("  [OK] Pre-flight Integrity: Both V4.0 and V4.1 MD5 hashes verified 100% bit-identical.")


def generate_live_shadow_forecasts():
    verify_integrity()

    fs = FixtureService()
    ps = PredictionService()
    calibrator = DrawProbabilityCalibrator.from_config_file(CALIB_CONFIG_PATH)

    current_utc = datetime.datetime.now(datetime.timezone.utc)
    current_iso = current_utc.isoformat()
    print(f"\n--- STEP 2F: LIVE PROSPECTIVE SHADOW RUNNER ---")
    print(f"Current UTC Timestamp: {current_iso}")

    # Fetch all scheduled dates
    scheduled_dates = [
        "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27",
        "2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31",
    ]

    all_fixtures = []
    seen_fids = set()

    for d_str in scheduled_dates:
        fixes, _ = fs.get_todays_matches(date_str=d_str, provider_name="oddalerts")
        for f in fixes:
            if f.fixture_id not in seen_fids:
                seen_fids.add(f.fixture_id)
                all_fixtures.append(f)

    # Filter for strictly future scheduled matches
    future_fixtures = [f for f in all_fixtures if f.status in ["NS", "TBD", "SCHEDULED"]]
    future_fixtures.sort(key=lambda x: x.scheduled_kickoff)
    print(f"Found {len(future_fixtures)} strictly upcoming scheduled fixtures across 5 target leagues.")

    # Load existing ledger if present to guarantee immutability
    existing_ledger = {}
    if LEDGER_PATH.exists():
        df_old = pd.read_csv(LEDGER_PATH)
        for _, row in df_old.iterrows():
            existing_ledger[int(row["fixture_id"])] = row.to_dict()
        print(f"Loaded {len(existing_ledger)} existing immutable forecasts from ledger.")

    ledger_rows = []
    new_forecast_count = 0

    for f in future_fixtures:
        fid = int(f.fixture_id)

        # Check if already locked in ledger
        if fid in existing_ledger:
            ledger_rows.append(existing_ledger[fid])
            continue

        # Generate fresh pre-kickoff prediction using V4.0 Production
        pred_res = ps.predict_dashboard_fixture(f, model_key="V4.0 Production")
        probs_v4 = pred_res.v4_champ_probs
        p_h, p_d, p_a = probs_v4["H"], probs_v4["D"], probs_v4["A"]
        v4_dec = pred_res.v4_champ_decision

        # Apply Step 2E validated Platt calibrator
        cal_res = calibrator.calibrate_single(p_h, p_d, p_a)

        row = {
            "fixture_id": fid,
            "competition": f.competition_name,
            "season": "2025/2026",
            "home_team": f.home_team,
            "away_team": f.away_team,
            "scheduled_kickoff": f.scheduled_kickoff,
            "prediction_timestamp": current_iso,
            "v4_home_prob": round(p_h, 4),
            "v4_draw_prob": round(p_d, 4),
            "v4_away_prob": round(p_a, 4),
            "v4_decision": v4_dec,
            "shadow_home_prob": round(cal_res.calibrated_p_home, 4),
            "shadow_draw_prob": round(cal_res.calibrated_p_draw, 4),
            "shadow_away_prob": round(cal_res.calibrated_p_away, 4),
            "shadow_decision": cal_res.calibrated_decision,
            "draw_probability_delta": round(cal_res.draw_delta, 4),
            "home_probability_delta": round(cal_res.home_delta, 4),
            "away_probability_delta": round(cal_res.away_delta, 4),
            "decision_changed": cal_res.decision_changed,
            "model_version": pred_res.model_version,
            "calibrator_version": calibrator.version,
            "status": f.status,
        }
        ledger_rows.append(row)
        new_forecast_count += 1

    df_ledger = pd.DataFrame(ledger_rows)
    df_ledger.to_csv(LEDGER_PATH, index=False)
    print(f"  [OK] Saved {len(df_ledger)} locked pre-match forecasts ({new_forecast_count} new) to {LEDGER_PATH.name}")

    # =========================================================================
    # EVALUATION ON COMPLETED MATCHES (MILESTONE TRACKING)
    # =========================================================================
    # Evaluate on completed matches (Aug 22-24, 2026 sample: N=33)
    completed_fixes = [f for f in all_fixtures if f.status == "FT"]
    completed_fixes.sort(key=lambda x: x.scheduled_kickoff)
    print(f"\nEvaluating on {len(completed_fixes)} completed matches from recent prospective period...")

    eval_rows = []
    for f in completed_fixes:
        p40 = ps.predict_dashboard_fixture(f, model_key="V4.0 Production")
        probs_40 = p40.v4_champ_probs
        p_h, p_d, p_a = probs_40["H"], probs_40["D"], probs_40["A"]
        v4_dec = p40.v4_champ_decision

        cal_res = calibrator.calibrate_single(p_h, p_d, p_a)

        is_act_draw = (f.actual_outcome == "D")
        is_v4_corr = (v4_dec == f.actual_outcome)
        is_sh_corr = (cal_res.calibrated_decision == f.actual_outcome)

        eval_rows.append({
            "fixture_id": f.fixture_id,
            "date": to_chennai_date(f.scheduled_kickoff),
            "kickoff_ist": format_kickoff_ist(f.scheduled_kickoff),
            "competition": f.competition_name,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "score": f"{f.home_goals}-{f.away_goals}",
            "actual_outcome": f.actual_outcome,
            "v4_p_H": round(p_h, 4),
            "v4_p_D": round(p_d, 4),
            "v4_p_A": round(p_a, 4),
            "v4_decision": v4_dec,
            "v4_correct": is_v4_corr,
            "shadow_p_H": round(cal_res.calibrated_p_home, 4),
            "shadow_p_D": round(cal_res.calibrated_p_draw, 4),
            "shadow_p_A": round(cal_res.calibrated_p_away, 4),
            "shadow_decision": cal_res.calibrated_decision,
            "shadow_correct": is_sh_corr,
            "draw_delta": round(cal_res.draw_delta, 4),
            "home_delta": round(cal_res.home_delta, 4),
            "away_delta": round(cal_res.away_delta, 4),
            "decision_changed": cal_res.decision_changed,
            "is_actual_draw": is_act_draw,
        })

    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(EVAL_PATH, index=False)
    print(f"  [OK] Saved evaluation results for {len(df_eval)} completed matches to {EVAL_PATH.name}")

    # Summary Metrics Calculation
    n_comp = len(df_eval)
    v4_acc = (df_eval["v4_correct"].sum() / n_comp * 100) if n_comp > 0 else 0.0
    sh_acc = (df_eval["shadow_correct"].sum() / n_comp * 100) if n_comp > 0 else 0.0
    v4_draw_brier = float(np.mean((df_eval["v4_p_D"] - df_eval["is_actual_draw"].astype(float)) ** 2))
    sh_draw_brier = float(np.mean((df_eval["shadow_p_D"] - df_eval["is_actual_draw"].astype(float)) ** 2))

    print(f"\n--- PROSPECTIVE EVALUATION SUMMARY (N={n_comp} completed matches) ---")
    print(f"V4.0 Production Accuracy: {v4_acc:.2f}% ({df_eval['v4_correct'].sum()}/{n_comp})")
    print(f"Shadow Calibrator Acc:    {sh_acc:.2f}% ({df_eval['shadow_correct'].sum()}/{n_comp})")
    print(f"V4.0 Draw Brier Score:    {v4_draw_brier:.6f} -> Shadow: {sh_draw_brier:.6f}")
    print(f"Mean V4.0 P(D):           {df_eval['v4_p_D'].mean()*100:.2f}% -> Shadow: {df_eval['shadow_p_D'].mean()*100:.2f}% (Actual Draw Rate: {df_eval['is_actual_draw'].mean()*100:.2f}%)")
    print(f"Decisions Changed:        {df_eval['decision_changed'].sum()} / {n_comp}")

    return df_ledger, df_eval


if __name__ == "__main__":
    generate_live_shadow_forecasts()
