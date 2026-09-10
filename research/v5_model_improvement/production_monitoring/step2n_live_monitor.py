"""Step 2N — Live Production Outcome Monitoring & Prospective Validation Engine.

Monitors completed outcomes against locked pre-kickoff prospective forecasts,
evaluating V4.0 Production accuracy and Draw Risk Advisory metrics with zero leakage.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("step2n_live_monitor")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys_paths = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2k_advisory_integration"),
    str(PROJECT_ROOT / "research/v5_model_improvement/production_monitoring"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from dashboard.draw_risk_advisor import DrawRiskAdvisor

MONITORING_DIR = PROJECT_ROOT / "research/v5_model_improvement/production_monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
STEP2F_LEDGER_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2f_live_prospective/live_shadow_forecast_ledger.csv"
STEP2E_33_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration/shadow_33_match_audit.csv"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def verify_md5(path: Path, expected: str, name: str) -> bool:
    act = hashlib.md5(path.read_bytes()).hexdigest()
    if act != expected:
        raise RuntimeError(f"{name} MD5 violation! Expected {expected}, got {act}")
    return True


def multiclass_brier_score(y_true: List[str], probs: np.ndarray, classes: List[str] = ["H", "D", "A"]) -> float:
    """Calculate multiclass Brier score sum_k (p_ik - y_ik)^2 / N."""
    brier_sum = 0.0
    for i, actual in enumerate(y_true):
        one_hot = np.array([1.0 if c == actual else 0.0 for c in classes])
        brier_sum += np.sum((probs[i] - one_hot) ** 2)
    return float(brier_sum / len(y_true)) if len(y_true) > 0 else 0.0


def multiclass_log_loss(y_true: List[str], probs: np.ndarray, classes: List[str] = ["H", "D", "A"]) -> float:
    """Calculate multiclass categorical cross entropy log loss."""
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    # Re-normalize
    probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
    loss_sum = 0.0
    for i, actual in enumerate(y_true):
        idx = classes.index(actual)
        loss_sum -= np.log(probs_clipped[i, idx])
    return float(loss_sum / len(y_true)) if len(y_true) > 0 else 0.0


def run_live_outcome_monitoring():
    logger.info("================================================================")
    logger.info("STEP 2N — LIVE PRODUCTION OUTCOME MONITORING & VALIDATION")
    logger.info("================================================================")

    # 1. Pre-flight Hash Check
    assert verify_md5(V4_PATH, EXP_V4_MD5, "V4.0 Baseline")
    assert verify_md5(V4_1_PATH, EXP_V4_1_MD5, "V4.1 Production Candidate")

    advisor = DrawRiskAdvisor()

    # 2. Load Completed Prospective Cohort (Aug 22–24, 2026, N=33)
    df_33_raw = pd.read_csv(STEP2E_33_PATH)
    logger.info(f"Loaded {len(df_33_raw)} completed prospective matches from Aug 22–24, 2026.")

    # 3. Load Future Locked Prospective Cohort (Step 2F Ledger, N=54)
    df_54_raw = pd.read_csv(STEP2F_LEDGER_PATH)
    logger.info(f"Loaded {len(df_54_raw)} upcoming locked prospective matches from Step 2F ledger.")

    # 4. Build Full Fixture Outcome Audit (Completed + Pending = 87 matches)
    audit_rows = []

    # Process Completed Matches (N=33)
    for _, row in df_33_raw.iterrows():
        fid = int(row["fixture_id"])
        h_name = str(row["home_team"])
        a_name = str(row["away_team"])
        league = str(row["league"])
        act_res = str(row["actual_outcome"])
        score = str(row["score"])
        hg, ag = [int(s) for s in score.split("-")] if "-" in score else (None, None)

        v4_h = float(row["v4_p_H"])
        v4_d = float(row["v4_p_D"])
        v4_a = float(row["v4_p_A"])
        cal_h = float(row["calibrated_p_H"])
        cal_d = float(row["calibrated_p_D"])
        cal_a = float(row["calibrated_p_A"])

        win_diff = abs(cal_h - cal_a)
        approx_score_draw = 0.28 if cal_d >= 0.28 else 0.24
        approx_lambda_gap = 0.20 if win_diff <= 0.06 else 0.50

        risk_res = advisor.evaluate_match_risk(
            cal_p_d=cal_d,
            cal_win_diff=win_diff,
            score_space_draw=approx_score_draw,
            lambda_gap=approx_lambda_gap,
            tot_goals=2.50,
            elo_gap=40.0,
        )

        v4_dec = str(row["v4_decision"])
        v4_corr = (v4_dec == act_res)
        is_draw = (act_res == "D")
        is_draw_flagged = bool(risk_res["draw_risk_tier"] in ("MEDIUM", "HIGH", "CRITICAL"))

        audit_rows.append({
            "fixture_id": fid,
            "competition": league,
            "season": "2025/2026",
            "home_team": h_name,
            "away_team": a_name,
            "scheduled_kickoff": str(row.get("kickoff_ist", "2026-08-22")),
            "prediction_timestamp": "2026-08-22T00:00:00Z",
            "status": "FT",
            "v4_p_home": v4_h,
            "v4_p_draw": v4_d,
            "v4_p_away": v4_a,
            "v4_decision": v4_dec,
            "draw_risk_score": risk_res["draw_risk_score"],
            "draw_risk_tier": risk_res["draw_risk_tier"],
            "draw_risk_label": risk_res["draw_risk_label"],
            "actual_home_goals": hg,
            "actual_away_goals": ag,
            "actual_result": act_res,
            "v4_correct": v4_corr,
            "draw_occurred": is_draw,
            "draw_risk_detected": is_draw_flagged,
            "is_pending": False,
        })

    # Process Pending Upcoming Matches (N=54)
    for _, row in df_54_raw.iterrows():
        fid = int(row["fixture_id"])
        h_name = str(row["home_team"])
        a_name = str(row["away_team"])
        league = str(row["competition"])
        kickoff = str(row["scheduled_kickoff"])
        pred_time = str(row["prediction_timestamp"])

        v4_h = float(row["v4_home_prob"])
        v4_d = float(row["v4_draw_prob"])
        v4_a = float(row["v4_away_prob"])
        cal_d = float(row["shadow_draw_prob"])
        cal_h = float(row["shadow_home_prob"])
        cal_a = float(row["shadow_away_prob"])

        win_diff = abs(cal_h - cal_a)
        approx_score_draw = 0.28 if cal_d >= 0.28 else 0.24
        approx_lambda_gap = 0.20 if win_diff <= 0.06 else 0.50

        risk_res = advisor.evaluate_match_risk(
            cal_p_d=cal_d,
            cal_win_diff=win_diff,
            score_space_draw=approx_score_draw,
            lambda_gap=approx_lambda_gap,
            tot_goals=2.50,
            elo_gap=40.0,
        )

        v4_dec = str(row["v4_decision"])

        audit_rows.append({
            "fixture_id": fid,
            "competition": league,
            "season": "2025/2026",
            "home_team": h_name,
            "away_team": a_name,
            "scheduled_kickoff": kickoff,
            "prediction_timestamp": pred_time,
            "status": "NS",
            "v4_p_home": v4_h,
            "v4_p_draw": v4_d,
            "v4_p_away": v4_a,
            "v4_decision": v4_dec,
            "draw_risk_score": risk_res["draw_risk_score"],
            "draw_risk_tier": risk_res["draw_risk_tier"],
            "draw_risk_label": risk_res["draw_risk_label"],
            "actual_home_goals": None,
            "actual_away_goals": None,
            "actual_result": None,
            "v4_correct": None,
            "draw_occurred": None,
            "draw_risk_detected": None,
            "is_pending": True,
        })

    df_full_audit = pd.DataFrame(audit_rows)
    df_full_audit.to_csv(MONITORING_DIR / "fixture_outcome_audit.csv", index=False)
    logger.info(f"Saved {len(df_full_audit)} matches to fixture_outcome_audit.csv ({len(df_33_raw)} completed, {len(df_54_raw)} pending).")

    # 5. Live Completed Outcome Ledger (N=33 completed matches)
    df_completed = df_full_audit[df_full_audit["status"] == "FT"].copy().reset_index(drop=True)
    df_completed.to_csv(MONITORING_DIR / "live_outcome_ledger.csv", index=False)
    logger.info("Saved completed outcome ledger to live_outcome_ledger.csv")

    # 6. Detailed V4.0 Performance Metrics
    n_comp = len(df_completed)
    n_corr = df_completed["v4_correct"].sum()
    overall_acc = (n_corr / n_comp * 100.0)

    # Class-specific accuracies
    h_sub = df_completed[df_completed["actual_result"] == "H"]
    d_sub = df_completed[df_completed["actual_result"] == "D"]
    a_sub = df_completed[df_completed["actual_result"] == "A"]

    h_acc = (h_sub["v4_decision"] == "H").mean() * 100.0 if len(h_sub) > 0 else 0.0
    d_acc = (d_sub["v4_decision"] == "D").mean() * 100.0 if len(d_sub) > 0 else 0.0
    a_acc = (a_sub["v4_decision"] == "A").mean() * 100.0 if len(a_sub) > 0 else 0.0

    probs_arr = df_completed[["v4_p_home", "v4_p_draw", "v4_p_away"]].values
    y_actual = df_completed["actual_result"].tolist()

    brier = multiclass_brier_score(y_actual, probs_arr)
    ll = multiclass_log_loss(y_actual, probs_arr)

    # 95% Wilson Score Confidence Interval for Accuracy
    p_hat = n_corr / n_comp
    z = 1.96
    ci_lower = (p_hat + z**2 / (2 * n_comp) - z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_comp)) / n_comp)) / (1 + z**2 / n_comp) * 100.0
    ci_upper = (p_hat + z**2 / (2 * n_comp) + z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_comp)) / n_comp)) / (1 + z**2 / n_comp) * 100.0

    summary_df = pd.DataFrame([{
        "total_locked_forecasts": len(df_full_audit),
        "completed_matches": n_comp,
        "pending_matches": len(df_full_audit) - n_comp,
        "cancelled_postponed": 0,
        "v4_0_correct": int(n_corr),
        "v4_0_accuracy_pct": round(overall_acc, 2),
        "accuracy_95ci_lower": round(ci_lower, 2),
        "accuracy_95ci_upper": round(ci_upper, 2),
        "home_accuracy_pct": round(h_acc, 2),
        "draw_accuracy_pct": round(d_acc, 2),
        "away_accuracy_pct": round(a_acc, 2),
        "multiclass_brier_score": round(brier, 4),
        "multiclass_log_loss": round(ll, 4),
    }])
    summary_df.to_csv(MONITORING_DIR / "live_performance_summary.csv", index=False)
    logger.info(f"Saved performance summary:\n{summary_df.to_string(index=False)}")

    # 7. Draw Risk Outcome Analysis
    risk_rows = []
    for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        sub_t = df_completed[df_completed["draw_risk_tier"] == tier]
        n_t = len(sub_t)
        n_draws_t = (sub_t["actual_result"] == "D").sum()
        draw_rate_t = (n_draws_t / n_t * 100.0) if n_t > 0 else 0.0
        acc_t = (sub_t["v4_correct"].mean() * 100.0) if n_t > 0 else 0.0
        fail_t = (100.0 - acc_t) if n_t > 0 else 0.0

        risk_rows.append({
            "risk_tier": tier,
            "match_count": n_t,
            "pct_of_completed": round(n_t / n_comp * 100.0, 1),
            "actual_draw_count": int(n_draws_t),
            "actual_draw_rate_pct": round(draw_rate_t, 2),
            "v4_0_accuracy_pct": round(acc_t, 2),
            "favorite_failure_rate_pct": round(fail_t, 2),
        })

    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(MONITORING_DIR / "draw_risk_outcome_analysis.csv", index=False)
    logger.info(f"Saved Draw Risk outcome analysis:\n{risk_df.to_string(index=False)}")

    # 8. Missed Draw Audit
    actual_draws = df_completed[df_completed["actual_result"] == "D"].copy()
    n_tot_draws = len(actual_draws)
    n_elev_draws = actual_draws["draw_risk_tier"].isin(["MEDIUM", "HIGH", "CRITICAL"]).sum()
    n_low_draws = (actual_draws["draw_risk_tier"] == "LOW").sum()

    logger.info(f"Actual Draws in Completed Sample: {n_tot_draws}")
    logger.info(f"Draws Tagged MEDIUM/HIGH/CRITICAL: {n_elev_draws} / {n_tot_draws} ({n_elev_draws/n_tot_draws*100:.1f}%)")
    logger.info(f"Draws Tagged LOW: {n_low_draws} / {n_tot_draws} ({n_low_draws/n_tot_draws*100:.1f}%)")

    # 9. Post-flight Hash Check
    assert verify_md5(V4_PATH, EXP_V4_MD5, "V4.0 Baseline")
    assert verify_md5(V4_1_PATH, EXP_V4_1_MD5, "V4.1 Production Candidate")

    logger.info("Step 2N Live Monitoring Execution Finished Successfully.")


if __name__ == "__main__":
    run_live_outcome_monitoring()
