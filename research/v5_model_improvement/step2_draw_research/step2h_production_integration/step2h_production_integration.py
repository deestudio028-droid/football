"""Step 2H — V4.0 Draw-Enhanced Production Pipeline Integration & Evaluation.

Validates the integration of the Platt Draw Probability Calibrator into the
production prediction pipeline, comparing V4.0 Production Baseline vs V4.0 Draw-Enhanced.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("step2h_production_integration")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys_paths = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from dashboard.fixture_service import DashboardFixture
from dashboard.model_registry import get_model_registry
from dashboard.prediction_service import PredictionService

STEP2H_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2h_production_integration"
STEP2H_DIR.mkdir(parents=True, exist_ok=True)

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
MATCHES_DB = PROJECT_ROOT / "data/processed/matches.db"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def verify_md5(path: Path, expected: str, name: str) -> bool:
    if not path.exists():
        logger.error(f"{name} artifact missing at {path}")
        return False
    act = hashlib.md5(path.read_bytes()).hexdigest()
    if act != expected:
        logger.error(f"{name} MD5 MISMATCH! Expected {expected}, got {act}")
        return False
    logger.info(f"{name} MD5 Verified: {act}")
    return True


def compute_binary_ece(y_true_binary: np.ndarray, probs_binary: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_binary)
    for i in range(n_bins):
        b_low, b_high = bins[i], bins[i + 1]
        mask = (probs_binary >= b_low) & (probs_binary < b_high) if i < n_bins - 1 else (probs_binary >= b_low) & (probs_binary <= b_high)
        n_b = np.sum(mask)
        if n_b > 0:
            bin_acc = np.mean(y_true_binary[mask])
            bin_conf = np.mean(probs_binary[mask])
            ece += (n_b / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_metrics(
    y_true: List[str],
    probs_v4: np.ndarray,
    probs_enh: np.ndarray,
    dec_v4: List[str],
    dec_enh: List[str],
    dataset_name: str,
) -> Dict[str, Any]:
    n = len(y_true)
    y_arr = np.array(y_true)
    dec_v4_arr = np.array(dec_v4)
    dec_enh_arr = np.array(dec_enh)

    # Actual outcome counts
    n_h = int(np.sum(y_arr == "H"))
    n_d = int(np.sum(y_arr == "D"))
    n_a = int(np.sum(y_arr == "A"))

    # V4.0 metrics
    acc_v4 = float(np.mean(y_arr == dec_v4_arr)) * 100
    h_pred_v4 = int(np.sum(dec_v4_arr == "H"))
    h_corr_v4 = int(np.sum((dec_v4_arr == "H") & (y_arr == "H")))
    a_pred_v4 = int(np.sum(dec_v4_arr == "A"))
    a_corr_v4 = int(np.sum((dec_v4_arr == "A") & (y_arr == "A")))
    d_pred_v4 = int(np.sum(dec_v4_arr == "D"))
    d_corr_v4 = int(np.sum((dec_v4_arr == "D") & (y_arr == "D")))

    rec_d_v4 = (d_corr_v4 / n_d * 100) if n_d > 0 else 0.0
    prec_d_v4 = (d_corr_v4 / d_pred_v4 * 100) if d_pred_v4 > 0 else 0.0
    f1_d_v4 = (2 * prec_d_v4 * rec_d_v4 / (prec_d_v4 + rec_d_v4)) if (prec_d_v4 + rec_d_v4) > 0 else 0.0

    # Draw-Enhanced metrics
    acc_enh = float(np.mean(y_arr == dec_enh_arr)) * 100
    h_pred_enh = int(np.sum(dec_enh_arr == "H"))
    h_corr_enh = int(np.sum((dec_enh_arr == "H") & (y_arr == "H")))
    a_pred_enh = int(np.sum(dec_enh_arr == "A"))
    a_corr_enh = int(np.sum((dec_enh_arr == "A") & (y_arr == "A")))
    d_pred_enh = int(np.sum(dec_enh_arr == "D"))
    d_corr_enh = int(np.sum((dec_enh_arr == "D") & (y_arr == "D")))

    rec_d_enh = (d_corr_enh / n_d * 100) if n_d > 0 else 0.0
    prec_d_enh = (d_corr_enh / d_pred_enh * 100) if d_pred_enh > 0 else 0.0
    f1_d_enh = (2 * prec_d_enh * rec_d_enh / (prec_d_enh + rec_d_enh)) if (prec_d_enh + rec_d_enh) > 0 else 0.0

    # Continuous losses
    y_d_bin = (y_arr == "D").astype(float)
    eps = 1e-15
    clip_p_d_v4 = np.clip(probs_v4[:, 1], eps, 1 - eps)
    clip_p_d_enh = np.clip(probs_enh[:, 1], eps, 1 - eps)

    brier_d_v4 = float(np.mean((probs_v4[:, 1] - y_d_bin) ** 2))
    brier_d_enh = float(np.mean((probs_enh[:, 1] - y_d_bin) ** 2))

    ll_d_v4 = float(-np.mean(y_d_bin * np.log(clip_p_d_v4) + (1 - y_d_bin) * np.log(1 - clip_p_d_v4)))
    ll_d_enh = float(-np.mean(y_d_bin * np.log(clip_p_d_enh) + (1 - y_d_bin) * np.log(1 - clip_p_d_enh)))

    ece_d_v4 = compute_binary_ece(y_d_bin, probs_v4[:, 1])
    ece_d_enh = compute_binary_ece(y_d_bin, probs_enh[:, 1])

    # Multiclass Log Loss
    class_map = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([class_map[c] for c in y_arr])
    p_true_v4 = np.clip(probs_v4[np.arange(n), y_idx], eps, 1.0)
    p_true_enh = np.clip(probs_enh[np.arange(n), y_idx], eps, 1.0)
    mc_ll_v4 = float(-np.mean(np.log(p_true_v4)))
    mc_ll_enh = float(-np.mean(np.log(p_true_enh)))

    # Decision shift analysis
    diff_mask = (dec_v4_arr != dec_enh_arr)
    n_diff = int(np.sum(diff_mask))
    h_to_d = int(np.sum((dec_v4_arr == "H") & (dec_enh_arr == "D")))
    a_to_d = int(np.sum((dec_v4_arr == "A") & (dec_enh_arr == "D")))
    d_to_h = int(np.sum((dec_v4_arr == "D") & (dec_enh_arr == "H")))
    d_to_a = int(np.sum((dec_v4_arr == "D") & (dec_enh_arr == "A")))

    rescued_draws = int(np.sum(diff_mask & (dec_enh_arr == "D") & (y_arr == "D")))
    damaged_wins = int(np.sum(diff_mask & (dec_v4_arr == y_arr) & (dec_enh_arr != y_arr)))
    net_accuracy_delta = acc_enh - acc_v4

    return {
        "dataset": dataset_name,
        "n_matches": n,
        "n_actual_h": n_h,
        "n_actual_d": n_d,
        "n_actual_a": n_a,
        "actual_draw_rate": float(n_d / n),
        # V4.0
        "v4_correct": int(np.sum(y_arr == dec_v4_arr)),
        "v4_accuracy": acc_v4,
        "v4_h_pred": h_pred_v4,
        "v4_h_corr": h_corr_v4,
        "v4_a_pred": a_pred_v4,
        "v4_a_corr": a_corr_v4,
        "v4_d_pred": d_pred_v4,
        "v4_d_corr": d_corr_v4,
        "v4_d_recall": rec_d_v4,
        "v4_d_precision": prec_d_v4,
        "v4_d_f1": f1_d_v4,
        "v4_d_brier": brier_d_v4,
        "v4_d_logloss": ll_d_v4,
        "v4_d_ece": ece_d_v4,
        "v4_mc_logloss": mc_ll_v4,
        "v4_mean_p_d": float(np.mean(probs_v4[:, 1])),
        # Draw-Enhanced
        "enh_correct": int(np.sum(y_arr == dec_enh_arr)),
        "enh_accuracy": acc_enh,
        "enh_h_pred": h_pred_enh,
        "enh_h_corr": h_corr_enh,
        "enh_a_pred": a_pred_enh,
        "enh_a_corr": a_corr_enh,
        "enh_d_pred": d_pred_enh,
        "enh_d_corr": d_corr_enh,
        "enh_d_recall": rec_d_enh,
        "enh_d_precision": prec_d_enh,
        "enh_d_f1": f1_d_enh,
        "enh_d_brier": brier_d_enh,
        "enh_d_logloss": ll_d_enh,
        "enh_d_ece": ece_d_enh,
        "enh_mc_logloss": mc_ll_enh,
        "enh_mean_p_d": float(np.mean(probs_enh[:, 1])),
        # Shifts
        "decisions_changed": n_diff,
        "pct_changed": float(n_diff / n * 100),
        "h_to_d": h_to_d,
        "a_to_d": a_to_d,
        "d_to_h": d_to_h,
        "d_to_a": d_to_a,
        "rescued_draws": rescued_draws,
        "damaged_wins": damaged_wins,
        "net_accuracy_delta": net_accuracy_delta,
    }


def main():
    logger.info("================================================================")
    logger.info("STEP 2H — V4.0 DRAW-ENHANCED PRODUCTION PIPELINE INTEGRATION")
    logger.info("================================================================")

    # 1. Pre-flight Hash Checks
    assert verify_md5(V4_PATH, EXP_V4_MD5, "V4.0 Baseline")
    assert verify_md5(V4_1_PATH, EXP_V4_1_MD5, "V4.1 Production Candidate")

    # 2. Initialize PredictionService
    ps = PredictionService()

    # 3. Load historical database and datasets
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    df_all = pd.read_sql_query(
        "SELECT fixture_id, date, season, home_id, away_id, home_name, away_name, competition_name, competition_id, home_goals, away_goals "
        "FROM fixtures WHERE status IN ('FT', 'AWARDED') AND home_goals IS NOT NULL ORDER BY unix ASC, fixture_id ASC",
        conn,
    )
    conn.close()

    logger.info(f"Loaded {len(df_all)} total completed historical fixtures from matches.db.")

    # 4. Generate Predictions across historical and prospective sets
    step2e_dir = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"
    df_r33_raw = pd.read_csv(step2e_dir / "shadow_33_match_audit.csv")

    r33_records = []
    for _, row in df_r33_raw.iterrows():
        fid = int(row["fixture_id"])
        h_name = str(row["home_team"])
        a_name = str(row["away_team"])
        comp_name = str(row["league"])
        act_res = str(row["actual_outcome"])
        score = str(row["score"])
        v4_h = float(row["v4_p_H"])
        v4_d = float(row["v4_p_D"])
        v4_a = float(row["v4_p_A"])
        v4_dec = str(row["v4_decision"])

        cal_res = ps._draw_calibrator.calibrate_single(v4_h, v4_d, v4_a)

        r33_records.append({
            "fixture_id": fid,
            "date": str(row.get("date_ist", "")),
            "competition": comp_name,
            "home_team": h_name,
            "away_team": a_name,
            "score": score,
            "actual_outcome": act_res,
            "v4_p_h": v4_h,
            "v4_p_d": v4_d,
            "v4_p_a": v4_a,
            "v4_decision": v4_dec,
            "v4_correct": v4_dec == act_res,
            "enh_p_h": cal_res.calibrated_p_home,
            "enh_p_d": cal_res.calibrated_p_draw,
            "enh_p_a": cal_res.calibrated_p_away,
            "enh_decision": cal_res.calibrated_decision,
            "enh_correct": cal_res.calibrated_decision == act_res,
            "draw_delta": cal_res.draw_delta,
            "decision_changed": v4_dec != cal_res.calibrated_decision,
        })

    df_33_out = pd.DataFrame(r33_records)
    df_33_out.to_csv(STEP2H_DIR / "step2h_33_match_audit.csv", index=False)
    logger.info(f"Saved {len(df_33_out)} records to step2h_33_match_audit.csv")

    # Now load the full historical dataset predictions from Step 2G reference
    step2g_preds = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2g_production_candidate/v40_draw_candidate_predictions.csv"
    df_hist_preds = pd.read_csv(step2g_preds)
    df_hist_preds.to_csv(STEP2H_DIR / "step2h_prediction_audit.csv", index=False)
    logger.info(f"Saved {len(df_hist_preds)} records to step2h_prediction_audit.csv")

    # Compute metric tables across datasets
    eval_list = []

    # 1. 33 Recent Prospective Matches
    probs_v4_33 = df_33_out[["v4_p_h", "v4_p_d", "v4_p_a"]].values
    probs_enh_33 = df_33_out[["enh_p_h", "enh_p_d", "enh_p_a"]].values
    y_33 = df_33_out["actual_outcome"].tolist()
    dec_v4_33 = df_33_out["v4_decision"].tolist()
    dec_enh_33 = df_33_out["enh_decision"].tolist()
    eval_list.append(compute_metrics(y_33, probs_v4_33, probs_enh_33, dec_v4_33, dec_enh_33, "Recent Prospective Sample (Aug 22–24)"))

    # 2. 2025/26 Holdout Season
    df_ho = df_hist_preds[df_hist_preds["season"] == "2025/2026"].reset_index(drop=True)
    probs_v4_ho = df_ho[["v4_h", "v4_d", "v4_a"]].values
    probs_enh_ho = df_ho[["candidate_h", "candidate_d", "candidate_a"]].values
    y_ho = df_ho["actual_result"].tolist()
    dec_v4_ho = df_ho["v4_decision"].tolist()
    dec_enh_ho = df_ho["candidate_decision"].tolist()
    eval_list.append(compute_metrics(y_ho, probs_v4_ho, probs_enh_ho, dec_v4_ho, dec_enh_ho, "Untouched Holdout Season (2025/2026)"))

    # 3. 2020/21–2024/25 Training Era
    df_tr = df_hist_preds[df_hist_preds["season"] != "2025/2026"].reset_index(drop=True)
    probs_v4_tr = df_tr[["v4_h", "v4_d", "v4_a"]].values
    probs_enh_tr = df_tr[["candidate_h", "candidate_d", "candidate_a"]].values
    y_tr = df_tr["actual_result"].tolist()
    dec_v4_tr = df_tr["v4_decision"].tolist()
    dec_enh_tr = df_tr["candidate_decision"].tolist()
    eval_list.append(compute_metrics(y_tr, probs_v4_tr, probs_enh_tr, dec_v4_tr, dec_enh_tr, "Historical Training Era (2020/21–2024/25)"))

    # 4. Combined Historical Dataset
    probs_v4_all = df_hist_preds[["v4_h", "v4_d", "v4_a"]].values
    probs_enh_all = df_hist_preds[["candidate_h", "candidate_d", "candidate_a"]].values
    y_all = df_hist_preds["actual_result"].tolist()
    dec_v4_all = df_hist_preds["v4_decision"].tolist()
    dec_enh_all = df_hist_preds["candidate_decision"].tolist()
    eval_list.append(compute_metrics(y_all, probs_v4_all, probs_enh_all, dec_v4_all, dec_enh_all, "Full Combined Historical (6 Seasons)"))

    df_eval = pd.DataFrame(eval_list)
    df_eval.to_csv(STEP2H_DIR / "step2h_comparison.csv", index=False)
    logger.info(f"Saved comparative metrics to step2h_comparison.csv")

    # Post-flight MD5 verification
    assert verify_md5(V4_PATH, EXP_V4_MD5, "V4.0 Baseline")
    assert verify_md5(V4_1_PATH, EXP_V4_1_MD5, "V4.1 Production Candidate")

    logger.info("STEP 2H Execution Finished Successfully.")


if __name__ == "__main__":
    main()
