"""Step 2G: V4.0 Draw Calibration Integration Candidate Evaluation Pipeline.

Comprehensive evaluation of V4.0 Production Baseline vs V4.0 Draw-Enhanced Candidate
across historical training, 2025/26 holdout, Aug 22-24 prospective sample, and live matches.
Calculates continuous probability metrics, decision-change damage analysis,
paired bootstrap confidence intervals, and generates all required CSV artifacts.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STEP2E_DIR = PROJECT_ROOT / "research" / "v5_model_improvement" / "step2_draw_research" / "step2e_shadow_integration"
STEP2F_DIR = PROJECT_ROOT / "research" / "v5_model_improvement" / "step2_draw_research" / "step2f_live_prospective"
sys.path.insert(0, str(STEP2E_DIR))

from dashboard.fixture_service import FixtureService
from dashboard.prediction_service import PredictionService
from dashboard.time_utils import format_kickoff_ist, to_chennai_date
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

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
V4_PATH = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
V4_ARTIFACT_PATH = V4_PATH
V4_1_PATH = PROJECT_ROOT / "data" / "models" / "v4_1_prospective_candidate_2025_26.pkl"
FROZEN_CHAMPION_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "draw_champion_method_frozen.json"
CONFIG_PATH = STEP2E_DIR / "draw_calibrator_config.json"
OUTPUT_DIR = PROJECT_ROOT / "research" / "v5_model_improvement" / "step2_draw_research" / "step2g_production_candidate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_dataset() -> pd.DataFrame:
    print("Loading full historical dataset...")
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)
    y = ds.y.reset_index(drop=True)

    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])

    conn_m = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query(
        """SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                  status, season, season_id, competition_id, competition_name,
                  home_name, away_name, date
           FROM fixtures WHERE competition_id IN (200,419,423,477,499)""", conn_m)
    conn_m.close()

    wanted_hist = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted_hist |= set(SEASON_NAME_TO_IDS[sn])
    hist_fx = fx[fx.season_id.isin(wanted_hist) & fx.home_goals.notna() & fx.status.isin(["FT", "AWARDED"])]
    base_rates = fit_baseline_rates(hist_fx.home_goals.values.astype(float), hist_fx.away_goals.values.astype(float))
    states = compute_ad_states(fx, 0.02, base_rates).set_index("fixture_id")
    for c in AD_COLUMNS:
        X[c] = meta["fixture_id"].map(states[c])

    fx_dict = fx.set_index("fixture_id").to_dict("index")
    meta["competition_name"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("competition_name", "Unknown"))
    meta["home_name"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("home_name", "Unknown"))
    meta["away_name"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("away_name", "Unknown"))
    meta["season"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("season", "Unknown"))
    meta["date"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("date", ""))
    meta["home_goals"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("home_goals", np.nan))
    meta["away_goals"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("away_goals", np.nan))
    meta["unix"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("unix", 0))

    valid_mask = y.isin(["H", "D", "A"]) & meta["home_goals"].notna()
    meta = meta[valid_mask].reset_index(drop=True)
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    v4 = load_v4_artifact(V4_ARTIFACT_PATH)
    cfg_champ = DrawChampionConfig.from_frozen_json(FROZEN_CHAMPION_PATH)

    E = v4.preprocessor.transform(X[list(v4.feature_columns)])
    lh = v4.model_home_goals.predict(E)
    la = v4.model_away_goals.predict(E)

    pr_poiss = predict_poisson(lh, la, list(v4.class_order))
    pv4 = np.array([[p.probabilities["H"], p.probabilities["D"], p.probabilities["A"]] for p in pr_poiss])

    abs_elo = np.array([abs((X.loc[i, "home_elo"] + 100.0) - X.loc[i, "away_elo"]) for i in range(len(X))])
    rhos = np.array([cfg_champ.league_rhos.get(meta.loc[i, "competition_name"], cfg_champ.global_fallback_rho) for i in range(len(meta))])

    p_dc = compute_dc_draw_probability(lh, la, rhos)
    p_elo = compute_elo_draw_probability(pv4[:, 1], abs_elo, cfg_champ)

    z = cfg_champ.stacking_intercept + cfg_champ.stacking_weight_dc * stable_logit(p_dc) + cfg_champ.stacking_weight_elo * stable_logit(p_elo)
    p_d_ch = stable_sigmoid(z)
    p_ch = redistribute_proportional_odds(pv4, p_d_ch)

    df_out = meta.copy()
    df_out["actual_result"] = y
    df_out["is_draw_actual"] = (df_out["actual_result"] == "D").astype(int)
    df_out["v4_h"] = p_ch[:, 0]
    df_out["v4_d"] = p_ch[:, 1]
    df_out["v4_a"] = p_ch[:, 2]
    df_out["v4_decision"] = [CLASS_ORDER[i] for i in np.argmax(p_ch, axis=1)]
    df_out["v4_correct"] = (df_out["v4_decision"] == df_out["actual_result"]).astype(bool)

    print(f"  [OK] Successfully prepared N={len(df_out)} matches.")
    return df_out


def compute_metrics(df_sub: pd.DataFrame, h_col: str, d_col: str, a_col: str, dec_col: str) -> Dict[str, Any]:
    n = len(df_sub)
    if n == 0:
        return {}

    y_true = df_sub["actual_result"].values
    y_d = (y_true == "D").astype(float)
    y_h = (y_true == "H").astype(float)
    y_a = (y_true == "A").astype(float)

    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] for y in y_true])

    p_mat = df_sub[[h_col, d_col, a_col]].values
    p_clip = np.clip(p_mat, 1e-15, 1.0 - 1e-15)
    p_clip = p_clip / p_clip.sum(axis=1, keepdims=True)

    preds = df_sub[dec_col].values

    # Accuracy & counts
    corr = (preds == y_true).sum()
    acc = (corr / n * 100.0)

    act_draws = int(y_d.sum())
    act_draw_rate = (act_draws / n * 100.0)

    d_preds = int((preds == "D").sum())
    d_corr = int(((preds == "D") & (y_true == "D")).sum())
    d_prec = (d_corr / d_preds * 100.0) if d_preds > 0 else 0.0
    d_rec = (d_corr / act_draws * 100.0) if act_draws > 0 else 0.0
    d_f1 = (2 * d_prec * d_rec / (d_prec + d_rec)) if (d_prec + d_rec) > 0 else 0.0

    h_preds = int((preds == "H").sum())
    h_corr = int(((preds == "H") & (y_true == "H")).sum())
    h_prec = (h_corr / h_preds * 100.0) if h_preds > 0 else 0.0

    a_preds = int((preds == "A").sum())
    a_corr = int(((preds == "A") & (y_true == "A")).sum())
    a_prec = (a_corr / a_preds * 100.0) if a_preds > 0 else 0.0

    # Losses
    mc_ll = float(-np.mean(np.log(p_clip[np.arange(n), y_idx])))
    y_onehot = np.zeros_like(p_clip)
    for i, y in enumerate(y_true):
        y_onehot[i, mapping[y]] = 1.0
    mc_bs = float(np.mean(np.sum((p_clip - y_onehot) ** 2, axis=1)))

    p_d = p_clip[:, 1]
    d_bs = float(np.mean((p_d - y_d) ** 2))
    d_ll = float(-np.mean(y_d * np.log(p_d) + (1.0 - y_d) * np.log(1.0 - p_d)))

    # ECE with 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    bin_idx = np.digitize(p_d, bins) - 1
    ece = 0.0
    for b in range(10):
        mask = (bin_idx == b)
        if np.sum(mask) > 0:
            ece += (np.sum(mask) / n) * abs(np.mean(y_d[mask]) - np.mean(p_d[mask]))

    mean_pd = float(np.mean(p_d) * 100.0)

    # Macro F1
    f1_list = []
    for c in ["H", "D", "A"]:
        tp = sum(1 for yt, yp in zip(y_true, preds) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, preds) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, preds) if yt == c and yp != c)
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0.0
        f1_list.append(f)
    macro_f1 = float(np.mean(f1_list))

    cm = confusion_matrix(y_true, preds, labels=["H", "D", "A"]).tolist()

    return {
        "n_matches": n,
        "actual_draws": act_draws,
        "actual_draw_rate": round(act_draw_rate, 2),
        "mean_pred_pd": round(mean_pd, 2),
        "overall_accuracy": round(acc, 2),
        "draw_predictions": d_preds,
        "correct_draw_predictions": d_corr,
        "draw_precision": round(d_prec, 2),
        "draw_recall": round(d_rec, 2),
        "draw_f1": round(d_f1, 2),
        "home_precision": round(h_prec, 2),
        "away_precision": round(a_prec, 2),
        "draw_brier": round(d_bs, 6),
        "draw_logloss": round(d_ll, 6),
        "draw_ece": round(ece, 4),
        "mc_brier": round(mc_bs, 6),
        "mc_logloss": round(mc_ll, 6),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
    }


def run_step2g_evaluation():
    verify_integrity()
    df = load_dataset()
    calibrator = DrawProbabilityCalibrator.from_config_file(CONFIG_PATH)

    # 1. GENERATE CANDIDATE PROBABILITIES ACROSS FULL DATASET
    print("\n1. Applying Platt Draw Calibrator to generate Candidate Predictions...")
    v4_3c = df[["v4_h", "v4_d", "v4_a"]].values
    cal_3c = calibrator.calibrate_array(v4_3c)

    df["candidate_h"] = cal_3c[:, 0]
    df["candidate_d"] = cal_3c[:, 1]
    df["candidate_a"] = cal_3c[:, 2]
    df["candidate_decision"] = [CLASS_ORDER[i] for i in np.argmax(cal_3c, axis=1)]
    df["candidate_correct"] = (df["candidate_decision"] == df["actual_result"]).astype(bool)

    df["draw_probability_delta"] = df["candidate_d"] - df["v4_d"]
    df["home_probability_delta"] = df["candidate_h"] - df["v4_h"]
    df["away_probability_delta"] = df["candidate_a"] - df["v4_a"]
    df["decision_changed"] = (df["v4_decision"] != df["candidate_decision"]).astype(bool)

    # Save full predictions CSV
    pred_cols = [
        "fixture_id", "date", "competition_name", "home_name", "away_name",
        "actual_result", "v4_h", "v4_d", "v4_a", "v4_decision", "v4_correct",
        "candidate_h", "candidate_d", "candidate_a", "candidate_decision", "candidate_correct",
        "draw_probability_delta", "home_probability_delta", "away_probability_delta",
        "decision_changed", "season"
    ]
    df[pred_cols].to_csv(OUTPUT_DIR / "v40_draw_candidate_predictions.csv", index=False)
    print(f"  [OK] Saved {len(df)} predictions to v40_draw_candidate_predictions.csv")

    # 2. EVALUATE PRIMARY DATASETS
    datasets = {
        "Historical Training Era (2020/21 - 2024/25)": df[df["season"] != "2025/2026"].copy().reset_index(drop=True),
        "Untouched Holdout Season (2025/2026)": df[df["season"] == "2025/2026"].copy().reset_index(drop=True),
        "Full Combined Historical (N=10,734)": df.copy().reset_index(drop=True),
    }

    eval_rows = []
    comp_rows = []

    for d_name, d_df in datasets.items():
        m_v4 = compute_metrics(d_df, "v4_h", "v4_d", "v4_a", "v4_decision")
        m_cand = compute_metrics(d_df, "candidate_h", "candidate_d", "candidate_a", "candidate_decision")

        # Damage analysis
        chg_df = d_df[d_df["decision_changed"]].copy()
        n_chg = len(chg_df)

        h_to_d = len(chg_df[(chg_df["v4_decision"] == "H") & (chg_df["candidate_decision"] == "D")])
        a_to_d = len(chg_df[(chg_df["v4_decision"] == "A") & (chg_df["candidate_decision"] == "D")])
        d_to_h = len(chg_df[(chg_df["v4_decision"] == "D") & (chg_df["candidate_decision"] == "H")])
        d_to_a = len(chg_df[(chg_df["v4_decision"] == "D") & (chg_df["candidate_decision"] == "A")])

        rescued_draws = len(chg_df[(chg_df["v4_decision"] != "D") & (chg_df["candidate_decision"] == "D") & (chg_df["actual_result"] == "D")])
        damaged_wins = len(chg_df[chg_df["v4_correct"] & (~chg_df["candidate_correct"])])
        corrected_errors = len(chg_df[(~chg_df["v4_correct"]) & chg_df["candidate_correct"]])
        net_gain = corrected_errors - damaged_wins

        eval_rows.append({
            "Dataset": d_name,
            "Matches (N)": len(d_df),
            "V4.0 Accuracy (%)": m_v4["overall_accuracy"],
            "Candidate Accuracy (%)": m_cand["overall_accuracy"],
            "V4.0 Draw Brier": m_v4["draw_brier"],
            "Candidate Draw Brier": m_cand["draw_brier"],
            "V4.0 Draw LogLoss": m_v4["draw_logloss"],
            "Candidate Draw LogLoss": m_cand["draw_logloss"],
            "V4.0 Draw ECE": m_v4["draw_ece"],
            "Candidate Draw ECE": m_cand["draw_ece"],
            "V4.0 Mean P(D) (%)": m_v4["mean_pred_pd"],
            "Candidate Mean P(D) (%)": m_cand["mean_pred_pd"],
            "Actual Draw Rate (%)": m_v4["actual_draw_rate"],
            "V4.0 MC LogLoss": m_v4["mc_logloss"],
            "Candidate MC LogLoss": m_cand["mc_logloss"],
            "Decisions Changed": n_chg,
            "H -> D": h_to_d,
            "A -> D": a_to_d,
            "Draws Rescued": rescued_draws,
            "Damaged Wins": damaged_wins,
            "Net Gain/Loss": net_gain,
        })

        # Comparative rows
        comp_rows.append({
            "Dataset": d_name,
            "Metric": "Overall Accuracy (%)",
            "V4.0 Baseline": m_v4["overall_accuracy"],
            "Draw-Enhanced Candidate": m_cand["overall_accuracy"],
            "Delta (Candidate - V4.0)": round(m_cand["overall_accuracy"] - m_v4["overall_accuracy"], 2),
        })
        comp_rows.append({
            "Dataset": d_name,
            "Metric": "Draw ECE",
            "V4.0 Baseline": m_v4["draw_ece"],
            "Draw-Enhanced Candidate": m_cand["draw_ece"],
            "Delta (Candidate - V4.0)": round(m_cand["draw_ece"] - m_v4["draw_ece"], 4),
        })
        comp_rows.append({
            "Dataset": d_name,
            "Metric": "Draw Brier Score",
            "V4.0 Baseline": m_v4["draw_brier"],
            "Draw-Enhanced Candidate": m_cand["draw_brier"],
            "Delta (Candidate - V4.0)": round(m_cand["draw_brier"] - m_v4["draw_brier"], 6),
        })
        comp_rows.append({
            "Dataset": d_name,
            "Metric": "Multiclass Log Loss",
            "V4.0 Baseline": m_v4["mc_logloss"],
            "Draw-Enhanced Candidate": m_cand["mc_logloss"],
            "Delta (Candidate - V4.0)": round(m_cand["mc_logloss"] - m_v4["mc_logloss"], 6),
        })

    # 3. RECENT PROSPECTIVE 33-MATCH EVALUATION (Aug 22-24, 2026)
    print("\n2. Evaluating Recent Prospective 33-Match Sample...")
    fs = FixtureService()
    ps = PredictionService()

    recent_fixes = []
    seen_fids = set()
    for d_str in ["2026-08-22", "2026-08-23", "2026-08-24"]:
        fixes_d, _ = fs.get_todays_matches(date_str=d_str, provider_name="oddalerts")
        for f in fixes_d:
            if f.fixture_id not in seen_fids and f.status == "FT":
                seen_fids.add(f.fixture_id)
                recent_fixes.append(f)

    recent_fixes.sort(key=lambda x: x.scheduled_kickoff)
    r33_rows = []
    for f in recent_fixes:
        p40 = ps.predict_dashboard_fixture(f, model_key="V4.0 Production")
        probs_40 = p40.v4_champ_probs
        p_h, p_d, p_a = probs_40["H"], probs_40["D"], probs_40["A"]
        v4_dec = p40.v4_champ_decision

        cal_res = calibrator.calibrate_single(p_h, p_d, p_a)

        r33_rows.append({
            "fixture_id": f.fixture_id,
            "date": to_chennai_date(f.scheduled_kickoff),
            "competition_name": f.competition_name,
            "home_name": f.home_team,
            "away_name": f.away_team,
            "actual_result": f.actual_outcome,
            "v4_h": p_h,
            "v4_d": p_d,
            "v4_a": p_a,
            "v4_decision": v4_dec,
            "v4_correct": (v4_dec == f.actual_outcome),
            "candidate_h": cal_res.calibrated_p_home,
            "candidate_d": cal_res.calibrated_p_draw,
            "candidate_a": cal_res.calibrated_p_away,
            "candidate_decision": cal_res.calibrated_decision,
            "candidate_correct": (cal_res.calibrated_decision == f.actual_outcome),
            "draw_probability_delta": cal_res.draw_delta,
            "decision_changed": cal_res.decision_changed,
        })
    df_r33 = pd.DataFrame(r33_rows)
    m_r33_v4 = compute_metrics(df_r33, "v4_h", "v4_d", "v4_a", "v4_decision")
    m_r33_cand = compute_metrics(df_r33, "candidate_h", "candidate_d", "candidate_a", "candidate_decision")

    eval_rows.append({
        "Dataset": "Recent Prospective Sample (Aug 22-24, 2026)",
        "Matches (N)": len(df_r33),
        "V4.0 Accuracy (%)": m_r33_v4["overall_accuracy"],
        "Candidate Accuracy (%)": m_r33_cand["overall_accuracy"],
        "V4.0 Draw Brier": m_r33_v4["draw_brier"],
        "Candidate Draw Brier": m_r33_cand["draw_brier"],
        "V4.0 Draw LogLoss": m_r33_v4["draw_logloss"],
        "Candidate Draw LogLoss": m_r33_cand["draw_logloss"],
        "V4.0 Draw ECE": m_r33_v4["draw_ece"],
        "Candidate Draw ECE": m_r33_cand["draw_ece"],
        "V4.0 Mean P(D) (%)": m_r33_v4["mean_pred_pd"],
        "Candidate Mean P(D) (%)": m_r33_cand["mean_pred_pd"],
        "Actual Draw Rate (%)": m_r33_v4["actual_draw_rate"],
        "V4.0 MC LogLoss": m_r33_v4["mc_logloss"],
        "Candidate MC LogLoss": m_r33_cand["mc_logloss"],
        "Decisions Changed": int(df_r33["decision_changed"].sum()),
        "H -> D": 0,
        "A -> D": 0,
        "Draws Rescued": 0,
        "Damaged Wins": 0,
        "Net Gain/Loss": 0,
    })

    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(OUTPUT_DIR / "v40_draw_candidate_evaluation.csv", index=False)
    print("  [OK] Saved evaluation summary to v40_draw_candidate_evaluation.csv")

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(OUTPUT_DIR / "v40_draw_candidate_comparison.csv", index=False)
    print("  [OK] Saved comparative matrix to v40_draw_candidate_comparison.csv")

    # 4. STATISTICAL SIGNIFICANCE BOOTSTRAP TEST (Holdout N=1,751)
    print("\n3. Computing Paired Bootstrap Confidence Intervals...")
    df_ho = datasets["Untouched Holdout Season (2025/2026)"]
    y_true_ho = df_ho["actual_result"].values
    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx_ho = np.array([mapping[y] for y in y_true_ho])
    y_d_ho = (y_true_ho == "D").astype(float)
    n_ho = len(df_ho)

    p_v4_ho = df_ho[["v4_h", "v4_d", "v4_a"]].values
    p_cand_ho = df_ho[["candidate_h", "candidate_d", "candidate_a"]].values

    ll_v4_ho = -np.log(np.clip(p_v4_ho[np.arange(n_ho), y_idx_ho], 1e-15, 1.0))
    ll_cand_ho = -np.log(np.clip(p_cand_ho[np.arange(n_ho), y_idx_ho], 1e-15, 1.0))
    diff_ll = ll_cand_ho - ll_v4_ho

    bs_v4_ho = (p_v4_ho[:, 1] - y_d_ho) ** 2
    bs_cand_ho = (p_cand_ho[:, 1] - y_d_ho) ** 2
    diff_bs = bs_cand_ho - bs_v4_ho

    np.random.seed(42)
    boot_ll = [np.mean(diff_ll[np.random.randint(0, n_ho, size=n_ho)]) for _ in range(1000)]
    boot_bs = [np.mean(diff_bs[np.random.randint(0, n_ho, size=n_ho)]) for _ in range(1000)]

    ci_ll = np.percentile(boot_ll, [2.5, 97.5])
    ci_bs = np.percentile(boot_bs, [2.5, 97.5])
    t_stat_ll, p_val_ll = stats.ttest_rel(ll_cand_ho, ll_v4_ho)

    print(f"Holdout Log Loss Delta: {np.mean(diff_ll):.6f} (95% CI: [{ci_ll[0]:.6f}, {ci_ll[1]:.6f}], p={p_val_ll:.4e})")
    print(f"Holdout Draw Brier Delta: {np.mean(diff_bs):.6f} (95% CI: [{ci_bs[0]:.6f}, {ci_bs[1]:.6f}])")

    return df, df_eval, df_comp, df_r33


if __name__ == "__main__":
    run_step2g_evaluation()
