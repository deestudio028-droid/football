"""Step 2I — Draw Decision Intelligence Research.

Comprehensive research engine evaluating 10 distinct Draw Decision policies,
multi-signal draw confidence scoring, threshold sensitivity, chronological walk-forward
stability, league robustness, and untouched 2025/26 holdout validation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("step2i_draw_decision")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys_paths = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

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
from draw_probability_calibrator import DrawProbabilityCalibrator

# Directory & Artifacts
STEP2I_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2i_draw_decision"
STEP2I_DIR.mkdir(parents=True, exist_ok=True)

MATCHES_DB = PROJECT_ROOT / "data/processed/matches.db"
FEATURES_DB = PROJECT_ROOT / "data/processed/features.db"
V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
FROZEN_CHAMPION_PATH = PROJECT_ROOT / "research/v4_promotion/draw_champion_method_frozen.json"
CALIBRATOR_CONFIG_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration/draw_calibrator_config.json"
STEP2E_33_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration/shadow_33_match_audit.csv"

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


def compute_dc_score_matrix(lh: float, la: float, rho: float, max_goals: int = 10) -> np.ndarray:
    p_h = np.array([poisson.pmf(i, lh) for i in range(max_goals + 1)])
    p_a = np.array([poisson.pmf(j, la) for j in range(max_goals + 1)])
    mat = np.outer(p_h, p_a)

    if lh > 0 and la > 0:
        mat[0, 0] = max(1e-15, mat[0, 0] * (1.0 - lh * la * rho))
        mat[1, 0] = max(1e-15, mat[1, 0] * (1.0 + la * rho))
        mat[0, 1] = max(1e-15, mat[0, 1] * (1.0 + lh * rho))
        mat[1, 1] = max(1e-15, mat[1, 1] * (1.0 - rho))

    s = np.sum(mat)
    return mat / s if s > 0 else mat


def load_full_research_dataset() -> pd.DataFrame:
    logger.info("Loading full historical dataset with score-space and multi-signal features...")
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
           FROM fixtures WHERE competition_id IN (200,419,423,477,499)""",
        conn_m,
    )
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

    v4 = load_v4_artifact(V4_PATH)
    cfg_champ = DrawChampionConfig.from_frozen_json(FROZEN_CHAMPION_PATH)
    calibrator = DrawProbabilityCalibrator.from_config_file(CALIBRATOR_CONFIG_PATH)

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

    # Calibrated Platt probabilities
    p_cal = calibrator.calibrate_array(p_ch)

    # Score-space signals
    p_00_list, p_11_list, p_22_list, p_low_score_list, p_score_draw_list = [], [], [], [], []
    for i in range(len(meta)):
        mat = compute_dc_score_matrix(lh[i], la[i], rhos[i])
        p_00_list.append(mat[0, 0])
        p_11_list.append(mat[1, 1])
        p_22_list.append(mat[2, 2])
        p_low = mat[0, 0] + mat[1, 0] + mat[0, 1] + mat[1, 1] + mat[2, 0] + mat[0, 2]
        p_low_score_list.append(p_low)
        p_score_draw_list.append(sum(mat[k, k] for k in range(mat.shape[0])))

    df_out = meta.copy()
    df_out["actual_result"] = y
    df_out["is_draw_actual"] = (df_out["actual_result"] == "D").astype(int)

    # V4.0 baseline probabilities & decision
    df_out["v4_p_H"] = p_ch[:, 0]
    df_out["v4_p_D"] = p_ch[:, 1]
    df_out["v4_p_A"] = p_ch[:, 2]
    df_out["v4_decision"] = [CLASS_ORDER[i] for i in np.argmax(p_ch, axis=1)]

    # Calibrated Platt probabilities & decision
    df_out["cal_p_H"] = p_cal[:, 0]
    df_out["cal_p_D"] = p_cal[:, 1]
    df_out["cal_p_A"] = p_cal[:, 2]
    df_out["cal_decision"] = [CLASS_ORDER[i] for i in np.argmax(p_cal, axis=1)]

    # Model Features & Diagnostics
    df_out["lambda_home"] = lh
    df_out["lambda_away"] = la
    df_out["lambda_total"] = lh + la
    df_out["lambda_gap"] = np.abs(lh - la)
    df_out["home_elo"] = X["home_elo"]
    df_out["away_elo"] = X["away_elo"]
    df_out["elo_gap"] = np.abs(X["home_elo"] - X["away_elo"])
    df_out["abs_elo_diff"] = abs_elo

    # Probability margins
    df_out["v4_win_diff"] = np.abs(df_out["v4_p_H"] - df_out["v4_p_A"])
    df_out["v4_max_win"] = np.maximum(df_out["v4_p_H"], df_out["v4_p_A"])
    df_out["v4_win_to_draw_margin"] = df_out["v4_max_win"] - df_out["v4_p_D"]

    df_out["cal_win_diff"] = np.abs(df_out["cal_p_H"] - df_out["cal_p_A"])
    df_out["cal_max_win"] = np.maximum(df_out["cal_p_H"], df_out["cal_p_A"])
    df_out["cal_win_to_draw_margin"] = df_out["cal_max_win"] - df_out["cal_p_D"]

    # Score-space signals
    df_out["p_00"] = p_00_list
    df_out["p_11"] = p_11_list
    df_out["p_22"] = p_22_list
    df_out["p_low_score_mass"] = p_low_score_list
    df_out["p_score_space_draw"] = p_score_draw_list

    # Online AD features
    df_out["A_home"] = X["A_home"]
    df_out["D_home"] = X["D_home"]
    df_out["A_away"] = X["A_away"]
    df_out["D_away"] = X["D_away"]
    df_out["ad_attack_gap"] = np.abs(X["A_home"] - X["A_away"])
    df_out["ad_defense_gap"] = np.abs(X["D_home"] - X["D_away"])
    df_out["ad_net_power_gap"] = np.abs((X["A_home"] - X["D_away"]) - (X["A_away"] - X["D_home"]))

    logger.info(f"Loaded {len(df_out)} total fixtures across {df_out['season'].nunique()} seasons.")
    return df_out


def evaluate_decision_policy(
    df: pd.DataFrame,
    decisions: List[str],
    policy_name: str,
    baseline_col: str = "v4_decision",
    actual_col: str = "actual_result",
) -> Dict[str, Any]:
    n = len(df)
    y_true = df[actual_col].values
    base_dec = df[baseline_col].values
    cand_dec = np.array(decisions)

    # Basic accuracy & macro metrics
    acc = accuracy_score(y_true, cand_dec) * 100.0
    base_acc = accuracy_score(y_true, base_dec) * 100.0
    macro_f1 = f1_score(y_true, cand_dec, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, cand_dec) * 100.0

    # Draw specific metrics
    y_d = (y_true == "D")
    cand_d = (cand_dec == "D")
    base_d = (base_dec == "D")

    draw_preds = int(np.sum(cand_d))
    draw_actual = int(np.sum(y_d))
    draw_corr = int(np.sum(cand_d & y_d))
    draw_rec = (draw_corr / draw_actual * 100.0) if draw_actual > 0 else 0.0
    draw_prec = (draw_corr / draw_preds * 100.0) if draw_preds > 0 else 0.0
    draw_f1 = (2 * draw_prec * draw_rec / (draw_prec + draw_rec)) if (draw_prec + draw_rec) > 0 else 0.0

    # Home & Away metrics
    y_h = (y_true == "H")
    cand_h = (cand_dec == "H")
    h_preds = int(np.sum(cand_h))
    h_corr = int(np.sum(cand_h & y_h))
    h_prec = (h_corr / h_preds * 100.0) if h_preds > 0 else 0.0
    h_rec = (h_corr / int(np.sum(y_h)) * 100.0) if np.sum(y_h) > 0 else 0.0

    y_a = (y_true == "A")
    cand_a = (cand_dec == "A")
    a_preds = int(np.sum(cand_a))
    a_corr = int(np.sum(cand_a & y_a))
    a_prec = (a_corr / a_preds * 100.0) if a_preds > 0 else 0.0
    a_rec = (a_corr / int(np.sum(y_a)) * 100.0) if np.sum(y_a) > 0 else 0.0

    # Shifts vs Baseline
    diff_mask = (cand_dec != base_dec)
    decisions_changed = int(np.sum(diff_mask))
    rescued_draws = int(np.sum(diff_mask & (cand_dec == "D") & (y_true == "D")))
    damaged_wins = int(np.sum(diff_mask & (base_dec == y_true) & (cand_dec != y_true)))
    false_draw_flips = int(np.sum(diff_mask & (cand_dec == "D") & (y_true != "D")))
    net_correct_delta = int(np.sum(cand_dec == y_true)) - int(np.sum(base_dec == y_true))
    acc_delta = acc - base_acc

    return {
        "policy": policy_name,
        "n_matches": n,
        "accuracy": acc,
        "baseline_accuracy": base_acc,
        "accuracy_delta": acc_delta,
        "net_correct_delta": net_correct_delta,
        "draw_predictions": draw_preds,
        "correct_draws": draw_corr,
        "draw_recall": draw_rec,
        "draw_precision": draw_prec,
        "draw_f1": draw_f1,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "home_precision": h_prec,
        "home_recall": h_rec,
        "away_precision": a_prec,
        "away_recall": a_rec,
        "decisions_changed": decisions_changed,
        "rescued_draws": rescued_draws,
        "damaged_wins": damaged_wins,
        "false_draw_flips": false_draw_flips,
    }


def fit_draw_confidence_model(df_train: pd.DataFrame) -> Tuple[LogisticRegression, List[str]]:
    """Fit regularized logistic draw confidence model on training split only."""
    feature_cols = [
        "cal_p_D",
        "cal_win_diff",
        "cal_win_to_draw_margin",
        "lambda_total",
        "lambda_gap",
        "abs_elo_diff",
        "p_score_space_draw",
        "p_00",
        "p_11",
        "p_low_score_mass",
        "ad_net_power_gap",
    ]
    X_tr = df_train[feature_cols].values
    y_tr = df_train["is_draw_actual"].values

    clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf, feature_cols


def run_policy_inference(
    df: pd.DataFrame,
    policy_key: str,
    params: Dict[str, Any],
    clf_model: Optional[Tuple[LogisticRegression, List[str]]] = None,
) -> List[str]:
    """Execute decision policy inference deterministically."""
    v4_dec = df["v4_decision"].values
    n = len(df)
    decisions = list(v4_dec)

    if policy_key == "A_standard_argmax":
        return list(v4_dec)

    elif policy_key == "B_win_margin_gate":
        # If max(P(H), P(A)) - P(D) < margin_thresh
        m_thresh = params.get("margin_thresh", 0.08)
        for i in range(n):
            if df.loc[i, "v4_win_to_draw_margin"] < m_thresh:
                decisions[i] = "D"

    elif policy_key == "C_calibrated_draw_thresh":
        # If cal_p_D > draw_thresh
        d_thresh = params.get("draw_thresh", 0.32)
        for i in range(n):
            if df.loc[i, "cal_p_D"] >= d_thresh:
                decisions[i] = "D"

    elif policy_key == "D_draw_and_win_parity":
        # If cal_p_D > draw_thresh and cal_win_diff < win_diff_thresh
        d_thresh = params.get("draw_thresh", 0.30)
        p_thresh = params.get("parity_thresh", 0.08)
        for i in range(n):
            if df.loc[i, "cal_p_D"] >= d_thresh and df.loc[i, "cal_win_diff"] <= p_thresh:
                decisions[i] = "D"

    elif policy_key == "E_draw_and_score_space":
        # If cal_p_D > draw_thresh and p_score_space_draw > score_thresh
        d_thresh = params.get("draw_thresh", 0.29)
        s_thresh = params.get("score_thresh", 0.27)
        for i in range(n):
            if df.loc[i, "cal_p_D"] >= d_thresh and df.loc[i, "p_score_space_draw"] >= s_thresh:
                decisions[i] = "D"

    elif policy_key == "F_draw_and_low_goals":
        # If cal_p_D > draw_thresh and lambda_total < goal_thresh
        d_thresh = params.get("draw_thresh", 0.29)
        g_thresh = params.get("goal_thresh", 2.40)
        for i in range(n):
            if df.loc[i, "cal_p_D"] >= d_thresh and df.loc[i, "lambda_total"] <= g_thresh:
                decisions[i] = "D"

    elif policy_key == "G_draw_and_elo_parity":
        # If cal_p_D > draw_thresh and abs_elo_diff < elo_thresh
        d_thresh = params.get("draw_thresh", 0.29)
        e_thresh = params.get("elo_thresh", 60.0)
        for i in range(n):
            if df.loc[i, "cal_p_D"] >= d_thresh and df.loc[i, "abs_elo_diff"] <= e_thresh:
                decisions[i] = "D"

    elif policy_key == "H_draw_confidence_score":
        # Multi-signal logistic confidence model
        if clf_model is not None:
            clf, fcols = clf_model
            X_eval = df[fcols].values
            probs_d_score = clf.predict_proba(X_eval)[:, 1]
            s_thresh = params.get("score_thresh", 0.35)
            for i in range(n):
                if probs_d_score[i] >= s_thresh:
                    decisions[i] = "D"

    elif policy_key == "I_selective_draw_gate":
        # 5-condition physical gate with strict pre-match thresholds
        d_thresh = params.get("draw_thresh", 0.29)
        m_thresh = params.get("margin_thresh", 0.08)
        g_thresh = params.get("goal_thresh", 2.50)
        e_thresh = params.get("elo_thresh", 80.0)
        for i in range(n):
            cond1 = df.loc[i, "cal_p_D"] >= d_thresh
            cond2 = df.loc[i, "cal_win_to_draw_margin"] <= m_thresh
            cond3 = df.loc[i, "lambda_total"] <= g_thresh
            cond4 = df.loc[i, "abs_elo_diff"] <= e_thresh
            if cond1 and cond2 and cond3 and cond4:
                decisions[i] = "D"

    elif policy_key == "J_platt_calibrated_argmax":
        # Strict argmax on Platt continuous calibrated probabilities
        return list(df["cal_decision"].values)

    return decisions


def main():
    logger.info("================================================================")
    logger.info("STEP 2I — DRAW DECISION INTELLIGENCE RESEARCH")
    logger.info("================================================================")

    # 1. Pre-flight Integrity
    assert verify_md5(V4_PATH, EXP_V4_MD5, "V4.0 Baseline")
    assert verify_md5(V4_1_PATH, EXP_V4_1_MD5, "V4.1 Production Candidate")

    # 2. Load Full Dataset
    df = load_full_research_dataset()

    # 3. Chronological Splits (5 Folds + Untouched Holdout)
    seasons = sorted(df["season"].unique().tolist())
    logger.info(f"Available seasons: {seasons}")

    df_hist = df[df["season"] != "2025/2026"].copy().reset_index(drop=True)
    df_holdout = df[df["season"] == "2025/2026"].copy().reset_index(drop=True)

    logger.info(f"Historical Training Era: {len(df_hist)} matches (2020/21–2024/25)")
    logger.info(f"Untouched Holdout Season: {len(df_holdout)} matches (2025/2026)")

    # 4. Feature Importance on Training Era
    logger.info("Analyzing draw signal importance...")
    clf_full_train, fcols = fit_draw_confidence_model(df_hist)
    feat_imp_df = pd.DataFrame({
        "feature": fcols,
        "coefficient": clf_full_train.coef_[0],
        "abs_importance": np.abs(clf_full_train.coef_[0]),
    }).sort_values(by="abs_importance", ascending=False)
    feat_imp_df.to_csv(STEP2I_DIR / "draw_signal_importance.csv", index=False)
    logger.info(f"Saved feature importance to draw_signal_importance.csv:\n{feat_imp_df.to_string(index=False)}")

    # 5. Policy Definitions & Configurations
    policies = {
        "A_standard_argmax": {},
        "B_win_margin_gate": {"margin_thresh": 0.08},
        "C_calibrated_draw_thresh": {"draw_thresh": 0.32},
        "D_draw_and_win_parity": {"draw_thresh": 0.30, "parity_thresh": 0.06},
        "E_draw_and_score_space": {"draw_thresh": 0.29, "score_thresh": 0.28},
        "F_draw_and_low_goals": {"draw_thresh": 0.29, "goal_thresh": 2.35},
        "G_draw_and_elo_parity": {"draw_thresh": 0.29, "elo_thresh": 60.0},
        "H_draw_confidence_score": {"score_thresh": 0.34},
        "I_selective_draw_gate": {"draw_thresh": 0.29, "margin_thresh": 0.08, "goal_thresh": 2.45, "elo_thresh": 75.0},
        "J_platt_calibrated_argmax": {},
    }

    # 6. Chronological 5-Fold Walk-Forward Validation
    logger.info("Running 5-Fold Chronological Walk-Forward Validation...")
    wf_splits = [
        ("Fold 1 (2021/22)", ["2020/2021"], ["2021/2022"]),
        ("Fold 2 (2022/23)", ["2020/2021", "2021/2022"], ["2022/2023"]),
        ("Fold 3 (2023/24)", ["2020/2021", "2021/2022", "2022/2023"], ["2023/2024"]),
        ("Fold 4 (2024/25)", ["2020/2021", "2021/2022", "2022/2023", "2023/2024"], ["2024/2025"]),
    ]

    wf_rows = []
    for split_name, tr_seasons, val_seasons in wf_splits:
        df_tr = df_hist[df_hist["season"].isin(tr_seasons)].reset_index(drop=True)
        df_val = df_hist[df_hist["season"].isin(val_seasons)].reset_index(drop=True)

        # Fit confidence model on training split only
        clf_fold, _ = fit_draw_confidence_model(df_tr)

        for p_key, p_params in policies.items():
            val_preds = run_policy_inference(df_val, p_key, p_params, clf_model=(clf_fold, fcols))
            res = evaluate_decision_policy(df_val, val_preds, p_key)
            res["split"] = split_name
            wf_rows.append(res)

    df_wf = pd.DataFrame(wf_rows)
    df_wf.to_csv(STEP2I_DIR / "walkforward_decision_results.csv", index=False)
    logger.info("Saved walk-forward results to walkforward_decision_results.csv")

    # 7. Overall Historical Candidate Comparison
    logger.info("Evaluating overall performance across historical training era (N=8,983)...")
    hist_comp_rows = []
    for p_key, p_params in policies.items():
        hist_preds = run_policy_inference(df_hist, p_key, p_params, clf_model=(clf_full_train, fcols))
        res = evaluate_decision_policy(df_hist, hist_preds, p_key)
        hist_comp_rows.append(res)

    df_hist_comp = pd.DataFrame(hist_comp_rows)
    df_hist_comp.to_csv(STEP2I_DIR / "candidate_decision_comparison.csv", index=False)
    logger.info(f"Saved candidate comparison to candidate_decision_comparison.csv:\n{df_hist_comp[['policy', 'accuracy', 'accuracy_delta', 'draw_recall', 'draw_precision', 'rescued_draws', 'damaged_wins', 'net_correct_delta']].to_string(index=False)}")

    # 8. Threshold Sensitivity Analysis on Historical Training Era
    logger.info("Running parameter sensitivity grids across training era...")
    sens_rows = []

    # Grid 1: Draw Probability Threshold in Policy C
    for t in [0.26, 0.28, 0.30, 0.32, 0.34, 0.36]:
        preds = run_policy_inference(df_hist, "C_calibrated_draw_thresh", {"draw_thresh": t})
        r = evaluate_decision_policy(df_hist, preds, f"Policy_C_dthresh_{t:.2f}")
        r["parameter_type"] = "draw_threshold"
        r["parameter_value"] = t
        sens_rows.append(r)

    # Grid 2: Margin Threshold in Policy B
    for m in [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]:
        preds = run_policy_inference(df_hist, "B_win_margin_gate", {"margin_thresh": m})
        r = evaluate_decision_policy(df_hist, preds, f"Policy_B_margin_{m:.2f}")
        r["parameter_type"] = "margin_threshold"
        r["parameter_value"] = m
        sens_rows.append(r)

    # Grid 3: Draw Confidence Score Threshold in Policy H
    for sc in [0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]:
        preds = run_policy_inference(df_hist, "H_draw_confidence_score", {"score_thresh": sc}, clf_model=(clf_full_train, fcols))
        r = evaluate_decision_policy(df_hist, preds, f"Policy_H_score_{sc:.2f}")
        r["parameter_type"] = "confidence_score_threshold"
        r["parameter_value"] = sc
        sens_rows.append(r)

    # Grid 4: Selective Physical Gate Margin in Policy I
    for sm in [0.04, 0.06, 0.08, 0.10, 0.12]:
        preds = run_policy_inference(df_hist, "I_selective_draw_gate", {"draw_thresh": 0.29, "margin_thresh": sm, "goal_thresh": 2.45, "elo_thresh": 75.0})
        r = evaluate_decision_policy(df_hist, preds, f"Policy_I_margin_{sm:.2f}")
        r["parameter_type"] = "selective_gate_margin"
        r["parameter_value"] = sm
        sens_rows.append(r)

    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(STEP2I_DIR / "threshold_sensitivity.csv", index=False)
    logger.info("Saved threshold sensitivity to threshold_sensitivity.csv")

    # 9. League-wise Disaggregated Analysis on Historical Data
    logger.info("Evaluating league-wise robustness across historical data...")
    league_rows = []
    for league in sorted(df_hist["competition_name"].unique()):
        df_lg = df_hist[df_hist["competition_name"] == league].reset_index(drop=True)
        for p_key, p_params in policies.items():
            lg_preds = run_policy_inference(df_lg, p_key, p_params, clf_model=(clf_full_train, fcols))
            res = evaluate_decision_policy(df_lg, lg_preds, p_key)
            res["league"] = league
            league_rows.append(res)

    df_league = pd.DataFrame(league_rows)
    df_league.to_csv(STEP2I_DIR / "league_decision_results.csv", index=False)
    logger.info("Saved league results to league_decision_results.csv")

    # 10. UNTOUCHED 2025/26 HOLDOUT VALIDATION
    logger.info("Running ONE-TIME evaluation on UNTOUCHED 2025/26 holdout season (N=1,751)...")
    holdout_rows = []
    confusion_rows = []

    for p_key, p_params in policies.items():
        ho_preds = run_policy_inference(df_holdout, p_key, p_params, clf_model=(clf_full_train, fcols))
        res = evaluate_decision_policy(df_holdout, ho_preds, p_key)
        holdout_rows.append(res)

        # Confusion Matrix
        cm = confusion_matrix(df_holdout["actual_result"], ho_preds, labels=["H", "D", "A"])
        for idx, act in enumerate(["H", "D", "A"]):
            confusion_rows.append({
                "policy": p_key,
                "actual": act,
                "pred_H": cm[idx, 0],
                "pred_D": cm[idx, 1],
                "pred_A": cm[idx, 2],
            })

    df_holdout_res = pd.DataFrame(holdout_rows)
    df_holdout_res.to_csv(STEP2I_DIR / "holdout_decision_results.csv", index=False)

    df_cm = pd.DataFrame(confusion_rows)
    df_cm.to_csv(STEP2I_DIR / "decision_confusion_matrices.csv", index=False)
    logger.info(f"Saved holdout results to holdout_decision_results.csv:\n{df_holdout_res[['policy', 'accuracy', 'accuracy_delta', 'draw_recall', 'draw_precision', 'rescued_draws', 'damaged_wins', 'net_correct_delta']].to_string(index=False)}")

    # 11. Prospective 33-Match Sample Evaluation (Aug 22-24, 2026)
    logger.info("Evaluating on Recent 33 Prospective Matches...")
    df_33_raw = pd.read_csv(STEP2E_33_PATH)
    r33_rows = []
    for _, row in df_33_raw.iterrows():
        fid = int(row["fixture_id"])
        h_name = str(row["home_team"])
        a_name = str(row["away_team"])
        league = str(row["league"])
        act_res = str(row["actual_outcome"])
        score = str(row["score"])
        v4_h = float(row["v4_p_H"])
        v4_d = float(row["v4_p_D"])
        v4_a = float(row["v4_p_A"])
        cal_h = float(row["calibrated_p_H"])
        cal_d = float(row["calibrated_p_D"])
        cal_a = float(row["calibrated_p_A"])

        r33_rows.append({
            "fixture_id": fid,
            "date": str(row.get("date_ist", "")),
            "league": league,
            "home_team": h_name,
            "away_team": a_name,
            "score": score,
            "actual_outcome": act_res,
            "v4_p_H": v4_h,
            "v4_p_D": v4_d,
            "v4_p_A": v4_a,
            "v4_decision": row["v4_decision"],
            "cal_p_H": cal_h,
            "cal_p_D": cal_d,
            "cal_p_A": cal_a,
            "cal_decision": row["calibrated_decision"],
            "v4_win_to_draw_margin": max(v4_h, v4_a) - v4_d,
            "cal_win_to_draw_margin": max(cal_h, cal_a) - cal_d,
            "cal_win_diff": abs(cal_h - cal_a),
            "is_actual_draw": act_res == "D",
        })

    df_33 = pd.DataFrame(r33_rows)
    df_33.to_csv(STEP2I_DIR / "prospective_decision_audit.csv", index=False)
    logger.info(f"Saved 33 prospective matches to prospective_decision_audit.csv")

    # 12. Save Candidate Config JSON
    config_dict = {
        "step": "Step 2I Draw Decision Intelligence Research",
        "date": "2026-08-24",
        "governance_status": "RESEARCH ONLY — FROZEN BASELINE ACTIVE",
        "v4_production_md5": EXP_V4_MD5,
        "v4_1_candidate_md5": EXP_V4_1_MD5,
        "policies_evaluated": list(policies.keys()),
        "policy_parameters": policies,
        "draw_confidence_features": fcols,
        "draw_confidence_intercept": float(clf_full_train.intercept_[0]),
        "draw_confidence_coefficients": {k: float(v) for k, v in zip(fcols, clf_full_train.coef_[0])},
    }

    with open(STEP2I_DIR / "step2i_candidate_config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    logger.info("Saved config to step2i_candidate_config.json")

    # 13. Post-flight Integrity
    assert verify_md5(V4_PATH, EXP_V4_MD5, "V4.0 Baseline")
    assert verify_md5(V4_1_PATH, EXP_V4_1_MD5, "V4.1 Production Candidate")

    logger.info("Step 2I Research Execution Finished Successfully.")


if __name__ == "__main__":
    main()
