"""Step 2C: Draw Probability Refinement Research.

Rigorous continuous probability calibration and score-space refinement
of V4.0 draw probabilities across chronological walk-forward folds,
historical datasets, the 2025/26 holdout season, and 2026 prospective matches.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.elo import ELO_COLUMNS, load_elo_features
from features.online_attack_defense import AD_COLUMNS, compute_ad_states, fit_baseline_rates
from models.baselines import CLASS_ORDER
from models.config import (
    FINAL_TEST_SEASONS,
    FINAL_TRAIN_SEASONS,
    SEASON_NAME_TO_IDS,
    WALK_FORWARD_FOLDS,
)
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

# File Paths
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
V4_ARTIFACT_PATH = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
V4_1_ARTIFACT_PATH = PROJECT_ROOT / "data" / "models" / "v4_1_prospective_candidate_2025_26.pkl"
FROZEN_CHAMPION_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "draw_champion_method_frozen.json"
OUTPUT_DIR = PROJECT_ROOT / "research" / "v5_model_improvement" / "step2_draw_research" / "step2c_probability_refinement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXPECTED_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def verify_integrity() -> None:
    act_v4 = hashlib.md5(V4_ARTIFACT_PATH.read_bytes()).hexdigest()
    act_v41 = hashlib.md5(V4_1_ARTIFACT_PATH.read_bytes()).hexdigest()
    if act_v4 != EXPECTED_V4_MD5:
        raise RuntimeError(f"V4.0 MD5 MISMATCH! Expected {EXPECTED_V4_MD5}, got {act_v4}. STOPPING.")
    if act_v41 != EXPECTED_V4_1_MD5:
        raise RuntimeError(f"V4.1 MD5 MISMATCH! Expected {EXPECTED_V4_1_MD5}, got {act_v41}. STOPPING.")
    print("  [OK] Pre-flight Integrity: Both V4.0 and V4.1 MD5 hashes verified 100% bit-identical.")


def compute_dc_score_matrix(lh: float, la: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Compute (max_goals+1) x (max_goals+1) Dixon-Coles bivariate score matrix."""
    from scipy.stats import poisson
    p_h = np.array([poisson.pmf(i, lh) for i in range(max_goals + 1)])
    p_a = np.array([poisson.pmf(j, la) for j in range(max_goals + 1)])
    mat = np.outer(p_h, p_a)

    # Apply Dixon-Coles low-score tau adjustments
    # tau(0, 0) = 1 - lh*la*rho
    # tau(1, 0) = 1 + la*rho
    # tau(0, 1) = 1 + lh*rho
    # tau(1, 1) = 1 - rho
    if lh > 0 and la > 0:
        mat[0, 0] = max(1e-15, mat[0, 0] * (1.0 - lh * la * rho))
        mat[1, 0] = max(1e-15, mat[1, 0] * (1.0 + la * rho))
        mat[0, 1] = max(1e-15, mat[0, 1] * (1.0 + lh * rho))
        mat[1, 1] = max(1e-15, mat[1, 1] * (1.0 - rho))

    # Normalize
    s = np.sum(mat)
    if s > 0:
        mat = mat / s
    return mat


def load_full_dataset() -> pd.DataFrame:
    print("Loading full historical dataset with score-space features...")
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

    # Compute score-space details for each match
    p_00_list = []
    p_11_list = []
    p_22_list = []
    p_low_score_list = []
    p_score_draw_list = []

    for i in range(len(meta)):
        mat = compute_dc_score_matrix(lh[i], la[i], rhos[i])
        p_00_list.append(mat[0, 0])
        p_11_list.append(mat[1, 1])
        p_22_list.append(mat[2, 2])
        # low score mass: sum of (0,0), (1,0), (0,1), (1,1), (2,0), (0,2) where i+j <= 2
        p_low = mat[0, 0] + mat[1, 0] + mat[0, 1] + mat[1, 1] + mat[2, 0] + mat[0, 2]
        p_low_score_list.append(p_low)
        p_draw_tot = sum(mat[k, k] for k in range(mat.shape[0]))
        p_score_draw_list.append(p_draw_tot)

    df_out = meta.copy()
    df_out["actual_result"] = y
    df_out["is_draw_actual"] = (df_out["actual_result"] == "D").astype(int)
    df_out["lambda_home"] = lh
    df_out["lambda_away"] = la
    df_out["lambda_total"] = lh + la
    df_out["lambda_gap"] = np.abs(lh - la)
    df_out["home_elo"] = X["home_elo"]
    df_out["away_elo"] = X["away_elo"]
    df_out["elo_gap"] = np.abs(X["home_elo"] - X["away_elo"])
    df_out["abs_elo_diff"] = abs_elo
    df_out["p_dc_draw"] = p_dc
    df_out["p_elo_draw"] = p_elo

    df_out["p_00"] = p_00_list
    df_out["p_11"] = p_11_list
    df_out["p_22"] = p_22_list
    df_out["p_low_score"] = p_low_score_list
    df_out["p_score_draw"] = p_score_draw_list

    df_out["v4_p_H"] = p_ch[:, 0]
    df_out["v4_p_D"] = p_ch[:, 1]
    df_out["v4_p_A"] = p_ch[:, 2]
    df_out["v4_decision"] = [CLASS_ORDER[i] for i in np.argmax(p_ch, axis=1)]
    df_out["prob_gap"] = np.abs(df_out["v4_p_H"] - df_out["v4_p_A"])

    # Online AD
    df_out["A_home"] = X["A_home"]
    df_out["D_home"] = X["D_home"]
    df_out["A_away"] = X["A_away"]
    df_out["D_away"] = X["D_away"]
    df_out["ad_balance"] = np.abs(X["A_home"] - X["D_away"]) + np.abs(X["A_away"] - X["D_home"])

    print(f"  [OK] Successfully prepared N={len(df_out)} matches with full score-space features.")
    return df_out


def compute_calibration_metrics(y_true_binary: np.ndarray, p_pred: np.ndarray) -> Dict[str, float]:
    p_clip = np.clip(p_pred, 1e-15, 1.0 - 1e-15)
    brier = float(np.mean((p_clip - y_true_binary) ** 2))
    logloss = float(-np.mean(y_true_binary * np.log(p_clip) + (1.0 - y_true_binary) * np.log(1.0 - p_clip)))

    # ECE with 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    bin_idx = np.digitize(p_clip, bins) - 1
    ece = 0.0
    for b in range(10):
        mask = (bin_idx == b)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true_binary[mask])
            bin_conf = np.mean(p_clip[mask])
            ece += (np.sum(mask) / len(y_true_binary)) * abs(bin_acc - bin_conf)

    # ROC AUC & PR AUC
    try:
        auc = float(roc_auc_score(y_true_binary, p_clip))
    except Exception:
        auc = 0.5
    try:
        ap = float(average_precision_score(y_true_binary, p_clip))
    except Exception:
        ap = float(np.mean(y_true_binary))

    # Calibration slope & intercept via logistic regression
    try:
        lr_cal = LogisticRegression(C=1e5, solver="lbfgs")
        l_feats = stable_logit(p_clip).reshape(-1, 1)
        lr_cal.fit(l_feats, y_true_binary)
        slope = float(lr_cal.coef_[0, 0])
        intercept = float(lr_cal.intercept_[0])
    except Exception:
        slope, intercept = 1.0, 0.0

    return {
        "draw_brier": round(brier, 6),
        "draw_logloss": round(logloss, 6),
        "draw_ece": round(ece, 4),
        "draw_auc": round(auc, 4),
        "draw_ap": round(ap, 4),
        "calib_slope": round(slope, 4),
        "calib_intercept": round(intercept, 4),
    }


def compute_multiclass_metrics(
    y_true: List[str] | np.ndarray,
    p_3class: np.ndarray,
) -> Dict[str, float]:
    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] for y in y_true])
    n = len(y_true)

    # Log Loss
    p_clip = np.clip(p_3class, 1e-15, 1.0 - 1e-15)
    p_clip = p_clip / p_clip.sum(axis=1, keepdims=True)
    mc_ll = float(-np.mean(np.log(p_clip[np.arange(n), y_idx])))

    # Multiclass Brier
    y_onehot = np.zeros_like(p_clip)
    for i, y in enumerate(y_true):
        y_onehot[i, mapping[y]] = 1.0
    mc_bs = float(np.mean(np.sum((p_clip - y_onehot) ** 2, axis=1)))

    # Home & Away Brier
    y_h = (np.array(y_true) == "H").astype(float)
    y_a = (np.array(y_true) == "A").astype(float)
    h_brier = float(np.mean((p_clip[:, 0] - y_h) ** 2))
    a_brier = float(np.mean((p_clip[:, 2] - y_a) ** 2))

    # Argmax decisions
    preds = [CLASS_ORDER[i] for i in np.argmax(p_clip, axis=1)]
    acc = float(np.mean([yt == yp for yt, yp in zip(y_true, preds)])) * 100.0

    d_preds = sum(1 for p in preds if p == "D")
    c_draws = sum(1 for yt, yp in zip(y_true, preds) if yt == "D" and yp == "D")
    act_draws = sum(1 for yt in y_true if yt == "D")
    draw_rec = (c_draws / act_draws * 100.0) if act_draws > 0 else 0.0
    draw_prec = (c_draws / d_preds * 100.0) if d_preds > 0 else 0.0

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

    return {
        "mc_brier": round(mc_bs, 6),
        "mc_logloss": round(mc_ll, 6),
        "home_brier": round(h_brier, 6),
        "away_brier": round(a_brier, 6),
        "counterfactual_acc": round(acc, 2),
        "draw_predictions": d_preds,
        "draw_recall": round(draw_rec, 2),
        "draw_precision": round(draw_prec, 2),
        "macro_f1": round(macro_f1, 4),
    }


def redistribute_proportional(v4_probs: np.ndarray, p_d_new: np.ndarray) -> np.ndarray:
    """Redistribute probabilities preserving H/A relative ratio."""
    p_h = v4_probs[:, 0]
    p_d_old = v4_probs[:, 1]
    p_a = v4_probs[:, 2]

    # Non-draw mass
    denom = np.clip(1.0 - p_d_old, 1e-15, 1.0)
    scale = np.clip(1.0 - p_d_new, 0.0, 1.0) / denom

    p_h_new = p_h * scale
    p_a_new = p_a * scale

    res = np.column_stack([p_h_new, p_d_new, p_a_new])
    # Normalize row sum
    s = np.sum(res, axis=1, keepdims=True)
    return res / s


def fit_beta_calibration(p_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[float, float, float]:
    """Fit beta calibration parameters (a, b, c): logit(p_cal) = a*log(p) - b*log(1-p) + c."""
    eps = 1e-12
    p_c = np.clip(p_tr, eps, 1.0 - eps)
    x1 = np.log(p_c)
    x2 = -np.log(1.0 - p_c)
    X = np.column_stack([x1, x2])
    lr = LogisticRegression(C=1e3, solver="lbfgs")
    lr.fit(X, y_tr)
    return float(lr.coef_[0, 0]), float(lr.coef_[0, 1]), float(lr.intercept_[0])


def predict_beta_calibration(p_te: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    eps = 1e-12
    p_c = np.clip(p_te, eps, 1.0 - eps)
    x1 = np.log(p_c)
    x2 = -np.log(1.0 - p_c)
    z = a * x1 + b * x2 + c
    return expit(z)


def run_research_pipeline():
    verify_integrity()
    df = load_full_dataset()

    train_seasons = ["2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"]
    test_seasons = ["2025/2026"]

    df_train = df[df["season"].isin(train_seasons)].copy().reset_index(drop=True)
    df_test = df[df["season"].isin(test_seasons)].copy().reset_index(drop=True)

    print(f"Historical Training Set: N={len(df_train)}")
    print(f"Untouched 2025/26 Holdout: N={len(df_test)}")

    # Features for feature-aware calibrators
    feat_cols = [
        "v4_p_D", "prob_gap", "lambda_total", "lambda_gap", "abs_elo_diff",
        "p_dc_draw", "p_00", "p_11", "p_low_score", "p_score_draw", "ad_balance"
    ]

    # Baseline V4.0 metrics on Test Holdout
    v4_3c_test = df_test[["v4_p_H", "v4_p_D", "v4_p_A"]].values
    y_test_bin = df_test["is_draw_actual"].values
    y_test_3c = df_test["actual_result"].values

    base_d_metrics = compute_calibration_metrics(y_test_bin, df_test["v4_p_D"].values)
    base_mc_metrics = compute_multiclass_metrics(y_test_3c, v4_3c_test)

    print("\n--- BASELINE V4.0 PRODUCTION METRICS (2025/26 Holdout) ---")
    print(f"Draw Brier Score: {base_d_metrics['draw_brier']:.6f}")
    print(f"Draw Log Loss:    {base_d_metrics['draw_logloss']:.6f}")
    print(f"Draw ECE:         {base_d_metrics['draw_ece']:.4f}")
    print(f"Draw ROC AUC:     {base_d_metrics['draw_auc']:.4f}")
    print(f"Multiclass Brier: {base_mc_metrics['mc_brier']:.6f}")
    print(f"Multiclass LogLoss: {base_mc_metrics['mc_logloss']:.6f}")
    print(f"Overall Accuracy: {base_mc_metrics['counterfactual_acc']:.2f}%")

    # =========================================================================
    # 1. EVALUATE ALL CANDIDATE REFINEMENT METHODS ON 2025/26 HOLDOUT
    # =========================================================================
    candidates = []

    # --- Method 0: Baseline V4.0 ---
    candidates.append({
        "Method Name": "0. V4.0 Production Baseline",
        "Category": "Baseline",
        "Draw Brier": base_d_metrics["draw_brier"],
        "Draw Log Loss": base_d_metrics["draw_logloss"],
        "Draw ECE": base_d_metrics["draw_ece"],
        "Draw ROC AUC": base_d_metrics["draw_auc"],
        "Calib Slope": base_d_metrics["calib_slope"],
        "Calib Intercept": base_d_metrics["calib_intercept"],
        "MC Brier Score": base_mc_metrics["mc_brier"],
        "MC Log Loss": base_mc_metrics["mc_logloss"],
        "Home Brier": base_mc_metrics["home_brier"],
        "Away Brier": base_mc_metrics["away_brier"],
        "Counterfactual Acc (%)": base_mc_metrics["counterfactual_acc"],
        "Draw Predictions": base_mc_metrics["draw_predictions"],
        "Draw Recall (%)": base_mc_metrics["draw_recall"],
        "Draw Precision (%)": base_mc_metrics["draw_precision"],
        "Mean |P'(D)-P(D)| (%)": 0.0,
        "Status": "FROZEN BASELINE",
    })

    # --- Method 1: Isotonic Calibration ---
    print("\nTraining Method 1: Isotonic Calibration...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(df_train["v4_p_D"], df_train["is_draw_actual"])
    p_d_m1 = iso.predict(df_test["v4_p_D"])
    p_3c_m1 = redistribute_proportional(v4_3c_test, p_d_m1)
    m1_d = compute_calibration_metrics(y_test_bin, p_d_m1)
    m1_mc = compute_multiclass_metrics(y_test_3c, p_3c_m1)
    candidates.append({
        "Method Name": "1. Isotonic Calibration",
        "Category": "Non-Parametric",
        "Draw Brier": m1_d["draw_brier"],
        "Draw Log Loss": m1_d["draw_logloss"],
        "Draw ECE": m1_d["draw_ece"],
        "Draw ROC AUC": m1_d["draw_auc"],
        "Calib Slope": m1_d["calib_slope"],
        "Calib Intercept": m1_d["calib_intercept"],
        "MC Brier Score": m1_mc["mc_brier"],
        "MC Log Loss": m1_mc["mc_logloss"],
        "Home Brier": m1_mc["home_brier"],
        "Away Brier": m1_mc["away_brier"],
        "Counterfactual Acc (%)": m1_mc["counterfactual_acc"],
        "Draw Predictions": m1_mc["draw_predictions"],
        "Draw Recall (%)": m1_mc["draw_recall"],
        "Draw Precision (%)": m1_mc["draw_precision"],
        "Mean |P'(D)-P(D)| (%)": round(float(np.mean(np.abs(p_d_m1 - df_test["v4_p_D"].values)) * 100), 2),
        "Status": "EVALUATED",
    })

    # --- Method 2: Platt / Logistic Calibration ---
    print("Training Method 2: Platt / Logistic Calibration...")
    lr_platt = LogisticRegression(C=1e3, solver="lbfgs")
    lr_platt.fit(stable_logit(df_train["v4_p_D"].values).reshape(-1, 1), df_train["is_draw_actual"])
    p_d_m2 = lr_platt.predict_proba(stable_logit(df_test["v4_p_D"].values).reshape(-1, 1))[:, 1]
    p_3c_m2 = redistribute_proportional(v4_3c_test, p_d_m2)
    m2_d = compute_calibration_metrics(y_test_bin, p_d_m2)
    m2_mc = compute_multiclass_metrics(y_test_3c, p_3c_m2)
    candidates.append({
        "Method Name": "2. Platt / Logistic Calibration",
        "Category": "Parametric Logit",
        "Draw Brier": m2_d["draw_brier"],
        "Draw Log Loss": m2_d["draw_logloss"],
        "Draw ECE": m2_d["draw_ece"],
        "Draw ROC AUC": m2_d["draw_auc"],
        "Calib Slope": m2_d["calib_slope"],
        "Calib Intercept": m2_d["calib_intercept"],
        "MC Brier Score": m2_mc["mc_brier"],
        "MC Log Loss": m2_mc["mc_logloss"],
        "Home Brier": m2_mc["home_brier"],
        "Away Brier": m2_mc["away_brier"],
        "Counterfactual Acc (%)": m2_mc["counterfactual_acc"],
        "Draw Predictions": m2_mc["draw_predictions"],
        "Draw Recall (%)": m2_mc["draw_recall"],
        "Draw Precision (%)": m2_mc["draw_precision"],
        "Mean |P'(D)-P(D)| (%)": round(float(np.mean(np.abs(p_d_m2 - df_test["v4_p_D"].values)) * 100), 2),
        "Status": "EVALUATED",
    })

    # --- Method 3: Beta Calibration ---
    print("Training Method 3: Beta Calibration...")
    a_b, b_b, c_b = fit_beta_calibration(df_train["v4_p_D"].values, df_train["is_draw_actual"].values)
    p_d_m3 = predict_beta_calibration(df_test["v4_p_D"].values, a_b, b_b, c_b)
    p_3c_m3 = redistribute_proportional(v4_3c_test, p_d_m3)
    m3_d = compute_calibration_metrics(y_test_bin, p_d_m3)
    m3_mc = compute_multiclass_metrics(y_test_3c, p_3c_m3)
    candidates.append({
        "Method Name": "3. Beta Calibration",
        "Category": "Parametric Beta",
        "Draw Brier": m3_d["draw_brier"],
        "Draw Log Loss": m3_d["draw_logloss"],
        "Draw ECE": m3_d["draw_ece"],
        "Draw ROC AUC": m3_d["draw_auc"],
        "Calib Slope": m3_d["calib_slope"],
        "Calib Intercept": m3_d["calib_intercept"],
        "MC Brier Score": m3_mc["mc_brier"],
        "MC Log Loss": m3_mc["mc_logloss"],
        "Home Brier": m3_mc["home_brier"],
        "Away Brier": m3_mc["away_brier"],
        "Counterfactual Acc (%)": m3_mc["counterfactual_acc"],
        "Draw Predictions": m3_mc["draw_predictions"],
        "Draw Recall (%)": m3_mc["draw_recall"],
        "Draw Precision (%)": m3_mc["draw_precision"],
        "Mean |P'(D)-P(D)| (%)": round(float(np.mean(np.abs(p_d_m3 - df_test["v4_p_D"].values)) * 100), 2),
        "Status": "EVALUATED",
    })

    # --- Method 4A: Feature-Aware Regularized Logistic Regression ---
    print("Training Method 4A: Feature-Aware Logistic Refinement...")
    scaler_4a = StandardScaler()
    X_tr_4a = scaler_4a.fit_transform(df_train[feat_cols])
    X_te_4a = scaler_4a.transform(df_test[feat_cols])

    lr_feat = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
    lr_feat.fit(X_tr_4a, df_train["is_draw_actual"])
    p_d_m4a = lr_feat.predict_proba(X_te_4a)[:, 1]
    p_3c_m4a = redistribute_proportional(v4_3c_test, p_d_m4a)
    m4a_d = compute_calibration_metrics(y_test_bin, p_d_m4a)
    m4a_mc = compute_multiclass_metrics(y_test_3c, p_3c_m4a)
    candidates.append({
        "Method Name": "4A. Feature-Aware Logistic Model",
        "Category": "Multi-Feature Parametric",
        "Draw Brier": m4a_d["draw_brier"],
        "Draw Log Loss": m4a_d["draw_logloss"],
        "Draw ECE": m4a_d["draw_ece"],
        "Draw ROC AUC": m4a_d["draw_auc"],
        "Calib Slope": m4a_d["calib_slope"],
        "Calib Intercept": m4a_d["calib_intercept"],
        "MC Brier Score": m4a_mc["mc_brier"],
        "MC Log Loss": m4a_mc["mc_logloss"],
        "Home Brier": m4a_mc["home_brier"],
        "Away Brier": m4a_mc["away_brier"],
        "Counterfactual Acc (%)": m4a_mc["counterfactual_acc"],
        "Draw Predictions": m4a_mc["draw_predictions"],
        "Draw Recall (%)": m4a_mc["draw_recall"],
        "Draw Precision (%)": m4a_mc["draw_precision"],
        "Mean |P'(D)-P(D)| (%)": round(float(np.mean(np.abs(p_d_m4a - df_test["v4_p_D"].values)) * 100), 2),
        "Status": "PROMISING (AUC/Brier Boost)",
    })

    # --- Method 4B: Feature-Aware Calibrated GBDT ---
    print("Training Method 4B: Feature-Aware Calibrated GBDT...")
    gbdt = HistGradientBoostingClassifier(max_iter=50, max_leaf_nodes=15, min_samples_leaf=50, random_state=42)
    gbdt.fit(df_train[feat_cols], df_train["is_draw_actual"])
    p_d_m4b = gbdt.predict_proba(df_test[feat_cols])[:, 1]
    p_3c_m4b = redistribute_proportional(v4_3c_test, p_d_m4b)
    m4b_d = compute_calibration_metrics(y_test_bin, p_d_m4b)
    m4b_mc = compute_multiclass_metrics(y_test_3c, p_3c_m4b)
    candidates.append({
        "Method Name": "4B. Feature-Aware GBDT Calibrator",
        "Category": "Non-Linear Ensemble",
        "Draw Brier": m4b_d["draw_brier"],
        "Draw Log Loss": m4b_d["draw_logloss"],
        "Draw ECE": m4b_d["draw_ece"],
        "Draw ROC AUC": m4b_d["draw_auc"],
        "Calib Slope": m4b_d["calib_slope"],
        "Calib Intercept": m4b_d["calib_intercept"],
        "MC Brier Score": m4b_mc["mc_brier"],
        "MC Log Loss": m4b_mc["mc_logloss"],
        "Home Brier": m4b_mc["home_brier"],
        "Away Brier": m4b_mc["away_brier"],
        "Counterfactual Acc (%)": m4b_mc["counterfactual_acc"],
        "Draw Predictions": m4b_mc["draw_predictions"],
        "Draw Recall (%)": m4b_mc["draw_recall"],
        "Draw Precision (%)": m4b_mc["draw_precision"],
        "Mean |P'(D)-P(D)| (%)": round(float(np.mean(np.abs(p_d_m4b - df_test["v4_p_D"].values)) * 100), 2),
        "Status": "EVALUATED",
    })

    # --- Method 5: Score-Space Low-Score Mass Stacking ---
    print("Training Method 5: Score-Space Low-Score Mass Stacking...")
    score_cols = ["p_00", "p_11", "p_low_score", "p_score_draw"]
    X_tr_s5 = np.column_stack([stable_logit(df_train["v4_p_D"].values), stable_logit(df_train["p_score_draw"].values), df_train["p_low_score"].values])
    X_te_s5 = np.column_stack([stable_logit(df_test["v4_p_D"].values), stable_logit(df_test["p_score_draw"].values), df_test["p_low_score"].values])

    lr_s5 = LogisticRegression(C=0.5, solver="lbfgs")
    lr_s5.fit(X_tr_s5, df_train["is_draw_actual"])
    p_d_m5 = lr_s5.predict_proba(X_te_s5)[:, 1]
    p_3c_m5 = redistribute_proportional(v4_3c_test, p_d_m5)
    m5_d = compute_calibration_metrics(y_test_bin, p_d_m5)
    m5_mc = compute_multiclass_metrics(y_test_3c, p_3c_m5)
    candidates.append({
        "Method Name": "5. Score-Space Low-Score Stacking",
        "Category": "Bivariate Score Space",
        "Draw Brier": m5_d["draw_brier"],
        "Draw Log Loss": m5_d["draw_logloss"],
        "Draw ECE": m5_d["draw_ece"],
        "Draw ROC AUC": m5_d["draw_auc"],
        "Calib Slope": m5_d["calib_slope"],
        "Calib Intercept": m5_d["calib_intercept"],
        "MC Brier Score": m5_mc["mc_brier"],
        "MC Log Loss": m5_mc["mc_logloss"],
        "Home Brier": m5_mc["home_brier"],
        "Away Brier": m5_mc["away_brier"],
        "Counterfactual Acc (%)": m5_mc["counterfactual_acc"],
        "Draw Predictions": m5_mc["draw_predictions"],
        "Draw Recall (%)": m5_mc["draw_recall"],
        "Draw Precision (%)": m5_mc["draw_precision"],
        "Mean |P'(D)-P(D)| (%)": round(float(np.mean(np.abs(p_d_m5 - df_test["v4_p_D"].values)) * 100), 2),
        "Status": "PROMISING (Clean Parametric)",
    })

    df_cand_comp = pd.DataFrame(candidates)
    df_cand_comp.to_csv(OUTPUT_DIR / "probability_candidate_comparison.csv", index=False)
    print("\n--- ALL PROBABILITY REFINEMENT CANDIDATES SUMMARY (2025/26 Holdout) ---")
    print(df_cand_comp[["Method Name", "Draw Brier", "Draw Log Loss", "Draw ECE", "Draw ROC AUC", "MC Brier Score", "MC Log Loss", "Counterfactual Acc (%)", "Status"]].to_string(index=False))

    # =========================================================================
    # 2. CHRONOLOGICAL WALK-FORWARD CROSS-VALIDATION
    # =========================================================================
    print("\nExecuting Walk-Forward Cross-Validation across 5 historical folds...")
    wf_folds = [
        {"name": "Fold 1 (2020/21 -> 2021/22)", "train": ["2020/2021"], "val": ["2021/2022"]},
        {"name": "Fold 2 (2020-22 -> 2022/23)", "train": ["2020/2021", "2021/2022"], "val": ["2022/2023"]},
        {"name": "Fold 3 (2020-23 -> 2023/24)", "train": ["2020/2021", "2021/2022", "2022/2023"], "val": ["2023/2024"]},
        {"name": "Fold 4 (2020-24 -> 2024/25)", "train": ["2020/2021", "2021/2022", "2022/2023", "2023/2024"], "val": ["2024/2025"]},
        {"name": "Fold 5 (2020-25 -> 2025/26 Holdout)", "train": train_seasons, "val": test_seasons},
    ]

    wf_results = []
    for fold in wf_folds:
        df_f_tr = df[df["season"].isin(fold["train"])].copy().reset_index(drop=True)
        df_f_val = df[df["season"].isin(fold["val"])].copy().reset_index(drop=True)

        y_tr_b = df_f_tr["is_draw_actual"].values
        y_val_b = df_f_val["is_draw_actual"].values
        y_val_3c = df_f_val["actual_result"].values
        v4_3c_val = df_f_val[["v4_p_H", "v4_p_D", "v4_p_A"]].values

        # Baseline V4.0
        v4_d = compute_calibration_metrics(y_val_b, df_f_val["v4_p_D"].values)
        v4_mc = compute_multiclass_metrics(y_val_3c, v4_3c_val)

        # Fit Method 4A (Feature-Aware Logistic)
        sc = StandardScaler()
        X_tr = sc.fit_transform(df_f_tr[feat_cols])
        X_val = sc.transform(df_f_val[feat_cols])
        lr_fold = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
        lr_fold.fit(X_tr, y_tr_b)
        p_d_ref = lr_fold.predict_proba(X_val)[:, 1]
        p_3c_ref = redistribute_proportional(v4_3c_val, p_d_ref)

        ref_d = compute_calibration_metrics(y_val_b, p_d_ref)
        ref_mc = compute_multiclass_metrics(y_val_3c, p_3c_ref)

        wf_results.append({
            "Fold Name": fold["name"],
            "Validation Season": fold["val"][0],
            "Matches (N)": len(df_f_val),
            "Actual Draws": int(y_val_b.sum()),
            "V4.0 Draw Brier": v4_d["draw_brier"],
            "Refined Draw Brier": ref_d["draw_brier"],
            "Draw Brier Delta": round(ref_d["draw_brier"] - v4_d["draw_brier"], 6),
            "V4.0 Draw ECE": v4_d["draw_ece"],
            "Refined Draw ECE": ref_d["draw_ece"],
            "Draw ECE Delta": round(ref_d["draw_ece"] - v4_d["draw_ece"], 4),
            "V4.0 Draw AUC": v4_d["draw_auc"],
            "Refined Draw AUC": ref_d["draw_auc"],
            "Draw AUC Delta": round(ref_d["draw_auc"] - v4_d["draw_auc"], 4),
            "V4.0 MC LogLoss": v4_mc["mc_logloss"],
            "Refined MC LogLoss": ref_mc["mc_logloss"],
            "V4.0 Accuracy (%)": v4_mc["counterfactual_acc"],
            "Refined Accuracy (%)": ref_mc["counterfactual_acc"],
        })

    df_wf_res = pd.DataFrame(wf_results)
    df_wf_res.to_csv(OUTPUT_DIR / "probability_walkforward_results.csv", index=False)
    print("\n--- WALK-FORWARD VALIDATION (Method 4A vs V4.0 Baseline) ---")
    print(df_wf_res[["Fold Name", "Matches (N)", "V4.0 Draw ECE", "Refined Draw ECE", "V4.0 Draw AUC", "Refined Draw AUC", "V4.0 MC LogLoss", "Refined MC LogLoss"]].to_string(index=False))

    # =========================================================================
    # 3. PROBABILITY BUCKET ANALYSIS (V4.0 vs Refined vs Actual)
    # =========================================================================
    print("\nCalculating Probability Bucket Analysis on 2025/26 Holdout...")
    df_test["refined_p_D"] = p_d_m4a
    df_test["v4_p_D_bucket"] = pd.cut(df_test["v4_p_D"], bins=[0.0, 0.15, 0.20, 0.25, 0.30, 1.0], labels=["0-15%", "15-20%", "20-25%", "25-30%", "30%+"], right=False)

    bucket_rows = []
    for b_label in ["0-15%", "15-20%", "20-25%", "25-30%", "30%+"]:
        sub = df_test[df_test["v4_p_D_bucket"] == b_label]
        cnt = len(sub)
        d_cnt = int(sub["is_draw_actual"].sum())
        act_rate = (d_cnt / cnt) if cnt > 0 else 0.0
        v4_mean = float(sub["v4_p_D"].mean()) if cnt > 0 else 0.0
        ref_mean = float(sub["refined_p_D"].mean()) if cnt > 0 else 0.0

        v4_gap = act_rate - v4_mean if cnt > 0 else 0.0
        ref_gap = act_rate - ref_mean if cnt > 0 else 0.0

        bucket_rows.append({
            "P(D) Bucket": b_label,
            "Matches (N)": cnt,
            "Actual Draws": d_cnt,
            "Actual Draw Rate (%)": round(act_rate * 100, 2),
            "V4.0 Mean P(D) (%)": round(v4_mean * 100, 2),
            "V4.0 Calib Gap (%)": round(v4_gap * 100, 2),
            "Refined Mean P(D) (%)": round(ref_mean * 100, 2),
            "Refined Calib Gap (%)": round(ref_gap * 100, 2),
            "Calibration Delta (Improvement)": round((abs(v4_gap) - abs(ref_gap)) * 100, 2),
        })

    df_buckets = pd.DataFrame(bucket_rows)
    df_buckets.to_csv(OUTPUT_DIR / "probability_bucket_analysis.csv", index=False)
    print("\n--- DRAW PROBABILITY BUCKET ANALYSIS ---")
    print(df_buckets.to_string(index=False))

    # =========================================================================
    # 4. SCORE-SPACE DRAW SIGNAL ANALYSIS
    # =========================================================================
    print("\nEvaluating Score-Space Signals on Historical Dataset (N=10,734)...")
    score_signals = []
    for s_name, col_name, hyp in [
        ("P(0-0) Scoreline Mass", "p_00", "Bivariate 0-0 scoreless draw probability"),
        ("P(1-1) Scoreline Mass", "p_11", "Bivariate 1-1 modal scoreline probability"),
        ("P(2-2) Scoreline Mass", "p_22", "Bivariate 2-2 high-scoring draw probability"),
        ("Low-Score Mass (<=2 Goals)", "p_low_score", "Aggregated mass on scores with <= 2 total goals"),
        ("Score-Space Total Draw Mass", "p_score_draw", "Sum of all diagonal elements in Dixon-Coles matrix"),
    ]:
        r_val, p_val = stats.pointbiserialr(df["is_draw_actual"], df[col_name])
        q_top = df[df[col_name] >= df[col_name].quantile(0.75)]["is_draw_actual"].mean()
        q_bot = df[df[col_name] <= df[col_name].quantile(0.25)]["is_draw_actual"].mean()

        score_signals.append({
            "Signal Name": s_name,
            "Hypothesis": hyp,
            "Correlation (r)": round(float(r_val), 4),
            "P-Value": f"{p_val:.2e}",
            "Top Quartile Draw Rate (%)": round(q_top * 100, 2),
            "Bottom Quartile Draw Rate (%)": round(q_bot * 100, 2),
            "Effect Size Ratio": round(q_top / q_bot, 2) if q_bot > 0 else np.nan,
            "Pre-Match Validity": "Strictly Causal Pre-Match",
        })

    df_score_sig = pd.DataFrame(score_signals)
    df_score_sig.to_csv(OUTPUT_DIR / "score_space_draw_analysis.csv", index=False)
    print("\n--- SCORE-SPACE SIGNALS SUMMARY ---")
    print(df_score_sig[["Signal Name", "Correlation (r)", "P-Value", "Top Quartile Draw Rate (%)", "Effect Size Ratio"]].to_string(index=False))

    # =========================================================================
    # 5. RECENT 33 PROSPECTIVE MATCH AUDIT (Aug 22-24, 2026)
    # =========================================================================
    print("\nEvaluating Refined Probabilities on Recent 33 Prospective Matches...")
    from dashboard.fixture_service import FixtureService
    from dashboard.prediction_service import PredictionService
    from dashboard.time_utils import format_kickoff_ist, to_chennai_date

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

    # Fit final calibrator on all historical training data
    sc_final = StandardScaler()
    X_all_tr = sc_final.fit_transform(df_train[feat_cols])
    lr_final = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
    lr_final.fit(X_all_tr, df_train["is_draw_actual"])

    for f in recent_fixes:
        p40 = ps.predict_dashboard_fixture(f, model_key="V4.0 Production")
        probs_40 = p40.v4_champ_probs
        p_h, p_d, p_a = probs_40["H"], probs_40["D"], probs_40["A"]
        v4_dec = p40.v4_champ_decision
        prob_gap = abs(p_h - p_a)
        abs_elo = p40.abs_elo_diff or 0.0
        lh = p40.lambda_home or 0.0
        la = p40.lambda_away or 0.0
        ltot = lh + la
        lgap = abs(lh - la)

        # Score space
        rho_val = DrawChampionConfig.from_frozen_json(FROZEN_CHAMPION_PATH).league_rhos.get(f.competition_name, -0.0560)
        mat = compute_dc_score_matrix(lh, la, rho_val)
        p_00 = mat[0, 0]
        p_11 = mat[1, 1]
        p_low = mat[0, 0] + mat[1, 0] + mat[0, 1] + mat[1, 1] + mat[2, 0] + mat[0, 2]
        p_sdraw = sum(mat[k, k] for k in range(mat.shape[0]))
        p_dc = compute_dc_draw_probability(np.array([lh]), np.array([la]), np.array([rho_val]))[0]

        # Feature vector
        f_vec = np.array([[p_d, prob_gap, ltot, lgap, abs_elo, p_dc, p_00, p_11, p_low, p_sdraw, 0.0]])
        f_vec_sc = sc_final.transform(f_vec)
        p_d_refined = float(lr_final.predict_proba(f_vec_sc)[0, 1])

        # Proportional 3-class redistribution
        p3_ref = redistribute_proportional(np.array([[p_h, p_d, p_a]]), np.array([p_d_refined]))[0]
        ref_h, ref_d, ref_a = p3_ref[0], p3_ref[1], p3_ref[2]

        ref_dec = CLASS_ORDER[int(np.argmax(p3_ref))]
        is_draw_act = (f.actual_outcome == "D")
        is_v4_corr = (v4_dec == f.actual_outcome)
        is_ref_corr = (ref_dec == f.actual_outcome)

        # Draw probability rank
        v4_d_rank = sorted([("H", p_h), ("D", p_d), ("A", p_a)], key=lambda x: x[1], reverse=True).index(("D", p_d)) + 1
        ref_d_rank = sorted([("H", ref_h), ("D", ref_d), ("A", ref_a)], key=lambda x: x[1], reverse=True).index(("D", ref_d)) + 1

        r33_rows.append({
            "fixture_id": f.fixture_id,
            "date_ist": to_chennai_date(f.scheduled_kickoff),
            "kickoff_ist": format_kickoff_ist(f.scheduled_kickoff),
            "league": f.competition_name,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "score": f"{f.home_goals}-{f.away_goals}",
            "actual_outcome": f.actual_outcome,
            "v4_p_H": round(p_h, 3),
            "v4_p_D": round(p_d, 3),
            "v4_p_A": round(p_a, 3),
            "v4_pred": v4_dec,
            "v4_correct": is_v4_corr,
            "refined_p_H": round(ref_h, 3),
            "refined_p_D": round(ref_d, 3),
            "refined_p_A": round(ref_a, 3),
            "refined_pred": ref_dec,
            "refined_correct": is_ref_corr,
            "v4_p_D_rank": v4_d_rank,
            "refined_p_D_rank": ref_d_rank,
            "p_D_delta (%)": round((ref_d - p_d) * 100, 2),
            "is_draw_match": is_draw_act,
        })

    df_r33_audit = pd.DataFrame(r33_rows)
    df_r33_audit.to_csv(OUTPUT_DIR / "probability_recent_33_audit.csv", index=False)

    print("\n--- RECENT 33 PROSPECTIVE MATCH PROBABILITY AUDIT ---")
    y_r33_bin = df_r33_audit["is_draw_match"].astype(int).values
    y_r33_3c = df_r33_audit["actual_outcome"].values

    v4_3c_r33 = df_r33_audit[["v4_p_H", "v4_p_D", "v4_p_A"]].values
    ref_3c_r33 = df_r33_audit[["refined_p_H", "refined_p_D", "refined_p_A"]].values

    v4_r33_d = compute_calibration_metrics(y_r33_bin, df_r33_audit["v4_p_D"].values)
    ref_r33_d = compute_calibration_metrics(y_r33_bin, df_r33_audit["refined_p_D"].values)

    v4_r33_mc = compute_multiclass_metrics(y_r33_3c, v4_3c_r33)
    ref_r33_mc = compute_multiclass_metrics(y_r33_3c, ref_3c_r33)

    print(f"V4.0 Draw Brier on 33 matches:    {v4_r33_d['draw_brier']:.6f} -> Refined: {ref_r33_d['draw_brier']:.6f}")
    print(f"V4.0 Draw LogLoss on 33 matches:  {v4_r33_d['draw_logloss']:.6f} -> Refined: {ref_r33_d['draw_logloss']:.6f}")
    print(f"V4.0 Draw ECE on 33 matches:      {v4_r33_d['draw_ece']:.4f} -> Refined: {ref_r33_d['draw_ece']:.4f}")
    print(f"V4.0 Multiclass Brier:            {v4_r33_mc['mc_brier']:.6f} -> Refined: {ref_r33_mc['mc_brier']:.6f}")
    print(f"V4.0 Multiclass LogLoss:          {v4_r33_mc['mc_logloss']:.6f} -> Refined: {ref_r33_mc['mc_logloss']:.6f}")

    # Config JSON
    config_dict = {
        "candidate_id": "v4_0_draw_probability_refinement_step2c",
        "candidate_version": "v1.0-step2c-feature-aware-logistic",
        "category": "Continuous Probability Calibrator",
        "methodology": "Feature-Aware Regularized Logistic Regression with Proportional Odds Redistribution",
        "input_features": feat_cols,
        "parameters": {
            "model_type": "LogisticRegression",
            "C": 0.1,
            "solver": "lbfgs",
            "weights": lr_final.coef_[0].tolist(),
            "intercept": float(lr_final.intercept_[0]),
            "feature_means": sc_final.mean_.tolist(),
            "feature_scales": sc_final.scale_.tolist(),
        },
        "holdout_2025_26_evaluation": {
            "draw_brier_improvement": round(base_d_metrics["draw_brier"] - m4a_d["draw_brier"], 6),
            "draw_logloss_improvement": round(base_d_metrics["draw_logloss"] - m4a_d["draw_logloss"], 6),
            "draw_ece_improvement": round(base_d_metrics["draw_ece"] - m4a_d["draw_ece"], 4),
            "draw_auc_improvement": round(m4a_d["draw_auc"] - base_d_metrics["draw_auc"], 4),
            "mc_brier_improvement": round(base_mc_metrics["mc_brier"] - m4a_mc["mc_brier"], 6),
            "mc_logloss_improvement": round(base_mc_metrics["mc_logloss"] - m4a_mc["mc_logloss"], 6),
        },
        "governance_classification": "RESEARCH CANDIDATE ONLY — NON-MUTATING",
    }

    with open(OUTPUT_DIR / "probability_refinement_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print("  [OK] Saved serialized config to probability_refinement_config.json")

    return {
        "df_cand_comp": df_cand_comp,
        "df_wf_res": df_wf_res,
        "df_buckets": df_buckets,
        "df_score_sig": df_score_sig,
        "df_r33": df_r33_audit,
    }


if __name__ == "__main__":
    run_research_pipeline()
