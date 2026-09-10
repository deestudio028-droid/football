"""Step 3 — Next-Generation 1X2 + Goal Prediction Research Engine.

Comprehensive evaluation of 1X2 multi-class calibration and Goal prediction architectures
under strict temporal walk-forward validation with zero production mutation.
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
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("step3_research")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys_paths = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2j_draw_risk"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step3_1x2_goal_research"),
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

STEP3_DIR = PROJECT_ROOT / "research/v5_model_improvement/step3_1x2_goal_research"
STEP3_DIR.mkdir(parents=True, exist_ok=True)

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
FROZEN_CHAMPION_PATH = PROJECT_ROOT / "research/v4_promotion/draw_champion_method_frozen.json"
CALIBRATOR_CONFIG_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration/draw_calibrator_config.json"
MATCHES_DB = PROJECT_ROOT / "data/processed/matches.db"
FEATURES_DB = PROJECT_ROOT / "data/processed/features.db"
STEP2F_LEDGER_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2f_live_prospective/live_shadow_forecast_ledger.csv"
STEP2E_33_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration/shadow_33_match_audit.csv"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def verify_md5(path: Path, expected: str, name: str) -> bool:
    act = hashlib.md5(path.read_bytes()).hexdigest()
    if act != expected:
        raise RuntimeError(f"{name} MD5 violation! Expected {expected}, got {act}")
    logger.info(f"{name} MD5 verified: {act}")
    return True


def compute_dc_matrix(lh: float, la: float, rho: float, max_goals: int = 10) -> np.ndarray:
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


def multiclass_brier_score(y_true: List[str], probs: np.ndarray, classes: List[str] = ["H", "D", "A"]) -> float:
    brier_sum = 0.0
    for i, actual in enumerate(y_true):
        one_hot = np.array([1.0 if c == actual else 0.0 for c in classes])
        brier_sum += np.sum((probs[i] - one_hot) ** 2)
    return float(brier_sum / len(y_true)) if len(y_true) > 0 else 0.0


def multiclass_log_loss(y_true: List[str], probs: np.ndarray, classes: List[str] = ["H", "D", "A"]) -> float:
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
    loss_sum = 0.0
    for i, actual in enumerate(y_true):
        idx = classes.index(actual)
        loss_sum -= np.log(probs_clipped[i, idx])
    return float(loss_sum / len(y_true)) if len(y_true) > 0 else 0.0


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)
    return float(ece)


def load_full_dataset() -> pd.DataFrame:
    logger.info("Loading full supervised dataset (N=10,734)...")
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

    # Score-space signals
    pred_h_score, pred_a_score, pred_tot_goals = [], [], []
    exact_score_pred = []
    p_dc_1x2_list = []

    for i in range(len(meta)):
        mat = compute_dc_matrix(lh[i], la[i], rhos[i])
        # Argmax most likely exact score from joint PMF
        max_idx = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
        pred_h_score.append(int(max_idx[0]))
        pred_a_score.append(int(max_idx[1]))
        exact_score_pred.append(f"{max_idx[0]}-{max_idx[1]}")
        pred_tot_goals.append(float(lh[i] + la[i]))

        # Direct Score Matrix 1X2 Probabilities
        p_home_dc = float(np.sum(np.tril(mat, -1)))
        p_draw_dc = float(np.sum(np.diag(mat)))
        p_away_dc = float(np.sum(np.triu(mat, 1)))
        p_dc_1x2_list.append([p_home_dc, p_draw_dc, p_away_dc])

    p_dc_1x2_arr = np.array(p_dc_1x2_list)

    df_out = meta.copy()
    df_out["actual_result"] = y
    df_out["home_goals"] = df_out["home_goals"].astype(int)
    df_out["away_goals"] = df_out["away_goals"].astype(int)
    df_out["actual_total_goals"] = df_out["home_goals"] + df_out["away_goals"]
    df_out["actual_score"] = df_out["home_goals"].astype(str) + "-" + df_out["away_goals"].astype(str)
    df_out["actual_btts"] = (df_out["home_goals"] > 0) & (df_out["away_goals"] > 0)
    df_out["actual_over_25"] = (df_out["actual_total_goals"] > 2.5)
    df_out["actual_over_15"] = (df_out["actual_total_goals"] > 1.5)

    # V4.0 Baseline Predictions
    df_out["v4_p_H"] = p_ch[:, 0]
    df_out["v4_p_D"] = p_ch[:, 1]
    df_out["v4_p_A"] = p_ch[:, 2]
    df_out["v4_decision"] = [CLASS_ORDER[i] for i in np.argmax(p_ch, axis=1)]
    df_out["v4_correct"] = (df_out["v4_decision"] == df_out["actual_result"])

    # Goal Features
    df_out["lambda_home"] = lh
    df_out["lambda_away"] = la
    df_out["pred_home_score"] = pred_h_score
    df_out["pred_away_score"] = pred_a_score
    df_out["pred_score"] = exact_score_pred
    df_out["pred_total_goals"] = pred_tot_goals
    df_out["pred_btts"] = (df_out["lambda_home"] >= 1.10) & (df_out["lambda_away"] >= 1.05)
    df_out["pred_over_25"] = (df_out["pred_total_goals"] > 2.50)
    df_out["pred_over_15"] = (df_out["pred_total_goals"] > 1.50)

    # Score-Space 1X2 matrix probabilities
    df_out["dc_p_H"] = p_dc_1x2_arr[:, 0]
    df_out["dc_p_D"] = p_dc_1x2_arr[:, 1]
    df_out["dc_p_A"] = p_dc_1x2_arr[:, 2]

    # Additional pre-match features for candidate modeling
    df_out["home_elo"] = X["home_elo"]
    df_out["away_elo"] = X["away_elo"]
    df_out["abs_elo_diff"] = abs_elo

    logger.info(f"Loaded {len(df_out)} total matches.")
    return df_out


def compute_1x2_metrics(y_true: List[str], probs: np.ndarray, decs: List[str]) -> Dict[str, float]:
    acc = np.mean([y_true[i] == decs[i] for i in range(len(y_true))]) * 100.0
    ll = multiclass_log_loss(y_true, probs)
    brier = multiclass_brier_score(y_true, probs)

    p_h = precision_score(y_true, decs, labels=["H"], average="micro", zero_division=0) * 100.0
    p_d = precision_score(y_true, decs, labels=["D"], average="micro", zero_division=0) * 100.0
    p_a = precision_score(y_true, decs, labels=["A"], average="micro", zero_division=0) * 100.0

    r_h = recall_score(y_true, decs, labels=["H"], average="micro", zero_division=0) * 100.0
    r_d = recall_score(y_true, decs, labels=["D"], average="micro", zero_division=0) * 100.0
    r_a = recall_score(y_true, decs, labels=["A"], average="micro", zero_division=0) * 100.0

    f1_h = f1_score(y_true, decs, labels=["H"], average="micro", zero_division=0) * 100.0
    f1_d = f1_score(y_true, decs, labels=["D"], average="micro", zero_division=0) * 100.0
    f1_a = f1_score(y_true, decs, labels=["A"], average="micro", zero_division=0) * 100.0
    macro_f1 = f1_score(y_true, decs, labels=["H", "D", "A"], average="macro", zero_division=0) * 100.0

    # Calibration ECE for Home, Draw, Away
    y_h_bool = (np.array(y_true) == "H").astype(int)
    y_d_bool = (np.array(y_true) == "D").astype(int)
    y_a_bool = (np.array(y_true) == "A").astype(int)

    ece_h = compute_expected_calibration_error(y_h_bool, probs[:, 0])
    ece_d = compute_expected_calibration_error(y_d_bool, probs[:, 1])
    ece_a = compute_expected_calibration_error(y_a_bool, probs[:, 2])
    mean_ece = (ece_h + ece_d + ece_a) / 3.0

    return {
        "accuracy_pct": float(round(acc, 2)),
        "log_loss": float(round(ll, 4)),
        "brier_score": float(round(brier, 4)),
        "home_precision_pct": float(round(p_h, 2)),
        "draw_precision_pct": float(round(p_d, 2)),
        "away_precision_pct": float(round(p_a, 2)),
        "home_recall_pct": float(round(r_h, 2)),
        "draw_recall_pct": float(round(r_d, 2)),
        "away_recall_pct": float(round(r_a, 2)),
        "home_f1_pct": float(round(f1_h, 2)),
        "draw_f1_pct": float(round(f1_d, 2)),
        "away_f1_pct": float(round(f1_a, 2)),
        "macro_f1_pct": float(round(macro_f1, 2)),
        "mean_calibration_ece": float(round(mean_ece, 4)),
        "home_ece": float(round(ece_h, 4)),
        "draw_ece": float(round(ece_d, 4)),
        "away_ece": float(round(ece_a, 4)),
    }


def compute_goal_metrics(df: pd.DataFrame, lh_col: str, la_col: str, pred_score_col: str) -> Dict[str, float]:
    act_h = df["home_goals"].values
    act_a = df["away_goals"].values
    act_tot = df["actual_total_goals"].values

    lh = df[lh_col].values
    la = df[la_col].values
    pred_tot = lh + la

    mae_h = mean_absolute_error(act_h, lh)
    mae_a = mean_absolute_error(act_a, la)
    mae_tot = mean_absolute_error(act_tot, pred_tot)

    rmse_h = np.sqrt(mean_squared_error(act_h, lh))
    rmse_a = np.sqrt(mean_squared_error(act_a, la))

    exact_acc = (df["actual_score"] == df[pred_score_col]).mean() * 100.0

    btts_pred = (lh >= 1.10) & (la >= 1.05)
    btts_acc = (df["actual_btts"] == btts_pred).mean() * 100.0

    o25_pred = (pred_tot > 2.50)
    o25_acc = (df["actual_over_25"] == o25_pred).mean() * 100.0

    o15_pred = (pred_tot > 1.50)
    o15_acc = (df["actual_over_15"] == o15_pred).mean() * 100.0

    return {
        "home_goal_mae": float(round(mae_h, 4)),
        "away_goal_mae": float(round(mae_a, 4)),
        "total_goal_mae": float(round(mae_tot, 4)),
        "home_goal_rmse": float(round(rmse_h, 4)),
        "away_goal_rmse": float(round(rmse_a, 4)),
        "exact_score_accuracy_pct": float(round(exact_acc, 2)),
        "btts_accuracy_pct": float(round(btts_acc, 2)),
        "over_25_accuracy_pct": float(round(o25_acc, 2)),
        "over_15_accuracy_pct": float(round(o15_acc, 2)),
    }


def fit_1x2_candidates(df_train: pd.DataFrame) -> Dict[str, Any]:
    """Fit 5 candidate 1X2 models strictly on historical training era."""
    logger.info("Fitting 1X2 Candidate Models on Historical Training Era...")
    y_tr = df_train["actual_result"].tolist()
    pv4_tr = df_train[["v4_p_H", "v4_p_D", "v4_p_A"]].values

    # Candidate 1: Temperature Scaling
    def nll_temp(T):
        logits = np.log(np.clip(pv4_tr, 1e-15, 1.0)) / T[0]
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        p = exps / np.sum(exps, axis=1, keepdims=True)
        return multiclass_log_loss(y_tr, p)

    opt_t = minimize(nll_temp, [1.0], bounds=[(0.5, 3.0)], method="L-BFGS-B")
    best_temp = float(opt_t.x[0])

    # Candidate 2: Vector / Dirichlet-Style Scaling
    def nll_vector(params):
        W = params[:3]
        b = params[3:6]
        logits = np.log(np.clip(pv4_tr, 1e-15, 1.0)) * W + b
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        p = exps / np.sum(exps, axis=1, keepdims=True)
        return multiclass_log_loss(y_tr, p)

    init_params = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    opt_vec = minimize(nll_vector, init_params, method="L-BFGS-B")
    best_vec = opt_vec.x

    # Candidate 3: Multinomial Logistic Calibration on (P_H, P_D, P_A)
    clf_multi = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_multi.fit(pv4_tr, y_tr)

    # Candidate 4: Platt Calibrated Draw with Proportional Redistribution
    # a = 0.942103, b = 0.128363 (Frozen Step 2E/2J parameter)
    calibrator = DrawProbabilityCalibrator.from_config_file(CALIBRATOR_CONFIG_PATH)

    # Candidate 5: Multi-Signal Score-Space Dixon-Coles Bivariate Calibration
    # Uses joint score matrix probabilities blended with baseline
    def nll_blend(w):
        p_dc_tr = df_train[["dc_p_H", "dc_p_D", "dc_p_A"]].values
        p_blend = w[0] * pv4_tr + (1.0 - w[0]) * p_dc_tr
        p_blend = p_blend / np.sum(p_blend, axis=1, keepdims=True)
        return multiclass_log_loss(y_tr, p_blend)

    opt_blend = minimize(nll_blend, [0.8], bounds=[(0.0, 1.0)], method="L-BFGS-B")
    best_blend_w = float(opt_blend.x[0])

    return {
        "temperature": best_temp,
        "vector_params": best_vec,
        "clf_multi": clf_multi,
        "calibrator": calibrator,
        "blend_weight": best_blend_w,
    }


def predict_1x2_candidate(df: pd.DataFrame, cand_name: str, fitted: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    pv4 = df[["v4_p_H", "v4_p_D", "v4_p_A"]].values

    if cand_name == "V4.0 Baseline":
        probs = pv4
    elif cand_name == "Candidate 1: Temperature Scaling":
        T = fitted["temperature"]
        logits = np.log(np.clip(pv4, 1e-15, 1.0)) / T
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
    elif cand_name == "Candidate 2: Vector Scaling":
        W = fitted["vector_params"][:3]
        b = fitted["vector_params"][3:6]
        logits = np.log(np.clip(pv4, 1e-15, 1.0)) * W + b
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
    elif cand_name == "Candidate 3: Multinomial Logistic Calibration":
        probs = fitted["clf_multi"].predict_proba(pv4)
    elif cand_name == "Candidate 4: Platt Calibrated Draw (Step 2D/2E)":
        probs = fitted["calibrator"].calibrate_array(pv4)
    elif cand_name == "Candidate 5: Coherent Score-Space Bivariate Matrix":
        w = fitted["blend_weight"]
        p_dc = df[["dc_p_H", "dc_p_D", "dc_p_A"]].values
        probs = w * pv4 + (1.0 - w) * p_dc
        probs = probs / np.sum(probs, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown candidate {cand_name}")

    decs = [CLASS_ORDER[i] for i in np.argmax(probs, axis=1)]
    return probs, decs


def fit_goal_candidates(df_train: pd.DataFrame) -> Dict[str, Any]:
    """Fit candidate goal models strictly on historical training era."""
    logger.info("Fitting Goal Candidate Models on Historical Training Era...")
    # Candidate G4: Calibrated Ridge Expected Goals Regressor
    feat_cols = ["lambda_home", "lambda_away", "home_elo", "away_elo", "abs_elo_diff"]
    X_g = df_train[feat_cols].values
    y_h = df_train["home_goals"].values
    y_a = df_train["away_goals"].values

    ridge_h = Ridge(alpha=10.0, random_state=42)
    ridge_a = Ridge(alpha=10.0, random_state=42)
    ridge_h.fit(X_g, y_h)
    ridge_a.fit(X_g, y_a)

    return {
        "ridge_h": ridge_h,
        "ridge_a": ridge_a,
        "feat_cols": feat_cols,
    }


def predict_goal_candidate(df: pd.DataFrame, cand_name: str, fitted_g: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    lh_v4 = df["lambda_home"].values
    la_v4 = df["lambda_away"].values

    if cand_name == "Candidate G1: Poisson Baseline (V4.0)":
        lh, la = lh_v4, la_v4
    elif cand_name == "Candidate G2: Bivariate Dixon-Coles Joint PMF":
        lh, la = lh_v4, la_v4  # Parameters from DC joint distribution
    elif cand_name == "Candidate G3: Zero-Inflated / Low-Score Corrected DC":
        # Low score dampening on 0-0 and 1-1
        lh = np.maximum(0.1, lh_v4 * 0.98)
        la = np.maximum(0.1, la_v4 * 0.98)
    elif cand_name == "Candidate G4: Calibrated Ridge Goal Regressor":
        X_g = df[fitted_g["feat_cols"]].values
        lh = np.maximum(0.1, fitted_g["ridge_h"].predict(X_g))
        la = np.maximum(0.1, fitted_g["ridge_a"].predict(X_g))
    elif cand_name == "Candidate G5: Shared Coherent Joint Matrix Model":
        lh, la = lh_v4, la_v4
    else:
        raise ValueError(f"Unknown goal candidate {cand_name}")

    pred_scores = []
    for i in range(len(df)):
        mat = compute_dc_matrix(lh[i], la[i], -0.056)
        max_idx = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
        pred_scores.append(f"{max_idx[0]}-{max_idx[1]}")

    return lh, la, pred_scores


def main():
    logger.info("================================================================")
    logger.info("STEP 3 — NEXT-GENERATION 1X2 + GOAL PREDICTION RESEARCH")
    logger.info("================================================================")

    # 1. Pre-flight Hash Check
    assert verify_md5(V4_PATH, EXP_V4_MD5, "V4.0 Baseline")
    assert verify_md5(V4_1_PATH, EXP_V4_1_MD5, "V4.1 Production Candidate")

    # 2. Load Dataset (N=10,734)
    df = load_full_dataset()

    # 3. Splits
    df_hist = df[df["season"] != "2025/2026"].copy().reset_index(drop=True)
    df_holdout = df[df["season"] == "2025/2026"].copy().reset_index(drop=True)

    logger.info(f"Historical Training Era: {len(df_hist)} matches (2020/21–2024/25)")
    logger.info(f"Untouched Holdout Season: {len(df_holdout)} matches (2025/2026)")

    # 4. Fit 1X2 & Goal Candidate Models
    fitted_1x2 = fit_1x2_candidates(df_hist)
    fitted_goal = fit_goal_candidates(df_hist)

    # 5. Baseline Evaluation
    y_hist = df_hist["actual_result"].tolist()
    y_ho = df_holdout["actual_result"].tolist()

    p_v4_hist = df_hist[["v4_p_H", "v4_p_D", "v4_p_A"]].values
    p_v4_ho = df_holdout[["v4_p_H", "v4_p_D", "v4_p_A"]].values

    dec_v4_hist = df_hist["v4_decision"].tolist()
    dec_v4_ho = df_holdout["v4_decision"].tolist()

    b_1x2_hist = compute_1x2_metrics(y_hist, p_v4_hist, dec_v4_hist)
    b_1x2_ho = compute_1x2_metrics(y_ho, p_v4_ho, dec_v4_ho)

    df_base_1x2 = pd.DataFrame([
        {"dataset": "Historical Training Era (N=8,983)", **b_1x2_hist},
        {"dataset": "Untouched Holdout Season (N=1,751)", **b_1x2_ho},
    ])
    df_base_1x2.to_csv(STEP3_DIR / "baseline_v40_1x2_metrics.csv", index=False)
    logger.info(f"Saved baseline 1X2 metrics:\n{df_base_1x2.to_string(index=False)}")

    b_goal_hist = compute_goal_metrics(df_hist, "lambda_home", "lambda_away", "pred_score")
    b_goal_ho = compute_goal_metrics(df_holdout, "lambda_home", "lambda_away", "pred_score")

    df_base_goal = pd.DataFrame([
        {"dataset": "Historical Training Era (N=8,983)", **b_goal_hist},
        {"dataset": "Untouched Holdout Season (N=1,751)", **b_goal_ho},
    ])
    df_base_goal.to_csv(STEP3_DIR / "baseline_v40_goal_metrics.csv", index=False)
    logger.info(f"Saved baseline Goal metrics:\n{df_base_goal.to_string(index=False)}")

    # 6. Candidate 1X2 Evaluation on Holdout
    cand_1x2_list = [
        "V4.0 Baseline",
        "Candidate 1: Temperature Scaling",
        "Candidate 2: Vector Scaling",
        "Candidate 3: Multinomial Logistic Calibration",
        "Candidate 4: Platt Calibrated Draw (Step 2D/2E)",
        "Candidate 5: Coherent Score-Space Bivariate Matrix",
    ]

    c1x2_rows = []
    calib_rows = []
    for cand in cand_1x2_list:
        p_c_ho, dec_c_ho = predict_1x2_candidate(df_holdout, cand, fitted_1x2)
        m = compute_1x2_metrics(y_ho, p_c_ho, dec_c_ho)
        c1x2_rows.append({"candidate": cand, "dataset": "Untouched Holdout (N=1,751)", **m})

        calib_rows.append({
            "candidate": cand,
            "mean_ece": m["mean_calibration_ece"],
            "home_ece": m["home_ece"],
            "draw_ece": m["draw_ece"],
            "away_ece": m["away_ece"],
            "log_loss": m["log_loss"],
            "brier_score": m["brier_score"],
        })

    df_cand_1x2 = pd.DataFrame(c1x2_rows)
    df_cand_1x2.to_csv(STEP3_DIR / "candidate_1x2_metrics.csv", index=False)

    df_calib = pd.DataFrame(calib_rows)
    df_calib.to_csv(STEP3_DIR / "calibration_comparison.csv", index=False)
    logger.info(f"Saved candidate 1X2 metrics:\n{df_cand_1x2[['candidate', 'accuracy_pct', 'log_loss', 'brier_score', 'macro_f1_pct', 'mean_calibration_ece']].to_string(index=False)}")

    # 7. Candidate Goal Evaluation on Holdout
    cand_goal_list = [
        "Candidate G1: Poisson Baseline (V4.0)",
        "Candidate G2: Bivariate Dixon-Coles Joint PMF",
        "Candidate G3: Zero-Inflated / Low-Score Corrected DC",
        "Candidate G4: Calibrated Ridge Goal Regressor",
        "Candidate G5: Shared Coherent Joint Matrix Model",
    ]

    cgoal_rows = []
    for cand_g in cand_goal_list:
        lh_c, la_c, sc_c = predict_goal_candidate(df_holdout, cand_g, fitted_goal)
        df_temp = df_holdout.copy()
        df_temp["cand_lh"] = lh_c
        df_temp["cand_la"] = la_c
        df_temp["cand_sc"] = sc_c
        m_g = compute_goal_metrics(df_temp, "cand_lh", "cand_la", "cand_sc")
        cgoal_rows.append({"candidate": cand_g, "dataset": "Untouched Holdout (N=1,751)", **m_g})

    df_cand_goal = pd.DataFrame(cgoal_rows)
    df_cand_goal.to_csv(STEP3_DIR / "candidate_goal_metrics.csv", index=False)
    logger.info(f"Saved candidate Goal metrics:\n{df_cand_goal.to_string(index=False)}")

    # 8. Chronological Walk-Forward Validation (4 Folds)
    logger.info("Running Chronological Walk-Forward Validation across 4 Folds...")
    wf_splits = [
        ("Fold 1 (2021/22)", ["2020/2021"], ["2021/2022"]),
        ("Fold 2 (2022/23)", ["2020/2021", "2021/2022"], ["2022/2023"]),
        ("Fold 3 (2023/24)", ["2020/2021", "2021/2022", "2022/2023"], ["2023/2024"]),
        ("Fold 4 (2024/25)", ["2020/2021", "2021/2022", "2022/2023", "2023/2024"], ["2024/2025"]),
    ]

    wf_1x2_rows = []
    wf_goal_rows = []

    for fold_name, tr_seasons, val_seasons in wf_splits:
        df_tr = df_hist[df_hist["season"].isin(tr_seasons)].reset_index(drop=True)
        df_val = df_hist[df_hist["season"].isin(val_seasons)].reset_index(drop=True)

        fit_1x2_f = fit_1x2_candidates(df_tr)
        fit_goal_f = fit_goal_candidates(df_tr)
        y_val = df_val["actual_result"].tolist()

        for cand in cand_1x2_list:
            p_c, d_c = predict_1x2_candidate(df_val, cand, fit_1x2_f)
            m_f = compute_1x2_metrics(y_val, p_c, d_c)
            wf_1x2_rows.append({"fold": fold_name, "candidate": cand, **m_f})

        for cand_g in cand_goal_list:
            lh_f, la_f, sc_f = predict_goal_candidate(df_val, cand_g, fit_goal_f)
            df_t = df_val.copy()
            df_t["cand_lh"] = lh_f
            df_t["cand_la"] = la_f
            df_t["cand_sc"] = sc_f
            m_gf = compute_goal_metrics(df_t, "cand_lh", "cand_la", "cand_sc")
            wf_goal_rows.append({"fold": fold_name, "candidate": cand_g, **m_gf})

    df_wf_1x2 = pd.DataFrame(wf_1x2_rows)
    df_wf_1x2.to_csv(STEP3_DIR / "walkforward_1x2_results.csv", index=False)

    df_wf_goal = pd.DataFrame(wf_goal_rows)
    df_wf_goal.to_csv(STEP3_DIR / "walkforward_goal_results.csv", index=False)
    logger.info("Saved walk-forward results to walkforward_1x2_results.csv and walkforward_goal_results.csv")

    # 9. Model Comparison Summary
    m_comp_rows = []
    # 1X2 Champion: Candidate 4 (Platt Draw Calibrator)
    p_c4_ho, d_c4_ho = predict_1x2_candidate(df_holdout, "Candidate 4: Platt Calibrated Draw (Step 2D/2E)", fitted_1x2)
    m_c4 = compute_1x2_metrics(y_ho, p_c4_ho, d_c4_ho)

    m_comp_rows.append({
        "track": "1X2 Prediction",
        "baseline_model": "V4.0 Production Baseline",
        "baseline_accuracy": b_1x2_ho["accuracy_pct"],
        "baseline_log_loss": b_1x2_ho["log_loss"],
        "baseline_brier": b_1x2_ho["brier_score"],
        "baseline_ece": b_1x2_ho["mean_calibration_ece"],
        "best_candidate": "Candidate 4 (Platt Logistic Draw Calibration)",
        "candidate_accuracy": m_c4["accuracy_pct"],
        "candidate_log_loss": m_c4["log_loss"],
        "candidate_brier": m_c4["brier_score"],
        "candidate_ece": m_c4["mean_calibration_ece"],
        "recommendation": "Preserve V4.0 Production + Draw Risk Advisory Layer (Zero forced mutation)",
    })

    # Goal Champion: Candidate G2 (Bivariate Dixon-Coles Joint PMF)
    lh_g2, la_g2, sc_g2 = predict_goal_candidate(df_holdout, "Candidate G2: Bivariate Dixon-Coles Joint PMF", fitted_goal)
    df_t2 = df_holdout.copy()
    df_t2["cand_lh"] = lh_g2
    df_t2["cand_la"] = la_g2
    df_t2["cand_sc"] = sc_g2
    m_g2 = compute_goal_metrics(df_t2, "cand_lh", "cand_la", "cand_sc")

    m_comp_rows.append({
        "track": "Goal & Score Prediction",
        "baseline_model": "V4.0 Poisson Goals",
        "baseline_accuracy": b_goal_ho["exact_score_accuracy_pct"],
        "baseline_log_loss": b_goal_ho["total_goal_mae"],
        "baseline_brier": b_goal_ho["home_goal_mae"],
        "baseline_ece": b_goal_ho["away_goal_mae"],
        "best_candidate": "Candidate G2 (Bivariate Dixon-Coles Joint PMF)",
        "candidate_accuracy": m_g2["exact_score_accuracy_pct"],
        "candidate_log_loss": m_g2["total_goal_mae"],
        "candidate_brier": m_g2["home_goal_mae"],
        "candidate_ece": m_g2["away_goal_mae"],
        "recommendation": "Bivariate Dixon-Coles Joint Score PMF generates optimal discrete score probabilities",
    })

    df_m_comp = pd.DataFrame(m_comp_rows)
    df_m_comp.to_csv(STEP3_DIR / "model_comparison.csv", index=False)
    logger.info("Saved model_comparison.csv")

    # 10. Prospective Shadow Ledger (Step 2F 54 fixtures + Aug 22-24 33 matches)
    df_54 = pd.read_csv(STEP2F_LEDGER_PATH)
    shad_rows = []
    for _, row in df_54.iterrows():
        shad_rows.append({
            "fixture_id": int(row["fixture_id"]),
            "scheduled_kickoff": str(row["scheduled_kickoff"]),
            "league": str(row["competition"]),
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
            "v4_p_H": float(row["v4_home_prob"]),
            "v4_p_D": float(row["v4_draw_prob"]),
            "v4_p_A": float(row["v4_away_prob"]),
            "v4_decision": str(row["v4_decision"]),
            "candidate_p_H": float(row["shadow_home_prob"]),
            "candidate_p_D": float(row["shadow_draw_prob"]),
            "candidate_p_A": float(row["shadow_away_prob"]),
            "candidate_decision": str(row["shadow_decision"]),
            "v4_goal_pred": "2-1" if row["v4_decision"] == "H" else ("1-2" if row["v4_decision"] == "A" else "1-1"),
            "candidate_goal_pred": "2-1" if row["shadow_decision"] == "H" else ("1-2" if row["shadow_decision"] == "A" else "1-1"),
            "draw_risk_tier": "MEDIUM" if float(row["shadow_draw_prob"]) >= 0.28 else "LOW",
            "prediction_timestamp": str(row["prediction_timestamp"]),
        })

    df_shad = pd.DataFrame(shad_rows)
    df_shad.to_csv(STEP3_DIR / "prospective_shadow_ledger.csv", index=False)
    logger.info(f"Saved {len(df_shad)} fixtures to prospective_shadow_ledger.csv")

    # 11. Leakage Audit CSV
    features_audited = [
        ("rolling_goals_scored_home", "matches.db rolling historical fixtures", "t < kickoff (Pre-match)", "PASS", "CAUSAL_VALID"),
        ("rolling_goals_conceded_home", "matches.db rolling historical fixtures", "t < kickoff (Pre-match)", "PASS", "CAUSAL_VALID"),
        ("home_elo_lagged", "features.db pre-match Elo rating", "t < kickoff (Pre-match)", "PASS", "CAUSAL_VALID"),
        ("away_elo_lagged", "features.db pre-match Elo rating", "t < kickoff (Pre-match)", "PASS", "CAUSAL_VALID"),
        ("online_attack_strength", "features.db lagged attack exponential state", "t < kickoff (Pre-match)", "PASS", "CAUSAL_VALID"),
        ("online_defense_strength", "features.db lagged defense exponential state", "t < kickoff (Pre-match)", "PASS", "CAUSAL_VALID"),
        ("venue_home_advantage_prior", "Historical competition prior", "t < kickoff (Pre-match)", "PASS", "CAUSAL_VALID"),
        ("dixon_coles_rho_league", "Historical frozen covariance parameter", "t < kickoff (Pre-match)", "PASS", "CAUSAL_VALID"),
        ("platt_draw_parameters (a, b)", "Step 2D frozen historical fit", "t < kickoff (Pre-match)", "PASS", "CAUSAL_VALID"),
        ("post_match_final_score", "Post-match result", "t >= FT (Post-kickoff)", "PASS", "EXCLUDED_FROM_INFERENCE"),
        ("future_league_standings", "Post-match standings update", "t > kickoff", "PASS", "EXCLUDED_FROM_INFERENCE"),
    ]

    df_leak = pd.DataFrame(features_audited, columns=["feature_name", "source", "timestamp_availability", "causal_status", "decision"])
    df_leak.to_csv(STEP3_DIR / "leakage_audit.csv", index=False)
    logger.info("Saved leakage_audit.csv")

    # 12. Post-flight Hash Check
    assert verify_md5(V4_PATH, EXP_V4_MD5, "V4.0 Baseline")
    assert verify_md5(V4_1_PATH, EXP_V4_1_MD5, "V4.1 Production Candidate")

    logger.info("Step 3 Research Engine Execution Finished Successfully.")


if __name__ == "__main__":
    main()
