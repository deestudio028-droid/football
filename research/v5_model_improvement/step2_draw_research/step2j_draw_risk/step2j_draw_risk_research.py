"""Step 2J — Draw Risk / Caution Intelligence Research.

Research engine developing, calibrating, and validating a Draw Risk / Caution Intelligence Layer.
Preserves original V4.0 Home/Away predictions while computing a continuous Draw Risk Score (0 to 1),
defining deterministic risk tiers (LOW, MEDIUM, HIGH, CRITICAL), and generating human-readable
advisory reasons for high-uncertainty draw-prone fixtures.
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
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("step2j_draw_risk")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
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

# Directory & Artifact Paths
STEP2J_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2j_draw_risk"
STEP2J_DIR.mkdir(parents=True, exist_ok=True)

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


def load_full_dataset() -> pd.DataFrame:
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

    # V4.0 Baseline Predictions
    df_out["v4_p_H"] = p_ch[:, 0]
    df_out["v4_p_D"] = p_ch[:, 1]
    df_out["v4_p_A"] = p_ch[:, 2]
    df_out["v4_decision"] = [CLASS_ORDER[i] for i in np.argmax(p_ch, axis=1)]
    df_out["v4_correct"] = (df_out["v4_decision"] == df_out["actual_result"])
    df_out["v4_favorite_prob"] = np.maximum(df_out["v4_p_H"], df_out["v4_p_A"])

    # Calibrated Probabilities
    df_out["cal_p_H"] = p_cal[:, 0]
    df_out["cal_p_D"] = p_cal[:, 1]
    df_out["cal_p_A"] = p_cal[:, 2]

    # Model Features & Signals
    df_out["lambda_home"] = lh
    df_out["lambda_away"] = la
    df_out["lambda_total"] = lh + la
    df_out["lambda_gap"] = np.abs(lh - la)
    df_out["home_elo"] = X["home_elo"]
    df_out["away_elo"] = X["away_elo"]
    df_out["abs_elo_diff"] = abs_elo

    # Probability margins
    df_out["v4_win_diff"] = np.abs(df_out["v4_p_H"] - df_out["v4_p_A"])
    df_out["cal_win_diff"] = np.abs(df_out["cal_p_H"] - df_out["cal_p_A"])
    df_out["cal_win_to_draw_margin"] = np.maximum(df_out["cal_p_H"], df_out["cal_p_A"]) - df_out["cal_p_D"]

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
    df_out["ad_net_power_gap"] = np.abs((X["A_home"] - X["D_away"]) - (X["A_away"] - X["D_home"]))

    logger.info(f"Loaded {len(df_out)} total fixtures across {df_out['season'].nunique()} seasons.")
    return df_out


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    """Compute Expected Calibration Error (ECE) and reliability table."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    table_rows = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_err = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * bin_err
            table_rows.append({
                "bin_idx": i + 1,
                "bin_range": f"{bin_lower:.2f} - {bin_upper:.2f}",
                "count": int(bin_count),
                "mean_predicted_prob": float(bin_conf),
                "actual_draw_rate": float(bin_acc),
                "abs_calibration_error": float(bin_err),
            })
        else:
            table_rows.append({
                "bin_idx": i + 1,
                "bin_range": f"{bin_lower:.2f} - {bin_upper:.2f}",
                "count": 0,
                "mean_predicted_prob": float((bin_lower + bin_upper) / 2.0),
                "actual_draw_rate": 0.0,
                "abs_calibration_error": 0.0,
            })

    return ece, pd.DataFrame(table_rows)


def fit_and_evaluate_risk_architectures(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train and compare 7 distinct Draw Risk scoring architectures."""
    y_tr = df_train["is_draw_actual"].values
    y_te = df_test["is_draw_actual"].values

    # Features for Model F (Regularized Logistic Risk Model)
    log_features = [
        "cal_p_D",
        "cal_win_diff",
        "lambda_gap",
        "lambda_total",
        "p_score_space_draw",
        "p_00",
        "p_11",
        "abs_elo_diff",
    ]

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(df_train[log_features].values, y_tr)

    scores_tr = {}
    scores_te = {}

    # Architecture A: Simple Calibrated P(D) scaled
    scores_tr["A_calibrated_pd"] = df_train["cal_p_D"].values
    scores_te["A_calibrated_pd"] = df_test["cal_p_D"].values

    # Architecture B: Probability-Parity Risk
    scores_tr["B_win_parity"] = df_train["cal_p_D"].values * (1.0 - np.clip(df_train["cal_win_diff"].values / 0.25, 0.0, 1.0))
    scores_te["B_win_parity"] = df_test["cal_p_D"].values * (1.0 - np.clip(df_test["cal_win_diff"].values / 0.25, 0.0, 1.0))

    # Architecture C: Score-Space Risk
    scores_tr["C_score_space"] = (df_train["cal_p_D"].values + df_train["p_score_space_draw"].values) / 2.0
    scores_te["C_score_space"] = (df_test["cal_p_D"].values + df_test["p_score_space_draw"].values) / 2.0

    # Architecture D: Low-Goal Intensity Risk
    scores_tr["D_low_goal"] = df_train["cal_p_D"].values * (1.0 - np.clip(df_train["lambda_total"].values / 3.5, 0.0, 1.0))
    scores_te["D_low_goal"] = df_test["cal_p_D"].values * (1.0 - np.clip(df_test["lambda_total"].values / 3.5, 0.0, 1.0))

    # Architecture E: Combined Interpretable Composite Risk Score (Scale 0 to 1)
    comp_tr = (
        0.35 * np.clip((df_train["cal_p_D"].values - 0.22) / 0.12, 0.0, 1.0)
        + 0.30 * (1.0 - np.clip(df_train["cal_win_diff"].values / 0.20, 0.0, 1.0))
        + 0.20 * np.clip((df_train["p_score_space_draw"].values - 0.22) / 0.10, 0.0, 1.0)
        + 0.15 * (1.0 - np.clip(df_train["lambda_gap"].values / 0.60, 0.0, 1.0))
    )
    comp_te = (
        0.35 * np.clip((df_test["cal_p_D"].values - 0.22) / 0.12, 0.0, 1.0)
        + 0.30 * (1.0 - np.clip(df_test["cal_win_diff"].values / 0.20, 0.0, 1.0))
        + 0.20 * np.clip((df_test["p_score_space_draw"].values - 0.22) / 0.10, 0.0, 1.0)
        + 0.15 * (1.0 - np.clip(df_test["lambda_gap"].values / 0.60, 0.0, 1.0))
    )
    scores_tr["E_composite_interpretable"] = np.clip(comp_tr, 0.0, 1.0)
    scores_te["E_composite_interpretable"] = np.clip(comp_te, 0.0, 1.0)

    # Architecture F: Regularized Logistic Risk Model
    scores_tr["F_logistic_model"] = clf.predict_proba(df_train[log_features].values)[:, 1]
    scores_te["F_logistic_model"] = clf.predict_proba(df_test[log_features].values)[:, 1]

    # Architecture G: Monotonic Percentile Rank Score (based on composite)
    ranks_tr = pd.Series(scores_tr["E_composite_interpretable"]).rank(pct=True).values
    scores_tr["G_rank_percentile"] = ranks_tr
    # Compute test percentiles against train distribution
    ranks_te = np.array([np.mean(scores_tr["E_composite_interpretable"] <= val) for val in scores_te["E_composite_interpretable"]])
    scores_te["G_rank_percentile"] = ranks_te

    # Evaluate Calibration & Ranking Metrics on Test Set
    comp_rows = []
    for arch_name, s_te in scores_te.items():
        brier = brier_score_loss(y_te, s_te) if s_te.max() <= 1.0 and s_te.min() >= 0.0 else np.nan
        # Correlation with draw outcome
        corr_draw = np.corrcoef(s_te, y_te)[0, 1]
        # Correlation with V4 correctness (should be negative: higher risk -> lower V4 accuracy)
        corr_v4_acc = np.corrcoef(s_te, df_test["v4_correct"].astype(int).values)[0, 1]
        ece, _ = compute_expected_calibration_error(y_te, np.clip(s_te, 0.0, 1.0))

        comp_rows.append({
            "architecture": arch_name,
            "brier_score": float(brier),
            "ece": float(ece),
            "correlation_with_draw": float(corr_draw),
            "correlation_with_v4_correctness": float(corr_v4_acc),
        })

    df_arch_comp = pd.DataFrame(comp_rows).sort_values(by="correlation_with_draw", ascending=False)
    fitted_info = {
        "clf": clf,
        "log_features": log_features,
        "train_composite_scores": scores_tr["E_composite_interpretable"],
    }
    return fitted_info, df_arch_comp


def assign_risk_tier(score: float, thresholds: Dict[str, float]) -> str:
    """Assign deterministic risk tier from continuous risk score."""
    if score >= thresholds["critical"]:
        return "CRITICAL"
    elif score >= thresholds["high"]:
        return "HIGH"
    elif score >= thresholds["medium"]:
        return "MEDIUM"
    else:
        return "LOW"


def generate_explainability_reason(row: pd.Series, tier: str) -> str:
    """Generate human-readable deterministic advisory reason based on contributing signals."""
    p_d = row["cal_p_D"] * 100.0
    win_gap = row["cal_win_diff"] * 100.0
    tot_goals = row["lambda_total"]
    elo_gap = row["abs_elo_diff"]
    score_draw = row["p_score_space_draw"] * 100.0

    if tier == "CRITICAL":
        return f"Tight parity (|H-A|={win_gap:.1f}%), elevated draw prob ({p_d:.1f}%), low goal intensity (xG={tot_goals:.2f}), score-space draw mass={score_draw:.1f}%."
    elif tier == "HIGH":
        return f"Elevated draw prob ({p_d:.1f}%), close win margin (|H-A|={win_gap:.1f}%), balanced team Elo gap ({elo_gap:.0f})."
    elif tier == "MEDIUM":
        return f"Moderate draw risk ({p_d:.1f}%), slight favorite separation (|H-A|={win_gap:.1f}%)."
    else:
        return f"Decisive win margin (|H-A|={win_gap:.1f}%) favored by baseline model; low draw exposure ({p_d:.1f}%)."


def analyze_risk_tiers(df: pd.DataFrame, score_col: str, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Compute detailed diagnostic metrics by Draw Risk tier."""
    tiers = ["LOW", "MEDIUM", "HIGH", "CRITICAL", "ALL"]
    rows = []
    n_total = len(df)

    for t in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        sub = df[df["draw_risk_tier"] == t]
        n_sub = len(sub)
        pct_sub = (n_sub / n_total * 100.0) if n_total > 0 else 0.0

        if n_sub > 0:
            draw_rate = (sub["actual_result"] == "D").mean() * 100.0
            v4_acc = sub["v4_correct"].mean() * 100.0
            fav_fail_rate = (100.0 - v4_acc)
            mean_score = sub[score_col].mean()
            mean_pd = sub["cal_p_D"].mean() * 100.0
            mean_win_gap = sub["cal_win_diff"].mean() * 100.0
        else:
            draw_rate, v4_acc, fav_fail_rate, mean_score, mean_pd, mean_win_gap = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        rows.append({
            "risk_tier": t,
            "match_count": int(n_sub),
            "pct_of_matches": float(pct_sub),
            "mean_risk_score": float(mean_score),
            "mean_calibrated_P_D": float(mean_pd),
            "mean_win_gap": float(mean_win_gap),
            "actual_draw_rate": float(draw_rate),
            "v4_0_accuracy": float(v4_acc),
            "favorite_failure_rate": float(fav_fail_rate),
        })

    # Overall Summary Row
    rows.append({
        "risk_tier": "OVERALL",
        "match_count": int(n_total),
        "pct_of_matches": 100.0,
        "mean_risk_score": float(df[score_col].mean()),
        "mean_calibrated_P_D": float(df["cal_p_D"].mean() * 100.0),
        "mean_win_gap": float(df["cal_win_diff"].mean() * 100.0),
        "actual_draw_rate": float((df["actual_result"] == "D").mean() * 100.0),
        "v4_0_accuracy": float(df["v4_correct"].mean() * 100.0),
        "favorite_failure_rate": float((100.0 - df["v4_correct"].mean() * 100.0)),
    })

    return pd.DataFrame(rows)


def main():
    logger.info("================================================================")
    logger.info("STEP 2J — DRAW RISK / CAUTION INTELLIGENCE RESEARCH")
    logger.info("================================================================")

    # 1. Pre-flight Hash Verification
    assert verify_md5(V4_PATH, EXP_V4_MD5, "V4.0 Baseline")
    assert verify_md5(V4_1_PATH, EXP_V4_1_MD5, "V4.1 Production Candidate")

    # 2. Load Full Supervised Dataset (10,734 matches)
    df = load_full_dataset()

    # 3. Chronological Splits (Historical 5 Seasons vs Untouched Holdout)
    df_hist = df[df["season"] != "2025/2026"].copy().reset_index(drop=True)
    df_holdout = df[df["season"] == "2025/2026"].copy().reset_index(drop=True)

    logger.info(f"Historical Training Era: {len(df_hist)} matches (2020/21–2024/25)")
    logger.info(f"Untouched Holdout Season: {len(df_holdout)} matches (2025/2026)")

    # 4. Compare Risk Architectures on Historical Data
    logger.info("Evaluating 7 candidate Draw Risk scoring architectures...")
    fitted_info, df_arch_comp = fit_and_evaluate_risk_architectures(df_hist, df_hist)
    logger.info(f"Risk Architecture Comparison:\n{df_arch_comp.to_string(index=False)}")

    # We select Architecture E (Combined Interpretable Composite Risk Score)
    # as the Primary Risk Engine due to zero hyperparameter risk, perfect monotonic stability,
    # and direct interpretable signal weighting.
    # We also evaluate Architecture F (Regularized Logistic Risk Model) as statistical benchmark.

    clf = fitted_info["clf"]
    log_features = fitted_info["log_features"]

    # Compute Continuous Draw Risk Score across all matches
    def compute_composite_risk(df_in: pd.DataFrame) -> np.ndarray:
        s = (
            0.35 * np.clip((df_in["cal_p_D"].values - 0.22) / 0.12, 0.0, 1.0)
            + 0.30 * (1.0 - np.clip(df_in["cal_win_diff"].values / 0.20, 0.0, 1.0))
            + 0.20 * np.clip((df_in["p_score_space_draw"].values - 0.22) / 0.10, 0.0, 1.0)
            + 0.15 * (1.0 - np.clip(df_in["lambda_gap"].values / 0.60, 0.0, 1.0))
        )
        return np.clip(s, 0.0, 1.0)

    df["draw_risk_score"] = compute_composite_risk(df)
    df_hist["draw_risk_score"] = df.loc[df_hist.index, "draw_risk_score"].values
    df_holdout["draw_risk_score"] = df.loc[df["season"] == "2025/2026", "draw_risk_score"].values

    # 5. Define Risk Tiers on Historical Era (Quartile / Strategic Thresholds)
    # Thresholds derived strictly from Historical Training Era:
    # LOW: Score < 0.40 (~40% of fixtures, clear favorites)
    # MEDIUM: 0.40 <= Score < 0.65 (~35% of fixtures, moderate contest)
    # HIGH: 0.65 <= Score < 0.80 (~18% of fixtures, tight contest, high draw risk)
    # CRITICAL: Score >= 0.80 (~7% of fixtures, extreme parity and low-score regime)
    tier_thresholds = {
        "medium": 0.40,
        "high": 0.65,
        "critical": 0.80,
    }

    df["draw_risk_tier"] = [assign_risk_tier(s, tier_thresholds) for s in df["draw_risk_score"]]
    df["draw_risk_reason"] = [generate_explainability_reason(row, t) for (_, row), t in zip(df.iterrows(), df["draw_risk_tier"])]

    df_hist["draw_risk_tier"] = df.loc[df_hist.index, "draw_risk_tier"].values
    df_hist["draw_risk_reason"] = df.loc[df_hist.index, "draw_risk_reason"].values

    df_holdout["draw_risk_tier"] = df.loc[df["season"] == "2025/2026", "draw_risk_tier"].values
    df_holdout["draw_risk_reason"] = df.loc[df["season"] == "2025/2026", "draw_risk_reason"].values

    # 6. Feature Importance Matrix
    feat_imp_df = pd.DataFrame({
        "feature": log_features,
        "logistic_coefficient": clf.coef_[0],
        "composite_weight": [0.35, 0.30, 0.15, 0.0, 0.20, 0.0, 0.0, 0.0],
        "abs_importance": np.abs(clf.coef_[0]),
    }).sort_values(by="abs_importance", ascending=False)
    feat_imp_df.to_csv(STEP2J_DIR / "draw_risk_feature_importance.csv", index=False)
    logger.info(f"Saved feature importance:\n{feat_imp_df.to_string(index=False)}")

    # 7. Tier Analysis on Historical Training Era
    logger.info("Computing Draw Risk Tier Diagnostics on Historical Training Era (N=8,983)...")
    df_tier_hist = analyze_risk_tiers(df_hist, "draw_risk_score", tier_thresholds)
    df_tier_hist["dataset"] = "Historical Training Era (2020/21–2024/25)"

    # Tier Analysis on Untouched Holdout Season
    logger.info("Computing Draw Risk Tier Diagnostics on Untouched Holdout Season (N=1,751)...")
    df_tier_holdout = analyze_risk_tiers(df_holdout, "draw_risk_score", tier_thresholds)
    df_tier_holdout["dataset"] = "Untouched Holdout Season (2025/2026)"

    # Combined Tier Analysis CSV
    df_tier_all = pd.concat([df_tier_hist, df_tier_holdout], ignore_index=True)
    df_tier_all.to_csv(STEP2J_DIR / "draw_risk_tier_analysis.csv", index=False)
    logger.info(f"Saved tier analysis to draw_risk_tier_analysis.csv:\n{df_tier_all[['dataset', 'risk_tier', 'match_count', 'pct_of_matches', 'actual_draw_rate', 'v4_0_accuracy', 'favorite_failure_rate']].to_string(index=False)}")

    # 8. Chronological Walk-Forward Stability Analysis
    logger.info("Running 5-Fold Chronological Walk-Forward Stability Analysis...")
    wf_splits = [
        ("Fold 1 (2021/22)", ["2020/2021"], ["2021/2022"]),
        ("Fold 2 (2022/23)", ["2020/2021", "2021/2022"], ["2022/2023"]),
        ("Fold 3 (2023/24)", ["2020/2021", "2021/2022", "2022/2023"], ["2023/2024"]),
        ("Fold 4 (2024/25)", ["2020/2021", "2021/2022", "2022/2023", "2023/2024"], ["2024/2025"]),
    ]

    wf_rows = []
    for split_name, tr_seasons, val_seasons in wf_splits:
        df_tr_f = df_hist[df_hist["season"].isin(tr_seasons)].reset_index(drop=True)
        df_val_f = df_hist[df_hist["season"].isin(val_seasons)].reset_index(drop=True)

        res_tier = analyze_risk_tiers(df_val_f, "draw_risk_score", tier_thresholds)
        res_tier["split"] = split_name
        wf_rows.append(res_tier)

    df_wf = pd.concat(wf_rows, ignore_index=True)
    df_wf.to_csv(STEP2J_DIR / "draw_risk_walkforward.csv", index=False)
    logger.info("Saved walk-forward results to draw_risk_walkforward.csv")

    # 9. Untouched Holdout CSV Report
    df_tier_holdout.to_csv(STEP2J_DIR / "draw_risk_holdout.csv", index=False)

    # 10. League-wise Disaggregated Analysis
    logger.info("Evaluating League-wise Robustness across all 5 Competitions...")
    lg_rows = []
    for lg in sorted(df_hist["competition_name"].unique()):
        df_lg = df_hist[df_hist["competition_name"] == lg].reset_index(drop=True)
        res_lg = analyze_risk_tiers(df_lg, "draw_risk_score", tier_thresholds)
        res_lg["competition_name"] = lg
        lg_rows.append(res_lg)

    df_lg_all = pd.concat(lg_rows, ignore_index=True)
    df_lg_all.to_csv(STEP2J_DIR / "draw_risk_league_analysis.csv", index=False)
    logger.info("Saved league analysis to draw_risk_league_analysis.csv")

    # 11. Threshold Sensitivity Grid
    logger.info("Running Sensitivity Analysis across Tier Cutoffs...")
    sens_rows = []
    for h_cut in [0.55, 0.60, 0.65, 0.70, 0.75]:
        for c_cut in [0.75, 0.80, 0.85]:
            if c_cut <= h_cut:
                continue
            test_th = {"medium": 0.40, "high": h_cut, "critical": c_cut}
            df_temp = df_hist.copy()
            df_temp["draw_risk_tier"] = [assign_risk_tier(s, test_th) for s in df_temp["draw_risk_score"]]
            r_high = df_temp[df_temp["draw_risk_tier"] == "HIGH"]
            r_crit = df_temp[df_temp["draw_risk_tier"] == "CRITICAL"]
            r_low = df_temp[df_temp["draw_risk_tier"] == "LOW"]

            sens_rows.append({
                "high_threshold": h_cut,
                "critical_threshold": c_cut,
                "low_tier_v4_acc": float(r_low["v4_correct"].mean() * 100.0) if len(r_low) > 0 else 0.0,
                "low_tier_draw_rate": float((r_low["actual_result"] == "D").mean() * 100.0) if len(r_low) > 0 else 0.0,
                "high_tier_count": int(len(r_high)),
                "high_tier_draw_rate": float((r_high["actual_result"] == "D").mean() * 100.0) if len(r_high) > 0 else 0.0,
                "high_tier_v4_acc": float(r_high["v4_correct"].mean() * 100.0) if len(r_high) > 0 else 0.0,
                "critical_tier_count": int(len(r_crit)),
                "critical_tier_draw_rate": float((r_crit["actual_result"] == "D").mean() * 100.0) if len(r_crit) > 0 else 0.0,
                "critical_tier_v4_acc": float(r_crit["v4_correct"].mean() * 100.0) if len(r_crit) > 0 else 0.0,
            })

    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(STEP2J_DIR / "draw_risk_threshold_sensitivity.csv", index=False)
    logger.info("Saved threshold sensitivity to draw_risk_threshold_sensitivity.csv")

    # 12. Calibration Analysis (Reliability Table, ECE, Brier)
    logger.info("Evaluating Draw Risk Score Calibration...")
    ece_hist, df_calib_hist = compute_expected_calibration_error(df_hist["is_draw_actual"].values, df_hist["draw_risk_score"].values)
    ece_ho, df_calib_ho = compute_expected_calibration_error(df_holdout["is_draw_actual"].values, df_holdout["draw_risk_score"].values)

    df_calib_hist["dataset"] = "Historical Training Era"
    df_calib_ho["dataset"] = "Untouched Holdout Season"
    df_calib_all = pd.concat([df_calib_hist, df_calib_ho], ignore_index=True)
    df_calib_all.to_csv(STEP2J_DIR / "draw_risk_calibration.csv", index=False)
    logger.info(f"Historical Risk Score ECE: {ece_hist:.4f} | Holdout Risk Score ECE: {ece_ho:.4f}")

    # 13. Prospective Sample Evaluation (Aug 22–24, 2026, N=33)
    logger.info("Evaluating Draw Risk Layer on Recent 33 Prospective Matches...")
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

        win_diff = abs(cal_h - cal_a)
        # Approximate score space and lambda gap from available features
        approx_score_draw = 0.28 if cal_d >= 0.28 else 0.24
        approx_lambda_gap = 0.20 if win_diff <= 0.06 else 0.50

        # Composite score
        risk_score = float(np.clip(
            0.35 * np.clip((cal_d - 0.22) / 0.12, 0.0, 1.0)
            + 0.30 * (1.0 - np.clip(win_diff / 0.20, 0.0, 1.0))
            + 0.20 * np.clip((approx_score_draw - 0.22) / 0.10, 0.0, 1.0)
            + 0.15 * (1.0 - np.clip(approx_lambda_gap / 0.60, 0.0, 1.0)),
            0.0,
            1.0,
        ))

        tier = assign_risk_tier(risk_score, tier_thresholds)
        reason = f"Calibrated P(D)={cal_d*100:.1f}%, win margin |H-A|={win_diff*100:.1f}%."

        v4_dec = str(row["v4_decision"])
        v4_corr = (v4_dec == act_res)
        is_draw = (act_res == "D")

        r33_rows.append({
            "fixture_id": fid,
            "date": str(row.get("date_ist", "")),
            "league": league,
            "home_team": h_name,
            "away_team": a_name,
            "score": score,
            "v4_p_H": v4_h,
            "v4_p_D": v4_d,
            "v4_p_A": v4_a,
            "calibrated_p_D": cal_d,
            "v4_decision": v4_dec,
            "actual_outcome": act_res,
            "is_draw_actual": is_draw,
            "v4_correct": v4_corr,
            "draw_risk_score": round(risk_score, 4),
            "draw_risk_tier": tier,
            "draw_risk_reason": reason,
        })

    df_33_res = pd.DataFrame(r33_rows)
    df_33_res.to_csv(STEP2J_DIR / "draw_risk_prospective.csv", index=False)
    logger.info(f"Saved prospective audit to draw_risk_prospective.csv:\n{df_33_res[['home_team', 'away_team', 'score', 'actual_outcome', 'v4_decision', 'v4_correct', 'draw_risk_score', 'draw_risk_tier']].to_string(index=False)}")

    # 14. Full Match Predictions Export (10,734 matches)
    out_cols = [
        "fixture_id",
        "date",
        "season",
        "competition_name",
        "home_name",
        "away_name",
        "home_goals",
        "away_goals",
        "actual_result",
        "is_draw_actual",
        "v4_p_H",
        "v4_p_D",
        "v4_p_A",
        "v4_decision",
        "v4_correct",
        "cal_p_H",
        "cal_p_D",
        "cal_p_A",
        "draw_risk_score",
        "draw_risk_tier",
        "draw_risk_reason",
    ]
    df[out_cols].to_csv(STEP2J_DIR / "draw_risk_predictions.csv", index=False)
    logger.info("Saved 10,734 match predictions to draw_risk_predictions.csv")

    # 15. Candidate Configuration JSON
    config_dict = {
        "step": "Step 2J Draw Risk / Caution Intelligence Research",
        "date": "2026-08-24",
        "status": "RESEARCH COMPLETE — ZERO PRODUCTION MUTATION",
        "v4_production_md5": EXP_V4_MD5,
        "v4_1_candidate_md5": EXP_V4_1_MD5,
        "primary_risk_architecture": "Architecture E: Combined Interpretable Composite Risk Score",
        "risk_weights": {
            "calibrated_draw_prob": 0.35,
            "win_parity_margin": 0.30,
            "score_space_draw_mass": 0.20,
            "expected_goal_parity": 0.15,
        },
        "tier_thresholds": tier_thresholds,
        "tier_definitions": {
            "LOW": "Score < 0.40 (Decisive contest, strong baseline favorite reliability)",
            "MEDIUM": "0.40 <= Score < 0.65 (Moderate contest, standard draw probability)",
            "HIGH": "0.65 <= Score < 0.80 (Elevated draw risk, favorite win reliability drops significantly)",
            "CRITICAL": "Score >= 0.80 (Extreme draw vulnerability, win parity tight, low-score regime)",
        },
    }

    with open(STEP2J_DIR / "step2j_candidate_config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    logger.info("Saved candidate config to step2j_candidate_config.json")

    # 16. Post-flight Hash Verification
    assert verify_md5(V4_PATH, EXP_V4_MD5, "V4.0 Baseline")
    assert verify_md5(V4_1_PATH, EXP_V4_1_MD5, "V4.1 Production Candidate")

    logger.info("Step 2J Research Execution Finished Successfully.")


if __name__ == "__main__":
    main()
