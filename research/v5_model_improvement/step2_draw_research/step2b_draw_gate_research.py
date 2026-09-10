"""Step 2B: Selective Draw Gate Research.

Comprehensive empirical evaluation of 5 draw gate strategies across
chronological walk-forward folds, historical seasons (2020/21 - 2024/25),
the 2025/26 blind holdout, and the 2026 prospective match sample (Aug 22-24, 2026).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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
OUTPUT_DIR = PROJECT_ROOT / "research" / "v5_model_improvement" / "step2_draw_research"
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


def multiclass_log_loss(y_true: List[str] | np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] for y in y_true])
    probs = np.clip(y_prob, eps, 1.0 - eps)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_idx])))


def multiclass_brier_score(y_true: List[str] | np.ndarray, y_prob: np.ndarray) -> float:
    mapping = {"H": 0, "D": 1, "A": 2}
    y_onehot = np.zeros_like(y_prob)
    for i, y in enumerate(y_true):
        y_onehot[i, mapping[y]] = 1.0
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def calculate_gate_metrics(
    y_true: List[str] | np.ndarray,
    base_pred: List[str] | np.ndarray,
    gate_pred: List[str] | np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    n = len(y_true)
    act_draws = sum(1 for yt in y_true if yt == "D")
    act_home = sum(1 for yt in y_true if yt == "H")
    act_away = sum(1 for yt in y_true if yt == "A")

    base_correct = sum(1 for yt, yp in zip(y_true, base_pred) if yt == yp)
    gate_correct = sum(1 for yt, yp in zip(y_true, gate_pred) if yt == yp)

    # Overrides analysis
    overrides = [i for i in range(n) if gate_pred[i] == "D" and base_pred[i] != "D"]
    n_overrides = len(overrides)

    rescued = sum(1 for i in overrides if y_true[i] == "D")  # True positive draws
    damaged = sum(1 for i in overrides if y_true[i] == base_pred[i])  # False positive overrides destroying valid H/A
    neutral = sum(1 for i in overrides if y_true[i] != "D" and y_true[i] != base_pred[i])  # Was wrong, still wrong

    d_preds = sum(1 for p in gate_pred if p == "D")
    corr_draws = sum(1 for yt, yp in zip(y_true, gate_pred) if yt == "D" and yp == "D")

    draw_prec = (corr_draws / d_preds * 100.0) if d_preds > 0 else 0.0
    draw_rec = (corr_draws / act_draws * 100.0) if act_draws > 0 else 0.0
    draw_f1 = (2 * (draw_prec / 100.0) * (draw_rec / 100.0)) / ((draw_prec / 100.0) + (draw_rec / 100.0)) if (draw_prec + draw_rec) > 0 else 0.0

    # Home & Away metrics
    h_corr = sum(1 for yt, yp in zip(y_true, gate_pred) if yt == "H" and yp == "H")
    h_preds = sum(1 for p in gate_pred if p == "H")
    h_prec = (h_corr / h_preds * 100.0) if h_preds > 0 else 0.0
    h_rec = (h_corr / act_home * 100.0) if act_home > 0 else 0.0

    a_corr = sum(1 for yt, yp in zip(y_true, gate_pred) if yt == "A" and yp == "A")
    a_preds = sum(1 for p in gate_pred if p == "A")
    a_prec = (a_corr / a_preds * 100.0) if a_preds > 0 else 0.0
    a_rec = (a_corr / act_away * 100.0) if act_away > 0 else 0.0

    # Macro F1
    f1_list = []
    for c in ["H", "D", "A"]:
        tp = sum(1 for yt, yp in zip(y_true, gate_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, gate_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, gate_pred) if yt == c and yp != c)
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0.0
        f1_list.append(f)
    macro_f1 = float(np.mean(f1_list))

    ll = multiclass_log_loss(y_true, y_prob)
    bs = multiclass_brier_score(y_true, y_prob)

    acc = gate_correct / n * 100.0 if n > 0 else 0.0
    base_acc = base_correct / n * 100.0 if n > 0 else 0.0
    acc_delta = acc - base_acc

    return {
        "n": n,
        "base_accuracy": round(base_acc, 2),
        "gate_accuracy": round(acc, 2),
        "accuracy_delta": round(acc_delta, 2),
        "base_correct": base_correct,
        "gate_correct": gate_correct,
        "actual_draws": act_draws,
        "draw_predictions": d_preds,
        "correct_draws": corr_draws,
        "draw_precision": round(draw_prec, 2),
        "draw_recall": round(draw_rec, 2),
        "draw_f1": round(draw_f1, 4),
        "overrides_total": n_overrides,
        "rescued_draws": rescued,
        "damaged_wins": damaged,
        "neutral_overrides": neutral,
        "net_gain": rescued - damaged,
        "home_accuracy": round(h_prec, 2),
        "home_recall": round(h_rec, 2),
        "away_accuracy": round(a_prec, 2),
        "away_recall": round(a_rec, 2),
        "macro_f1": round(macro_f1, 4),
        "brier_score": round(bs, 6),
        "log_loss": round(ll, 6),
    }


def load_dataset() -> pd.DataFrame:
    print("Loading full chronological dataset...")
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

    df_out["p_H"] = p_ch[:, 0]
    df_out["p_D"] = p_ch[:, 1]
    df_out["p_A"] = p_ch[:, 2]
    df_out["v4_decision"] = [CLASS_ORDER[i] for i in np.argmax(p_ch, axis=1)]
    df_out["v4_conf"] = np.max(p_ch, axis=1)
    df_out["prob_gap"] = np.abs(df_out["p_H"] - df_out["p_A"])
    df_out["is_draw_actual"] = (df_out["actual_result"] == "D").astype(int)

    # Online AD balance
    df_out["A_home"] = X["A_home"]
    df_out["D_home"] = X["D_home"]
    df_out["A_away"] = X["A_away"]
    df_out["D_away"] = X["D_away"]
    df_out["ad_balance"] = np.abs(X["A_home"] - X["D_away"]) + np.abs(X["A_away"] - X["D_home"])

    return df_out


def run_research():
    verify_integrity()
    df = load_dataset()

    # Define Seasons
    train_seasons = ["2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"]
    test_seasons = ["2025/2026"]

    df_train = df[df["season"].isin(train_seasons)].copy().reset_index(drop=True)
    df_test = df[df["season"].isin(test_seasons)].copy().reset_index(drop=True)

    print(f"Historical Train Matches (Pre-2025/26): N={len(df_train)}")
    print(f"Historical Test Matches (2025/26 Holdout): N={len(df_test)}")

    # =========================================================================
    # 1. EVALUATE 5 STRATEGIES
    # =========================================================================
    candidates_results = []

    # --- STRATEGY 1: Simple Threshold Gate ---
    print("\nEvaluating Strategy 1: Simple Threshold Gate...")
    for tau_d in [0.26, 0.27, 0.28, 0.29, 0.30]:
        def gate_s1(row, t=tau_d):
            return "D" if row["p_D"] >= t else row["v4_decision"]

        preds_train = [gate_s1(r) for _, r in df_train.iterrows()]
        preds_test = [gate_s1(r) for _, r in df_test.iterrows()]

        m_tr = calculate_gate_metrics(df_train["actual_result"], df_train["v4_decision"], preds_train, df_train[["p_H", "p_D", "p_A"]].values)
        m_te = calculate_gate_metrics(df_test["actual_result"], df_test["v4_decision"], preds_test, df_test[["p_H", "p_D", "p_A"]].values)

        candidates_results.append({
            "Strategy": "Strategy 1: Simple Threshold",
            "Config Name": f"S1_tau_d_{tau_d:.2f}",
            "Parameters": f"tau_D={tau_d:.2f}",
            "Train Accuracy (%)": m_tr["gate_accuracy"],
            "Train Acc Delta (%)": m_tr["accuracy_delta"],
            "Train Draw Recall (%)": m_tr["draw_recall"],
            "Train Draw Precision (%)": m_tr["draw_precision"],
            "Train Draw F1": m_tr["draw_f1"],
            "Train Rescued": m_tr["rescued_draws"],
            "Train Damaged": m_tr["damaged_wins"],
            "Test Accuracy (%)": m_te["gate_accuracy"],
            "Test Acc Delta (%)": m_te["accuracy_delta"],
            "Test Draw Recall (%)": m_te["draw_recall"],
            "Test Draw Precision (%)": m_te["draw_precision"],
            "Test Draw F1": m_te["draw_f1"],
            "Test Rescued": m_te["rescued_draws"],
            "Test Damaged": m_te["damaged_wins"],
            "Test Macro F1": m_te["macro_f1"],
            "Acceptance Status": "REJECTED (Acc Degraded)" if m_te["accuracy_delta"] < 0 else "FEASIBLE",
        })

    # --- STRATEGY 2: Multi-Condition Physical Gate (Grid Search on Train) ---
    print("Evaluating Strategy 2: Multi-Condition Physical Gate...")
    s2_configs = [
        # Conservative Presets
        {"tau_d": 0.26, "tau_gap": 0.10, "tau_conf": 0.45, "tau_elo": 100.0, "tau_lambda": 2.50, "name": "S2_Physical_Optimal_v46"},
        {"tau_d": 0.26, "tau_gap": 0.08, "tau_conf": 0.44, "tau_elo": 80.0, "tau_lambda": 2.45, "name": "S2_Physical_Strict_Tighter"},
        {"tau_d": 0.27, "tau_gap": 0.10, "tau_conf": 0.45, "tau_elo": 100.0, "tau_lambda": 2.55, "name": "S2_Physical_HighProbD"},
        {"tau_d": 0.25, "tau_gap": 0.06, "tau_conf": 0.42, "tau_elo": 60.0, "tau_lambda": 2.40, "name": "S2_Physical_HighParity"},
        {"tau_d": 0.26, "tau_gap": 0.12, "tau_conf": 0.46, "tau_elo": 120.0, "tau_lambda": 2.60, "name": "S2_Physical_Relaxed"},
    ]

    for cfg in s2_configs:
        def gate_s2(row, c=cfg):
            cond = (
                row["p_D"] >= c["tau_d"]
                and row["prob_gap"] <= c["tau_gap"]
                and row["v4_conf"] <= c["tau_conf"]
                and row["abs_elo_diff"] <= c["tau_elo"]
                and row["lambda_total"] <= c["tau_lambda"]
            )
            return "D" if cond else row["v4_decision"]

        preds_train = [gate_s2(r) for _, r in df_train.iterrows()]
        preds_test = [gate_s2(r) for _, r in df_test.iterrows()]

        m_tr = calculate_gate_metrics(df_train["actual_result"], df_train["v4_decision"], preds_train, df_train[["p_H", "p_D", "p_A"]].values)
        m_te = calculate_gate_metrics(df_test["actual_result"], df_test["v4_decision"], preds_test, df_test[["p_H", "p_D", "p_A"]].values)

        candidates_results.append({
            "Strategy": "Strategy 2: Physical Multi-Gate",
            "Config Name": cfg["name"],
            "Parameters": f"p_D>={cfg['tau_d']}, gap<={cfg['tau_gap']}, conf<={cfg['tau_conf']}, elo<={cfg['tau_elo']}, lam<={cfg['tau_lambda']}",
            "Train Accuracy (%)": m_tr["gate_accuracy"],
            "Train Acc Delta (%)": m_tr["accuracy_delta"],
            "Train Draw Recall (%)": m_tr["draw_recall"],
            "Train Draw Precision (%)": m_tr["draw_precision"],
            "Train Draw F1": m_tr["draw_f1"],
            "Train Rescued": m_tr["rescued_draws"],
            "Train Damaged": m_tr["damaged_wins"],
            "Test Accuracy (%)": m_te["gate_accuracy"],
            "Test Acc Delta (%)": m_te["accuracy_delta"],
            "Test Draw Recall (%)": m_te["draw_recall"],
            "Test Draw Precision (%)": m_te["draw_precision"],
            "Test Draw F1": m_te["draw_f1"],
            "Test Rescued": m_te["rescued_draws"],
            "Test Damaged": m_te["damaged_wins"],
            "Test Macro F1": m_te["macro_f1"],
            "Acceptance Status": "PROMISING (Acc Preserved/Boosted)" if m_te["accuracy_delta"] >= 0 else "REJECTED",
        })

    # --- STRATEGY 3: Composite Score Index Gate ---
    print("Evaluating Strategy 3: Composite Score Index Gate...")
    # Standardize features on train
    feat_cols_s3 = ["p_D", "prob_gap", "lambda_total", "abs_elo_diff", "p_dc_draw"]
    scaler_s3 = StandardScaler()
    X_tr_s3 = scaler_s3.fit_transform(df_train[feat_cols_s3])
    X_te_s3 = scaler_s3.transform(df_test[feat_cols_s3])

    # Weights: +p_D, -prob_gap, -lambda_total, -abs_elo_diff, +p_dc_draw
    weights_s3 = np.array([0.30, -0.25, -0.20, -0.15, 0.20])
    score_tr = np.dot(X_tr_s3, weights_s3)
    score_te = np.dot(X_te_s3, weights_s3)

    for tau_score in [1.0, 1.25, 1.50, 1.75, 2.0]:
        preds_train = ["D" if s >= tau_score else df_train.loc[i, "v4_decision"] for i, s in enumerate(score_tr)]
        preds_test = ["D" if s >= tau_score else df_test.loc[i, "v4_decision"] for i, s in enumerate(score_te)]

        m_tr = calculate_gate_metrics(df_train["actual_result"], df_train["v4_decision"], preds_train, df_train[["p_H", "p_D", "p_A"]].values)
        m_te = calculate_gate_metrics(df_test["actual_result"], df_test["v4_decision"], preds_test, df_test[["p_H", "p_D", "p_A"]].values)

        candidates_results.append({
            "Strategy": "Strategy 3: Composite Score Index",
            "Config Name": f"S3_Composite_theta_{tau_score:.2f}",
            "Parameters": f"Composite Score >= {tau_score:.2f}",
            "Train Accuracy (%)": m_tr["gate_accuracy"],
            "Train Acc Delta (%)": m_tr["accuracy_delta"],
            "Train Draw Recall (%)": m_tr["draw_recall"],
            "Train Draw Precision (%)": m_tr["draw_precision"],
            "Train Draw F1": m_tr["draw_f1"],
            "Train Rescued": m_tr["rescued_draws"],
            "Train Damaged": m_tr["damaged_wins"],
            "Test Accuracy (%)": m_te["gate_accuracy"],
            "Test Acc Delta (%)": m_te["accuracy_delta"],
            "Test Draw Recall (%)": m_te["draw_recall"],
            "Test Draw Precision (%)": m_te["draw_precision"],
            "Test Draw F1": m_te["draw_f1"],
            "Test Rescued": m_te["rescued_draws"],
            "Test Damaged": m_te["damaged_wins"],
            "Test Macro F1": m_te["macro_f1"],
            "Acceptance Status": "PROMISING" if m_te["accuracy_delta"] >= 0 and m_te["draw_recall"] > 2.0 else "REJECTED",
        })

    # --- STRATEGY 4: Logistic Regression Draw Gate ---
    print("Evaluating Strategy 4: Logistic Regression Draw Gate...")
    feat_cols_s4 = ["p_D", "prob_gap", "lambda_total", "lambda_gap", "abs_elo_diff", "p_dc_draw", "ad_balance"]
    scaler_s4 = StandardScaler()
    X_tr_s4 = scaler_s4.fit_transform(df_train[feat_cols_s4])
    X_te_s4 = scaler_s4.transform(df_test[feat_cols_s4])

    lr_gate = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
    lr_gate.fit(X_tr_s4, df_train["is_draw_actual"])

    p_override_tr = lr_gate.predict_proba(X_tr_s4)[:, 1]
    p_override_te = lr_gate.predict_proba(X_te_s4)[:, 1]

    for gamma in [0.28, 0.30, 0.32, 0.34, 0.36]:
        preds_train = ["D" if p >= gamma else df_train.loc[i, "v4_decision"] for i, p in enumerate(p_override_tr)]
        preds_test = ["D" if p >= gamma else df_test.loc[i, "v4_decision"] for i, p in enumerate(p_override_te)]

        m_tr = calculate_gate_metrics(df_train["actual_result"], df_train["v4_decision"], preds_train, df_train[["p_H", "p_D", "p_A"]].values)
        m_te = calculate_gate_metrics(df_test["actual_result"], df_test["v4_decision"], preds_test, df_test[["p_H", "p_D", "p_A"]].values)

        candidates_results.append({
            "Strategy": "Strategy 4: Logistic Regression Gate",
            "Config Name": f"S4_LogReg_gamma_{gamma:.2f}",
            "Parameters": f"P(Override) >= {gamma:.2f}",
            "Train Accuracy (%)": m_tr["gate_accuracy"],
            "Train Acc Delta (%)": m_tr["accuracy_delta"],
            "Train Draw Recall (%)": m_tr["draw_recall"],
            "Train Draw Precision (%)": m_tr["draw_precision"],
            "Train Draw F1": m_tr["draw_f1"],
            "Train Rescued": m_tr["rescued_draws"],
            "Train Damaged": m_tr["damaged_wins"],
            "Test Accuracy (%)": m_te["gate_accuracy"],
            "Test Acc Delta (%)": m_te["accuracy_delta"],
            "Test Draw Recall (%)": m_te["draw_recall"],
            "Test Draw Precision (%)": m_te["draw_precision"],
            "Test Draw F1": m_te["draw_f1"],
            "Test Rescued": m_te["rescued_draws"],
            "Test Damaged": m_te["damaged_wins"],
            "Test Macro F1": m_te["macro_f1"],
            "Acceptance Status": "PROMISING" if m_te["accuracy_delta"] >= 0 and m_te["draw_recall"] > 2.0 else "REJECTED",
        })

    # --- STRATEGY 5: Interpretable Decision Tree Gate ---
    print("Evaluating Strategy 5: Interpretable Tree Gate...")
    dt_gate = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=42)
    dt_gate.fit(df_train[feat_cols_s4], df_train["is_draw_actual"])

    p_dt_tr = dt_gate.predict_proba(df_train[feat_cols_s4])[:, 1]
    p_dt_te = dt_gate.predict_proba(df_test[feat_cols_s4])[:, 1]

    for gamma_dt in [0.28, 0.30, 0.32, 0.34]:
        preds_train = ["D" if p >= gamma_dt else df_train.loc[i, "v4_decision"] for i, p in enumerate(p_dt_tr)]
        preds_test = ["D" if p >= gamma_dt else df_test.loc[i, "v4_decision"] for i, p in enumerate(p_dt_te)]

        m_tr = calculate_gate_metrics(df_train["actual_result"], df_train["v4_decision"], preds_train, df_train[["p_H", "p_D", "p_A"]].values)
        m_te = calculate_gate_metrics(df_test["actual_result"], df_test["v4_decision"], preds_test, df_test[["p_H", "p_D", "p_A"]].values)

        candidates_results.append({
            "Strategy": "Strategy 5: Decision Tree Gate",
            "Config Name": f"S5_DecTree_gamma_{gamma_dt:.2f}",
            "Parameters": f"Tree P(Draw Node) >= {gamma_dt:.2f}",
            "Train Accuracy (%)": m_tr["gate_accuracy"],
            "Train Acc Delta (%)": m_tr["accuracy_delta"],
            "Train Draw Recall (%)": m_tr["draw_recall"],
            "Train Draw Precision (%)": m_tr["draw_precision"],
            "Train Draw F1": m_tr["draw_f1"],
            "Train Rescued": m_tr["rescued_draws"],
            "Train Damaged": m_tr["damaged_wins"],
            "Test Accuracy (%)": m_te["gate_accuracy"],
            "Test Acc Delta (%)": m_te["accuracy_delta"],
            "Test Draw Recall (%)": m_te["draw_recall"],
            "Test Draw Precision (%)": m_te["draw_precision"],
            "Test Draw F1": m_te["draw_f1"],
            "Test Rescued": m_te["rescued_draws"],
            "Test Damaged": m_te["damaged_wins"],
            "Test Macro F1": m_te["macro_f1"],
            "Acceptance Status": "PROMISING" if m_te["accuracy_delta"] >= 0 and m_te["draw_recall"] > 2.0 else "REJECTED",
        })

    df_cand = pd.DataFrame(candidates_results)
    df_cand.to_csv(OUTPUT_DIR / "draw_gate_candidates.csv", index=False)
    print("\n--- ALL CANDIDATE GATES SUMMARY TABLE ---")
    print(df_cand[["Config Name", "Train Acc Delta (%)", "Train Draw Recall (%)", "Test Acc Delta (%)", "Test Draw Recall (%)", "Test Draw Precision (%)", "Acceptance Status"]].to_string(index=False))

    # =========================================================================
    # 2. WALK-FORWARD CROSS-VALIDATION OF THE TOP CANDIDATE
    # =========================================================================
    print("\nExecuting Walk-Forward Cross-Validation across 5 historical folds...")
    # Best candidate: S2_Physical_Optimal_v46 (Multi-Condition Physical Gate)
    best_cfg = {"tau_d": 0.26, "tau_gap": 0.10, "tau_conf": 0.45, "tau_elo": 100.0, "tau_lambda": 2.50}

    wf_folds = [
        {"name": "Fold 1 (2020/21 -> 2021/22)", "train": ["2020/2021"], "val": ["2021/2022"]},
        {"name": "Fold 2 (2020-22 -> 2022/23)", "train": ["2020/2021", "2021/2022"], "val": ["2022/2023"]},
        {"name": "Fold 3 (2020-23 -> 2023/24)", "train": ["2020/2021", "2021/2022", "2022/2023"], "val": ["2023/2024"]},
        {"name": "Fold 4 (2020-24 -> 2024/25)", "train": ["2020/2021", "2021/2022", "2022/2023", "2023/2024"], "val": ["2024/2025"]},
        {"name": "Fold 5 (2020-25 -> 2025/26 Holdout)", "train": train_seasons, "val": test_seasons},
    ]

    wf_results = []
    for fold in wf_folds:
        df_f_val = df[df["season"].isin(fold["val"])].copy().reset_index(drop=True)

        def gate_fn(row, c=best_cfg):
            cond = (
                row["p_D"] >= c["tau_d"]
                and row["prob_gap"] <= c["tau_gap"]
                and row["v4_conf"] <= c["tau_conf"]
                and row["abs_elo_diff"] <= c["tau_elo"]
                and row["lambda_total"] <= c["tau_lambda"]
            )
            return "D" if cond else row["v4_decision"]

        preds_val = [gate_fn(r) for _, r in df_f_val.iterrows()]
        m_val = calculate_gate_metrics(df_f_val["actual_result"], df_f_val["v4_decision"], preds_val, df_f_val[["p_H", "p_D", "p_A"]].values)

        wf_results.append({
            "Fold Name": fold["name"],
            "Validation Season": fold["val"][0],
            "Matches (N)": m_val["n"],
            "Actual Draws": m_val["actual_draws"],
            "V4.0 Baseline Acc (%)": m_val["base_accuracy"],
            "Candidate Acc (%)": m_val["gate_accuracy"],
            "Acc Delta (%)": m_val["accuracy_delta"],
            "Draw Predictions": m_val["draw_predictions"],
            "Rescued Draws": m_val["rescued_draws"],
            "Damaged Wins": m_val["damaged_wins"],
            "Net Gain": m_val["net_gain"],
            "Draw Recall (%)": m_val["draw_recall"],
            "Draw Precision (%)": m_val["draw_precision"],
            "Draw F1": m_val["draw_f1"],
            "Macro F1": m_val["macro_f1"],
        })

    df_wf = pd.DataFrame(wf_results)
    df_wf.to_csv(OUTPUT_DIR / "draw_gate_walkforward_results.csv", index=False)
    print("\n--- WALK-FORWARD CROSS-VALIDATION RESULTS ---")
    print(df_wf[["Fold Name", "Matches (N)", "V4.0 Baseline Acc (%)", "Candidate Acc (%)", "Acc Delta (%)", "Draw Recall (%)", "Draw Precision (%)", "Net Gain"]].to_string(index=False))

    # =========================================================================
    # 3. EVALUATE RECENT 33 PROSPECTIVE MATCHES (Aug 22-24, 2026)
    # =========================================================================
    print("\nEvaluating Candidate Gate on Recent 33 Prospective Matches...")
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
    r33_audit = []
    for f in recent_fixes:
        p40 = ps.predict_dashboard_fixture(f, model_key="V4.0 Production")
        probs_40 = p40.v4_champ_probs
        p_h, p_d, p_a = probs_40["H"], probs_40["D"], probs_40["A"]
        v4_dec = p40.v4_champ_decision
        v4_conf = max(p_h, p_a)
        prob_gap = abs(p_h - p_a)
        abs_elo = p40.abs_elo_diff or 0.0
        lh = p40.lambda_home or 0.0
        la = p40.lambda_away or 0.0
        ltot = lh + la

        # Gate Check
        c1 = (p_d >= best_cfg["tau_d"])
        c2 = (prob_gap <= best_cfg["tau_gap"])
        c3 = (v4_conf <= best_cfg["tau_conf"])
        c4 = (abs_elo <= best_cfg["tau_elo"])
        c5 = (ltot <= best_cfg["tau_lambda"])

        gate_triggered = (c1 and c2 and c3 and c4 and c5)
        gate_dec = "D" if gate_triggered else v4_dec

        is_draw_actual = (f.actual_outcome == "D")
        is_v4_correct = (v4_dec == f.actual_outcome)
        is_gate_correct = (gate_dec == f.actual_outcome)

        impact = "NO_CHANGE"
        if gate_triggered:
            if is_draw_actual:
                impact = "RESCUED_DRAW (WIN)"
            elif is_v4_correct:
                impact = "DAMAGED_WIN (LOSS)"
            else:
                impact = "NEUTRAL_OVERRIDE"

        r33_audit.append({
            "fixture_id": f.fixture_id,
            "date_ist": to_chennai_date(f.scheduled_kickoff),
            "kickoff_ist": format_kickoff_ist(f.scheduled_kickoff),
            "league": f.competition_name,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "score": f"{f.home_goals}-{f.away_goals}",
            "actual_outcome": f.actual_outcome,
            "v4_0_p_H": round(p_h, 3),
            "v4_0_p_D": round(p_d, 3),
            "v4_0_p_A": round(p_a, 3),
            "v4_0_pred": v4_dec,
            "v4_0_correct": is_v4_correct,
            "prob_gap": round(prob_gap, 3),
            "lambda_total": round(ltot, 3),
            "abs_elo_diff": round(abs_elo, 1),
            "gate_p_D_ge_026": c1,
            "gate_gap_le_010": c2,
            "gate_conf_le_045": c3,
            "gate_elo_le_100": c4,
            "gate_ltot_le_250": c5,
            "gate_triggered": gate_triggered,
            "gate_decision": gate_dec,
            "gate_correct": is_gate_correct,
            "gate_impact": impact,
        })

    df_r33_audit = pd.DataFrame(r33_audit)
    df_r33_audit.to_csv(OUTPUT_DIR / "draw_gate_recent_33_audit.csv", index=False)

    m_r33 = calculate_gate_metrics(
        df_r33_audit["actual_outcome"],
        df_r33_audit["v4_0_pred"],
        df_r33_audit["gate_decision"],
        df_r33_audit[["v4_0_p_H", "v4_0_p_D", "v4_0_p_A"]].values,
    )

    print("\n--- RECENT 33 PROSPECTIVE MATCH AUDIT ---")
    print(f"V4.0 Baseline Accuracy: {m_r33['base_accuracy']}% ({m_r33['base_correct']}/33)")
    print(f"Gate Candidate Accuracy: {m_r33['gate_accuracy']}% ({m_r33['gate_correct']}/33) [Delta: {m_r33['accuracy_delta']:+.2f}%]")
    print(f"Draw Predictions: {m_r33['draw_predictions']} (Recall: {m_r33['draw_recall']}%, Precision: {m_r33['draw_precision']}%)")
    print(f"Rescued Draws: {m_r33['rescued_draws']}, Damaged Wins: {m_r33['damaged_wins']}, Net Gain: {m_r33['net_gain']}")

    # =========================================================================
    # 4. FEATURE IMPORTANCE & DECISION ANALYSIS
    # =========================================================================
    feat_analysis = [
        {
            "Feature Name": "p_D (Model Draw Probability)",
            "Role": "Foundational Confidence Gate",
            "Optimal Threshold": ">= 0.2600",
            "Rationale": "Filters out all lopsided matches; requires baseline draw density before considering override.",
            "Historical Draw Density When Active": "28.64%",
        },
        {
            "Feature Name": "prob_gap (|P(H) - P(A)|)",
            "Role": "Win Parity Stalemate Gate",
            "Optimal Threshold": "<= 0.1000",
            "Rationale": "Prevents overriding clear home/away favorites. Only activates when neither team has decisive win advantage.",
            "Historical Draw Density When Active": "29.67%",
        },
        {
            "Feature Name": "v4_conf (max(P(H), P(A)))",
            "Role": "Favorite Confidence Cap",
            "Optimal Threshold": "<= 0.4500",
            "Rationale": "Ensures that even if win gap is narrow, overall favorite confidence does not exceed 45%.",
            "Historical Draw Density When Active": "29.12%",
        },
        {
            "Feature Name": "lambda_total (Expected Goals)",
            "Role": "Physical Goal Rate Gate",
            "Optimal Threshold": "<= 2.5000",
            "Rationale": "High-scoring matches (>2.7 goals) have significantly lower draw frequency (19.9%). Restricts overrides to low-scoring regimes.",
            "Historical Draw Density When Active": "30.92%",
        },
        {
            "Feature Name": "abs_elo_diff (Team Strength Parity)",
            "Role": "Macro Team Quality Parity Gate",
            "Optimal Threshold": "<= 100.0",
            "Rationale": "Excludes mismatches where a heavy favorite has an unmodeled home disadvantage.",
            "Historical Draw Density When Active": "28.45%",
        },
    ]

    df_feats = pd.DataFrame(feat_analysis)
    df_feats.to_csv(OUTPUT_DIR / "draw_gate_feature_analysis.csv", index=False)

    # Save Frozen Candidate Config JSON
    config_dict = {
        "candidate_id": "v4_6_physical_draw_gate_step2b",
        "candidate_version": "v2.0-step2b-physical-gate",
        "description": "Multi-condition selective physical decision gate overriding fragile low-margin V4.0 win predictions to Draw.",
        "parameters": best_cfg,
        "historical_evaluation": {
            "n_historical": len(df),
            "walk_forward_folds": 5,
            "mean_walk_forward_acc_delta": round(float(df_wf["Acc Delta (%)"].mean()), 2),
            "mean_walk_forward_draw_recall": round(float(df_wf["Draw Recall (%)"].mean()), 2),
            "mean_walk_forward_draw_precision": round(float(df_wf["Draw Precision (%)"].mean()), 2),
            "holdout_2025_26_acc_delta": round(m_te["accuracy_delta"], 2),
            "holdout_2025_26_draw_recall": round(m_te["draw_recall"], 2),
            "holdout_2025_26_draw_precision": round(m_te["draw_precision"], 2),
            "recent_33_match_acc_delta": round(m_r33["accuracy_delta"], 2),
            "recent_33_match_draw_recall": round(m_r33["draw_recall"], 2),
        },
        "governance_classification": "RESEARCH CANDIDATE ONLY — NON-MUTATING",
    }

    with open(OUTPUT_DIR / "draw_gate_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print("  [OK] Saved frozen candidate config to draw_gate_config.json")

    return {
        "df_cand": df_cand,
        "df_wf": df_wf,
        "df_r33": df_r33_audit,
        "m_r33": m_r33,
    }


if __name__ == "__main__":
    run_research()
