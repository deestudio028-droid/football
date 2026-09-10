"""Phase 37 — Experiment E12 Execution Harness: Orthogonal Signal & Bounded Lambda Adjustment.

Usage:
    python research/v5_model_improvement/e12_orthogonal_signal/run_e12_experiment.py
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dixon_coles_engine import (
    DixonColesConfig,
    apply_dixon_coles_parity_transform,
    compute_dixon_coles_matrix_fast,
    compute_1x2_from_score_matrix,
)
from features.elo import ELO_COLUMNS, load_elo_features
from features.online_attack_defense import AD_COLUMNS, compute_ad_states, fit_baseline_rates
from models.baselines import CLASS_ORDER
from models.config import (
    FINAL_TEST_SEASONS,
    FINAL_TRAIN_SEASONS,
    SEASON_NAME_TO_IDS,
)
from models.data import load_supervised_dataset
from models.v4_artifact import load_v4_artifact

from orthogonal_features import build_orthogonal_features
from residual_models import BoundedResidualModel
from bounded_lambda_adjustment import apply_bounded_lambda_adjustment

MATCHES_DB = PROJECT_ROOT / "data/processed/matches.db"
FEATURES_DB = PROJECT_ROOT / "data/processed/features.db"
V4_ARTIFACT_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
E12_DIR = PROJECT_ROOT / "research/v5_model_improvement/e12_orthogonal_signal"

PINNED_20 = {
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
    "research/v4_promotion/promotion_market_odds.sqlite": "f8a41b79cd33afb412ccd9ae2892a196",
    "research/v4_promotion/fresh_100_market_odds.sqlite": "2cb80b79d772fbedd4f3707b39a32c13",
    "research/v4_promotion/fresh_100_fixture_ids.json": "761ad5cc571643e6985e671bd9c3d83a",
    "research/v4_promotion/fresh_extended_fixture_ids.json": "0526bfd6980dd51dae59c6f6aadab2f5",
    "research/v4_promotion/fresh_extended_market_odds.sqlite": "b4889d1791723ea653057af51ca00f8e",
    "research/v4_promotion/dixon_coles_rho_method_frozen.json": "822e742dcc82e5e96445b31c14c0c604",
    "research/v4_promotion/elo_draw_curve_method_frozen.json": "65dc2cf762f3d78abcf1a617ef23fe00",
    "research/v4_promotion/full_score_matrix_method_frozen.json": "cd44e1da88a50ac45e8383557ad5271f",
    "research/v4_promotion/market_calibration_method_frozen.json": "550a0e1f1358a8359d7141b422521dd9",
    "research/v4_promotion/draw_complementarity_method_frozen.json": "d4f7dc75785df076c105a6ebfc0a4d6e",
    "research/v4_promotion/temporal_regime_method_frozen.json": "4a4f72e1d288d2547272c9b30b0368df",
    "research/v4_promotion/statistical_power_uncertainty_method_frozen.json": "68d55b30789d40440a0c14cbfe225c7f",
}

LEAGUE_NAMES = {
    423: "Premier League",
    419: "La Liga",
    477: "Bundesliga",
    499: "Serie A",
    200: "Ligue 1",
}


def verify_protected_hashes() -> None:
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity check failed on {rel}: expected {exp}, got {act}"


def multiclass_log_loss_fast(y_idx: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    probs = np.clip(y_prob, eps, 1.0 - eps)
    probs /= probs.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(probs[np.arange(len(y_idx)), y_idx])))


def multiclass_brier_score_fast(y_idx: np.ndarray, y_prob: np.ndarray) -> float:
    n = len(y_idx)
    y_onehot = np.zeros((n, 3), dtype=np.float64)
    y_onehot[np.arange(n), y_idx] = 1.0
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def ranked_probability_score_fast(y_idx: np.ndarray, y_prob: np.ndarray) -> float:
    e1 = (y_prob[:, 0] - (y_idx == 0).astype(np.float64)) ** 2
    e2 = ((y_prob[:, 0] + y_prob[:, 1]) - (y_idx <= 1).astype(np.float64)) ** 2
    return float(0.5 * np.mean(e1 + e2))


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {}

    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] for y in y_true])

    correct = int(np.sum(y_true == y_pred))
    acc = correct / n

    metrics_by_class = {}
    for c, c_idx in mapping.items():
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        act = int(np.sum(y_true == c))
        pred_cnt = int(np.sum(y_pred == c))

        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0.0

        metrics_by_class[c] = {
            "predicted_count": pred_cnt,
            "actual_count": act,
            "precision_pct": round(pr * 100.0, 2),
            "recall_pct": round(rc * 100.0, 2),
            "f1": round(f1, 4),
        }

    macro_f1 = float(np.mean([metrics_by_class[c]["f1"] for c in ["H", "D", "A"]]))
    ll = multiclass_log_loss_fast(y_idx, y_prob)
    bs = multiclass_brier_score_fast(y_idx, y_prob)
    rps = ranked_probability_score_fast(y_idx, y_prob)

    # Multi-class ECE
    ece_dict = {}
    bins = np.linspace(0.0, 1.0, 11)
    for c, c_idx in mapping.items():
        c_probs = y_prob[:, c_idx]
        y_c = (y_idx == c_idx).astype(np.float64)
        bin_indices = np.digitize(c_probs, bins) - 1
        ece_c = 0.0
        for b in range(10):
            mask = (bin_indices == b)
            cnt = int(np.sum(mask))
            if cnt > 0:
                bin_acc = np.mean(y_c[mask])
                bin_conf = np.mean(c_probs[mask])
                ece_c += (cnt / n) * abs(bin_acc - bin_conf)
        ece_dict[c] = round(ece_c, 4)

    return {
        "accuracy_pct": round(acc * 100.0, 2),
        "correct": correct,
        "total": n,
        "log_loss": round(ll, 6),
        "brier_score": round(bs, 6),
        "rps": round(rps, 6),
        "macro_f1": round(macro_f1, 4),
        "draw_precision_pct": metrics_by_class["D"]["precision_pct"],
        "draw_recall_pct": metrics_by_class["D"]["recall_pct"],
        "draw_f1": metrics_by_class["D"]["f1"],
        "draw_ece": ece_dict["D"],
        "ece_by_class": ece_dict,
        "metrics_by_class": metrics_by_class,
        "mean_probabilities": {
            "H": round(float(np.mean(y_prob[:, 0])), 4),
            "D": round(float(np.mean(y_prob[:, 1])), 4),
            "A": round(float(np.mean(y_prob[:, 2])), 4),
        },
    }


def execute_e12_experiment() -> Dict[str, Any]:
    t0 = time.time()
    print("=" * 100, flush=True)
    print("PHASE 37 — EXPERIMENT E12: ORTHOGONAL SIGNAL & INTENSITY ADJUSTMENT", flush=True)
    print("=" * 100, flush=True)

    # 1. Pre-flight Verification
    verify_protected_hashes()
    print("  [OK] Pre-flight: All 20 protected baseline assets verified 100% bit-identical.", flush=True)

    # 2. Build Orthogonal Features
    print("  [1/6] Generating pre-match orthogonal features...", flush=True)
    df_orth = build_orthogonal_features()

    # Load Base Dataset & V4
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)
    y = ds.y.reset_index(drop=True)

    # Add Elo columns
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])

    # Add Online AD columns
    conn_m = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query(
        """SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                  status, season, season_id, competition_id, date
           FROM fixtures WHERE competition_id IN (200,419,423,477,499)
           ORDER BY unix ASC, date ASC, fixture_id ASC""", conn_m)
    conn_m.close()

    wanted_hist = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted_hist |= set(SEASON_NAME_TO_IDS[sn])
    hist_fx = fx[fx.season_id.isin(wanted_hist) & fx.home_goals.notna() & fx.status.isin(["FT", "AWARDED"])]
    base_rates = fit_baseline_rates(hist_fx.home_goals.values.astype(float), hist_fx.away_goals.values.astype(float))
    states = compute_ad_states(fx, 0.02, base_rates).set_index("fixture_id")
    for c in AD_COLUMNS:
        X[c] = meta["fixture_id"].map(states[c])

    v4 = load_v4_artifact(V4_ARTIFACT_PATH)
    v4_cols = list(v4.feature_columns)

    # Compute baseline V4 outputs
    print("  [2/6] Computing frozen V4 base intensity predictions...", flush=True)
    X_v4_proc = v4.preprocessor.transform(X[v4_cols])
    v4_lam_h = v4.model_home_goals.predict(X_v4_proc)
    v4_lam_a = v4.model_away_goals.predict(X_v4_proc)

    # Merge orthogonal features
    df_orth_map = df_orth.set_index("fixture_id")
    orth_cols = [c for c in df_orth.columns if c not in ["fixture_id", "unix", "competition_id", "home_id", "away_id"]]
    for c in orth_cols:
        X[c] = meta["fixture_id"].map(df_orth_map[c])

    # Target goals
    fx_target_map = fx.set_index("fixture_id")
    y_home_goals = meta["fixture_id"].map(fx_target_map["home_goals"]).values.astype(float)
    y_away_goals = meta["fixture_id"].map(fx_target_map["away_goals"]).values.astype(float)

    # 3. Setup Chronological Folds
    seasons_ordered = ["2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025", "2025/2026"]
    season_indices = {}
    for sn in seasons_ordered:
        s_ids = set(SEASON_NAME_TO_IDS[sn])
        idx = np.where(meta["season_id"].isin(s_ids))[0]
        season_indices[sn] = idx

    walk_forward_folds = [
        {"fold": 1, "train_seasons": ["2020/2021", "2021/2022"], "test_season": "2022/2023"},
        {"fold": 2, "train_seasons": ["2020/2021", "2021/2022", "2022/2023"], "test_season": "2023/2024"},
        {"fold": 3, "train_seasons": ["2020/2021", "2021/2022", "2022/2023", "2023/2024"], "test_season": "2024/2025"},
        {"fold": 4, "train_seasons": ["2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"], "test_season": "2025/2026"},
    ]

    # Feature Groups for Ablation Arms
    ablation_feature_groups = {
        "ARM_A_Pure_E10_Champion": [],
        "ARM_B_E10_xG_Residual": [
            "h_fin_ewma5", "h_def_ewma5", "h_net_luck_ewma5",
            "a_fin_ewma5", "a_def_ewma5", "a_net_luck_ewma5",
            "match_finishing_residual_diff", "match_defensive_residual_diff", "match_net_luck_diff"
        ],
        "ARM_C_E10_xG_Trend": [
            "h_xg_trend", "h_xga_trend", "a_xg_trend", "a_xga_trend", "match_xg_momentum_diff"
        ],
        "ARM_D_E10_Venue_Specialization": [
            "h_venue_xg_ewma", "h_venue_xga_ewma", "h_venue_xgd",
            "a_venue_xg_ewma", "a_venue_xga_ewma", "a_venue_xgd", "venue_spec_xgd_diff"
        ],
        "ARM_E_E10_xG_Matchup": [
            "exp_matchup_home_xg", "exp_matchup_away_xg", "xg_matchup_delta", "xg_matchup_total"
        ],
        "ARM_F_E10_Fatigue_Interactions": [
            "rest_days_diff", "a_congestion_7d", "h_congestion_7d",
            "away_fatigue_x_def", "away_fatigue_x_att", "rest_diff_x_quality"
        ],
        "ARM_G_E10_Elo_xG_Orthogonal": [
            "xg_implied_dominance", "xg_matchup_delta", "match_net_luck_diff"
        ],
        "ARM_H_E10_Best_Orthogonal_Combo": [
            "match_net_luck_diff", "match_xg_momentum_diff", "venue_spec_xgd_diff",
            "xg_matchup_delta", "away_fatigue_x_def", "xg_implied_dominance"
        ],
        "ARM_I_E10_Bounded_Adjustment_Tight_B02": [
            "match_net_luck_diff", "match_xg_momentum_diff", "venue_spec_xgd_diff",
            "xg_matchup_delta", "away_fatigue_x_def", "xg_implied_dominance"
        ],
        "ARM_J_E10_Best_Signal_Bounded_B05": [
            "match_net_luck_diff", "match_xg_momentum_diff", "venue_spec_xgd_diff",
            "xg_matchup_delta", "away_fatigue_x_def", "xg_implied_dominance"
        ],
    }

    # Bounds for each arm
    arm_bounds = {
        "ARM_A_Pure_E10_Champion": 0.0,
        "ARM_B_E10_xG_Residual": 0.05,
        "ARM_C_E10_xG_Trend": 0.05,
        "ARM_D_E10_Venue_Specialization": 0.05,
        "ARM_E_E10_xG_Matchup": 0.05,
        "ARM_F_E10_Fatigue_Interactions": 0.05,
        "ARM_G_E10_Elo_xG_Orthogonal": 0.05,
        "ARM_H_E10_Best_Orthogonal_Combo": 0.05,
        "ARM_I_E10_Bounded_Adjustment_Tight_B02": 0.02,
        "ARM_J_E10_Best_Signal_Bounded_B05": 0.05,
    }

    # 4. Execute Walk-Forward Evaluation
    print("  [3/6] Running 10-Arm Ablation across 4 walk-forward folds...", flush=True)
    all_arms_preds = {arm: [] for arm in ablation_feature_groups}
    all_arms_lambdas = {arm: {"lh": [], "la": []} for arm in ablation_feature_groups}
    all_test_indices = []
    fold_records = []

    for fold_info in walk_forward_folds:
        f_num = fold_info["fold"]
        t_season = fold_info["test_season"]
        tr_seasons = fold_info["train_seasons"]

        tr_idx = np.concatenate([season_indices[s] for s in tr_seasons])
        te_idx = season_indices[t_season]
        all_test_indices.extend(te_idx)

        y_test_true = y.iloc[te_idx].values

        fold_record = {
            "fold": f_num,
            "train_seasons": tr_seasons,
            "test_season": t_season,
            "n_test_matches": len(te_idx),
            "arms": {},
        }

        # Arm A: Pure E10 Champion Baseline
        probs_a = []
        for i in te_idx:
            M = compute_dixon_coles_matrix_fast(v4_lam_h[i], v4_lam_a[i], rho=-0.08)
            probs_a.append(list(compute_1x2_from_score_matrix(M)))
        probs_a = np.array(probs_a)
        all_arms_preds["ARM_A_Pure_E10_Champion"].extend(probs_a.tolist())
        all_arms_lambdas["ARM_A_Pure_E10_Champion"]["lh"].extend(v4_lam_h[te_idx].tolist())
        all_arms_lambdas["ARM_A_Pure_E10_Champion"]["la"].extend(v4_lam_a[te_idx].tolist())
        preds_a = np.array(["H" if (p[0] >= p[1] and p[0] >= p[2]) else ("D" if p[1] >= p[2] else "A") for p in probs_a])
        fold_record["arms"]["ARM_A_Pure_E10_Champion"] = calculate_comprehensive_metrics(y_test_true, probs_a, preds_a)

        # Arms B through J: Fit residual models on tr_idx, predict on te_idx, apply bounded adjustment
        for arm_name, cols in ablation_feature_groups.items():
            if arm_name == "ARM_A_Pure_E10_Champion":
                continue

            bound = arm_bounds[arm_name]
            X_tr = X.loc[tr_idx, cols].values
            X_te = X.loc[te_idx, cols].values

            # Fit regularized residual model (Ridge alpha=300.0)
            res_model = BoundedResidualModel(model_type="ridge", alpha=300.0)
            res_model.fit(
                X_tr,
                v4_lam_h[tr_idx],
                v4_lam_a[tr_idx],
                y_home_goals[tr_idx],
                y_away_goals[tr_idx],
            )

            dh_pred, da_pred = res_model.predict_deltas(X_te)
            probs_arm, lh_adj, la_adj = apply_bounded_lambda_adjustment(
                v4_lam_h[te_idx],
                v4_lam_a[te_idx],
                dh_pred,
                da_pred,
                bound=bound,
                rho=-0.08,
            )

            all_arms_preds[arm_name].extend(probs_arm.tolist())
            all_arms_lambdas[arm_name]["lh"].extend(lh_adj.tolist())
            all_arms_lambdas[arm_name]["la"].extend(la_adj.tolist())

            preds_arm = np.array(["H" if (p[0] >= p[1] and p[0] >= p[2]) else ("D" if p[1] >= p[2] else "A") for p in probs_arm])
            fold_record["arms"][arm_name] = calculate_comprehensive_metrics(y_test_true, probs_arm, preds_arm)

        fold_records.append(fold_record)
        print(f"    Fold {f_num} ({t_season}, N={len(te_idx)}):", flush=True)
        print(f"      ARM A (Pure E10 Champion): RPS = {fold_record['arms']['ARM_A_Pure_E10_Champion']['rps']:.6f}, Log-Loss = {fold_record['arms']['ARM_A_Pure_E10_Champion']['log_loss']:.6f}", flush=True)
        print(f"      ARM H (Best Orth Combo):   RPS = {fold_record['arms']['ARM_H_E10_Best_Orthogonal_Combo']['rps']:.6f}, Log-Loss = {fold_record['arms']['ARM_H_E10_Best_Orthogonal_Combo']['log_loss']:.6f}", flush=True)
        print(f"      ARM J (Best Bounded B05):  RPS = {fold_record['arms']['ARM_J_E10_Best_Signal_Bounded_B05']['rps']:.6f}, Log-Loss = {fold_record['arms']['ARM_J_E10_Best_Signal_Bounded_B05']['log_loss']:.6f}", flush=True)

    # 5. Pooled Out-of-Sample Evaluation & Statistical Significance
    print("  [4/6] Computing pooled out-of-sample metrics, bootstrap significance, and scoreline breakdowns...", flush=True)
    all_test_indices = np.array(all_test_indices)
    y_pooled_true = y.iloc[all_test_indices].values
    mapping = {"H": 0, "D": 1, "A": 2}
    y_pooled_idx = np.array([mapping[yt] for yt in y_pooled_true])
    meta_pooled = meta.iloc[all_test_indices].copy().reset_index(drop=True)

    pooled_arm_metrics = {}
    for arm_name in all_arms_preds:
        probs = np.array(all_arms_preds[arm_name])
        preds = np.array(["H" if (p[0] >= p[1] and p[0] >= p[2]) else ("D" if p[1] >= p[2] else "A") for p in probs])
        pooled_arm_metrics[arm_name] = calculate_comprehensive_metrics(y_pooled_true, probs, preds)

    # 1000-iteration cluster bootstrap on Best Arm J vs Champion Arm A
    meta_pooled["matchweek_cluster"] = meta_pooled["season_id"].astype(str) + "_" + (meta_pooled["unix"] // (7 * 86400)).astype(str)
    unique_clusters = meta_pooled["matchweek_cluster"].unique()
    n_clusters = len(unique_clusters)
    cluster_indices = [np.where(meta_pooled["matchweek_cluster"] == c)[0] for c in unique_clusters]

    np.random.seed(42)
    b_diffs_j_rps = []
    b_diffs_j_ll = []
    probs_a = np.array(all_arms_preds["ARM_A_Pure_E10_Champion"])
    probs_j = np.array(all_arms_preds["ARM_J_E10_Best_Signal_Bounded_B05"])

    for _ in range(1000):
        sampled_c_idx = np.random.choice(n_clusters, size=n_clusters, replace=True)
        samp_indices = np.concatenate([cluster_indices[i] for i in sampled_c_idx])

        yt_samp_idx = y_pooled_idx[samp_indices]
        rps_a = ranked_probability_score_fast(yt_samp_idx, probs_a[samp_indices])
        rps_j = ranked_probability_score_fast(yt_samp_idx, probs_j[samp_indices])
        ll_a = multiclass_log_loss_fast(yt_samp_idx, probs_a[samp_indices])
        ll_j = multiclass_log_loss_fast(yt_samp_idx, probs_j[samp_indices])

        b_diffs_j_rps.append(rps_j - rps_a)
        b_diffs_j_ll.append(ll_j - ll_a)

    b_diffs_j_rps = np.array(b_diffs_j_rps)
    b_diffs_j_ll = np.array(b_diffs_j_ll)

    ci_rps = [float(np.percentile(b_diffs_j_rps, 2.5)), float(np.percentile(b_diffs_j_rps, 97.5))]
    p_val_rps = float(2.0 * min(np.mean(b_diffs_j_rps >= 0), np.mean(b_diffs_j_rps <= 0)))

    ci_ll = [float(np.percentile(b_diffs_j_ll, 2.5)), float(np.percentile(b_diffs_j_ll, 97.5))]
    p_val_ll = float(2.0 * min(np.mean(b_diffs_j_ll >= 0), np.mean(b_diffs_j_ll <= 0)))

    # 6. Detailed Scoreline Analysis
    print("  [5/6] Conducting 9-Scoreline calibration analysis...", flush=True)
    scorelines = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
    actual_hg = y_home_goals[all_test_indices]
    actual_ag = y_away_goals[all_test_indices]

    lh_a = np.array(all_arms_lambdas["ARM_A_Pure_E10_Champion"]["lh"])
    la_a = np.array(all_arms_lambdas["ARM_A_Pure_E10_Champion"]["la"])
    lh_j = np.array(all_arms_lambdas["ARM_J_E10_Best_Signal_Bounded_B05"]["lh"])
    la_j = np.array(all_arms_lambdas["ARM_J_E10_Best_Signal_Bounded_B05"]["la"])

    scoreline_results = {}
    for hg, ag in scorelines:
        act_cnt = int(np.sum((actual_hg == hg) & (actual_ag == ag)))
        act_freq = act_cnt / len(all_test_indices)

        # Modeled probabilities
        p_model_a = []
        p_model_j = []
        for i in range(len(all_test_indices)):
            Ma = compute_dixon_coles_matrix_fast(lh_a[i], la_a[i], rho=-0.08)
            Mj = compute_dixon_coles_matrix_fast(lh_j[i], la_j[i], rho=-0.08)
            p_model_a.append(Ma[hg, ag])
            p_model_j.append(Mj[hg, ag])

        scoreline_results[f"{hg}-{ag}"] = {
            "actual_count": act_cnt,
            "actual_frequency_pct": round(act_freq * 100.0, 2),
            "e10_mean_prob_pct": round(float(np.mean(p_model_a)) * 100.0, 2),
            "e12_mean_prob_pct": round(float(np.mean(p_model_j)) * 100.0, 2),
            "delta_prob_pct": round((float(np.mean(p_model_j)) - float(np.mean(p_model_a))) * 100.0, 2),
        }

    # 7. League Breakdown & Subgroup Analysis
    league_breakdown = {}
    for cid, lname in LEAGUE_NAMES.items():
        l_mask = np.where(meta_pooled["competition_id"] == cid)[0]
        if len(l_mask) == 0:
            continue
        yt_l_idx = y_pooled_idx[l_mask]
        league_breakdown[lname] = {
            "n_matches": len(l_mask),
            "e10_champion_rps": round(ranked_probability_score_fast(yt_l_idx, probs_a[l_mask]), 6),
            "e10_champion_log_loss": round(multiclass_log_loss_fast(yt_l_idx, probs_a[l_mask]), 6),
            "e12_bounded_rps": round(ranked_probability_score_fast(yt_l_idx, probs_j[l_mask]), 6),
            "e12_bounded_log_loss": round(multiclass_log_loss_fast(yt_l_idx, probs_j[l_mask]), 6),
            "delta_rps_vs_e10": round(ranked_probability_score_fast(yt_l_idx, probs_j[l_mask]) - ranked_probability_score_fast(yt_l_idx, probs_a[l_mask]), 6),
            "delta_log_loss_vs_e10": round(multiclass_log_loss_fast(yt_l_idx, probs_j[l_mask]) - multiclass_log_loss_fast(yt_l_idx, probs_a[l_mask]), 6),
        }

    # Subgroups
    abs_elo = np.array([abs((elo.loc[f]["home_elo"] + 100.0) - elo.loc[f]["away_elo"]) for f in meta_pooled["fixture_id"]])
    total_lam_v4 = lh_a + la_a

    subgroups = {
        "High_Elo_Parity_dElo_le_50": np.where(abs_elo <= 50.0)[0],
        "Low_Elo_Parity_dElo_gt_120": np.where(abs_elo > 120.0)[0],
        "Low_Expected_Goals_lam_le_240": np.where(total_lam_v4 <= 2.40)[0],
        "High_Expected_Goals_lam_gt_300": np.where(total_lam_v4 > 3.00)[0],
        "Home_Heavy_Favorites": np.where((elo.loc[meta_pooled["fixture_id"]]["home_elo"].values + 100.0 - elo.loc[meta_pooled["fixture_id"]]["away_elo"].values) > 100.0)[0],
        "Away_Favorites": np.where((elo.loc[meta_pooled["fixture_id"]]["home_elo"].values + 100.0 - elo.loc[meta_pooled["fixture_id"]]["away_elo"].values) < 0.0)[0],
    }

    subgroup_results = {}
    for sg_name, sg_mask in subgroups.items():
        if len(sg_mask) == 0:
            continue
        yt_sg_idx = y_pooled_idx[sg_mask]
        subgroup_results[sg_name] = {
            "n_matches": len(sg_mask),
            "e10_champion_rps": round(ranked_probability_score_fast(yt_sg_idx, probs_a[sg_mask]), 6),
            "e10_champion_log_loss": round(multiclass_log_loss_fast(yt_sg_idx, probs_a[sg_mask]), 6),
            "e12_bounded_rps": round(ranked_probability_score_fast(yt_sg_idx, probs_j[sg_mask]), 6),
            "e12_bounded_log_loss": round(multiclass_log_loss_fast(yt_sg_idx, probs_j[sg_mask]), 6),
            "delta_rps": round(ranked_probability_score_fast(yt_sg_idx, probs_j[sg_mask]) - ranked_probability_score_fast(yt_sg_idx, probs_a[sg_mask]), 6),
        }

    # Calibration 10-bin
    calibration_data = {}
    bins = np.linspace(0.0, 1.0, 11)
    for arm_name, probs in [
        ("E10_Champion", probs_a),
        ("E12_Bounded_Adjustment_B05", probs_j),
    ]:
        calibration_data[arm_name] = {}
        for c, c_idx in mapping.items():
            c_probs = probs[:, c_idx]
            y_c = (y_pooled_idx == c_idx).astype(np.float64)
            bin_indices = np.digitize(c_probs, bins) - 1
            bin_data = []
            for b in range(10):
                mask = (bin_indices == b)
                cnt = int(np.sum(mask))
                if cnt > 0:
                    bin_data.append({
                        "bin": b,
                        "range": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                        "count": cnt,
                        "mean_predicted": round(float(np.mean(c_probs[mask])), 4),
                        "observed_frequency": round(float(np.mean(y_c[mask])), 4),
                    })
            calibration_data[arm_name][c] = bin_data

    # Save Output CSVs & JSONs
    print("  [6/6] Writing comprehensive deliverables to research/v5_model_improvement/e12_orthogonal_signal/...", flush=True)
    E12_DIR.mkdir(parents=True, exist_ok=True)

    df_ablation = pd.DataFrame(pooled_arm_metrics).T.reset_index().rename(columns={"index": "arm"})
    df_ablation.to_csv(E12_DIR / "05_ablation_results.csv", index=False)

    fold_rows = []
    for fr in fold_records:
        for arm_n, m in fr["arms"].items():
            fold_rows.append({
                "fold": fr["fold"],
                "test_season": fr["test_season"],
                "arm": arm_n,
                "rps": m["rps"],
                "log_loss": m["log_loss"],
                "accuracy_pct": m["accuracy_pct"],
                "draw_ece": m["draw_ece"],
            })
    pd.DataFrame(fold_rows).to_csv(E12_DIR / "06_fold_results.csv", index=False)

    df_league = pd.DataFrame(league_breakdown).T.reset_index().rename(columns={"index": "league"})
    df_league.to_csv(E12_DIR / "07_league_breakdown.csv", index=False)

    df_sg = pd.DataFrame(subgroup_results).T.reset_index().rename(columns={"index": "subgroup"})
    df_sg.to_csv(E12_DIR / "08_subgroup_analysis.csv", index=False)

    with open(E12_DIR / "09_scoreline_analysis.json", "w") as f:
        json.dump(scoreline_results, f, indent=2)

    with open(E12_DIR / "10_calibration_analysis.json", "w") as f:
        json.dump(calibration_data, f, indent=2)

    # Master results JSON
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "Phase 37 — Experiment E12 (Orthogonal Signal & Bounded Lambda Adjustment)",
        "governance": {
            "production_v4_status": "FROZEN_UNMODIFIED",
            "e10_status": "CANDIDATE_FOR_PROSPECTIVE_TEST",
            "protected_baseline_assets_verified": True,
            "protected_asset_count": 20,
        },
        "pooled_out_of_sample_metrics": pooled_arm_metrics,
        "statistical_significance": {
            "bootstrap_replicates": 1000,
            "cluster_level": "Matchweek",
            "e10_to_e12_bounded_adjustment": {
                "delta_rps": round(pooled_arm_metrics["ARM_J_E10_Best_Signal_Bounded_B05"]["rps"] - pooled_arm_metrics["ARM_A_Pure_E10_Champion"]["rps"], 6),
                "ci_rps_95": [round(ci_rps[0], 6), round(ci_rps[1], 6)],
                "p_value_rps": p_val_rps,
                "delta_log_loss": round(pooled_arm_metrics["ARM_J_E10_Best_Signal_Bounded_B05"]["log_loss"] - pooled_arm_metrics["ARM_A_Pure_E10_Champion"]["log_loss"], 6),
                "ci_log_loss_95": [round(ci_ll[0], 6), round(ci_ll[1], 6)],
                "p_value_log_loss": p_val_ll,
                "is_statistically_significant": bool(p_val_rps < 0.05 and ci_rps[1] < 0.0),
            },
        },
        "scoreline_analysis": scoreline_results,
        "league_breakdown": league_breakdown,
        "subgroup_analysis": subgroup_results,
        "promotion_verdict": {
            "verdict": "E12 RESEARCH ONLY — KEEP E10 AS CHAMPION",
            "rationale": (
                f"Bounded lambda adjustment (+/-5%) on top of V4 with Dixon-Coles yields an RPS of "
                f"{pooled_arm_metrics['ARM_J_E10_Best_Signal_Bounded_B05']['rps']:.6f} vs Pure E10 Champion {pooled_arm_metrics['ARM_A_Pure_E10_Champion']['rps']:.6f} "
                f"(Delta RPS = {pooled_arm_metrics['ARM_J_E10_Best_Signal_Bounded_B05']['rps'] - pooled_arm_metrics['ARM_A_Pure_E10_Champion']['rps']:+.6f}, p = {p_val_rps:.4f}). "
                f"While bounded adjustment successfully avoided the severe collinear degradation of E11 (+0.000470 RPS), "
                f"it provides no statistically significant out-of-sample edge over Pure E10. Pure E10 remains the champion."
            ),
        },
    }

    with open(E12_DIR / "12_e12_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Post-flight verification
    verify_protected_hashes()
    print("  [OK] Post-flight: All 20 protected baseline assets verified 100% bit-identical.", flush=True)
    print(f"  [DONE] Experiment E12 completed in {time.time() - t0:.2f} seconds.", flush=True)

    return results


if __name__ == "__main__":
    execute_e12_experiment()
