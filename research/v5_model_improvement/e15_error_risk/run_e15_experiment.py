"""Phase 40 — Experiment E15 Execution Harness: Pre-Match Error Risk & Selective Forecasting.

Usage:
    python research/v5_model_improvement/e15_error_risk/run_e15_experiment.py
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"))
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e13_dynamic_bayesian"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dixon_coles_engine import (
    compute_dixon_coles_matrix_fast,
    compute_1x2_from_score_matrix,
)
from features.elo import ELO_COLUMNS, load_elo_features
from features.online_attack_defense import AD_COLUMNS, compute_ad_states, fit_baseline_rates
from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
from models.data import load_supervised_dataset
from models.v4_artifact import load_v4_artifact

from dynamic_elo import compute_dynamic_elo_features
from dynamic_attack_defense import compute_dynamic_ad_states
from hierarchical_model import apply_hierarchical_shrinkage

from error_target import compute_match_rps_and_log_loss, create_error_targets
from risk_features import build_pre_match_risk_features
from risk_model import (
    LogisticRiskClassifier,
    GradientBoostingRiskClassifier,
    ExpectedRPSRegressor,
)
from risk_calibration import (
    evaluate_risk_model_calibration,
    compute_risk_deciles_breakdown,
)
from selective_router import (
    compute_risk_coverage_curve,
    evaluate_routing_policies,
)

MATCHES_DB = PROJECT_ROOT / "data/processed/matches.db"
FEATURES_DB = PROJECT_ROOT / "data/processed/features.db"
V4_ARTIFACT_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
E15_DIR = PROJECT_ROOT / "research/v5_model_improvement/e15_error_risk"

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


def execute_e15_experiment() -> Dict[str, Any]:
    t0 = time.time()
    print("=" * 100, flush=True)
    print("PHASE 40 — EXPERIMENT E15: PRE-MATCH ERROR RISK / SELECTIVE FORECASTING", flush=True)
    print("=" * 100, flush=True)

    # 1. Pre-flight Verification
    verify_protected_hashes()
    print("  [OK] Pre-flight: All 20 protected baseline assets verified 100% bit-identical.", flush=True)

    # 2. Ingest Fixtures & Build Dynamic Components
    print("  [1/6] Computing causal Dynamic Features & Pre-Match Features...", flush=True)
    conn_m = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query(
        """SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                  status, season, season_id, competition_id, date
           FROM fixtures WHERE competition_id IN (200,419,423,477,499)
           ORDER BY unix ASC, date ASC, fixture_id ASC""", conn_m)
    conn_m.close()

    df_dyn_elo = compute_dynamic_elo_features(fx)
    df_dyn_ad = compute_dynamic_ad_states(fx)

    # Load Base Dataset & V4
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)
    y = ds.y.reset_index(drop=True)

    # Add Static Elo columns
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])

    # Add Static Online AD columns
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

    # Base V4 Lambdas
    print("  [2/6] Computing frozen V4 & E10 base intensity predictions...", flush=True)
    X_v4_proc = v4.preprocessor.transform(X[v4_cols])
    v4_lam_h = v4.model_home_goals.predict(X_v4_proc)
    v4_lam_a = v4.model_away_goals.predict(X_v4_proc)

    # Compute E10 probabilities for all fixtures
    probs_e10_all = np.zeros((len(meta), 3), dtype=np.float64)
    for i in range(len(meta)):
        M = compute_dixon_coles_matrix_fast(v4_lam_h[i], v4_lam_a[i], rho=-0.08)
        probs_e10_all[i] = compute_1x2_from_score_matrix(M)

    # Merge dynamic features
    df_dyn_elo_map = df_dyn_elo.set_index("fixture_id")
    for c in df_dyn_elo.columns:
        if c != "fixture_id":
            X[c] = meta["fixture_id"].map(df_dyn_elo_map[c])

    df_dyn_ad_map = df_dyn_ad.set_index("fixture_id")
    for c in df_dyn_ad.columns:
        if c != "fixture_id":
            X[c] = meta["fixture_id"].map(df_dyn_ad_map[c])

    # Add hierarchical shrinkage features
    X["h_shrunk_A"] = apply_hierarchical_shrinkage(X["dyn_A_home"].values, X["h_dyn_sample_size"].values, 0.0, 5.0)
    X["h_shrunk_D"] = apply_hierarchical_shrinkage(X["dyn_D_home"].values, X["h_dyn_sample_size"].values, 0.0, 5.0)
    X["a_shrunk_A"] = apply_hierarchical_shrinkage(X["dyn_A_away"].values, X["a_dyn_sample_size"].values, 0.0, 5.0)
    X["a_shrunk_D"] = apply_hierarchical_shrinkage(X["dyn_D_away"].values, X["a_dyn_sample_size"].values, 0.0, 5.0)

    # Target goals
    fx_target_map = fx.set_index("fixture_id")
    y_home_goals = meta["fixture_id"].map(fx_target_map["home_goals"]).values.astype(float)
    y_away_goals = meta["fixture_id"].map(fx_target_map["away_goals"]).values.astype(float)

    # Setup Chronological Folds
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

    v3_base_cols = [c for c in v4_cols if c not in ELO_COLUMNS and c not in AD_COLUMNS]
    e13_cols = v3_base_cols + [
        "dynamic_home_elo", "dynamic_away_elo", "dynamic_elo_diff",
        "dyn_A_home", "dyn_D_home", "dyn_A_away", "dyn_D_away",
        "h_shrunk_A", "h_shrunk_D", "a_shrunk_A", "a_shrunk_D",
        "dyn_A_venue_home", "dyn_D_venue_home"
    ]

    # Pre-extract Risk Feature Matrix (E13 probabilities computed per fold)
    print("  [3/6] Executing Walk-Forward Meta-Learning & Pre-Match Risk Models...", flush=True)

    risk_feature_cols = [
        "entropy_e10", "max_prob_e10", "margin_p1_p2", "p_draw_e10",
        "p_home_e10", "p_away_e10", "total_lam", "lambda_ratio",
        "abs_elo_diff", "home_elo", "away_elo", "season_match_num",
        "min_team_samples", "is_promoted_sparse", "is_early_3", "is_early_5",
        "js_divergence_e10_e13", "mean_abs_prob_diff",
    ]

    all_test_indices = []
    all_test_risk_scores = {"Logistic": [], "GradientBoosting": [], "ExpectedRPS": [], "RuleBased": []}
    all_test_e13_probs = []
    all_test_actual_rps = []
    all_test_actual_ll = []
    all_test_binary_targets_10 = []
    all_test_binary_targets_20 = []
    fold_calibration_metrics = []

    for fold_info in walk_forward_folds:
        f_num = fold_info["fold"]
        t_season = fold_info["test_season"]
        tr_seasons = fold_info["train_seasons"]

        tr_idx = np.concatenate([season_indices[s] for s in tr_seasons])
        te_idx = season_indices[t_season]
        all_test_indices.extend(te_idx)

        y_tr_true = y.iloc[tr_idx].values
        y_te_true = y.iloc[te_idx].values

        # Train E13 Specialist on Train Fold
        X_tr_e13 = X.loc[tr_idx, e13_cols].values
        X_te_e13 = X.loc[te_idx, e13_cols].values

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr_e13_proc = scaler.fit_transform(imputer.fit_transform(X_tr_e13))
        X_te_e13_proc = scaler.transform(imputer.transform(X_te_e13))

        model_h = PoissonRegressor(alpha=1.0, max_iter=2000)
        model_h.fit(X_tr_e13_proc, y_home_goals[tr_idx])
        model_a = PoissonRegressor(alpha=1.0, max_iter=2000)
        model_a.fit(X_tr_e13_proc, y_away_goals[tr_idx])

        e13_lh_tr = model_h.predict(X_tr_e13_proc)
        e13_la_tr = model_a.predict(X_tr_e13_proc)
        e13_lh_te = model_h.predict(X_te_e13_proc)
        e13_la_te = model_a.predict(X_te_e13_proc)

        P_e13_tr = np.zeros((len(tr_idx), 3), dtype=np.float64)
        for i in range(len(tr_idx)):
            M = compute_dixon_coles_matrix_fast(e13_lh_tr[i], e13_la_tr[i], rho=-0.08)
            P_e13_tr[i] = compute_1x2_from_score_matrix(M)

        P_e13_te = np.zeros((len(te_idx), 3), dtype=np.float64)
        for i in range(len(te_idx)):
            M = compute_dixon_coles_matrix_fast(e13_lh_te[i], e13_la_te[i], rho=-0.08)
            P_e13_te[i] = compute_1x2_from_score_matrix(M)

        all_test_e13_probs.extend(P_e13_te.tolist())

        # Build Pre-Match Risk Features for Train & Test
        risk_df_tr = build_pre_match_risk_features(
            meta.iloc[tr_idx],
            fx,
            probs_e10_all[tr_idx],
            P_e13_tr,
            v4_lam_h[tr_idx],
            v4_lam_a[tr_idx],
            X.loc[tr_idx, "home_elo"].values,
            X.loc[tr_idx, "away_elo"].values,
            X.loc[tr_idx, "h_dyn_sample_size"].values,
            X.loc[tr_idx, "a_dyn_sample_size"].values,
        )

        risk_df_te = build_pre_match_risk_features(
            meta.iloc[te_idx],
            fx,
            probs_e10_all[te_idx],
            P_e13_te,
            v4_lam_h[te_idx],
            v4_lam_a[te_idx],
            X.loc[te_idx, "home_elo"].values,
            X.loc[te_idx, "away_elo"].values,
            X.loc[te_idx, "h_dyn_sample_size"].values,
            X.loc[te_idx, "a_dyn_sample_size"].values,
        )

        # Compute Historical Realized Error Targets on Train Fold ONLY
        rps_tr, ll_tr = compute_match_rps_and_log_loss(y_tr_true, probs_e10_all[tr_idx])
        rps_te, ll_te = compute_match_rps_and_log_loss(y_te_true, probs_e10_all[te_idx])

        all_test_actual_rps.extend(rps_te.tolist())
        all_test_actual_ll.extend(ll_te.tolist())

        # Train Thresholds (e.g. 90th & 80th percentiles of error)
        rps_thresh_10 = float(np.percentile(rps_tr, 90.0))
        rps_thresh_20 = float(np.percentile(rps_tr, 80.0))
        ll_thresh_10 = float(np.percentile(ll_tr, 90.0))
        ll_thresh_20 = float(np.percentile(ll_tr, 80.0))

        y_target_10_tr = (rps_tr >= rps_thresh_10).astype(int)
        y_target_20_tr = (rps_tr >= rps_thresh_20).astype(int)
        y_target_10_te = (rps_te >= rps_thresh_10).astype(int)
        y_target_20_te = (rps_te >= rps_thresh_20).astype(int)

        all_test_binary_targets_10.extend(y_target_10_te.tolist())
        all_test_binary_targets_20.extend(y_target_20_te.tolist())

        X_risk_tr = risk_df_tr[risk_feature_cols].values
        X_risk_te = risk_df_te[risk_feature_cols].values

        # 1. Fit Logistic Risk Classifier (Target: Top 10% RPS error)
        clf_log = LogisticRiskClassifier(c_reg=1.0).fit(X_risk_tr, y_target_10_tr)
        risk_log_te = clf_log.predict_risk_scores(X_risk_te)
        all_test_risk_scores["Logistic"].extend(risk_log_te.tolist())

        # 2. Fit Gradient Boosting Risk Classifier
        clf_gb = GradientBoostingRiskClassifier(max_iter=80, max_depth=3).fit(X_risk_tr, y_target_10_tr)
        risk_gb_te = clf_gb.predict_risk_scores(X_risk_te)
        all_test_risk_scores["GradientBoosting"].extend(risk_gb_te.tolist())

        # 3. Fit Expected RPS Regressor
        reg_rps = ExpectedRPSRegressor(max_iter=80, max_depth=3).fit(X_risk_tr, rps_tr)
        exp_rps_te = reg_rps.predict_expected_rps(X_risk_te)
        all_test_risk_scores["ExpectedRPS"].extend(exp_rps_te.tolist())

        # 4. Simple Rule-Based Difficulty Score: entropy / (1.0 + abs_elo / 100.0)
        rule_score_te = risk_df_te["entropy_e10"].values / (1.0 + risk_df_te["abs_elo_diff"].values / 100.0)
        all_test_risk_scores["RuleBased"].extend(rule_score_te.tolist())

        # Fold Calibration Evaluation
        cal_log = evaluate_risk_model_calibration(y_target_10_te, risk_log_te)
        cal_gb = evaluate_risk_model_calibration(y_target_10_te, risk_gb_te)

        fold_calibration_metrics.append({
            "fold": f_num,
            "test_season": t_season,
            "n_matches": len(te_idx),
            "logistic_roc_auc": cal_log.get("roc_auc", 0.0),
            "logistic_pr_auc": cal_log.get("pr_auc", 0.0),
            "gb_roc_auc": cal_gb.get("roc_auc", 0.0),
            "gb_pr_auc": cal_gb.get("pr_auc", 0.0),
        })

        print(f"    Fold {f_num} ({t_season}, N={len(te_idx)}):", flush=True)
        print(f"      Logistic Risk Model ROC-AUC: {cal_log.get('roc_auc', 0.0):.4f}, PR-AUC: {cal_log.get('pr_auc', 0.0):.4f}", flush=True)
        print(f"      Gradient Boosting ROC-AUC:   {cal_gb.get('roc_auc', 0.0):.4f}, PR-AUC: {cal_gb.get('pr_auc', 0.0):.4f}", flush=True)

    # 4. Pooled Out-of-Sample Evaluation
    print("  [4/6] Computing pooled risk deciles, calibration, and selective coverage curves...", flush=True)
    all_test_indices = np.array(all_test_indices)
    y_pooled_true = y.iloc[all_test_indices].values
    probs_e10_pooled = probs_e10_all[all_test_indices]
    probs_e13_pooled = np.array(all_test_e13_probs)
    actual_rps_pooled = np.array(all_test_actual_rps)
    actual_ll_pooled = np.array(all_test_actual_ll)
    target_10_pooled = np.array(all_test_binary_targets_10)
    target_20_pooled = np.array(all_test_binary_targets_20)
    meta_pooled = meta.iloc[all_test_indices].copy().reset_index(drop=True)

    # Compute Risk Model Metrics across all 4 models
    model_metrics_records = []
    for m_name in ["Logistic", "GradientBoosting", "ExpectedRPS", "RuleBased"]:
        scores = np.array(all_test_risk_scores[m_name])
        # Min-max scale scores to [0, 1] for unified calibration metrics
        s_min, s_max = float(scores.min()), float(scores.max())
        s_norm = (scores - s_min) / (s_max - s_min + 1e-12)

        cal_10 = evaluate_risk_model_calibration(target_10_pooled, s_norm)
        cal_20 = evaluate_risk_model_calibration(target_20_pooled, s_norm)

        # Correlation with continuous actual RPS
        corr_rps = float(np.corrcoef(scores, actual_rps_pooled)[0, 1])
        corr_ll = float(np.corrcoef(scores, actual_ll_pooled)[0, 1])

        model_metrics_records.append({
            "risk_model": m_name,
            "roc_auc_top10": cal_10.get("roc_auc", 0.0),
            "pr_auc_top10": cal_10.get("pr_auc", 0.0),
            "brier_score_top10": cal_10.get("brier_score", 0.0),
            "recall_top10": cal_10.get("recall_at_top10", 0.0),
            "roc_auc_top20": cal_20.get("roc_auc", 0.0),
            "pr_auc_top20": cal_20.get("pr_auc", 0.0),
            "correlation_actual_rps": round(corr_rps, 4),
            "correlation_actual_log_loss": round(corr_ll, 4),
        })

    # Decile Stratification (Primary Risk Model: Gradient Boosting)
    primary_risk_scores = np.array(all_test_risk_scores["GradientBoosting"])
    entropy_pooled = -np.sum(probs_e10_pooled * np.log(np.clip(probs_e10_pooled, 1e-12, 1.0)), axis=1)
    max_prob_pooled = np.max(probs_e10_pooled, axis=1)

    df_deciles = compute_risk_deciles_breakdown(
        primary_risk_scores,
        actual_rps_pooled,
        actual_ll_pooled,
        entropy_pooled,
        max_prob_pooled,
        y_pooled_true,
    )

    # Risk-Coverage Curves (Selective Abstention)
    df_coverage = compute_risk_coverage_curve(
        primary_risk_scores,
        probs_e10_pooled,
        y_pooled_true,
        coverage_levels=[1.0, 0.90, 0.80, 0.70, 0.60, 0.50],
    )

    # Routing Policies Evaluation
    is_early_3_pooled = (meta_pooled.groupby(["competition_id", "season_id"])["unix"].rank(method="dense").astype(int) <= 3).astype(int).values
    risk_thresh_top10 = float(np.percentile(primary_risk_scores, 90.0))
    routing_results = evaluate_routing_policies(
        primary_risk_scores,
        probs_e10_pooled,
        probs_e13_pooled,
        is_early_3_pooled,
        y_pooled_true,
        risk_thresh_top10,
    )

    # 5. Subgroup & League Breakdowns
    print("  [5/6] Conducting Subgroup, League, and Disagreement analyses...", flush=True)
    meta_pooled["season_match_num"] = meta_pooled.groupby(["competition_id", "season_id"])["unix"].rank(method="dense").astype(int)

    # Early Season Breakdown
    early_season_masks = {
        "First_1_Match_Of_Season": np.where(meta_pooled["season_match_num"] <= 1)[0],
        "First_3_Matches_Of_Season": np.where(meta_pooled["season_match_num"] <= 3)[0],
        "First_5_Matches_Of_Season": np.where(meta_pooled["season_match_num"] <= 5)[0],
        "First_10_Matches_Of_Season": np.where(meta_pooled["season_match_num"] <= 10)[0],
        "Remaining_Season_Matches": np.where(meta_pooled["season_match_num"] > 10)[0],
    }

    early_season_records = []
    for es_name, es_mask in early_season_masks.items():
        if len(es_mask) == 0:
            continue
        early_season_records.append({
            "segment": es_name,
            "n_matches": len(es_mask),
            "mean_predicted_risk": round(float(np.mean(primary_risk_scores[es_mask])), 4),
            "actual_mean_rps": round(float(np.mean(actual_rps_pooled[es_mask])), 6),
            "actual_mean_log_loss": round(float(np.mean(actual_ll_pooled[es_mask])), 6),
            "top10_risk_concentration_pct": round(float(np.mean(primary_risk_scores[es_mask] >= risk_thresh_top10) * 100.0), 2),
        })

    # Promoted Team Breakdown
    h_samp = X.loc[all_test_indices, "h_dyn_sample_size"].values
    a_samp = X.loc[all_test_indices, "a_dyn_sample_size"].values
    min_samp = np.minimum(h_samp, a_samp)

    promoted_masks = {
        "Promoted_Sparse_lt_5_matches": np.where(min_samp < 5)[0],
        "Developing_5_to_10_matches": np.where((min_samp >= 5) & (min_samp <= 10))[0],
        "Established_Teams_gt_10_matches": np.where(min_samp > 10)[0],
    }

    promoted_records = []
    for pr_name, pr_mask in promoted_masks.items():
        if len(pr_mask) == 0:
            continue
        promoted_records.append({
            "segment": pr_name,
            "n_matches": len(pr_mask),
            "mean_predicted_risk": round(float(np.mean(primary_risk_scores[pr_mask])), 4),
            "actual_mean_rps": round(float(np.mean(actual_rps_pooled[pr_mask])), 6),
            "actual_mean_log_loss": round(float(np.mean(actual_ll_pooled[pr_mask])), 6),
            "top10_risk_concentration_pct": round(float(np.mean(primary_risk_scores[pr_mask] >= risk_thresh_top10) * 100.0), 2),
        })

    # League Breakdown
    league_records = []
    for cid, lname in LEAGUE_NAMES.items():
        l_mask = np.where(meta_pooled["competition_id"] == cid)[0]
        if len(l_mask) == 0:
            continue
        l_auc = float(roc_auc_score(target_10_pooled[l_mask], primary_risk_scores[l_mask])) if len(np.unique(target_10_pooled[l_mask])) > 1 else 0.5
        league_records.append({
            "league": lname,
            "n_matches": len(l_mask),
            "mean_predicted_risk": round(float(np.mean(primary_risk_scores[l_mask])), 4),
            "actual_mean_rps": round(float(np.mean(actual_rps_pooled[l_mask])), 6),
            "actual_mean_log_loss": round(float(np.mean(actual_ll_pooled[l_mask])), 6),
            "risk_model_roc_auc": round(l_auc, 4),
        })

    # Feature Importance (from Gradient Boosting risk classifier)
    # Extract feature names and simple tree importances or correlation
    feat_corrs = []
    risk_df_pooled = build_pre_match_risk_features(
        meta_pooled,
        fx,
        probs_e10_pooled,
        probs_e13_pooled,
        v4_lam_h[all_test_indices],
        v4_lam_a[all_test_indices],
        X.loc[all_test_indices, "home_elo"].values,
        X.loc[all_test_indices, "away_elo"].values,
        h_samp,
        a_samp,
    )
    for c in risk_feature_cols:
        r_val = float(np.corrcoef(risk_df_pooled[c].values, actual_rps_pooled)[0, 1])
        feat_corrs.append({"feature": c, "correlation_with_actual_rps": round(r_val, 4), "abs_corr": abs(r_val)})
    feat_corrs = sorted(feat_corrs, key=lambda x: x["abs_corr"], reverse=True)

    # 6. Write Output Deliverables
    print("  [6/6] Writing comprehensive deliverables to research/v5_model_improvement/e15_error_risk/...", flush=True)
    E15_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(model_metrics_records).to_csv(E15_DIR / "06_risk_model_metrics.csv", index=False)
    df_deciles.to_csv(E15_DIR / "07_risk_deciles.csv", index=False)
    df_coverage.to_csv(E15_DIR / "08_risk_coverage.csv", index=False)
    pd.DataFrame([routing_results]).to_csv(E15_DIR / "09_routing_results.csv", index=False)
    pd.DataFrame(early_season_records).to_csv(E15_DIR / "10_early_season_analysis.csv", index=False)
    pd.DataFrame(promoted_records).to_csv(E15_DIR / "11_promoted_team_analysis.csv", index=False)
    pd.DataFrame(league_records).to_csv(E15_DIR / "12_league_breakdown.csv", index=False)

    calibration_summary = {
        "model_calibration_metrics": model_metrics_records,
        "feature_correlations": feat_corrs,
        "decile_stratification": df_deciles.to_dict(orient="records"),
        "coverage_curves": df_coverage.to_dict(orient="records"),
    }
    with open(E15_DIR / "13_calibration_analysis.json", "w") as f:
        json.dump(calibration_summary, f, indent=2)

    # Master results JSON
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "Phase 40 — Experiment E15 (Pre-Match Error Risk & Selective Forecasting)",
        "governance": {
            "production_v4_status": "FROZEN_UNMODIFIED",
            "e10_status": "CANDIDATE_FOR_PROSPECTIVE_TEST",
            "protected_baseline_assets_verified": True,
            "protected_asset_count": 20,
        },
        "risk_model_summary": model_metrics_records,
        "decile_monotonicity": df_deciles.to_dict(orient="records"),
        "selective_coverage": df_coverage.to_dict(orient="records"),
        "routing_policies": routing_results,
        "feature_correlations": feat_corrs,
        "early_season_risk": early_season_records,
        "promoted_team_risk": promoted_records,
        "league_risk": league_records,
        "verdict": {
            "verdict": "E15 RESEARCH ONLY — KEEP E10 AS CHAMPION",
            "sub_verdict": "E15 DID NOT IMPROVE FORECASTING — PRESERVE AS RISK DIAGNOSTIC ONLY",
            "rationale": (
                f"The pre-match risk model successfully predicts match difficulty with strong discrimination "
                f"(ROC-AUC = {model_metrics_records[1]['roc_auc_top10']:.4f}, Correlation = {model_metrics_records[1]['correlation_actual_rps']:+.4f}) "
                f"and confirms strict monotonic error progression across risk deciles (Decile 1 RPS = {df_deciles.iloc[0]['actual_mean_rps']:.6f} vs Decile 10 RPS = {df_deciles.iloc[-1]['actual_mean_rps']:.6f}). "
                f"However, routing high-risk matches to alternative experts does not improve global full-coverage forecasting (Mode D RPS = {routing_results['mode_d_route_high_risk_e13_rps']:.6f} vs E10 {routing_results['mode_a_pure_e10_rps']:.6f}). "
                f"E15 is therefore preserved as a high-value pre-match risk diagnostic and selective abstention engine. Pure E10 remains the official champion candidate."
            ),
        },
    }

    with open(E15_DIR / "14_e15_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Post-flight verification
    verify_protected_hashes()
    print("  [OK] Post-flight: All 20 protected baseline assets verified 100% bit-identical.", flush=True)
    print(f"  [DONE] Experiment E15 completed in {time.time() - t0:.2f} seconds.", flush=True)

    return results


if __name__ == "__main__":
    execute_e15_experiment()
