"""FPP World Cup Elo Controlled Research Experiment Runner.

Executes:
  - 3 approved walk-forward folds (reused from src/models/splits.py)
  - Champion V2 Poisson+Venue (84 features)
  - Elo Challengers (85, 87 features)
  - Standalone Elo baseline
  - Controlled Elo variants
  - Subgroup & Early-season analysis (matches 1-5, 6-10, 11+, promoted vs established, elo mismatch)
  - Pre/post MD5 integrity checks
"""
from __future__ import annotations
from collections import defaultdict

import hashlib
import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add project src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.ablation import MODEL_B_COLUMNS
from models.baselines import CLASS_ORDER
from models.config import (
    FINAL_TEST_SEASONS,
    SEASON_NAME_TO_IDS,
    WALK_FORWARD_FOLDS,
)
from models.data import load_supervised_dataset, SupervisedDataset
from models.evaluate import accuracy, brier_score, confusion_matrix, evaluate, log_loss
from models.splits import iter_walk_forward_folds
from models.train import LogisticRegressionPreprocessor
from models.v2_artifact import VENUE_COLUMNS

from elo_engine import EloEngine, load_matches_and_build_elo

# Pinned files for integrity gate
PROTECTED_FILES = {
    "v2_poisson_venue.pkl": PROJECT_ROOT / "data" / "models" / "v2_poisson_venue.pkl",
    "v1_logreg.pkl": PROJECT_ROOT / "data" / "models" / "v1_logreg.pkl",
    "features.db": PROJECT_ROOT / "data" / "processed" / "features.db",
    "matches.db": PROJECT_ROOT / "data" / "processed" / "matches.db",
}

EXPECTED_MD5 = {
    "v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "matches.db": "fdeed042096fa1c851aaee6c84995247",
}

def get_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def check_integrity():
    for name, p in PROTECTED_FILES.items():
        actual = get_md5(p)
        exp = EXPECTED_MD5[name]
        assert actual == exp, f"Integrity failure on {name}: {actual} != {exp}"
    print("INTEGRITY GATE PASSED: All 4 protected files match expected MD5 checksums.")

def compute_rps(y_true: np.ndarray, P: np.ndarray) -> float:
    """Ranked Probability Score for ordered classes ['H', 'D', 'A']."""
    y_idx = np.array([CLASS_ORDER.index(c) for c in y_true])
    Y = np.eye(3)[y_idx]
    
    # Cumulative distributions
    cum_P = np.cumsum(P, axis=1)
    cum_Y = np.cumsum(Y, axis=1)
    
    # RPS = 1/(K-1) sum_{k=1}^{K-1} (cum_P_k - cum_Y_k)^2
    # K=3 -> ( (cum_P_0 - cum_Y_0)^2 + (cum_P_1 - cum_Y_1)^2 ) / 2
    sq_diff = (cum_P[:, :2] - cum_Y[:, :2]) ** 2
    rps_per_row = np.sum(sq_diff, axis=1) / 2.0
    return float(np.mean(rps_per_row))

# Tail-safe Poisson conversion
TAIL_TOL = 1e-15
MASS_TOL = 1e-10
COMPLEMENT_TOL = 1e-10
ROWSUM_TOL = 1e-10

def _grid_size(max_lambda: float, tol: float = TAIL_TOL) -> int:
    from scipy.stats import poisson
    K = int(np.ceil(max_lambda + 1))
    while True:
        if poisson.sf(K, max_lambda) < tol:
            return K + 1
        K = max(K + 1, int(np.ceil(K * 1.2)))

def _pmf_grid(lambdas: np.ndarray, K: int) -> np.ndarray:
    n = len(lambdas)
    grid = np.zeros((n, K), dtype=float)
    grid[:, 0] = np.exp(-lambdas)
    for j in range(1, K):
        grid[:, j] = grid[:, j - 1] * lambdas / j
    return grid

def hda_tail_safe(lam_h: np.ndarray, lam_a: np.ndarray, tol: float = TAIL_TOL):
    K = _grid_size(float(max(np.max(lam_h), np.max(lam_a))), tol)
    ph, pa = _pmf_grid(lam_h, K), _pmf_grid(lam_a, K)
    residual = float(max(np.max(np.abs(1.0 - ph.sum(axis=1))),
                         np.max(np.abs(1.0 - pa.sum(axis=1)))))
    Fa = np.cumsum(pa, axis=1)
    p_home = (ph[:, 1:] * Fa[:, :-1]).sum(axis=1)
    p_draw = (ph * pa).sum(axis=1)
    p_away_direct = (pa[:, 1:] * np.cumsum(ph, axis=1)[:, :-1]).sum(axis=1)
    p_away = 1.0 - p_home - p_draw
    control = float(np.max(np.abs(p_away - p_away_direct)))
    P = np.column_stack([p_home, p_draw, p_away])
    return P, K, residual, control

def fit_poisson_arm(cols: List[str], tr_X: pd.DataFrame, va_X: pd.DataFrame, 
                    htr: np.ndarray, atr: np.ndarray, alpha: float = 1.0, max_iter: int = 2000):
    prep = LogisticRegressionPreprocessor()
    prep.fit(tr_X[cols])
    Etr = prep.transform(tr_X[cols])
    Eva = prep.transform(va_X[cols])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mh = PoissonRegressor(alpha=alpha, max_iter=max_iter).fit(Etr, htr)
        ma = PoissonRegressor(alpha=alpha, max_iter=max_iter).fit(Etr, atr)

    lam_h = mh.predict(Eva)
    lam_a = ma.predict(Eva)
    P, K, res, ctl = hda_tail_safe(lam_h, lam_a)
    return P, lam_h, lam_a

def fit_elo_prob_head(tr_elo_diff: np.ndarray, tr_y: np.ndarray, va_elo_diff: np.ndarray):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(np.column_stack([tr_elo_diff, np.abs(tr_elo_diff)]))
    X_va = scaler.transform(np.column_stack([va_elo_diff, np.abs(va_elo_diff)]))
    
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, tr_y)
    p_raw = clf.predict_proba(X_va)
    order = [list(clf.classes_).index(c) for c in CLASS_ORDER]
    return p_raw[:, order]

def compute_all_metrics(y_true: np.ndarray, P: np.ndarray, 
                        lam_h: np.ndarray | None = None, lam_a: np.ndarray | None = None,
                        y_h: np.ndarray | None = None, y_a: np.ndarray | None = None) -> Dict:
    ev = evaluate(y_true, P)
    rps_val = compute_rps(y_true, P)
    
    # Draw metrics
    y_is_draw = (y_true == "D")
    draw_prob = P[:, 1]
    draw_auc = float(roc_auc_score(y_is_draw.astype(int), draw_prob)) if len(np.unique(y_is_draw)) > 1 else np.nan
    
    cm = ev.confusion_matrix
    draw_idx = CLASS_ORDER.index("D")
    draw_tp = cm[draw_idx, draw_idx]
    draw_actual = cm[draw_idx, :].sum()
    draw_pred = cm[:, draw_idx].sum()
    draw_recall = draw_tp / draw_actual if draw_actual > 0 else 0.0
    draw_precision = draw_tp / draw_pred if draw_pred > 0 else 0.0
    draw_f1 = (2 * draw_precision * draw_recall / (draw_precision + draw_recall)) if (draw_precision + draw_recall) > 0 else 0.0

    out = {
        "accuracy": ev.accuracy,
        "log_loss": ev.log_loss,
        "brier": ev.brier,
        "rps": rps_val,
        "draw_auc": draw_auc,
        "draw_recall": draw_recall,
        "draw_precision": draw_precision,
        "draw_f1": draw_f1,
        "draw_argmax_share": float(draw_pred / len(y_true)),
        "mean_p_home": float(P[:, 0].mean()),
        "mean_p_draw": float(P[:, 1].mean()),
        "mean_p_away": float(P[:, 2].mean()),
        "sd_p_draw": float(P[:, 1].std()),
        "p_draw_gt_third": float((P[:, 1] > 1/3).mean()),
    }
    
    if lam_h is not None and y_h is not None:
        out["home_goal_mae"] = float(np.mean(np.abs(y_h - lam_h)))
        out["away_goal_mae"] = float(np.mean(np.abs(y_a - lam_a)))
    
    return out

def run_experiment():
    print("=" * 80)
    print("PHASE 11 PRE-EXPERIMENT INTEGRITY GATE")
    print("=" * 80)
    check_integrity()

    print("\n" + "=" * 80)
    print("PHASE 3 & 4: DATA PREPARATION & CAUSAL ELO MERGE")
    print("=" * 80)
    fdb_path = PROTECTED_FILES["features.db"]
    mdb_path = PROTECTED_FILES["matches.db"]

    ds = load_supervised_dataset(fdb_path)
    print(f"Loaded supervised dataset: {len(ds)} rows")

    # Goals lookup
    con = sqlite3.connect(f"file:{fdb_path.resolve()}?mode=ro", uri=True)
    try:
        goals = pd.read_sql_query(
            "SELECT fixture_id, label_home_goals, label_away_goals "
            "FROM feature_rows WHERE label_result IS NOT NULL", con)
    finally:
        con.close()
    gh = dict(zip(goals["fixture_id"], goals["label_home_goals"]))
    ga = dict(zip(goals["fixture_id"], goals["label_away_goals"]))

    # 1. Base Elo Engine (K=20, HomeAdv=100, GoalDiff=True, Reversion=0)
    print("Computing Baseline Causal Elo Features...")
    elo_df_base = load_matches_and_build_elo(mdb_path, k_factor=20.0, home_advantage=100.0, use_goal_diff=True, mean_reversion=0.0)
    
    # 2. Variants for Phase 9
    print("Computing Elo Variant Features...")
    elo_df_k30 = load_matches_and_build_elo(mdb_path, k_factor=30.0, home_advantage=100.0, use_goal_diff=True, mean_reversion=0.0)
    elo_df_rev20 = load_matches_and_build_elo(mdb_path, k_factor=20.0, home_advantage=100.0, use_goal_diff=True, mean_reversion=0.20)
    elo_df_no_hadv = load_matches_and_build_elo(mdb_path, k_factor=20.0, home_advantage=0.0, use_goal_diff=True, mean_reversion=0.0)

    # Merge Elo features onto dataset
    elo_map_base = elo_df_base.set_index("fixture_id")
    ds.X["elo_diff"] = ds.metadata["fixture_id"].map(elo_map_base["elo_diff"])
    ds.X["home_elo"] = ds.metadata["fixture_id"].map(elo_map_base["home_elo"])
    ds.X["away_elo"] = ds.metadata["fixture_id"].map(elo_map_base["away_elo"])
    ds.X["abs_elo_diff"] = ds.metadata["fixture_id"].map(elo_map_base["abs_elo_diff"])
    
    # Variants features
    ds.X["elo_diff_k30"] = ds.metadata["fixture_id"].map(elo_df_k30.set_index("fixture_id")["elo_diff"])
    ds.X["elo_diff_rev20"] = ds.metadata["fixture_id"].map(elo_df_rev20.set_index("fixture_id")["elo_diff"])
    ds.X["elo_diff_no_hadv"] = ds.metadata["fixture_id"].map(elo_df_no_hadv.set_index("fixture_id")["elo_diff"])

    # Define Feature Sets
    BASE_84_COLS = list(MODEL_B_COLUMNS) + list(VENUE_COLUMNS)
    assert len(BASE_84_COLS) == 84, f"Expected 84 features, got {len(BASE_84_COLS)}"

    CHALLENGER_85_COLS = BASE_84_COLS + ["elo_diff"]
    CHALLENGER_87_COLS = BASE_84_COLS + ["home_elo", "away_elo", "elo_diff"]
    
    VARIANT_K30_COLS = BASE_84_COLS + ["elo_diff_k30"]
    VARIANT_REV20_COLS = BASE_84_COLS + ["elo_diff_rev20"]
    VARIANT_NO_HADV_COLS = BASE_84_COLS + ["elo_diff_no_hadv"]

    ARMS = {
        "Champion V2 (84 cols)": BASE_84_COLS,
        "Challenger + elo_diff (85 cols)": CHALLENGER_85_COLS,
        "Challenger + 3 Elo (87 cols)": CHALLENGER_87_COLS,
        "Variant: Elo K=30 (85 cols)": VARIANT_K30_COLS,
        "Variant: Elo Reversion 0.20 (85 cols)": VARIANT_REV20_COLS,
        "Variant: Elo No HomeAdv (85 cols)": VARIANT_NO_HADV_COLS,
    }

    print("\n" + "=" * 80)
    print("PHASE 4 & 6: WALK-FORWARD FOLD EXECUTION")
    print("=" * 80)

    results_by_fold = {arm: {} for arm in ARMS}
    results_by_fold["Standalone Elo Benchmark"] = {}
    
    # Store predictions and validation metadata for subgroup analysis
    val_records = []

    for fold, tr, va in iter_walk_forward_folds(ds):
        print(f"\n--- {fold.name.upper()} ---")
        print(f"  Train Seasons: {fold.train_seasons} (n={len(tr)})  Unix: {tr.metadata['unix'].min()} -> {tr.metadata['unix'].max()}")
        print(f"  Val Seasons:   {fold.validation_seasons} (n={len(va)})  Unix: {va.metadata['unix'].min()} -> {va.metadata['unix'].max()}")
        
        # Targets
        fids_tr = tr.metadata["fixture_id"].to_numpy()
        htr = np.array([gh[f] for f in fids_tr], dtype=float)
        atr = np.array([ga[f] for f in fids_tr], dtype=float)

        fids_va = va.metadata["fixture_id"].to_numpy()
        hva = np.array([gh[f] for f in fids_va], dtype=float)
        ava = np.array([ga[f] for f in fids_va], dtype=float)
        y_va = va.y.to_numpy()

        # Standalone Elo Benchmark
        p_elo_bench = fit_elo_prob_head(tr.X["elo_diff"].to_numpy(), tr.y.to_numpy(), va.X["elo_diff"].to_numpy())
        results_by_fold["Standalone Elo Benchmark"][fold.name] = compute_all_metrics(y_va, p_elo_bench)

        fold_preds = {"y_true": y_va, "hva": hva, "ava": ava, "fixture_id": fids_va, "season_id": va.metadata["season_id"].to_numpy(),
                      "competition_id": va.metadata["competition_id"].to_numpy(), "unix": va.metadata["unix"].to_numpy(),
                      "elo_diff": va.X["elo_diff"].to_numpy(), "home_elo": va.X["home_elo"].to_numpy(), "away_elo": va.X["away_elo"].to_numpy()}

        # Fit all arms
        for arm_name, cols in ARMS.items():
            P, lam_h, lam_a = fit_poisson_arm(cols, tr.X, va.X, htr, atr)
            metrics = compute_all_metrics(y_va, P, lam_h, lam_a, hva, ava)
            results_by_fold[arm_name][fold.name] = metrics
            fold_preds[f"P_{arm_name}"] = P
            fold_preds[f"lam_h_{arm_name}"] = lam_h
            fold_preds[f"lam_a_{arm_name}"] = lam_a

        val_records.append(fold_preds)

    # Summary table per fold and pooled
    print("\n" + "=" * 80)
    print("POOLED METRICS COMPARISON (Unweighted mean across 3 folds)")
    print("=" * 80)
    
    summary_table = []
    for arm_name in results_by_fold:
        pooled = {}
        for m in ["accuracy", "log_loss", "brier", "rps", "home_goal_mae", "away_goal_mae", "draw_auc", "draw_recall", "draw_argmax_share"]:
            vals = [results_by_fold[arm_name][f.name].get(m, np.nan) for f in WALK_FORWARD_FOLDS]
            pooled[m] = float(np.mean(vals))
        summary_table.append({"Arm": arm_name, **pooled})

    summary_df = pd.DataFrame(summary_table)
    print(summary_df.to_string(index=False))

    # Delta table vs Champion
    champ_pooled = summary_df.set_index("Arm").loc["Champion V2 (84 cols)"]
    print("\n" + "=" * 80)
    print("DELTAS VS CHAMPION V2 (Candidate - Champion; Negative is better for LL/Brier/RPS/MAE)")
    print("=" * 80)
    deltas = []
    for arm_name in summary_df["Arm"]:
        if arm_name == "Champion V2 (84 cols)":
            continue
        row = summary_df.set_index("Arm").loc[arm_name]
        d = {"Arm": arm_name}
        for m in ["accuracy", "log_loss", "brier", "rps", "home_goal_mae", "away_goal_mae", "draw_auc", "draw_recall"]:
            d[f"d_{m}"] = row[m] - champ_pooled[m]
        deltas.append(d)
    print(pd.DataFrame(deltas).to_string(index=False))

    print("\n" + "=" * 80)
    print("FOLD-BY-FOLD DETAILED BREAKDOWN")
    print("=" * 80)
    for f in WALK_FORWARD_FOLDS:
        print(f"\n>>> {f.name.upper()} <<<")
        fold_rows = []
        for arm_name in results_by_fold:
            m = results_by_fold[arm_name][f.name]
            fold_rows.append({
                "Arm": arm_name,
                "Accuracy": f"{m['accuracy']*100:.2f}%",
                "LogLoss": f"{m['log_loss']:.6f}",
                "Brier": f"{m['brier']:.6f}",
                "RPS": f"{m['rps']:.6f}",
                "DrawRecall": f"{m['draw_recall']:.4f}",
                "DrawAUC": f"{m.get('draw_auc', np.nan):.4f}",
                "H_MAE": f"{m.get('home_goal_mae', np.nan):.4f}",
                "A_MAE": f"{m.get('away_goal_mae', np.nan):.4f}",
            })
        print(pd.DataFrame(fold_rows).to_string(index=False))

    # Fold wins
    print("\n" + "=" * 80)
    print("FOLD CONSISTENCY & DECISION RULE EVALUATION")
    print("=" * 80)
    for arm_name in ARMS:
        if arm_name == "Champion V2 (84 cols)":
            continue
        ll_wins = sum(results_by_fold[arm_name][f.name]["log_loss"] < results_by_fold["Champion V2 (84 cols)"][f.name]["log_loss"] for f in WALK_FORWARD_FOLDS)
        acc_wins = sum(results_by_fold[arm_name][f.name]["accuracy"] > results_by_fold["Champion V2 (84 cols)"][f.name]["accuracy"] for f in WALK_FORWARD_FOLDS)
        brier_wins = sum(results_by_fold[arm_name][f.name]["brier"] < results_by_fold["Champion V2 (84 cols)"][f.name]["brier"] for f in WALK_FORWARD_FOLDS)
        rps_wins = sum(results_by_fold[arm_name][f.name]["rps"] < results_by_fold["Champion V2 (84 cols)"][f.name]["rps"] for f in WALK_FORWARD_FOLDS)
        
        d_ll_pooled = summary_df.set_index("Arm").loc[arm_name, "log_loss"] - champ_pooled["log_loss"]
        d_acc_pooled = summary_df.set_index("Arm").loc[arm_name, "accuracy"] - champ_pooled["accuracy"]
        
        pass_rule = (d_ll_pooled < 0) and (ll_wins >= 2)
        print(f"Arm: {arm_name}")
        print(f"  Log Loss Pooled Delta: {d_ll_pooled:+.6f} | Fold Wins: {ll_wins}/3")
        print(f"  Accuracy Pooled Delta: {d_acc_pooled*100:+.2f}pp | Fold Wins: {acc_wins}/3")
        print(f"  Brier Fold Wins: {brier_wins}/3 | RPS Fold Wins: {rps_wins}/3")
        print(f"  Preregistered Rule Pass: {'YES (PROMISING)' if pass_rule else 'NO (REJECT/INVESTIGATE)'}")

    # =========================================================================
    # PHASE 7: SUBGROUP & EARLY-SEASON ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 7: EARLY-SEASON & SUBGROUP ANALYSIS")
    print("=" * 80)

    # Combine all validation fixtures across 3 folds
    all_fids = np.concatenate([r["fixture_id"] for r in val_records])
    all_y = np.concatenate([r["y_true"] for r in val_records])
    all_unix = np.concatenate([r["unix"] for r in val_records])
    all_season = np.concatenate([r["season_id"] for r in val_records])
    all_comp = np.concatenate([r["competition_id"] for r in val_records])
    all_elo_diff = np.concatenate([r["elo_diff"] for r in val_records])
    all_P_champ = np.vstack([r["P_Champion V2 (84 cols)"] for r in val_records])
    all_P_elo85 = np.vstack([r["P_Challenger + elo_diff (85 cols)"] for r in val_records])
    all_P_elo87 = np.vstack([r["P_Challenger + 3 Elo (87 cols)"] for r in val_records])

    # To calculate match index of season for each team:
    # Load all fixtures from matches.db to compute team match number within each season
    uri = f"file:{mdb_path.resolve()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    all_matches_df = pd.read_sql_query(
        "SELECT fixture_id, season_id, competition_id, unix, home_id, away_id, status "
        "FROM fixtures ORDER BY unix ASC, fixture_id ASC", con
    )
    con.close()

    # Calculate match number within season for each team
    team_season_match_num = {}
    for row in all_matches_df.itertuples(index=False):
        for side, tid in [("home", row.home_id), ("away", row.away_id)]:
            key = (row.season_id, tid)
            team_season_match_num[key] = team_season_match_num.get(key, 0) + 1
            if row.fixture_id not in team_season_match_num:
                team_season_match_num[row.fixture_id] = {}
            team_season_match_num[row.fixture_id][side] = team_season_match_num[key]

    match_nums_home = np.array([team_season_match_num.get(f, {}).get("home", 99) for f in all_fids])
    match_nums_away = np.array([team_season_match_num.get(f, {}).get("away", 99) for f in all_fids])
    max_match_num = np.maximum(match_nums_home, match_nums_away)
    min_match_num = np.minimum(match_nums_home, match_nums_away)

    # 1. Early-Season breakdown (Matches 1-5, 6-10, 11+)
    buckets = {
        "Matches 1-5 (Early Season)": max_match_num <= 5,
        "Matches 6-10 (Transition)": (min_match_num > 5) & (max_match_num <= 10),
        "Matches 11+ (Mature Season)": min_match_num > 10,
    }

    print("\n--- PERFORMANCE BY MATCH-OF-SEASON PHASE ---")
    subgroup_rows = []
    for bname, mask in buckets.items():
        n_m = mask.sum()
        if n_m == 0:
            continue
        y_b = all_y[mask]
        P_c = all_P_champ[mask]
        P_e85 = all_P_elo85[mask]
        
        acc_c, ll_c, br_c = accuracy(y_b, P_c), log_loss(y_b, P_c), brier_score(y_b, P_c)
        acc_e, ll_e, br_e = accuracy(y_b, P_e85), log_loss(y_b, P_e85), brier_score(y_b, P_e85)
        rps_c = compute_rps(y_b, P_c)
        rps_e = compute_rps(y_b, P_e85)

        subgroup_rows.append({
            "Bucket": bname,
            "N": n_m,
            "Champ Acc": f"{acc_c*100:.2f}%",
            "Elo85 Acc": f"{acc_e*100:.2f}%",
            "Acc Delta": f"{(acc_e-acc_c)*100:+.2f}pp",
            "Champ LL": f"{ll_c:.4f}",
            "Elo85 LL": f"{ll_e:.4f}",
            "LL Delta": f"{ll_e-ll_c:+.4f}",
            "Champ RPS": f"{rps_c:.4f}",
            "Elo85 RPS": f"{rps_e:.4f}",
            "RPS Delta": f"{rps_e-rps_c:+.4f}",
        })
    print(pd.DataFrame(subgroup_rows).to_string(index=False))

    # 2. Elo Mismatch breakdown
    mismatch_buckets = {
        "Large Elo Mismatch (|Δ| >= 150)": np.abs(all_elo_diff) >= 150,
        "Medium Elo Mismatch (75 <= |Δ| < 150)": (np.abs(all_elo_diff) >= 75) & (np.abs(all_elo_diff) < 150),
        "Small Elo Mismatch (|Δ| < 75)": np.abs(all_elo_diff) < 75,
    }

    print("\n--- PERFORMANCE BY ELO MISMATCH MAGNITUDE ---")
    mismatch_rows = []
    for bname, mask in mismatch_buckets.items():
        n_m = mask.sum()
        if n_m == 0:
            continue
        y_b = all_y[mask]
        P_c = all_P_champ[mask]
        P_e85 = all_P_elo85[mask]
        
        acc_c, ll_c = accuracy(y_b, P_c), log_loss(y_b, P_c)
        acc_e, ll_e = accuracy(y_b, P_e85), log_loss(y_b, P_e85)
        rps_c = compute_rps(y_b, P_c)
        rps_e = compute_rps(y_b, P_e85)

        mismatch_rows.append({
            "Bucket": bname,
            "N": n_m,
            "Champ Acc": f"{acc_c*100:.2f}%",
            "Elo85 Acc": f"{acc_e*100:.2f}%",
            "Acc Delta": f"{(acc_e-acc_c)*100:+.2f}pp",
            "Champ LL": f"{ll_c:.4f}",
            "Elo85 LL": f"{ll_e:.4f}",
            "LL Delta": f"{ll_e-ll_c:+.4f}",
            "Champ RPS": f"{rps_c:.4f}",
            "Elo85 RPS": f"{rps_e:.4f}",
            "RPS Delta": f"{rps_e-rps_c:+.4f}",
        })
    print(pd.DataFrame(mismatch_rows).to_string(index=False))

    # 3. Promoted Teams Analysis
    # Teams with fewer than 38 prior matches in dataset at match kickoff
    # Track historical appearances prior to each match
    team_history_count = defaultdict(int)
    is_promoted_match = []
    for row in all_matches_df.itertuples(index=False):
        h_prev = team_history_count[row.home_id]
        a_prev = team_history_count[row.away_id]
        # Flag if either team had <= 38 prior games in dataset
        promoted = (h_prev <= 38) or (a_prev <= 38)
        if row.fixture_id in set(all_fids):
            is_promoted_match.append((row.fixture_id, promoted))
        team_history_count[row.home_id] += 1
        team_history_count[row.away_id] += 1

    prom_map = dict(is_promoted_match)
    prom_mask = np.array([prom_map.get(f, False) for f in all_fids])

    print("\n--- PERFORMANCE ON PROMOTED / LOW-HISTORY TEAMS ---")
    prom_rows = []
    for bname, mask in [("Promoted / Low-History Teams Involved", prom_mask), ("Established Teams Only", ~prom_mask)]:
        n_m = mask.sum()
        y_b = all_y[mask]
        P_c = all_P_champ[mask]
        P_e85 = all_P_elo85[mask]
        
        acc_c, ll_c = accuracy(y_b, P_c), log_loss(y_b, P_c)
        acc_e, ll_e = accuracy(y_b, P_e85), log_loss(y_b, P_e85)
        rps_c = compute_rps(y_b, P_c)
        rps_e = compute_rps(y_b, P_e85)

        prom_rows.append({
            "Group": bname,
            "N": n_m,
            "Champ Acc": f"{acc_c*100:.2f}%",
            "Elo85 Acc": f"{acc_e*100:.2f}%",
            "Acc Delta": f"{(acc_e-acc_c)*100:+.2f}pp",
            "Champ LL": f"{ll_c:.4f}",
            "Elo85 LL": f"{ll_e:.4f}",
            "LL Delta": f"{ll_e-ll_c:+.4f}",
            "Champ RPS": f"{rps_c:.4f}",
            "Elo85 RPS": f"{rps_e:.4f}",
            "RPS Delta": f"{rps_e-rps_c:+.4f}",
        })
    print(pd.DataFrame(prom_rows).to_string(index=False))

    # Save detailed results to JSON for final reporting
    out_payload = {
        "summary": summary_df.to_dict(orient="records"),
        "per_fold": results_by_fold,
        "subgroup_early_season": subgroup_rows,
        "subgroup_mismatch": mismatch_rows,
        "subgroup_promoted": prom_rows,
    }
    out_json = Path(__file__).resolve().parent / "experiment_results.json"
    out_json.write_text(json.dumps(out_payload, indent=2, cls=NumpyEncoder), encoding="utf-8")
    print(f"\nWrote structured experiment results to {out_json}")

    print("\n" + "=" * 80)
    print("PHASE 11 POST-EXPERIMENT INTEGRITY GATE")
    print("=" * 80)
    check_integrity()

if __name__ == "__main__":
    run_experiment()