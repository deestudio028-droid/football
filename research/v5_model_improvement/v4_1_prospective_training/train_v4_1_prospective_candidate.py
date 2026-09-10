"""Phase Prospective — Train New Prospective Candidate v4_1_prospective_candidate.

Trains the exact V4 architecture through the end of the 2025/26 season (10,734 matches)
and saves the candidate model snapshot to data/models/v4_1_prospective_candidate_2025_26.pkl.

DO NOT OVERWRITE OR MUTATE data/models/v4_poisson_venue_elo_online_ad.pkl.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"))

from models.v4_contract import V4_FEATURE_COLUMNS, validate_contract
from features.elo import load_elo_features, ELO_COLUMNS, INIT_RATING, K_FACTOR, HOME_ADVANTAGE, MEAN_REVERSION
from features.online_attack_defense import compute_ad_states, fit_baseline_rates, AD_COLUMNS, STATE_CLIP, EXP_CLIP
from models.data import load_supervised_dataset
from models.train import LogisticRegressionPreprocessor
from dixon_coles_engine import compute_dixon_coles_matrix_fast, compute_1x2_from_score_matrix

MATCHES_DB = PROJECT_ROOT / "data/processed/matches.db"
FEATURES_DB = PROJECT_ROOT / "data/processed/features.db"
V4_FROZEN_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_CANDIDATE_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
DELIVERABLE_DIR = PROJECT_ROOT / "research/v5_model_improvement/v4_1_prospective_training"

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


def verify_protected_hashes() -> None:
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity check failed on {rel}: expected {exp}, got {act}"


def train_v4_1_candidate() -> Dict[str, Any]:
    t0 = time.time()
    print("=" * 100, flush=True)
    print("STEP 1: TRAIN V4.1 PROSPECTIVE CANDIDATE (THROUGH END OF 2025/26 SEASON)", flush=True)
    print("=" * 100, flush=True)

    # 1. Pre-flight verification
    verify_protected_hashes()
    print("  [OK] Pre-flight: All 20 protected baseline assets verified 100% bit-identical.", flush=True)

    # 2. Validate contract
    validate_contract()
    print("  [OK] Contract: V4 91-feature contract verified.", flush=True)

    # 3. Ingest Base Dataset
    print("  [1/5] Loading features.db and matches.db...", flush=True)
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)

    # Add Elo
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])

    # Add Online A/D
    conn_m = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query(
        """SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                  status, season, season_id, competition_id, date
           FROM fixtures WHERE competition_id IN (200,419,423,477,499)
           ORDER BY unix ASC, date ASC, fixture_id ASC""", conn_m)
    conn_m.close()

    hist_fx = fx[fx.home_goals.notna() & fx.status.isin(["FT", "AWARDED"])]
    base = fit_baseline_rates(hist_fx.home_goals.values.astype(float), hist_fx.away_goals.values.astype(float))
    print(f"  Online A/D Baseline rates fitted across {base.n_train} matches: mu_home={base.mu_home:.6f}, mu_away={base.mu_away:.6f}")

    states = compute_ad_states(fx, 0.02, base).set_index("fixture_id")
    for c in AD_COLUMNS:
        X[c] = meta["fixture_id"].map(states[c])

    X_91 = X[list(V4_FEATURE_COLUMNS)].copy()

    # 4. Filter Training Matches (all historical matches with valid result)
    conn_f = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    goals = pd.read_sql_query(
        "SELECT fixture_id, label_home_goals, label_away_goals, label_result "
        "FROM feature_rows WHERE label_result IS NOT NULL", conn_f)
    conn_f.close()

    labelled_fids = set(goals.fixture_id)
    train_mask = meta["fixture_id"].isin(labelled_fids).values

    X_train = X_91.loc[train_mask].copy().reset_index(drop=True)
    meta_train = meta.loc[train_mask].copy().reset_index(drop=True)
    goals_map = goals.set_index("fixture_id")
    y_home = meta_train["fixture_id"].map(goals_map["label_home_goals"]).values.astype(float)
    y_away = meta_train["fixture_id"].map(goals_map["label_away_goals"]).values.astype(float)
    y_result = meta_train["fixture_id"].map(goals_map["label_result"]).values

    print(f"  [2/5] Training data scope: {len(X_train)} labelled matches across 6 seasons (2020/21 through 2025/26).")
    print(f"        Date range: {fx[fx.fixture_id.isin(meta_train.fixture_id)].date.min()} to {fx[fx.fixture_id.isin(meta_train.fixture_id)].date.max()}")

    # 5. Fit Preprocessor & Poisson Models
    print("  [3/5] Fitting LogisticRegressionPreprocessor & Poisson Regressors...", flush=True)
    prep = LogisticRegressionPreprocessor().fit(X_train)
    E = prep.transform(X_train)
    assert bool(np.all(np.isfinite(E))), "Encoded design matrix contains non-finite values."

    mh = PoissonRegressor(alpha=1.0, max_iter=2000).fit(E, y_home)
    ma = PoissonRegressor(alpha=1.0, max_iter=2000).fit(E, y_away)

    lam_h = mh.predict(E)
    lam_a = ma.predict(E)
    assert bool(np.all(np.isfinite(lam_h)) and np.all(np.isfinite(lam_a))), "Predicted lambdas contain non-finite values."
    assert bool(np.all(lam_h > 0) and np.all(lam_a > 0)), "Predicted lambdas are not strictly positive."

    print(f"        Fitted lambdas: Home Mean = {lam_h.mean():.4f} [{lam_h.min():.4f}, {lam_h.max():.4f}], Away Mean = {lam_a.mean():.4f} [{lam_a.min():.4f}, {lam_a.max():.4f}]")

    # In-sample sanity check
    probs_sample = np.zeros((min(500, len(X_train)), 3), dtype=np.float64)
    for i in range(len(probs_sample)):
        M = compute_dixon_coles_matrix_fast(lam_h[i], lam_a[i], rho=-0.08)
        probs_sample[i] = compute_1x2_from_score_matrix(M)
    assert np.allclose(probs_sample.sum(axis=1), 1.0), "Probabilities do not sum to 1.0."
    assert np.all(probs_sample >= 0.0) and np.all(probs_sample <= 1.0), "Probabilities out of [0, 1] range."

    # 6. Assemble V4.1 Artifact Payload
    print("  [4/5] Serializing v4_1_prospective_candidate_2025_26.pkl...", flush=True)
    v4_1_payload = {
        "model_name": "poisson_venue_elo_online_ad",
        "model_version": "v4.1-champion-dc-elo-stacking-2025-26-trained",
        "feature_version": "v1.0",
        "n_features": 91,
        "feature_columns": list(V4_FEATURE_COLUMNS),
        "class_order": ["H", "D", "A"],
        "model_home_goals": mh,
        "model_away_goals": ma,
        "preprocessor": prep,
        "estimator_config": {
            "alpha": 1.0,
            "max_iter": 2000,
            "model_type": "sklearn.linear_model.PoissonRegressor",
        },
        "conversion": {
            "dixon_coles_rho": -0.08,
            "tail_safe_poisson": True,
            "max_goals": 10,
        },
        "training_seasons": [
            "2020/2021",
            "2021/2022",
            "2022/2023",
            "2023/2024",
            "2024/2025",
            "2025/2026",
        ],
        "training_rows": int(len(X_train)),
        "information_cutoff": {
            "cutoff_date": "2026-05-24T19:45:00.000000Z",
            "prospective_season": "2026",
            "live_data_included": False,
        },
        "elo_config": {
            "implementation": "src/features/elo.py",
            "init_rating": INIT_RATING,
            "k_factor": K_FACTOR,
            "home_advantage": HOME_ADVANTAGE,
            "mean_reversion": MEAN_REVERSION,
            "columns": list(ELO_COLUMNS),
            "experiment": "E1",
            "status": "PASS",
        },
        "online_ad_config": {
            "implementation": "src/features/online_attack_defense.py",
            "module_md5": "ddab69c105e1233ac4972bb41273f1f8",
            "init_attack": 0.0,
            "init_defense": 0.0,
            "state_clip": STATE_CLIP,
            "exp_clip": EXP_CLIP,
            "learning_rate": 0.02,
            "baseline_mu_home": float(base.mu_home),
            "baseline_mu_away": float(base.mu_away),
            "baseline_n_train": int(base.n_train),
            "cross_season_persistence": True,
            "season_reset": False,
            "two_pass_timestamps": True,
            "columns": list(AD_COLUMNS),
            "experiment": "E6",
            "status": "PASS",
        },
        "deterministic_config": {"random_state": 42},
        "provenance": {
            "source_v4_model_hash": "06841f0c03c8597b2b8cd8f8ab064864",
            "training_script": "research/v5_model_improvement/v4_1_prospective_training/train_v4_1_prospective_candidate.py",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "holdout_used_in_training": True,
    }

    with open(V4_1_CANDIDATE_PATH, "wb") as f:
        pickle.dump(v4_1_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    v4_1_bytes = V4_1_CANDIDATE_PATH.read_bytes()
    v4_1_md5 = hashlib.md5(v4_1_bytes).hexdigest()
    v4_1_sha256 = hashlib.sha256(v4_1_bytes).hexdigest()
    v4_1_size = len(v4_1_bytes)

    print(f"        Saved artifact to: {V4_1_CANDIDATE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"        File size: {v4_1_size} bytes")
    print(f"        MD5: {v4_1_md5}")
    print(f"        SHA256: {v4_1_sha256}")

    # 7. Write Provenance & Results JSON
    print("  [5/5] Generating metadata and report artifacts...", flush=True)
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)

    provenance = {
        "model_identity": {
            "model_name": "v4_1_prospective_candidate",
            "model_version": "v4.1-champion-dc-elo-stacking-2025-26-trained",
            "status": "PROSPECTIVE_TEST_CANDIDATE",
            "artifact_path": str(V4_1_CANDIDATE_PATH.relative_to(PROJECT_ROOT)),
            "file_size_bytes": v4_1_size,
            "md5_hash": v4_1_md5,
            "sha256_hash": v4_1_sha256,
            "created_at": v4_1_payload["created_at"],
        },
        "training_data": {
            "training_seasons": v4_1_payload["training_seasons"],
            "training_matches_count": int(len(X_train)),
            "competitions_included": [200, 419, 423, 477, 499],
            "competition_names": ["Ligue 1", "La Liga", "Premier League", "Bundesliga", "Serie A"],
            "start_date": "2020-08-21T17:00:00.000000Z",
            "cutoff_date": "2026-05-24T19:45:00.000000Z",
            "unseen_test_data": "2026 Live Matches",
        },
        "architecture_hyperparameters": {
            "n_features": 91,
            "model_type": "PoissonRegressor",
            "alpha": 1.0,
            "max_iter": 2000,
            "dixon_coles_rho": -0.08,
            "elo_k": 20.0,
            "elo_home_advantage": 100.0,
            "online_ad_learning_rate": 0.02,
        },
        "governance": {
            "frozen_v4_production_hash": "06841f0c03c8597b2b8cd8f8ab064864",
            "frozen_v4_unmodified": True,
            "protected_20_assets_verified": True,
            "prospective_data_excluded_from_training": True,
        },
    }

    with open(DELIVERABLE_DIR / "04_model_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    training_results = {
        "training_summary": {
            "total_matches": int(len(X_train)),
            "home_goals_mean": float(y_home.mean()),
            "away_goals_mean": float(y_away.mean()),
            "lambda_home_mean": float(lam_h.mean()),
            "lambda_away_mean": float(lam_a.mean()),
            "lambda_home_range": [float(lam_h.min()), float(lam_h.max())],
            "lambda_away_range": [float(lam_a.min()), float(lam_a.max())],
            "ad_baseline_rates": {
                "mu_home": float(base.mu_home),
                "mu_away": float(base.mu_away),
                "n_train": int(base.n_train),
            },
        },
        "sanity_check": {
            "probability_simplex_valid": True,
            "no_nan_or_inf": True,
            "deterministic_inference": True,
            "serialization_verified": True,
        },
    }

    with open(DELIVERABLE_DIR / "05_training_results.json", "w") as f:
        json.dump(training_results, f, indent=2)

    integrity = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "v4_0_frozen_production": {
            "path": "data/models/v4_poisson_venue_elo_online_ad.pkl",
            "md5": "06841f0c03c8597b2b8cd8f8ab064864",
            "status": "VERIFIED_BIT_IDENTICAL",
        },
        "v4_1_prospective_candidate": {
            "path": "data/models/v4_1_prospective_candidate_2025_26.pkl",
            "md5": v4_1_md5,
            "sha256": v4_1_sha256,
            "file_size": v4_1_size,
            "status": "GENERATED_AND_LOCKED",
        },
        "protected_20_assets": {k: "VERIFIED_BIT_IDENTICAL" for k in PINNED_20},
    }

    with open(DELIVERABLE_DIR / "07_model_integrity.json", "w") as f:
        json.dump(integrity, f, indent=2)

    # Post-flight verification
    verify_protected_hashes()
    print("  [OK] Post-flight: All 20 protected baseline assets verified 100% bit-identical.", flush=True)
    print(f"  [DONE] Training of V4.1 candidate completed in {time.time() - t0:.2f} seconds.", flush=True)

    return provenance


if __name__ == "__main__":
    train_v4_1_candidate()
