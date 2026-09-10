"""Trains the V3 Poisson+Venue+Persistent Elo candidate artifact.

Trains exclusively on 2020/21 through 2024/25 (FINAL_TRAIN_SEASONS).
2025/26 is NEVER accessed or trained on.
Saves to data/models/v3_poisson_venue_elo_candidate.pkl.
Leaves v2_poisson_venue.pkl and v1_logreg.pkl untouched.
"""
from __future__ import annotations

import hashlib
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.baselines import CLASS_ORDER
from models.config import FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
from models.data import load_supervised_dataset
from models.splits import season_ids_for
from models.train import LogisticRegressionPreprocessor
from models.v3_contract import V3_FEATURE_COLUMNS, V3_N_FEATURES
from elo_engine import load_matches_and_build_elo

CANDIDATE_OUTPUT_PATH = PROJECT_ROOT / "data" / "models" / "v3_poisson_venue_elo_candidate.pkl"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"

def main():
    print("=== TRAINING V3 POISSON+VENUE+ELO CANDIDATE ARTIFACT ===")
    
    # 1. Load supervised dataset
    ds = load_supervised_dataset(FEATURES_DB)
    print(f"Loaded feature rows: {len(ds)}")

    # 2. Compute pre-match Elo features
    print("Computing causal Elo features...")
    elo_df = load_matches_and_build_elo(MATCHES_DB, k_factor=20.0, home_advantage=100.0, use_goal_diff=True, mean_reversion=0.0)
    elo_map = elo_df.set_index("fixture_id")

    ds.X["home_elo"] = ds.metadata["fixture_id"].map(elo_map["home_elo"])
    ds.X["away_elo"] = ds.metadata["fixture_id"].map(elo_map["away_elo"])
    ds.X["elo_diff"] = ds.metadata["fixture_id"].map(elo_map["elo_diff"])

    # Verify all 87 features exist
    for col in V3_FEATURE_COLUMNS:
        assert col in ds.X.columns, f"Missing feature: {col}"
    assert len(V3_FEATURE_COLUMNS) == 87, f"Expected 87 features, got {len(V3_FEATURE_COLUMNS)}"

    # 3. Filter to training seasons (2020/21 through 2024/25)
    train_ids = season_ids_for(FINAL_TRAIN_SEASONS)
    test_ids = season_ids_for(FINAL_TEST_SEASONS)
    
    train_mask = ds.metadata["season_id"].isin(train_ids)
    test_mask = ds.metadata["season_id"].isin(test_ids)
    
    assert not (set(ds.metadata.loc[train_mask, "season_id"]) & set(test_ids)), "2025/26 leaked into train set!"
    print(f"Training fixtures count: {train_mask.sum()}")
    print(f"Quarantined 2025/26 count: {test_mask.sum()} (NOT USED)")

    X_train = ds.X.loc[train_mask, list(V3_FEATURE_COLUMNS)].copy()
    fids_train = ds.metadata.loc[train_mask, "fixture_id"].to_numpy()

    # 4. Load goal targets
    con = sqlite3.connect(f"file:{FEATURES_DB.resolve()}?mode=ro", uri=True)
    try:
        goals = pd.read_sql_query(
            "SELECT fixture_id, label_home_goals, label_away_goals "
            "FROM feature_rows WHERE label_result IS NOT NULL", con)
    finally:
        con.close()
    gh = dict(zip(goals["fixture_id"], goals["label_home_goals"]))
    ga = dict(zip(goals["fixture_id"], goals["label_away_goals"]))

    htr = np.array([gh[f] for f in fids_train], dtype=float)
    atr = np.array([ga[f] for f in fids_train], dtype=float)

    # 5. Fit preprocessor and models
    print("Fitting LogisticRegressionPreprocessor on training rows...")
    prep = LogisticRegressionPreprocessor()
    prep.fit(X_train)
    Etr = prep.transform(X_train)

    print("Fitting PoissonRegressor on home goals...")
    mh = PoissonRegressor(alpha=1.0, max_iter=2000).fit(Etr, htr)
    print("Fitting PoissonRegressor on away goals...")
    ma = PoissonRegressor(alpha=1.0, max_iter=2000).fit(Etr, atr)

    # 6. Package artifact
    artifact_payload = {
        "model_name": "poisson_venue_elo",
        "model_version": "v3.0-poisson-venue-elo",
        "feature_version": "v1.0",
        "n_features": 87,
        "feature_columns": list(V3_FEATURE_COLUMNS),
        "class_order": list(CLASS_ORDER),
        "model_home_goals": mh,
        "model_away_goals": ma,
        "preprocessor": prep,
        "estimator_config": {"alpha": 1.0, "max_iter": 2000},
        "conversion": {
            "type": "tail_safe_bivariate_poisson",
            "method": "hda_tail_safe",
            "tail_tol": 1e-15,
        },
        "training_seasons": list(FINAL_TRAIN_SEASONS),
        "holdout_used_in_training": False,
    }

    # 7. Write candidate artifact
    CANDIDATE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATE_OUTPUT_PATH, "wb") as f:
        pickle.dump(artifact_payload, f, protocol=5)

    md5 = hashlib.md5(CANDIDATE_OUTPUT_PATH.read_bytes()).hexdigest()
    print(f"Successfully saved V3 Candidate Artifact:")
    print(f"  Path: {CANDIDATE_OUTPUT_PATH}")
    print(f"  MD5:  {md5}")
    print(f"  Size: {CANDIDATE_OUTPUT_PATH.stat().st_size} bytes")

if __name__ == "__main__":
    main()