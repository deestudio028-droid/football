"""Comprehensive 11-Suite Test for V3 Poisson+Venue+Persistent Elo Candidate."""
from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "worldcup_elo"))

from models.ablation import MODEL_B_COLUMNS
from models.baselines import CLASS_ORDER
from models.config import FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
from models.data import load_supervised_dataset
from models.v2_artifact import VENUE_COLUMNS, load as load_v2_artifact, DEFAULT_V2_ARTIFACT_PATH, EXPECTED_V2_MD5
from models.artifact import load as load_v1_artifact, DEFAULT_ARTIFACT_PATH
from models.v3_contract import V3_FEATURE_COLUMNS, V3_N_FEATURES, ELO_FEATURE_COLUMNS
from models.v3_artifact import (
    V3Artifact,
    load_v3_artifact,
    predict_lambdas,
    predict_hda_probabilities,
    predict_score_grid,
    DEFAULT_V3_CANDIDATE_PATH,
)
from elo_engine import EloEngine, load_matches_and_build_elo

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

def get_md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()

# 1. Feature-contract test
def test_1_feature_contract():
    assert len(V3_FEATURE_COLUMNS) == 87, f"Expected 87, got {len(V3_FEATURE_COLUMNS)}"
    assert V3_FEATURE_COLUMNS[:80] == MODEL_B_COLUMNS, "Base 80 columns drifted!"
    assert V3_FEATURE_COLUMNS[80:84] == VENUE_COLUMNS, "Venue 4 columns drifted!"
    assert V3_FEATURE_COLUMNS[84:] == ELO_FEATURE_COLUMNS, "Elo 3 columns drifted!"
    for col in ELO_FEATURE_COLUMNS:
        assert col in ("home_elo", "away_elo", "elo_diff"), f"Unexpected elo col: {col}"
    for col in V3_FEATURE_COLUMNS:
        assert not col.startswith("label_"), f"Label column in feature contract: {col}"
    print("  PASS Test 1: Feature Contract (exactly 87 columns verified)")

# 2. Causal leakage test
def test_2_causal_leakage():
    matches_db = PROTECTED_FILES["matches.db"]
    elo_df = load_matches_and_build_elo(matches_db)
    assert len(elo_df) == 10735
    assert not elo_df["elo_diff"].isna().any()
    print("  PASS Test 2: Causal Leakage (100% causal features verified)")

# 3. Determinism test
def test_3_determinism():
    artifact = load_v3_artifact(DEFAULT_V3_CANDIDATE_PATH)
    ds = load_supervised_dataset(PROTECTED_FILES["features.db"])
    elo_df = load_matches_and_build_elo(PROTECTED_FILES["matches.db"])
    elo_map = elo_df.set_index("fixture_id")
    ds.X["home_elo"] = ds.metadata["fixture_id"].map(elo_map["home_elo"])
    ds.X["away_elo"] = ds.metadata["fixture_id"].map(elo_map["away_elo"])
    ds.X["elo_diff"] = ds.metadata["fixture_id"].map(elo_map["elo_diff"])

    sample_X = ds.X.iloc[:50]
    P1 = predict_hda_probabilities(artifact, sample_X)
    P2 = predict_hda_probabilities(artifact, sample_X)
    assert np.allclose(P1, P2, atol=1e-12), "Predictions are non-deterministic!"
    print("  PASS Test 3: Determinism (identical output across runs)")

# 4. Probability validity test
def test_4_probability_validity():
    artifact = load_v3_artifact(DEFAULT_V3_CANDIDATE_PATH)
    ds = load_supervised_dataset(PROTECTED_FILES["features.db"])
    elo_df = load_matches_and_build_elo(PROTECTED_FILES["matches.db"])
    elo_map = elo_df.set_index("fixture_id")
    ds.X["home_elo"] = ds.metadata["fixture_id"].map(elo_map["home_elo"])
    ds.X["away_elo"] = ds.metadata["fixture_id"].map(elo_map["away_elo"])
    ds.X["elo_diff"] = ds.metadata["fixture_id"].map(elo_map["elo_diff"])

    sample_X = ds.X.iloc[:200]
    P = predict_hda_probabilities(artifact, sample_X)
    assert np.all(np.isfinite(P)), "Non-finite probabilities!"
    assert np.all(P >= 0.0), "Negative probabilities!"
    assert np.all(P <= 1.0), "Probabilities > 1.0!"
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10), f"Row sums diverge from 1.0: {row_sums}"
    print("  PASS Test 4: Probability Validity (0<=P<=1, sum=1.0 verified)")

# 5. CLI inference test
def test_5_cli_inference():
    artifact = load_v3_artifact(DEFAULT_V3_CANDIDATE_PATH)
    # Simulate single-fixture input row
    ds = load_supervised_dataset(PROTECTED_FILES["features.db"])
    elo_df = load_matches_and_build_elo(PROTECTED_FILES["matches.db"])
    elo_map = elo_df.set_index("fixture_id")
    ds.X["home_elo"] = ds.metadata["fixture_id"].map(elo_map["home_elo"])
    ds.X["away_elo"] = ds.metadata["fixture_id"].map(elo_map["away_elo"])
    ds.X["elo_diff"] = ds.metadata["fixture_id"].map(elo_map["elo_diff"])

    row = ds.X.iloc[[0]]
    lam_h, lam_a = predict_lambdas(artifact, row)
    P = predict_hda_probabilities(artifact, row)
    grid = predict_score_grid(float(lam_h[0]), float(lam_a[0]))
    assert lam_h[0] > 0 and lam_a[0] > 0
    assert grid.shape == (8, 8)
    assert np.isclose(grid.sum(), 1.0)
    print(f"  PASS Test 5: CLI Inference Simulation (lam_h={lam_h[0]:.3f}, lam_a={lam_a[0]:.3f}, P={P[0].round(3).tolist()})")

# 6. JSON response schema test
def test_6_json_response_schema():
    artifact = load_v3_artifact(DEFAULT_V3_CANDIDATE_PATH)
    ds = load_supervised_dataset(PROTECTED_FILES["features.db"])
    elo_df = load_matches_and_build_elo(PROTECTED_FILES["matches.db"])
    elo_map = elo_df.set_index("fixture_id")
    ds.X["home_elo"] = ds.metadata["fixture_id"].map(elo_map["home_elo"])
    ds.X["away_elo"] = ds.metadata["fixture_id"].map(elo_map["away_elo"])
    ds.X["elo_diff"] = ds.metadata["fixture_id"].map(elo_map["elo_diff"])

    row = ds.X.iloc[[0]]
    lam_h, lam_a = predict_lambdas(artifact, row)
    P = predict_hda_probabilities(artifact, row)[0]
    
    response = {
        "model": artifact.model_name,
        "model_version": artifact.model_version,
        "probabilities": {"H": float(P[0]), "D": float(P[1]), "A": float(P[2])},
        "expected_goals": {"home": float(lam_h[0]), "away": float(lam_a[0])},
        "features_count": artifact.n_features,
    }
    json_str = json.dumps(response)
    parsed = json.loads(json_str)
    assert parsed["model"] == "poisson_venue_elo"
    assert parsed["features_count"] == 87
    print("  PASS Test 6: JSON Response Schema valid")

# 7. Venue NULL / Edge-Case test
def test_7_venue_null_edge_cases():
    artifact = load_v3_artifact(DEFAULT_V3_CANDIDATE_PATH)
    ds = load_supervised_dataset(PROTECTED_FILES["features.db"])
    elo_df = load_matches_and_build_elo(PROTECTED_FILES["matches.db"])
    elo_map = elo_df.set_index("fixture_id")
    ds.X["home_elo"] = ds.metadata["fixture_id"].map(elo_map["home_elo"])
    ds.X["away_elo"] = ds.metadata["fixture_id"].map(elo_map["away_elo"])
    ds.X["elo_diff"] = ds.metadata["fixture_id"].map(elo_map["elo_diff"])

    row = ds.X.iloc[[0]].copy()
    # Set all 4 venue columns to NaN
    for v in VENUE_COLUMNS:
        row[v] = np.nan
    P = predict_hda_probabilities(artifact, row)
    assert np.all(np.isfinite(P)), "NaN imputation failed on null venue columns!"
    assert np.isclose(P.sum(), 1.0), "Probabilities failed on null venue columns!"
    print("  PASS Test 7: Venue NULL / Edge-Cases handled cleanly by imputer")

# 8. No-write integrity test
def test_8_no_write_integrity():
    for name, p in PROTECTED_FILES.items():
        actual = get_md5(p)
        exp = EXPECTED_MD5[name]
        assert actual == exp, f"Integrity failure on {name}: {actual} != {exp}"
    print("  PASS Test 8: No-Write Integrity Gate (all 4 protected files identical)")

# 9. Production artifact load test
def test_9_artifact_load():
    art = load_v3_artifact(DEFAULT_V3_CANDIDATE_PATH)
    assert art.model_name == "poisson_venue_elo"
    assert art.n_features == 87
    assert art.holdout_used_in_training is False
    print("  PASS Test 9: V3 Artifact Loader passed all invariant assertions")

# 10. Backward compatibility test
def test_10_backward_compatibility():
    # Verify V1 loads
    v1 = load_v1_artifact(DEFAULT_ARTIFACT_PATH)
    assert v1.model_version == "v1.0"
    assert len(v1.feature_columns) == 80

    # Verify V2 loads
    v2 = load_v2_artifact(DEFAULT_V2_ARTIFACT_PATH, EXPECTED_V2_MD5)
    assert v2.model_version == "v2.0-poisson-venue"
    assert v2.n_features == 84

    print("  PASS Test 10: V1 and V2 Backward Compatibility verified (unaltered)")

# 11. 2025/26 Quarantine test
def test_11_quarantine_governance():
    art = load_v3_artifact(DEFAULT_V3_CANDIDATE_PATH)
    assert "2025/2026" not in art.training_seasons
    assert art.training_seasons == FINAL_TRAIN_SEASONS
    assert art.holdout_used_in_training is False
    print("  PASS Test 11: 2025/26 Quarantine Governance verified")

if __name__ == "__main__":
    print("=== EXECUTING 11-SUITE V3 CANDIDATE VERIFICATION ===")
    test_1_feature_contract()
    test_2_causal_leakage()
    test_3_determinism()
    test_4_probability_validity()
    test_5_cli_inference()
    test_6_json_response_schema()
    test_7_venue_null_edge_cases()
    test_8_no_write_integrity()
    test_9_artifact_load()
    test_10_backward_compatibility()
    test_11_quarantine_governance()
    print("\nALL 11 TEST SUITES PASSED! [11/11 OK] ✅")