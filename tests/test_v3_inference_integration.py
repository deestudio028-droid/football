"""V3 Inference Integration Tests.

Tests the production inference pipeline, NOT the model itself.
Covers: feature contract, causal Elo invariants, production-path Elo
equivalence, V2/V3 side-by-side inference, determinism, and integrity.

Run with: python tests/test_v3_inference_integration.py
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "worldcup_elo"))

FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"

# Protected file checksums
EXPECTED_MD5 = {
    "v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "matches.db": "fdeed042096fa1c851aaee6c84995247",
}

PROTECTED_FILES = {
    "v2_poisson_venue.pkl": PROJECT_ROOT / "data" / "models" / "v2_poisson_venue.pkl",
    "v1_logreg.pkl": PROJECT_ROOT / "data" / "models" / "v1_logreg.pkl",
    "features.db": FEATURES_DB,
    "matches.db": MATCHES_DB,
}

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} -- {detail}")
        failed += 1


def get_md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ======================================================================
# SUITE 1: Feature Contract
# ======================================================================
def test_feature_contract():
    print("\n=== SUITE 1: Feature Contract ===")
    from models.ablation import MODEL_B_COLUMNS
    from models.v2_artifact import VENUE_COLUMNS
    from models.v3_contract import V3_FEATURE_COLUMNS, V3_N_FEATURES, ELO_FEATURE_COLUMNS
    from features.elo import ELO_COLUMNS

    check("V2 has exactly 84 features",
          len(MODEL_B_COLUMNS) + len(VENUE_COLUMNS) == 84,
          f"got {len(MODEL_B_COLUMNS) + len(VENUE_COLUMNS)}")

    check("V3 has exactly 87 features",
          V3_N_FEATURES == 87,
          f"got {V3_N_FEATURES}")

    check("V3 first 80 columns == MODEL_B_COLUMNS",
          V3_FEATURE_COLUMNS[:80] == MODEL_B_COLUMNS)

    check("V3 columns 80-83 == VENUE_COLUMNS",
          V3_FEATURE_COLUMNS[80:84] == VENUE_COLUMNS)

    check("V3 columns 84-86 == ELO_FEATURE_COLUMNS",
          V3_FEATURE_COLUMNS[84:] == ELO_FEATURE_COLUMNS)

    check("ELO_FEATURE_COLUMNS are exactly (home_elo, away_elo, elo_diff)",
          ELO_FEATURE_COLUMNS == ("home_elo", "away_elo", "elo_diff"))

    check("Production ELO_COLUMNS match contract ELO_FEATURE_COLUMNS",
          ELO_COLUMNS == ELO_FEATURE_COLUMNS)

    check("No label columns in V3 contract",
          not any(c.startswith("label_") for c in V3_FEATURE_COLUMNS))

    check("No duplicate columns in V3 contract",
          len(V3_FEATURE_COLUMNS) == len(set(V3_FEATURE_COLUMNS)))

    # Verify V3 is a strict superset of V2
    v2_cols = MODEL_B_COLUMNS + VENUE_COLUMNS
    check("V3[:84] == V2 contract",
          V3_FEATURE_COLUMNS[:84] == v2_cols)


# ======================================================================
# SUITE 2: Causal Elo Mechanical Tests
# ======================================================================
def test_causal_elo():
    print("\n=== SUITE 2: Causal Elo Invariants ===")
    from features.elo import compute_elo_features, INIT_RATING

    # Load fixtures
    uri = f"file:{MATCHES_DB.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        fixtures = pd.read_sql_query(
            "SELECT fixture_id, unix, home_id, away_id, "
            "home_goals, away_goals, status "
            "FROM fixtures ORDER BY unix ASC, fixture_id ASC",
            conn,
        )
    finally:
        conn.close()

    full_elo = compute_elo_features(fixtures)

    # Test 1: Current fixture outcome cannot affect its own Elo
    # Modify fixture 250's goals to 10-0, verify its pre-match Elo unchanged
    modified = fixtures.copy()
    idx = 250
    modified.loc[idx, "home_goals"] = 10
    modified.loc[idx, "away_goals"] = 0
    modified_elo = compute_elo_features(modified)

    fid_250 = fixtures.loc[idx, "fixture_id"]
    orig_row = full_elo.loc[full_elo["fixture_id"] == fid_250].iloc[0]
    mod_row = modified_elo.loc[modified_elo["fixture_id"] == fid_250].iloc[0]

    check("Outcome insulation: modifying fixture 250 goals doesn't change its Elo",
          abs(orig_row["home_elo"] - mod_row["home_elo"]) < 1e-10
          and abs(orig_row["away_elo"] - mod_row["away_elo"]) < 1e-10
          and abs(orig_row["elo_diff"] - mod_row["elo_diff"]) < 1e-10)

    # Test 2: Adding a future fixture cannot change earlier Elos (truncation invariance)
    for n in [1000, 3000, 5000]:
        trunc = fixtures.iloc[:n].copy()
        trunc_elo = compute_elo_features(trunc)
        merged = full_elo.iloc[:n].merge(trunc_elo, on="fixture_id", suffixes=("_full", "_trunc"))
        max_diff = max(
            (merged["home_elo_full"] - merged["home_elo_trunc"]).abs().max(),
            (merged["away_elo_full"] - merged["away_elo_trunc"]).abs().max(),
            (merged["elo_diff_full"] - merged["elo_diff_trunc"]).abs().max(),
        )
        check(f"Truncation invariance at N={n}: max_diff={max_diff:.2e}",
              max_diff < 1e-10)

    # Test 3: Removing current fixture outcome doesn't change pre-match Elo
    removed = fixtures.copy()
    removed.loc[idx, "home_goals"] = np.nan
    removed.loc[idx, "away_goals"] = np.nan
    removed.loc[idx, "status"] = "NS"  # not started
    removed_elo = compute_elo_features(removed)
    rem_row = removed_elo.loc[removed_elo["fixture_id"] == fid_250].iloc[0]
    check("Outcome removal: setting fixture 250 to NS doesn't change its Elo",
          abs(orig_row["home_elo"] - rem_row["home_elo"]) < 1e-10
          and abs(orig_row["away_elo"] - rem_row["away_elo"]) < 1e-10)

    # Test 4: Determinism
    elo_run2 = compute_elo_features(fixtures)
    max_det_diff = max(
        (full_elo["home_elo"] - elo_run2["home_elo"]).abs().max(),
        (full_elo["away_elo"] - elo_run2["away_elo"]).abs().max(),
        (full_elo["elo_diff"] - elo_run2["elo_diff"]).abs().max(),
    )
    check(f"Determinism: two runs produce max_diff={max_det_diff:.2e}",
          max_det_diff < 1e-12)

    # Test 5: All Elo values are finite
    check("All home_elo finite",
          np.all(np.isfinite(full_elo["home_elo"])))
    check("All away_elo finite",
          np.all(np.isfinite(full_elo["away_elo"])))
    check("All elo_diff finite",
          np.all(np.isfinite(full_elo["elo_diff"])))

    # Test 6: Elo values within reasonable bounds (700-2300 is generous)
    min_elo = min(full_elo["home_elo"].min(), full_elo["away_elo"].min())
    max_elo = max(full_elo["home_elo"].max(), full_elo["away_elo"].max())
    check(f"Elo bounds: [{min_elo:.1f}, {max_elo:.1f}] within [700, 2300]",
          min_elo >= 700 and max_elo <= 2300,
          f"range [{min_elo:.1f}, {max_elo:.1f}]")

    # Test 7: Cross-season persistence
    # Get a team that appears in multiple seasons, verify Elo carries forward
    conn = sqlite3.connect(f"file:{MATCHES_DB.resolve()}?mode=ro", uri=True)
    try:
        seasons = pd.read_sql_query(
            "SELECT DISTINCT season_id FROM fixtures ORDER BY season_id", conn)
    finally:
        conn.close()
    # Elo should differ from INIT_RATING for experienced teams
    last_elo = full_elo.iloc[-100:]
    non_init = last_elo[(last_elo["home_elo"] != INIT_RATING) |
                        (last_elo["away_elo"] != INIT_RATING)]
    check("Cross-season persistence: recent fixtures have non-initial Elo",
          len(non_init) > 90,
          f"only {len(non_init)}/100 have non-initial Elo")

    # Test 8: Teams absent from history receive initial rating
    check(f"Initial rating is {INIT_RATING}",
          INIT_RATING == 1500.0)
    # First fixture's teams should start at INIT_RATING
    first_row = full_elo.iloc[0]
    check("First fixture: home_elo == INIT_RATING",
          abs(first_row["home_elo"] - INIT_RATING) < 1e-10)
    check("First fixture: away_elo == INIT_RATING",
          abs(first_row["away_elo"] - INIT_RATING) < 1e-10)


# ======================================================================
# SUITE 3: Production vs Research Elo Equivalence
# ======================================================================
def test_elo_equivalence():
    print("\n=== SUITE 3: Production Elo == Research Elo ===")
    from features.elo import load_elo_features
    from elo_engine import load_matches_and_build_elo

    prod_elo = load_elo_features(MATCHES_DB)
    research_elo = load_matches_and_build_elo(MATCHES_DB)

    # Join on fixture_id
    merged = prod_elo.merge(
        research_elo[["fixture_id", "home_elo", "away_elo", "elo_diff"]],
        on="fixture_id", suffixes=("_prod", "_research"),
    )

    check(f"Same fixture count: prod={len(prod_elo)}, research={len(research_elo)}",
          len(prod_elo) == len(research_elo))

    max_home = (merged["home_elo_prod"] - merged["home_elo_research"]).abs().max()
    max_away = (merged["away_elo_prod"] - merged["away_elo_research"]).abs().max()
    max_diff = (merged["elo_diff_prod"] - merged["elo_diff_research"]).abs().max()

    check(f"home_elo max diff: {max_home:.2e}",
          max_home < 1e-10)
    check(f"away_elo max diff: {max_away:.2e}",
          max_away < 1e-10)
    check(f"elo_diff max diff: {max_diff:.2e}",
          max_diff < 1e-10)


# ======================================================================
# SUITE 4: V2/V3 Side-by-Side Inference
# ======================================================================
def test_inference():
    print("\n=== SUITE 4: V2/V3 Side-by-Side Inference ===")
    from models import v2_artifact as v2_mod
    from models.v3_artifact import load_v3_artifact, DEFAULT_V3_CANDIDATE_PATH
    from models.v3_contract import V3_FEATURE_COLUMNS
    from models.poisson import predict_poisson
    from models.data import load_supervised_dataset
    from features.elo import load_elo_features, ELO_COLUMNS

    # Load V2
    v2_path = PROJECT_ROOT / v2_mod.DEFAULT_V2_ARTIFACT_PATH
    v2_art = v2_mod.load(v2_path)

    # Load V3
    v3_prod = PROJECT_ROOT / "data" / "models" / "v3_poisson_venue_elo.pkl"
    v3_cand = PROJECT_ROOT / DEFAULT_V3_CANDIDATE_PATH
    v3_path = v3_prod if v3_prod.exists() else v3_cand
    v3_art = load_v3_artifact(v3_path)

    # Load features
    ds = load_supervised_dataset(FEATURES_DB)
    elo_df = load_elo_features(MATCHES_DB)
    elo_map = elo_df.set_index("fixture_id")

    # Pick 5 fixtures from validation seasons (2022/23, 2023/24, 2024/25)
    from models.config import SEASON_NAME_TO_IDS
    val_ids = []
    for s in ["2022/2023", "2023/2024", "2024/2025"]:
        val_ids.extend(SEASON_NAME_TO_IDS[s])
    val_mask = ds.metadata["season_id"].isin(val_ids)
    val_fixtures = ds.metadata.loc[val_mask, "fixture_id"].values[:5]

    check(f"Found {len(val_fixtures)} test fixtures",
          len(val_fixtures) >= 5)

    for fid in val_fixtures:
        mask = (ds.metadata["fixture_id"] == fid).values

        # V2 prediction
        X_v2 = ds.X[mask][list(v2_art.feature_columns)]
        X_v2_enc = v2_art.preprocessor.transform(X_v2)
        lam_h_v2 = v2_art.model_home_goals.predict(X_v2_enc)
        lam_a_v2 = v2_art.model_away_goals.predict(X_v2_enc)
        preds_v2 = predict_poisson(lam_h_v2, lam_a_v2, v2_art.class_order)

        # V3 prediction (via production path: base cols + Elo join)
        base_cols = [c for c in V3_FEATURE_COLUMNS if c not in ELO_COLUMNS]
        X_v3 = ds.X[mask][base_cols].copy()
        for col in ELO_COLUMNS:
            X_v3[col] = elo_map.loc[fid, col]

        check(f"Fixture {fid}: V3 has 87 columns",
              len(X_v3.columns) == 87)
        check(f"Fixture {fid}: V3 column order matches contract",
              list(X_v3.columns) == list(V3_FEATURE_COLUMNS))

        X_v3_enc = v3_art.preprocessor.transform(X_v3)
        lam_h_v3 = v3_art.model_home_goals.predict(X_v3_enc)
        lam_a_v3 = v3_art.model_away_goals.predict(X_v3_enc)
        preds_v3 = predict_poisson(lam_h_v3, lam_a_v3, v3_art.class_order)

        p_v2 = preds_v2[0]
        p_v3 = preds_v3[0]

        # V2 still works
        check(f"Fixture {fid}: V2 lambda_home > 0", p_v2.lambda_home > 0)
        check(f"Fixture {fid}: V2 lambda_away > 0", p_v2.lambda_away > 0)
        v2_sum = sum(p_v2.probabilities.values())
        check(f"Fixture {fid}: V2 probs sum to 1 (got {v2_sum:.10f})",
              abs(v2_sum - 1.0) < 1e-8)

        # V3 works
        check(f"Fixture {fid}: V3 lambda_home > 0", p_v3.lambda_home > 0)
        check(f"Fixture {fid}: V3 lambda_away > 0", p_v3.lambda_away > 0)
        v3_sum = sum(p_v3.probabilities.values())
        check(f"Fixture {fid}: V3 probs sum to 1 (got {v3_sum:.10f})",
              abs(v3_sum - 1.0) < 1e-8)

        # All probabilities finite and in [0, 1]
        for cls in ["H", "D", "A"]:
            check(f"Fixture {fid}: V3 P({cls}) in [0,1]",
                  0 <= p_v3.probabilities[cls] <= 1)

    # Determinism: run V3 twice on same fixture
    fid0 = val_fixtures[0]
    mask0 = (ds.metadata["fixture_id"] == fid0).values
    X_v3_a = ds.X[mask0][base_cols].copy()
    X_v3_b = ds.X[mask0][base_cols].copy()
    for col in ELO_COLUMNS:
        X_v3_a[col] = elo_map.loc[fid0, col]
        X_v3_b[col] = elo_map.loc[fid0, col]
    enc_a = v3_art.preprocessor.transform(X_v3_a)
    enc_b = v3_art.preprocessor.transform(X_v3_b)
    lam_ha = v3_art.model_home_goals.predict(enc_a)
    lam_hb = v3_art.model_home_goals.predict(enc_b)
    check("V3 determinism: identical lambdas across runs",
          np.allclose(lam_ha, lam_hb, atol=1e-12))


# ======================================================================
# SUITE 5: Integrity Gate
# ======================================================================
def test_integrity():
    print("\n=== SUITE 5: Protected File Integrity ===")
    for name, path in PROTECTED_FILES.items():
        actual = get_md5(path)
        expected = EXPECTED_MD5[name]
        check(f"{name}: {actual}",
              actual == expected,
              f"expected {expected}")


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("V3 INFERENCE INTEGRATION TEST SUITE")
    print("=" * 60)

    test_feature_contract()
    test_causal_elo()
    test_elo_equivalence()
    test_inference()
    test_integrity()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        print("INFERENCE INTEGRATION: FAIL")
        sys.exit(1)
    else:
        print("INFERENCE INTEGRATION: ALL TESTS PASSED")
        sys.exit(0)
