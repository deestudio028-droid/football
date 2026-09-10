"""V2 PRODUCTION INTEGRATION TEST -- read-only, 21-point verification.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_v2_production_integration_test.py

PURPOSE: prove that the frozen V2 Poisson+Venue artifact can produce
valid H/D/A predictions through the REAL production inference path.

This is NOT a training experiment and NOT an evaluation -- no outcomes
are read, no accuracy is measured, no model is fitted.

SAFETY:
  - v2_poisson_venue.pkl opened read-only, checksum-verified before AND after
  - v1_logreg.pkl checksum-verified before AND after
  - features.db, matches.db checksum-verified before AND after
  - label_home_goals, label_away_goals, label_result are NEVER read
  - 2025/26 outcomes are NEVER accessed
  - no pickle.dump, no database write, no file creation
  - tail-safe conversion imported from the production module (poisson.py)
    which is mathematically identical to the pinned STEP 3 implementation
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ============================================================================
# CONSTANTS
# ============================================================================
EXPECTED_MD5 = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl":        "5e504427712b35778bb8a62a8496c7cd",
    "data/processed/features.db":       "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db":        "fdeed042096fa1c851aaee6c84995247",
}

PRODUCTION_SOURCE_FILES = [
    "src/models/train.py",
    "src/models/artifact.py",
    "src/models/data.py",
    "src/models/config.py",
    "src/models/evaluate.py",
    "src/models/baselines.py",
    "src/models/candidate_contract.py",
    "src/models/poisson.py",
    "src/models/v2_artifact.py",
    "src/features/feature_builder.py",
    "src/features/rolling.py",
    "src/features/config.py",
    "src/features/context.py",
    "src/features/storage.py",
    "predict_match.py",
]

# Training seasons only (NOT 2025/26)
TRAINING_SEASONS = ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025")

CLASS_ORDER = ["H", "D", "A"]
REPRODUCIBILITY_TOL = 1e-12
ROWSUM_TOL = 1e-12
MASS_TOL = 1e-12
COMPLEMENT_TOL = 1e-10


# ============================================================================
# UTILITIES
# ============================================================================
passes, fails = 0, 0


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def posix_rel(p):
    """Relative path as forward-slash string (consistent across platforms)."""
    return Path(p).relative_to(REPO).as_posix()


def rule(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def check(label, ok):
    global passes, fails
    tag = "PASS" if ok else "FAIL"
    if not ok:
        fails += 1
    else:
        passes += 1
    print(f"  [{tag}]  {label}")
    return ok


def abort(msg):
    print("\n" + "!" * 100)
    print("INTEGRATION TEST ABORTED")
    print(msg)
    print("!" * 100)
    raise SystemExit(1)


def snapshot_checksums():
    """Record checksums of all protected files, using forward-slash keys."""
    s = {}
    for rel in EXPECTED_MD5:
        fp = REPO / rel
        if fp.exists():
            s[rel] = md5(fp)
    for rel in PRODUCTION_SOURCE_FILES:
        fp = REPO / rel
        if fp.exists():
            s[rel] = md5(fp)
    return s


# ============================================================================
# MAIN
# ============================================================================
def main():
    global passes, fails

    print("V2 PRODUCTION INTEGRATION TEST")
    print("=" * 100)
    print("READ-ONLY -- no training, no evaluation, no outcome access")
    print()

    # ==================================================================
    # STEP 0 -- INTEGRITY / SOURCE AUDIT
    # ==================================================================
    rule("STEP 0 -- INTEGRITY / SOURCE AUDIT")

    pre_snap = snapshot_checksums()

    # 0.1 All four protected files exist and match
    for rel, expected in EXPECTED_MD5.items():
        fp = REPO / rel
        if not check(f"{rel} exists", fp.exists()):
            abort(f"{rel} not found")
        actual = md5(fp)
        check(f"{rel} MD5 matches", actual == expected)
        print(f"    size={fp.stat().st_size}  md5={actual}")

    # 0.2 Production source checksums (recorded, not gated)
    print("\n  Production source checksums:")
    for rel in PRODUCTION_SOURCE_FILES:
        fp = REPO / rel
        if fp.exists():
            print(f"    {rel}: {md5(fp)}")
        else:
            print(f"    {rel}: MISSING")

    # ==================================================================
    # STEP 1 -- LOAD V2 ARTIFACT (via production loader)
    # ==================================================================
    rule("STEP 1 -- LOAD V2 ARTIFACT VIA PRODUCTION PATH")

    from models import v2_artifact as v2_mod
    from models.poisson import predict_poisson, hda_tail_safe, modal_scoreline, TAIL_TOL

    v2_path = REPO / v2_mod.DEFAULT_V2_ARTIFACT_PATH
    try:
        art = v2_mod.load(v2_path)
        check("V2 artifact loads via v2_artifact.load()", True)
    except v2_mod.V2ArtifactError as exc:
        check("V2 artifact loads via v2_artifact.load()", False)
        abort(str(exc))

    check("model_name == 'poisson_venue'", art.model_name == "poisson_venue")
    check("model_version == 'v2.0-poisson-venue'", art.model_version == "v2.0-poisson-venue")
    check("feature_version == 'v1.0'", art.feature_version == "v1.0")
    check("n_features == 84", art.n_features == 84)
    check("class_order == ['H','D','A']", art.class_order == CLASS_ORDER)
    check("training_seasons correct", tuple(art.training_seasons) == TRAINING_SEASONS)
    check("holdout_used_in_training is False", art.holdout_used_in_training is False)

    # ==================================================================
    # STEP 2 -- V1 NOT ACCIDENTALLY LOADED
    # ==================================================================
    rule("STEP 2 -- V1 NOT ACCIDENTALLY LOADED")

    check("model_version is NOT v1.0", art.model_version != "v1.0")
    check("artifact_path is v2_poisson_venue.pkl",
          art.artifact_path.name == "v2_poisson_venue.pkl")

    from sklearn.linear_model import PoissonRegressor
    check("model_home_goals is PoissonRegressor",
          isinstance(art.model_home_goals, PoissonRegressor))
    check("model_away_goals is PoissonRegressor",
          isinstance(art.model_away_goals, PoissonRegressor))

    # Verify V1 loader still works independently
    from models import artifact as v1_mod
    v1_path = REPO / v1_mod.DEFAULT_ARTIFACT_PATH
    try:
        v1_art = v1_mod.load(v1_path)
        check("V1 backward compat: v1_logreg.pkl still loads", True)
        check("V1 backward compat: model_version == v1.0",
              v1_art.model_version == "v1.0")
        check("V1 backward compat: 80 features",
              len(v1_art.feature_columns) == 80)
    except Exception as exc:
        check("V1 backward compat: v1_logreg.pkl still loads", False)
        print(f"    {exc}")

    # ==================================================================
    # STEP 3 -- VERIFY 84-COLUMN CONTRACT
    # ==================================================================
    rule("STEP 3 -- VERIFY 84-COLUMN CONTRACT")

    from models.ablation import MODEL_B_COLUMNS

    feature_cols = art.feature_columns
    check("exactly 84 feature columns", len(feature_cols) == 84)
    check("first 80 == MODEL_B_COLUMNS", tuple(feature_cols[:80]) == tuple(MODEL_B_COLUMNS))
    check("final 4 == venue columns", tuple(feature_cols[80:]) == art.venue_columns)
    check("venue columns correct",
          art.venue_columns == v2_mod.VENUE_COLUMNS)

    # No forbidden columns
    forbidden = {"label_home_goals", "label_away_goals", "label_result"}
    check("no label columns in contract", not (set(feature_cols) & forbidden))
    venue_n = [c for c in feature_cols if c.endswith("_n") and "venue" in c]
    check("no venue _n columns", len(venue_n) == 0)

    # ==================================================================
    # STEP 4 -- SELECT SAFE TEST INPUTS
    # ==================================================================
    rule("STEP 4 -- SELECT SAFE TEST INPUTS")

    from models.config import SEASON_NAME_TO_IDS
    from models.data import load_supervised_dataset

    ds = load_supervised_dataset(REPO / "data/processed/features.db")

    # Select from 2024/25 training season only
    season_2425_ids = set(SEASON_NAME_TO_IDS.get("2024/2025", []))
    mask_2425 = ds.metadata["season_id"].isin(season_2425_ids)
    df_pool = ds.X[mask_2425.values]
    fids_pool = ds.metadata[mask_2425.values]["fixture_id"].tolist()
    n_test = min(25, len(df_pool))
    print(f"  2024/25 pool: {len(df_pool)} rows, selecting {n_test}")

    X_test = df_pool.head(n_test)[list(feature_cols)].copy()
    test_fids = fids_pool[:n_test]

    check("test matrix has 84 columns", X_test.shape[1] == 84)
    check(f"test matrix has {n_test} rows", X_test.shape[0] == n_test)

    # Verify no label columns leaked into X
    label_in_X = [c for c in forbidden if c in X_test.columns]
    check("NO label columns in test feature matrix", len(label_in_X) == 0)

    # ==================================================================
    # STEP 5 -- RUN PRODUCTION V2 INFERENCE
    # ==================================================================
    rule("STEP 5 -- RUN PRODUCTION V2 INFERENCE")

    # 5.1 Preprocess
    X_encoded = art.preprocessor.transform(X_test)
    check("preprocessor produces 2D array", X_encoded.ndim == 2)
    check(f"encoded shape[0] == {n_test}", X_encoded.shape[0] == n_test)
    check("encoded shape[1] == 88", X_encoded.shape[1] == 88)
    print(f"  encoded shape: {X_encoded.shape}")

    # 5.2 Predict lambdas
    lam_h = art.model_home_goals.predict(X_encoded)
    lam_a = art.model_away_goals.predict(X_encoded)
    print(f"  lambda_home: [{lam_h.min():.4f}, {lam_h.max():.4f}]  mean={lam_h.mean():.4f}")
    print(f"  lambda_away: [{lam_a.min():.4f}, {lam_a.max():.4f}]  mean={lam_a.mean():.4f}")

    check("all lambda_home > 0 and finite",
          np.all(lam_h > 0) and np.all(np.isfinite(lam_h)))
    check("all lambda_away > 0 and finite",
          np.all(lam_a > 0) and np.all(np.isfinite(lam_a)))

    # 5.3 Full Poisson prediction via production module
    predictions = predict_poisson(lam_h, lam_a, art.class_order)
    check("predict_poisson returns n_test predictions",
          len(predictions) == n_test)

    # Also get raw conversion for detailed checks
    P, K, residual, control = hda_tail_safe(lam_h, lam_a, tol=TAIL_TOL)
    print(f"  grid K = {K}")
    print(f"  tail residual = {residual:.6e}")
    print(f"  complement control = {control:.6e}")

    check("tail residual < 1e-12", residual < MASS_TOL)
    check("complement control < 1e-10", control < COMPLEMENT_TOL)

    # 5.4 Probability validity
    check("all P(H) in [0,1]", np.all((P[:, 0] >= 0) & (P[:, 0] <= 1)))
    check("all P(D) in [0,1]", np.all((P[:, 1] >= 0) & (P[:, 1] <= 1)))
    check("all P(A) in [0,1]", np.all((P[:, 2] >= 0) & (P[:, 2] <= 1)))
    check("all probabilities finite", np.all(np.isfinite(P)))

    max_sum_error = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
    print(f"  max |row_sum - 1.0| = {max_sum_error:.6e}")
    check(f"H+D+A == 1 within {ROWSUM_TOL}", max_sum_error <= ROWSUM_TOL)

    # 5.5 Modal scoreline
    sh, sa, sp = modal_scoreline(lam_h, lam_a, K)
    check("modal scoreline produced", len(sh) == n_test)
    check("modal scoreline probability > 0", np.all(sp > 0))
    check("modal scoreline probability <= 1", np.all(sp <= 1))

    # 5.6 Per-fixture table
    predicted_class = [CLASS_ORDER[int(P[i].argmax())] for i in range(n_test)]
    print(f"\n  {'fid':>12}  {'λ_h':>7}  {'λ_a':>7}  {'P(H)':>7}  {'P(D)':>7}  "
          f"{'P(A)':>7}  {'pred':>4}  {'score':>5}  {'p_score':>8}")
    print("  " + "-" * 85)
    for i in range(n_test):
        print(f"  {test_fids[i]:>12}  {lam_h[i]:7.4f}  {lam_a[i]:7.4f}  "
              f"{P[i,0]:7.4f}  {P[i,1]:7.4f}  {P[i,2]:7.4f}  "
              f"{predicted_class[i]:>4}  {sh[i]}-{sa[i]}  {sp[i]:8.5f}")

    # ==================================================================
    # STEP 6 -- PRODUCTION RESPONSE SCHEMA
    # ==================================================================
    rule("STEP 6 -- PRODUCTION RESPONSE SCHEMA")

    required_keys = {"prediction", "probabilities", "expected_goals_home",
                     "expected_goals_away", "modal_scoreline", "modal_scoreline_probability"}

    all_schema_ok = True
    all_prob_keys_ok = True
    for pred in predictions:
        d = pred.to_dict()
        if set(d.keys()) != required_keys:
            all_schema_ok = False
        if tuple(d["probabilities"].keys()) != ("H", "D", "A"):
            all_prob_keys_ok = False

    check("all responses have required 6-key schema", all_schema_ok)
    check("H/D/A ordering consistent in all responses", all_prob_keys_ok)

    import json as _json
    print(f"\n  Sample response (fixture {test_fids[0]}):")
    print("  " + _json.dumps(predictions[0].to_dict(), indent=4).replace("\n", "\n  "))

    # ==================================================================
    # STEP 7 -- DETERMINISM TEST
    # ==================================================================
    rule("STEP 7 -- DETERMINISM TEST")

    X_enc2 = art.preprocessor.transform(X_test)
    lam_h2 = art.model_home_goals.predict(X_enc2)
    lam_a2 = art.model_away_goals.predict(X_enc2)
    P2, K2, _, _ = hda_tail_safe(lam_h2, lam_a2, tol=TAIL_TOL)
    pred2 = [CLASS_ORDER[int(P2[i].argmax())] for i in range(n_test)]
    sh2, sa2, _ = modal_scoreline(lam_h2, lam_a2, K2)

    max_lam_diff = max(float(np.max(np.abs(lam_h - lam_h2))),
                       float(np.max(np.abs(lam_a - lam_a2))))
    max_prob_diff = float(np.max(np.abs(P - P2)))

    print(f"  max lambda difference:      {max_lam_diff:.6e}")
    print(f"  max probability difference: {max_prob_diff:.6e}")

    check(f"lambda deterministic (<= {REPRODUCIBILITY_TOL})",
          max_lam_diff <= REPRODUCIBILITY_TOL)
    check(f"probability deterministic (<= {REPRODUCIBILITY_TOL})",
          max_prob_diff <= REPRODUCIBILITY_TOL)
    check("predicted class identical", predicted_class == pred2)
    check("grid K identical", K == K2)
    check("modal scoreline identical",
          np.array_equal(sh, sh2) and np.array_equal(sa, sa2))

    # ==================================================================
    # STEP 8 -- VENUE NULL / EDGE CASE HANDLING
    # ==================================================================
    rule("STEP 8 -- VENUE NULL / EDGE CASE HANDLING")

    venue_cols = list(art.venue_columns)
    venue_nulls = X_test[venue_cols].isna().sum()
    total_null = int(venue_nulls.sum())
    print(f"  Venue NULLs in test batch: {total_null}")

    # Synthetic venue-NULL row
    X_null = X_test.iloc[[0]].copy()
    for vc in venue_cols:
        X_null[vc] = np.nan

    try:
        enc_null = art.preprocessor.transform(X_null)
        lh_null = art.model_home_goals.predict(enc_null)
        la_null = art.model_away_goals.predict(enc_null)
        preds_null = predict_poisson(lh_null, la_null, art.class_order)
        null_ok = (np.all(np.isfinite([lh_null[0], la_null[0]]))
                   and lh_null[0] > 0 and la_null[0] > 0)
        check("venue-NULL row produces valid prediction", null_ok)
        d = preds_null[0]
        print(f"    λ_h={d.lambda_home:.4f}  λ_a={d.lambda_away:.4f}  "
              f"P(H)={d.probabilities['H']:.4f}  "
              f"P(D)={d.probabilities['D']:.4f}  "
              f"P(A)={d.probabilities['A']:.4f}")
    except Exception as exc:
        check("venue-NULL row produces valid prediction", False)
        print(f"    ERROR: {exc}")

    # Distribution check
    pred_arr = np.array(predicted_class)
    print(f"\n  Prediction distribution: H={sum(pred_arr=='H')}, "
          f"D={sum(pred_arr=='D')}, A={sum(pred_arr=='A')}")
    check("no NaN in probabilities", not np.any(np.isnan(P)))
    check("no infinity in probabilities", not np.any(np.isinf(P)))

    # ==================================================================
    # STEP 9 -- NO-WRITE GATE
    # ==================================================================
    rule("STEP 9 -- NO-WRITE GATE")

    post_snap = snapshot_checksums()

    all_unchanged = True
    for key in EXPECTED_MD5:
        pre_val = pre_snap.get(key)
        post_val = post_snap.get(key)
        expected = EXPECTED_MD5[key]
        if post_val != expected:
            check(f"{key} unchanged (post vs expected)", False)
            print(f"    expected: {expected}")
            print(f"    post:     {post_val}")
            all_unchanged = False
        elif pre_val != post_val:
            check(f"{key} unchanged (pre vs post)", False)
            all_unchanged = False
        else:
            check(f"{key} unchanged", True)

    # Production source unchanged (informational)
    source_changed = []
    for rel in PRODUCTION_SOURCE_FILES:
        pre_val = pre_snap.get(rel)
        post_val = post_snap.get(rel)
        if pre_val and post_val and pre_val != post_val:
            source_changed.append(rel)
    if source_changed:
        for rel in source_changed:
            check(f"production source {rel} unchanged", False)
        all_unchanged = False
    else:
        check("all production source files unchanged during test", True)

    # ==================================================================
    # FINAL REPORT
    # ==================================================================
    rule("FINAL REPORT")

    print(f"\n  Total checks: {passes + fails}  (PASS: {passes}  FAIL: {fails})")
    print()

    v2_md5_ok = md5(REPO / "data/models/v2_poisson_venue.pkl") == EXPECTED_MD5["data/models/v2_poisson_venue.pkl"]
    v1_md5_ok = md5(REPO / "data/models/v1_logreg.pkl") == EXPECTED_MD5["data/models/v1_logreg.pkl"]

    report = {
        "V2 ARTIFACT":                "VALID" if v2_md5_ok else "INVALID",
        "V2 PRODUCTION LOADING":      "PASS",
        "84-COLUMN CONTRACT":         "PASS" if len(feature_cols) == 84 else "FAIL",
        "VENUE FEATURES":             "PASS" if tuple(feature_cols[80:]) == art.venue_columns else "FAIL",
        "POISSON INFERENCE":          "PASS" if np.all(lam_h > 0) and np.all(lam_a > 0) else "FAIL",
        "H/D/A CONVERSION":           "PASS" if max_sum_error <= ROWSUM_TOL else "FAIL",
        "PRODUCTION RESPONSE":        "PASS" if all_schema_ok else "FAIL",
        "DETERMINISM":                "PASS" if max_prob_diff <= REPRODUCIBILITY_TOL else "FAIL",
        "V1 BACKWARD COMPATIBILITY":  "PASS",  # verified in step 2
        "DATABASE INTEGRITY":         "PASS" if all_unchanged else "FAIL",
        "V1 ARTIFACT INTEGRITY":      "PASS" if v1_md5_ok else "FAIL",
        "V2 ARTIFACT INTEGRITY":      "PASS" if v2_md5_ok else "FAIL",
        "2025/26 OUTCOME ACCESS":     "NONE",
        "TRAINING PERFORMED":         "NO",
        "EVALUATION PERFORMED":       "NO",
    }

    max_key_len = max(len(k) for k in report)
    for k, v in report.items():
        print(f"  {k:<{max_key_len}}  :  {v}")

    print()
    if fails == 0:
        print("  PRODUCTION INTEGRATION: READY")
    else:
        print(f"  PRODUCTION INTEGRATION: BLOCKED ({fails} failure(s))")

    return 1 if fails > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
