"""Predict a single EXISTING fixture by fixture_id.

    python predict_match.py --fixture-id 123456           # uses V2 (default)
    python predict_match.py --fixture-id 123456 --v1      # forces V1
    python predict_match.py --fixture-id 123456 --v3      # uses V3 (Elo)

SCOPE -- READ THIS BEFORE USING THE OUTPUT.

This is NOT a live or upcoming-match predictor. It can only predict
fixtures that ALREADY have a feature row in data/processed/features.db.
Those are historical fixtures, all of which are in the model's training
partition, so the probabilities printed here are IN-SAMPLE and are not
evidence of predictive performance. Nothing here should be read as a
backtest or an accuracy claim.

Predicting an upcoming match would require upcoming-fixture ingestion,
team-name resolution and a defined current-season semantics, none of
which exist (see the D-28 trace). That is deliberately out of scope.

LEAKAGE. The predictor reads exactly the contract columns (80 for V1,
84 for V2, 87 for V3) and nothing else. `label_result`,
`label_home_goals` and `label_away_goals` are never read, never passed
to the model, and never printed. V3 Elo features are computed causally
from completed matches strictly before the fixture's timestamp.

V2 INTEGRATION. The default model is now V2 (Poisson+Venue), which
provides expected goals, modal scoreline, and H/D/A probabilities.
V1 remains available via --v1 for backward compatibility.

V3 INTEGRATION. V3 (Poisson+Venue+Elo) extends V2 with 3 persistent
Elo features. Available via --v3. V2 remains the default.

Nothing is fitted. No database is opened writable.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

RESULT_LABEL = {"H": "HOME WIN", "D": "DRAW", "A": "AWAY WIN"}


def fail(msg: str, code: int = 2) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return code


def team_names(fixture_id: int) -> tuple[str, str]:
    """Fixture metadata only -- never a post-match field."""
    db = REPO / "data/processed/matches.db"
    if not db.exists():
        return ("<unknown>", "<unknown>")
    con = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT home_name, away_name FROM fixtures WHERE fixture_id = ?",
            (fixture_id,)).fetchone()
    except sqlite3.Error:
        return ("<unknown>", "<unknown>")
    finally:
        con.close()
    return (row[0], row[1]) if row else ("<unknown>", "<unknown>")


# ======================================================================
# V1 PREDICTION PATH (backward compatible)
# ======================================================================
def predict_v1(fixture_id: int, artifact_path: Path | None) -> int:
    """Original V1 LogisticRegression prediction path."""
    from models import artifact as artifact_mod
    from models.ablation import MODEL_B_COLUMNS
    from models.candidate_contract import predict_candidate
    from models.data import load_supervised_dataset

    path = artifact_path or artifact_mod.DEFAULT_ARTIFACT_PATH
    if not path.is_absolute():
        path = REPO / path
    try:
        art = artifact_mod.load(path)
    except artifact_mod.ArtifactError as exc:
        return fail(str(exc))

    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    mask = (ds.metadata["fixture_id"] == fixture_id).values
    if not mask.any():
        return fail(
            f"fixture_id {fixture_id} has no feature row in features.db. "
            "This predictor only supports fixtures already present in the "
            "feature database; it is not an upcoming-match predictor.")

    X = ds.X[mask][list(art.feature_columns)]
    if list(X.columns) != list(MODEL_B_COLUMNS):
        return fail("feature column order does not match the frozen V1 contract")

    result = predict_candidate(art.model, art.preprocessor, X, [fixture_id])
    probs = result.probabilities[0]
    order = list(result.class_order)
    p = dict(zip(order, probs))
    predicted = order[int(probs.argmax())]

    home, away = team_names(fixture_id)
    print("=" * 60)
    print("MATCH PREDICTION (V1 -- LogisticRegression)")
    print("=" * 60)
    print(f"Fixture ID      : {fixture_id}")
    print(f"Home Team       : {home}")
    print(f"Away Team       : {away}")
    print(f"Predicted Result: {RESULT_LABEL[predicted]}")
    print("Win Probabilities:")
    print(f"  Home : {100 * p['H']:.2f}%")
    print(f"  Draw : {100 * p['D']:.2f}%")
    print(f"  Away : {100 * p['A']:.2f}%")
    print("Model:")
    print(f"  Version  : {art.model_version}")
    print(f"  Features : {len(art.feature_columns)}")
    print("Expected Goals:")
    print("  NOT AVAILABLE - V1 has no production expected-goals model.")
    print("=" * 60)
    print("Note: this fixture is historical and is in the model's training")
    print("partition, so these probabilities are IN-SAMPLE, not a forecast.")
    return 0


# ======================================================================
# V2 PREDICTION PATH (Poisson + Venue)
# ======================================================================
def predict_v2(fixture_id: int, artifact_path: Path | None,
               output_json: bool = False) -> int:
    """V2 Poisson+Venue prediction path."""
    import numpy as np
    from models import v2_artifact as v2_mod
    from models.poisson import predict_poisson
    from models.data import load_supervised_dataset

    path = artifact_path or (REPO / v2_mod.DEFAULT_V2_ARTIFACT_PATH)
    if not path.is_absolute():
        path = REPO / path
    try:
        art = v2_mod.load(path)
    except v2_mod.V2ArtifactError as exc:
        return fail(str(exc))

    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    mask = (ds.metadata["fixture_id"] == fixture_id).values
    if not mask.any():
        return fail(
            f"fixture_id {fixture_id} has no feature row in features.db. "
            "This predictor only supports fixtures already present in the "
            "feature database; it is not an upcoming-match predictor.")

    # Select exactly the 84 V2 contract columns. Labels are not in ds.X
    # (X_EXCLUDED_COLUMNS removes them), and this selection would drop
    # them regardless.
    feature_cols = list(art.feature_columns)
    missing = [c for c in feature_cols if c not in ds.X.columns]
    if missing:
        return fail(
            f"V2 contract columns missing from features.db: {missing}. "
            "The feature database may need regeneration with venue features.")

    X = ds.X[mask][feature_cols]
    if list(X.columns) != feature_cols:
        return fail("feature column order does not match the V2 84-column contract")
    if len(X.columns) != 84:
        return fail(f"expected 84 feature columns, got {len(X.columns)}")

    # Preprocess
    X_encoded = art.preprocessor.transform(X)

    # Predict lambdas
    lam_h = art.model_home_goals.predict(X_encoded)
    lam_a = art.model_away_goals.predict(X_encoded)

    # Full Poisson prediction
    predictions = predict_poisson(lam_h, lam_a, art.class_order)
    pred = predictions[0]

    if output_json:
        print(json.dumps(pred.to_dict(), indent=2))
        return 0

    home, away = team_names(fixture_id)
    print("=" * 60)
    print("MATCH PREDICTION (V2 -- Poisson + Venue)")
    print("=" * 60)
    print(f"Fixture ID      : {fixture_id}")
    print(f"Home Team       : {home}")
    print(f"Away Team       : {away}")
    print(f"Predicted Result: {RESULT_LABEL[pred.prediction]}")
    print("Win Probabilities:")
    print(f"  Home : {100 * pred.probabilities['H']:.2f}%")
    print(f"  Draw : {100 * pred.probabilities['D']:.2f}%")
    print(f"  Away : {100 * pred.probabilities['A']:.2f}%")
    print("Expected Goals:")
    print(f"  Home : {pred.expected_goals_home:.3f}")
    print(f"  Away : {pred.expected_goals_away:.3f}")
    print(f"Modal Scoreline : {pred.modal_scoreline}  "
          f"(p={pred.modal_scoreline_probability:.4f})")
    print("Model:")
    print(f"  Version  : {art.model_version}")
    print(f"  Name     : {art.model_name}")
    print(f"  Features : {art.n_features}")
    print("=" * 60)
    print("Note: this fixture is historical and is in the model's training")
    print("partition, so these probabilities are IN-SAMPLE, not a forecast.")
    return 0


# ======================================================================
# V3 PREDICTION PATH (Poisson + Venue + Elo)
# ======================================================================
def predict_v3(fixture_id: int, artifact_path: Path | None,
               output_json: bool = False) -> int:
    """V3 Poisson+Venue+Elo prediction path.

    Loads the 84 base features from features.db, computes causal Elo
    features from matches.db, joins to form the 87-column V3 contract,
    and predicts using the V3 candidate artifact.
    """
    import numpy as np
    from features.elo import get_elo_for_fixture, ELO_COLUMNS
    from models.v3_artifact import (
        load_v3_artifact, V3ArtifactError,
        DEFAULT_V3_PRODUCTION_PATH, DEFAULT_V3_CANDIDATE_PATH,
    )
    from models.v3_contract import V3_FEATURE_COLUMNS, V3_N_FEATURES
    from models.poisson import predict_poisson
    from models.data import load_supervised_dataset

    # Resolve artifact: prefer promoted production path, fall back to candidate
    if artifact_path:
        path = artifact_path if artifact_path.is_absolute() else REPO / artifact_path
    else:
        prod_path = REPO / DEFAULT_V3_PRODUCTION_PATH
        cand_path = REPO / DEFAULT_V3_CANDIDATE_PATH
        if prod_path.exists():
            path = prod_path
        elif cand_path.exists():
            path = cand_path
        else:
            return fail(
                "No V3 artifact found. Expected one of:\n"
                f"  {prod_path}\n  {cand_path}")

    try:
        art = load_v3_artifact(path)
    except (V3ArtifactError, FileNotFoundError) as exc:
        return fail(str(exc))

    # Load base features from features.db
    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    mask = (ds.metadata["fixture_id"] == fixture_id).values
    if not mask.any():
        return fail(
            f"fixture_id {fixture_id} has no feature row in features.db. "
            "This predictor only supports fixtures already present in the "
            "feature database; it is not an upcoming-match predictor.")

    # Get the base 84 columns that are in features.db
    base_cols = [c for c in V3_FEATURE_COLUMNS if c not in ELO_COLUMNS]
    missing_base = [c for c in base_cols if c not in ds.X.columns]
    if missing_base:
        return fail(
            f"V3 base contract columns missing from features.db: {missing_base}")

    X = ds.X[mask][base_cols].copy()

    # Compute causal Elo features from matches.db
    matches_db = REPO / "data/processed/matches.db"
    try:
        elo_feats = get_elo_for_fixture(matches_db, fixture_id)
    except KeyError as exc:
        return fail(str(exc))

    # Append Elo columns
    for col in ELO_COLUMNS:
        X[col] = elo_feats[col]

    # Verify contract
    if list(X.columns) != list(V3_FEATURE_COLUMNS):
        return fail(
            f"V3 feature column order mismatch.\n"
            f"  Expected: {list(V3_FEATURE_COLUMNS)}\n"
            f"  Got:      {list(X.columns)}")
    if len(X.columns) != V3_N_FEATURES:
        return fail(f"expected {V3_N_FEATURES} feature columns, got {len(X.columns)}")

    # Preprocess
    X_encoded = art.preprocessor.transform(X)

    # Predict lambdas
    lam_h = art.model_home_goals.predict(X_encoded)
    lam_a = art.model_away_goals.predict(X_encoded)

    # Full Poisson prediction (same conversion as V2)
    predictions = predict_poisson(lam_h, lam_a, art.class_order)
    pred = predictions[0]

    if output_json:
        out = pred.to_dict()
        out["elo"] = elo_feats
        print(json.dumps(out, indent=2))
        return 0

    home, away = team_names(fixture_id)
    print("=" * 60)
    print("MATCH PREDICTION (V3 -- Poisson + Venue + Elo)")
    print("=" * 60)
    print(f"Fixture ID      : {fixture_id}")
    print(f"Home Team       : {home}")
    print(f"Away Team       : {away}")
    print(f"Predicted Result: {RESULT_LABEL[pred.prediction]}")
    print("Win Probabilities:")
    print(f"  Home : {100 * pred.probabilities['H']:.2f}%")
    print(f"  Draw : {100 * pred.probabilities['D']:.2f}%")
    print(f"  Away : {100 * pred.probabilities['A']:.2f}%")
    print("Expected Goals:")
    print(f"  Home : {pred.expected_goals_home:.3f}")
    print(f"  Away : {pred.expected_goals_away:.3f}")
    print(f"Modal Scoreline : {pred.modal_scoreline}  "
          f"(p={pred.modal_scoreline_probability:.4f})")
    print("Elo Ratings (pre-match):")
    print(f"  Home Elo : {elo_feats['home_elo']:.1f}")
    print(f"  Away Elo : {elo_feats['away_elo']:.1f}")
    print(f"  Elo Diff : {elo_feats['elo_diff']:.1f}")
    print("Model:")
    print(f"  Version  : {art.model_version}")
    print(f"  Name     : {art.model_name}")
    print(f"  Features : {art.n_features}")
    print(f"  Artifact : {path}")
    print("=" * 60)
    print("Note: this fixture is historical and is in the model's training")
    print("partition, so these probabilities are IN-SAMPLE, not a forecast.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Predict an existing fixture by fixture_id.")
    ap.add_argument("--fixture-id", required=True,
                    help="fixture_id that already exists in features.db")
    ap.add_argument("--artifact", default=None,
                    help="model artifact path (overrides default)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--v1", action="store_true",
                     help="use V1 (LogisticRegression)")
    grp.add_argument("--v3", action="store_true",
                     help="use V3 (Poisson+Venue+Elo)")
    ap.add_argument("--json", action="store_true",
                    help="output prediction as JSON (V2/V3 only)")
    args = ap.parse_args(argv)

    try:
        fixture_id = int(args.fixture_id)
    except (TypeError, ValueError):
        return fail(f"--fixture-id must be an integer, got {args.fixture_id!r}")

    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        return fail(f"scikit-learn is required to load the model artifact: {exc}")

    artifact_path = Path(args.artifact) if args.artifact else None

    if args.v1:
        return predict_v1(fixture_id, artifact_path)
    elif args.v3:
        return predict_v3(fixture_id, artifact_path, output_json=args.json)
    else:
        return predict_v2(fixture_id, artifact_path, output_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
