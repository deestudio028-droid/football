"""V4 — Build the production candidate artifact.

Usage:
    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python research/v4_promotion/build_v4.py

WHAT THIS BUILDS
----------------
    V4 = V3's exact 87 columns + A_home, D_home, A_away, D_away  =  91

Only independently validated components are promoted:

    E1  causal Elo             PASS  — via src/features/elo.py (inside the 87)
    E6  online attack/defense  PASS  — via src/features/online_attack_defense.py
    E2  market odds            PASS  — deliberately ABSENT from the design matrix

Market probabilities are a post-hoc signal in E2's validated architecture.
Putting them in this matrix would reinterpret E2 AND recreate the E7 Arm D
construction, which FAILED. They never enter here.

Excluded entirely: E3 Dixon-Coles, E4 time decay, E5 quadratic Elo,
E7 blend, E8 temperature scaling, E9 LightGBM.

MODEL LOGIC IS V3'S, UNCHANGED
------------------------------
Same targets (label_home_goals / label_away_goals), same
LogisticRegressionPreprocessor, same PoissonRegressor(alpha=1.0,
max_iter=2000), same tail-safe conversion metadata. The ONLY difference
from V3 is the four appended E6 columns.

TRAINING SCOPE
--------------
FINAL_TRAIN_SEASONS = 2020/21 .. 2024/25. **2025/26 is excluded and the
build aborts if a single 2025/26 row reaches the fitting set.**

SAFETY
------
Every gate below is a hard failure. This script writes NOTHING until all
pre-write gates pass, and it refuses to overwrite an existing V4 artifact.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
ARTIFACT_PATH = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
BUILD_MANIFEST = HERE / "v4_build_manifest.json"

PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
E2_FILES = {
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
}
E6_PROMOTED_MD5 = "ddab69c105e1233ac4972bb41273f1f8"

QUARANTINED_SEASON = "2025/2026"
#: E6 learning rate. E6's local walk-forward selected [0.01, 0.02, 0.01]
#: across its three folds. For a single final fit spanning all training
#: seasons the mid-grid value is used, matching E6's own fallback
#: convention. NOT tuned here, and never against 2025/26.
E6_LEARNING_RATE = 0.02


class BuildError(RuntimeError):
    """Raised when a V4 build gate fails."""


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def gate(cond: bool, label: str, detail: str = "") -> None:
    mark = "OK  " if cond else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise BuildError(f"GATE FAILED: {label}" + (f" ({detail})" if detail else ""))


def check_hashes(label: str, spec: dict) -> dict:
    print(f"\n--- {label} ---")
    out = {}
    for rel, exp in spec.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            out[rel] = {"status": "missing"}
            gate(False, rel, "file missing")
        a = md5(p)
        out[rel] = {"expected": exp, "actual": a, "sha256": sha256(p),
                    "status": "identical" if a == exp else "CHANGED"}
        gate(a == exp, rel, f"MD5={a}")
    return out


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_training_frame():
    """Assemble the 91-column training matrix from validated sources only."""
    from features.elo import (ELO_COLUMNS, HOME_ADVANTAGE, INIT_RATING,
                              K_FACTOR, MEAN_REVERSION, load_elo_features)
    from features.online_attack_defense import (AD_COLUMNS, EXP_CLIP,
                                                INIT_ATTACK, INIT_DEFENSE,
                                                STATE_CLIP, compute_ad_states,
                                                fit_baseline_rates)
    from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
    from models.data import load_supervised_dataset
    from models.v3_contract import V3_FEATURE_COLUMNS
    from models.v4_contract import V4_FEATURE_COLUMNS, validate_contract

    print("\n--- Contract verification ---")
    validate_contract()
    gate(len(V4_FEATURE_COLUMNS) == 91, "V4 contract == 91 columns",
         f"n={len(V4_FEATURE_COLUMNS)}")
    gate(tuple(V4_FEATURE_COLUMNS[:87]) == tuple(V3_FEATURE_COLUMNS),
         "V4[:87] == V3 contract, exact order")
    gate(tuple(V4_FEATURE_COLUMNS[87:]) == tuple(AD_COLUMNS),
         "V4[87:] == E6 A/D columns, exact order", f"{AD_COLUMNS}")
    gate(V3_FEATURE_COLUMNS[84:87] == ("home_elo", "away_elo", "elo_diff"),
         "E1 causal Elo at columns 85-87")

    print("\n--- Loading base features ---")
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)
    print(f"  feature rows loaded: {len(meta)}")

    print("\n--- E1 causal Elo (production implementation, verbatim) ---")
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])
    gate(not X[list(ELO_COLUMNS)].isna().any().any(),
         "Elo present for every feature row")
    print(f"  init={INIT_RATING} K={K_FACTOR} HA={HOME_ADVANTAGE} "
          f"mean_reversion={MEAN_REVERSION}")

    # ---- training season selection, BEFORE any A/D fitting -------------
    train_ids = SEASON_NAME_TO_IDS
    wanted = set()
    for s in FINAL_TRAIN_SEASONS:
        wanted |= set(train_ids[s])
    quarantined = set(train_ids[QUARANTINED_SEASON])

    print("\n--- Training scope ---")
    print(f"  FINAL_TRAIN_SEASONS: {list(FINAL_TRAIN_SEASONS)}")
    gate(QUARANTINED_SEASON not in FINAL_TRAIN_SEASONS,
         f"{QUARANTINED_SEASON} not in training seasons")
    gate(not (wanted & quarantined),
         f"No {QUARANTINED_SEASON} season_id in the training set")

    train_mask = meta["season_id"].isin(wanted).values
    quar_mask = meta["season_id"].isin(quarantined).values
    gate(int(quar_mask.sum()) > 0,
         f"{QUARANTINED_SEASON} rows exist and are being excluded",
         f"{int(quar_mask.sum())} rows excluded")
    gate(not (train_mask & quar_mask).any(),
         "Training mask and quarantine mask are disjoint")
    print(f"  training rows: {int(train_mask.sum())}")

    # ---- E6 online A/D, baseline fitted on TRAINING fixtures only -------
    print("\n--- E6 online attack/defense (promoted module, verbatim) ---")
    promoted = PROJECT_ROOT / "src/features/online_attack_defense.py"
    gate(md5(promoted) == E6_PROMOTED_MD5,
         "Promoted E6 module is the validated build", f"MD5={md5(promoted)}")

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query(
        """SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                  status, season, season_id, competition_id
           FROM fixtures WHERE competition_id IN (200,419,423,477,499)""",
        conn)
    conn.close()

    # baseline rates from TRAINING seasons only — never 2025/26
    hist = fx[fx.season_id.isin(wanted) & fx.home_goals.notna()
              & fx.status.isin(["FT", "AWARDED"])]
    gate(QUARANTINED_SEASON not in set(hist.season),
         "Baseline-rate fixtures exclude the quarantined season")
    base = fit_baseline_rates(hist.home_goals.values.astype(float),
                              hist.away_goals.values.astype(float))
    print(f"  mu_home={base.mu_home:.6f} mu_away={base.mu_away:.6f} "
          f"n={base.n_train}")
    print(f"  init={INIT_ATTACK}/{INIT_DEFENSE} clip=±{STATE_CLIP} "
          f"exp_clip=±{EXP_CLIP} lr={E6_LEARNING_RATE}")

    # States are computed over the FULL ordered history: the two-pass
    # engine guarantees each fixture reads state built only from strictly
    # earlier matches, so including later rows cannot contaminate earlier
    # ones. Only TRAINING rows are then used for fitting.
    states = compute_ad_states(fx, E6_LEARNING_RATE, base).set_index("fixture_id")
    for c in AD_COLUMNS:
        X[c] = meta["fixture_id"].map(states[c])
    gate(not X[list(AD_COLUMNS)].isna().any().any(),
         "A/D states present for every feature row")
    sub = X[list(AD_COLUMNS)].to_numpy(float)
    gate(bool(np.all(np.abs(sub) <= STATE_CLIP + 1e-9)),
         f"All A/D states within ±{STATE_CLIP}",
         f"max |state| = {float(np.max(np.abs(sub))):.6f}")

    # ---- assemble the 91-column matrix ---------------------------------
    print("\n--- Assembling the 91-column design matrix ---")
    for c in V4_FEATURE_COLUMNS:
        if c not in X.columns:
            raise BuildError(f"Contract column missing from features: {c}")
    X_all = X[list(V4_FEATURE_COLUMNS)].copy()
    gate(list(X_all.columns) == list(V4_FEATURE_COLUMNS),
         "Design matrix column order matches the contract exactly")
    gate(X_all.shape[1] == 91, "Design matrix width == 91",
         f"{X_all.shape}")

    X_train = X_all.loc[train_mask].copy()
    fids_train = meta.loc[train_mask, "fixture_id"].to_numpy()
    seasons_train = set(meta.loc[train_mask, "season_id"])
    gate(not (seasons_train & quarantined),
         "No quarantined season_id survived into X_train")

    # ---- targets --------------------------------------------------------
    conn = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    goals = pd.read_sql_query(
        "SELECT fixture_id, label_home_goals, label_away_goals "
        "FROM feature_rows WHERE label_result IS NOT NULL", conn)
    conn.close()
    gh = dict(zip(goals.fixture_id, goals.label_home_goals))
    ga = dict(zip(goals.fixture_id, goals.label_away_goals))
    missing = [f for f in fids_train if f not in gh]
    gate(not missing, "Every training fixture has goal labels",
         f"{len(fids_train)} labelled")

    y_home = np.array([gh[f] for f in fids_train], dtype=float)
    y_away = np.array([ga[f] for f in fids_train], dtype=float)
    gate(bool(np.all(np.isfinite(y_home)) and np.all(np.isfinite(y_away))),
         "All goal labels finite")
    gate(bool(np.all(y_home >= 0) and np.all(y_away >= 0)),
         "All goal labels non-negative")

    # 2025/26 labels must not be reachable from the training fixture ids
    quar_fids = set(meta.loc[quar_mask, "fixture_id"])
    gate(not (set(fids_train) & quar_fids),
         f"No {QUARANTINED_SEASON} label can enter the fit",
         f"{len(quar_fids)} quarantined fixtures excluded")

    elo_cfg = {"implementation": "src/features/elo.py",
               "init_rating": INIT_RATING, "k_factor": K_FACTOR,
               "home_advantage": HOME_ADVANTAGE,
               "mean_reversion": MEAN_REVERSION,
               "columns": list(ELO_COLUMNS), "experiment": "E1", "status": "PASS"}
    ad_cfg = {"implementation": "src/features/online_attack_defense.py",
              "module_md5": E6_PROMOTED_MD5,
              "init_attack": INIT_ATTACK, "init_defense": INIT_DEFENSE,
              "state_clip": STATE_CLIP, "exp_clip": EXP_CLIP,
              "learning_rate": E6_LEARNING_RATE,
              "baseline_mu_home": base.mu_home, "baseline_mu_away": base.mu_away,
              "baseline_n_train": base.n_train,
              "cross_season_persistence": True, "season_reset": False,
              "two_pass_timestamps": True,
              "columns": list(AD_COLUMNS), "experiment": "E6", "status": "PASS"}

    return (X_train, y_home, y_away, fids_train,
            list(FINAL_TRAIN_SEASONS), elo_cfg, ad_cfg,
            int(quar_mask.sum()))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_v4(X_train, y_home, y_away):
    """Fit V4 using V3's validated model logic, unchanged."""
    from sklearn.linear_model import PoissonRegressor
    from models.train import LogisticRegressionPreprocessor

    prep = LogisticRegressionPreprocessor().fit(X_train)
    E = prep.transform(X_train)
    gate(bool(np.all(np.isfinite(E))), "Encoded matrix contains no NaN/Inf",
         f"shape={E.shape}")

    mh = PoissonRegressor(alpha=1.0, max_iter=2000).fit(E, y_home)
    ma = PoissonRegressor(alpha=1.0, max_iter=2000).fit(E, y_away)

    lam_h, lam_a = mh.predict(E), ma.predict(E)
    gate(bool(np.all(np.isfinite(lam_h)) and np.all(np.isfinite(lam_a))),
         "In-sample lambdas finite")
    gate(bool(np.all(lam_h > 0) and np.all(lam_a > 0)),
         "In-sample lambdas strictly positive",
         f"min home={lam_h.min():.6f} min away={lam_a.min():.6f}")
    print(f"  lambda_home: mean={lam_h.mean():.4f} "
          f"[{lam_h.min():.4f}, {lam_h.max():.4f}]")
    print(f"  lambda_away: mean={lam_a.mean():.4f} "
          f"[{lam_a.min():.4f}, {lam_a.max():.4f}]")
    return prep, mh, ma


def main() -> int:
    print("=" * 78)
    print("V4 PRODUCTION CANDIDATE BUILD — E1 causal Elo + E6 online A/D")
    print("=" * 78)

    # ---- PRE-WRITE GATES ------------------------------------------------
    pre_protected = check_hashes("Protected files (PRE)", PROTECTED_FILES)
    pre_e2 = check_hashes("E2 research artefacts (PRE)", E2_FILES)

    print("\n--- Artifact path guard ---")
    gate(not ARTIFACT_PATH.exists(),
         "V4 artifact does not already exist (refusing to overwrite)",
         str(ARTIFACT_PATH.relative_to(PROJECT_ROOT)))

    (X_train, y_home, y_away, fids, train_seasons,
     elo_cfg, ad_cfg, n_quarantined) = build_training_frame()

    print("\n--- Fitting V4 (V3 model logic, unchanged) ---")
    prep, mh, ma = fit_v4(X_train, y_home, y_away)

    # ---- determinism ----------------------------------------------------
    print("\n--- Determinism: repeated fit ---")
    prep2, mh2, ma2 = fit_v4(X_train, y_home, y_away)
    E1_, E2_ = prep.transform(X_train), prep2.transform(X_train)
    d_prep = float(np.max(np.abs(E1_ - E2_)))
    d_h = float(np.max(np.abs(mh.predict(E1_) - mh2.predict(E2_))))
    d_a = float(np.max(np.abs(ma.predict(E1_) - ma2.predict(E2_))))
    gate(d_prep == 0.0, "Preprocessing bit-identical on repeat",
         f"max diff {d_prep:.3e}")
    gate(d_h < 1e-12 and d_a < 1e-12,
         "Repeated fit reproduces predictions",
         f"home {d_h:.3e}, away {d_a:.3e}")

    # ---- assemble payload ------------------------------------------------
    from models.baselines import CLASS_ORDER
    from models.v4_contract import V4_FEATURE_COLUMNS

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "model_name": "poisson_venue_elo_online_ad",
        "model_version": "v4.0-poisson-venue-elo-online-ad",
        "feature_version": "v1.0",
        "n_features": 91,
        "feature_columns": list(V4_FEATURE_COLUMNS),
        "class_order": list(CLASS_ORDER),
        "model_home_goals": mh,
        "model_away_goals": ma,
        "preprocessor": prep,
        "estimator_config": {"alpha": 1.0, "max_iter": 2000,
                             "model_type": "sklearn.linear_model.PoissonRegressor"},
        "conversion": {"type": "tail_safe_bivariate_poisson",
                       "method": "hda_tail_safe", "tail_tol": 1e-15,
                       "grid": "adaptive, derived from batch max lambda"},
        "training_seasons": list(train_seasons),
        "information_cutoff": {
            "training_seasons": list(train_seasons),
            "quarantined_season": QUARANTINED_SEASON,
            "quarantined_rows_excluded": n_quarantined,
            "holdout_note": ("2025/26 excluded from every feature, "
                             "statistic and fitted parameter"),
        },
        "elo_config": elo_cfg,
        "online_ad_config": ad_cfg,
        "deterministic_config": {
            "rng_used": False,
            "solver": "lbfgs (PoissonRegressor default)",
            "repeated_fit_verified": True,
            "max_abs_prediction_diff_on_refit": max(d_h, d_a),
        },
        "provenance": {
            "promoted_components": {"E1_causal_elo": "PASS",
                                    "E6_online_attack_defense": "PASS"},
            "market_signal": ("E2 PASS — kept as a SEPARATE post-hoc signal, "
                              "deliberately NOT a model feature"),
            "excluded": {"E3_dixon_coles": "INCONCLUSIVE",
                         "E4_time_decay": "INCONCLUSIVE",
                         "E5_quadratic_elo": "INCONCLUSIVE",
                         "E7_blend": "FAIL", "E8_temperature": "FAIL",
                         "E9_lightgbm": "FAIL"},
        },
        "created_at": now,
        "training_rows": int(len(X_train)),
        "holdout_used_in_training": False,
    }

    print("\n--- Pre-write payload verification ---")
    gate(len(payload["feature_columns"]) == 91, "Payload declares 91 features")
    gate(payload["holdout_used_in_training"] is False,
         "holdout_used_in_training is False")
    gate(QUARANTINED_SEASON not in payload["training_seasons"],
         f"{QUARANTINED_SEASON} absent from training_seasons")
    from models.v4_contract import FORBIDDEN_MARKET_COLUMNS
    gate(not (set(payload["feature_columns"]) & FORBIDDEN_MARKET_COLUMNS),
         "No market column in the artifact feature list")

    # ---- WRITE -----------------------------------------------------------
    print("\n--- Writing artifact ---")
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=5)
    a_md5, a_sha = md5(ARTIFACT_PATH), sha256(ARTIFACT_PATH)
    print(f"  path : {ARTIFACT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  MD5  : {a_md5}")
    print(f"  bytes: {ARTIFACT_PATH.stat().st_size}")

    # ---- POST-WRITE GATES -------------------------------------------------
    print("\n--- Artifact load verification ---")
    from models.v4_artifact import load_v4_artifact
    art = load_v4_artifact(ARTIFACT_PATH)
    gate(art.n_features == 91, "Loaded artifact declares 91 features")
    gate(tuple(art.feature_columns) == tuple(V4_FEATURE_COLUMNS),
         "Loaded feature_columns match the contract exactly")
    gate(art.holdout_used_in_training is False,
         "Loaded artifact confirms holdout not used")

    E_chk = art.preprocessor.transform(X_train)
    lh = art.model_home_goals.predict(E_chk)
    la = art.model_away_goals.predict(E_chk)
    gate(float(np.max(np.abs(lh - mh.predict(E1_)))) < 1e-12,
         "Loaded artifact reproduces home lambdas bit-identically")
    gate(float(np.max(np.abs(la - ma.predict(E1_)))) < 1e-12,
         "Loaded artifact reproduces away lambdas bit-identically")

    post_protected = check_hashes("Protected files (POST)", PROTECTED_FILES)
    post_e2 = check_hashes("E2 research artefacts (POST)", E2_FILES)

    BUILD_MANIFEST.write_text(json.dumps({
        "generated_at": now,
        "artifact": {"path": str(ARTIFACT_PATH.relative_to(PROJECT_ROOT)),
                     "md5": a_md5, "sha256": a_sha,
                     "bytes": ARTIFACT_PATH.stat().st_size},
        "training": {"rows": int(len(X_train)),
                     "seasons": list(train_seasons),
                     "quarantined_rows_excluded": n_quarantined,
                     "features": 91},
        "contract": {"n": 91, "v3_prefix_verified": True,
                     "ad_suffix_verified": True,
                     "market_columns_present": False},
        "elo_config": elo_cfg, "online_ad_config": ad_cfg,
        "determinism": {"preprocessing_max_diff": d_prep,
                        "home_max_diff": d_h, "away_max_diff": d_a,
                        "passed": True},
        "integrity": {"protected_pre": pre_protected,
                      "protected_post": post_protected,
                      "e2_pre": pre_e2, "e2_post": post_e2},
    }, indent=2, default=str))
    print(f"\n  Manifest -> {BUILD_MANIFEST.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 78)
    print("V4 CANDIDATE ARTIFACT BUILD: PASS")
    print("=" * 78)
    print(f"  artifact        : {ARTIFACT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  MD5             : {a_md5}")
    print(f"  training rows   : {len(X_train)}")
    print(f"  training seasons: {train_seasons}")
    print(f"  features        : 91")
    print("\n  This is a CANDIDATE. It is not production-ready and has not")
    print("  been evaluated against V2. Next: run the V4 promotion tests.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"\n{'=' * 78}\nBUILD ABORTED\n{'=' * 78}\n  {exc}")
        raise SystemExit(1)
