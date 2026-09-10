"""V4 Poisson+Venue+Elo+OnlineAD model artifact: load and verify.

Parallel to v3_artifact.py, v2_artifact.py and artifact.py (V1). Leaves
every one of them frozen and unchanged.

This module handles ONLY the V4 artifact structure:

    model_name           "poisson_venue_elo_online_ad"
    model_version        "v4.0-poisson-venue-elo-online-ad"
    feature_version      "v1.0"
    n_features           91
    feature_columns      87 V3 columns + 4 online A/D columns
    class_order          ["H", "D", "A"]
    model_home_goals     fitted PoissonRegressor
    model_away_goals     fitted PoissonRegressor
    preprocessor         fitted LogisticRegressionPreprocessor
    estimator_config     {alpha: 1.0, max_iter: 2000}
    conversion           tail-safe Poisson metadata
    training_seasons     2020/21 through 2024/25
    information_cutoff   metadata for the promotion validation
    elo_config           E1 causal Elo parameters
    online_ad_config     E6 online attack/defense parameters
    holdout_used_in_training  False

PROVENANCE — only independently validated components:
    E1 causal Elo             PASS  (inside the 87 via v3_contract)
    E6 online attack/defense  PASS  (the 4 appended columns)
    E2 market odds            PASS  — deliberately ABSENT from features

Market probabilities are NEVER model features. E2's validated form is a
post-hoc probability signal; putting it in this design matrix would both
reinterpret E2 and recreate the E7 Arm D construction, which FAILED.

WHAT THIS MODULE DOES NOT DO:
  - Fit, train, or tune anything.
  - Write or replace the V1/V2/V3 artifacts.
  - Read label columns or holdout outcomes.
  - Promote V4 to production default.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baselines import CLASS_ORDER
from .v4_contract import (V4_FEATURE_COLUMNS, V4_N_FEATURES,
                          FORBIDDEN_MARKET_COLUMNS,
                          FORBIDDEN_EXPERIMENT_COLUMNS)

#: Default V4 candidate and production paths.
DEFAULT_V4_CANDIDATE_PATH = Path(
    "data/models/v4_poisson_venue_elo_online_ad.pkl")
DEFAULT_V4_PRODUCTION_PATH = Path("data/models/v4_production.pkl")

EXPECTED_MODEL_NAME = "poisson_venue_elo_online_ad"
EXPECTED_MODEL_VERSION = "v4.0-poisson-venue-elo-online-ad"
EXPECTED_FEATURE_VERSION = "v1.0"
EXPECTED_N_FEATURES = 91
EXPECTED_ALPHA = 1.0
EXPECTED_MAX_ITER = 2000

REQUIRED_KEYS = (
    "model_name",
    "model_version",
    "feature_version",
    "n_features",
    "feature_columns",
    "class_order",
    "model_home_goals",
    "model_away_goals",
    "preprocessor",
    "estimator_config",
    "conversion",
    "training_seasons",
    "information_cutoff",
    "elo_config",
    "online_ad_config",
    "deterministic_config",
    "created_at",
    "holdout_used_in_training",
)


class V4ArtifactError(RuntimeError):
    """Raised when a V4 artifact fails structural verification."""


@dataclass
class V4Artifact:
    model_name: str
    model_version: str
    feature_version: str
    n_features: int
    feature_columns: tuple[str, ...]
    class_order: list[str]
    model_home_goals: Any
    model_away_goals: Any
    preprocessor: Any
    estimator_config: dict[str, Any]
    conversion: dict[str, Any]
    training_seasons: tuple[str, ...]
    information_cutoff: dict[str, Any]
    elo_config: dict[str, Any]
    online_ad_config: dict[str, Any]
    deterministic_config: dict[str, Any]
    created_at: str
    holdout_used_in_training: bool

    source_path: Path


def load_v4_artifact(
    path: Path | str = DEFAULT_V4_CANDIDATE_PATH,
) -> V4Artifact:
    """Load and structurally verify a V4 artifact.

    Every check is a hard failure. A silently-wrong artifact reaching
    inference is worse than a loud refusal to load.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"V4 artifact not found: {path}")

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:                                # noqa: BLE001
        raise V4ArtifactError(f"Could not unpickle {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise V4ArtifactError(f"Artifact payload is not a dict: {type(payload)}")

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise V4ArtifactError(f"Artifact missing required keys: {missing}")

    if payload["model_name"] != EXPECTED_MODEL_NAME:
        raise V4ArtifactError(
            f"Unexpected model_name: {payload['model_name']!r}")
    if payload["model_version"] != EXPECTED_MODEL_VERSION:
        raise V4ArtifactError(
            f"Unexpected model_version: {payload['model_version']!r}")
    if payload["feature_version"] != EXPECTED_FEATURE_VERSION:
        raise V4ArtifactError(
            f"Unexpected feature_version: {payload['feature_version']!r}")

    if payload["n_features"] != EXPECTED_N_FEATURES:
        raise V4ArtifactError(
            f"Unexpected n_features: {payload['n_features']} "
            f"(expected {EXPECTED_N_FEATURES})")

    cols = tuple(payload["feature_columns"])
    if len(cols) != V4_N_FEATURES:
        raise V4ArtifactError(
            f"feature_columns has {len(cols)} entries, expected {V4_N_FEATURES}")
    if cols != tuple(V4_FEATURE_COLUMNS):
        # locate the first divergence so the error is actionable
        for i, (a, b) in enumerate(zip(cols, V4_FEATURE_COLUMNS)):
            if a != b:
                raise V4ArtifactError(
                    f"feature_columns diverges from the V4 contract at "
                    f"index {i}: artifact {a!r} vs contract {b!r}")
        raise V4ArtifactError("feature_columns does not match the V4 contract")

    forbidden = set(cols) & FORBIDDEN_MARKET_COLUMNS
    if forbidden:
        raise V4ArtifactError(
            f"Market columns present in artifact features: {sorted(forbidden)}")
    forbidden = set(cols) & FORBIDDEN_EXPERIMENT_COLUMNS
    if forbidden:
        raise V4ArtifactError(
            f"Non-promoted experiment columns in artifact: {sorted(forbidden)}")

    if list(payload["class_order"]) != list(CLASS_ORDER):
        raise V4ArtifactError(
            f"Unexpected class_order: {payload['class_order']}")

    cfg = payload["estimator_config"]
    if cfg.get("alpha") != EXPECTED_ALPHA:
        raise V4ArtifactError(f"Unexpected alpha: {cfg.get('alpha')}")
    if cfg.get("max_iter") != EXPECTED_MAX_ITER:
        raise V4ArtifactError(f"Unexpected max_iter: {cfg.get('max_iter')}")

    if payload["holdout_used_in_training"] is not False:
        raise V4ArtifactError(
            "holdout_used_in_training must be False — 2025/26 is quarantined")

    seasons = tuple(payload["training_seasons"])
    if "2025/2026" in seasons:
        raise V4ArtifactError(
            "2025/2026 appears in training_seasons — quarantine violated")

    for key in ("model_home_goals", "model_away_goals"):
        if not hasattr(payload[key], "predict"):
            raise V4ArtifactError(f"{key} has no .predict() method")
    if not hasattr(payload["preprocessor"], "transform"):
        raise V4ArtifactError("preprocessor has no .transform() method")

    return V4Artifact(
        model_name=payload["model_name"],
        model_version=payload["model_version"],
        feature_version=payload["feature_version"],
        n_features=payload["n_features"],
        feature_columns=cols,
        class_order=list(payload["class_order"]),
        model_home_goals=payload["model_home_goals"],
        model_away_goals=payload["model_away_goals"],
        preprocessor=payload["preprocessor"],
        estimator_config=dict(payload["estimator_config"]),
        conversion=dict(payload["conversion"]),
        training_seasons=seasons,
        information_cutoff=dict(payload["information_cutoff"]),
        elo_config=dict(payload["elo_config"]),
        online_ad_config=dict(payload["online_ad_config"]),
        deterministic_config=dict(payload["deterministic_config"]),
        created_at=payload["created_at"],
        holdout_used_in_training=payload["holdout_used_in_training"],
        source_path=path,
    )


__all__ = [
    "DEFAULT_V4_CANDIDATE_PATH",
    "DEFAULT_V4_PRODUCTION_PATH",
    "EXPECTED_MODEL_NAME",
    "EXPECTED_MODEL_VERSION",
    "EXPECTED_N_FEATURES",
    "V4ArtifactError",
    "V4Artifact",
    "load_v4_artifact",
]
