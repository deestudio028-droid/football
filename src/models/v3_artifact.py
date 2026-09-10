"""V3 Poisson+Venue+Persistent Elo model artifact: load and verify.

Parallel to v2_artifact.py and artifact.py (V1). Leaves both frozen and
unchanged.

This module handles ONLY the V3 Poisson+Venue+Elo artifact structure:

    model_name           "poisson_venue_elo"
    model_version        "v3.0-poisson-venue-elo"
    feature_version      "v1.0"
    n_features           87
    feature_columns      80 base + 4 venue + 3 elo columns
    class_order          ["H", "D", "A"]
    model_home_goals     fitted PoissonRegressor
    model_away_goals     fitted PoissonRegressor
    preprocessor         fitted LogisticRegressionPreprocessor
    estimator_config     {alpha: 1.0, max_iter: 2000}
    conversion           tail-safe Poisson metadata
    training_seasons     2020/21 through 2024/25
    holdout_used_in_training  False

WHAT THIS MODULE DOES NOT DO:
  - Fit, train, or tune anything.
  - Write or replace the V2 champion artifact.
  - Read label columns or holdout outcomes.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .baselines import CLASS_ORDER
from .v3_contract import V3_FEATURE_COLUMNS, V3_N_FEATURES, ELO_FEATURE_COLUMNS
from .v2_artifact import VENUE_COLUMNS

#: Default candidate and production paths
DEFAULT_V3_CANDIDATE_PATH = Path("data/models/v3_poisson_venue_elo_candidate.pkl")
DEFAULT_V3_PRODUCTION_PATH = Path("data/models/v3_poisson_venue_elo.pkl")

EXPECTED_MODEL_NAME = "poisson_venue_elo"
EXPECTED_MODEL_VERSION = "v3.0-poisson-venue-elo"
EXPECTED_FEATURE_VERSION = "v1.0"
EXPECTED_N_FEATURES = 87
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
    "holdout_used_in_training",
)


class V3ArtifactError(RuntimeError):
    """Raised when artifact contents or structure violate invariants."""


@dataclass(frozen=True)
class V3Artifact:
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
    holdout_used_in_training: bool
    md5: str
    source_path: Path


def load_v3_artifact(
    path: Path | str = DEFAULT_V3_CANDIDATE_PATH,
    expected_md5: str | None = None,
) -> V3Artifact:
    """Load and verify a V3 candidate or production artifact."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"V3 artifact not found at {p.resolve()}")

    data_bytes = p.read_bytes()
    actual_md5 = hashlib.md5(data_bytes).hexdigest()

    if expected_md5 is not None and actual_md5 != expected_md5:
        raise V3ArtifactError(
            f"V3 artifact MD5 mismatch at {p}: expected {expected_md5}, got {actual_md5}"
        )

    try:
        payload = pickle.loads(data_bytes)
    except Exception as exc:
        raise V3ArtifactError(f"Failed to unpickle V3 artifact at {p}: {exc}") from exc

    if not isinstance(payload, dict):
        raise V3ArtifactError(f"V3 artifact must be a dict, got {type(payload)}")

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise V3ArtifactError(f"V3 artifact missing required keys: {missing}")

    if payload["model_name"] != EXPECTED_MODEL_NAME:
        raise V3ArtifactError(f"Unexpected model_name: {payload['model_name']}")
    if payload["model_version"] != EXPECTED_MODEL_VERSION:
        raise V3ArtifactError(f"Unexpected model_version: {payload['model_version']}")
    if payload["feature_version"] != EXPECTED_FEATURE_VERSION:
        raise V3ArtifactError(f"Unexpected feature_version: {payload['feature_version']}")
    if payload["n_features"] != EXPECTED_N_FEATURES:
        raise V3ArtifactError(f"Unexpected n_features: {payload['n_features']}")
    if tuple(payload["feature_columns"]) != V3_FEATURE_COLUMNS:
        raise V3ArtifactError("feature_columns does not match V3_FEATURE_COLUMNS contract")
    if list(payload["class_order"]) != list(CLASS_ORDER):
        raise V3ArtifactError(f"class_order mismatch: {payload['class_order']}")
    if payload["holdout_used_in_training"] is not False:
        raise V3ArtifactError("holdout_used_in_training must be False")

    return V3Artifact(
        model_name=payload["model_name"],
        model_version=payload["model_version"],
        feature_version=payload["feature_version"],
        n_features=payload["n_features"],
        feature_columns=tuple(payload["feature_columns"]),
        class_order=list(payload["class_order"]),
        model_home_goals=payload["model_home_goals"],
        model_away_goals=payload["model_away_goals"],
        preprocessor=payload["preprocessor"],
        estimator_config=dict(payload["estimator_config"]),
        conversion=dict(payload["conversion"]),
        training_seasons=tuple(payload["training_seasons"]),
        holdout_used_in_training=payload["holdout_used_in_training"],
        md5=actual_md5,
        source_path=p,
    )


def predict_lambdas(artifact: V3Artifact, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict expected home and away goal rates (lambdas)."""
    X_input = X[list(artifact.feature_columns)]
    E = artifact.preprocessor.transform(X_input)
    lam_h = artifact.model_home_goals.predict(E)
    lam_a = artifact.model_away_goals.predict(E)
    return np.asarray(lam_h, dtype=float), np.asarray(lam_a, dtype=float)


def _grid_size(max_lambda: float, tol: float = 1e-15) -> int:
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


def predict_hda_probabilities(
    artifact: V3Artifact, X: pd.DataFrame, tol: float = 1e-15
) -> np.ndarray:
    """Predict [P(H), P(D), P(A)] probabilities using tail-safe Poisson conversion."""
    lam_h, lam_a = predict_lambdas(artifact, X)
    max_lam = float(max(np.max(lam_h), np.max(lam_a)))
    K = _grid_size(max_lam, tol)
    ph = _pmf_grid(lam_h, K)
    pa = _pmf_grid(lam_a, K)
    Fa = np.cumsum(pa, axis=1)
    p_home = (ph[:, 1:] * Fa[:, :-1]).sum(axis=1)
    p_draw = (ph * pa).sum(axis=1)
    p_away = 1.0 - p_home - p_draw
    return np.column_stack([p_home, p_draw, p_away])


def predict_score_grid(
    lam_h: float, lam_a: float, max_goals: int = 7
) -> np.ndarray:
    """Compute (max_goals+1, max_goals+1) score probability grid."""
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k, lam_h)
    pa = poisson.pmf(k, lam_a)
    M = np.outer(ph, pa)
    return M / M.sum()