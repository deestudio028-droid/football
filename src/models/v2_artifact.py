"""V2 Poisson+Venue model artifact: load and verify.

Parallel to artifact.py (V1), not a replacement. artifact.py is frozen
and unchanged; V1 loading continues to work through that module.

This module handles ONLY the V2 Poisson+Venue artifact structure:

    model_name           "poisson_venue"
    model_version        "v2.0-poisson-venue"
    feature_version      "v1.0"
    n_features           84
    feature_columns      80 base + 4 venue columns
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
  - Write, save, or modify any artifact.
  - Read label columns or holdout outcomes.
  - Replace or modify artifact.py (V1).
  - Import from or modify the V1 artifact path.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baselines import CLASS_ORDER

#: Default V2 artifact location.
DEFAULT_V2_ARTIFACT_PATH = Path("data/models/v2_poisson_venue.pkl")

#: Expected artifact checksum.
EXPECTED_V2_MD5 = "25935b4e93fc4074f67f16e3181ed4df"

#: Expected V2 metadata.
EXPECTED_MODEL_NAME = "poisson_venue"
EXPECTED_MODEL_VERSION = "v2.0-poisson-venue"
EXPECTED_FEATURE_VERSION = "v1.0"
EXPECTED_N_FEATURES = 84
EXPECTED_ALPHA = 1.0
EXPECTED_MAX_ITER = 2000

#: The 4 venue columns that extend the V1 80-column contract.
VENUE_COLUMNS = (
    "home_goals_for_home_venue_season",
    "home_goals_against_home_venue_season",
    "away_goals_for_away_venue_season",
    "away_goals_against_away_venue_season",
)

#: Required top-level keys in the artifact dict.
REQUIRED_KEYS = (
    "model_name", "model_version", "feature_version",
    "feature_columns", "n_features", "class_order",
    "model_home_goals", "model_away_goals", "preprocessor",
    "estimator_config", "conversion",
    "base_contract_columns", "venue_columns",
    "training_seasons", "holdout_used_in_training",
)


class V2ArtifactError(RuntimeError):
    """Raised when the V2 artifact fails any verification gate."""


@dataclass(frozen=True)
class V2Artifact:
    """Loaded and verified V2 Poisson+Venue artifact."""
    model_home_goals: Any
    model_away_goals: Any
    preprocessor: Any
    feature_columns: tuple[str, ...]
    base_contract_columns: tuple[str, ...]
    venue_columns: tuple[str, ...]
    model_name: str
    model_version: str
    feature_version: str
    n_features: int
    class_order: list[str]
    estimator_config: dict
    conversion: dict
    training_seasons: tuple[str, ...]
    holdout_used_in_training: bool
    holdout_metrics: dict | None
    artifact_path: Path


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _fail(where: str, msg: str) -> V2ArtifactError:
    return V2ArtifactError(f"{where}: {msg}")


def load(
    path: Path = DEFAULT_V2_ARTIFACT_PATH,
    verify_checksum: bool = True,
) -> V2Artifact:
    """Load and verify the V2 Poisson+Venue artifact.

    Every verification gate from the integration test specification is
    enforced here. Failure is always loud (V2ArtifactError), never silent.

    Args:
        path: path to the V2 pickle artifact.
        verify_checksum: if True, verify the file's MD5 against
            EXPECTED_V2_MD5. Set False only for development/testing
            with a different artifact file.

    Returns:
        V2Artifact with all fields verified.
    """
    path = Path(path)
    if not path.exists():
        raise _fail("load", f"no V2 artifact at {path}")

    # Checksum gate
    if verify_checksum:
        actual_md5 = _md5(path)
        if actual_md5 != EXPECTED_V2_MD5:
            raise _fail("load",
                f"V2 artifact MD5 mismatch: expected {EXPECTED_V2_MD5}, "
                f"got {actual_md5}")

    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    if not isinstance(payload, dict):
        raise _fail("load", "artifact is not a dict")

    # Required keys
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise _fail("load", f"missing key(s): {missing}")

    # Model identity
    if payload["model_name"] != EXPECTED_MODEL_NAME:
        raise _fail("load",
            f"model_name is {payload['model_name']!r}, "
            f"expected {EXPECTED_MODEL_NAME!r}")
    if payload["model_version"] != EXPECTED_MODEL_VERSION:
        raise _fail("load",
            f"model_version is {payload['model_version']!r}, "
            f"expected {EXPECTED_MODEL_VERSION!r}")
    if payload["feature_version"] != EXPECTED_FEATURE_VERSION:
        raise _fail("load",
            f"feature_version is {payload['feature_version']!r}, "
            f"expected {EXPECTED_FEATURE_VERSION!r}")

    # Feature contract
    feature_cols = tuple(payload["feature_columns"])
    if payload["n_features"] != EXPECTED_N_FEATURES:
        raise _fail("load",
            f"n_features is {payload['n_features']}, expected {EXPECTED_N_FEATURES}")
    if len(feature_cols) != EXPECTED_N_FEATURES:
        raise _fail("load",
            f"feature_columns has {len(feature_cols)} entries, "
            f"expected {EXPECTED_N_FEATURES}")

    base_cols = tuple(payload["base_contract_columns"])
    venue_cols = tuple(payload["venue_columns"])
    if len(base_cols) != 80:
        raise _fail("load",
            f"base_contract_columns has {len(base_cols)} entries, expected 80")
    if venue_cols != VENUE_COLUMNS:
        raise _fail("load",
            f"venue_columns is {venue_cols}, expected {VENUE_COLUMNS}")
    if feature_cols != base_cols + venue_cols:
        raise _fail("load",
            "feature_columns != base_contract_columns + venue_columns")

    # Class order
    if list(payload["class_order"]) != list(CLASS_ORDER):
        raise _fail("load",
            f"class_order is {payload['class_order']}, expected {CLASS_ORDER}")

    # Estimator config
    cfg = payload.get("estimator_config", {})
    if cfg.get("alpha") != EXPECTED_ALPHA:
        raise _fail("load",
            f"estimator alpha is {cfg.get('alpha')}, expected {EXPECTED_ALPHA}")
    if cfg.get("max_iter") != EXPECTED_MAX_ITER:
        raise _fail("load",
            f"estimator max_iter is {cfg.get('max_iter')}, expected {EXPECTED_MAX_ITER}")

    # Model types (import lazily to match project pattern)
    from sklearn.linear_model import PoissonRegressor
    if not isinstance(payload["model_home_goals"], PoissonRegressor):
        raise _fail("load",
            f"model_home_goals is {type(payload['model_home_goals']).__name__}, "
            "expected PoissonRegressor")
    if not isinstance(payload["model_away_goals"], PoissonRegressor):
        raise _fail("load",
            f"model_away_goals is {type(payload['model_away_goals']).__name__}, "
            "expected PoissonRegressor")

    # Preprocessor
    if payload["preprocessor"] is None:
        raise _fail("load", "preprocessor is None")

    # Holdout safety
    if payload.get("holdout_used_in_training") is not False:
        raise _fail("load",
            f"holdout_used_in_training is {payload.get('holdout_used_in_training')}, "
            "expected False")

    # Forbidden columns in feature contract
    forbidden = {"label_home_goals", "label_away_goals", "label_result"}
    overlap = forbidden & set(feature_cols)
    if overlap:
        raise _fail("load", f"label columns in feature contract: {overlap}")

    return V2Artifact(
        model_home_goals=payload["model_home_goals"],
        model_away_goals=payload["model_away_goals"],
        preprocessor=payload["preprocessor"],
        feature_columns=feature_cols,
        base_contract_columns=base_cols,
        venue_columns=venue_cols,
        model_name=payload["model_name"],
        model_version=payload["model_version"],
        feature_version=payload["feature_version"],
        n_features=len(feature_cols),
        class_order=list(payload["class_order"]),
        estimator_config=dict(cfg),
        conversion=dict(payload.get("conversion", {})),
        training_seasons=tuple(payload.get("training_seasons", ())),
        holdout_used_in_training=False,
        holdout_metrics=payload.get("holdout_metrics"),
        artifact_path=path,
    )


__all__ = [
    "DEFAULT_V2_ARTIFACT_PATH", "EXPECTED_V2_MD5", "VENUE_COLUMNS",
    "V2ArtifactError", "V2Artifact", "load",
]
