"""Read-only loader for the v1.1 successor model layer (D-22 / D-23).

WHY THIS MODULE EXISTS
`models.data` compares each row's `feature_version` against the
module-global `config.REQUIRED_FEATURE_VERSION` ("v1.0"), so it rejects
`data/processed/features_v1_1.db` outright. Both `data.py` and
`config.py` are frozen V1 sources pinned in the 13 LOCKED_INPUTS, so the
required version cannot be parameterised where it currently lives
without modifying them. D-15 anticipated exactly this and directed that
a successor-specific required-version constant be kept ISOLATED from the
V1 constant rather than mutating it. That is what this module does.

WHAT IS REUSED, NOT REIMPLEMENTED
Every production safety check is imported from `models.data` and called
directly -- the read-only connection, the label whitelist, the known-
unlabeled-fixture sanity check, the feature-column selection rules, the
SupervisedDataset container and the exception hierarchy. Nothing is
re-derived and nothing is bypassed. A successor loader that opened its
own sqlite connection and assembled X/y by hand would give the control
and candidate arms materially different loading paths and invalidate the
parity requirement of the D-22 experiment; that is precisely what this
module avoids.

THE ONE THING THAT DIFFERS
`_verify_single_feature_version` closes over the V1 constant, so the
three-branch version check (no version / multiple versions / wrong
version) is restated here against the successor constant. It is
deliberately branch-for-branch identical to the V1 original -- same
conditions, same exception types, same refusal to silently blend mixed
versions. tests/test_successor_data_loader.py proves that equivalence
two ways: a behavioural parity table across all four version scenarios,
and an AST comparison of the two functions with the constant name
normalised.

BOUNDARIES
No model is imported, fitted, tuned or evaluated here. No write is ever
possible: the connection comes from `models.data._read_only_connection`,
so any accidental INSERT/UPDATE/DDL raises. `MODEL_VERSION`,
`REQUIRED_FEATURE_VERSION`, `MODEL_B_COLUMNS`, `features.db` and
`matches.db` are neither read for modification nor touched. This module
does not promote anything.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import KNOWN_UNLABELED_FIXTURE_IDS
from .data import (
    FeatureVersionError,
    ModelDataError,
    SupervisedDataset,
    _read_only_connection,
    _verify_allowed_labels,
    get_feature_columns,
)

#: The only feature_version this successor model layer will accept.
#:
#: Declared as an independent literal, mirroring how V1 declares
#: `config.REQUIRED_FEATURE_VERSION` independently of
#: `features.config.FEATURE_VERSION`. The test suite asserts it equals
#: `features.config.SUCCESSOR_FEATURE_VERSION`, so the value the builder
#: stamps and the value this loader demands cannot drift apart.
SUCCESSOR_REQUIRED_FEATURE_VERSION: str = "v1.1"


def _verify_single_successor_feature_version(df: pd.DataFrame) -> None:
    """Successor counterpart of `data._verify_single_feature_version`.

    Branch-for-branch identical to the V1 original -- the only
    difference is the constant compared against. Mixed-version data is
    refused rather than silently resolved, exactly as in V1.
    """
    versions = sorted(df["feature_version"].dropna().unique().tolist())
    if len(versions) == 0:
        raise FeatureVersionError("feature_rows has no rows with a feature_version set.")
    if len(versions) > 1:
        raise FeatureVersionError(
            f"feature_rows contains multiple feature_version values: {versions}. "
            "Refusing to silently select one -- this must be resolved (e.g. by "
            "regenerating the feature dataset under a single version) before any "
            "model can be trained on it."
        )
    if versions[0] != SUCCESSOR_REQUIRED_FEATURE_VERSION:
        raise FeatureVersionError(
            f"feature_rows has feature_version={versions[0]!r}, but this model "
            f"layer requires {SUCCESSOR_REQUIRED_FEATURE_VERSION!r}. Refusing to proceed."
        )


def load_successor_feature_rows(db_path: Path) -> pd.DataFrame:
    """Successor counterpart of `data.load_feature_rows`.

    Loads the entire `feature_rows` table, unfiltered, and verifies
    exactly one feature_version is present and that it is v1.1.
    """
    conn = _read_only_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM feature_rows", conn)
    finally:
        conn.close()

    _verify_single_successor_feature_version(df)
    return df


def load_successor_supervised_dataset(db_path: Path) -> SupervisedDataset:
    """Successor counterpart of `data.load_supervised_dataset`.

    Applies the identical row filter (`label_result IS NOT NULL`, the
    only row filter -- never a blanket dropna, so legitimately-null
    feature values survive), the identical label whitelist, and the
    identical known-unlabeled-fixture sanity check. Metadata and feature
    column selection come from the same production rules, so the
    successor's X excludes exactly what V1's X excludes.
    """
    conn = _read_only_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM feature_rows WHERE label_result IS NOT NULL", conn
        )
    finally:
        conn.close()

    _verify_single_successor_feature_version(df)
    _verify_allowed_labels(df["label_result"])

    present_unlabeled = set(KNOWN_UNLABELED_FIXTURE_IDS) & set(df["fixture_id"].tolist())
    if present_unlabeled:
        raise ModelDataError(
            f"Fixture(s) {sorted(present_unlabeled)} are known to have no valid "
            "result (e.g. ABANDONED) but appeared in the supervised dataset anyway. "
            "This means the label_result IS NOT NULL filter did not behave as expected."
        )

    metadata_cols = ["fixture_id", "competition_id", "season_id", "unix", "feature_version"]
    metadata = df[metadata_cols].copy()
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["label_result"].copy()

    return SupervisedDataset(metadata=metadata, X=X, y=y)


__all__ = [
    "SUCCESSOR_REQUIRED_FEATURE_VERSION",
    "load_successor_feature_rows",
    "load_successor_supervised_dataset",
]
