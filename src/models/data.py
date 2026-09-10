"""Read-only loader for the V1 model layer. Reads exclusively from
`data/processed/features.db` (table `feature_rows`) -- never
`matches.db`, never `data/raw/`. The connection is opened in SQLite's
own read-only URI mode, so "never writes back into features.db" is a
structural guarantee (any accidental INSERT/UPDATE/DDL raises
immediately) rather than just a convention this module promises to follow.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    ALLOWED_LABELS,
    KNOWN_UNLABELED_FIXTURE_IDS,
    REQUIRED_FEATURE_VERSION,
    X_EXCLUDED_COLUMNS,
)


class ModelDataError(RuntimeError):
    """Base class for data-integrity problems this layer refuses to
    silently work around.
    """


class FeatureVersionError(ModelDataError):
    pass


class UnexpectedLabelError(ModelDataError):
    pass


def _read_only_connection(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_feature_rows(db_path: Path) -> pd.DataFrame:
    """Loads the entire `feature_rows` table, unfiltered. Verifies
    exactly one `feature_version` is present and that it matches
    REQUIRED_FEATURE_VERSION -- fails loudly (raises) rather than
    picking one version if more than one is found, per the explicit
    requirement that mixed-version data must never be silently blended.
    """
    conn = _read_only_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM feature_rows", conn)
    finally:
        conn.close()

    _verify_single_feature_version(df)
    return df


def _verify_single_feature_version(df: pd.DataFrame) -> None:
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
    if versions[0] != REQUIRED_FEATURE_VERSION:
        raise FeatureVersionError(
            f"feature_rows has feature_version={versions[0]!r}, but this model "
            f"layer requires {REQUIRED_FEATURE_VERSION!r}. Refusing to proceed."
        )


def _verify_allowed_labels(labels: pd.Series) -> None:
    seen = set(labels.dropna().unique().tolist())
    unexpected = seen - set(ALLOWED_LABELS)
    if unexpected:
        raise UnexpectedLabelError(
            f"label_result contains values outside {ALLOWED_LABELS}: {sorted(unexpected)}. "
            "Refusing to proceed -- this indicates a data-integrity problem upstream, "
            "not something this layer should coerce or filter around silently."
        )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column legal to feed into X: everything in `df` except the
    label columns and the metadata/bookkeeping columns explicitly
    excluded per docs/PHASE3_MODEL_DEVELOPMENT_DESIGN_REPORT.md's
    "feature matrix rules" (season_id, unix, home_id, away_id,
    fixture_id, generated_at, feature_version, and the 3 label_*
    columns). `competition_id` is deliberately NOT excluded -- it is
    the design report's recommended league-context feature.

    This performs no null-handling of any kind -- a column with
    legitimate NULLs (e.g. coverage-gated xG features) stays exactly as
    wide and exactly as null as it is in the source table. No blanket
    dropna() happens anywhere in this module.
    """
    return [c for c in df.columns if c not in X_EXCLUDED_COLUMNS]


@dataclass
class SupervisedDataset:
    """Bundles the three things every downstream step needs, kept
    explicitly separate so nothing can accidentally end up in the wrong one:

    - `metadata`: fixture_id, competition_id, season_id, unix,
      feature_version -- used for splitting/auditing, NOT fed to a model.
    - `X`: the legal feature matrix (see get_feature_columns) --
      includes competition_id, excludes season_id/unix/home_id/away_id/
      fixture_id/generated_at/feature_version/labels.
    - `y`: label_result ("H"/"D"/"A" only, enforced).
    """
    metadata: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series

    def __len__(self) -> int:
        return len(self.y)


def load_supervised_dataset(db_path: Path) -> SupervisedDataset:
    """The one function training/evaluation code should actually call.

    Filters to `label_result IS NOT NULL` (the explicit supervised-
    training filter -- this is the ONLY row filter applied; it is not a
    blanket dropna() over feature columns, so legitimately-null feature
    values are preserved for every row that has a valid label). Verifies
    the resulting labels are exclusively H/D/A. Confirms the one known
    unlabeled fixture (the ABANDONED match) is excluded as an
    independent sanity check, not just a side effect of the SQL filter.
    """
    conn = _read_only_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM feature_rows WHERE label_result IS NOT NULL", conn
        )
    finally:
        conn.close()

    _verify_single_feature_version(df)
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
