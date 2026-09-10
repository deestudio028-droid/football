"""Production causal Elo ratings for V3 inference.

Derived from research/worldcup_elo/elo_engine.py (the validated causal
engine that passed all 11 mechanical property tests and produced the V3
candidate artifact).

CAUSAL INVARIANT:
For every fixture at timestamp T, the pre-match Elo features depend
EXCLUSIVELY on fixtures completed strictly before T. Simultaneous
fixtures (same unix timestamp) have their features extracted in a first
pass before any ratings are updated in a second pass.

WHAT THIS MODULE DOES:
  - Reads matches.db in read-only mode.
  - Computes home_elo, away_elo, elo_diff for every fixture.
  - Returns a DataFrame keyed on fixture_id.

WHAT THIS MODULE DOES NOT DO:
  - Fit, train, or tune anything.
  - Write to any file or database.
  - Access label columns or outcomes for feature construction.
  - Modify the V2 or V1 artifacts.
  - Add features beyond the 3 in the V3 contract.

ELO PARAMETERS (frozen, matching the V3 candidate artifact):
  init_rating:   1500.0
  k_factor:      20.0
  home_advantage: 100.0
  goal_diff_multiplier: eloratings.net standard
  mean_reversion: 0.0 (continuous carry-forward)
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd


# --- Frozen Elo parameters (must match V3 candidate training) ----------

INIT_RATING: float = 1500.0
K_FACTOR: float = 20.0
HOME_ADVANTAGE: float = 100.0
MEAN_REVERSION: float = 0.0  # no season-boundary regression

# V3 contract column names
ELO_COLUMNS: tuple[str, ...] = ("home_elo", "away_elo", "elo_diff")


def _goal_diff_multiplier(diff: int) -> float:
    """eloratings.net standard margin multiplier.

    Identical to research/worldcup_elo/elo_engine.py::goal_diff_multiplier.
    """
    ad = abs(int(diff))
    if ad <= 1:
        return 1.0
    elif ad == 2:
        return 1.5
    else:
        return (11.0 + ad) / 8.0


def _expected_score(r_home: float, r_away: float) -> float:
    """Expected score for the home team including home advantage."""
    delta = (r_home + HOME_ADVANTAGE) - r_away
    return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))


def compute_elo_features(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Compute causal pre-match Elo features for all fixtures.

    Args:
        fixtures_df: Must contain columns:
            fixture_id, unix, home_id, away_id,
            home_goals, away_goals, status

    Returns:
        DataFrame with columns: fixture_id, home_elo, away_elo, elo_diff
        One row per input fixture, in input order.

    The two-pass-per-timestamp design guarantees that simultaneous
    fixtures never leak into each other's pre-match ratings.
    """
    df = fixtures_df.sort_values(["unix", "fixture_id"]).reset_index(drop=True)
    ratings: dict[int, float] = defaultdict(lambda: INIT_RATING)

    records: list[dict] = []

    for _unix_ts, group in df.groupby("unix", sort=True):
        # --- First pass: extract pre-match features ---
        for row in group.itertuples(index=False):
            r_home = ratings[row.home_id]
            r_away = ratings[row.away_id]
            records.append({
                "fixture_id": row.fixture_id,
                "home_elo": r_home,
                "away_elo": r_away,
                "elo_diff": (r_home + HOME_ADVANTAGE) - r_away,
            })

        # --- Second pass: update ratings with completed match outcomes ---
        for row in group.itertuples(index=False):
            if (row.status in ("FT", "AWARDED")
                    and pd.notna(row.home_goals)
                    and pd.notna(row.away_goals)):
                hg = int(row.home_goals)
                ag = int(row.away_goals)

                r_home = ratings[row.home_id]
                r_away = ratings[row.away_id]

                we = _expected_score(r_home, r_away)
                w = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
                g = _goal_diff_multiplier(hg - ag)
                delta_r = K_FACTOR * g * (w - we)

                ratings[row.home_id] = r_home + delta_r
                ratings[row.away_id] = r_away - delta_r

    return pd.DataFrame(records)


def load_elo_features(matches_db: Path) -> pd.DataFrame:
    """Load all fixtures from matches.db and compute Elo features.

    Opens matches.db in SQLite read-only URI mode (structural guarantee
    that no write can occur).

    Returns:
        DataFrame with columns: fixture_id, home_elo, away_elo, elo_diff
    """
    uri = f"file:{Path(matches_db).resolve()}?mode=ro"
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

    return compute_elo_features(fixtures)


def get_elo_for_fixture(matches_db: Path, fixture_id: int) -> dict[str, float]:
    """Get Elo features for a single fixture.

    Computes the full Elo chain (required for causality) and returns
    the row matching the requested fixture_id.

    Returns:
        Dict with keys: home_elo, away_elo, elo_diff

    Raises:
        KeyError if fixture_id not found in matches.db
    """
    elo_df = load_elo_features(matches_db)
    match = elo_df.loc[elo_df["fixture_id"] == fixture_id]
    if match.empty:
        raise KeyError(
            f"fixture_id {fixture_id} not found in matches.db Elo computation. "
            "The fixture may not exist in matches.db."
        )
    row = match.iloc[0]
    return {
        "home_elo": float(row["home_elo"]),
        "away_elo": float(row["away_elo"]),
        "elo_diff": float(row["elo_diff"]),
    }


__all__ = [
    "ELO_COLUMNS",
    "INIT_RATING",
    "K_FACTOR",
    "HOME_ADVANTAGE",
    "compute_elo_features",
    "load_elo_features",
    "get_elo_for_fixture",
]
