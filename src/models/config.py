"""Constants for the V1 model data/split layer. Every boundary here is
grounded in docs/PHASE3_MODEL_DEVELOPMENT_DESIGN_REPORT.md sections 5-6
(temporal split strategy) -- change one, change the other, and bump
MODEL_VERSION if the change would alter what a trained model actually
sees.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MODEL_VERSION = "v1.0"

# The only feature_version this model layer will accept. data.py fails
# loudly (does not silently proceed) if features.db contains anything
# else, or more than one value at once. See docs/PHASE3_MODEL_DEVELOPMENT_DESIGN_REPORT.md §1.
REQUIRED_FEATURE_VERSION = "v1.0"

# The only legal values for label_result. Anything else in the data is
# a data-integrity problem to fail loudly on, not silently drop or coerce.
ALLOWED_LABELS = ("H", "D", "A")

# Re-exported from features.storage rather than redefined here, so the
# two packages can never drift out of sync about what a "label" or
# "metadata" column is. See src/features/storage.py for the source of
# truth and its own rationale.
from features.storage import LABEL_COLUMNS, METADATA_COLUMNS  # noqa: E402

# Columns that must never enter the model feature matrix X, beyond the
# label columns themselves (design report, "feature matrix rules").
# `season_id` and `unix` are deliberately EXCLUDED from X by default --
# they're split/audit metadata, not features a match-result model
# should be trained on (a model that learns "season_id=6484 predicts
# home win" would be learning a leakage-adjacent shortcut, not football).
# `competition_id` is deliberately KEPT as a candidate feature (the
# design report's recommended league-context input) -- it is NOT in
# this exclusion list.
X_EXCLUDED_COLUMNS = list(LABEL_COLUMNS) + [
    "fixture_id", "generated_at", "feature_version", "season_id", "unix",
    "home_id", "away_id",
]

# Recommended-but-not-mandatory context feature, called out explicitly
# per the design report so callers don't have to guess whether
# competition_id belongs in X.
RECOMMENDED_CONTEXT_FEATURE = "competition_id"

# The exact 15-column feature set approved in the Phase 3 design report's
# Step 2 sanity check (14 model features + competition_id). This is the
# ONLY feature set Step 2/3 candidate models (LogisticRegression,
# HistGradientBoosting) may be trained on -- see
# docs/PHASE3_STEP3_DIAGNOSTIC_REPORT.md for why: the full 264-column
# feature_rows table includes xG feature groups that are structurally
# all-null in early walk-forward training folds (pre-xG-cutover), and
# training on them crashes HistGradientBoostingClassifier's binning step.
# xG features belong only to the (not-yet-approved) Model C/D ablation
# variants, never to this approved subset. Column names verified to
# exist verbatim in data/processed/features.db during Step 2 -- if any
# of these is ever missing, callers must fail loudly, not substitute or
# invent a column.
APPROVED_FEATURE_COLUMNS_V1: tuple[str, ...] = (
    "home_goals_for_per_match_season",
    "home_goals_against_per_match_season",
    "away_goals_for_per_match_season",
    "away_goals_against_per_match_season",
    "home_goals_for_per_match_last5",
    "away_goals_for_per_match_last5",
    "home_points_last5",
    "away_points_last5",
    "home_attack_strength_score",
    "home_defence_strength_score",
    "away_attack_strength_score",
    "away_defence_strength_score",
    "strength_diff",
    "league_home_advantage_season",
    RECOMMENDED_CONTEXT_FEATURE,
)

# Fixture(s) known to have no valid result and therefore no label.
# Explicit, not inferred, per docs/PHASE3_MODEL_DEVELOPMENT_DESIGN_REPORT.md §4.
KNOWN_UNLABELED_FIXTURE_IDS = (420450481,)  # Nantes vs Toulouse, ABANDONED


def _project_root() -> Path:
    # src/models/config.py -> src/models -> src -> project root
    return Path(__file__).resolve().parents[2]


def _load_season_name_to_ids(project_root: Path | None = None) -> dict[str, list[int]]:
    """Maps a season name (e.g. "2020/2021") to every season_id that
    represents it across all 5 competitions -- one match on the
    calendar season spans 5 distinct season_id values, one per league.
    """
    root = project_root or _project_root()
    raw = json.loads((root / "config" / "competitions.json").read_text(encoding="utf-8"))
    mapping: dict[str, list[int]] = {}
    for comp in raw["competitions"]:
        for season in comp["seasons"]:
            mapping.setdefault(season["season_name"], []).append(season["season_id"])
    return mapping


SEASON_NAME_TO_IDS = _load_season_name_to_ids()

# Chronological order, confirmed against the ingested dataset in the
# design report (§1/§5) -- all 6 seasons are complete, no partial season.
SEASON_ORDER = [
    "2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025", "2025/2026",
]


@dataclass(frozen=True)
class Fold:
    name: str
    train_seasons: tuple[str, ...]
    validation_seasons: tuple[str, ...]


# Walk-forward folds, exactly as approved in
# docs/PHASE3_MODEL_DEVELOPMENT_DESIGN_REPORT.md §5. Model selection
# (a later step) may only use these three folds -- never the final test season.
WALK_FORWARD_FOLDS = (
    Fold("fold_1", ("2020/2021", "2021/2022"), ("2022/2023",)),
    Fold("fold_2", ("2020/2021", "2021/2022", "2022/2023"), ("2023/2024",)),
    Fold("fold_3", ("2020/2021", "2021/2022", "2022/2023", "2023/2024"), ("2024/2025",)),
)

# The untouched final test split. 2025/26 must never appear in
# WALK_FORWARD_FOLDS above.
FINAL_TRAIN_SEASONS: tuple[str, ...] = (
    "2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025",
)
FINAL_TEST_SEASONS: tuple[str, ...] = ("2025/2026",)
