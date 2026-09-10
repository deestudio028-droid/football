"""Phase 4A: feature ablation study infrastructure (Model A/B/C/D).

STATUS: tier mapping FROZEN, xG fold-eligibility policy FROZEN (both
resolved by explicit decision -- see
docs/PHASE4A_ABLATION_DESIGN_DIAGNOSTIC_REPORT.md for the diagnostic
that prompted them, and docs/PHASE4A_FEATURE_ABLATION_REPORT.md for
results).

DECISION 1 -- xG handling (option (c)): Models C/D are an
xG-coverage-specific experiment, evaluated ONLY on folds whose
TRAINING partition has real observed xG values. With the current
dataset that means fold_3 only; fold_1 and fold_2 have 100% missing xG
across all 22 xG columns in their training partitions, so no imputation
can be *learned* from them. Such folds are marked INVALID_FOR_XG and
recorded explicitly with a reason -- never silently skipped, never
filled with zero/mean/placeholder values, and never dropped as a
feature group. Consequence, stated plainly: **C/D cannot be ranked
against A/B on the 3-fold mean**, because folds 1-2 are structurally
invalid for xG-inclusive training. Any "3-fold mean" for C/D would be
fabricated, and this module refuses to produce one.

DECISION 2 -- tier composition, frozen exactly as proposed:
  A = goals_core + form + competition_id + strength
  B = A + shots_core + shots_on_core
  C = B + xg_core
  D = C + goals_venue + tempo + discipline + league_home_advantage
          + league_mean_goals + venue_goal_diff + history_flags

Note (documented consequence of the frozen mapping, not a bug): Model A
is NOT a superset of V1's approved 15-column feature set --
`league_home_advantage_season` is a V1-approved column that the frozen
mapping places at tier D, so it is absent from A, B and C. Every other
V1-approved column is present in Model A. See
tests/test_model_ablation.py::test_model_a_contains_every_v1_column_except_league_home_advantage.

Nothing here modifies `config.APPROVED_FEATURE_COLUMNS_V1`,
`config.WALK_FORWARD_FOLDS`, `config.MODEL_VERSION`,
`config.REQUIRED_FEATURE_VERSION`, `config.FINAL_TEST_SEASONS`, or any
other V1 contract item. This module is purely additive; V1 remains
frozen.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------
# Schema-verified column groups. Every column listed here was confirmed
# to exist in data/processed/features.db's feature_rows table (via
# models.data.get_feature_columns) at the time this module was written
# -- see tests/test_model_ablation.py::TestFeatureGroupColumnsExistInSchema
# for the mechanical, re-runnable proof. "_n" and "_coverage_n" companion
# (sample-size / coverage-count) columns are deliberately excluded from
# every group below, consistent with how V1's own approved feature set
# (config.APPROVED_FEATURE_COLUMNS_V1) only uses rate/score-level
# columns, never their raw sample-size companions.
# ---------------------------------------------------------------------

GOALS_CORE_COLUMNS: tuple[str, ...] = (
    "home_goals_for_per_match_last5", "home_goals_against_per_match_last5", "home_goals_diff_last5",
    "home_goals_for_per_match_last10", "home_goals_against_per_match_last10", "home_goals_diff_last10",
    "home_goals_for_per_match_season", "home_goals_against_per_match_season", "home_goals_diff_season",
    "away_goals_for_per_match_last5", "away_goals_against_per_match_last5", "away_goals_diff_last5",
    "away_goals_for_per_match_last10", "away_goals_against_per_match_last10", "away_goals_diff_last10",
    "away_goals_for_per_match_season", "away_goals_against_per_match_season", "away_goals_diff_season",
)  # 18 columns

FORM_COLUMNS: tuple[str, ...] = (
    "home_points_last5", "home_win_rate_last5", "home_draw_rate_last5", "home_loss_rate_last5", "home_goal_diff_last5",
    "home_points_last10", "home_win_rate_last10", "home_draw_rate_last10", "home_loss_rate_last10", "home_goal_diff_last10",
    "away_points_last5", "away_win_rate_last5", "away_draw_rate_last5", "away_loss_rate_last5", "away_goal_diff_last5",
    "away_points_last10", "away_win_rate_last10", "away_draw_rate_last10", "away_loss_rate_last10", "away_goal_diff_last10",
)  # 20 columns

SHOTS_CORE_COLUMNS: tuple[str, ...] = (
    "home_shots_for_per_match_last5", "home_shots_against_per_match_last5", "home_shots_diff_last5",
    "home_shots_for_per_match_last10", "home_shots_against_per_match_last10", "home_shots_diff_last10",
    "home_shots_for_per_match_season", "home_shots_against_per_match_season", "home_shots_diff_season",
    "away_shots_for_per_match_last5", "away_shots_against_per_match_last5", "away_shots_diff_last5",
    "away_shots_for_per_match_last10", "away_shots_against_per_match_last10", "away_shots_diff_last10",
    "away_shots_for_per_match_season", "away_shots_against_per_match_season", "away_shots_diff_season",
)  # 18 columns

SHOTS_ON_CORE_COLUMNS: tuple[str, ...] = (
    "home_shots_on_for_per_match_last5", "home_shots_on_against_per_match_last5", "home_shots_on_diff_last5",
    "home_shots_on_for_per_match_last10", "home_shots_on_against_per_match_last10", "home_shots_on_diff_last10",
    "home_shots_on_for_per_match_season", "home_shots_on_against_per_match_season", "home_shots_on_diff_season",
    "away_shots_on_for_per_match_last5", "away_shots_on_against_per_match_last5", "away_shots_on_diff_last5",
    "away_shots_on_for_per_match_last10", "away_shots_on_against_per_match_last10", "away_shots_on_diff_last10",
    "away_shots_on_for_per_match_season", "away_shots_on_against_per_match_season", "away_shots_on_diff_season",
)  # 18 columns

XG_CORE_COLUMNS: tuple[str, ...] = (
    "home_xg_for_per_match_last5", "home_xg_against_per_match_last5", "home_xg_diff_last5",
    "home_xg_for_per_match_last10", "home_xg_against_per_match_last10", "home_xg_diff_last10",
    "home_xg_for_per_match_season", "home_xg_against_per_match_season", "home_xg_diff_season",
    "away_xg_for_per_match_last5", "away_xg_against_per_match_last5", "away_xg_diff_last5",
    "away_xg_for_per_match_last10", "away_xg_against_per_match_last10", "away_xg_diff_last10",
    "away_xg_for_per_match_season", "away_xg_against_per_match_season", "away_xg_diff_season",
    "home_xg_diff_form_last5", "home_xg_diff_form_last10", "away_xg_diff_form_last5", "away_xg_diff_form_last10",
)  # 22 columns -- 100% NULL in fold_1 and fold_2 training partitions, see module docstring item 2

STRENGTH_COLUMNS: tuple[str, ...] = (
    "home_attack_strength_score", "home_defence_strength_score",
    "away_attack_strength_score", "away_defence_strength_score", "strength_diff",
)  # 5 columns -- part of V1's approved set; NOT assigned to any tier in the design report's A-D prose

COMPETITION_ID_COLUMN: tuple[str, ...] = ("competition_id",)  # not assigned to any tier in the design report's A-D prose

LEAGUE_HOME_ADVANTAGE_COLUMN: tuple[str, ...] = ("league_home_advantage_season",)  # design report explicitly places this at Model D only

LEAGUE_MEAN_GOALS_COLUMN: tuple[str, ...] = ("league_mean_goals_per_team_match_season",)  # not assigned to any tier

GOALS_VENUE_COLUMNS: tuple[str, ...] = (
    "home_goals_for_home_venue_season", "home_goals_against_home_venue_season",
    "away_goals_for_away_venue_season", "away_goals_against_away_venue_season",
)  # 4 columns -- design report explicitly places "venue-split goals" at Model D

TEMPO_COLUMNS: tuple[str, ...] = (
    "home_attacks_per_match_season", "home_dang_attacks_per_match_season", "home_pressure_per_match_season",
    "away_attacks_per_match_season", "away_dang_attacks_per_match_season", "away_pressure_per_match_season",
)  # 6 columns -- design report's "tempo... proxies", placed at Model D

DISCIPLINE_COLUMNS: tuple[str, ...] = (
    "home_fouls_per_match_season", "home_yellow_cards_per_match_season",
    "away_fouls_per_match_season", "away_yellow_cards_per_match_season",
)  # 4 columns -- design report's "discipline proxies", placed at Model D

VENUE_GOAL_DIFF_COLUMNS: tuple[str, ...] = (
    "home_team_home_goal_diff_season", "away_team_away_goal_diff_season",
)  # 2 columns -- not assigned to any tier in the design report's A-D prose

HISTORY_FLAG_COLUMNS: tuple[str, ...] = (
    "home_matches_played_before_target", "away_matches_played_before_target",
    "home_insufficient_history", "away_insufficient_history",
)  # 4 columns -- not assigned to any tier in the design report's A-D prose

# shrinkage_k is deliberately excluded from every group: it is a fixed
# hyperparameter recorded on every row (constant value = 5 across the
# entire dataset, confirmed in data/audit/phase4a_feature_group_diagnostic.json),
# not a per-match signal -- a constant column carries zero information
# for any linear or tree-based classifier.

ALL_SCHEMA_VERIFIED_GROUPS: dict[str, tuple[str, ...]] = {
    "goals_core": GOALS_CORE_COLUMNS,
    "goals_venue": GOALS_VENUE_COLUMNS,
    "form": FORM_COLUMNS,
    "shots_core": SHOTS_CORE_COLUMNS,
    "shots_on_core": SHOTS_ON_CORE_COLUMNS,
    "xg_core": XG_CORE_COLUMNS,
    "strength": STRENGTH_COLUMNS,
    "competition_id": COMPETITION_ID_COLUMN,
    "league_home_advantage": LEAGUE_HOME_ADVANTAGE_COLUMN,
    "league_mean_goals": LEAGUE_MEAN_GOALS_COLUMN,
    "tempo": TEMPO_COLUMNS,
    "discipline": DISCIPLINE_COLUMNS,
    "venue_goal_diff": VENUE_GOAL_DIFF_COLUMNS,
    "history_flags": HISTORY_FLAG_COLUMNS,
}


# ---------------------------------------------------------------------
# FROZEN tier compositions (Decision 2). The 14 columns the Phase 3
# design report's A-D prose never assigned to a tier are resolved as:
# competition_id and strength start at Model A (neither is itself an
# ablated dimension, and strength is goals-derived); league_home_advantage,
# goals_venue, tempo and discipline follow the design report's explicit
# placement at Model D; league_mean_goals, venue_goal_diff and
# history_flags land at Model D as part of "every V1-eligible feature."
# Each tier is a strict superset of the previous one, so every A->B->C->D
# comparison isolates exactly one added feature dimension.
# ---------------------------------------------------------------------
MODEL_A_COLUMNS: tuple[str, ...] = GOALS_CORE_COLUMNS + FORM_COLUMNS + STRENGTH_COLUMNS + COMPETITION_ID_COLUMN
MODEL_B_COLUMNS: tuple[str, ...] = MODEL_A_COLUMNS + SHOTS_CORE_COLUMNS + SHOTS_ON_CORE_COLUMNS
MODEL_C_COLUMNS: tuple[str, ...] = MODEL_B_COLUMNS + XG_CORE_COLUMNS
MODEL_D_COLUMNS: tuple[str, ...] = (
    MODEL_C_COLUMNS + GOALS_VENUE_COLUMNS + TEMPO_COLUMNS + DISCIPLINE_COLUMNS
    + LEAGUE_HOME_ADVANTAGE_COLUMN + LEAGUE_MEAN_GOALS_COLUMN + VENUE_GOAL_DIFF_COLUMNS + HISTORY_FLAG_COLUMNS
)

MODEL_A = "model_a"
MODEL_B = "model_b"
MODEL_C = "model_c"
MODEL_D = "model_d"
ABLATION_MODELS = (MODEL_A, MODEL_B, MODEL_C, MODEL_D)

MODEL_COLUMNS: dict[str, tuple[str, ...]] = {
    MODEL_A: MODEL_A_COLUMNS,
    MODEL_B: MODEL_B_COLUMNS,
    MODEL_C: MODEL_C_COLUMNS,
    MODEL_D: MODEL_D_COLUMNS,
}

MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    MODEL_A: ("goals_core", "form", "competition_id", "strength"),
    MODEL_B: ("goals_core", "form", "competition_id", "strength", "shots_core", "shots_on_core"),
    MODEL_C: ("goals_core", "form", "competition_id", "strength", "shots_core", "shots_on_core", "xg_core"),
    MODEL_D: (
        "goals_core", "form", "competition_id", "strength", "shots_core", "shots_on_core", "xg_core",
        "goals_venue", "tempo", "discipline", "league_home_advantage", "league_mean_goals",
        "venue_goal_diff", "history_flags",
    ),
}

# Models whose feature set includes the xG group, and which are
# therefore subject to the fold-eligibility guard below.
XG_INCLUSIVE_MODELS = (MODEL_C, MODEL_D)


def select_feature_columns(X: pd.DataFrame, columns: tuple[str, ...], group_name: str = "") -> pd.DataFrame:
    """Restricts a feature matrix to exactly `columns`, in that order,
    failing loudly (never substituting or silently dropping) if any is
    missing from `X` -- same leakage-safe pattern as
    run_experiments._select_approved_features().
    """
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise RuntimeError(
            f"Feature group {group_name!r} references column(s) not present in the "
            f"feature matrix: {missing}. Refusing to silently substitute or drop them."
        )
    return X[list(columns)]


# ---------------------------------------------------------------------
# xG fold-eligibility guard (Decision 1).
# ---------------------------------------------------------------------
class XgCoverageError(RuntimeError):
    """Raised when an xG-inclusive model is requested for a fold whose
    TRAINING partition has no observed values for a required xG column.
    Deliberately a hard failure: a caller must never be able to train
    C/D on such a fold, and must never silently skip it either -- the
    experiment runner catches this explicitly and records the fold as
    INVALID_FOR_XG with this message as the recorded reason.
    """


@dataclass(frozen=True)
class XgFoldEligibility:
    fold_name: str
    is_valid: bool
    reason: str | None
    xg_columns_total: int
    xg_columns_with_zero_training_observations: tuple[str, ...]
    min_observed_count_across_xg_columns: int
    max_null_rate_across_xg_columns: float

    def to_dict(self) -> dict:
        return {
            "fold_name": self.fold_name,
            "xg_training_observed": self.is_valid,
            "is_valid_for_xg": self.is_valid,
            "invalid_reason": self.reason,
            "xg_columns_total": self.xg_columns_total,
            "xg_columns_with_zero_training_observations": list(
                self.xg_columns_with_zero_training_observations
            ),
            "min_observed_count_across_xg_columns": self.min_observed_count_across_xg_columns,
            "max_null_rate_across_xg_columns": self.max_null_rate_across_xg_columns,
        }


def assess_xg_fold_eligibility(
    fold_name: str, X_train: pd.DataFrame, xg_columns: tuple[str, ...] = XG_CORE_COLUMNS
) -> XgFoldEligibility:
    """Decides whether a fold's TRAINING partition can support
    xG-inclusive training, using only that fold's own training data.

    A fold is INVALID_FOR_XG if ANY required xG column has zero observed
    (non-null) values in the training partition -- deliberately stricter
    than "all xG columns are empty". Rationale: `SimpleImputer` cannot
    learn a statistic for an all-NaN column and drops it, which would
    silently remove part of the xG feature group -- exactly the
    forbidden behaviour. Failing the whole fold instead keeps the
    xG group intact-or-absent, never partially and invisibly gutted.
    """
    zero_observation_columns = tuple(c for c in xg_columns if int(X_train[c].notna().sum()) == 0)
    observed_counts = [int(X_train[c].notna().sum()) for c in xg_columns]
    null_rates = [float(X_train[c].isna().mean()) for c in xg_columns]

    if zero_observation_columns:
        reason = (
            f"INVALID_FOR_XG: {len(zero_observation_columns)} of {len(xg_columns)} required xG "
            f"columns have ZERO observed values in {fold_name}'s training partition "
            f"(e.g. {zero_observation_columns[0]}). No imputation statistic can be learned from "
            f"an entirely-empty training column, and filling one would require fabricating a "
            f"value (forbidden). This fold is structurally unusable for xG-inclusive training; "
            f"it is recorded as invalid rather than skipped or filled."
        )
        return XgFoldEligibility(
            fold_name=fold_name, is_valid=False, reason=reason,
            xg_columns_total=len(xg_columns),
            xg_columns_with_zero_training_observations=zero_observation_columns,
            min_observed_count_across_xg_columns=min(observed_counts) if observed_counts else 0,
            max_null_rate_across_xg_columns=max(null_rates) if null_rates else 1.0,
        )

    return XgFoldEligibility(
        fold_name=fold_name, is_valid=True, reason=None,
        xg_columns_total=len(xg_columns),
        xg_columns_with_zero_training_observations=(),
        min_observed_count_across_xg_columns=min(observed_counts) if observed_counts else 0,
        max_null_rate_across_xg_columns=max(null_rates) if null_rates else 0.0,
    )


def require_xg_fold_valid(
    fold_name: str, X_train: pd.DataFrame, xg_columns: tuple[str, ...] = XG_CORE_COLUMNS
) -> XgFoldEligibility:
    """Hard guard: raises XgCoverageError if the fold cannot support
    xG-inclusive training. Use this at the point of training an
    xG-inclusive model, so no code path can reach a `fit()` call on a
    fold with unusable xG coverage.
    """
    eligibility = assess_xg_fold_eligibility(fold_name, X_train, xg_columns)
    if not eligibility.is_valid:
        raise XgCoverageError(eligibility.reason)
    return eligibility
