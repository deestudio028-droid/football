"""Orchestrates one leakage-safe feature row per fixture.

The core loop processes every fixture in strict chronological order
(`unix` ascending, `fixture_id` as tiebreak). For each fixture:
  1. Read the home/away teams' accumulated history and the league's
     accumulated season state -- all of which, by construction, only
     contains fixtures processed in earlier iterations of this same loop.
  2. Compute every feature family from that state.
  3. Only THEN append the current fixture's own result to the team
     histories and league accumulator (`FeatureContext.record`).

Because step 3 always happens after step 1-2, and the loop is strictly
chronological, no feature for fixture T can ever be influenced by T
itself or by any fixture with a later kickoff time. This is the
architectural leakage guarantee that tests/test_feature_leakage.py
verifies empirically.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .config import FEATURE_VERSION, SUCCESSOR_FEATURE_VERSION, WINDOWS
from .context import home_advantage_features, sample_size_features
from .form import form_features
from .history import FeatureContext, load_fixtures_chronological
from .missingness import EXTRACTORS
from .rolling import gated_mean, venue_restricted_mean
from .strength import strength_features

# Families computed over last5/last10/season (goals, xg, shots, shots_on).
_ROLLING_FAMILIES = ["goals", "xg", "shots", "shots_on"]
# Families computed over season only (tempo/discipline proxies).
_SEASON_ONLY_FAMILIES = ["attacks", "dang_attacks", "pressure", "fouls", "yellow_cards"]

_FAMILY_TO_EXTRACTOR_SUFFIX = {
    "goals": ("goals_for", "goals_against"),
    "xg": ("xg_for", "xg_against"),
    "shots": ("shots_for", "shots_against"),
    "shots_on": ("shots_on_for", "shots_on_against"),
}
_SEASON_ONLY_EXTRACTOR = MappingProxyType({
    "attacks": "attacks_for",
    "dang_attacks": "dang_attacks_for",
    "pressure": "pressure_for",
    "fouls": "fouls_for",
    "yellow_cards": "yellow_cards_for",
})

# --- v1.1 successor registry (D-17 Option B, D-18) ---------------------
# This is the ONLY registry that differs between V1 and the successor,
# because it is the only one the builder ITERATES to decide which columns
# get emitted (see _season_only_family_features). Mutating the V1 dict in
# place would therefore have silently changed V1's output, which is
# exactly the hazard D-17 identified -- hence MappingProxyType above: an
# in-place mutation now raises instead of corrupting V1.
#
# Derived from the V1 registry rather than restated, so the five V1
# entries can never drift between versions, and dict insertion order
# guarantees the successor's first five entries are V1's, in V1's order.
_SEASON_ONLY_EXTRACTOR_V1_1 = MappingProxyType({
    **_SEASON_ONLY_EXTRACTOR,
    "possession": "possession_for",
    "corners": "corners_for",
    "red_cards": "red_cards_for",
})


def _rolling_family_features(history: list[dict[str, Any]], target_season_id: int, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in _ROLLING_FAMILIES:
        for_key, against_key = _FAMILY_TO_EXTRACTOR_SUFFIX[family]
        for_extractor = EXTRACTORS[for_key]
        against_extractor = EXTRACTORS[against_key]
        for window in WINDOWS:
            for_val, for_n, for_cov = gated_mean(history, window, target_season_id, for_extractor)
            out[f"{prefix}_{family}_for_per_match_{window}"] = for_val
            out[f"{prefix}_{family}_for_per_match_{window}_n"] = for_n
            out[f"{prefix}_{family}_for_per_match_{window}_coverage_n"] = for_cov

            against_val, against_n, against_cov = gated_mean(history, window, target_season_id, against_extractor)
            out[f"{prefix}_{family}_against_per_match_{window}"] = against_val
            out[f"{prefix}_{family}_against_per_match_{window}_n"] = against_n
            out[f"{prefix}_{family}_against_per_match_{window}_coverage_n"] = against_cov

            if for_val is not None and against_val is not None:
                out[f"{prefix}_{family}_diff_{window}"] = for_val - against_val
            else:
                out[f"{prefix}_{family}_diff_{window}"] = None
    return out


def _season_only_family_features(
    history: list[dict[str, Any]],
    target_season_id: int,
    prefix: str,
    extractor_registry: Mapping[str, str] = _SEASON_ONLY_EXTRACTOR,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family, extractor_key in extractor_registry.items():
        extractor = EXTRACTORS[extractor_key]
        val, n, cov = gated_mean(history, "season", target_season_id, extractor)
        out[f"{prefix}_{family}_per_match_season"] = val
        out[f"{prefix}_{family}_per_match_season_n"] = n
        out[f"{prefix}_{family}_per_match_season_coverage_n"] = cov
    return out


def _venue_features(history: list[dict[str, Any]], target_season_id: int, prefix: str, home_only: bool) -> dict[str, Any]:
    venue_label = "home_venue" if home_only else "away_venue"
    out: dict[str, Any] = {}
    for_val, for_n, for_cov = venue_restricted_mean(history, target_season_id, EXTRACTORS["goals_for"], home_only)
    against_val, against_n, against_cov = venue_restricted_mean(history, target_season_id, EXTRACTORS["goals_against"], home_only)
    out[f"{prefix}_goals_for_{venue_label}_season"] = for_val
    out[f"{prefix}_goals_for_{venue_label}_season_n"] = for_n
    out[f"{prefix}_goals_against_{venue_label}_season"] = against_val
    out[f"{prefix}_goals_against_{venue_label}_season_n"] = against_n
    return out


def build_feature_row(
    fixture: dict[str, Any],
    ctx: FeatureContext,
    *,
    _season_only_extractor: Mapping[str, str] = _SEASON_ONLY_EXTRACTOR,
    _feature_version: str = FEATURE_VERSION,
) -> dict[str, Any]:
    """Compute the full feature row for one fixture. Must be called
    BEFORE `ctx.record(fixture)` for this same fixture.

    The two underscore-prefixed keyword-only parameters exist solely so
    the v1.1 successor path can reuse this exact function instead of
    duplicating the leakage-sensitive construction (D-17). They default
    to the frozen V1 registry and "v1.0"; calling this function the way
    every historical caller calls it therefore yields V1 behaviour
    unchanged. They are not a supported public API -- the only
    non-default caller is build_successor_feature_dataset.
    """
    home_id, away_id = fixture["home_id"], fixture["away_id"]
    season_id = fixture["season_id"]

    home_history = ctx.history_before(home_id) if home_id is not None else []
    away_history = ctx.history_before(away_id) if away_id is not None else []
    league_acc = ctx.league_accumulator(fixture["competition_id"], season_id)

    row: dict[str, Any] = {
        "fixture_id": fixture["fixture_id"],
        "competition_id": fixture["competition_id"],
        "season_id": season_id,
        "unix": fixture["unix"],
        "home_id": home_id,
        "away_id": away_id,
        "feature_version": _feature_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # labels -- never to be used as inputs, see storage.py LABEL_COLUMNS
        "label_home_goals": fixture.get("home_goals"),
        "label_away_goals": fixture.get("away_goals"),
        "label_result": (
            "H" if (fixture.get("home_goals") or 0) > (fixture.get("away_goals") or 0)
            else "A" if (fixture.get("home_goals") or 0) < (fixture.get("away_goals") or 0)
            else "D"
        ) if fixture.get("home_goals") is not None and fixture.get("away_goals") is not None else None,
    }

    row.update(_rolling_family_features(home_history, season_id, "home"))
    row.update(_rolling_family_features(away_history, season_id, "away"))
    row.update(_season_only_family_features(home_history, season_id, "home", _season_only_extractor))
    row.update(_season_only_family_features(away_history, season_id, "away", _season_only_extractor))
    row.update(_venue_features(home_history, season_id, "home", home_only=True))
    row.update(_venue_features(away_history, season_id, "away", home_only=False))
    row.update(form_features(home_history, season_id, "home"))
    row.update(form_features(away_history, season_id, "away"))
    row.update(home_advantage_features(home_history, away_history, season_id, league_acc))
    row.update(sample_size_features(home_history, away_history))
    row.update(strength_features(home_history, away_history, season_id, league_acc))

    return row


def build_feature_dataset(db_path: Path, fixture_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Process the entire dataset (or, if `fixture_ids` is given, still
    walk the ENTIRE chronological sequence but only keep rows for the
    requested fixtures -- this is deliberate: history for fixture #9000
    still needs fixtures #1-8999 to have been accumulated, even if we
    only want to inspect a handful of rows, e.g. for the validation
    sample in step 11).
    """
    return _walk_and_build(
        db_path, fixture_ids,
        season_only_extractor=_SEASON_ONLY_EXTRACTOR,
        feature_version=FEATURE_VERSION,
    )


def _walk_and_build(
    db_path: Path,
    fixture_ids: list[int] | None,
    *,
    season_only_extractor: Mapping[str, str],
    feature_version: str,
) -> list[dict[str, Any]]:
    """The single chronological walk shared by V1 and the successor.

    Private on purpose: it is the one place that takes a registry, so
    there is exactly one route by which a version choice can be made,
    and it is not reachable from the historical public API.
    """
    fixtures = load_fixtures_chronological(db_path)
    ctx = FeatureContext()
    wanted = set(fixture_ids) if fixture_ids is not None else None
    rows = []
    for fixture in fixtures:
        if wanted is None or fixture["fixture_id"] in wanted:
            rows.append(build_feature_row(
                fixture, ctx,
                _season_only_extractor=season_only_extractor,
                _feature_version=feature_version,
            ))
        # Always AFTER build_feature_row for this same fixture: this is the
        # leakage guarantee, and it is shared by both versions because both
        # versions run this identical loop.
        ctx.record(fixture)
    return rows


def build_successor_feature_dataset(
    db_path: Path, fixture_ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """v1.1 successor feature construction (D-14/D-15/D-17/D-18).

    Identical to `build_feature_dataset` except that it selects the
    v1.1 registry and stamps SUCCESSOR_FEATURE_VERSION, so it emits the
    V1 columns plus the three additive season-only families
    (possession, corners, red_cards) for both sides.

    `db_path` is the READ-ONLY matches source, exactly as for the V1
    builder. Like the V1 builder, this function writes nothing and has
    no output-path parameter, so it cannot write to (or default to)
    data/processed/features.db. Persisting these rows is a separate,
    currently unauthorized step; the caller must supply its own
    destination via features.storage.FeatureDB.

    V1 is untouched: this does not mutate _SEASON_ONLY_EXTRACTOR, does
    not read or change FEATURE_VERSION, MODEL_VERSION,
    REQUIRED_FEATURE_VERSION, or MODEL_B_COLUMNS.
    """
    return _walk_and_build(
        db_path, fixture_ids,
        season_only_extractor=_SEASON_ONLY_EXTRACTOR_V1_1,
        feature_version=SUCCESSOR_FEATURE_VERSION,
    )
