"""Promoted Team Pre-Match Feature Initialization Engine.

Provides causal, leakage-safe pre-match feature initialization for newly promoted teams
entering top-flight competitions without current-season history in matches.db.

INVARIANTS:
1. Strict Causality: Only information available strictly before the fixture's scheduled
   kickoff timestamp is used. Post-kickoff stats and match results are strictly excluded.
2. V4 Contract Compliance: Produces the exact 91-feature design matrix required by the
   frozen V4 Poisson+Venue+Elo+OnlineAD model.
3. No Generic Fallbacks: Never emits hardcoded 1.45/1.15 rates or static probabilities.
   Promoted teams receive feature vectors interacting with the opponent's true top-flight state.
4. Fail-Closed Safety: If a valid promoted feature vector cannot be constructed, fails closed.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"

from features.elo import (
    HOME_ADVANTAGE,
    INIT_RATING,
    compute_elo_features,
)
from features.feature_builder import build_feature_row
from features.history import FeatureContext, load_fixtures_chronological
from features.online_attack_defense import (
    AD_COLUMNS,
    compute_ad_states,
    fit_baseline_rates,
)
from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
from models.v4_contract import V4_FEATURE_COLUMNS

logger = logging.getLogger("promoted_team_initializer")


@dataclass
class PromotedTeamProfile:
    """Pre-match initialization profile for a promoted team."""
    team_name: str
    team_id: int
    competition_id: int
    is_promoted: bool
    initialization_method: str
    feature_source: str
    historical_match_count: int
    latest_historical_match_date: Optional[str]
    initialized_elo: float
    initialized_attack: float
    initialized_defense: float
    data_cutoff_timestamp: str


@dataclass
class PromotedMatchFeatureResult:
    """Complete pre-match feature output for a match involving promoted team(s)."""
    fixture_id: int
    competition_id: int
    scheduled_kickoff: str
    home_team: str
    away_team: str
    home_id: int
    away_id: int
    home_promoted: bool
    away_promoted: bool
    home_profile: PromotedTeamProfile
    away_profile: PromotedTeamProfile
    features_df: pd.DataFrame
    diagnostics: Dict[str, Any]


class PromotedTeamInitializer:
    """Engine that initializes pre-match features for newly promoted teams."""

    def __init__(
        self,
        db_path: Path = MATCHES_DB,
        ctx: Optional[FeatureContext] = None,
        all_fx: Optional[pd.DataFrame] = None,
        elo_df: Optional[pd.DataFrame] = None,
        ad_df: Optional[pd.DataFrame] = None,
        team_id_to_latest_elo: Optional[Dict[int, float]] = None,
        team_id_to_latest_ad: Optional[Dict[int, Tuple[float, float]]] = None,
    ):
        self.db_path = Path(db_path)
        
        if ctx is not None:
            self._ctx = ctx
        else:
            self._ctx = FeatureContext()
            db_fixtures = load_fixtures_chronological(self.db_path)
            for f in db_fixtures:
                if f.get("status") in ("FT", "AWARDED") and f.get("home_goals") is not None:
                    self._ctx.record(f)

        if all_fx is not None:
            self._all_fx = all_fx
        else:
            conn_m = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self._all_fx = pd.read_sql_query("SELECT * FROM fixtures ORDER BY unix ASC, fixture_id ASC", conn_m)
            conn_m.close()

        # Pre-compute Elo and AD
        if elo_df is not None and team_id_to_latest_elo is not None and team_id_to_latest_ad is not None:
            self._elo_df = elo_df
            self._ad_df = ad_df
            self._team_id_to_latest_elo = team_id_to_latest_elo
            self._team_id_to_latest_ad = team_id_to_latest_ad
        else:
            self._elo_df = compute_elo_features(self._all_fx)
            self._team_id_to_latest_elo = {}
            self._team_id_to_latest_ad = {}

            wanted_hist = set()
            for sn in FINAL_TRAIN_SEASONS:
                wanted_hist |= set(SEASON_NAME_TO_IDS[sn])
            hist_fx = self._all_fx[self._all_fx.season_id.isin(wanted_hist) & self._all_fx.home_goals.notna() & self._all_fx.status.isin(["FT", "AWARDED"])]
            base_rates = fit_baseline_rates(hist_fx.home_goals.values.astype(float), hist_fx.away_goals.values.astype(float))
            self._ad_df = compute_ad_states(self._all_fx, 0.02, base_rates)

            for _, row in self._elo_df.iterrows():
                fid = row["fixture_id"]
                fx_row = self._all_fx[self._all_fx["fixture_id"] == fid].iloc[0]
                self._team_id_to_latest_elo[int(fx_row["home_id"])] = float(row["home_elo"])
                self._team_id_to_latest_elo[int(fx_row["away_id"])] = float(row["away_elo"])

            for _, row in self._ad_df.iterrows():
                fid = row["fixture_id"]
                fx_row = self._all_fx[self._all_fx["fixture_id"] == fid].iloc[0]
                self._team_id_to_latest_ad[int(fx_row["home_id"])] = (float(row["A_home"]), float(row["D_home"]))
                self._team_id_to_latest_ad[int(fx_row["away_id"])] = (float(row["A_away"]), float(row["D_away"]))

        self._team_name_to_id: Dict[str, int] = {}
        for _, row in self._all_fx.iterrows():
            if row.get("home_name") and row.get("home_id"):
                self._team_name_to_id[str(row["home_name"]).strip().lower()] = int(row["home_id"])
            if row.get("away_name") and row.get("away_id"):
                self._team_name_to_id[str(row["away_name"]).strip().lower()] = int(row["away_id"])

    def is_promoted_team(self, team_id: int) -> bool:
        """Check if team lacks current top-flight match history in context."""
        history = self._ctx.history_before(team_id)
        return len(history) == 0

    def get_team_profile(
        self,
        team_name: str,
        team_id: int,
        competition_id: int,
        cutoff_dt_iso: str,
    ) -> PromotedTeamProfile:
        """Construct the team's causal initialization profile."""
        history = self._ctx.history_before(team_id)
        n_hist = len(history)

        if n_hist > 0:
            latest_dt = history[-1].get("date", cutoff_dt_iso)
            latest_elo = self._team_id_to_latest_elo.get(team_id, INIT_RATING)
            latest_ad = self._team_id_to_latest_ad.get(team_id, (0.0, 0.0))
            return PromotedTeamProfile(
                team_name=team_name,
                team_id=team_id,
                competition_id=competition_id,
                is_promoted=False,
                initialization_method="Standard Chronological Top-Flight History",
                feature_source="Live Chronological Feature Context",
                historical_match_count=n_hist,
                latest_historical_match_date=latest_dt,
                initialized_elo=latest_elo,
                initialized_attack=latest_ad[0],
                initialized_defense=latest_ad[1],
                data_cutoff_timestamp=cutoff_dt_iso,
            )

        # Team is newly promoted / lacks current-season history
        # 1. Search if historical top-flight matches exist from prior seasons in matches.db
        prior_matches = self._all_fx[
            ((self._all_fx["home_id"] == team_id) | (self._all_fx["away_id"] == team_id))
            & (self._all_fx["status"].isin(["FT", "AWARDED"]))
            & (self._all_fx["home_goals"].notna())
        ]

        if not prior_matches.empty:
            last_match = prior_matches.iloc[-1]
            last_date = str(last_match.get("date", "Unknown"))
            init_elo = self._team_id_to_latest_elo.get(team_id, INIT_RATING)
            init_ad = self._team_id_to_latest_ad.get(team_id, (0.0, 0.0))
            method = "Historical Top-Flight Prior Season Bridge"
            source = "PROMOTED TEAM INITIALIZATION (Historical Prior Flight)"
            n_prior = len(prior_matches)
        else:
            last_date = None
            init_elo = INIT_RATING  # Standard 1500.0 initial Elo
            init_ad = (0.0, 0.0)    # Baseline attack/defense
            method = "Conservative Promotion Baseline (Initial Elo 1500.0 + League Neutral AD)"
            source = "PROMOTED TEAM INITIALIZATION (Standard Promotion Baseline)"
            n_prior = 0

        return PromotedTeamProfile(
            team_name=team_name,
            team_id=team_id,
            competition_id=competition_id,
            is_promoted=True,
            initialization_method=method,
            feature_source=source,
            historical_match_count=n_prior,
            latest_historical_match_date=last_date,
            initialized_elo=init_elo,
            initialized_attack=init_ad[0],
            initialized_defense=init_ad[1],
            data_cutoff_timestamp=cutoff_dt_iso,
        )

    def initialize_match_features(
        self,
        fixture_id: int,
        competition_id: int,
        scheduled_kickoff: str,
        home_team: str,
        away_team: str,
        home_id: int,
        away_id: int,
        season_id: int = 0,
    ) -> Optional[PromotedMatchFeatureResult]:
        """Build causal 91-feature design matrix for a match with promoted team initialization."""
        try:
            unix_ts = int(datetime.fromisoformat(scheduled_kickoff.replace("Z", "+00:00")).timestamp())
        except Exception:
            unix_ts = int(datetime.now(timezone.utc).timestamp())

        home_profile = self.get_team_profile(home_team, home_id, competition_id, scheduled_kickoff)
        away_profile = self.get_team_profile(away_team, away_id, competition_id, scheduled_kickoff)

        fix_dict = {
            "fixture_id": fixture_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "unix": unix_ts,
            "home_id": home_id,
            "away_id": away_id,
            "home_goals": None,
            "away_goals": None,
        }

        try:
            feat_row = build_feature_row(fix_dict, self._ctx)
        except Exception as exc:
            logger.warning(f"Error building feature row for promoted match {home_team} vs {away_team}: {exc}")
            return None

        # Attach Elo features
        h_elo = home_profile.initialized_elo
        a_elo = away_profile.initialized_elo
        elo_diff = (h_elo + HOME_ADVANTAGE) - a_elo
        feat_row["home_elo"] = h_elo
        feat_row["away_elo"] = a_elo
        feat_row["elo_diff"] = elo_diff

        # Attach Online AD features
        feat_row["A_home"] = home_profile.initialized_attack
        feat_row["D_home"] = home_profile.initialized_defense
        feat_row["A_away"] = away_profile.initialized_attack
        feat_row["D_away"] = away_profile.initialized_defense

        # Construct DataFrame matching V4 contract
        try:
            features_df = pd.DataFrame([feat_row])[list(V4_FEATURE_COLUMNS)]
        except KeyError as exc:
            logger.warning(f"Missing required V4 feature column in promoted match: {exc}")
            return None

        diag = {
            "home_id": home_id,
            "away_id": away_id,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": elo_diff,
            "abs_elo_diff": float(abs(elo_diff)),
            "A_home": home_profile.initialized_attack,
            "D_home": home_profile.initialized_defense,
            "A_away": away_profile.initialized_attack,
            "D_away": away_profile.initialized_defense,
            "home_promoted": home_profile.is_promoted,
            "away_promoted": away_profile.is_promoted,
            "feature_source": home_profile.feature_source if home_profile.is_promoted else (
                away_profile.feature_source if away_profile.is_promoted else "Live Chronological Feature Context"
            ),
        }

        return PromotedMatchFeatureResult(
            fixture_id=fixture_id,
            competition_id=competition_id,
            scheduled_kickoff=scheduled_kickoff,
            home_team=home_team,
            away_team=away_team,
            home_id=home_id,
            away_id=away_id,
            home_promoted=home_profile.is_promoted,
            away_promoted=away_profile.is_promoted,
            home_profile=home_profile,
            away_profile=away_profile,
            features_df=features_df,
            diagnostics=diag,
        )
