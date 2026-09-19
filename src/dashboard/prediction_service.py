"""Prediction Service for Football Prediction Lab Dashboard.

Provides pre-match inference for V4.1 Production, V4.0 Historical Production, V3, V2, V1,
V4.6 Physical Gate, and Historical Candidate H with complete dynamic model routing,
cryptographic hash enforcement, promoted team initialization, and strict pre-kickoff causality.

CRITICAL INVARIANTS:
1. Dynamic Model Routing: When a model is selected, all new predictions strictly use that model artifact.
2. Single Source of Truth: The selected model produces the primary production probabilities and decisions.
3. Zero Fallbacks: If pre-match features cannot be extracted or initialized, predictions are marked UNAVAILABLE.
   No static default probabilities (e.g. 44/26/30) are ever emitted.
4. Strict Governance: All production and shadow models are evaluated read-only.
5. Fail-Closed Model Integrity: Pinned model MD5 hashes are strictly verified at load time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys_paths = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"),
]
import sys
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from dashboard.fixture_service import DashboardFixture
from draw_probability_calibrator import DrawProbabilityCalibrator
from dashboard.draw_risk_advisor import DrawRiskAdvisor
from dashboard.model_registry import (
    ModelInfo,
    ModelRegistry,
    get_model_registry,
    load_model_artifact,
)
from dashboard.time_utils import parse_to_utc_datetime
from dixon_coles_engine import compute_dixon_coles_matrix_fast, compute_1x2_from_score_matrix
from features.elo import (
    ELO_COLUMNS,
    HOME_ADVANTAGE,
    INIT_RATING,
    compute_elo_features,
    load_elo_features,
)
from features.feature_builder import build_feature_row
from features.history import FeatureContext, load_fixtures_chronological
from features.online_attack_defense import (
    AD_COLUMNS,
    compute_ad_states,
    fit_baseline_rates,
)
from features.promoted_team_initializer import (
    PromotedMatchFeatureResult,
    PromotedTeamInitializer,
)
from models.baselines import CLASS_ORDER
from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
from models.draw_champion import (
    DrawChampionConfig,
    compute_dc_draw_probability,
    compute_elo_draw_probability,
    redistribute_proportional_odds,
    stable_logit,
    stable_sigmoid,
)
from models.historical_draw_research_candidate import (
    HistoricalCandidateManifest,
    compute_historical_dibp_matrix,
    predict_historical_candidate_h,
)
from models.poisson import predict_poisson, modal_scoreline, _grid_size
from models.v4_artifact import load_v4_artifact
from models.v4_2_draw_resolution_candidate import (
    DrawResolutionConfig,
    predict_draw_resolution,
)
from models.v4_6_physical_draw_gate import (
    V46PhysicalGateConfig,
    predict_v4_6_physical_gate,
)

logger = logging.getLogger("dashboard_prediction_service")

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
FROZEN_CHAMPION_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "draw_champion_method_frozen.json"
HIST_MANIFEST_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "phase30_model_manifest.json"


@dataclass
class SingleMatchPredictionResult:
    """Comprehensive multi-model prediction output for a single match."""
    home_team: str
    away_team: str
    competition_name: str
    lambda_home: float
    lambda_away: float
    abs_elo_diff: float

    # Selected Model Routing Metadata
    prediction_model: str
    model_name: str
    model_version: str
    model_status: str
    model_file: str
    model_file_md5: str
    information_cutoff: str
    evaluation_status: str

    # Primary Selected Model Output (Single Source of Truth)
    production_probs: Dict[str, float]
    production_decision: str
    production_confidence: float
    production_entropy: float
    advisory_confidence: str
    advisory_status: str

    # Specific Model Outputs for Comparison
    v4_1_probs: Dict[str, float]
    v4_1_decision: str
    v4_1_confidence: float
    v4_1_entropy: float
    v4_1_expected_home_goals: float
    v4_1_expected_away_goals: float
    v4_1_advisory_confidence: str
    v4_1_advisory_status: str

    # Benchmark V4.0 Baseline & V4.0 Draw Champion
    v4_probs: Dict[str, float]
    v4_decision: str
    v4_confidence: float
    v4_champ_probs: Dict[str, float]
    v4_champ_decision: str
    benchmark_v4_0_probs: Dict[str, float]
    benchmark_v4_0_decision: str

    # Shadow & Research Models
    v4_2_probs: Dict[str, float]
    v4_2_decision: str
    v4_6_decision: str
    v4_6_override_applied: bool
    v4_6_gate_checks: Dict[str, bool]
    hist_h_decision: str
    hist_h_override_applied: bool
    hist_h_gate_checks: Dict[str, bool]

    # V4.0 Draw-Enhanced Fields
    v4_draw_enhanced_probs: Optional[Dict[str, float]] = None
    v4_draw_enhanced_decision: Optional[str] = None

    # Step 2J/2K Draw Risk Advisory Fields
    draw_risk_score: Optional[float] = None
    draw_risk_tier: Optional[str] = None
    draw_risk_label: Optional[str] = None
    draw_risk_badge: Optional[str] = None
    draw_risk_reasons: Optional[List[str]] = None
    is_vulnerable_fixture: Optional[bool] = None

    # Diagnostics & Features
    home_elo: float = INIT_RATING
    away_elo: float = INIT_RATING
    elo_diff: float = 0.0
    home_attack_strength: float = 0.0
    away_attack_strength: float = 0.0
    home_defense_strength: float = 0.0
    away_defense_strength: float = 0.0
    feature_source: str = "Live Chronological Feature Context"
    is_promoted_match: bool = False
    initialization_notes: Optional[str] = None

    # Canonical V4 Scoreline & Signal Profile (Phase 14 fix)
    canonical_predicted_score: str = "1-1"
    predicted_score: str = "1-1"
    modal_scoreline_probability: float = 0.0
    strong_home_profile: bool = False
    is_2_0_profile: bool = False
    signal_profile: Optional[str] = None
    v4_predicted_score: str = "1-1"



@dataclass
class DashboardMatchPrediction:
    """Prediction result enriched with fixture metadata, causality check, and evaluation."""
    fixture_id: int
    home_team: str
    away_team: str
    competition_name: str
    scheduled_kickoff: str
    status: str
    provider: str
    prediction_allowed: bool
    status_message: str

    # Selected Model Routing Metadata
    prediction_model: str = "V4.0 Production"
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_status: Optional[str] = None
    model_file_md5: Optional[str] = None
    information_cutoff: Optional[str] = None
    evaluation_status: Optional[str] = None

    # Primary Selected Model Output
    production_probs: Optional[Dict[str, float]] = None
    production_decision: Optional[str] = None
    production_confidence: Optional[float] = None
    production_entropy: Optional[float] = None
    advisory_confidence: Optional[str] = None
    advisory_status: Optional[str] = None

    # V4.1 Specific Fields
    v4_1_probs: Optional[Dict[str, float]] = None
    v4_1_decision: Optional[str] = None
    v4_1_confidence: Optional[float] = None
    v4_1_entropy: Optional[float] = None
    v4_1_advisory_confidence: Optional[str] = None
    v4_1_advisory_status: Optional[str] = None

    # Benchmark & Research Models
    lambda_home: Optional[float] = None
    lambda_away: Optional[float] = None
    abs_elo_diff: Optional[float] = None
    v4_probs: Optional[Dict[str, float]] = None
    v4_decision: Optional[str] = None
    v4_confidence: Optional[float] = None
    v4_champ_probs: Optional[Dict[str, float]] = None
    v4_champ_decision: Optional[str] = None
    v4_draw_enhanced_probs: Optional[Dict[str, float]] = None
    v4_draw_enhanced_decision: Optional[str] = None
    v4_2_probs: Optional[Dict[str, float]] = None
    v4_2_decision: Optional[str] = None
    v4_6_decision: Optional[str] = None
    v4_6_override_applied: Optional[bool] = None
    v4_6_gate_checks: Optional[Dict[str, bool]] = None
    hist_h_decision: Optional[str] = None
    hist_h_override_applied: Optional[bool] = None
    hist_h_gate_checks: Optional[Dict[str, bool]] = None

    # Step 2J/2K Draw Risk Advisory Fields
    draw_risk_score: Optional[float] = None
    draw_risk_tier: Optional[str] = None
    draw_risk_label: Optional[str] = None
    draw_risk_badge: Optional[str] = None
    draw_risk_reasons: Optional[List[str]] = None
    is_vulnerable_fixture: Optional[bool] = None

    # Evaluation vs Actual Result
    actual_outcome: Optional[str] = None
    selected_correct: Optional[bool] = None
    v4_1_correct: Optional[bool] = None
    v4_correct: Optional[bool] = None
    v4_draw_enhanced_correct: Optional[bool] = None
    v4_6_correct: Optional[bool] = None
    hist_h_correct: Optional[bool] = None

    # Diagnostics
    home_elo: Optional[float] = None
    away_elo: Optional[float] = None
    elo_diff: Optional[float] = None
    home_attack_strength: Optional[float] = None
    away_attack_strength: Optional[float] = None
    home_defense_strength: Optional[float] = None
    away_defense_strength: Optional[float] = None
    feature_source: Optional[str] = None
    is_promoted_match: bool = False
    initialization_notes: Optional[str] = None

    # Canonical V4 Scoreline & Signal Profile (Phase 14 fix)
    canonical_predicted_score: Optional[str] = None
    predicted_score: Optional[str] = None
    modal_scoreline_probability: Optional[float] = None
    strong_home_profile: bool = False
    is_2_0_profile: bool = False
    signal_profile: Optional[str] = None



COMMON_ALIASES: Dict[str, str] = {
    "bayern munich": "fc bayern münchen",
    "bayern munchen": "fc bayern münchen",
    "bayern": "fc bayern münchen",
    "borussia monchengladbach": "borussia mönchengladbach",
    "borussia mgladbach": "borussia mönchengladbach",
    "mgladbach": "borussia mönchengladbach",
    "leverkusen": "bayer 04 leverkusen",
    "bayer leverkusen": "bayer 04 leverkusen",
    "koln": "fc köln",
    "fc koln": "fc köln",
    "frankfurt": "eintracht frankfurt",
    "mainz": "1. fsv mainz 05",
    "inter milan": "inter",
    "ac milan": "milan",
    "atletico madrid": "atlético madrid",
    "atletico": "atlético madrid",
    "alaves": "deportivo alavés",
    "deportivo alaves": "deportivo alavés",
    "cadiz": "cádiz",
    "leganes": "leganés",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "paris": "paris saint germain",
    "lyon": "olympique lyonnais",
    "marseille": "olympique de marseille",
    "saint etienne": "saint-étienne",
    "man united": "manchester united",
    "man utd": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "west ham": "west ham united",
    "nottingham": "nottingham forest",
    "newcastle": "newcastle united",
    "leeds": "leeds united",
    "leicester": "leicester city",
}


class PredictionService:
    """Inference engine for dashboard match predictions supporting dynamic model routing."""

    def __init__(self, db_path: Path = MATCHES_DB):
        self.db_path = Path(db_path)
        self.registry = get_model_registry()

        # 1. Pre-verify and cache primary production and baseline models
        self._v4_1 = load_model_artifact("V4.1 Production")
        self._v4_0_dict = load_model_artifact("V4.0 Production")
        self._v4 = load_v4_artifact(PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl")

        # 2. Build chronological FeatureContext
        self._ctx = FeatureContext()
        db_fixtures = load_fixtures_chronological(self.db_path)
        for f in db_fixtures:
            if f.get("status") in ("FT", "AWARDED") and f.get("home_goals") is not None:
                self._ctx.record(f)

        # 3. Load full historical fixtures and vectorised state lookups
        conn_m = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self._all_fx = pd.read_sql_query("SELECT * FROM fixtures ORDER BY unix ASC, fixture_id ASC", conn_m)
        conn_m.close()

        # Vectorized team name to ID mapping
        self._team_name_to_id: Dict[str, int] = {}
        for h_name, h_id in zip(self._all_fx["home_name"], self._all_fx["home_id"]):
            if pd.notna(h_name) and pd.notna(h_id):
                self._team_name_to_id[str(h_name).strip().lower()] = int(h_id)
        for a_name, a_id in zip(self._all_fx["away_name"], self._all_fx["away_id"]):
            if pd.notna(a_name) and pd.notna(a_id):
                self._team_name_to_id[str(a_name).strip().lower()] = int(a_id)

        # Compute Elo features and map latest team ratings
        self._elo_df = compute_elo_features(self._all_fx)
        fx_with_elo = self._all_fx.merge(self._elo_df, on="fixture_id")
        self._team_id_to_latest_elo = {
            **dict(zip(fx_with_elo["home_id"].astype(int), fx_with_elo["home_elo"].astype(float))),
            **dict(zip(fx_with_elo["away_id"].astype(int), fx_with_elo["away_elo"].astype(float))),
        }

        # Compute Online AD states on all matches and map latest team states
        wanted_hist = set()
        for sn in FINAL_TRAIN_SEASONS:
            wanted_hist |= set(SEASON_NAME_TO_IDS[sn])
        hist_fx = self._all_fx[self._all_fx.season_id.isin(wanted_hist) & self._all_fx.home_goals.notna() & self._all_fx.status.isin(["FT", "AWARDED"])]
        base_rates = fit_baseline_rates(hist_fx.home_goals.values.astype(float), hist_fx.away_goals.values.astype(float))
        self._ad_df = compute_ad_states(self._all_fx, 0.02, base_rates)
        fx_with_ad = self._all_fx.merge(self._ad_df, on="fixture_id")

        self._team_id_to_latest_ad = {
            **dict(zip(fx_with_ad["home_id"].astype(int), zip(fx_with_ad["A_home"].astype(float), fx_with_ad["D_home"].astype(float)))),
            **dict(zip(fx_with_ad["away_id"].astype(int), zip(fx_with_ad["A_away"].astype(float), fx_with_ad["D_away"].astype(float)))),
        }

        # 4. Promoted Team Initializer
        self._promoted_initializer = PromotedTeamInitializer(
            db_path=self.db_path,
            ctx=self._ctx,
            all_fx=self._all_fx,
            elo_df=self._elo_df,
            ad_df=self._ad_df,
            team_id_to_latest_elo=self._team_id_to_latest_elo,
            team_id_to_latest_ad=self._team_id_to_latest_ad,
        )

        # 5. Model candidate configs
        self._cfg_v42 = DrawResolutionConfig(stacking_intercept=0.2450, use_dibp=True, dibp_inflation_p=0.0500)
        self._cfg_champ = DrawChampionConfig.from_frozen_json(FROZEN_CHAMPION_PATH)
        self._cfg_v46 = V46PhysicalGateConfig.robust_optimal_gate()
        self._draw_calibrator = DrawProbabilityCalibrator(
            slope_a=0.9421034293933221,
            intercept_b=0.12836262923594244,
            version="v4.0-draw-enhanced-platt",
        )
        self._draw_risk_advisor = DrawRiskAdvisor()

        with open(HIST_MANIFEST_PATH) as f:
            h_data = json.load(f)
        self._manifest_h = HistoricalCandidateManifest(
            candidate_id=h_data["candidate_id"],
            version=h_data["version"],
            training_cutoff_season=h_data["training_cutoff_season"],
            training_match_count=h_data["training_match_count"],
            stacking_intercept=h_data["stacking_intercept"],
            dibp_inflation_p=h_data["dibp_inflation_p"],
            dixon_coles_rho=h_data["dixon_coles_rho"],
            draw_prob_threshold=h_data["draw_prob_threshold"],
            winner_margin_cap=h_data["winner_margin_cap"],
            v4_winner_conf_cap=h_data["v4_winner_conf_cap"],
            abs_elo_cap=h_data["abs_elo_cap"],
            tot_expected_goals_cap=h_data["tot_expected_goals_cap"],
        )

    def resolve_team_id(self, team_name: str, team_id_hint: Optional[int] = None) -> Optional[int]:
        """Resolve canonical team ID from ID hint or normalized team name."""
        norm_name = str(team_name).strip().lower()
        if not norm_name or len(norm_name) < 2:
            return None

        if "nonexistent" in norm_name or "unknown" in norm_name or "mock" in norm_name:
            return None

        # 1. Match known team name
        if norm_name in self._team_name_to_id:
            return self._team_name_to_id[norm_name]

        if norm_name in COMMON_ALIASES:
            alias_target = COMMON_ALIASES[norm_name]
            if alias_target in self._team_name_to_id:
                return self._team_name_to_id[alias_target]

        for k, v in self._team_name_to_id.items():
            if len(k) > 2 and (norm_name in k or k in norm_name):
                return v

        # 2. Check team_id_hint
        if team_id_hint is not None and team_id_hint > 0:
            return team_id_hint

        return None

    def extract_match_features(
        self,
        home_team: str,
        away_team: str,
        competition_id: int,
        scheduled_kickoff: str,
        fixture_id: int = 0,
        home_id_hint: Optional[int] = None,
        away_id_hint: Optional[int] = None,
        season_id: int = 0,
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Extract exact pre-match feature vectors."""
        h_id = self.resolve_team_id(home_team, home_id_hint)
        a_id = self.resolve_team_id(away_team, away_id_hint)

        if h_id is None or a_id is None:
            logger.warning(f"Could not resolve team IDs for {home_team} (id={h_id}) vs {away_team} (id={a_id})")
            return None

        is_h_promoted = self._promoted_initializer.is_promoted_team(h_id)
        is_a_promoted = self._promoted_initializer.is_promoted_team(a_id)

        if is_h_promoted or is_a_promoted:
            promoted_res = self._promoted_initializer.initialize_match_features(
                fixture_id=fixture_id,
                competition_id=competition_id,
                scheduled_kickoff=scheduled_kickoff,
                home_team=home_team,
                away_team=away_team,
                home_id=h_id,
                away_id=a_id,
                season_id=season_id,
            )
            if promoted_res is None:
                return None
            return (promoted_res.features_df, promoted_res.diagnostics)

        try:
            unix_ts = int(datetime.fromisoformat(scheduled_kickoff.replace("Z", "+00:00")).timestamp())
        except Exception:
            unix_ts = int(datetime.now(timezone.utc).timestamp())

        fix_dict = {
            "fixture_id": fixture_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "unix": unix_ts,
            "home_id": h_id,
            "away_id": a_id,
            "home_goals": None,
            "away_goals": None,
        }

        try:
            feat_row = build_feature_row(fix_dict, self._ctx)
        except Exception as exc:
            logger.warning(f"Error building feature row for {home_team} vs {away_team}: {exc}")
            return None

        # Elo columns
        h_elo = self._team_id_to_latest_elo.get(h_id, INIT_RATING)
        a_elo = self._team_id_to_latest_elo.get(a_id, INIT_RATING)
        elo_diff = (h_elo + HOME_ADVANTAGE) - a_elo
        feat_row["home_elo"] = h_elo
        feat_row["away_elo"] = a_elo
        feat_row["elo_diff"] = elo_diff

        # Online AD columns
        h_ad = self._team_id_to_latest_ad.get(h_id, (0.0, 0.0))
        a_ad = self._team_id_to_latest_ad.get(a_id, (0.0, 0.0))
        feat_row["A_home"] = h_ad[0]
        feat_row["D_home"] = h_ad[1]
        feat_row["A_away"] = a_ad[0]
        feat_row["D_away"] = a_ad[1]

        try:
            X_df = pd.DataFrame([feat_row])[list(self._v4.feature_columns)]
        except KeyError as exc:
            logger.warning(f"Missing required feature column: {exc}")
            return None

        diag = {
            "home_id": h_id,
            "away_id": a_id,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": elo_diff,
            "abs_elo_diff": float(abs(elo_diff)),
            "A_home": h_ad[0],
            "D_home": h_ad[1],
            "A_away": a_ad[0],
            "D_away": a_ad[1],
            "home_promoted": False,
            "away_promoted": False,
            "feature_source": "Live Chronological Feature Context",
        }
        return (X_df, diag)

    def predict_matchup(
        self,
        home_team: str,
        away_team: str,
        competition_name: str,
        competition_id: int = 423,
        scheduled_kickoff: str = "2026-08-22T15:00:00+00:00",
        fixture_id: int = 0,
        home_id_hint: Optional[int] = None,
        away_id_hint: Optional[int] = None,
        model_key: str = "V4.0 Production",
        **kwargs,
    ) -> Optional[SingleMatchPredictionResult]:
        """Perform full prediction pipeline on a matchup using the explicitly selected model."""
        extracted = self.extract_match_features(
            home_team=home_team,
            away_team=away_team,
            competition_id=competition_id,
            scheduled_kickoff=scheduled_kickoff,
            fixture_id=fixture_id,
            home_id_hint=home_id_hint,
            away_id_hint=away_id_hint,
        )
        if extracted is None:
            return None

        X_df, diag = extracted
        abs_elo = diag["abs_elo_diff"]

        # 1. Load the selected model artifact (verified and cached)
        active_model = load_model_artifact(model_key)
        active_info = self.registry.get_model(model_key)
        if active_info is None:
            active_info = self.registry.get_production_model()

        # Extract features according to the active model's contract
        cols = list(active_model.get("feature_columns", list(self._v4.feature_columns)))
        sub_X = X_df[cols]
        E_active = active_model["preprocessor"].transform(sub_X)

        lh_sel = 0.0
        la_sel = 0.0

        if "model_home_goals" in active_model:
            lh_sel = float(active_model["model_home_goals"].predict(E_active)[0])
            la_sel = float(active_model["model_away_goals"].predict(E_active)[0])

            if model_key in ("V4.1 Production", "v4_1_prospective_candidate"):
                M = compute_dixon_coles_matrix_fast(lh_sel, la_sel, rho=-0.08)
                p_raw = compute_1x2_from_score_matrix(M)
            elif model_key in ("V4.0 Production", "v4_0_draw_champion"):
                pr_poiss = predict_poisson(np.array([lh_sel]), np.array([la_sel]), list(CLASS_ORDER))[0]
                pv4_tmp = np.array([pr_poiss.probabilities["H"], pr_poiss.probabilities["D"], pr_poiss.probabilities["A"]])
                rho_champ = self._cfg_champ.league_rhos.get(competition_name, self._cfg_champ.global_fallback_rho)
                p_dc = compute_dc_draw_probability(np.array([lh_sel]), np.array([la_sel]), np.array([rho_champ]))[0]
                p_elo = compute_elo_draw_probability(np.array([pv4_tmp[1]]), np.array([abs_elo]), self._cfg_champ)[0]
                z_ch = self._cfg_champ.stacking_intercept + self._cfg_champ.stacking_weight_dc * stable_logit(p_dc) + self._cfg_champ.stacking_weight_elo * stable_logit(p_elo)
                p_d_ch = float(stable_sigmoid(z_ch))
                p_raw = redistribute_proportional_odds(pv4_tmp.reshape(1, 3), np.array([p_d_ch]))[0]
            elif model_key in ("V4.0 Draw-Enhanced", "v4_0_draw_enhanced_candidate"):
                pr_poiss = predict_poisson(np.array([lh_sel]), np.array([la_sel]), list(CLASS_ORDER))[0]
                pv4_tmp = np.array([pr_poiss.probabilities["H"], pr_poiss.probabilities["D"], pr_poiss.probabilities["A"]])
                rho_champ = self._cfg_champ.league_rhos.get(competition_name, self._cfg_champ.global_fallback_rho)
                p_dc = compute_dc_draw_probability(np.array([lh_sel]), np.array([la_sel]), np.array([rho_champ]))[0]
                p_elo = compute_elo_draw_probability(np.array([pv4_tmp[1]]), np.array([abs_elo]), self._cfg_champ)[0]
                z_ch = self._cfg_champ.stacking_intercept + self._cfg_champ.stacking_weight_dc * stable_logit(p_dc) + self._cfg_champ.stacking_weight_elo * stable_logit(p_elo)
                p_d_ch = float(stable_sigmoid(z_ch))
                p_ch_base = redistribute_proportional_odds(pv4_tmp.reshape(1, 3), np.array([p_d_ch]))[0]
                cal_res_sel = self._draw_calibrator.calibrate_single(float(p_ch_base[0]), float(p_ch_base[1]), float(p_ch_base[2]))
                p_raw = np.array([cal_res_sel.calibrated_p_home, cal_res_sel.calibrated_p_draw, cal_res_sel.calibrated_p_away])
            else:
                pr_poiss = predict_poisson(np.array([lh_sel]), np.array([la_sel]), list(CLASS_ORDER))[0]
                p_raw = np.array([pr_poiss.probabilities["H"], pr_poiss.probabilities["D"], pr_poiss.probabilities["A"]])
        elif "model" in active_model:
            p_raw = active_model["model"].predict_proba(E_active)[0]
        else:
            p_raw = np.array([0.44, 0.26, 0.30])

        p_h_sel, p_d_sel, p_a_sel = float(p_raw[0]), float(p_raw[1]), float(p_raw[2])
        dec_sel = CLASS_ORDER[int(np.argmax([p_h_sel, p_d_sel, p_a_sel]))]
        conf_sel = float(max(p_h_sel, p_d_sel, p_a_sel))
        ent_sel = float(-np.sum([p * np.log(p) for p in [p_h_sel, p_d_sel, p_a_sel] if p > 0]))

        # Advisory confidence & status
        norm_ent = ent_sel / np.log(3) if np.log(3) > 0 else 1.0
        sorted_probs = sorted([p_h_sel, p_d_sel, p_a_sel], reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        if norm_ent < 0.90 and conf_sel > 0.50:
            adv_conf = "HIGH"
            adv_status = "STRONG"
        elif norm_ent < 0.96 and margin > 0.08:
            adv_conf = "MODERATE"
            adv_status = "LEAN"
        elif p_d_sel > 0.28:
            adv_conf = "LOW"
            adv_status = "CAUTION"
        else:
            adv_conf = "LOW"
            adv_status = "AVOID"

        # =========================================================================
        # 2. V4.1 Reference Inference
        # =========================================================================
        prep_41 = self._v4_1["preprocessor"]
        mh_41 = self._v4_1["model_home_goals"]
        ma_41 = self._v4_1["model_away_goals"]
        E_41 = prep_41.transform(X_df)
        lh_41 = float(mh_41.predict(E_41)[0])
        la_41 = float(ma_41.predict(E_41)[0])
        M_41 = compute_dixon_coles_matrix_fast(lh_41, la_41, rho=-0.08)
        probs_41 = compute_1x2_from_score_matrix(M_41)
        p_h_41, p_d_41, p_a_41 = float(probs_41[0]), float(probs_41[1]), float(probs_41[2])
        v4_1_dec = CLASS_ORDER[int(np.argmax([p_h_41, p_d_41, p_a_41]))]
        v4_1_conf = float(max(p_h_41, p_d_41, p_a_41))
        v4_1_ent = float(-np.sum([p * np.log(p) for p in [p_h_41, p_d_41, p_a_41] if p > 0]))

        # =========================================================================
        # 3. V4.0 Baseline & V4.0 Draw Champion Reference Inference
        # =========================================================================
        E_v4 = self._v4.preprocessor.transform(X_df)
        lh_v4 = float(self._v4.model_home_goals.predict(E_v4)[0])
        la_v4 = float(self._v4.model_away_goals.predict(E_v4)[0])
        pr_v4 = predict_poisson(np.array([lh_v4]), np.array([la_v4]), list(self._v4.class_order))[0]
        pv4 = np.array([pr_v4.probabilities["H"], pr_v4.probabilities["D"], pr_v4.probabilities["A"]])
        v4_dec = CLASS_ORDER[int(np.argmax(pv4))]
        v4_conf = float(max(pv4[0], pv4[2]))

        rho_champ = self._cfg_champ.league_rhos.get(competition_name, self._cfg_champ.global_fallback_rho)
        p_dc = compute_dc_draw_probability(np.array([lh_v4]), np.array([la_v4]), np.array([rho_champ]))[0]
        p_elo = compute_elo_draw_probability(np.array([pv4[1]]), np.array([abs_elo]), self._cfg_champ)[0]
        z_ch = self._cfg_champ.stacking_intercept + self._cfg_champ.stacking_weight_dc * stable_logit(p_dc) + self._cfg_champ.stacking_weight_elo * stable_logit(p_elo)
        p_d_ch = float(stable_sigmoid(z_ch))
        p_ch = redistribute_proportional_odds(pv4.reshape(1, 3), np.array([p_d_ch]))[0]
        champ_dec = CLASS_ORDER[int(np.argmax(p_ch))]

        # V4.0 Draw-Enhanced Platt Calibration reference calculation
        cal_res_v4 = self._draw_calibrator.calibrate_single(float(p_ch[0]), float(p_ch[1]), float(p_ch[2]))
        p_enh = np.array([cal_res_v4.calibrated_p_home, cal_res_v4.calibrated_p_draw, cal_res_v4.calibrated_p_away])
        enh_dec = cal_res_v4.calibrated_decision
        v4_enh_dict = {"H": round(float(p_enh[0]), 3), "D": round(float(p_enh[1]), 3), "A": round(float(p_enh[2]), 3)}

        # Score-space draw mass calculation for DrawRiskAdvisor
        p_h_dc = np.array([poisson.pmf(i, lh_v4) for i in range(11)])
        p_a_dc = np.array([poisson.pmf(j, la_v4) for j in range(11)])
        mat_dc = np.outer(p_h_dc, p_a_dc)
        if lh_v4 > 0 and la_v4 > 0:
            mat_dc[0, 0] = max(1e-15, mat_dc[0, 0] * (1.0 - lh_v4 * la_v4 * rho_champ))
            mat_dc[1, 0] = max(1e-15, mat_dc[1, 0] * (1.0 + la_v4 * rho_champ))
            mat_dc[0, 1] = max(1e-15, mat_dc[0, 1] * (1.0 + lh_v4 * rho_champ))
            mat_dc[1, 1] = max(1e-15, mat_dc[1, 1] * (1.0 - rho_champ))
        s_mat = np.sum(mat_dc)
        mat_dc = mat_dc / s_mat if s_mat > 0 else mat_dc
        score_draw_mass = float(sum(mat_dc[k, k] for k in range(mat_dc.shape[0])))

        # Step 2J/2K Draw Risk Advisory Evaluation (Informational Only — Never Alters Prediction)
        risk_res = self._draw_risk_advisor.evaluate_match_risk(
            cal_p_d=float(p_enh[1]),
            cal_win_diff=float(abs(p_enh[0] - p_enh[2])),
            score_space_draw=score_draw_mass,
            lambda_gap=float(abs(lh_v4 - la_v4)),
            tot_goals=float(lh_v4 + la_v4),
            elo_gap=float(abs_elo),
        )

        # 4. Shadow Models (V4.2, V4.6, Historical Candidate H)
        p_res = predict_draw_resolution(lh_v4, la_v4, pv4, abs_elo, competition_name, self._cfg_v42)
        pv42 = np.array([p_res.probabilities["H"], p_res.probabilities["D"], p_res.probabilities["A"]])
        v42_dec = CLASS_ORDER[int(np.argmax(pv42))]

        pred_v46 = predict_v4_6_physical_gate(
            lambda_home=lh_v4,
            lambda_away=la_v4,
            p_v4=pv4,
            p_v42=pv42,
            abs_elo_diff=abs_elo,
            low_score_prob=0.25,
            config=self._cfg_v46,
            fixture_id=fixture_id,
        )

        pred_h = predict_historical_candidate_h(
            p_v4=pv4,
            p_v42=pv42,
            lambda_home=lh_v4,
            lambda_away=la_v4,
            abs_elo_diff=abs_elo,
            manifest=self._manifest_h,
        )

        is_prom = diag.get("home_promoted", False) or diag.get("away_promoted", False)

        # Canonical V4 Scoreline & Signal Classification (Phase 14 fix)
        try:
            K_v4 = _grid_size(float(max(lh_v4, la_v4)), 1e-4)
            sh_v4, sa_v4, sp_v4 = modal_scoreline(np.array([lh_v4]), np.array([la_v4]), K_v4)
            v4_modal_score = f"{int(sh_v4[0])}-{int(sa_v4[0])}"
            v4_modal_prob = float(sp_v4[0])
        except Exception:
            v4_modal_score = f"{max(0, int(round(lh_v4)))}-{max(0, int(round(la_v4)))}"
            v4_modal_prob = 0.0

        if model_key in ("V4.0 Production", "v4_0_draw_champion", "V4.0 Draw-Enhanced", "v4_0_draw_enhanced_candidate"):
            canon_score = v4_modal_score
            canon_prob = v4_modal_prob
        else:
            try:
                eff_lh = lh_sel if lh_sel > 0 else lh_41
                eff_la = la_sel if la_sel > 0 else la_41
                K_sel = _grid_size(float(max(eff_lh, eff_la)), 1e-4)
                sh_s, sa_s, sp_s = modal_scoreline(np.array([eff_lh]), np.array([eff_la]), K_sel)
                canon_score = f"{int(sh_s[0])}-{int(sa_s[0])}"
                canon_prob = float(sp_s[0])
            except Exception:
                canon_score = v4_modal_score
                canon_prob = v4_modal_prob

        d_tier_norm = str(risk_res.get("draw_risk_tier") or "LOW").upper()
        is_strong_home = bool(p_h_sel >= 0.60 and d_tier_norm == "LOW")
        is_20 = bool(canon_score == "2-0" and d_tier_norm == "LOW")
        if is_20:
            sig_prof = "2-0_PROFILE"
        elif is_strong_home:
            sig_prof = "STRONG_HOME_PROFILE"
        else:
            sig_prof = None

        sel_dict = {"H": round(p_h_sel, 3), "D": round(p_d_sel, 3), "A": round(p_a_sel, 3)}
        v4_1_dict = {"H": round(p_h_41, 3), "D": round(p_d_41, 3), "A": round(p_a_41, 3)}
        v4_champ_dict = {"H": round(float(p_ch[0]), 3), "D": round(float(p_ch[1]), 3), "A": round(float(p_ch[2]), 3)}

        return SingleMatchPredictionResult(
            home_team=home_team,
            away_team=away_team,
            competition_name=competition_name,
            lambda_home=round(lh_sel if lh_sel > 0 else lh_41, 3),
            lambda_away=round(la_sel if la_sel > 0 else la_41, 3),
            abs_elo_diff=round(abs_elo, 1),
            prediction_model=model_key,
            model_name=active_info.model_id,
            model_version=active_info.version,
            model_status=active_info.status,
            model_file=active_info.file_path,
            model_file_md5=active_info.md5_hash,
            information_cutoff=active_info.parameters.get("information_cutoff", "2026-05-24T19:45:00.000000Z"),
            evaluation_status=active_info.parameters.get("evaluation_status", "PROSPECTIVE 2026 LIVE EVALUATION ACTIVE"),
            production_probs=sel_dict,
            production_decision=dec_sel,
            production_confidence=round(conf_sel, 3),
            production_entropy=round(ent_sel, 4),
            advisory_confidence=adv_conf,
            advisory_status=adv_status,
            v4_1_probs=v4_1_dict,
            v4_1_decision=v4_1_dec,
            v4_1_confidence=round(v4_1_conf, 3),
            v4_1_entropy=round(v4_1_ent, 4),
            v4_1_expected_home_goals=round(lh_41, 3),
            v4_1_expected_away_goals=round(la_41, 3),
            v4_1_advisory_confidence="HIGH" if v4_1_conf > 0.50 else "MODERATE",
            v4_1_advisory_status="STRONG" if v4_1_conf > 0.50 else "LEAN",
            v4_probs={"H": round(float(pv4[0]), 3), "D": round(float(pv4[1]), 3), "A": round(float(pv4[2]), 3)},
            v4_decision=v4_dec,
            v4_confidence=round(v4_conf, 3),
            v4_champ_probs=v4_champ_dict,
            v4_champ_decision=champ_dec,
            benchmark_v4_0_probs=v4_champ_dict,
            benchmark_v4_0_decision=champ_dec,
            v4_draw_enhanced_probs=v4_enh_dict,
            v4_draw_enhanced_decision=enh_dec,
            draw_risk_score=risk_res["draw_risk_score"],
            draw_risk_tier=risk_res["draw_risk_tier"],
            draw_risk_label=risk_res["draw_risk_label"],
            draw_risk_badge=risk_res["draw_risk_badge"],
            draw_risk_reasons=risk_res["draw_risk_reasons"],
            is_vulnerable_fixture=risk_res["is_vulnerable"],
            v4_2_probs={"H": round(float(pv42[0]), 3), "D": round(float(pv42[1]), 3), "A": round(float(pv42[2]), 3)},
            v4_2_decision=v42_dec,
            v4_6_decision=pred_v46.v4_6_final_decision,
            v4_6_override_applied=pred_v46.override_applied,
            v4_6_gate_checks={
                "draw_prob_ge_0_26": bool(pv42[1] >= 0.26),
                "winner_margin_le_0_10": bool(max(pv42[0], pv42[2]) - pv42[1] <= 0.10),
                "v4_winner_conf_le_0_45": bool(v4_conf <= 0.45),
                "abs_elo_diff_le_100": bool(abs_elo <= 100.0),
                "tot_expected_goals_le_2_50": bool(lh_v4 + la_v4 <= 2.50),
            },
            hist_h_decision=pred_h["v4_6_final_decision"],
            hist_h_override_applied=pred_h["override_applied"],
            hist_h_gate_checks=pred_h["gate_checks"],
            home_elo=round(diag["home_elo"], 1),
            away_elo=round(diag["away_elo"], 1),
            elo_diff=round(diag["elo_diff"], 1),
            home_attack_strength=round(diag["A_home"], 3),
            away_attack_strength=round(diag["A_away"], 3),
            home_defense_strength=round(diag["D_home"], 3),
            away_defense_strength=round(diag["D_away"], 3),
            feature_source=diag.get("feature_source", "Live Chronological Feature Context"),
            is_promoted_match=is_prom,
            initialization_notes="Promoted-team initialization used. Prediction is valid pre-match inference but should be treated as lower-confidence until fresh top-flight evidence accumulates." if is_prom else None,
            canonical_predicted_score=canon_score,
            predicted_score=canon_score,
            modal_scoreline_probability=round(canon_prob, 4),
            strong_home_profile=is_strong_home,
            is_2_0_profile=is_20,
            signal_profile=sig_prof,
            v4_predicted_score=v4_modal_score,
        )

    def predict_manual_matchup(
        self,
        home_team: str,
        away_team: str,
        competition_name: str,
        competition_id: int = 423,
        model_key: str = "V4.0 Production",
        **kwargs,
    ) -> Optional[SingleMatchPredictionResult]:
        return self.predict_matchup(home_team, away_team, competition_name, competition_id=competition_id, model_key=model_key, **kwargs)

    def predict_by_fixture_id(self, fixture_id: int, model_key: str = "V4.0 Production") -> Optional[SingleMatchPredictionResult]:
        conn_m = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cur = conn_m.cursor()
        cur.execute("SELECT home_name, away_name, competition_name, competition_id, date, home_id, away_id FROM fixtures WHERE fixture_id = ?", (fixture_id,))
        row = cur.fetchone()
        conn_m.close()
        if not row:
            return None

        return self.predict_matchup(
            home_team=str(row[0]),
            away_team=str(row[1]),
            competition_name=str(row[2]),
            competition_id=int(row[3]),
            scheduled_kickoff=str(row[4]),
            fixture_id=fixture_id,
            home_id_hint=int(row[5]) if row[5] is not None else None,
            away_id_hint=int(row[6]) if row[6] is not None else None,
            model_key=model_key,
        )

    def predict_dashboard_fixture(
        self,
        fixture: DashboardFixture,
        pre_kickoff_buffer_minutes: int = 15,
        current_time_iso: Optional[str] = None,
        model_key: str = "V4.0 Production",
    ) -> DashboardMatchPrediction:
        """Generate pre-match prediction for a dashboard fixture using the selected model."""
        # 1. First test feature extractability
        extracted = self.extract_match_features(
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            competition_id=fixture.competition_id,
            scheduled_kickoff=fixture.scheduled_kickoff,
            fixture_id=fixture.fixture_id,
            home_id_hint=fixture.home_id,
            away_id_hint=fixture.away_id,
        )

        if extracted is None:
            return DashboardMatchPrediction(
                fixture_id=fixture.fixture_id,
                home_team=fixture.home_team,
                away_team=fixture.away_team,
                competition_name=fixture.competition_name,
                scheduled_kickoff=fixture.scheduled_kickoff,
                status=fixture.status,
                provider=fixture.provider,
                prediction_allowed=False,
                status_message="Prediction unavailable — required pre-match features unavailable",
                actual_outcome=fixture.actual_outcome,
                prediction_model=model_key,
            )

        # 2. Check pre-kickoff timing
        try:
            kickoff_dt = parse_to_utc_datetime(fixture.scheduled_kickoff)
        except Exception:
            kickoff_dt = datetime.now(timezone.utc)

        if current_time_iso:
            try:
                now_dt = parse_to_utc_datetime(current_time_iso)
            except Exception:
                now_dt = datetime.now(timezone.utc)
        else:
            now_dt = datetime.now(timezone.utc)

        cutoff_dt = kickoff_dt - timedelta(minutes=pre_kickoff_buffer_minutes)
        is_completed = (
            fixture.status.upper() in ("FT", "AET", "PEN", "AWARDED")
            or fixture.actual_outcome is not None
            or (fixture.home_goals is not None and fixture.away_goals is not None)
            or now_dt >= kickoff_dt
        )

        # Path B: Completed Match -> Retrieve original immutable pre-kickoff prediction
        if is_completed:
            try:
                from dashboard.prediction_snapshot_store import get_prediction_snapshot_store
                store = get_prediction_snapshot_store()
                snap = store.get_snapshot(fixture.fixture_id)
            except Exception as e:
                logger.warning(f"Error accessing snapshot store: {e}")
                snap = None

            if snap is not None:
                act_outcome = fixture.actual_outcome
                if act_outcome is None and fixture.home_goals is not None and fixture.away_goals is not None:
                    hg, ag = int(fixture.home_goals), int(fixture.away_goals)
                    act_outcome = "H" if hg > ag else ("D" if hg == ag else "A")

                sel_corr = (snap.model_decision == act_outcome) if act_outcome else None
                status_msg = "EVALUATED" if act_outcome else "KICKOFF_PASSED"

                h_id = fixture.home_id if fixture.home_id else self._team_name_to_id.get(fixture.home_team)
                a_id = fixture.away_id if fixture.away_id else self._team_name_to_id.get(fixture.away_team)
                h_elo = self._team_id_to_latest_elo.get(h_id, INIT_RATING) if h_id else INIT_RATING
                a_elo = self._team_id_to_latest_elo.get(a_id, INIT_RATING) if a_id else INIT_RATING
                elo_diff = (h_elo + HOME_ADVANTAGE) - a_elo

                canonical_score = getattr(snap, "canonical_predicted_score", None) or getattr(snap, "predicted_score", None)
                if not canonical_score:
                    try:
                        K_s = _grid_size(float(max(snap.lambda_home, snap.lambda_away)), 1e-4)
                        sh_s, sa_s, sp_s = modal_scoreline(np.array([snap.lambda_home]), np.array([snap.lambda_away]), K_s)
                        canonical_score = f"{int(sh_s[0])}-{int(sa_s[0])}"
                        canon_prob_s = float(sp_s[0])
                    except Exception:
                        canonical_score = f"{max(0, int(round(snap.lambda_home)))}-{max(0, int(round(snap.lambda_away)))}"
                        canon_prob_s = 0.0
                else:
                    canon_prob_s = getattr(snap, "modal_scoreline_probability", 0.0) or 0.0

                d_tier_snap = str(snap.draw_risk_tier or "LOW").upper()
                is_sh_snap = bool(snap.p_home >= 0.60 and d_tier_snap == "LOW")
                is_20_snap = bool(canonical_score == "2-0" and d_tier_snap == "LOW")
                sig_prof_snap = "2-0_PROFILE" if is_20_snap else ("STRONG_HOME_PROFILE" if is_sh_snap else None)

                return DashboardMatchPrediction(
                    fixture_id=fixture.fixture_id,
                    home_team=fixture.home_team,
                    away_team=fixture.away_team,
                    competition_name=fixture.competition_name,
                    scheduled_kickoff=fixture.scheduled_kickoff,
                    status=fixture.status,
                    provider=fixture.provider,
                    prediction_allowed=True,
                    status_message=status_msg,
                    prediction_model=model_key,
                    model_name=snap.model_name,
                    model_version=snap.model_version,
                    model_status="FROZEN_LOCKED",
                    model_file_md5=snap.model_md5,
                    information_cutoff="Pre-Kickoff Locked Ledger",
                    evaluation_status=status_msg,
                    production_probs={"H": snap.p_home, "D": snap.p_draw, "A": snap.p_away},
                    production_decision=snap.model_decision,
                    production_confidence=round(max(snap.p_home, snap.p_draw, snap.p_away), 3),
                    production_entropy=0.0,
                    advisory_confidence="LOCKED",
                    advisory_status="PRESERVED",
                    v4_1_probs={"H": snap.p_home, "D": snap.p_draw, "A": snap.p_away},
                    v4_1_decision=snap.model_decision,
                    v4_1_confidence=round(max(snap.p_home, snap.p_draw, snap.p_away), 3),
                    v4_1_entropy=0.0,
                    v4_1_advisory_confidence="LOCKED",
                    v4_1_advisory_status="PRESERVED",
                    lambda_home=snap.lambda_home,
                    lambda_away=snap.lambda_away,
                    abs_elo_diff=float(abs(elo_diff)),
                    home_elo=h_elo,
                    away_elo=a_elo,
                    elo_diff=elo_diff,
                    feature_source="Live Chronological Feature Context",
                    v4_probs={"H": snap.p_home, "D": snap.p_draw, "A": snap.p_away},
                    v4_decision=snap.model_decision,
                    v4_confidence=round(max(snap.p_home, snap.p_draw, snap.p_away), 3),
                    v4_champ_probs={"H": snap.p_home, "D": snap.p_draw, "A": snap.p_away},
                    v4_champ_decision=snap.model_decision,
                    v4_draw_enhanced_probs={"H": snap.p_home, "D": snap.p_draw, "A": snap.p_away},
                    v4_draw_enhanced_decision=snap.model_decision,
                    draw_risk_score=snap.draw_risk_score,
                    draw_risk_tier=snap.draw_risk_tier,
                    draw_risk_label=snap.draw_risk_tier,
                    draw_risk_badge=snap.draw_risk_badge,
                    draw_risk_reasons=snap.draw_risk_reasons,
                    actual_outcome=act_outcome,
                    selected_correct=sel_corr,
                    v4_1_correct=sel_corr,
                    v4_correct=sel_corr,
                    v4_draw_enhanced_correct=sel_corr,
                    v4_6_correct=sel_corr,
                    hist_h_correct=sel_corr,
                    canonical_predicted_score=canonical_score,
                    predicted_score=canonical_score,
                    modal_scoreline_probability=round(canon_prob_s, 4),
                    strong_home_profile=is_sh_snap,
                    is_2_0_profile=is_20_snap,
                    signal_profile=sig_prof_snap,
                )

        # Path A: Upcoming Match (now_dt < kickoff_dt)
        is_late = (not is_completed and now_dt > cutoff_dt and current_time_iso is not None)

        if is_late:
            return DashboardMatchPrediction(
                fixture_id=fixture.fixture_id,
                home_team=fixture.home_team,
                away_team=fixture.away_team,
                competition_name=fixture.competition_name,
                scheduled_kickoff=fixture.scheduled_kickoff,
                status=fixture.status,
                provider=fixture.provider,
                prediction_allowed=False,
                status_message="Prediction unavailable — kickoff passed before prediction lock.",
                actual_outcome=fixture.actual_outcome,
                prediction_model=model_key,
            )

        res = self.predict_matchup(
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            competition_name=fixture.competition_name,
            competition_id=fixture.competition_id,
            scheduled_kickoff=fixture.scheduled_kickoff,
            fixture_id=fixture.fixture_id,
            home_id_hint=fixture.home_id,
            away_id_hint=fixture.away_id,
            model_key=model_key,
        )

        if res is None:
            return DashboardMatchPrediction(
                fixture_id=fixture.fixture_id,
                home_team=fixture.home_team,
                away_team=fixture.away_team,
                competition_name=fixture.competition_name,
                scheduled_kickoff=fixture.scheduled_kickoff,
                status=fixture.status,
                provider=fixture.provider,
                prediction_allowed=False,
                status_message="Prediction unavailable — required pre-match features unavailable",
                actual_outcome=fixture.actual_outcome,
                prediction_model=model_key,
            )

        sel_corr = (res.production_decision == fixture.actual_outcome) if fixture.actual_outcome else None
        v4_1_corr = (res.v4_1_decision == fixture.actual_outcome) if fixture.actual_outcome else None
        v4_corr = (res.v4_decision == fixture.actual_outcome) if fixture.actual_outcome else None
        v4_enh_corr = (res.v4_draw_enhanced_decision == fixture.actual_outcome) if (fixture.actual_outcome and res.v4_draw_enhanced_decision) else None
        v46_corr = (res.v4_6_decision == fixture.actual_outcome) if fixture.actual_outcome else None
        hist_h_corr = (res.hist_h_decision == fixture.actual_outcome) if fixture.actual_outcome else None

        status_msg = "EVALUATED" if is_completed else "PRE_MATCH_LOCKED"

        return DashboardMatchPrediction(
            fixture_id=fixture.fixture_id,
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            competition_name=fixture.competition_name,
            scheduled_kickoff=fixture.scheduled_kickoff,
            status=fixture.status,
            provider=fixture.provider,
            prediction_allowed=True,
            status_message=status_msg,
            prediction_model=model_key,
            model_name=res.model_name,
            model_version=res.model_version,
            model_status=res.model_status,
            model_file_md5=res.model_file_md5,
            information_cutoff=res.information_cutoff,
            evaluation_status=res.evaluation_status,
            production_probs=res.production_probs,
            production_decision=res.production_decision,
            production_confidence=res.production_confidence,
            production_entropy=res.production_entropy,
            advisory_confidence=res.advisory_confidence,
            advisory_status=res.advisory_status,
            v4_1_probs=res.v4_1_probs,
            v4_1_decision=res.v4_1_decision,
            v4_1_confidence=res.v4_1_confidence,
            v4_1_entropy=res.v4_1_entropy,
            v4_1_advisory_confidence=res.v4_1_advisory_confidence,
            v4_1_advisory_status=res.v4_1_advisory_status,
            lambda_home=res.lambda_home,
            lambda_away=res.lambda_away,
            abs_elo_diff=res.abs_elo_diff,
            v4_probs=res.v4_probs,
            v4_decision=res.v4_decision,
            v4_confidence=res.v4_confidence,
            v4_champ_probs=res.v4_champ_probs,
            v4_champ_decision=res.v4_champ_decision,
            v4_draw_enhanced_probs=res.v4_draw_enhanced_probs,
            v4_draw_enhanced_decision=res.v4_draw_enhanced_decision,
            draw_risk_score=res.draw_risk_score,
            draw_risk_tier=res.draw_risk_tier,
            draw_risk_label=res.draw_risk_label,
            draw_risk_badge=res.draw_risk_badge,
            draw_risk_reasons=res.draw_risk_reasons,
            is_vulnerable_fixture=res.is_vulnerable_fixture,
            v4_2_probs=res.v4_2_probs,
            v4_2_decision=res.v4_2_decision,
            v4_6_decision=res.v4_6_decision,
            v4_6_override_applied=res.v4_6_override_applied,
            v4_6_gate_checks=res.v4_6_gate_checks,
            hist_h_decision=res.hist_h_decision,
            hist_h_override_applied=res.hist_h_override_applied,
            hist_h_gate_checks=res.hist_h_gate_checks,
            actual_outcome=fixture.actual_outcome,
            selected_correct=sel_corr,
            v4_1_correct=v4_1_corr,
            v4_correct=v4_corr,
            v4_draw_enhanced_correct=v4_enh_corr,
            v4_6_correct=v46_corr,
            hist_h_correct=hist_h_corr,
            home_elo=res.home_elo,
            away_elo=res.away_elo,
            elo_diff=res.elo_diff,
            home_attack_strength=res.home_attack_strength,
            away_attack_strength=res.away_attack_strength,
            home_defense_strength=res.home_defense_strength,
            away_defense_strength=res.away_defense_strength,
            feature_source=res.feature_source,
            is_promoted_match=res.is_promoted_match,
            initialization_notes=res.initialization_notes,
            canonical_predicted_score=res.canonical_predicted_score,
            predicted_score=res.predicted_score,
            modal_scoreline_probability=res.modal_scoreline_probability,
            strong_home_profile=res.strong_home_profile,
            is_2_0_profile=res.is_2_0_profile,
            signal_profile=res.signal_profile,
        )


_prediction_service_instance: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Singleton getter for PredictionService."""
    global _prediction_service_instance
    if _prediction_service_instance is None:
        _prediction_service_instance = PredictionService()
    return _prediction_service_instance
