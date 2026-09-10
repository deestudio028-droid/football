"""Phase 39 — Experiment E14: Causal Pre-Match Regime Detector.

Extracts causal regime features available strictly prior to kickoff:
- Early-season match indicators (match 1-3, 4-5, 6-10)
- Team sample maturity & promotion status (n < 5, 5-10, >10)
- Elo parity tiers (|delta_elo| <= 25, 25-50, 50-100, 100-150, >150)
- Total expected goals environment (<2.4, 2.4-3.0, >3.0)
- Unsupervised Gaussian Mixture Model / KMeans cluster assignment
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def extract_regime_features(
    meta_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    elo_diff: np.ndarray,
    v4_lam_h: np.ndarray,
    v4_lam_a: np.ndarray,
    team_sample_sizes_h: np.ndarray,
    team_sample_sizes_a: np.ndarray,
) -> pd.DataFrame:
    """Extracts strictly causal pre-match regime features for every fixture in meta_df."""
    # Compute season match rank on full fixtures history
    fx = fixtures_df.copy()
    fx["season_match_num"] = fx.groupby(["competition_id", "season_id"])["unix"].rank(method="dense").astype(int)
    fx_match_num_map = fx.set_index("fixture_id")["season_match_num"]

    season_match_num = meta_df["fixture_id"].map(fx_match_num_map).fillna(10).astype(int).values

    # 2. Team Maturity & Promotion Flags
    min_samples = np.minimum(team_sample_sizes_h, team_sample_sizes_a)
    is_promoted_sparse = (min_samples < 5).astype(int)
    is_developing = ((min_samples >= 5) & (min_samples <= 10)).astype(int)
    is_established = (min_samples > 10).astype(int)

    # 3. Elo Parity Tiers
    abs_elo_diff = np.abs(elo_diff)
    parity_tight = (abs_elo_diff <= 25.0).astype(int)
    parity_close = ((abs_elo_diff > 25.0) & (abs_elo_diff <= 50.0)).astype(int)
    parity_moderate = ((abs_elo_diff > 50.0) & (abs_elo_diff <= 100.0)).astype(int)
    parity_wide = (abs_elo_diff > 100.0).astype(int)

    # 4. Total Expected Goals Environment
    total_lam = v4_lam_h + v4_lam_a
    env_low_goals = (total_lam < 2.40).astype(int)
    env_med_goals = ((total_lam >= 2.40) & (total_lam <= 3.00)).astype(int)
    env_high_goals = (total_lam > 3.00).astype(int)

    # 5. Early Season Indicators
    is_early_3 = (season_match_num <= 3).astype(int)
    is_early_5 = ((season_match_num > 3) & (season_match_num <= 5)).astype(int)
    is_early_10 = ((season_match_num > 5) & (season_match_num <= 10)).astype(int)
    is_late_season = (season_match_num > 10).astype(int)

    regime_df = pd.DataFrame({
        "fixture_id": meta_df["fixture_id"].values,
        "season_match_num": season_match_num,
        "min_team_samples": min_samples,
        "is_promoted_sparse": is_promoted_sparse,
        "is_developing": is_developing,
        "is_established": is_established,
        "abs_elo_diff": abs_elo_diff,
        "parity_tight": parity_tight,
        "parity_close": parity_close,
        "parity_moderate": parity_moderate,
        "parity_wide": parity_wide,
        "total_lam": total_lam,
        "env_low_goals": env_low_goals,
        "env_med_goals": env_med_goals,
        "env_high_goals": env_high_goals,
        "is_early_3": is_early_3,
        "is_early_5": is_early_5,
        "is_early_10": is_early_10,
        "is_late_season": is_late_season,
    })

    return regime_df


class CausalClusterRegimeDetector:
    """Unsupervised data-driven regime discovery using Gaussian Mixture Models."""

    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_components, random_state=random_state, covariance_type="diag")

    def fit(self, X_regime: np.ndarray) -> CausalClusterRegimeDetector:
        X_scaled = self.scaler.fit_transform(X_regime)
        self.gmm.fit(X_scaled)
        return self

    def predict_regimes(self, X_regime: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_regime)
        return self.gmm.predict(X_scaled)

    def predict_regime_posteriors(self, X_regime: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_regime)
        return self.gmm.predict_proba(X_scaled)
