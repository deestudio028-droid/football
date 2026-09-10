"""Phase 38 — Experiment E13: Dynamic Attack/Defense State Space & Variance Engine.

Tracks dynamically evolving latent attack & defense states with uncertainty variances:
    theta_i(t) = [A_i(t), D_i(t)]
    Sigma_i(t) = diag(var_A_i(t), var_D_i(t))

Supports:
- Causal Poisson score updating
- Mean-reverting process noise
- Venue-specific latent components (Home A/D vs. Away A/D)
- Parameter uncertainty tracking
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

INIT_VAR: float = 0.25
PROCESS_NOISE_Q: float = 0.005
MEAN_REVERSION_RATE: float = 0.015


def compute_dynamic_ad_states(
    fixtures_df: pd.DataFrame,
    mu_home: float = 1.5348,
    mu_away: float = 1.2740,
    lr: float = 0.025,
    q_noise: float = 0.005,
    reversion_rate: float = 0.015,
) -> pd.DataFrame:
    """Computes causal dynamic attack/defense states and posterior variance estimates."""
    df = fixtures_df.sort_values(["unix", "date", "fixture_id"]).reset_index(drop=True)

    # General latent states & variances
    A: Dict[int, float] = defaultdict(float)
    D: Dict[int, float] = defaultdict(float)
    var_A: Dict[int, float] = defaultdict(lambda: INIT_VAR)
    var_D: Dict[int, float] = defaultdict(lambda: INIT_VAR)

    # Venue-specific latent states & variances
    A_h: Dict[int, float] = defaultdict(float)
    D_h: Dict[int, float] = defaultdict(float)
    A_a: Dict[int, float] = defaultdict(float)
    D_a: Dict[int, float] = defaultdict(float)

    match_counts: Dict[int, int] = defaultdict(int)
    records: List[Dict[str, Any]] = []

    for unix_ts, group in df.groupby("unix", sort=True):
        # --- PASS 1: Extract Pre-Match Dynamic States & Variances ---
        for row in group.itertuples(index=False):
            h_id = row.home_id
            a_id = row.away_id

            a_h = A[h_id]
            d_h = D[h_id]
            a_a = A[a_id]
            d_a = D[a_id]

            va_h = var_A[h_id]
            vd_h = var_D[h_id]
            va_a = var_A[a_id]
            vd_a = var_D[a_id]

            # Implied Poisson intensity parameters
            implied_lam_h = mu_home * math.exp(np.clip(a_h - d_a, -1.5, 1.5))
            implied_lam_a = mu_away * math.exp(np.clip(a_a - d_h, -1.5, 1.5))

            # Total parameter uncertainty for match
            total_var_h = va_h + vd_a
            total_var_a = va_a + vd_h

            records.append({
                "fixture_id": row.fixture_id,
                "dyn_A_home": a_h,
                "dyn_D_home": d_h,
                "dyn_A_away": a_a,
                "dyn_D_away": d_a,
                "var_A_home": va_h,
                "var_D_home": vd_h,
                "var_A_away": va_a,
                "var_D_away": vd_a,
                "dyn_implied_lam_h": implied_lam_h,
                "dyn_implied_lam_a": implied_lam_a,
                "total_param_var_h": total_var_h,
                "total_param_var_a": total_var_a,
                "dyn_A_venue_home": A_h[h_id],
                "dyn_D_venue_home": D_h[h_id],
                "dyn_A_venue_away": A_a[a_id],
                "dyn_D_venue_away": D_a[a_id],
                "h_dyn_sample_size": match_counts[h_id],
                "a_dyn_sample_size": match_counts[a_id],
            })

        # --- PASS 2: State-Space Time Evolution & Observation Updates ---
        for row in group.itertuples(index=False):
            if row.status not in ("FT", "AWARDED") or pd.isna(row.home_goals) or pd.isna(row.away_goals):
                continue

            h_id = row.home_id
            a_id = row.away_id
            hg = float(row.home_goals)
            ag = float(row.away_goals)

            # 1. State-Space Time Step: Process Noise & Mean Reversion
            A[h_id] = (1.0 - reversion_rate) * A[h_id]
            D[h_id] = (1.0 - reversion_rate) * D[h_id]
            A[a_id] = (1.0 - reversion_rate) * A[a_id]
            D[a_id] = (1.0 - reversion_rate) * D[a_id]

            var_A[h_id] = var_A[h_id] + q_noise
            var_D[h_id] = var_D[h_id] + q_noise
            var_A[a_id] = var_A[a_id] + q_noise
            var_D[a_id] = var_D[a_id] + q_noise

            # 2. Observation Likelihood Updates (Poisson Score Ascent)
            lam_h = mu_home * math.exp(np.clip(A[h_id] - D[a_id], -1.5, 1.5))
            lam_a = mu_away * math.exp(np.clip(A[a_id] - D[h_id], -1.5, 1.5))

            err_h = hg - lam_h
            err_a = ag - lam_a

            # Dynamic adaptive learning rate scaled by current uncertainty
            gain_ah = lr * (1.0 + 2.0 * var_A[h_id])
            gain_da = lr * (1.0 + 2.0 * var_D[a_id])
            gain_aa = lr * (1.0 + 2.0 * var_A[a_id])
            gain_dh = lr * (1.0 + 2.0 * var_D[h_id])

            A[h_id] = np.clip(A[h_id] + gain_ah * err_h, -1.5, 1.5)
            D[a_id] = np.clip(D[a_id] - gain_da * err_h, -1.5, 1.5)
            A[a_id] = np.clip(A[a_id] + gain_aa * err_a, -1.5, 1.5)
            D[h_id] = np.clip(D[h_id] - gain_dh * err_a, -1.5, 1.5)

            # Variance Reduction via Observation Information
            var_A[h_id] = max(var_A[h_id] * (1.0 - 0.08), 0.01)
            var_D[a_id] = max(var_D[a_id] * (1.0 - 0.08), 0.01)
            var_A[a_id] = max(var_A[a_id] * (1.0 - 0.08), 0.01)
            var_D[h_id] = max(var_D[h_id] * (1.0 - 0.08), 0.01)

            # Venue-specific updates
            A_h[h_id] = np.clip(A_h[h_id] + 0.03 * err_h, -1.5, 1.5)
            D_h[h_id] = np.clip(D_h[h_id] - 0.03 * err_a, -1.5, 1.5)
            A_a[a_id] = np.clip(A_a[a_id] + 0.03 * err_a, -1.5, 1.5)
            D_a[a_id] = np.clip(D_a[a_id] - 0.03 * err_h, -1.5, 1.5)

            match_counts[h_id] += 1
            match_counts[a_id] += 1

    return pd.DataFrame(records)
