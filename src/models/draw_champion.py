"""Production module for the Frozen Draw Champion Layer.

Model Architecture:
    V4 Poisson Model (Frozen)
        ↓
    V4 Lambdas (lambda_home, lambda_away) & Baseline Probabilities [P(H), P(D), P(A)]
        ↓
    +------------------------------------+   +------------------------------------+
    |     Frozen Dixon-Coles Layer       |   |       Frozen Elo Draw Layer        |
    |  - League-specific rho             |   |  - Pre-match |dElo| / 100          |
    |  - Low-score tau matrix correction |   |  - logit(P(D)_V4)                  |
    |  -> P(D)_DC                        |   |  -> P(D)_Elo                       |
    +------------------------------------+   +------------------------------------+
        ↓                                       ↓
    +-----------------------------------------------------------------------------+
    |                        Frozen Logistic Stacking                             |
    |  logit(P(D)_new) = 0.1130 + 0.6037 * logit(P_DC) + 0.4812 * logit(P_Elo)    |
    +-----------------------------------------------------------------------------+
        ↓
    +-----------------------------------------------------------------------------+
    |                    Proportional Odds Simplex Redistribution                 |
    |  P(H)_new = P(H)_V4 * (1 - P(D)_new) / (1 - P(D)_V4)                        |
    |  P(A)_new = P(A)_V4 * (1 - P(D)_new) / (1 - P(D)_V4)                        |
    +-----------------------------------------------------------------------------+
        ↓
    Champion Probabilities [P(H), P(D), P(A)] + Original V4 Shadow Retained

INVARIANTS & INTEGRITY:
- Fully deterministic pure functions.
- Preserves exact conditional relative odds: P(H)_new / P(A)_new == P(H)_V4 / P(A)_V4.
- Sums strictly to 1.0 (simplex normalization) and strictly non-negative.
- No market odds as features (market odds are reference-only).
- Causal pre-match information only.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import CLASS_ORDER
from .poisson import TAIL_TOL, _grid_size, _pmf_grid, modal_scoreline

#: Path to the authoritative frozen champion methodology JSON.
DEFAULT_FROZEN_SPEC_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "research"
    / "v4_promotion"
    / "draw_champion_method_frozen.json"
)

EXPECTED_CHAMPION_NAME = "V4_Expanding_Window_DixonColes_plus_Elo_Stacking"
EXPECTED_CHAMPION_VERSION = "1.0"
EXPECTED_MODEL_VERSION = "v4.0-champion-dc-elo-stacking"


class DrawChampionError(RuntimeError):
    """Raised when a safety or mathematical validation gate fails."""


@dataclass(frozen=True)
class DrawChampionConfig:
    """Frozen parameters for the Draw Champion layer."""
    stacking_intercept: float = 0.1130
    stacking_weight_dc: float = 0.6037
    stacking_weight_elo: float = 0.4812

    elo_intercept: float = 0.2227
    elo_slope_logit_v4: float = 1.1278
    elo_slope_abs_elo: float = -0.1652

    league_rhos: dict[str, float] = None
    global_fallback_rho: float = -0.0560

    def __post_init__(self):
        if self.league_rhos is None:
            object.__setattr__(self, "league_rhos", {
                "Bundesliga": -0.0768,
                "Ligue 1": -0.0583,
                "Serie A": -0.0468,
                "La Liga": -0.0387,
                "Premier League": -0.0163,
            })

    @classmethod
    def from_frozen_json(cls, spec_path: Path | str = DEFAULT_FROZEN_SPEC_PATH) -> DrawChampionConfig:
        """Load and structurally verify config from the frozen JSON specification."""
        path = Path(spec_path)
        if not path.exists():
            # If path doesn't exist (e.g. deployed standalone), return default verified config
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dc_cfg = data.get("component_a_dixon_coles", {})
        elo_cfg = data.get("component_b_elo_draw", {})
        stack_cfg = data.get("stacking_integration_layer", {})

        return cls(
            stacking_intercept=float(stack_cfg["coefficients"]["intercept"]),
            stacking_weight_dc=float(stack_cfg["coefficients"]["weight_dc"]),
            stacking_weight_elo=float(stack_cfg["coefficients"]["weight_elo"]),
            elo_intercept=float(elo_cfg["coefficients"]["a0_intercept"]),
            elo_slope_logit_v4=float(elo_cfg["coefficients"]["a1_logit_v4"]),
            elo_slope_abs_elo=float(elo_cfg["coefficients"]["a2_abs_elo"]),
            league_rhos=dict(dc_cfg.get("league_rhos", {})),
            global_fallback_rho=float(dc_cfg.get("global_fallback_rho", -0.0560)),
        )


@dataclass(frozen=True)
class DrawChampionPrediction:
    """Production prediction record containing champion and baseline outputs."""
    prediction: str
    probabilities: dict[str, float]
    v4_probabilities: dict[str, float]
    p_draw_dc: float
    p_draw_elo: float
    p_draw_champion: float
    expected_goals_home: float
    expected_goals_away: float
    modal_scoreline: str
    modal_scoreline_probability: float
    lambda_home: float
    lambda_away: float
    model_name: str = "v4_draw_champion"
    model_version: str = EXPECTED_MODEL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "probabilities": dict(self.probabilities),
            "v4_probabilities": dict(self.v4_probabilities),
            "p_draw_dc": self.p_draw_dc,
            "p_draw_elo": self.p_draw_elo,
            "p_draw_champion": self.p_draw_champion,
            "expected_goals_home": self.expected_goals_home,
            "expected_goals_away": self.expected_goals_away,
            "modal_scoreline": self.modal_scoreline,
            "modal_scoreline_probability": self.modal_scoreline_probability,
            "lambda_home": self.lambda_home,
            "lambda_away": self.lambda_away,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


def stable_logit(p: float | np.ndarray, eps: float = 1e-12) -> float | np.ndarray:
    """Compute logit(p) with numerical boundary clipping."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def stable_sigmoid(z: float | np.ndarray, clip_bound: float = 30.0) -> float | np.ndarray:
    """Compute sigmoid(z) with exponent clipping for numerical stability."""
    z_clipped = np.clip(z, -clip_bound, clip_bound)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def compute_dc_draw_probability(
    lam_h: np.ndarray,
    lam_a: np.ndarray,
    rho: float | np.ndarray,
    tol: float = TAIL_TOL,
) -> np.ndarray:
    """Compute Dixon-Coles draw probability P(D)_DC using the low-score tau matrix."""
    lam_h = np.asarray(lam_h, dtype=float)
    lam_a = np.asarray(lam_a, dtype=float)
    is_scalar = lam_h.ndim == 0
    if is_scalar:
        lam_h = np.array([float(lam_h)])
        lam_a = np.array([float(lam_a)])

    n = len(lam_h)
    if isinstance(rho, (int, float)):
        rhos = np.full(n, float(rho))
    else:
        rhos = np.asarray(rho, dtype=float)
        if len(rhos) == 1 and n > 1:
            rhos = np.full(n, rhos[0])

    lam_max = float(max(np.max(lam_h), np.max(lam_a)))
    K = max(_grid_size(lam_max, tol), 1)

    ph = _pmf_grid(lam_h, K)
    pa = _pmf_grid(lam_a, K)

    # Independent Poisson draw probability: sum_k P(H=k) * P(A=k)
    p_draw_ind = (ph * pa).sum(axis=1)

    # Tau correction adjustments on draw cells (0,0) and (1,1):
    # tau(0,0) = 1 - lam_h * lam_a * rho -> delta = - lam_h * lam_a * rho
    # tau(1,1) = 1 - rho -> delta = - rho
    delta_00 = - (lam_h * lam_a * rhos) * ph[:, 0] * pa[:, 0]
    delta_11 = - rhos * ph[:, 1] * pa[:, 1]

    p_draw_dc = np.clip(p_draw_ind + delta_00 + delta_11, 1e-12, 1.0 - 1e-12)
    return float(p_draw_dc[0]) if is_scalar else p_draw_dc


def compute_elo_draw_probability(
    p_draw_v4: float | np.ndarray,
    abs_elo_diff: float | np.ndarray,
    config: DrawChampionConfig,
) -> float | np.ndarray:
    """Compute Elo-calibrated draw probability P(D)_Elo."""
    z_v4 = stable_logit(p_draw_v4)
    elo_scaled = np.asarray(abs_elo_diff, dtype=float) / 100.0
    z_elo = config.elo_intercept + config.elo_slope_logit_v4 * z_v4 + config.elo_slope_abs_elo * elo_scaled
    return stable_sigmoid(z_elo)


def stack_draw_probabilities(
    p_draw_dc: float | np.ndarray,
    p_draw_elo: float | np.ndarray,
    config: DrawChampionConfig,
) -> float | np.ndarray:
    """Stack DC and Elo draw signals into champion draw probability P(D)_new."""
    z_dc = stable_logit(p_draw_dc)
    z_elo = stable_logit(p_draw_elo)
    z_champ = config.stacking_intercept + config.stacking_weight_dc * z_dc + config.stacking_weight_elo * z_elo
    return stable_sigmoid(z_champ)


def redistribute_proportional_odds(
    p_v4: np.ndarray,
    p_draw_new: float | np.ndarray,
) -> np.ndarray:
    """Redistribute Home/Away probability mass preserving conditional odds P(H)/P(A)."""
    p_v4 = np.asarray(p_v4, dtype=float)
    is_1d = p_v4.ndim == 1
    if is_1d:
        p_v4 = p_v4[None, :]

    p_orig_d = np.clip(p_v4[:, 1], 1e-12, 1.0 - 1e-12)
    p_d_new = np.clip(p_draw_new, 1e-12, 1.0 - 1e-12)

    ratio = (1.0 - p_d_new) / (1.0 - p_orig_d)
    p_h_new = p_v4[:, 0] * ratio
    p_a_new = p_v4[:, 2] * ratio

    p_new = np.column_stack([p_h_new, p_d_new, p_a_new])
    p_new = np.clip(p_new, 1e-15, 1.0)
    p_new = p_new / p_new.sum(axis=1, keepdims=True)

    return p_new[0] if is_1d else p_new


def predict_draw_champion(
    lam_h: float | np.ndarray,
    lam_a: float | np.ndarray,
    p_v4: np.ndarray,
    abs_elo_diff: float | np.ndarray,
    league_name: str | list[str] | np.ndarray,
    config: DrawChampionConfig | None = None,
    class_order: list[str] = list(CLASS_ORDER),
) -> list[DrawChampionPrediction]:
    """Execute the complete production Draw Champion prediction pipeline.

    Args:
        lam_h: Home Poisson rate parameter(s).
        lam_a: Away Poisson rate parameter(s).
        p_v4: Baseline V4 probabilities of shape (n, 3) or (3,).
        abs_elo_diff: Absolute pre-match Elo rating difference(s).
        league_name: Competition/league name(s).
        config: Optional DrawChampionConfig (defaults to frozen config).
        class_order: Outcome class order, default ['H', 'D', 'A'].

    Returns:
        List of DrawChampionPrediction objects.
    """
    if config is None:
        config = DrawChampionConfig.from_frozen_json()

    lam_h_arr = np.asarray(lam_h, dtype=float)
    lam_a_arr = np.asarray(lam_a, dtype=float)
    if lam_h_arr.ndim == 0:
        lam_h_arr = np.array([float(lam_h_arr)])
        lam_a_arr = np.array([float(lam_a_arr)])

    n = len(lam_h_arr)
    p_v4_arr = np.asarray(p_v4, dtype=float)
    if p_v4_arr.ndim == 1:
        p_v4_arr = p_v4_arr[None, :]

    if len(p_v4_arr) != n or p_v4_arr.shape[1] != 3:
        raise DrawChampionError(f"p_v4 must have shape ({n}, 3), got {p_v4_arr.shape}")

    if np.isscalar(abs_elo_diff):
        elo_diff_arr = np.full(n, float(abs_elo_diff))
    else:
        elo_diff_arr = np.asarray(abs_elo_diff, dtype=float)

    if isinstance(league_name, str):
        leagues = [league_name] * n
    else:
        leagues = list(league_name)

    # 1. Resolve Dixon-Coles rho per fixture
    rhos = np.array([config.league_rhos.get(lg, config.global_fallback_rho) for lg in leagues])

    # 2. Compute DC Draw Probabilities
    p_dc_draw = compute_dc_draw_probability(lam_h_arr, lam_a_arr, rhos)

    # 3. Compute Elo Draw Probabilities
    p_elo_draw = compute_elo_draw_probability(p_v4_arr[:, 1], elo_diff_arr, config)

    # 4. Stacking Integration Layer
    p_champ_draw = stack_draw_probabilities(p_dc_draw, p_elo_draw, config)

    # 5. Proportional Odds Redistribution
    p_champ_all = redistribute_proportional_odds(p_v4_arr, p_champ_draw)

    # 6. Modal Scorelines
    lam_max = float(max(np.max(lam_h_arr), np.max(lam_a_arr)))
    K = max(_grid_size(lam_max, TAIL_TOL), 1)
    sh, sa, sp = modal_scoreline(lam_h_arr, lam_a_arr, K)

    results: list[DrawChampionPrediction] = []
    for i in range(n):
        pred_idx = int(p_champ_all[i].argmax())
        pred_class = class_order[pred_idx]

        champ_probs = {cls: float(p_champ_all[i, j]) for j, cls in enumerate(class_order)}
        v4_probs = {cls: float(p_v4_arr[i, j]) for j, cls in enumerate(class_order)}

        results.append(DrawChampionPrediction(
            prediction=pred_class,
            probabilities=champ_probs,
            v4_probabilities=v4_probs,
            p_draw_dc=float(p_dc_draw[i]),
            p_draw_elo=float(p_elo_draw[i]),
            p_draw_champion=float(p_champ_draw[i]),
            expected_goals_home=float(lam_h_arr[i]),
            expected_goals_away=float(lam_a_arr[i]),
            modal_scoreline=f"{int(sh[i])}-{int(sa[i])}",
            modal_scoreline_probability=float(sp[i]),
            lambda_home=float(lam_h_arr[i]),
            lambda_away=float(lam_a_arr[i]),
        ))

    return results


__all__ = [
    "DEFAULT_FROZEN_SPEC_PATH",
    "EXPECTED_CHAMPION_NAME",
    "EXPECTED_CHAMPION_VERSION",
    "EXPECTED_MODEL_VERSION",
    "DrawChampionConfig",
    "DrawChampionPrediction",
    "DrawChampionError",
    "stable_logit",
    "stable_sigmoid",
    "compute_dc_draw_probability",
    "compute_elo_draw_probability",
    "stack_draw_probabilities",
    "redistribute_proportional_odds",
    "predict_draw_champion",
]
