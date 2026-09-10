"""E7 — Best Proven Combination: arm construction and diagnostics.

APPROVED ARCHITECTURE (E7_AUDIT_REPORT.md, confirmed by the user)
-----------------------------------------------------------------
    ARM A   P_A = P_V3                                  87 features
    ARM B   P_B = w * P_market + (1-w) * P_V3           E2 blend, verbatim
    ARM C   P_C from a 91-column refit                  E6 features, verbatim
    ARM D   P_D = w * P_market + (1-w) * P_C            E6 then E2's blend

Arm D is Arm C's output passed through Arm B's blend. Each validated
component keeps its own architecture:

  - the market NEVER enters a Poisson design matrix (E2 was a post-hoc
    probability blend, not a regression feature -- reinterpreting it is
    explicitly forbidden)
  - the A/D states NEVER enter the blend as probabilities; they are
    design-matrix columns, exactly as E6 validated them

There is only ONE weight, w, and it is E2's existing one. No second
combination parameter is introduced.

MARKET-ALONE is carried as a reference column, not a fifth arm. E2
showed market alone (0.955905) beat the E2 blend (0.957130), so it is
the incumbent any combination must actually clear.

ZERO-COMPONENT CONTROLS
    w = 0                 ->  Arm D reduces to Arm C
    A/D states disabled   ->  Arm D reduces to Arm B
    both disabled         ->  Arm A

DEPENDENCIES: numpy / pandas only. No scipy, no sklearn.
DETERMINISM: pure functions. No RNG anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASS_ORDER: tuple[str, str, str] = ("H", "D", "A")

#: E2's blend weight grid, verbatim from run_e2_experiment.py.
WEIGHT_GRID: tuple[float, ...] = tuple(round(x, 2)
                                       for x in np.arange(0.0, 1.001, 0.05))

#: Tolerance for the zero-component controls.
CONTROL_TOL: float = 1e-12


class BestCombinationError(ValueError):
    """Raised when an E7 safety gate fails."""


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def _onehot(y: np.ndarray) -> np.ndarray:
    o = np.zeros((len(y), 3), dtype=float)
    for i, c in enumerate(CLASS_ORDER):
        o[:, i] = (y == c)
    return o


def log_loss(y: np.ndarray, P: np.ndarray) -> float:
    Pc = np.clip(P, 1e-15, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(_onehot(y) * np.log(Pc), axis=1)))


def per_fixture_log_loss(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Per-fixture log-loss contribution. Used by the complementarity
    diagnostic only -- never for tuning."""
    Pc = np.clip(P, 1e-15, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return -np.sum(_onehot(y) * np.log(Pc), axis=1)


def validate_probs(P: np.ndarray, name: str = "probs") -> None:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise BestCombinationError(f"{name}: expected (n, 3), got {P.shape}")
    if not np.all(np.isfinite(P)):
        raise BestCombinationError(f"{name}: non-finite values")
    if np.any(P < 0) or np.any(P > 1):
        raise BestCombinationError(f"{name}: values outside [0, 1]")
    s = P.sum(axis=1)
    if not np.allclose(s, 1.0, atol=1e-9):
        bad = int(np.argmax(np.abs(s - 1.0)))
        raise BestCombinationError(
            f"{name}: rows do not sum to 1 (worst {s[bad]:.12f})")


# ---------------------------------------------------------------------------
# E2 blend — reproduced verbatim
# ---------------------------------------------------------------------------

def blend(p_base: np.ndarray, p_market: np.ndarray, w: float) -> np.ndarray:
    """P_final = w * P_market + (1 - w) * P_base, renormalized.

    Identical in form to research/market_odds/run_e2_experiment.py::blend.
    `p_base` is P_V3 for Arm B and P_C for Arm D -- that substitution is
    the ONLY difference between the two arms.
    """
    if not (0.0 <= w <= 1.0):
        raise BestCombinationError(f"w must be in [0, 1], got {w}")
    out = w * np.asarray(p_market, float) + (1.0 - w) * np.asarray(p_base, float)
    return out / out.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class WeightSelection:
    """Outcome of a training-only blend-weight search."""
    w: float
    train_log_loss: float
    curve: list[dict]
    n_train: int


def learn_weight(y_train: np.ndarray, p_base_train: np.ndarray,
                 p_market_train: np.ndarray,
                 grid: tuple[float, ...] = WEIGHT_GRID) -> WeightSelection:
    """Select w minimizing TRAINING log loss over E2's fixed grid.

    CAUSALITY: the signature carries training arrays only. No test-fold
    outcome or probability can reach it. Identical procedure to E2's
    learn_weight(); only the `p_base` argument differs between arms.
    """
    if len(y_train) == 0:
        raise BestCombinationError("Cannot learn w on an empty training set")

    curve, best_w, best = [], None, float("inf")
    for w in grid:
        ll = log_loss(y_train, blend(p_base_train, p_market_train, w))
        curve.append({"w": w, "train_log_loss": round(ll, 8)})
        if ll < best - 1e-12:
            best, best_w = ll, w
    return WeightSelection(w=best_w, train_log_loss=best, curve=curve,
                           n_train=int(len(y_train)))


# ---------------------------------------------------------------------------
# Zero-component controls
# ---------------------------------------------------------------------------

def control_market_disabled(p_c: np.ndarray, p_market: np.ndarray) -> dict:
    """CONTROL 1 — w = 0 must reduce Arm D to Arm C."""
    p_d0 = blend(p_c, p_market, 0.0)
    d = float(np.max(np.abs(p_d0 - np.asarray(p_c, float))))
    return {"control": "market_disabled_w0_equals_arm_c",
            "max_abs_diff": d, "tolerance": CONTROL_TOL,
            "passed": bool(d <= CONTROL_TOL)}


def control_ad_disabled(p_b: np.ndarray, p_c_zero_state: np.ndarray,
                        p_market: np.ndarray, w: float) -> dict:
    """CONTROL 2 — zero A/D states must reduce Arm D to Arm B.

    `p_c_zero_state` is Arm C refitted with all four state columns set to
    zero; it should already equal P_V3, so blending it at the same w must
    equal Arm B.
    """
    p_d = blend(p_c_zero_state, p_market, w)
    d = float(np.max(np.abs(p_d - np.asarray(p_b, float))))
    return {"control": "ad_disabled_equals_arm_b",
            "max_abs_diff": d, "tolerance": CONTROL_TOL,
            "passed": bool(d <= CONTROL_TOL)}


def control_both_disabled(p_a: np.ndarray, p_c_zero_state: np.ndarray,
                          p_market: np.ndarray) -> dict:
    """CONTROL 3 — zero states and w = 0 must reproduce Arm A."""
    p_d = blend(p_c_zero_state, p_market, 0.0)
    d = float(np.max(np.abs(p_d - np.asarray(p_a, float))))
    return {"control": "both_disabled_equals_arm_a",
            "max_abs_diff": d, "tolerance": CONTROL_TOL,
            "passed": bool(d <= CONTROL_TOL)}


# ---------------------------------------------------------------------------
# Complementarity diagnostics (descriptive only — never used for tuning)
# ---------------------------------------------------------------------------

def complementarity(y: np.ndarray, p_a: np.ndarray, p_market: np.ndarray,
                    p_b: np.ndarray, p_c: np.ndarray, p_d: np.ndarray,
                    material: float = 0.01) -> dict:
    """Per-fixture error decomposition across the arms.

    DESCRIPTIVE ONLY. Nothing here feeds model fitting or selection.
    """
    ll_a = per_fixture_log_loss(y, p_a)
    ll_m = per_fixture_log_loss(y, p_market)
    ll_c = per_fixture_log_loss(y, p_c)
    ll_d = per_fixture_log_loss(y, p_d)

    oh = _onehot(y)
    err_m = np.abs(np.asarray(p_market, float) - oh).mean(axis=1)
    err_c = np.abs(np.asarray(p_c, float) - oh).mean(axis=1)

    market_helps = ll_m < ll_a          # market beats V3 on this fixture
    e6_helps = ll_c < ll_a              # E6 beats V3 on this fixture

    n = len(y)
    return {
        "corr_logloss_market_vs_e6": round(float(np.corrcoef(ll_m, ll_c)[0, 1]), 6),
        "corr_logloss_v3_vs_market": round(float(np.corrcoef(ll_a, ll_m)[0, 1]), 6),
        "corr_logloss_v3_vs_e6": round(float(np.corrcoef(ll_a, ll_c)[0, 1]), 6),
        "corr_prob_error_market_vs_e6": round(float(np.corrcoef(err_m, err_c)[0, 1]), 6),
        "pct_market_helps_vs_v3": round(100 * float(market_helps.mean()), 4),
        "pct_e6_helps_vs_v3": round(100 * float(e6_helps.mean()), 4),
        "pct_both_help": round(100 * float((market_helps & e6_helps).mean()), 4),
        "pct_both_hurt": round(100 * float((~market_helps & ~e6_helps).mean()), 4),
        "pct_market_helps_e6_hurts": round(
            100 * float((market_helps & ~e6_helps).mean()), 4),
        "pct_e6_helps_market_hurts": round(
            100 * float((~market_helps & e6_helps).mean()), 4),
        "pct_e6_materially_changes_pc": round(
            100 * float(np.mean(np.abs(np.asarray(p_c, float)
                                       - np.asarray(p_a, float)).max(axis=1)
                                > material)), 4),
        "pct_market_materially_changes_pd": round(
            100 * float(np.mean(np.abs(np.asarray(p_d, float)
                                       - np.asarray(p_c, float)).max(axis=1)
                                > material)), 4),
        "pct_top_class_changed_c_vs_a": round(
            100 * float(np.mean(np.asarray(p_a).argmax(1)
                                != np.asarray(p_c).argmax(1))), 4),
        "pct_top_class_changed_d_vs_c": round(
            100 * float(np.mean(np.asarray(p_c).argmax(1)
                                != np.asarray(p_d).argmax(1))), 4),
        "pct_top_class_changed_d_vs_b": round(
            100 * float(np.mean(np.asarray(p_b).argmax(1)
                                != np.asarray(p_d).argmax(1))), 4),
        "mean_logloss_d_minus_c": round(float((ll_d - ll_c).mean()), 8),
        "material_threshold": material,
        "n": int(n),
    }


def pairwise_deltas(metrics: dict[str, dict]) -> dict:
    """All required pairwise comparisons. Positive = first arm better."""
    def d(x, yk, key):
        return round(metrics[yk][key] - metrics[x][key], 6)

    pairs = [("B", "A"), ("C", "A"), ("D", "A"), ("D", "B"), ("D", "C"),
             ("MARKET", "A"), ("MARKET", "B"), ("MARKET", "C"),
             ("D", "MARKET")]
    out = {}
    for better, base in pairs:
        out[f"{better}_vs_{base}"] = {
            k: d(better, base, k)
            for k in ("log_loss", "brier", "rps", "ece")
            if k in metrics[better] and k in metrics[base]
        }
    return out


__all__ = [
    "CLASS_ORDER", "WEIGHT_GRID", "CONTROL_TOL", "BestCombinationError",
    "log_loss", "per_fixture_log_loss", "validate_probs",
    "blend", "WeightSelection", "learn_weight",
    "control_market_disabled", "control_ad_disabled", "control_both_disabled",
    "complementarity", "pairwise_deltas",
]
