"""E3 — Dixon-Coles low-score correction applied to FIXED V3 lambdas.

SCOPE — read this before using anything here.

This module does NOT implement the full Dixon-Coles model. The full DC
model jointly fits attack/defence/home-advantage/rho by maximum
likelihood (see research/worldcup_repo/src/dixon_coles.py for such an
implementation, built for international football).

E3 tests ONE thing: the low-score dependence correction tau(x, y)
applied to the score grid implied by V3's EXISTING lambda_home and
lambda_away. V3's lambdas are never recomputed, refitted, or adjusted.

    A:  V3 lambdas -> independent Poisson       -> 1X2
    B:  V3 lambdas -> DC-corrected score grid   -> 1X2

The tau correction (Dixon & Coles 1997):

    tau(0,0) = 1 - lam_h * lam_a * rho
    tau(1,0) = 1 + lam_a * rho
    tau(0,1) = 1 + lam_h * rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1                     otherwise

Indexing convention: index [i, j] is home i goals, away j goals.

WHAT THIS MODULE DOES NOT DO:
  - Fit, train, or tune V3
  - Estimate attack/defence ratings
  - Apply time decay
  - Read or write any production artifact or database
  - Touch market odds

DEPENDENCIES: numpy only. No scipy, no sklearn.

DETERMINISM: every function here is a pure function of its arguments.
No RNG is used anywhere in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Tail tolerance — identical to src/models/poisson.py TAIL_TOL.
TAIL_TOL: float = 1e-15

#: Row-sum tolerance for the returned 1X2 probabilities.
ROWSUM_TOL: float = 1e-9

#: Minimum allowed tau value. tau <= 0 makes the joint PMF invalid.
TAU_FLOOR: float = 1e-9

#: Class order, fixed for the whole E3 experiment.
CLASS_ORDER: tuple[str, str, str] = ("H", "D", "A")


class DixonColesError(RuntimeError):
    """Raised when a DC safety gate fails."""


# ---------------------------------------------------------------------------
# Grid construction — mirrors src/models/poisson.py exactly
# ---------------------------------------------------------------------------

def grid_size(lam_max: float, tol: float = TAIL_TOL, cap: int = 20000) -> int:
    """Smallest K with Poisson(lam_max) upper-tail P(X > K) < tol.

    Byte-for-byte the same recursion as src/models/poisson.py::_grid_size.
    Reproduced here (rather than imported) so this research module has no
    import dependency on production code, while remaining verifiably
    identical — test_dixon_coles.py asserts equality against the
    production function across a wide lambda range.
    """
    k, term = 0, math.exp(-lam_max)
    cdf = term
    while cdf < 1.0 - tol and k < cap:
        k += 1
        term *= lam_max / k
        cdf += term
    return max(k, 1)


def pmf_grid(lam: np.ndarray, K: int) -> np.ndarray:
    """Poisson PMF for 0..K in log space. Shape (n, K+1).

    Same computation as src/models/poisson.py::_pmf_grid.
    """
    k = np.arange(K + 1)
    log_fact = np.array([math.lgamma(i + 1) for i in k])
    l = np.asarray(lam, dtype=float)[:, None]
    return np.exp(-l + k * np.log(l) - log_fact)


# ---------------------------------------------------------------------------
# The tau correction
# ---------------------------------------------------------------------------

def tau_matrix(lam_h: np.ndarray, lam_a: np.ndarray, rho: float,
               K: int) -> np.ndarray:
    """Build the (n, K+1, K+1) tau correction tensor.

    Only the four cells (0,0), (1,0), (0,1), (1,1) differ from 1.0.
    Every other cell is exactly 1.0.
    """
    lam_h = np.asarray(lam_h, dtype=float)
    lam_a = np.asarray(lam_a, dtype=float)
    n = len(lam_h)

    tau = np.ones((n, K + 1, K + 1), dtype=float)
    if K >= 1:
        tau[:, 0, 0] = 1.0 - lam_h * lam_a * rho
        tau[:, 1, 0] = 1.0 + lam_a * rho
        tau[:, 0, 1] = 1.0 + lam_h * rho
        tau[:, 1, 1] = 1.0 - rho
    else:
        tau[:, 0, 0] = 1.0 - lam_h * lam_a * rho
    return tau


def rho_validity_bounds(lam_h: np.ndarray,
                        lam_a: np.ndarray) -> tuple[float, float]:
    """Return (rho_min, rho_max) keeping all four tau cells strictly positive.

        tau(0,0) > 0  ->  rho <  1 / (lam_h * lam_a)
        tau(1,1) > 0  ->  rho <  1
        tau(1,0) > 0  ->  rho > -1 / lam_a
        tau(0,1) > 0  ->  rho > -1 / lam_h

    Bounds are taken over the whole batch, so a rho inside them is valid
    for every fixture supplied.
    """
    lam_h = np.asarray(lam_h, dtype=float)
    lam_a = np.asarray(lam_a, dtype=float)
    lo = float(max(np.max(-1.0 / lam_h), np.max(-1.0 / lam_a)))
    hi = float(min(1.0, np.min(1.0 / (lam_h * lam_a))))
    return lo, hi


# ---------------------------------------------------------------------------
# Score grid and 1X2 conversion
# ---------------------------------------------------------------------------

def score_grid(lam_h: np.ndarray, lam_a: np.ndarray, rho: float,
               K: int) -> np.ndarray:
    """DC-corrected, renormalized joint score distribution.

    Returns shape (n, K+1, K+1); entry [n, i, j] = P(home i, away j).
    Each fixture's grid sums to exactly 1.

    rho = 0.0 reduces to the independent Poisson grid (tau == 1), so this
    function is a strict superset of the baseline.
    """
    ph = pmf_grid(lam_h, K)          # (n, K+1)
    pa = pmf_grid(lam_a, K)          # (n, K+1)
    joint = ph[:, :, None] * pa[:, None, :]

    if rho != 0.0:
        tau = tau_matrix(lam_h, lam_a, rho, K)
        if np.any(tau <= 0.0):
            raise DixonColesError(
                f"rho={rho} produces non-positive tau; "
                f"valid range for this batch is {rho_validity_bounds(lam_h, lam_a)}"
            )
        joint = joint * tau

    joint = np.clip(joint, 0.0, None)
    totals = joint.sum(axis=(1, 2), keepdims=True)
    if np.any(totals <= 0) or not np.all(np.isfinite(totals)):
        raise DixonColesError("Score grid has non-positive or non-finite mass")
    return joint / totals


def hda_from_grid(joint: np.ndarray) -> np.ndarray:
    """Collapse a joint score grid to (n, 3) [P(H), P(D), P(A)].

    P(H) = mass strictly below the diagonal (home > away)
    P(D) = diagonal mass
    P(A) = mass strictly above the diagonal (away > home)
    """
    n, r, c = joint.shape
    i = np.arange(r)[:, None]
    j = np.arange(c)[None, :]
    home_mask = (i > j)
    draw_mask = (i == j)
    away_mask = (i < j)

    p_h = (joint * home_mask).sum(axis=(1, 2))
    p_d = (joint * draw_mask).sum(axis=(1, 2))
    p_a = (joint * away_mask).sum(axis=(1, 2))

    P = np.column_stack([p_h, p_d, p_a])
    row_sums = P.sum(axis=1)
    if not np.all(np.abs(row_sums - 1.0) <= ROWSUM_TOL):
        worst = float(np.max(np.abs(row_sums - 1.0)))
        raise DixonColesError(f"1X2 row sums deviate from 1 by up to {worst:.3e}")
    return P


def predict_dc(lam_h: np.ndarray, lam_a: np.ndarray, rho: float,
               K: int | None = None) -> tuple[np.ndarray, int]:
    """Full pipeline: fixed lambdas + rho -> (n, 3) 1X2 probabilities.

    Returns (P, K). If K is None it is derived adaptively from the batch
    maximum lambda, exactly as production does.
    """
    lam_h = np.asarray(lam_h, dtype=float)
    lam_a = np.asarray(lam_a, dtype=float)

    if not np.all(np.isfinite(lam_h)) or not np.all(np.isfinite(lam_a)):
        raise DixonColesError("Non-finite lambda values")
    if np.any(lam_h <= 0) or np.any(lam_a <= 0):
        raise DixonColesError("Lambda values must be > 0")

    if K is None:
        K = grid_size(float(max(np.max(lam_h), np.max(lam_a))))

    joint = score_grid(lam_h, lam_a, rho, K)
    P = hda_from_grid(joint)

    if not np.all(np.isfinite(P)):
        raise DixonColesError("Non-finite 1X2 probabilities")
    if np.any(P < 0.0) or np.any(P > 1.0):
        raise DixonColesError("1X2 probabilities outside [0, 1]")
    return P, K


def low_score_probs(lam_h: np.ndarray, lam_a: np.ndarray, rho: float,
                    K: int) -> dict[str, np.ndarray]:
    """Probabilities of the four cells DC modifies, plus their total."""
    joint = score_grid(lam_h, lam_a, rho, K)
    return {
        "p_0_0": joint[:, 0, 0],
        "p_1_0": joint[:, 1, 0],
        "p_0_1": joint[:, 0, 1],
        "p_1_1": joint[:, 1, 1],
        "p_low_total": (joint[:, 0, 0] + joint[:, 1, 0]
                        + joint[:, 0, 1] + joint[:, 1, 1]),
    }


# ---------------------------------------------------------------------------
# rho selection — TRAINING DATA ONLY
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RhoSelection:
    """Outcome of a training-only rho search."""
    rho: float
    train_log_loss: float
    curve: list[dict]
    grid: list[float]
    n_train: int
    valid_bounds: tuple[float, float]


def _log_loss(y: np.ndarray, P: np.ndarray) -> float:
    """Multiclass log loss with CLASS_ORDER columns."""
    oh = np.zeros((len(y), 3), dtype=float)
    for i, cls in enumerate(CLASS_ORDER):
        oh[:, i] = (y == cls).astype(float)
    Pc = np.clip(P, 1e-15, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(oh * np.log(Pc), axis=1)))


def select_rho(
    y_train: np.ndarray,
    lam_h_train: np.ndarray,
    lam_a_train: np.ndarray,
    rho_grid: list[float],
    K: int,
) -> RhoSelection:
    """Choose rho minimizing TRAINING log loss over a fixed grid.

    CAUSALITY: this function's signature contains no test-fold data. It
    physically cannot see a test outcome. The selected rho is returned to
    the caller, which freezes it before touching the test fold.

    Candidate rho values outside the batch validity bounds are skipped
    (recorded as invalid in the curve) rather than silently clipped.
    """
    lo, hi = rho_validity_bounds(lam_h_train, lam_a_train)

    curve: list[dict] = []
    best_rho: float | None = None
    best_ll = float("inf")

    for rho in rho_grid:
        if not (lo < rho < hi):
            curve.append({"rho": rho, "train_log_loss": None,
                          "status": "invalid_out_of_bounds"})
            continue
        P, _ = predict_dc(lam_h_train, lam_a_train, rho, K=K)
        ll = _log_loss(y_train, P)
        curve.append({"rho": rho, "train_log_loss": round(ll, 8),
                      "status": "ok"})
        if ll < best_ll - 1e-12:
            best_ll, best_rho = ll, rho

    if best_rho is None:
        raise DixonColesError(
            f"No valid rho in grid {rho_grid} for bounds ({lo:.4f}, {hi:.4f})"
        )

    return RhoSelection(
        rho=best_rho,
        train_log_loss=best_ll,
        curve=curve,
        grid=list(rho_grid),
        n_train=int(len(y_train)),
        valid_bounds=(lo, hi),
    )


__all__ = [
    "CLASS_ORDER", "TAIL_TOL", "TAU_FLOOR", "DixonColesError",
    "grid_size", "pmf_grid", "tau_matrix", "rho_validity_bounds",
    "score_grid", "hda_from_grid", "predict_dc", "low_score_probs",
    "RhoSelection", "select_rho",
]
