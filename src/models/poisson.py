"""Production Poisson conversion: lambda -> H/D/A probabilities.

Extracted from run_step_poisson_3_first_experiment.py (the validated
STEP 3 script, md5 24827e92bc0acb04b9fc1dbe5a9958ae).

Every function here is MATHEMATICALLY IDENTICAL to its STEP 3 origin.
The only changes are:

  1. Module-level imports instead of inline.
  2. Docstrings expanded for production use.
  3. No experiment-specific helpers (snapshot, check, abort, etc.).

NOTHING about the numerical logic -- grid sizing, PMF computation,
cumulative-sum P(H), diagonal P(D), complement P(A), or the modal
scoreline factorisation -- has been changed, tuned, or "improved".

If a future change is needed, it must be validated against the pinned
STEP 3 implementation before deployment.

This module fits nothing, evaluates nothing, and writes nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Tail tolerance, identical to STEP 3's TAIL_TOL = 1e-15.
TAIL_TOL: float = 1e-15

#: Safety tolerances for production assertions.
MASS_TOL: float = 1e-12
COMPLEMENT_TOL: float = 1e-10
ROWSUM_TOL: float = 1e-9

#: STEP 3 script identity, for traceability.
STEP3_SOURCE: str = "run_step_poisson_3_first_experiment.py"
STEP3_MD5: str = "24827e92bc0acb04b9fc1dbe5a9958ae"


class PoissonConversionError(RuntimeError):
    """Raised when a production safety gate in the conversion fails."""


def _grid_size(lam_max: float, tol: float = TAIL_TOL, cap: int = 20000) -> int:
    """Smallest K with Poisson(lam_max) upper-tail P(X > K) < tol.

    Computed by exact forward CDF recursion (p_k = p_{k-1} * lam / k),
    so the grid length is DERIVED from the data rather than assumed.
    """
    k, term = 0, math.exp(-lam_max)
    cdf = term
    while cdf < 1.0 - tol and k < cap:
        k += 1
        term *= lam_max / k
        cdf += term
    return max(k, 1)


def _pmf_grid(lam: np.ndarray, K: int) -> np.ndarray:
    """Poisson PMF for 0..K, computed in log space for numerical stability.

    Args:
        lam: array of shape (n,) -- Poisson rate parameters.
        K: maximum goal count (grid runs 0..K inclusive).

    Returns:
        Array of shape (n, K+1).
    """
    k = np.arange(K + 1)
    log_fact = np.array([math.lgamma(i + 1) for i in k])
    l = np.asarray(lam, dtype=float)[:, None]
    return np.exp(-l + k * np.log(l) - log_fact)


def hda_tail_safe(
    lam_h: np.ndarray,
    lam_a: np.ndarray,
    tol: float = TAIL_TOL,
) -> tuple[np.ndarray, int, float, float]:
    """P(H), P(D), P(A) under independent Poisson -- tail-safe.

    Returns (P, K, residual_mass, complement_control) where:
      P             -- (n, 3) array, columns in CLASS_ORDER [H, D, A].
      K             -- adaptive grid size used.
      residual_mass -- max marginal tail mass lost (should be < tol).
      complement_control -- max |P(A) direct - P(A) complement|.

    The implementation is identical to STEP 3's hda_tail_safe.
    """
    K = _grid_size(float(max(np.max(lam_h), np.max(lam_a))), tol)
    ph, pa = _pmf_grid(lam_h, K), _pmf_grid(lam_a, K)
    residual = float(max(np.max(np.abs(1.0 - ph.sum(axis=1))),
                         np.max(np.abs(1.0 - pa.sum(axis=1)))))
    Fa = np.cumsum(pa, axis=1)                        # P(away <= j)
    p_home = (ph[:, 1:] * Fa[:, :-1]).sum(axis=1)
    p_draw = (ph * pa).sum(axis=1)
    p_away_direct = (pa[:, 1:] * np.cumsum(ph, axis=1)[:, :-1]).sum(axis=1)
    p_away = 1.0 - p_home - p_draw
    control = float(np.max(np.abs(p_away - p_away_direct)))
    return np.column_stack([p_home, p_draw, p_away]), K, residual, control


def modal_scoreline(
    lam_h: np.ndarray,
    lam_a: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Most likely scoreline and its probability.

    For independent Poisson the joint PMF factorises, so the argmax of
    the joint equals the pair of marginal argmaxes.

    Returns (home_goals, away_goals, probability) -- each shape (n,).
    """
    ph, pa = _pmf_grid(lam_h, K), _pmf_grid(lam_a, K)
    i, j = ph.argmax(axis=1), pa.argmax(axis=1)
    return i, j, ph[np.arange(len(i)), i] * pa[np.arange(len(j)), j]


@dataclass(frozen=True)
class PoissonPrediction:
    """Complete production prediction from the V2 Poisson model."""
    prediction: str
    probabilities: dict[str, float]
    expected_goals_home: float
    expected_goals_away: float
    modal_scoreline: str
    modal_scoreline_probability: float
    lambda_home: float
    lambda_away: float
    grid_K: int
    tail_residual: float
    complement_control: float

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "probabilities": dict(self.probabilities),
            "expected_goals_home": self.expected_goals_home,
            "expected_goals_away": self.expected_goals_away,
            "modal_scoreline": self.modal_scoreline,
            "modal_scoreline_probability": self.modal_scoreline_probability,
        }


def predict_poisson(
    lam_h: np.ndarray,
    lam_a: np.ndarray,
    class_order: list[str],
    tol: float = TAIL_TOL,
) -> list[PoissonPrediction]:
    """Full Poisson prediction pipeline: lambdas -> production predictions.

    Args:
        lam_h: home expected goals, shape (n,).
        lam_a: away expected goals, shape (n,).
        class_order: e.g. ['H', 'D', 'A'].
        tol: tail tolerance.

    Returns:
        List of n PoissonPrediction objects.

    Raises:
        PoissonConversionError on any safety gate failure.
    """
    lam_h = np.asarray(lam_h, dtype=float)
    lam_a = np.asarray(lam_a, dtype=float)

    if not np.all(np.isfinite(lam_h)) or not np.all(np.isfinite(lam_a)):
        raise PoissonConversionError("Non-finite lambda values")
    if np.any(lam_h <= 0) or np.any(lam_a <= 0):
        raise PoissonConversionError("Lambda values must be > 0")

    P, K, residual, control = hda_tail_safe(lam_h, lam_a, tol)

    if residual > MASS_TOL:
        raise PoissonConversionError(
            f"Marginal tail residual {residual:.6e} exceeds tolerance {MASS_TOL}")
    if control > COMPLEMENT_TOL:
        raise PoissonConversionError(
            f"Complement control {control:.6e} exceeds tolerance {COMPLEMENT_TOL}")
    row_sums = P.sum(axis=1)
    if not np.all(np.abs(row_sums - 1.0) <= ROWSUM_TOL):
        raise PoissonConversionError(
            f"Row sums deviate from 1.0 by up to {np.max(np.abs(row_sums - 1.0)):.6e}")
    if np.any(P < 0) or np.any(P > 1):
        raise PoissonConversionError("Probabilities outside [0, 1]")

    sh, sa, sp = modal_scoreline(lam_h, lam_a, K)

    results = []
    for i in range(len(lam_h)):
        pred_idx = int(P[i].argmax())
        pred_class = class_order[pred_idx]
        probs = {cls: float(P[i, j]) for j, cls in enumerate(class_order)}
        results.append(PoissonPrediction(
            prediction=pred_class,
            probabilities=probs,
            expected_goals_home=float(lam_h[i]),
            expected_goals_away=float(lam_a[i]),
            modal_scoreline=f"{int(sh[i])}-{int(sa[i])}",
            modal_scoreline_probability=float(sp[i]),
            lambda_home=float(lam_h[i]),
            lambda_away=float(lam_a[i]),
            grid_K=K,
            tail_residual=residual,
            complement_control=control,
        ))
    return results


__all__ = [
    "TAIL_TOL", "STEP3_SOURCE", "STEP3_MD5",
    "PoissonConversionError", "PoissonPrediction",
    "hda_tail_safe", "modal_scoreline", "predict_poisson",
]
