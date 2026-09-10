"""E8 — Post-hoc temperature scaling of a 1X2 probability vector.

    P'_k = P_k^(1/T) / sum_j P_j^(1/T)

implemented in log space for numerical stability:

    z_k  = log(P_k) / T
    P'_k = softmax(z)          with max-subtraction before exp

    T < 1  sharpen      T = 1  identity      T > 1  soften

SCOPE
-----
This module touches ONLY the final probability vector. It knows nothing
about football, features, models or fixtures. It cannot change the
underlying model because it never sees it.

IDENTITY AT T = 1
-----------------
`apply_temperature` short-circuits at T == 1.0 and returns a copy of the
input unchanged. No log, no epsilon, no exp is applied on that path, so
the identity holds EXACTLY (max abs diff 0.0), not merely within
tolerance. This matters because the epsilon below would otherwise
perturb the identity in the last bits.

EPSILON — documented, per the spec's requirement not to add one silently
-----------------------------------------------------------------------
log(0) is -inf. Probabilities are clipped to EPS = 1e-300 before the
log. That is the smallest value keeping log() finite in float64, and it
sits ~290 orders of magnitude below anything this pipeline produces
(the observed minimum across the market dataset is ~1e-3). It therefore
never binds in practice; it exists only so a pathological zero cannot
produce NaN.

RANKING
-------
Temperature scaling is a strictly monotone transform of each component
followed by a common normalisation, so it preserves the ordering of
classes whenever all probabilities are positive. `ranking_preserved()`
verifies this rather than assuming it.

DEPENDENCIES: numpy only. No scipy, no sklearn.
DETERMINISM: pure functions. No RNG anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASS_ORDER: tuple[str, str, str] = ("H", "D", "A")

#: Frozen temperature grid. Pre-registered before any outer-test result
#: was inspected. DO NOT EXPAND after seeing results.
TEMPERATURE_GRID: tuple[float, ...] = (
    0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20,
)

#: Documented clip guarding log(0). See module docstring.
EPS: float = 1e-300

#: Identity control tolerance.
IDENTITY_TOL: float = 1e-12


class TemperatureError(ValueError):
    """Raised when a temperature-scaling safety gate fails."""


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature T to an (n, 3) probability matrix.

    T == 1.0 returns an unchanged copy via an explicit short-circuit, so
    the identity control is exact.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise TemperatureError(f"expected (n, 3) probabilities, got {P.shape}")
    if not np.all(np.isfinite(P)):
        raise TemperatureError("non-finite input probabilities")
    if np.any(P < 0):
        raise TemperatureError("negative input probabilities")
    if T <= 0:
        raise TemperatureError(f"T must be > 0, got {T}")

    if T == 1.0:
        return P.copy()

    z = np.log(np.clip(P, EPS, None)) / T
    z = z - z.max(axis=1, keepdims=True)      # stabilise before exp
    e = np.exp(z)
    denom = e.sum(axis=1, keepdims=True)
    if np.any(denom <= 0) or not np.all(np.isfinite(denom)):
        raise TemperatureError("temperature scaling produced invalid mass")
    out = e / denom

    if not np.all(np.isfinite(out)):
        raise TemperatureError("non-finite output probabilities")
    return out


def validate_probs(P: np.ndarray, name: str = "probs",
                   atol: float = 1e-9) -> None:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise TemperatureError(f"{name}: expected (n, 3), got {P.shape}")
    if not np.all(np.isfinite(P)):
        raise TemperatureError(f"{name}: non-finite values")
    if np.any(P < 0) or np.any(P > 1):
        raise TemperatureError(f"{name}: values outside [0, 1]")
    s = P.sum(axis=1)
    if not np.allclose(s, 1.0, atol=atol):
        worst = float(np.max(np.abs(s - 1.0)))
        raise TemperatureError(f"{name}: rows deviate from 1 by up to {worst:.3e}")


def ranking_preserved(P: np.ndarray, P_scaled: np.ndarray) -> bool:
    """True if per-row class ordering is identical in both matrices."""
    a = np.argsort(np.asarray(P, float), axis=1)
    b = np.argsort(np.asarray(P_scaled, float), axis=1)
    return bool(np.array_equal(a, b))


# ---------------------------------------------------------------------------
# Metrics (self-contained: research isolation, no import from src/)
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


def brier(y: np.ndarray, P: np.ndarray) -> float:
    return float(np.mean(np.sum((P - _onehot(y)) ** 2, axis=1)))


def rps(y: np.ndarray, P: np.ndarray) -> float:
    cp, co = np.cumsum(P, axis=1), np.cumsum(_onehot(y), axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


def accuracy(y: np.ndarray, P: np.ndarray) -> float:
    pred = np.array([CLASS_ORDER[i] for i in np.asarray(P).argmax(1)])
    return float(np.mean(pred == y))


def draw_recall(y: np.ndarray, P: np.ndarray) -> float:
    pred = np.array([CLASS_ORDER[i] for i in np.asarray(P).argmax(1)])
    m = (y == "D")
    return float(np.mean(pred[m] == "D")) if m.sum() else float("nan")


def reliability_bins(y: np.ndarray, P: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Pooled reliability curve over all three classes."""
    oh, pf = _onehot(y).ravel(), np.asarray(P, float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (pf >= lo) & (pf <= hi) if i == n_bins - 1 else (pf >= lo) & (pf < hi)
        n = int(m.sum())
        if not n:
            out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": 0,
                        "mean_predicted": None, "observed_freq": None,
                        "gap": None})
            continue
        mp, ob = float(pf[m].mean()), float(oh[m].mean())
        out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": n,
                    "mean_predicted": round(mp, 6),
                    "observed_freq": round(ob, 6), "gap": round(ob - mp, 6)})
    return out


def ece(y: np.ndarray, P: np.ndarray, n_bins: int = 10) -> float:
    b = reliability_bins(y, P, n_bins)
    tot = sum(x["n"] for x in b)
    if not tot:
        return float("nan")
    return float(sum(x["n"] / tot * abs(x["gap"]) for x in b if x["n"]))


def max_calibration_error(y: np.ndarray, P: np.ndarray, n_bins: int = 10) -> float:
    b = [x for x in reliability_bins(y, P, n_bins) if x["n"]]
    return float(max(abs(x["gap"]) for x in b)) if b else float("nan")


def confidence_profile(P: np.ndarray) -> dict:
    """Top-class confidence distribution — the quantity T directly moves."""
    conf = np.asarray(P, float).max(axis=1)
    return {
        "mean_confidence": round(float(conf.mean()), 6),
        "max_confidence": round(float(conf.max()), 6),
        "min_confidence": round(float(conf.min()), 6),
        "p10": round(float(np.percentile(conf, 10)), 6),
        "p50": round(float(np.percentile(conf, 50)), 6),
        "p90": round(float(np.percentile(conf, 90)), 6),
    }


def all_metrics(y: np.ndarray, P: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "log_loss": round(log_loss(y, P), 6),
        "brier": round(brier(y, P), 6),
        "rps": round(rps(y, P), 6),
        "accuracy": round(accuracy(y, P), 6),
        "draw_recall": round(draw_recall(y, P), 6),
        "ece": round(ece(y, P), 6),
        "max_calibration_error": round(max_calibration_error(y, P), 6),
        **confidence_profile(P),
    }


# ---------------------------------------------------------------------------
# Temperature selection — TRAINING DATA ONLY
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemperatureSelection:
    """Outcome of a training-only temperature search."""
    T: float
    train_log_loss: float
    curve: list[dict]
    n_train: int
    grid: tuple[float, ...]
    method: str


def select_temperature(
    y_train: np.ndarray,
    P_train: np.ndarray,
    grid: tuple[float, ...] = TEMPERATURE_GRID,
    method: str = "outer_train_direct",
) -> TemperatureSelection:
    """Select T minimizing TRAINING log loss over the frozen grid.

    CAUSALITY: the signature accepts training arrays only. There is no
    parameter through which an outer-test label or probability could
    enter, so test data cannot influence T.
    """
    y_train = np.asarray(y_train)
    if len(y_train) == 0:
        raise TemperatureError("Cannot select T on an empty training set")

    curve, best_T, best = [], None, float("inf")
    for T in grid:
        ll = log_loss(y_train, apply_temperature(P_train, T))
        curve.append({"T": T, "train_log_loss": round(ll, 8)})
        if ll < best - 1e-12:
            best, best_T = ll, T
    return TemperatureSelection(T=best_T, train_log_loss=best, curve=curve,
                                n_train=int(len(y_train)), grid=tuple(grid),
                                method=method)


def select_temperature_nested(
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    P: np.ndarray,
    grid: tuple[float, ...] = TEMPERATURE_GRID,
) -> TemperatureSelection:
    """Select T by mean inner-validation log loss across nested splits.

    `splits` is a list of (train_idx, val_idx) built from the OUTER
    TRAINING period only. Used when the outer training period spans at
    least two seasons; otherwise the caller falls back to
    `select_temperature` on the outer training rows directly.
    """
    if not splits:
        raise TemperatureError("No inner splits supplied")

    curve, best_T, best = [], None, float("inf")
    n_val = 0
    for T in grid:
        per = []
        for _tr, va in splits:
            per.append(log_loss(y[va], apply_temperature(P[va], T)))
        mean_ll = float(np.mean(per))
        curve.append({"T": T, "mean_inner_log_loss": round(mean_ll, 8),
                      "per_split_log_loss": [round(v, 8) for v in per]})
        if mean_ll < best - 1e-12:
            best, best_T = mean_ll, T
    n_val = int(sum(len(va) for _tr, va in splits))
    return TemperatureSelection(T=best_T, train_log_loss=best, curve=curve,
                                n_train=n_val, grid=tuple(grid),
                                method="nested_inner_validation")


# ---------------------------------------------------------------------------
# Controls and diagnostics
# ---------------------------------------------------------------------------

def identity_control(y: np.ndarray, P: np.ndarray) -> dict:
    """T = 1.0 must reproduce the base exactly, on probabilities AND metrics."""
    P1 = apply_temperature(P, 1.0)
    d = float(np.max(np.abs(P1 - np.asarray(P, float))))
    mb, m1 = all_metrics(y, P), all_metrics(y, P1)
    pred_same = bool(np.array_equal(np.asarray(P).argmax(1), P1.argmax(1)))
    metric_same = all(
        mb[k] == m1[k] for k in ("log_loss", "brier", "rps", "ece", "accuracy")
    )
    return {
        "max_abs_prob_diff": d,
        "tolerance": IDENTITY_TOL,
        "probs_identical": bool(d <= IDENTITY_TOL),
        "predictions_identical": pred_same,
        "metrics_identical": metric_same,
        "base_metrics": mb,
        "t1_metrics": m1,
        "passed": bool(d <= IDENTITY_TOL and pred_same and metric_same),
    }


def temperature_effect(P_base: np.ndarray, P_scaled: np.ndarray,
                       T: float, material: float = 0.01) -> dict:
    """Quantify what the temperature actually did, and check direction."""
    Pb = np.asarray(P_base, float)
    Ps = np.asarray(P_scaled, float)
    dP = np.abs(Ps - Pb)
    conf_b = Pb.max(axis=1).mean()
    conf_s = Ps.max(axis=1).mean()

    if T < 1.0:
        direction, expected = "sharpen", conf_s > conf_b
    elif T > 1.0:
        direction, expected = "soften", conf_s < conf_b
    else:
        direction, expected = "identity", float(dP.max()) <= IDENTITY_TOL

    return {
        "T": T,
        "direction": direction,
        "direction_confirmed": bool(expected),
        "mean_abs_prob_change": round(float(dP.mean()), 8),
        "max_prob_change": round(float(dP.max()), 8),
        "pct_materially_changed": round(
            100 * float(np.mean(dP.max(axis=1) > material)), 4),
        "pct_top_class_changed": round(
            100 * float(np.mean(Pb.argmax(1) != Ps.argmax(1))), 4),
        "mean_confidence_base": round(float(conf_b), 6),
        "mean_confidence_scaled": round(float(conf_s), 6),
        "ranking_preserved": ranking_preserved(Pb, Ps),
        "material_threshold": material,
    }


def temperature_stability(selected: list[float],
                          grid: tuple[float, ...] = TEMPERATURE_GRID) -> dict:
    """Assess fold-to-fold stability and flag grid-boundary selections."""
    s = list(selected)
    lo, hi = min(grid), max(grid)
    at_boundary = [t for t in s if t in (lo, hi)]
    spread = max(s) - min(s)
    all_one = all(t == 1.0 for t in s)
    same = len(set(s)) == 1

    if all_one:
        assessment = "ALL T=1 — calibration declined to act"
    elif same:
        assessment = "STABLE — identical selection in every fold"
    elif spread <= 0.10:
        assessment = "REASONABLY STABLE — within one grid step"
    else:
        assessment = "UNSTABLE — selections vary by more than one grid step"

    return {
        "selected": s,
        "spread": round(spread, 6),
        "all_identity": all_one,
        "identical_across_folds": same,
        "n_at_grid_boundary": len(at_boundary),
        "boundary_warning": bool(at_boundary),
        "boundary_note": (
            f"{len(at_boundary)} fold(s) selected a grid endpoint "
            f"({lo} or {hi}); the grid may be too narrow. Reporting this "
            "rather than expanding the grid post hoc."
            if at_boundary else "No fold selected a grid endpoint."
        ),
        "assessment": assessment,
    }


__all__ = [
    "CLASS_ORDER", "TEMPERATURE_GRID", "EPS", "IDENTITY_TOL",
    "TemperatureError", "apply_temperature", "validate_probs",
    "ranking_preserved", "log_loss", "brier", "rps", "accuracy",
    "draw_recall", "reliability_bins", "ece", "max_calibration_error",
    "confidence_profile", "all_metrics",
    "TemperatureSelection", "select_temperature", "select_temperature_nested",
    "identity_control", "temperature_effect", "temperature_stability",
]
