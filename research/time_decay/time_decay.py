"""E4 — Exponential time-decay training weights.

    w_i = exp(-xi * DeltaDays_i)          DeltaDays_i = T_ref - t_i  >= 0

The decay weight is a TRAINING WEIGHT, never a model feature. The V3
87-column contract is untouched.

REFERENCE POINT: DeltaDays is measured against the PREDICTION-TIME
reference (the start of the test period), never against today's
real-world date. A run in 2026 and a run in 2030 produce identical
weights for the same fold.

xi IS NOT COPIED FROM ANY REPOSITORY. In particular the international
World Cup value xi = 0.001 is not used. The grid below is expressed in
half-lives and is selected on OUR domestic-league training data only.

DEPENDENCIES: numpy only. No scipy, no sklearn.
DETERMINISM: every function is pure. No RNG anywhere in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SECONDS_PER_DAY: float = 86400.0

#: Candidate half-lives in days. `None` == no decay (xi = 0).
#:
#: Rationale: a domestic league plays ~1 match/week/team, so a season is
#: roughly 280 days of fixtures. 60 days is about a third of a season
#: (very aggressive); 730 days is two full seasons (very mild). The grid
#: spans that range roughly geometrically, plus an explicit no-decay
#: control so the procedure can decline to apply decay at all.
HALF_LIFE_GRID_DAYS: tuple[float | None, ...] = (
    60.0, 90.0, 120.0, 180.0, 240.0, 365.0, 540.0, 730.0, None,
)


class TimeDecayError(ValueError):
    """Raised when a decay parameter or weight vector is invalid."""


# ---------------------------------------------------------------------------
# xi <-> half-life
# ---------------------------------------------------------------------------

def xi_from_half_life(half_life_days: float | None) -> float:
    """xi = ln(2) / half_life. `None` -> xi = 0 (no decay)."""
    if half_life_days is None:
        return 0.0
    if half_life_days <= 0:
        raise TimeDecayError(f"half_life_days must be > 0, got {half_life_days}")
    return math.log(2.0) / float(half_life_days)


def half_life_from_xi(xi: float) -> float | None:
    """half_life = ln(2) / xi. xi = 0 -> None (infinite half-life)."""
    if xi < 0:
        raise TimeDecayError(f"xi must be >= 0, got {xi}")
    if xi == 0.0:
        return None
    return math.log(2.0) / xi


def xi_grid() -> list[float]:
    """The candidate xi values, in the same order as HALF_LIFE_GRID_DAYS."""
    return [xi_from_half_life(h) for h in HALF_LIFE_GRID_DAYS]


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def delta_days(unix_train: np.ndarray, reference_unix: float) -> np.ndarray:
    """DeltaDays = (reference_unix - unix_train) / 86400, clipped at 0.

    The clip guards against a caller passing a training fixture at or
    after the reference; such a fixture would be a causality violation
    and is caught separately by `assert_causal`.
    """
    d = (float(reference_unix) - np.asarray(unix_train, dtype=float)) / SECONDS_PER_DAY
    return np.maximum(d, 0.0)


def assert_causal(unix_train: np.ndarray, reference_unix: float) -> None:
    """Hard-fail if any training fixture is at or after the reference time."""
    u = np.asarray(unix_train, dtype=float)
    bad = np.where(u >= float(reference_unix))[0]
    if len(bad):
        raise TimeDecayError(
            f"{len(bad)} training fixtures are at or after the prediction "
            f"reference time {reference_unix}. This is a causality violation."
        )


def decay_weights(unix_train: np.ndarray, reference_unix: float,
                  xi: float) -> np.ndarray:
    """w_i = exp(-xi * DeltaDays_i).

    xi = 0 returns an all-ones vector, reproducing unweighted training
    exactly (this is the zero-decay control).
    """
    if xi < 0:
        raise TimeDecayError(f"xi must be >= 0, got {xi}")
    dd = delta_days(unix_train, reference_unix)
    if xi == 0.0:
        w = np.ones_like(dd)
    else:
        w = np.exp(-xi * dd)
    if not np.all(np.isfinite(w)):
        raise TimeDecayError("Non-finite decay weights")
    if np.any(w <= 0.0):
        raise TimeDecayError(
            f"Non-positive decay weights (min {w.min():.3e}); "
            f"xi={xi} is too aggressive for this training span"
        )
    return w


def weight_profile(xi: float,
                   ages: tuple[int, ...] = (30, 90, 180, 365, 730)) -> dict:
    """Interpretable summary: effective weight at a set of ages in days."""
    return {
        "xi": xi,
        "half_life_days": half_life_from_xi(xi),
        "weights": {f"{a}d": round(float(math.exp(-xi * a)), 6) for a in ages},
    }


# ---------------------------------------------------------------------------
# Nested temporal selection of xi  (TRAINING PERIOD ONLY)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InnerSplit:
    """A temporal split strictly inside the outer training period."""
    inner_train_idx: np.ndarray
    inner_val_idx: np.ndarray
    inner_train_seasons: tuple[str, ...]
    inner_val_season: str
    reference_unix: float


def build_inner_splits(seasons: np.ndarray, unix: np.ndarray,
                       train_seasons: tuple[str, ...]) -> list[InnerSplit]:
    """Nested temporal splits inside the outer training seasons.

    For outer training seasons (s1, ..., sk) this yields k-1 inner
    splits: train on (s1..sj), validate on s(j+1). The outer test period
    is not visible to this function at all — it only receives the outer
    training seasons.

    The reference time for each inner split is the earliest kickoff in
    the inner validation season, matching how the outer fold's reference
    is defined.
    """
    ordered = tuple(train_seasons)
    splits: list[InnerSplit] = []
    for j in range(1, len(ordered)):
        tr_seasons = ordered[:j]
        va_season = ordered[j]
        tr_idx = np.where(np.isin(seasons, tr_seasons))[0]
        va_idx = np.where(seasons == va_season)[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        ref = float(np.min(unix[va_idx]))
        splits.append(InnerSplit(tr_idx, va_idx, tr_seasons, va_season, ref))
    return splits


@dataclass(frozen=True)
class XiSelection:
    """Outcome of a training-only xi search."""
    xi: float
    half_life_days: float | None
    mean_inner_log_loss: float
    curve: list[dict]
    n_inner_splits: int
    grid_half_lives: list[float | None]


def select_xi(
    inner_splits: list[InnerSplit],
    fit_predict,
    half_life_grid: tuple[float | None, ...] = HALF_LIFE_GRID_DAYS,
) -> XiSelection:
    """Select xi by mean log loss across nested temporal inner splits.

    CAUSALITY: this function receives only inner splits built from the
    outer TRAINING seasons. Its signature contains no outer-test data,
    so no outer-test label can reach it.

    Args:
        inner_splits: from build_inner_splits()
        fit_predict: callable(split: InnerSplit, xi: float) -> float
            Fits on the inner training rows using decay weights implied
            by `xi` and the split's own reference time, then returns the
            log loss on the inner validation rows. The model
            implementation stays with the caller so this module has no
            sklearn dependency.
        half_life_grid: candidate half-lives; None means no decay.

    Returns:
        XiSelection with the winning xi and the full search curve.
    """
    if not inner_splits:
        raise TimeDecayError("No inner splits available for xi selection")

    curve: list[dict] = []
    best_xi: float | None = None
    best_hl: float | None = None
    best_ll = float("inf")

    for hl in half_life_grid:
        xi = xi_from_half_life(hl)
        per_split = [float(fit_predict(sp, xi)) for sp in inner_splits]
        mean_ll = float(np.mean(per_split))
        curve.append({
            "half_life_days": hl,
            "xi": xi,
            "mean_inner_log_loss": round(mean_ll, 8),
            "per_split_log_loss": [round(v, 8) for v in per_split],
        })
        if mean_ll < best_ll - 1e-12:
            best_ll, best_xi, best_hl = mean_ll, xi, hl

    return XiSelection(
        xi=best_xi,
        half_life_days=best_hl,
        mean_inner_log_loss=best_ll,
        curve=curve,
        n_inner_splits=len(inner_splits),
        grid_half_lives=list(half_life_grid),
    )


def _log_loss(y: np.ndarray, P: np.ndarray) -> float:
    oh = np.zeros((len(y), 3))
    for i, c in enumerate(("H", "D", "A")):
        oh[:, i] = (y == c)
    Pc = np.clip(P, 1e-15, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(oh * np.log(Pc), axis=1)))


__all__ = [
    "SECONDS_PER_DAY", "HALF_LIFE_GRID_DAYS", "TimeDecayError",
    "xi_from_half_life", "half_life_from_xi", "xi_grid",
    "delta_days", "assert_causal", "decay_weights", "weight_profile",
    "InnerSplit", "build_inner_splits", "XiSelection", "select_xi",
]
