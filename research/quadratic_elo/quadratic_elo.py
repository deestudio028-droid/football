"""E5 — Quadratic Elo -> xG research feature.

IMPLEMENTATION DECISION (Option A, per §4)
------------------------------------------
V3's two goal models are `PoissonRegressor`, i.e. GLMs with a log link:

    log(lambda) = b0 + sum_j b_j * x_j

so the 87 V3 features already enter log(lambda) linearly. Adding one
column `elo_diff_sq` therefore places the quadratic term exactly where
§5 requires:

    log(lambda_H) = alpha_H + ... + c1_H * elo_diff + c2_H * elo_diff_sq
    log(lambda_A) = alpha_A + ... + c1_A * elo_diff + c2_A * elo_diff_sq

Both goal models receive the same column, so the term is fitted
consistently for home and away, each with its own free coefficient.
Option B (a bespoke Elo->goal-rate parameterisation) was rejected: it
would replace V3's architecture rather than extend it, making Arm A and
Arm B structurally different and the comparison unfair.

    V3 contract:  87 columns
    E5 contract:  88 columns = V3[0:87] + ("elo_diff_sq",)

The first 87 columns are byte-identical to V3's, in the same order.

CENTERING — AND WHY IT MATTERS
------------------------------
Raw elo_diff^2 correlates 0.668 with elo_diff across this dataset. Under
V3's ridge penalty (alpha=1.0) that collinearity distorts both
coefficients and confounds the curvature test. Centering first:

    elo_diff_sq = (elo_diff - mean_train)^2

drops the correlation to -0.0006, making the quadratic term effectively
orthogonal to the linear one.

`mean_train` is computed from TRAINING FIXTURES ONLY and then frozen and
applied to the test fold. It is a fitted statistic, exactly like the
preprocessor's median and scale, and is subject to the same causality
rule.

ZERO-QUADRATIC CONTROL (§18)
----------------------------
`make_quadratic_feature(..., enabled=False)` returns an all-zero column.
A constant-zero column has zero variance, contributes nothing to the
linear predictor, and its coefficient is driven to zero by the ridge
penalty, so the 88-column model reproduces the 87-column model. The
harness verifies this numerically rather than assuming it.

DEPENDENCIES: numpy / pandas only. No scipy, no sklearn.
DETERMINISM: every function is pure. No RNG anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The single research column E5 adds to the V3 contract.
QUADRATIC_COLUMN: str = "elo_diff_sq"

#: Column that the quadratic term is built from.
SOURCE_COLUMN: str = "elo_diff"


class QuadraticEloError(ValueError):
    """Raised when a quadratic-Elo safety gate fails."""


@dataclass(frozen=True)
class CenteringStat:
    """A centering constant fitted on training rows only."""
    mean: float
    n_train: int
    source_column: str = SOURCE_COLUMN

    def as_dict(self) -> dict:
        return {"mean": self.mean, "n_train": self.n_train,
                "source_column": self.source_column}


def fit_centering(elo_diff_train: np.ndarray) -> CenteringStat:
    """Fit the centering constant on TRAINING elo_diff values only.

    This function's signature accepts training values only — no label,
    no test array — so a test value cannot reach it.
    """
    v = np.asarray(elo_diff_train, dtype=float)
    if len(v) == 0:
        raise QuadraticEloError("Cannot fit centering on an empty training set")
    if not np.all(np.isfinite(v)):
        raise QuadraticEloError("Non-finite elo_diff in training set")
    return CenteringStat(mean=float(v.mean()), n_train=int(len(v)))


def make_quadratic_feature(
    elo_diff: np.ndarray,
    centering: CenteringStat,
    enabled: bool = True,
) -> np.ndarray:
    """Build the elo_diff_sq column.

    enabled=True  -> (elo_diff - centering.mean) ** 2
    enabled=False -> all zeros  (the §18 zero-quadratic control)
    """
    v = np.asarray(elo_diff, dtype=float)
    if not np.all(np.isfinite(v)):
        raise QuadraticEloError("Non-finite elo_diff supplied")
    if not enabled:
        return np.zeros(len(v), dtype=float)
    q = (v - centering.mean) ** 2
    if not np.all(np.isfinite(q)):
        raise QuadraticEloError("Non-finite elo_diff_sq produced")
    if np.any(q < 0):
        raise QuadraticEloError("Negative squared term — impossible")
    return q


def e5_feature_columns(v3_columns: tuple[str, ...]) -> tuple[str, ...]:
    """The 88-column E5 contract: V3's 87 columns then elo_diff_sq."""
    if QUADRATIC_COLUMN in v3_columns:
        raise QuadraticEloError(
            f"{QUADRATIC_COLUMN} already present in the V3 contract")
    if SOURCE_COLUMN not in v3_columns:
        raise QuadraticEloError(
            f"{SOURCE_COLUMN} missing from the V3 contract")
    return tuple(v3_columns) + (QUADRATIC_COLUMN,)


def build_design(
    X_v3: pd.DataFrame,
    v3_columns: tuple[str, ...],
    centering: CenteringStat,
    enabled: bool = True,
) -> pd.DataFrame:
    """Return the 88-column E5 design matrix.

    The first 87 columns are passed through untouched and in order; only
    elo_diff_sq is appended.
    """
    if list(X_v3.columns) != list(v3_columns):
        raise QuadraticEloError(
            "Input columns do not match the V3 contract order")
    out = X_v3.copy()
    out[QUADRATIC_COLUMN] = make_quadratic_feature(
        X_v3[SOURCE_COLUMN].to_numpy(dtype=float), centering, enabled)
    expected = e5_feature_columns(v3_columns)
    if tuple(out.columns) != expected:
        raise QuadraticEloError("E5 design column order is wrong")
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def curvature_table(
    elo_diff: np.ndarray,
    c0: float, c1: float, c2: float,
    centering: CenteringStat,
    grid: np.ndarray | None = None,
) -> list[dict]:
    """Tabulate linear vs quadratic predicted goal rate over an Elo grid.

    Uses training-fitted coefficients only. Reports both curves so the
    quadratic model can be checked for meaningful bending versus merely
    reproducing the linear fit.
    """
    v = np.asarray(elo_diff, dtype=float)
    if grid is None:
        grid = np.linspace(np.percentile(v, 1), np.percentile(v, 99), 13)
    z = grid - centering.mean
    lin = np.exp(c0 + c1 * grid)
    quad = np.exp(c0 + c1 * grid + c2 * z ** 2)
    return [{"elo_diff": round(float(g), 2),
             "linear_lambda": round(float(l), 6),
             "quadratic_lambda": round(float(q), 6),
             "ratio": round(float(q / l), 6)}
            for g, l, q in zip(grid, lin, quad)]


def check_extreme_safety(lam_h: np.ndarray, lam_a: np.ndarray,
                         max_lambda: float = 15.0) -> dict:
    """§16 — verify goal rates are positive, finite and not exploding."""
    lh = np.asarray(lam_h, dtype=float)
    la = np.asarray(lam_a, dtype=float)
    report = {
        "lambda_home_min": round(float(lh.min()), 6),
        "lambda_home_max": round(float(lh.max()), 6),
        "lambda_away_min": round(float(la.min()), 6),
        "lambda_away_max": round(float(la.max()), 6),
        "all_finite": bool(np.all(np.isfinite(lh)) and np.all(np.isfinite(la))),
        "all_positive": bool(np.all(lh > 0) and np.all(la > 0)),
        "within_max": bool(lh.max() <= max_lambda and la.max() <= max_lambda),
    }
    report["passed"] = (report["all_finite"] and report["all_positive"]
                        and report["within_max"])
    return report


__all__ = [
    "QUADRATIC_COLUMN", "SOURCE_COLUMN", "QuadraticEloError",
    "CenteringStat", "fit_centering", "make_quadratic_feature",
    "e5_feature_columns", "build_design",
    "curvature_table", "check_extreme_safety",
]
