"""Phase 3G — Power-method de-vigging for 1X2 decimal odds.

Solves for k in:
    (1/O_H)^k + (1/O_D)^k + (1/O_A)^k = 1

using scipy.optimize.brentq (bisection-family root-finding).

Returns de-vigged probabilities that sum exactly to 1.

This module is isolated research code. It does NOT modify any production
artifact, database, or model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DevigResult:
    """Result of power-method de-vigging."""
    p_home: float
    p_draw: float
    p_away: float
    k: float
    overround: float  # raw implied probability sum before de-vig
    home_odds: float
    draw_odds: float
    away_odds: float

    def probabilities(self) -> Tuple[float, float, float]:
        return (self.p_home, self.p_draw, self.p_away)

    def prob_sum(self) -> float:
        return self.p_home + self.p_draw + self.p_away


class DevigError(ValueError):
    """Raised when de-vigging fails due to invalid inputs."""


def _validate_odds(home: float, draw: float, away: float) -> None:
    """Validate that odds are valid decimal odds (> 1.0, finite)."""
    for name, val in [("home", home), ("draw", draw), ("away", away)]:
        if val is None:
            raise DevigError(f"{name} odds is None")
        if not isinstance(val, (int, float)):
            raise DevigError(f"{name} odds is not numeric: {val!r}")
        if math.isnan(val) or math.isinf(val):
            raise DevigError(f"{name} odds is not finite: {val}")
        if val <= 1.0:
            raise DevigError(
                f"{name} odds must be > 1.0 for decimal odds, got {val}"
            )


def _implied_sum(home: float, draw: float, away: float, k: float) -> float:
    """Compute sum of implied probabilities raised to power k."""
    return (1.0 / home) ** k + (1.0 / draw) ** k + (1.0 / away) ** k


def power_devig(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> DevigResult:
    """De-vig 1X2 decimal odds using the power method.

    Finds k such that (1/O_H)^k + (1/O_D)^k + (1/O_A)^k = 1,
    then returns P_i = (1/O_i)^k for each outcome.

    Args:
        home_odds: Decimal odds for home win (must be > 1.0)
        draw_odds: Decimal odds for draw (must be > 1.0)
        away_odds: Decimal odds for away win (must be > 1.0)
        tol: Convergence tolerance for bisection
        max_iter: Maximum bisection iterations

    Returns:
        DevigResult with de-vigged probabilities, k, and overround

    Raises:
        DevigError: If odds are invalid or solver fails
    """
    _validate_odds(home_odds, draw_odds, away_odds)

    # Raw overround
    overround = 1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds

    # Special case: already fair (overround ≈ 1.0)
    if abs(overround - 1.0) < tol:
        return DevigResult(
            p_home=1.0 / home_odds,
            p_draw=1.0 / draw_odds,
            p_away=1.0 / away_odds,
            k=1.0,
            overround=overround,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
        )

    # For vigged odds (overround > 1), k > 1 shrinks probabilities
    # For under-round (overround < 1), k < 1 expands them
    # Bisection bounds
    if overround > 1.0:
        k_lo, k_hi = 1.0, 100.0
    else:
        k_lo, k_hi = 0.01, 1.0

    # Verify bracket: f(k_lo) and f(k_hi) must straddle 0
    f_lo = _implied_sum(home_odds, draw_odds, away_odds, k_lo) - 1.0
    f_hi = _implied_sum(home_odds, draw_odds, away_odds, k_hi) - 1.0

    # Expand bracket if needed
    expansion_attempts = 0
    while f_lo * f_hi > 0 and expansion_attempts < 20:
        if overround > 1.0:
            k_hi *= 2
        else:
            k_lo /= 2
        f_lo = _implied_sum(home_odds, draw_odds, away_odds, k_lo) - 1.0
        f_hi = _implied_sum(home_odds, draw_odds, away_odds, k_hi) - 1.0
        expansion_attempts += 1

    if f_lo * f_hi > 0:
        raise DevigError(
            f"Cannot bracket root for odds ({home_odds}, {draw_odds}, {away_odds}). "
            f"f(k_lo={k_lo})={f_lo:.6e}, f(k_hi={k_hi})={f_hi:.6e}"
        )

    # Bisection
    for _ in range(max_iter):
        k_mid = (k_lo + k_hi) / 2.0
        f_mid = _implied_sum(home_odds, draw_odds, away_odds, k_mid) - 1.0

        if abs(f_mid) < tol:
            k = k_mid
            break

        if f_lo * f_mid < 0:
            k_hi = k_mid
            f_hi = f_mid
        else:
            k_lo = k_mid
            f_lo = f_mid
    else:
        # Use best available
        k = (k_lo + k_hi) / 2.0

    # Compute de-vigged probabilities
    p_home = (1.0 / home_odds) ** k
    p_draw = (1.0 / draw_odds) ** k
    p_away = (1.0 / away_odds) ** k

    # Normalize to handle any residual numerical error
    total = p_home + p_draw + p_away
    if total <= 0 or not math.isfinite(total):
        raise DevigError(
            f"De-vig produced invalid probability sum {total} "
            f"for odds ({home_odds}, {draw_odds}, {away_odds}), k={k}"
        )

    p_home /= total
    p_draw /= total
    p_away /= total

    return DevigResult(
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        k=k,
        overround=overround,
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
    )


def devig_batch(
    rows: list[tuple[float, float, float]],
) -> list[DevigResult | None]:
    """De-vig a list of (home, draw, away) odds tuples.

    Returns a list of DevigResult (or None for rows that fail validation).
    """
    results = []
    for home, draw, away in rows:
        try:
            results.append(power_devig(home, draw, away))
        except DevigError:
            results.append(None)
    return results
