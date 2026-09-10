"""v4.0 production candidate feature contract: V3 + Online Attack/Defense.

WHY THIS IS A SEPARATE MODULE:
Parallel to v3_contract.py and v2_artifact.py. It leaves every existing
contract untouched and defines the formal 91-column V4 contract:

    87 V3 columns (V3_FEATURE_COLUMNS, order preserved exactly)
  +  4 online attack/defense columns (AD_COLUMNS)
  ================================================================
    91 total V4 candidate features

PROVENANCE — only independently validated components
    E1  causal Elo              PASS  (already inside the 87 via v3_contract)
    E6  online attack/defense   PASS  (the 4 appended columns)
    E2  market odds             PASS  — deliberately ABSENT, see below

WHY MARKET ODDS ARE NOT IN THIS CONTRACT:
E2's validated architecture is a post-hoc probability signal, never a
regression feature. Adding market columns here would silently reinterpret
E2 and would additionally recreate the E7 Arm D construction, which
FAILED. The market is carried alongside V4 as a separate output, never
inside this design matrix.

EXPLICITLY EXCLUDED (all FAILED or INCONCLUSIVE, none promoted):
    E3 Dixon-Coles · E4 time decay · E5 quadratic Elo
    E7 blend · E8 temperature scaling · E9 LightGBM

WHAT THIS MODULE DOES NOT DO:
  - Fit, train, or tune anything.
  - Write, save, or modify any artifact.
  - Replace or modify v3_contract.py, v2_artifact.py or ablation.py.
  - Promote V4 to production.
"""
from __future__ import annotations

from features.online_attack_defense import AD_COLUMNS
from .v3_contract import V3_FEATURE_COLUMNS

#: The 4 online attack/defense columns appended by E6, in fixed order.
V4_AD_COLUMNS: tuple[str, ...] = tuple(AD_COLUMNS)

#: The 91-column V4 candidate contract. V3's 87 come first, in order.
V4_FEATURE_COLUMNS: tuple[str, ...] = tuple(V3_FEATURE_COLUMNS) + V4_AD_COLUMNS

#: Sizes.
V4_N_FEATURES: int = len(V4_FEATURE_COLUMNS)
V3_N_FEATURES_IN_V4: int = len(V3_FEATURE_COLUMNS)

#: Market columns must never appear in the design matrix. Listed so the
#: prohibition is checkable rather than merely documented.
FORBIDDEN_MARKET_COLUMNS: frozenset[str] = frozenset({
    "p_home", "p_draw", "p_away",
    "pm_h", "pm_d", "pm_a",
    "devig_closing_home", "devig_closing_draw", "devig_closing_away",
    "devig_opening_home", "devig_opening_draw", "devig_opening_away",
    "closing_home", "closing_draw", "closing_away",
    "opening_home", "opening_draw", "opening_away",
    "market_prob_home", "market_prob_draw", "market_prob_away",
})

#: Components from failed/inconclusive experiments, likewise checkable.
FORBIDDEN_EXPERIMENT_COLUMNS: frozenset[str] = frozenset({
    "elo_diff_sq",          # E5 quadratic Elo — INCONCLUSIVE
    "rho", "tau",           # E3 Dixon-Coles — INCONCLUSIVE
    "decay_weight", "xi",   # E4 time decay — INCONCLUSIVE
    "temperature", "T",     # E8 temperature scaling — FAIL
    "blend_weight", "w",    # E7 blend — FAIL
})


def validate_contract() -> None:
    """Assert every structural invariant of the V4 contract.

    Raises ValueError on any violation. Cheap enough to call at import
    time in tests and at the start of any V4 build.
    """
    if len(V3_FEATURE_COLUMNS) != 87:
        raise ValueError(
            f"V3 contract must be 87 columns, got {len(V3_FEATURE_COLUMNS)}")
    if len(V4_AD_COLUMNS) != 4:
        raise ValueError(
            f"Expected 4 online A/D columns, got {len(V4_AD_COLUMNS)}")
    if V4_N_FEATURES != 91:
        raise ValueError(f"V4 contract must be 91 columns, got {V4_N_FEATURES}")

    # V3's 87 must be preserved exactly, in order, at the front
    if tuple(V4_FEATURE_COLUMNS[:87]) != tuple(V3_FEATURE_COLUMNS):
        raise ValueError("V4[:87] does not equal the V3 contract in order")

    # the 4 appended columns must be exactly the E6 ones, in order
    if tuple(V4_FEATURE_COLUMNS[87:]) != V4_AD_COLUMNS:
        raise ValueError("V4[87:] does not equal the E6 A/D columns in order")

    if len(set(V4_FEATURE_COLUMNS)) != 91:
        raise ValueError("V4 contract contains duplicate column names")

    forbidden = set(V4_FEATURE_COLUMNS) & FORBIDDEN_MARKET_COLUMNS
    if forbidden:
        raise ValueError(f"Market columns present in V4 contract: {sorted(forbidden)}")

    forbidden = set(V4_FEATURE_COLUMNS) & FORBIDDEN_EXPERIMENT_COLUMNS
    if forbidden:
        raise ValueError(
            f"Non-promoted experiment columns in V4 contract: {sorted(forbidden)}")


__all__ = [
    "V4_AD_COLUMNS",
    "V4_FEATURE_COLUMNS",
    "V4_N_FEATURES",
    "V3_N_FEATURES_IN_V4",
    "FORBIDDEN_MARKET_COLUMNS",
    "FORBIDDEN_EXPERIMENT_COLUMNS",
    "validate_contract",
]
