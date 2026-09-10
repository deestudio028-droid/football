"""v3.0 candidate feature contract: Poisson + Venue + Persistent Elo.

WHY THIS IS A SEPARATE MODULE:
Parallel to v2_artifact.py and candidate_contract.py. Leaves all existing
contracts and models untouched while defining the formal 87-column contract:

    80 base columns (MODEL_B_COLUMNS)
  +  4 venue columns (VENUE_COLUMNS)
  +  3 persistent Elo columns (ELO_FEATURE_COLUMNS)
  ================================================
    87 total candidate features

WHAT THIS MODULE DOES NOT DO:
  - Fit, train, or tune anything.
  - Write, save, or modify any artifact.
  - Replace or modify v2_artifact.py or ablation.py.
  - Promote V3 to production.
"""
from __future__ import annotations

from .ablation import MODEL_B_COLUMNS
from .v2_artifact import VENUE_COLUMNS

#: The 3 persistent Elo columns.
ELO_FEATURE_COLUMNS: tuple[str, ...] = (
    "home_elo",
    "away_elo",
    "elo_diff",
)

#: The 87-column candidate feature contract.
V3_FEATURE_COLUMNS: tuple[str, ...] = MODEL_B_COLUMNS + VENUE_COLUMNS + ELO_FEATURE_COLUMNS

#: Sizes.
V3_N_FEATURES: int = len(V3_FEATURE_COLUMNS)
V2_N_FEATURES: int = len(MODEL_B_COLUMNS) + len(VENUE_COLUMNS)
V1_N_FEATURES: int = len(MODEL_B_COLUMNS)

__all__ = [
    "ELO_FEATURE_COLUMNS",
    "V3_FEATURE_COLUMNS",
    "V3_N_FEATURES",
    "V2_N_FEATURES",
    "V1_N_FEATURES",
]