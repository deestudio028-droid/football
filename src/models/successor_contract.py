"""v1.1 successor feature contract (D-14 / D-15 / D-19 / D-20).

WHY THIS IS A SEPARATE MODULE, NOT AN ADDITION TO ablation.py:
`src/models/ablation.py` is checksum-pinned at
9bf8bf4a1f332f8bb47d640a72384ed7 by nine governance test modules
(Gate 1, 3, 4, 5, 6, 7, 8, Phase 5A, Phase 5B) -- a pinning layer
independent of run_final_comparison.LOCKED_INPUTS. D-18 discovered this
by breaking all nine. Defining the successor constants here leaves that
checksum untouched, so every Phase 3/4A/4B/4C/5A artifact stays
reproducible and every pin stays green.

CONTRACT SOURCE OF TRUTH: `ablation.MODEL_B_COLUMNS`, imported and
CONCATENATED onto -- never restated. Duplicating the 80 names would
create two sources of truth that could silently diverge; concatenation
makes "the first 80 successor columns are exactly MODEL_B_COLUMNS, in
order" true by construction rather than by assertion.

MODULE NAMING: deliberately not `*v2*`. Nine governance tests iterate
`Path("src/models").glob("*.py")` and assert no module stem contains
"v2", encoding the rule "no V2 model or artifact has been created". A
feature-contract definition is not a V2 model, and the correct response
to that invariant is to respect it rather than weaken the test. This
follows the precedent set by `candidate_contract.py`.

WHAT THIS IS NOT:
  - Not a model, trainer, or estimator. No training logic.
  - Not a feature generator. Construction of the six columns lives in
    `features.feature_builder.build_successor_feature_dataset`.
  - Not a promotion. The six columns are absent from the pinned
    data/processed/features.db; MODEL_VERSION stays "v1.0"; nothing
    here is validated, fitted, or deployed.
  - No side effects: this module only defines two tuples.

STATUS: the successor contract is defined but NOT promoted. Whether
POSSESSION improves the model is a separate, later question.
"""
from __future__ import annotations

from .ablation import MODEL_B_COLUMNS

#: The three additive season-only families from D-5, two sides each.
#:
#: Order is D-5's FAMILIES order -- possession, corners, red_cards, and
#: home before away within each family -- carried forward unchanged so
#: the contract order matches the construction that passed D-5's 7/7
#: leakage audit and D-6's validation.
POSSESSION_FAMILY_COLUMNS: tuple[str, ...] = (
    "home_possession_per_match_season",
    "away_possession_per_match_season",
    "home_corners_per_match_season",
    "away_corners_per_match_season",
    "home_red_cards_per_match_season",
    "away_red_cards_per_match_season",
)

#: The 80-column frozen V1 contract plus the six POSSESSION columns.
#: Built by concatenation onto the untouched MODEL_B_COLUMNS.
MODEL_B_PLUS_POSSESSION_COLUMNS: tuple[str, ...] = MODEL_B_COLUMNS + POSSESSION_FAMILY_COLUMNS

#: Sizes, stated once so callers never hardcode them.
SUCCESSOR_N_FEATURES: int = len(MODEL_B_PLUS_POSSESSION_COLUMNS)
V1_N_FEATURES: int = len(MODEL_B_COLUMNS)

__all__ = [
    "POSSESSION_FAMILY_COLUMNS",
    "MODEL_B_PLUS_POSSESSION_COLUMNS",
    "SUCCESSOR_N_FEATURES",
    "V1_N_FEATURES",
]
