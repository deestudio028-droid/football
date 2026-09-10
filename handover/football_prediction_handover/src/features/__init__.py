"""Leakage-safe feature engineering for the V1 football prediction system.

Reads only from `data/processed/matches.db` (the immutable ingested
dataset). Writes only to `data/processed/features.db`. No network
access, no API calls, no model training happens anywhere in this
package -- see docs/PHASE2_FEATURE_SPEC.md for the full feature
contract this package implements.
"""

from .config import FEATURE_VERSION

__all__ = ["FEATURE_VERSION"]
