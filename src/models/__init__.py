"""V1 match-result model package (H/D/A classification).

Phase 3, Step 1 scope only: data loading and temporal splitting. No
model training, no calibration, no hyperparameter tuning, and no
model artifacts exist in this package yet -- see
docs/PHASE3_MODEL_DEVELOPMENT_DESIGN_REPORT.md for the full design and
the reason each of those is deliberately deferred.

Reads only from `data/processed/features.db` (read-only). Never
touches `data/processed/matches.db` or `data/raw/`.
"""

from .config import MODEL_VERSION

__all__ = ["MODEL_VERSION"]
