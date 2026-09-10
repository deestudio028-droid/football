"""Constants for the V1 feature pipeline. Every threshold here is
referenced from docs/PHASE2_FEATURE_SPEC.md -- change the doc and this
file together, and bump FEATURE_VERSION if the change alters any
feature's actual values.
"""
from __future__ import annotations

FEATURE_VERSION = "v1.0"

# Successor (v1.1) feature version -- D-15. Deliberately a SEPARATE
# constant rather than a mutation of FEATURE_VERSION: V1 generation must
# keep stamping "v1.0" forever, so the two versions cannot be expressed
# by one global. Only features.feature_builder.build_successor_feature_dataset
# stamps this value; the V1 path never reads it. See docs D-15/D-17.
SUCCESSOR_FEATURE_VERSION = "v1.1"

# Rolling windows and the minimum sample size (coverage_n) required for
# a window to emit a value instead of NULL. See spec §1.
WINDOWS = ["last5", "last10", "season"]
WINDOW_SIZE = {"last5": 5, "last10": 10}  # "season" has no fixed size
WINDOW_MIN_COVERAGE = {"last5": 3, "last10": 5, "season": 2}

# Form windows only (spec §E) -- a subset of WINDOWS.
FORM_WINDOWS = ["last5", "last10"]

# Venue-restricted windows only support "season" (spec §A/§B).
VENUE_MIN_COVERAGE = 2

# Empirical-Bayes shrinkage constant for attack/defence strength (spec §4).
SHRINKAGE_K = 5

# "Insufficient history" threshold used as the promotion/new-team proxy
# (spec §5) -- NOT a literal promotion detector.
MIN_HISTORY_THRESHOLD = 3

# Statuses that count as a "played" match for history purposes (spec §0).
PLAYED_STATUSES = ("FT", "AWARDED")
