"""Locked monitoring policy constants (Gate 10).

Every numeric value here was fixed by explicit human authorization. None
was inferred, tuned, or derived from data by this module. Provenance is
recorded inline so a reader can trace each number to its decision.

Frozen-source discipline: the six families and the known competition set
are IMPORTED from their existing frozen definitions rather than restated,
so they cannot silently diverge. The §10 availability ceilings are
likewise imported, never re-derived.
"""
from __future__ import annotations

from enum import Enum

from models.ablation import (
    COMPETITION_ID_COLUMN,
    FORM_COLUMNS,
    GOALS_CORE_COLUMNS,
    MODEL_B_COLUMNS,
    SHOTS_CORE_COLUMNS,
    SHOTS_ON_CORE_COLUMNS,
    STRENGTH_COLUMNS,
)
from models.candidate_contract import (
    KNOWN_COMPETITION_IDS,
    MAX_FINAL_SEASON_NULL_RATE,
    MAX_HISTORICAL_NULL_RATE,
)

SCHEMA_VERSION = "gate10-monitoring-1"


class Status(str, Enum):
    """The five authorized statuses. No additional semantics."""

    PASS = "PASS"
    ALERT = "ALERT"
    HARD_FAIL = "HARD_FAIL"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    DEGRADED = "DEGRADED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


#: Severity is a MECHANICAL function of status, not an independent policy.
#: Severity policy was classified deferrable, so this mapping is derived
#: rather than invented: it adds no information beyond the status.
SEVERITY_BY_STATUS: dict[Status, Severity] = {
    Status.PASS: Severity.INFO,
    Status.INSUFFICIENT_SAMPLE: Severity.INFO,
    Status.DEGRADED: Severity.WARNING,
    Status.ALERT: Severity.WARNING,
    Status.HARD_FAIL: Severity.CRITICAL,
}


def severity_for(status: Status) -> Severity:
    return SEVERITY_BY_STATUS[Status(status)]


# ---------------------------------------------------------------------
# Signal identifiers.
#
# Signal 1 is emitted as TWO ids because the absolute availability
# ceiling and the drift alert are different constructs that must never be
# merged (spec §12.1).
# ---------------------------------------------------------------------
SIGNAL_S1_CEILING = "S1_CEILING"
SIGNAL_S1_DRIFT = "S1_DRIFT"
SIGNAL_S2_UNSEEN_COMPETITION = "S2_UNSEEN_COMPETITION"
SIGNAL_S3_CLASS_DISTRIBUTION = "S3_CLASS_DISTRIBUTION"
SIGNAL_S4_CALIBRATION = "S4_CALIBRATION"
SIGNAL_S5_PROBABILITY_VALIDITY = "S5_PROBABILITY_VALIDITY"
SIGNAL_S6_DRAW_BEHAVIOUR = "S6_DRAW_BEHAVIOUR"

SIGNAL_NAMES: dict[str, str] = {
    SIGNAL_S1_CEILING: "per-family absolute availability ceiling",
    SIGNAL_S1_DRIFT: "per-family null-rate drift",
    SIGNAL_S2_UNSEEN_COMPETITION: "unseen competition_id occurrences",
    SIGNAL_S3_CLASS_DISTRIBUTION: "predicted class distribution vs historical base rates",
    SIGNAL_S4_CALIBRATION: "mean predicted probability per class vs realized outcome frequency",
    SIGNAL_S5_PROBABILITY_VALIDITY: "probability-validity failures",
    SIGNAL_S6_DRAW_BEHAVIOUR: "draw-class behaviour",
}

#: The six §12 requirements, mapped to the signal ids that satisfy them.
#: Used by the acceptance tests to prove full coverage.
REQUIRED_SIGNAL_COVERAGE: dict[str, tuple[str, ...]] = {
    "per-family null-rate drift": (SIGNAL_S1_DRIFT, SIGNAL_S1_CEILING),
    "unseen competition_id occurrences": (SIGNAL_S2_UNSEEN_COMPETITION,),
    "predicted class distribution vs historical base rates": (SIGNAL_S3_CLASS_DISTRIBUTION,),
    "mean predicted probability per class vs realized outcome frequency": (SIGNAL_S4_CALIBRATION,),
    "probability-validity failures": (SIGNAL_S5_PROBABILITY_VALIDITY,),
    "draw-class behaviour": (SIGNAL_S6_DRAW_BEHAVIOUR,),
}


# ---------------------------------------------------------------------
# LOCKED THRESHOLDS. Authorized values; not tuned, not inferred.
# ---------------------------------------------------------------------

#: Signal 1 drift alert: absolute percentage-point deviation ABOVE the
#: selected historical baseline. Authorized value: +2 percentage points.
DRIFT_ALERT_PP: float = 2.0

#: Signal 3: +/- percentage points per class, H/D/A evaluated
#: independently. Authorized value: 5 percentage points.
CLASS_DISTRIBUTION_TOLERANCE_PP: float = 5.0

#: Signals 4 and 6: +/- percentage points per class. Authorized value: 5.
CALIBRATION_TOLERANCE_PP: float = 5.0

#: Signals 4 and 6: minimum eligible settled predictions before a
#: PASS/ALERT verdict may be emitted. Authorized value: 200.
MIN_SETTLED_SAMPLE: int = 200

#: Signals 4 and 6: rolling observation window, in days. Authorized: 30.
OBSERVATION_WINDOW_DAYS: int = 30

#: Signals 4 and 6: settlement maturity, in days. Authorized: 7.
#:
#: IMPLEMENTED EXACTLY AS SPECIFIED: an observation is eligible only when
#: `settled_at - predicted_at >= 7 days`. This is the literal authorized
#: rule and was deliberately NOT reinterpreted. An operational note is
#: recorded in the decision matrix (§13.2) rather than resolved here.
SETTLEMENT_MATURITY_DAYS: int = 7

#: The §10 absolute availability ceilings, IMPORTED unchanged. These are
#: NOT drift thresholds and are never compared against DRIFT_ALERT_PP.
CEILING_HISTORICAL: float = MAX_HISTORICAL_NULL_RATE      # 0.078
CEILING_FINAL_SEASON: float = MAX_FINAL_SEASON_NULL_RATE  # 0.056


# ---------------------------------------------------------------------
# Family mapping, DERIVED from the frozen 80-column contract.
#
# Each family is an existing frozen tuple in models.ablation. Their
# concatenation is MODEL_B_COLUMNS in contract order -- asserted at import
# time, so a contract change breaks loudly instead of silently skewing
# every drift measurement. No unrelated feature group is included.
# ---------------------------------------------------------------------
FAMILY_COLUMNS: dict[str, tuple[str, ...]] = {
    "goals_core": tuple(GOALS_CORE_COLUMNS),
    "form": tuple(FORM_COLUMNS),
    "strength": tuple(STRENGTH_COLUMNS),
    "competition_id": tuple(COMPETITION_ID_COLUMN),
    "shots_core": tuple(SHOTS_CORE_COLUMNS),
    "shots_on_core": tuple(SHOTS_ON_CORE_COLUMNS),
}

FAMILY_NAMES: tuple[str, ...] = tuple(FAMILY_COLUMNS)


def _verify_family_partition() -> None:
    """Fail loudly at import if the families stop partitioning the contract."""
    flat: list[str] = []
    for cols in FAMILY_COLUMNS.values():
        flat.extend(cols)
    if len(flat) != len(set(flat)):
        raise RuntimeError("monitoring family mapping is not disjoint")
    if set(flat) != set(MODEL_B_COLUMNS):
        raise RuntimeError("monitoring family mapping does not cover exactly the 80-column contract")
    if tuple(flat) != tuple(MODEL_B_COLUMNS):
        raise RuntimeError("monitoring family mapping order differs from the frozen contract order")


_verify_family_partition()


# ---------------------------------------------------------------------
# Signal 1 baseline evidence: NON-TEST historical partitions only.
#
# Transcribed from the recorded per-family coverage table in
# docs/PHASE5_V2_CANDIDATE_REVIEW.md Part 4 ("Max null rate per family,
# per partition"). 2025/26 (the final test partition) is DELIBERATELY
# ABSENT and must never be added: it is excluded as a monitoring baseline
# by standing governance.
#
# No new computation produced these numbers -- they are a transcription of
# already-recorded evidence.
# ---------------------------------------------------------------------
NON_TEST_PARTITION_NULL_RATES: dict[str, dict[str, float]] = {
    "fold_1_train": {
        "goals_core": 0.0764, "form": 0.0764, "strength": 0.0027,
        "competition_id": 0.0000, "shots_core": 0.0772, "shots_on_core": 0.0772,
    },
    "fold_2_train": {
        "goals_core": 0.0555, "form": 0.0555, "strength": 0.0027,
        "competition_id": 0.0000, "shots_core": 0.0560, "shots_on_core": 0.0560,
    },
    "fold_3_train": {
        "goals_core": 0.0542, "form": 0.0444, "strength": 0.0028,
        "competition_id": 0.0000, "shots_core": 0.0550, "shots_on_core": 0.0550,
    },
    "final_train": {
        "goals_core": 0.0543, "form": 0.0374, "strength": 0.0028,
        "competition_id": 0.0000, "shots_core": 0.0551, "shots_on_core": 0.0550,
    },
}

#: Names of the partitions that may be used as a Signal 1 baseline.
ALLOWED_BASELINE_PARTITIONS: tuple[str, ...] = tuple(NON_TEST_PARTITION_NULL_RATES)

#: Aggregations available when combining several non-test partitions.
#: NO DEFAULT IS PROVIDED. Which partition(s) and which aggregation form
#: the operational baseline was NOT selected by the authorized decisions
#: (only the constraint "non-test only, exclude 2025/26" was), so the
#: caller must state it explicitly. See decision matrix §13.1.
BASELINE_AGGREGATIONS: tuple[str, ...] = ("max", "mean", "single")

#: Stored non-test base-rate reference for Signal 3. Read from the frozen
#: Phase 3 artifact rather than hardcoded; folds 1-3 validation only.
BASE_RATE_ARTIFACT = "data/audit/phase3_probability_diagnostics.json"

CLASS_ORDER_NAMES: tuple[str, ...] = ("H", "D", "A")

__all__ = [
    "SCHEMA_VERSION", "Status", "Severity", "SEVERITY_BY_STATUS", "severity_for",
    "SIGNAL_S1_CEILING", "SIGNAL_S1_DRIFT", "SIGNAL_S2_UNSEEN_COMPETITION",
    "SIGNAL_S3_CLASS_DISTRIBUTION", "SIGNAL_S4_CALIBRATION",
    "SIGNAL_S5_PROBABILITY_VALIDITY", "SIGNAL_S6_DRAW_BEHAVIOUR",
    "SIGNAL_NAMES", "REQUIRED_SIGNAL_COVERAGE",
    "DRIFT_ALERT_PP", "CLASS_DISTRIBUTION_TOLERANCE_PP", "CALIBRATION_TOLERANCE_PP",
    "MIN_SETTLED_SAMPLE", "OBSERVATION_WINDOW_DAYS", "SETTLEMENT_MATURITY_DAYS",
    "CEILING_HISTORICAL", "CEILING_FINAL_SEASON",
    "FAMILY_COLUMNS", "FAMILY_NAMES", "KNOWN_COMPETITION_IDS",
    "NON_TEST_PARTITION_NULL_RATES", "ALLOWED_BASELINE_PARTITIONS",
    "BASELINE_AGGREGATIONS", "BASE_RATE_ARTIFACT", "CLASS_ORDER_NAMES",
]
