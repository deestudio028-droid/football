"""Gate 10 monitoring — additive, local, dependency-light.

WHAT THIS IS
    The §12 monitoring implementation for the V2 candidate: six required
    signals, each with thresholds locked by explicit human authorization
    (see docs/GATE10_MONITORING_SPECIFICATION_DECISION_MATRIX.md §13 and
    docs/PHASE5_V2_MODEL_SPEC_DRAFT.md §12).

WHAT THIS IS NOT
    - Not a model, trainer, tuner, or calibrator. Nothing here fits or
      changes any estimator.
    - Not a V2 artifact. `MODEL_VERSION` is read only, never written, and
      remains "v1.0".
    - Not a dashboard, web API, scheduler, alert integration, or UI.
    - Not a modification of anything. Every frozen V1 file is untouched:
      config.py, train.py, splits.py, evaluate.py, data.py, baselines.py,
      run_experiments.py. `candidate_contract.py` and `calibration.py` are
      imported read-only and never modified.
    - Not a 2025/26 consumer. This package imports no split function and
      no season constant, and its baselines are restricted to non-test
      historical partitions by construction.

DESIGN NOTES
    - Signal 5 authority stays with `evaluate.validate_probabilities`.
      The wrapper records a failure and then re-raises the ORIGINAL
      exception object unchanged. It never swallows, normalizes, clips,
      repairs, or downgrades an invalid probability matrix.
    - Signal 1 reports the absolute availability ceiling and the drift
      alert as two separate events. They are different constructs and
      are never merged.
    - Signal 3's baseline is a STORED historical reference only; the
      rolling live realized-frequency baseline (former option C-4) is
      excluded, so Signal 3 is prediction-time only.
    - Signals 4 and 6 are post-outcome only and emit
      INSUFFICIENT_SAMPLE below the locked minimum sample size.
"""
from .config import (
    CALIBRATION_TOLERANCE_PP,
    CLASS_DISTRIBUTION_TOLERANCE_PP,
    DRIFT_ALERT_PP,
    FAMILY_COLUMNS,
    MIN_SETTLED_SAMPLE,
    NON_TEST_PARTITION_NULL_RATES,
    OBSERVATION_WINDOW_DAYS,
    SETTLEMENT_MATURITY_DAYS,
    SIGNAL_NAMES,
    Severity,
    Status,
)
from .events import MonitoringEvent, SchemaViolation, validate_event
from .signals import (
    build_null_rate_baseline,
    evaluate_class_distribution,
    evaluate_calibration_post_outcome,
    evaluate_draw_behaviour,
    evaluate_null_rate_ceiling,
    evaluate_null_rate_drift,
    evaluate_unseen_competition_ids,
    load_stored_base_rate_reference,
    observed_family_null_rates,
)
from .store import JsonlEventStore
from .validity import monitored_predict_candidate, record_validity_failure

__all__ = [
    "CALIBRATION_TOLERANCE_PP",
    "CLASS_DISTRIBUTION_TOLERANCE_PP",
    "DRIFT_ALERT_PP",
    "FAMILY_COLUMNS",
    "MIN_SETTLED_SAMPLE",
    "NON_TEST_PARTITION_NULL_RATES",
    "OBSERVATION_WINDOW_DAYS",
    "SETTLEMENT_MATURITY_DAYS",
    "SIGNAL_NAMES",
    "Severity",
    "Status",
    "MonitoringEvent",
    "SchemaViolation",
    "validate_event",
    "JsonlEventStore",
    "build_null_rate_baseline",
    "observed_family_null_rates",
    "load_stored_base_rate_reference",
    "evaluate_null_rate_ceiling",
    "evaluate_null_rate_drift",
    "evaluate_unseen_competition_ids",
    "evaluate_class_distribution",
    "evaluate_calibration_post_outcome",
    "evaluate_draw_behaviour",
    "monitored_predict_candidate",
    "record_validity_failure",
]
