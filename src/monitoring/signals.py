"""The six §12 monitoring signals.

Every threshold is imported from `monitoring.config`; none is written
here. Every comparison rule (direction, boundary, epsilon) is stated
explicitly in the docstring of the function that applies it, so a reader
never has to guess how a boundary case is decided.

Nothing in this module trains, tunes, calibrates, or fits anything. It
computes statistics over data it is handed and emits events.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    ALLOWED_BASELINE_PARTITIONS,
    BASE_RATE_ARTIFACT,
    BASELINE_AGGREGATIONS,
    CALIBRATION_TOLERANCE_PP,
    CEILING_HISTORICAL,
    CLASS_DISTRIBUTION_TOLERANCE_PP,
    CLASS_ORDER_NAMES,
    DRIFT_ALERT_PP,
    FAMILY_COLUMNS,
    KNOWN_COMPETITION_IDS,
    MIN_SETTLED_SAMPLE,
    NON_TEST_PARTITION_NULL_RATES,
    OBSERVATION_WINDOW_DAYS,
    SETTLEMENT_MATURITY_DAYS,
    SIGNAL_NAMES,
    SIGNAL_S1_CEILING,
    SIGNAL_S1_DRIFT,
    SIGNAL_S2_UNSEEN_COMPETITION,
    SIGNAL_S3_CLASS_DISTRIBUTION,
    SIGNAL_S4_CALIBRATION,
    SIGNAL_S6_DRAW_BEHAVIOUR,
    Status,
)
from .events import MonitoringEvent

#: Float-comparison epsilon for threshold boundaries. Small enough to be
#: physically irrelevant, large enough to absorb binary representation
#: noise so a value computed as 2.0000000000000004 pp is not treated as
#: exceeding a 2.0 pp threshold.
_EPS = 1e-9


class MonitoringInputError(ValueError):
    """Raised when monitoring input is structurally unusable."""


def _exceeds(value: float, threshold: float) -> bool:
    """Strictly-greater comparison with an epsilon guard.

    BOUNDARY RULE (explicit): a value exactly AT the threshold does NOT
    exceed it. A +2.00 pp drift is PASS; +2.01 pp is ALERT.
    """
    return float(value) > float(threshold) + _EPS


# =====================================================================
# SIGNAL 1 -- per-family null rate: ceiling and drift (never merged)
# =====================================================================

def observed_family_null_rates(X: pd.DataFrame) -> dict[str, float]:
    """Per-family observed null rate for a batch.

    AGGREGATION: the MAXIMUM null rate across the family's columns. This
    matches the semantics of the recorded baseline evidence, which is
    documented as "Max null rate per family, per partition" -- comparing a
    family mean against a family max would be an apples-to-oranges
    comparison and would understate drift.
    """
    if len(X) == 0:
        raise MonitoringInputError("cannot compute null rates for an empty batch")
    out: dict[str, float] = {}
    for family, columns in FAMILY_COLUMNS.items():
        missing = [c for c in columns if c not in X.columns]
        if missing:
            raise MonitoringInputError(
                f"family {family!r} is missing {len(missing)} contract column(s): {missing[:5]}"
            )
        out[family] = float(max(X[c].isna().mean() for c in columns))
    return out


def build_null_rate_baseline(
    aggregation: str,
    partitions: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-family Signal 1 baseline, from NON-TEST partitions only.

    `aggregation` is REQUIRED and has no default. The authorized decisions
    fixed the constraint (non-test partitions only, 2025/26 excluded) but
    did NOT select which partition or aggregation is operational, so this
    function refuses to choose one on the caller's behalf. See the Gate 10
    decision matrix §13.1.

    Args:
        aggregation: "max" or "mean" across `partitions`, or "single"
            (requires exactly one partition).
        partitions: names from `ALLOWED_BASELINE_PARTITIONS`. Defaults to
            all of them, which is only meaningful for "max"/"mean".

    Raises:
        MonitoringInputError: unknown aggregation, unknown partition, or
            any attempt to use a partition outside the non-test set --
            which is how a 2025/26 baseline is prevented structurally
            rather than by convention.
    """
    if aggregation not in BASELINE_AGGREGATIONS:
        raise MonitoringInputError(
            f"unknown aggregation {aggregation!r}; expected one of {BASELINE_AGGREGATIONS}"
        )
    names = tuple(partitions) if partitions is not None else ALLOWED_BASELINE_PARTITIONS
    if not names:
        raise MonitoringInputError("at least one baseline partition is required")
    unknown = [n for n in names if n not in NON_TEST_PARTITION_NULL_RATES]
    if unknown:
        raise MonitoringInputError(
            f"partition(s) {unknown} are not in the authorized non-test baseline evidence "
            f"{list(ALLOWED_BASELINE_PARTITIONS)}; the 2025/26 final-test partition is "
            "excluded from monitoring baselines by governance and is not available here"
        )
    if aggregation == "single" and len(names) != 1:
        raise MonitoringInputError('aggregation="single" requires exactly one partition')

    out: dict[str, float] = {}
    for family in FAMILY_COLUMNS:
        values = [NON_TEST_PARTITION_NULL_RATES[n][family] for n in names]
        if aggregation == "max":
            out[family] = float(max(values))
        elif aggregation == "mean":
            out[family] = float(sum(values) / len(values))
        else:  # single
            out[family] = float(values[0])
    return out


def evaluate_null_rate_ceiling(
    X: pd.DataFrame,
    ceiling: float = CEILING_HISTORICAL,
    context: Mapping[str, Any] | None = None,
) -> MonitoringEvent:
    """Signal 1(a): ABSOLUTE availability ceiling -- §10, not drift.

    Emits ALERT if any family's observed null rate exceeds `ceiling`.
    This function never reads a drift baseline and never references
    `DRIFT_ALERT_PP`: the ceiling and the drift alert are separate
    constructs and are reported as separate events.

    Default ceiling is the §10 historical operational bound (0.078). The
    final-season bound (0.056) exists for the 2025/26 partition, which
    monitoring must not use as a reference, so it is not the default here.
    """
    observed = observed_family_null_rates(X)
    violations = {f: r for f, r in observed.items() if _exceeds(r, ceiling)}
    status = Status.ALERT if violations else Status.PASS
    message = (
        f"{len(violations)} family/families exceed the absolute availability ceiling {ceiling}: "
        f"{sorted(violations)}"
        if violations
        else f"all {len(observed)} families within the absolute availability ceiling {ceiling}"
    )
    return MonitoringEvent(
        signal_id=SIGNAL_S1_CEILING,
        signal_name=SIGNAL_NAMES[SIGNAL_S1_CEILING],
        status=status,
        metrics={
            "observed_null_rate_by_family": observed,
            "violations": violations,
            "comparison": "absolute_ceiling",
        },
        baseline={"kind": "absolute_ceiling", "value": float(ceiling)},
        threshold={"ceiling": float(ceiling), "rule": "observed > ceiling"},
        sample_count=int(len(X)),
        context=dict(context or {}),
        message=message,
    )


def evaluate_null_rate_drift(
    X: pd.DataFrame,
    baseline: Mapping[str, float],
    context: Mapping[str, Any] | None = None,
) -> MonitoringEvent:
    """Signal 1(b): per-family null-rate DRIFT -- not the §10 ceiling.

    METRIC: absolute percentage-point deviation, `(observed - baseline) * 100`.
    DIRECTION: only deviation ABOVE the baseline can alert. A family whose
    missingness improves relative to baseline is PASS, never ALERT.
    THRESHOLD: `DRIFT_ALERT_PP` (+2.0 pp). Exactly +2.00 pp is PASS.

    This function never reads the §10 ceilings.
    """
    observed = observed_family_null_rates(X)
    missing = [f for f in FAMILY_COLUMNS if f not in baseline]
    if missing:
        raise MonitoringInputError(f"baseline is missing family/families: {missing}")

    deviations_pp = {f: (observed[f] - float(baseline[f])) * 100.0 for f in FAMILY_COLUMNS}
    breaches = {f: d for f, d in deviations_pp.items() if _exceeds(d, DRIFT_ALERT_PP)}
    status = Status.ALERT if breaches else Status.PASS
    worst = max(deviations_pp, key=deviations_pp.get)
    message = (
        f"{len(breaches)} family/families drifted more than +{DRIFT_ALERT_PP} pp above baseline: "
        f"{sorted(breaches)}"
        if breaches
        else f"no family exceeded +{DRIFT_ALERT_PP} pp drift (worst: {worst} "
             f"{deviations_pp[worst]:+.4f} pp)"
    )
    return MonitoringEvent(
        signal_id=SIGNAL_S1_DRIFT,
        signal_name=SIGNAL_NAMES[SIGNAL_S1_DRIFT],
        status=status,
        metrics={
            "observed_null_rate_by_family": observed,
            "deviation_pp_by_family": deviations_pp,
            "breaches_pp": breaches,
            "comparison": "drift_from_baseline",
        },
        baseline={"kind": "non_test_historical_partitions", "by_family": dict(baseline)},
        threshold={
            "drift_alert_pp": DRIFT_ALERT_PP,
            "rule": "(observed - baseline) * 100 > drift_alert_pp  (above-baseline only)",
        },
        sample_count=int(len(X)),
        context=dict(context or {}),
        message=message,
    )


# =====================================================================
# SIGNAL 2 -- unseen competition_id
# =====================================================================

def evaluate_unseen_competition_ids(
    competition_ids: Iterable[Any],
    context: Mapping[str, Any] | None = None,
) -> MonitoringEvent:
    """Signal 2: unseen-league occurrences.

    Emits DEGRADED (flagged) when any id falls outside the known set, and
    PASS otherwise. It NEVER emits HARD_FAIL and never signals that
    inference should be aborted: the authorized contract is
    degraded-not-failed, and predictive quality for an unseen league
    remains not established by current evidence.
    """
    values = [v for v in competition_ids]
    observed: list[int] = []
    null_count = 0
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            null_count += 1
            continue
        observed.append(int(v))
    unseen = sorted(set(observed) - set(KNOWN_COMPETITION_IDS))
    unseen_count = sum(1 for v in observed if v not in KNOWN_COMPETITION_IDS)

    status = Status.DEGRADED if unseen else Status.PASS
    message = (
        f"{unseen_count} row(s) carry {len(unseen)} unseen competition_id(s) {unseen}: flagged, "
        "inference not failed; predictive quality for unseen leagues is not established by "
        "current evidence"
        if unseen
        else f"all {len(observed)} row(s) carry known competition_ids"
    )
    return MonitoringEvent(
        signal_id=SIGNAL_S2_UNSEEN_COMPETITION,
        signal_name=SIGNAL_NAMES[SIGNAL_S2_UNSEEN_COMPETITION],
        status=status,
        metrics={
            "observed_ids": sorted(set(observed)),
            "unseen_ids": unseen,
            "unseen_row_count": unseen_count,
            "null_competition_id_count": null_count,
        },
        baseline={"kind": "known_competition_id_set", "value": sorted(KNOWN_COMPETITION_IDS)},
        threshold={"rule": "any id outside the known set is flagged; never a hard failure"},
        sample_count=len(values),
        context=dict(context or {}),
        message=message,
    )


# =====================================================================
# SIGNAL 3 -- predicted class distribution (prediction-time only)
# =====================================================================

def load_stored_base_rate_reference(
    artifact_path: str | Path = BASE_RATE_ARTIFACT,
) -> dict[str, float]:
    """Stored historical base-rate reference for Signal 3.

    Read from the frozen Phase 3 artifact (folds 1-3 validation, non-test)
    rather than hardcoded, so the reference cannot drift from the recorded
    evidence. The artifact is opened read-only and never written.

    2025/26 is not involved: the artifact records walk-forward validation
    diagnostics only.
    """
    path = Path(artifact_path)
    if not path.exists():
        raise MonitoringInputError(f"base-rate reference artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        freqs = payload["diagnostics"]["actual_class_frequency"]
    except (KeyError, TypeError) as exc:
        raise MonitoringInputError(
            f"artifact {path} does not contain diagnostics.actual_class_frequency"
        ) from exc
    missing = [c for c in CLASS_ORDER_NAMES if c not in freqs]
    if missing:
        raise MonitoringInputError(f"base-rate reference is missing class(es): {missing}")
    return {c: float(freqs[c]) for c in CLASS_ORDER_NAMES}


def _as_probability_matrix(P: Any) -> np.ndarray:
    arr = np.asarray(P, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(CLASS_ORDER_NAMES):
        raise MonitoringInputError(
            f"probabilities must have shape (n, {len(CLASS_ORDER_NAMES)}), got {arr.shape}"
        )
    if len(arr) == 0:
        raise MonitoringInputError("cannot evaluate an empty probability matrix")
    return arr


def evaluate_class_distribution(
    P: Any,
    base_rates: Mapping[str, float] | None = None,
    context: Mapping[str, Any] | None = None,
) -> MonitoringEvent:
    """Signal 3: mean predicted probability MASS per class vs base rates.

    PRIMARY QUANTITY: mean predicted probability mass per class. Argmax
    class share is computed and recorded as a SECONDARY DIAGNOSTIC and
    takes no part in determining `status` -- the status is a pure function
    of the probability-mass deviations.

    THRESHOLD: +/- `CLASS_DISTRIBUTION_TOLERANCE_PP` (5.0 pp), applied to
    H, D and A INDEPENDENTLY. A deviation exactly at 5.00 pp is PASS.

    TIMING: prediction-time only. The baseline is a stored historical
    reference; no realized outcomes are consulted, and the rolling live
    realized-frequency baseline (former option C-4) is excluded.
    """
    arr = _as_probability_matrix(P)
    reference = dict(base_rates) if base_rates is not None else load_stored_base_rate_reference()
    missing = [c for c in CLASS_ORDER_NAMES if c not in reference]
    if missing:
        raise MonitoringInputError(f"base-rate reference is missing class(es): {missing}")

    mass = {c: float(arr[:, i].mean()) for i, c in enumerate(CLASS_ORDER_NAMES)}
    deviations_pp = {c: (mass[c] - float(reference[c])) * 100.0 for c in CLASS_ORDER_NAMES}
    breaches = {
        c: d for c, d in deviations_pp.items() if _exceeds(abs(d), CLASS_DISTRIBUTION_TOLERANCE_PP)
    }

    # SECONDARY DIAGNOSTIC ONLY -- never used below this point.
    argmax_idx = arr.argmax(axis=1)
    argmax_share = {
        c: float((argmax_idx == i).mean()) for i, c in enumerate(CLASS_ORDER_NAMES)
    }

    status = Status.ALERT if breaches else Status.PASS
    message = (
        f"probability-mass deviation exceeded +/-{CLASS_DISTRIBUTION_TOLERANCE_PP} pp for "
        f"class(es) {sorted(breaches)}"
        if breaches
        else f"all classes within +/-{CLASS_DISTRIBUTION_TOLERANCE_PP} pp of the stored base rates"
    )
    return MonitoringEvent(
        signal_id=SIGNAL_S3_CLASS_DISTRIBUTION,
        signal_name=SIGNAL_NAMES[SIGNAL_S3_CLASS_DISTRIBUTION],
        status=status,
        metrics={
            "mean_predicted_probability_mass": mass,
            "deviation_pp": deviations_pp,
            "breaches_pp": breaches,
            "primary_quantity": "mean_predicted_probability_mass",
            "diagnostic_only_argmax_share": argmax_share,
            "diagnostic_note": (
                "argmax share is a secondary diagnostic and does not participate in the "
                "primary alert; the candidate is not a reliable Draw-class predictor "
                "(Condition 2, DRAW_LIMITATION_ACCEPTED)"
            ),
        },
        baseline={"kind": "stored_non_test_historical_base_rates", "by_class": reference},
        threshold={
            "tolerance_pp": CLASS_DISTRIBUTION_TOLERANCE_PP,
            "rule": "abs((mass - base_rate) * 100) > tolerance_pp, evaluated per class",
        },
        sample_count=int(len(arr)),
        context=dict(context or {}),
        message=message,
    )


# =====================================================================
# SIGNALS 4 and 6 -- post-outcome only
# =====================================================================

def _parse_ts(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise MonitoringInputError(f"{field_name} {value!r} is not ISO-8601") from exc


def select_eligible_observations(
    records: Iterable[Mapping[str, Any]],
    as_of: datetime,
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Apply the locked post-outcome eligibility policy.

    A record is eligible only when ALL of the following hold:

      1. it is settled (`settled_at` present and not None);
      2. MATURITY -- `settled_at - predicted_at >= SETTLEMENT_MATURITY_DAYS`
         (7 days). This is the literal authorized rule, implemented exactly
         as specified and deliberately not reinterpreted;
      3. WINDOW -- `settled_at` lies within the rolling
         `OBSERVATION_WINDOW_DAYS` (30 days) ending at `as_of`.

    The window is applied to `settled_at` because every eligible
    observation is by definition settled, and a post-outcome signal
    measures recently-settled results. This reading is recorded in the
    decision matrix (§13.2) as an implementation detail open to
    confirmation; both timestamps are preserved in the emitted metrics so
    the choice is auditable rather than hidden.

    Returns the eligible records and a breakdown of exclusion reasons.
    """
    window_start = as_of - timedelta(days=OBSERVATION_WINDOW_DAYS)
    maturity = timedelta(days=SETTLEMENT_MATURITY_DAYS)

    eligible: list[Mapping[str, Any]] = []
    reasons = {"unsettled": 0, "immature": 0, "outside_window": 0, "eligible": 0}

    for rec in records:
        settled_raw = rec.get("settled_at")
        if settled_raw is None:
            reasons["unsettled"] += 1
            continue
        settled_at = _parse_ts(settled_raw, "settled_at")
        predicted_at = _parse_ts(rec["predicted_at"], "predicted_at")
        if settled_at - predicted_at < maturity:
            reasons["immature"] += 1
            continue
        if not (window_start <= settled_at <= as_of):
            reasons["outside_window"] += 1
            continue
        eligible.append(rec)
        reasons["eligible"] += 1

    return eligible, reasons


def _post_outcome_stats(eligible: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Mean predicted mass, realized frequency, and argmax share per class."""
    P = np.asarray([[float(r["probabilities"][c]) for c in CLASS_ORDER_NAMES] for r in eligible], dtype=float)
    outcomes = [str(r["outcome"]) for r in eligible]
    mass = {c: float(P[:, i].mean()) for i, c in enumerate(CLASS_ORDER_NAMES)}
    realized = {c: float(sum(1 for o in outcomes if o == c) / len(outcomes)) for c in CLASS_ORDER_NAMES}
    argmax_idx = P.argmax(axis=1)
    argmax_share = {c: float((argmax_idx == i).mean()) for i, c in enumerate(CLASS_ORDER_NAMES)}
    return mass, realized, argmax_share


def _insufficient_sample_event(
    signal_id: str,
    n_eligible: int,
    reasons: Mapping[str, int],
    as_of: datetime,
    context: Mapping[str, Any] | None,
    extra_threshold: Mapping[str, Any] | None = None,
) -> MonitoringEvent:
    threshold: dict[str, Any] = {
        "tolerance_pp": CALIBRATION_TOLERANCE_PP,
        "minimum_sample": MIN_SETTLED_SAMPLE,
        "observation_window_days": OBSERVATION_WINDOW_DAYS,
        "settlement_maturity_days": SETTLEMENT_MATURITY_DAYS,
    }
    threshold.update(dict(extra_threshold or {}))
    return MonitoringEvent(
        signal_id=signal_id,
        signal_name=SIGNAL_NAMES[signal_id],
        status=Status.INSUFFICIENT_SAMPLE,
        metrics={
            "eligible_observations": n_eligible,
            "exclusion_breakdown": dict(reasons),
            "as_of": as_of.isoformat(),
            "note": (
                "below the authorized minimum sample size; no PASS or ALERT verdict is "
                "emitted, and no tolerance comparison was performed"
            ),
        },
        baseline={"kind": "realized_outcome_frequency_in_window", "value": None},
        threshold=threshold,
        sample_count=int(n_eligible),
        context=dict(context or {}),
        message=(
            f"{n_eligible} eligible settled observation(s) < required {MIN_SETTLED_SAMPLE}: "
            "INSUFFICIENT_SAMPLE"
        ),
    )


def evaluate_calibration_post_outcome(
    records: Iterable[Mapping[str, Any]],
    as_of: datetime,
    context: Mapping[str, Any] | None = None,
) -> MonitoringEvent:
    """Signal 4: mean predicted probability vs realized outcome frequency.

    POST-OUTCOME ONLY. Cannot be evaluated at prediction time, because
    realized outcomes do not exist then.

    POLICY (all authorized): +/- 5.0 pp per class, minimum 200 eligible
    settled observations, rolling 30-day window, 7-day settlement
    maturity. Below the minimum, the status is INSUFFICIENT_SAMPLE --
    never PASS, never ALERT.

    Each record needs: `predicted_at`, `settled_at`, `probabilities`
    (mapping with H/D/A) and `outcome` (one of H/D/A).
    """
    eligible, reasons = select_eligible_observations(records, as_of)
    if len(eligible) < MIN_SETTLED_SAMPLE:
        return _insufficient_sample_event(
            SIGNAL_S4_CALIBRATION, len(eligible), reasons, as_of, context
        )

    mass, realized, argmax_share = _post_outcome_stats(eligible)
    deviations_pp = {c: (mass[c] - realized[c]) * 100.0 for c in CLASS_ORDER_NAMES}
    breaches = {
        c: d for c, d in deviations_pp.items() if _exceeds(abs(d), CALIBRATION_TOLERANCE_PP)
    }
    status = Status.ALERT if breaches else Status.PASS
    return MonitoringEvent(
        signal_id=SIGNAL_S4_CALIBRATION,
        signal_name=SIGNAL_NAMES[SIGNAL_S4_CALIBRATION],
        status=status,
        metrics={
            "mean_predicted_probability": mass,
            "realized_outcome_frequency": realized,
            "deviation_pp": deviations_pp,
            "breaches_pp": breaches,
            "exclusion_breakdown": dict(reasons),
            "as_of": as_of.isoformat(),
            "diagnostic_only_argmax_share": argmax_share,
        },
        baseline={"kind": "realized_outcome_frequency_in_window", "by_class": realized},
        threshold={
            "tolerance_pp": CALIBRATION_TOLERANCE_PP,
            "minimum_sample": MIN_SETTLED_SAMPLE,
            "observation_window_days": OBSERVATION_WINDOW_DAYS,
            "settlement_maturity_days": SETTLEMENT_MATURITY_DAYS,
            "rule": "abs((mean_predicted - realized) * 100) > tolerance_pp, per class",
        },
        sample_count=int(len(eligible)),
        context=dict(context or {}),
        message=(
            f"calibration deviation exceeded +/-{CALIBRATION_TOLERANCE_PP} pp for class(es) "
            f"{sorted(breaches)} over {len(eligible)} settled observations"
            if breaches
            else f"all classes within +/-{CALIBRATION_TOLERANCE_PP} pp over "
                 f"{len(eligible)} settled observations"
        ),
    )


def evaluate_draw_behaviour(
    records: Iterable[Mapping[str, Any]],
    as_of: datetime,
    context: Mapping[str, Any] | None = None,
) -> MonitoringEvent:
    """Signal 6: mean predicted P(draw) vs realized draw frequency.

    Uses EXACTLY Signal 4's post-outcome policy: +/- 5.0 pp, minimum 200
    eligible settled observations, 30-day rolling window, 7-day settlement
    maturity, INSUFFICIENT_SAMPLE below the minimum.

    The previously rejected +/-0.03 Phase 3 heuristic is NOT used, and no
    argmax-based Draw alert exists: Draw argmax share is recorded as a
    secondary diagnostic only, so this signal can never conflict with
    Signal 3. Poor historical Draw classification remains an accepted
    limitation under Condition 2; nothing here claims the model is a
    reliable Draw classifier.
    """
    eligible, reasons = select_eligible_observations(records, as_of)
    if len(eligible) < MIN_SETTLED_SAMPLE:
        return _insufficient_sample_event(
            SIGNAL_S6_DRAW_BEHAVIOUR, len(eligible), reasons, as_of, context,
            extra_threshold={"monitored_class": "D"},
        )

    mass, realized, argmax_share = _post_outcome_stats(eligible)
    deviation_pp = (mass["D"] - realized["D"]) * 100.0
    breach = _exceeds(abs(deviation_pp), CALIBRATION_TOLERANCE_PP)
    status = Status.ALERT if breach else Status.PASS
    return MonitoringEvent(
        signal_id=SIGNAL_S6_DRAW_BEHAVIOUR,
        signal_name=SIGNAL_NAMES[SIGNAL_S6_DRAW_BEHAVIOUR],
        status=status,
        metrics={
            "mean_predicted_p_draw": mass["D"],
            "realized_draw_frequency": realized["D"],
            "deviation_pp": deviation_pp,
            "exclusion_breakdown": dict(reasons),
            "as_of": as_of.isoformat(),
            "diagnostic_only_argmax_draw_share": argmax_share["D"],
            "accepted_limitation": (
                "poor historical Draw classification is an accepted limitation under "
                "Condition 2 (DRAW_LIMITATION_ACCEPTED, scoped to the probability-output "
                "contract); this signal monitors probability mass, not class assignment"
            ),
        },
        baseline={"kind": "realized_draw_frequency_in_window", "value": realized["D"]},
        threshold={
            "tolerance_pp": CALIBRATION_TOLERANCE_PP,
            "minimum_sample": MIN_SETTLED_SAMPLE,
            "observation_window_days": OBSERVATION_WINDOW_DAYS,
            "settlement_maturity_days": SETTLEMENT_MATURITY_DAYS,
            "monitored_class": "D",
            "rule": "abs((mean_predicted_p_draw - realized_draw_frequency) * 100) > tolerance_pp",
        },
        sample_count=int(len(eligible)),
        context=dict(context or {}),
        message=(
            f"draw probability-mass deviation {deviation_pp:+.4f} pp exceeded "
            f"+/-{CALIBRATION_TOLERANCE_PP} pp over {len(eligible)} settled observations"
            if breach
            else f"draw probability-mass deviation {deviation_pp:+.4f} pp within "
                 f"+/-{CALIBRATION_TOLERANCE_PP} pp over {len(eligible)} settled observations"
        ),
    )


__all__ = [
    "MonitoringInputError",
    "observed_family_null_rates",
    "build_null_rate_baseline",
    "evaluate_null_rate_ceiling",
    "evaluate_null_rate_drift",
    "evaluate_unseen_competition_ids",
    "load_stored_base_rate_reference",
    "evaluate_class_distribution",
    "select_eligible_observations",
    "evaluate_calibration_post_outcome",
    "evaluate_draw_behaviour",
]
