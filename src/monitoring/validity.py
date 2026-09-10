"""Signal 5 -- probability-validity monitoring (external wrapper).

GOVERNANCE CONSTRAINTS ENCODED HERE
    - `evaluate.validate_probabilities` is the SOLE authority on validity.
      This module never re-implements it and never overrides its verdict.
    - `src/models/evaluate.py` is NOT modified. It is one of the seven
      frozen V1 files pinned in data/audit/phase4c_prerun_manifest.json.
    - `candidate_contract.predict_candidate` is NOT modified. Monitoring
      observes it from OUTSIDE via the wrapper below, so existing
      prediction semantics are untouched.
    - On failure the ORIGINAL exception object is re-raised unchanged --
      same type, same message, same instance -- so a validity failure
      remains a hard inference failure exactly as §8 requires.
    - Invalid probabilities are never swallowed, normalized, clipped,
      repaired, or downgraded. There is no code path here that returns a
      "fixed" probability matrix.

ROW-LEVEL COUNTS ARE DIAGNOSTIC ONLY
    `count_invalid_rows` exists so an operator can see HOW MANY rows were
    invalid, which a raised exception does not record. It is never used to
    decide pass/fail: that decision has already been made by
    `validate_probabilities` before the counting code runs.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import (
    CLASS_ORDER_NAMES,
    SIGNAL_NAMES,
    SIGNAL_S5_PROBABILITY_VALIDITY,
    Status,
)
from .events import MonitoringEvent


def count_invalid_rows(P: Any, atol: float = 1e-6) -> dict[str, int]:
    """Diagnostic row-level breakdown of why a matrix was rejected.

    NOT an authority on validity -- see the module docstring. Mirrors the
    §8 conditions purely to attribute counts, and is only ever called
    after `validate_probabilities` has already raised.

    The `atol` default matches the documented §8 figure. Note the known,
    disclosed deviation recorded in §8 of the spec: the authoritative
    validator uses `np.allclose(..., atol=1e-6)`, whose NumPy default
    `rtol=1e-5` makes its effective tolerance ~1.1e-5. Because this
    function is diagnostic only, a marginal disagreement between the two
    cannot change any verdict; it would only change an explanatory count.
    """
    arr = np.asarray(P, dtype=float)
    if arr.ndim != 2:
        return {
            "n_rows": 0, "invalid_rows": 0, "non_finite_rows": 0,
            "negative_rows": 0, "bad_sum_rows": 0, "malformed_shape": 1,
        }
    non_finite = ~np.isfinite(arr)
    negative = arr < 0
    with np.errstate(invalid="ignore"):
        row_sums = np.nansum(np.where(np.isfinite(arr), arr, 0.0), axis=1)
    bad_sum = np.abs(row_sums - 1.0) > atol
    non_finite_rows = non_finite.any(axis=1)
    negative_rows = negative.any(axis=1)
    invalid = non_finite_rows | negative_rows | bad_sum
    return {
        "n_rows": int(arr.shape[0]),
        "invalid_rows": int(invalid.sum()),
        "non_finite_rows": int(non_finite_rows.sum()),
        "negative_rows": int(negative_rows.sum()),
        "bad_sum_rows": int(bad_sum.sum()),
        "malformed_shape": 0 if arr.shape[1:] == (len(CLASS_ORDER_NAMES),) else 1,
    }


def record_validity_failure(
    exc: BaseException,
    P: Any = None,
    context: Mapping[str, Any] | None = None,
) -> MonitoringEvent:
    """Build the HARD_FAIL event for a validity failure.

    Zero tolerance: a single invalid row is a hard failure. This function
    only RECORDS; the caller is responsible for re-raising, and every
    caller in this module does so unchanged.
    """
    breakdown = count_invalid_rows(P) if P is not None else {}
    return MonitoringEvent(
        signal_id=SIGNAL_S5_PROBABILITY_VALIDITY,
        signal_name=SIGNAL_NAMES[SIGNAL_S5_PROBABILITY_VALIDITY],
        status=Status.HARD_FAIL,
        metrics={
            "invalid_batches": 1,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "row_breakdown_diagnostic_only": breakdown,
            "authority": "models.evaluate.validate_probabilities",
        },
        baseline={"kind": "absolute_validity_rule", "value": "zero_tolerance"},
        threshold={
            "tolerance": 0,
            "rule": "any invalid probability vector is a hard inference failure",
        },
        sample_count=int(breakdown.get("n_rows", 0)),
        context=dict(context or {}),
        message=f"probability validity failure ({type(exc).__name__}): hard inference failure",
    )


def record_validity_pass(
    P: Any,
    context: Mapping[str, Any] | None = None,
) -> MonitoringEvent:
    """Build the PASS event for a batch that satisfied the validity rule."""
    arr = np.asarray(P, dtype=float)
    return MonitoringEvent(
        signal_id=SIGNAL_S5_PROBABILITY_VALIDITY,
        signal_name=SIGNAL_NAMES[SIGNAL_S5_PROBABILITY_VALIDITY],
        status=Status.PASS,
        metrics={
            "invalid_rows": 0,
            "invalid_batches": 0,
            "authority": "models.evaluate.validate_probabilities",
        },
        baseline={"kind": "absolute_validity_rule", "value": "zero_tolerance"},
        threshold={
            "tolerance": 0,
            "rule": "any invalid probability vector is a hard inference failure",
        },
        sample_count=int(arr.shape[0]) if arr.ndim == 2 else 0,
        message=f"probability validity passed for {arr.shape[0] if arr.ndim == 2 else 0} row(s)",
        context=dict(context or {}),
    )


def _default_predict_candidate(*args: Any, **kwargs: Any) -> Any:
    """Resolve the real prediction path lazily.

    Imported inside the call so this module stays importable in
    environments without scikit-learn (`candidate_contract.predict_candidate`
    imports V1's training module at call time).
    """
    from models.candidate_contract import predict_candidate

    return predict_candidate(*args, **kwargs)


def monitored_predict_candidate(
    model: Any,
    preprocessor: Any,
    X: Any,
    fixture_ids: Sequence[Any],
    store: Any = None,
    context: Mapping[str, Any] | None = None,
    predict_fn: Callable[..., Any] | None = None,
) -> Any:
    """External monitoring wrapper around `predict_candidate`.

    Behaviour is deliberately transparent: on success the prediction is
    returned unchanged; on failure the ORIGINAL exception is re-raised
    unchanged. The only added effect is that a monitoring event is
    recorded.

    Args:
        store: optional event sink exposing `.append(event)`.
        predict_fn: injection point for tests; defaults to the real
            `candidate_contract.predict_candidate`. Production callers
            should not pass this.

    Returns:
        Exactly what `predict_candidate` returned.

    Raises:
        The original exception, unchanged, whenever prediction or
        validation fails.
    """
    fn = predict_fn or _default_predict_candidate
    try:
        prediction = fn(model, preprocessor, X, fixture_ids)
    except BaseException as exc:  # noqa: BLE001 -- re-raised unchanged below
        probabilities = getattr(exc, "probabilities", None)
        event = record_validity_failure(exc, probabilities, context)
        if store is not None:
            store.append(event)
        raise  # bare raise: same object, same traceback, semantics unchanged

    if store is not None:
        store.append(record_validity_pass(getattr(prediction, "probabilities", None), context))
    return prediction


__all__ = [
    "count_invalid_rows",
    "record_validity_failure",
    "record_validity_pass",
    "monitored_predict_candidate",
]
