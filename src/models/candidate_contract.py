"""V2 candidate feature contract, availability check, and the intended
candidate prediction path.

Contents, by the gate that introduced them:
  - Gate 1: the frozen 80-column feature contract (§2) and its verifier.
  - Gate 3: `check_feature_availability` -- the §10 pre-inference check.
  - Gate 8: `predict_candidate` -- the intended prediction path (§7).

WHAT THIS IS NOT:
  - Not a model and not a trainer. `predict_candidate` requires an
    already-fitted model and preprocessor produced by V1's frozen
    training path; nothing here fits, tunes, or calibrates anything.
  - Not a V2 production artifact. `MODEL_VERSION` stays "v1.0"; this
    module never reads it for modification and never writes it.
  - Not a replacement for V1. V1's contract
    (`config.APPROVED_FEATURE_COLUMNS_V1`) is untouched and independent.
  - No calibration, no hyperparameter search, no final-test access.
    2025/26 is unreachable from this module: it imports no split
    function and no season constant.

MODULE NAMING: deliberately not named `v2_*`. Existing governance tests
(e.g. tests/test_phase5_v2_review.py::TestNoV2SideEffects) assert that no
`src/models/*v2*.py` module exists, encoding the rule "no V2 model or
artifact has been created". A feature-contract definition is not a V2
model, and the correct response to that invariant is to respect it
rather than weaken the test.

CONTRACT SOURCE OF TRUTH: `ablation.MODEL_B_COLUMNS`, re-exported here in
its exact stored order. This module deliberately does NOT restate the 80
column names as a literal list — duplicating them would create two
sources of truth that could silently diverge. Gate 1's verifier asserts
identity with the Phase 4A definition instead.

CONDITION CLOSURES CARRIED FORWARD (Phase 5A / Phase 5B):
  - Condition 1: RETAIN_OMISSION_SUPPORTED. `league_home_advantage_season`
    stays OMITTED. Adding it worsened mean validation log loss
    (0.9993791056968738 -> 0.9996821579734537, delta +0.000303052276579896).
    It must not be added back here.
  - Condition 2: DRAW_LIMITATION_ACCEPTED (scoped to the probability
    output contract; reopen if a hard predicted class enters the contract).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .ablation import (
    MODEL_B_COLUMNS,
    SHOTS_CORE_COLUMNS,
    SHOTS_ON_CORE_COLUMNS,
    XG_CORE_COLUMNS,
    select_feature_columns,
)
from .config import ALLOWED_LABELS, APPROVED_FEATURE_COLUMNS_V1, X_EXCLUDED_COLUMNS

CANDIDATE_NAME = "model_b"

#: Local name for the league-context column, so the availability check
#: does not depend on importing V1's `RECOMMENDED_CONTEXT_FEATURE`.
RECOMMENDED_CONTEXT_FEATURE_NAME = "competition_id"

#: The frozen class order, as a tuple, for the inference boundary's
#: output contract. Kept as a literal-free derivation from the same
#: source of truth V1 uses (`config.ALLOWED_LABELS`), so it cannot drift
#: from `baselines.CLASS_ORDER`.
CLASS_ORDER_NAMES: tuple[str, ...] = tuple(ALLOWED_LABELS)

#: The frozen V2 candidate feature contract: exactly the Phase 4A Model B
#: columns, in their exact stored order. Order is part of the contract.
CANDIDATE_FEATURE_COLUMNS: tuple[str, ...] = tuple(MODEL_B_COLUMNS)

CANDIDATE_N_FEATURES: int = 80

#: Documented family composition (spec §2). Used only for verification
#: reporting -- the authoritative membership is the tuple above.
CANDIDATE_FAMILY_COUNTS: dict[str, int] = {
    "goals_core": 18,
    "form": 20,
    "strength": 5,
    "competition_id": 1,
    "shots_core": 18,
    "shots_on_core": 18,
}

#: Condition 1 (Phase 5A): this V1 feature stays omitted. Named
#: explicitly so any attempt to reintroduce it fails a test rather than
#: passing unnoticed.
OMITTED_V1_FEATURE: str = "league_home_advantage_season"

CONDITION_CLOSURES: dict[str, str] = {
    "condition_1": "RETAIN_OMISSION_SUPPORTED",
    "condition_2": "DRAW_LIMITATION_ACCEPTED",
}


# ---------------------------------------------------------------------
# Gate 3 — feature availability bounds (spec §10).
#
# §10 states: "Null rate per family within documented historical bounds
# (<= ~7.7% observed historically; ~5.5% on 2025/26)."
#
# PRECISION NOTE (documented rather than silently rounded): the "~7.7%"
# figure traces to the Phase 5 review's coverage table, where fold_1's
# training partition shows 0.0772 for shots_core / shots_on_core. The
# exact observed value is 0.077218 (six `away_*_last10` shots columns).
# A bound written to describe an observation cannot be violated by that
# same observation, but "<= 0.077" read literally would exclude it by
# 0.000218. The bound is therefore encoded at a precision that admits
# the documented observation, and the discrepancy is reported rather
# than papered over -- see the Gate 3 report.
MAX_HISTORICAL_NULL_RATE: float = 0.078
MAX_FINAL_SEASON_NULL_RATE: float = 0.056

#: The five league IDs present in the modelled data (spec §10). An
#: unseen value must be FLAGGED, not raised on: the V1 preprocessing
#: maps an unknown category to an all-zero one-hot row, so behaviour is
#: degraded-not-failed. Predictive quality for an unseen league is not
#: established by current evidence.
KNOWN_COMPETITION_IDS: frozenset[int] = frozenset({200, 419, 423, 477, 499})


class FeatureContractViolation(RuntimeError):
    """Raised when the implemented contract does not match the frozen
    Phase 4A Model B definition. Deliberately fatal: Gate 1's stop
    condition is 'any column added/removed/reordered'."""


def verify_feature_contract() -> dict[str, Any]:
    """Mechanically verifies Gate 1's acceptance condition:
    exact set AND exact order match against `ablation.MODEL_B_COLUMNS`.

    Raises `FeatureContractViolation` on any mismatch; returns a
    structured verification record on success.
    """
    problems: list[str] = []

    if len(CANDIDATE_FEATURE_COLUMNS) != CANDIDATE_N_FEATURES:
        problems.append(
            f"contract has {len(CANDIDATE_FEATURE_COLUMNS)} columns, expected {CANDIDATE_N_FEATURES}"
        )
    if tuple(CANDIDATE_FEATURE_COLUMNS) != tuple(MODEL_B_COLUMNS):
        if set(CANDIDATE_FEATURE_COLUMNS) == set(MODEL_B_COLUMNS):
            problems.append("contract set matches MODEL_B_COLUMNS but ORDER differs")
        else:
            added = sorted(set(CANDIDATE_FEATURE_COLUMNS) - set(MODEL_B_COLUMNS))
            removed = sorted(set(MODEL_B_COLUMNS) - set(CANDIDATE_FEATURE_COLUMNS))
            problems.append(f"contract set differs from MODEL_B_COLUMNS; added={added} removed={removed}")
    if len(set(CANDIDATE_FEATURE_COLUMNS)) != len(CANDIDATE_FEATURE_COLUMNS):
        problems.append("contract contains duplicate columns")
    if OMITTED_V1_FEATURE in CANDIDATE_FEATURE_COLUMNS:
        problems.append(
            f"{OMITTED_V1_FEATURE!r} is present but Condition 1 (RETAIN_OMISSION_SUPPORTED) requires it omitted"
        )
    xg = sorted(set(CANDIDATE_FEATURE_COLUMNS) & set(XG_CORE_COLUMNS))
    if xg:
        problems.append(f"xG columns present in contract: {xg}")
    excluded = sorted(set(CANDIDATE_FEATURE_COLUMNS) & set(X_EXCLUDED_COLUMNS))
    if excluded:
        problems.append(f"label/bookkeeping columns present in contract: {excluded}")

    if problems:
        raise FeatureContractViolation("Gate 1 feature contract invalid: " + "; ".join(problems))

    v1 = set(APPROVED_FEATURE_COLUMNS_V1)
    cand = set(CANDIDATE_FEATURE_COLUMNS)
    return {
        "candidate_name": CANDIDATE_NAME,
        "n_features": len(CANDIDATE_FEATURE_COLUMNS),
        "matches_model_b_exactly": True,
        "order_preserved": True,
        "omitted_v1_feature": OMITTED_V1_FEATURE,
        "omitted_v1_feature_absent": OMITTED_V1_FEATURE not in cand,
        "v1_minus_candidate": sorted(v1 - cand),
        "candidate_minus_v1_count": len(cand - v1),
        "intersection_with_v1": len(v1 & cand),
        "is_strict_superset_of_v1": v1 < cand,
        "xg_columns_present": 0,
        "excluded_columns_present": 0,
        "shots_columns": len(cand & set(SHOTS_CORE_COLUMNS)),
        "shots_on_columns": len(cand & set(SHOTS_ON_CORE_COLUMNS)),
        "condition_closures": dict(CONDITION_CLOSURES),
        "verified": True,
    }


def check_feature_availability(
    X: pd.DataFrame,
    max_null_rate: float = MAX_HISTORICAL_NULL_RATE,
    partition_name: str = "batch",
) -> dict[str, Any]:
    """Gate 3 pre-inference coverage check against spec §10.

    Returns a structured report; never raises on data conditions, so a
    caller can inspect every finding at once rather than only the first.
    `passed` is False if any §10 requirement is unmet.

    Checks: all 80 contract columns present; no contract column entirely
    null; every column's null rate within `max_null_rate`; no unexpected
    constant column; `competition_id` values within
    `KNOWN_COMPETITION_IDS`, with any unseen league flagged.
    """
    missing = [c for c in CANDIDATE_FEATURE_COLUMNS if c not in X.columns]
    if missing:
        return {
            "partition": partition_name, "passed": False, "n_rows": len(X),
            "missing_columns": missing, "all_null_columns": [], "constant_columns": [],
            "out_of_bounds_columns": [], "max_null_rate_observed": None,
            "max_null_rate_column": None, "max_null_rate_allowed": max_null_rate,
            "competition_id": None,
            "failures": [f"{len(missing)} contract column(s) missing"],
        }

    present = X[list(CANDIDATE_FEATURE_COLUMNS)]
    null_rates = {c: float(present[c].isna().mean()) for c in CANDIDATE_FEATURE_COLUMNS}

    all_null = [c for c in CANDIDATE_FEATURE_COLUMNS if present[c].isna().all()]
    constant = [
        c for c in CANDIDATE_FEATURE_COLUMNS
        if not present[c].isna().all() and present[c].dropna().nunique() <= 1
    ]
    out_of_bounds = {c: r for c, r in null_rates.items() if r > max_null_rate}

    worst_col = max(null_rates, key=null_rates.get) if null_rates else None
    worst_rate = null_rates[worst_col] if worst_col else None

    comp = present[RECOMMENDED_CONTEXT_FEATURE_NAME]
    observed_ids = sorted(int(v) for v in comp.dropna().unique())
    unseen = sorted(set(observed_ids) - KNOWN_COMPETITION_IDS)
    comp_report = {
        "observed_ids": observed_ids,
        "known_ids": sorted(KNOWN_COMPETITION_IDS),
        "unseen_ids": unseen,
        "unseen_flagged": bool(unseen),
        "null_count": int(comp.isna().sum()),
        "note": (
            "Unseen league IDs are flagged, not fatal: V1 preprocessing maps an unknown "
            "category to an all-zero one-hot row (degraded-not-failed). Predictive quality "
            "for an unseen league is not established by current evidence."
        ),
    }

    failures: list[str] = []
    if all_null:
        failures.append(f"{len(all_null)} contract column(s) entirely null: {all_null}")
    if constant:
        failures.append(f"{len(constant)} unexpected constant column(s): {constant}")
    if out_of_bounds:
        failures.append(
            f"{len(out_of_bounds)} column(s) exceed max null rate {max_null_rate}: "
            f"{sorted(out_of_bounds.items(), key=lambda kv: -kv[1])[:5]}"
        )
    if comp_report["null_count"]:
        failures.append(f"competition_id has {comp_report['null_count']} null value(s)")

    return {
        "partition": partition_name,
        "passed": not failures,
        "n_rows": len(X),
        "n_contract_columns_present": len(CANDIDATE_FEATURE_COLUMNS),
        "missing_columns": [],
        "all_null_columns": all_null,
        "constant_columns": constant,
        "out_of_bounds_columns": sorted(out_of_bounds),
        "max_null_rate_observed": worst_rate,
        "max_null_rate_column": worst_col,
        "max_null_rate_allowed": max_null_rate,
        "null_rate_by_column": null_rates,
        "competition_id": comp_report,
        "failures": failures,
    }


def select_candidate_features(X: pd.DataFrame) -> pd.DataFrame:
    """Restricts a feature frame to the candidate contract, in contract
    order, failing loudly if any column is absent.

    Reuses `ablation.select_feature_columns` rather than reimplementing
    the selection/failure semantics, so behaviour is identical to every
    other phase's feature selection.
    """
    verify_feature_contract()
    return select_feature_columns(X, CANDIDATE_FEATURE_COLUMNS, group_name=CANDIDATE_NAME)


# ---------------------------------------------------------------------
# Gate 8 — the intended candidate prediction path (spec §7).
#
# Deliberately minimal, and deliberately NOT a sixth copy of the
# experiment harnesses' private `_predictions_frame()` helpers. Those are
# reporting conveniences that bundle labels and argmax classes for
# audit CSVs; this is an inference boundary that enforces the §7
# interface contract and returns probabilities aligned to fixture_id.
#
# It does not fit anything. The caller supplies a model and preprocessor
# already fitted by V1's frozen `train.train_logistic_regression`, so V1
# training behaviour is reused unchanged rather than re-implemented.
#
# `fixture_id` is bookkeeping/alignment metadata ONLY. It is structurally
# incapable of becoming a feature: the design matrix is built by
# `select_candidate_features`, which returns exactly the 80 contract
# columns, and `fixture_id` is not one of them (it lives in
# `config.X_EXCLUDED_COLUMNS`).
# ---------------------------------------------------------------------
class InferenceContractViolation(RuntimeError):
    """Raised when an inference call violates the §7 interface contract.

    Deliberately fatal: Gate 8's stop condition is "interface mismatch or
    silent column substitution", so a mismatch must never be tolerated or
    silently repaired.
    """


@dataclass(frozen=True)
class CandidatePrediction:
    """Result of the intended prediction path.

    `fixture_ids[i]` corresponds to `probabilities[i]` -- alignment is
    verified at construction, never assumed.
    """
    fixture_ids: tuple
    probabilities: np.ndarray
    class_order: tuple
    n_rows: int

    def as_frame(self) -> pd.DataFrame:
        """Long-form view for callers that prefer a frame. Column names
        follow the project's existing p_home/p_draw/p_away convention,
        positionally derived from `class_order` rather than hard-coded.
        """
        name_by_class = {"H": "p_home", "D": "p_draw", "A": "p_away"}
        data: dict[str, Any] = {"fixture_id": list(self.fixture_ids)}
        for i, cls in enumerate(self.class_order):
            data[name_by_class[cls]] = self.probabilities[:, i]
        return pd.DataFrame(data)


def predict_candidate(
    model: Any,
    preprocessor: Any,
    X: pd.DataFrame,
    fixture_ids: Sequence,
) -> CandidatePrediction:
    """The intended candidate prediction path (spec §7).

    Args:
        model: an already-fitted estimator from V1's frozen training path.
        preprocessor: the matching already-fitted
            `train.LogisticRegressionPreprocessor` (transform-only here --
            it is never refitted).
        X: a frame containing the complete 80-column candidate contract.
            Extra columns are ignored for modelling purposes; missing
            contract columns raise.
        fixture_ids: alignment keys, one per row of `X`, in the same order.

    Returns:
        `CandidatePrediction` with `probabilities` of shape (n, 3) whose
        columns follow the frozen `CLASS_ORDER`, aligned row-for-row with
        `fixture_ids`.

    Raises:
        FeatureContractViolation: the contract itself has drifted.
        RuntimeError: a contract column is missing from `X`.
        InferenceContractViolation: fixture_id/row-count mismatch, or the
            returned probability matrix has the wrong shape.
        MetricInputError: the probabilities are not valid.
    """
    # Imported lazily so this module remains importable without
    # scikit-learn, matching the existing project pattern.
    from .evaluate import validate_probabilities
    from .train import _reorder_proba

    verify_feature_contract()

    # Contract enforcement + structural exclusion of fixture_id: this
    # returns exactly the 80 contract columns in frozen order, so no
    # bookkeeping column can reach the estimator.
    design = select_candidate_features(X)

    fixture_ids = tuple(fixture_ids)
    if len(fixture_ids) != len(design):
        raise InferenceContractViolation(
            f"fixture_ids length {len(fixture_ids)} does not match feature rows {len(design)}; "
            "alignment must be exact and is never inferred."
        )

    encoded = preprocessor.transform(design)
    P = _reorder_proba(model, encoded)

    expected_shape = (len(design), len(CLASS_ORDER_NAMES))
    if P.shape != expected_shape:
        raise InferenceContractViolation(
            f"prediction matrix has shape {P.shape}, expected {expected_shape}"
        )
    validate_probabilities(P)

    return CandidatePrediction(
        fixture_ids=fixture_ids,
        probabilities=P,
        class_order=CLASS_ORDER_NAMES,
        n_rows=len(design),
    )
