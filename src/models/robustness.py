"""Phase 4B: LogisticRegression regularization-sensitivity study (design layer).

STATUS: DESIGN FROZEN, TRAINING NOT STARTED.

Question this phase exists to answer: does Model B's Phase 4A advantage
over Model A (mean validation log loss -0.004634278243) survive
reasonable changes to LogisticRegression's regularization strength, or
is it an artifact of the arbitrary default C=1.0?

This is NOT a V2 selection exercise. No production model is chosen here.

--------------------------------------------------------------------
IMPORTANT DESIGN CONSTRAINT (reported, not silently worked around)
--------------------------------------------------------------------
`train.train_logistic_regression()` hardcodes
`LogisticRegression(max_iter=2000, C=1.0, random_state=0)` and exposes
NO `C` parameter. Varying C through it would require editing V1
production code, which is forbidden.

Resolution used here, chosen specifically to avoid modifying V1:
`train_logistic_regression_with_C()` below REUSES the unmodified V1
preprocessing (`train.LogisticRegressionPreprocessor`) and the
unmodified V1 class-order handling (`train._reorder_proba`), and
constructs the estimator with kwargs identical to V1's except for C.
Nothing about the architecture, preprocessing, imputation, scaling,
competition_id handling, solver, or random_state changes.

Two mechanisms guard that claim:
  1. `V1_LOGREG_BASE_KWARGS` is checked against the kwargs actually
     present in `train.py` by AST inspection, in
     tests/test_model_robustness.py (runs without scikit-learn).
  2. An equivalence test asserts that at C=1.0 this path reproduces
     Phase 4A's recorded per-fold Model A / Model B metrics exactly
     (scikit-learn required; runs on the project machine).

If either guard ever fails, the sensitivity study is not comparable to
Phase 4A and must stop rather than be reinterpreted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ablation import MODEL_A, MODEL_A_COLUMNS, MODEL_B, MODEL_B_COLUMNS, select_feature_columns

# ---------------------------------------------------------------------
# FROZEN sensitivity grid. Declared here, before any training, and never
# expanded at runtime. Deliberately a small, pre-declared, symmetric-ish
# sweep spanning four orders of magnitude around V1's C=1.0 -- NOT an
# automated hyperparameter search, and not a selection procedure.
# ---------------------------------------------------------------------
C_GRID: tuple[float, ...] = (0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0)

# V1's baseline setting, present in the grid so the Phase 4A result sits
# inside the swept region rather than at its edge.
V1_C_VALUE: float = 1.0

# Every LogisticRegression kwarg V1 uses EXCEPT C. Verified against
# train.py by AST inspection in the test suite -- if V1's configuration
# ever changes, that test fails loudly instead of this study silently
# diverging from the model it claims to be probing.
V1_LOGREG_BASE_KWARGS: dict = {"max_iter": 2000, "random_state": 0}

# The two Phase 4A tiers under study, imported unchanged from ablation.py.
# Models C and D are deliberately excluded: they require xG, which Phase
# 4A established cannot be evaluated across the 3-fold protocol.
PHASE4B_MODELS: dict[str, tuple[str, ...]] = {
    MODEL_A: MODEL_A_COLUMNS,
    MODEL_B: MODEL_B_COLUMNS,
}

METRICS: tuple[str, ...] = ("log_loss", "brier", "macro_f1", "balanced_accuracy", "accuracy")

# Phase 4A's frozen results, quoted for equivalence-checking at C=1.0
# only. Never used to select, tune, or gate anything.
PHASE4A_C1_MEAN_LOG_LOSS: dict[str, float] = {
    MODEL_A: 1.004013383939449,
    MODEL_B: 0.9993791056968738,
}


def build_logistic_regression(C: float):
    """Constructs the estimator with V1's exact kwargs plus the swept C.

    Kept as a single choke point so there is exactly one place where a
    Phase 4B estimator can be created, and so tests can assert its
    parameters against V1's without duplicating literals.
    """
    # Grid membership is validated BEFORE importing scikit-learn: the
    # frozen-grid guard is pure logic and must reject an off-grid C in
    # any environment, including one without scikit-learn installed.
    if C not in C_GRID:
        raise ValueError(
            f"C={C!r} is not in the frozen Phase 4B grid {C_GRID}. The grid is fixed before "
            "training by design; widening it at runtime would turn a pre-declared sensitivity "
            "check into an unregistered hyperparameter search."
        )

    from sklearn.linear_model import LogisticRegression  # lazy: sklearn only needed to train

    return LogisticRegression(C=C, **V1_LOGREG_BASE_KWARGS)


def train_logistic_regression_with_C(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, C: float
) -> np.ndarray:
    """V1's training path with C as the single varied parameter.

    Reuses `train.LogisticRegressionPreprocessor` (fit on X_train ONLY)
    and `train._reorder_proba` unmodified, so preprocessing, imputation,
    scaling, competition_id one-hot encoding and CLASS_ORDER handling
    are bit-for-bit V1 behaviour. Returns validation-fold probabilities
    in CLASS_ORDER column order.
    """
    from . import train as train_module

    preprocessor = train_module.LogisticRegressionPreprocessor().fit(X_train)
    X_train_enc = preprocessor.transform(X_train)
    X_val_enc = preprocessor.transform(X_val)

    model = build_logistic_regression(C)
    model.fit(X_train_enc, y_train)
    return train_module._reorder_proba(model, X_val_enc)


def select_model_features(X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Restricts X to the frozen Phase 4A columns for `model_name`,
    failing loudly if any is absent (never substituting or dropping).
    """
    if model_name not in PHASE4B_MODELS:
        raise ValueError(f"Unknown Phase 4B model {model_name!r}; expected one of {sorted(PHASE4B_MODELS)}")
    return select_feature_columns(X, PHASE4B_MODELS[model_name], group_name=model_name)
