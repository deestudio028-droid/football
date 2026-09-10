"""Real scikit-learn candidate models: LogisticRegression and
HistGradientBoostingClassifier, evaluated across the 3 approved
walk-forward folds only (never 2025/26 -- see splits.py).

Requires scikit-learn to be installed to actually run. This module
does not implement any fallback/substitute model -- if scikit-learn is
unavailable, importing this module raises ImportError immediately
(surfaced clearly, not masked behind a confusing downstream error) so
callers know exactly why nothing ran, and are pointed at the real fix
(install scikit-learn) rather than tempted to patch around it.

Preprocessing is fit exclusively on each fold's training partition and
applied unchanged to that fold's validation partition -- there is no
code path in this module that can fit a scaler/imputer/encoder on
anything other than X_train. This is verified mechanically, not just by
convention, in tests/test_model_training.py (fold-swap tests that prove
validation-fold values cannot influence the fitted parameters).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover -- exercised only when sklearn is absent
    raise ImportError(
        "scikit-learn is required to use src/models/train.py and is not installed "
        "in this Python environment. Install it (e.g. `pip install scikit-learn`) "
        "in the environment that will actually run model training -- do not work "
        "around this by substituting a different implementation here."
    ) from exc

from .baselines import CLASS_ORDER
from .config import RECOMMENDED_CONTEXT_FEATURE

MODEL_LOGISTIC_REGRESSION = "logistic_regression"
MODEL_HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"


# ---------------------------------------------------------------------
# competition_id categorical encoding for HistGradientBoostingClassifier.
#
# The real competition_id values in this dataset are {200, 419, 423,
# 477, 499} -- arbitrary-magnitude league IDs, not a contiguous 0..N-1
# range. scikit-learn's HistGradientBoostingClassifier requires
# categorical columns to be encoded as integers in [0, n_categories);
# passing the raw IDs directly violates that contract and is a
# confirmed contributor to the binning-stage crash documented in
# docs/PHASE3_STEP3_DIAGNOSTIC_REPORT.md. This encoder fixes that by
# fitting a deterministic train-only mapping to a dense code range.
#
# Unseen categories at transform time (should not happen in practice --
# all 5 leagues appear in every approved walk-forward fold -- but
# handled defensively) map to -1. scikit-learn's own documented
# convention for HistGradientBoostingClassifier is that negative values
# in a categorical column are treated as missing, not as a guessed
# valid category -- so an unseen category safely falls back to "no
# information" rather than silently colliding with a real, different
# league's code.
# ---------------------------------------------------------------------
@dataclass
class CompetitionCodeEncoder:
    categories_: list[int] = field(default_factory=list)
    _code_by_category: dict[int, int] = field(default_factory=dict)
    _fitted: bool = False

    def fit(self, competition_ids: pd.Series) -> "CompetitionCodeEncoder":
        # Sorted, not insertion-order, so the mapping is reproducible
        # regardless of row order within the training fold.
        self.categories_ = sorted(competition_ids.dropna().unique().tolist())
        self._code_by_category = {cat: i for i, cat in enumerate(self.categories_)}
        self._fitted = True
        return self

    def transform(self, competition_ids: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("CompetitionCodeEncoder.fit() must be called before transform().")
        return np.array(
            [self._code_by_category.get(v, -1) for v in competition_ids.tolist()],
            dtype=np.int64,
        )


# ---------------------------------------------------------------------
# LogisticRegression preprocessing: median-impute + standardize every
# numeric column, one-hot encode competition_id (a categorical league
# ID with no ordinal meaning -- feeding it to a linear model as a raw
# integer would wrongly imply "league 477 is between leagues 423 and
# 499," which is meaningless). All statistics are fit on X_train only.
# ---------------------------------------------------------------------
@dataclass
class LogisticRegressionPreprocessor:
    numeric_columns: list[str] = field(default_factory=list)
    competition_categories: list[Any] = field(default_factory=list)
    _imputer: SimpleImputer | None = None
    _scaler: StandardScaler | None = None
    _fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "LogisticRegressionPreprocessor":
        self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]
        self.competition_categories = sorted(X_train[RECOMMENDED_CONTEXT_FEATURE].dropna().unique().tolist())

        self._imputer = SimpleImputer(strategy="median")
        imputed = self._imputer.fit_transform(X_train[self.numeric_columns])

        self._scaler = StandardScaler()
        self._scaler.fit(imputed)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LogisticRegressionPreprocessor.fit() must be called before transform().")
        imputed = self._imputer.transform(X[self.numeric_columns])
        scaled = self._scaler.transform(imputed)

        # One-hot encode competition_id against the categories seen at fit
        # time. A category never seen in training (shouldn't happen given
        # all 5 leagues appear in every fold, but handled defensively)
        # gets an all-zero row rather than raising or inventing a column.
        onehot = np.zeros((len(X), len(self.competition_categories)))
        cat_index = {c: i for i, c in enumerate(self.competition_categories)}
        for row_i, val in enumerate(X[RECOMMENDED_CONTEXT_FEATURE].tolist()):
            col = cat_index.get(val)
            if col is not None:
                onehot[row_i, col] = 1.0

        return np.hstack([scaled, onehot])


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
) -> tuple[LogisticRegression, LogisticRegressionPreprocessor, np.ndarray]:
    """Fits preprocessing + LogisticRegression on (X_train, y_train) only,
    returns validation-fold probabilities in CLASS_ORDER column order.
    """
    preprocessor = LogisticRegressionPreprocessor().fit(X_train)
    X_train_enc = preprocessor.transform(X_train)
    X_val_enc = preprocessor.transform(X_val)

    model = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
    model.fit(X_train_enc, y_train)

    P_val = _reorder_proba(model, X_val_enc)
    return model, preprocessor, P_val


# ---------------------------------------------------------------------
# HistGradientBoostingClassifier: uses its native missing-value handling
# (no imputation) and native categorical-feature support for
# competition_id (no one-hot encoding, no scaling -- none of that is
# meaningful for a tree-based model, and imputing away real "coverage
# gap" information the way LogisticRegression's preprocessing must
# would only throw away signal a tree model can use directly).
# competition_id itself is re-coded to a dense [0, n_categories) integer
# range via CompetitionCodeEncoder (fit on X_train only) before being
# handed to the model -- see that class's docstring for why the raw IDs
# can't be passed directly.
# ---------------------------------------------------------------------
def train_hist_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
) -> tuple[HistGradientBoostingClassifier, CompetitionCodeEncoder | None, np.ndarray]:
    has_competition_id = RECOMMENDED_CONTEXT_FEATURE in X_train.columns

    encoder: CompetitionCodeEncoder | None = None
    X_train_enc = X_train
    X_val_enc = X_val
    if has_competition_id:
        encoder = CompetitionCodeEncoder().fit(X_train[RECOMMENDED_CONTEXT_FEATURE])
        X_train_enc = X_train.copy()
        X_train_enc[RECOMMENDED_CONTEXT_FEATURE] = encoder.transform(X_train[RECOMMENDED_CONTEXT_FEATURE])
        X_val_enc = X_val.copy()
        X_val_enc[RECOMMENDED_CONTEXT_FEATURE] = encoder.transform(X_val[RECOMMENDED_CONTEXT_FEATURE])

    categorical_features = [RECOMMENDED_CONTEXT_FEATURE] if has_competition_id else None
    model = HistGradientBoostingClassifier(
        random_state=0,
        categorical_features=categorical_features,
    )
    model.fit(X_train_enc, y_train)
    P_val = _reorder_proba(model, X_val_enc)
    return model, encoder, P_val


def _reorder_proba(model, X_encoded) -> np.ndarray:
    """sklearn orders predict_proba's columns by `model.classes_`
    (alphabetical for string labels: A, D, H), not necessarily our
    CLASS_ORDER (H, D, A). Reorder explicitly rather than assuming --
    silently trusting column order here would be a subtle, easy-to-miss
    correctness bug in every downstream metric.
    """
    raw_proba = model.predict_proba(X_encoded)
    class_to_col = {c: i for i, c in enumerate(model.classes_)}
    missing = [c for c in CLASS_ORDER if c not in class_to_col]
    if missing:
        raise RuntimeError(
            f"Model was not fit with all of {CLASS_ORDER} present in training labels; "
            f"missing: {missing}. Cannot safely reorder predict_proba output."
        )
    reorder = [class_to_col[c] for c in CLASS_ORDER]
    return raw_proba[:, reorder]
