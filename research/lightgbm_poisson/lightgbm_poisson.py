"""E9 — LightGBM Poisson vs V3 linear Poisson.

SINGLE-VARIABLE EXPERIMENT
--------------------------
    ARM A   87 features -> PoissonRegressor(alpha=1.0, max_iter=2000)
    ARM B   87 features -> LGBMRegressor(objective="poisson")

The ONLY conceptual difference is the learner. Both arms receive a
BYTE-IDENTICAL encoded matrix, identical targets, identical folds, and
the SAME adaptive score grid.

MISSING-DATA POLICY (approved)
------------------------------
Median imputation for BOTH arms, with medians computed from TRAINING
FOLD ROWS ONLY. Native LightGBM NaN handling is deliberately NOT used in
the primary experiment: it would change the missing-data treatment as
well as the learner, confounding the one variable E9 tests. Measured
missingness is 3.297% of cells across 83 of the 87 columns, so this is
a material choice, not a formality.

SHARED ADAPTIVE K — the subtle failure mode this module prevents
----------------------------------------------------------------
`src/models/poisson.py::_grid_size` derives K from
`max(lam_h.max(), lam_a.max())` ACROSS THE BATCH. If each arm called
`predict_poisson` independently they would derive DIFFERENT K whenever
their maximum lambdas differ, silently giving the two arms different
score grids and invalidating the comparison.

`shared_grid_k()` computes one K from the pooled maximum across BOTH
arms; `hda_from_lambdas()` then converts using that fixed K for both.

PREPROCESSING
-------------
`V3Preprocessor` reimplements `src/models/train.py::
LogisticRegressionPreprocessor` exactly -- median impute, standardise,
one-hot `competition_id` against fit-time categories -- in numpy only.
It is reimplemented rather than imported so that:
  * this research module has no dependency on production code,
  * the identical matrix handed to both arms is provably identical,
  * production code is not touched.
`test_lightgbm_poisson.py` asserts equivalence against the production
class when sklearn is available.

WHAT THIS MODULE DOES NOT DO
----------------------------
No market odds. No Dixon-Coles. No time decay. No quadratic Elo. No
online attack/defence. No calibration. No stacking. No feature changes.

DEPENDENCIES: numpy / pandas always; sklearn and lightgbm only inside
the fitting functions, so every contract, preprocessing, grid and
metric routine is importable and testable without them.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

CLASS_ORDER: tuple[str, str, str] = ("H", "D", "A")

#: Categorical column in the V3 contract (one-hot encoded by V3).
CATEGORICAL_COLUMN: str = "competition_id"

#: Tail tolerance, identical to src/models/poisson.py TAIL_TOL.
TAIL_TOL: float = 1e-15

#: V3 estimator config, verbatim from train_v3_candidate.py.
V3_ALPHA: float = 1.0
V3_MAX_ITER: int = 2000

#: Determinism / safety.
RANDOM_STATE: int = 42
MAX_PLAUSIBLE_LAMBDA: float = 15.0

#: Probability row-sum tolerance.
ROWSUM_TOL: float = 1e-9


class E9Error(RuntimeError):
    """Raised when an E9 safety gate fails."""


# ---------------------------------------------------------------------------
# Hyperparameter grid (frozen)
# ---------------------------------------------------------------------------

def load_grid(path: Path | None = None) -> dict:
    """Load the frozen pre-registered configuration set."""
    p = path or (HERE / "hyperparameter_grid.json")
    if not p.exists():
        raise E9Error(f"Frozen hyperparameter grid not found at {p}")
    grid = json.loads(p.read_text(encoding="utf-8"))
    if "configurations" not in grid or not grid["configurations"]:
        raise E9Error("Grid file contains no configurations")
    return grid


def config_params(cfg: dict, fixed: dict) -> dict:
    """Merge one configuration with the fixed settings into LGBM kwargs."""
    out = dict(fixed)
    for k in ("num_leaves", "learning_rate", "n_estimators",
              "min_child_samples", "subsample", "colsample_bytree",
              "reg_lambda"):
        if k not in cfg:
            raise E9Error(f"Configuration {cfg.get('id')} missing '{k}'")
        out[k] = cfg[k]
    # subsample only takes effect when bagging is enabled
    if out.get("subsample", 1.0) < 1.0:
        out["subsample_freq"] = 1
    return out


# ---------------------------------------------------------------------------
# Preprocessing — median impute + standardise + one-hot, training-only stats
# ---------------------------------------------------------------------------

@dataclass
class V3Preprocessor:
    """Faithful numpy reimplementation of LogisticRegressionPreprocessor.

    All statistics are fitted on training rows only. `transform` is a pure
    function of those frozen statistics.
    """
    numeric_columns: list[str] = field(default_factory=list)
    categories: list = field(default_factory=list)
    medians_: np.ndarray | None = None
    means_: np.ndarray | None = None
    scales_: np.ndarray | None = None
    fitted_: bool = False

    def fit(self, X_train: pd.DataFrame) -> "V3Preprocessor":
        self.numeric_columns = [c for c in X_train.columns
                                if c != CATEGORICAL_COLUMN]
        self.categories = sorted(
            X_train[CATEGORICAL_COLUMN].dropna().unique().tolist())

        num = X_train[self.numeric_columns].to_numpy(dtype=float)
        # SimpleImputer(strategy="median") ignores NaN when computing medians
        with np.errstate(all="ignore"):
            med = np.nanmedian(num, axis=0)
        # A fully-NaN column has no median; sklearn drops such columns, but
        # none exist here. Fail loudly rather than invent a value.
        if np.any(~np.isfinite(med)):
            bad = [self.numeric_columns[i]
                   for i in np.where(~np.isfinite(med))[0]]
            raise E9Error(f"Columns are entirely NaN in training fold: {bad}")
        self.medians_ = med

        imputed = np.where(np.isfinite(num), num, med)
        self.means_ = imputed.mean(axis=0)
        sd = imputed.std(axis=0)
        # StandardScaler maps zero-variance columns to 0 by using scale 1
        self.scales_ = np.where(sd > 0, sd, 1.0)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise E9Error("V3Preprocessor.fit() must be called before transform()")
        num = X[self.numeric_columns].to_numpy(dtype=float)
        imputed = np.where(np.isfinite(num), num, self.medians_)
        scaled = (imputed - self.means_) / self.scales_

        onehot = np.zeros((len(X), len(self.categories)), dtype=float)
        idx = {c: i for i, c in enumerate(self.categories)}
        for r, v in enumerate(X[CATEGORICAL_COLUMN].tolist()):
            j = idx.get(v)
            if j is not None:
                onehot[r, j] = 1.0

        out = np.hstack([scaled, onehot])
        if not np.all(np.isfinite(out)):
            raise E9Error("Non-finite values after preprocessing")
        return out

    @property
    def n_encoded_columns(self) -> int:
        return len(self.numeric_columns) + len(self.categories)


# ---------------------------------------------------------------------------
# Shared adaptive score grid
# ---------------------------------------------------------------------------

def grid_size(lam_max: float, tol: float = TAIL_TOL, cap: int = 20000) -> int:
    """Smallest K with Poisson(lam_max) upper tail < tol.

    Byte-for-byte the recursion used by src/models/poisson.py::_grid_size.
    Reproduced (not imported) so this module needs no production import;
    the test suite asserts equality against the production function.
    """
    k, term = 0, math.exp(-lam_max)
    cdf = term
    while cdf < 1.0 - tol and k < cap:
        k += 1
        term *= lam_max / k
        cdf += term
    return max(k, 1)


def shared_grid_k(*lambda_arrays: np.ndarray, tol: float = TAIL_TOL) -> int:
    """ONE K derived from the pooled maximum lambda across ALL arms.

    Pass every arm's lambda_home and lambda_away. The returned K is then
    used by both arms, so the score grid cannot silently differ.
    """
    mx = 0.0
    for a in lambda_arrays:
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            continue
        if not np.all(np.isfinite(a)):
            raise E9Error("Non-finite lambda supplied to shared_grid_k")
        # Every value must be a valid Poisson rate. Checking only the
        # maximum would let a corrupt negative rate pass unnoticed
        # whenever some other fixture happened to be positive.
        if np.any(a <= 0):
            n_bad = int(np.sum(a <= 0))
            raise E9Error(
                f"{n_bad} non-positive lambda value(s) supplied to "
                f"shared_grid_k (min {float(a.min()):.6g})")
        mx = max(mx, float(a.max()))
    if mx <= 0:
        raise E9Error("Pooled maximum lambda is non-positive")
    return grid_size(mx, tol)


def _pmf_grid(lam: np.ndarray, K: int) -> np.ndarray:
    k = np.arange(K + 1)
    log_fact = np.array([math.lgamma(i + 1) for i in k])
    l = np.asarray(lam, dtype=float)[:, None]
    return np.exp(-l + k * np.log(l) - log_fact)


def hda_from_lambdas(lam_h: np.ndarray, lam_a: np.ndarray,
                     K: int) -> np.ndarray:
    """Independent-Poisson 1X2 conversion at a FIXED K.

    Mirrors src/models/poisson.py::hda_tail_safe -- cumulative form for
    P(H) and P(D), P(A) by complement -- but takes K as an argument so
    both arms provably share one grid.
    """
    lam_h = np.asarray(lam_h, dtype=float)
    lam_a = np.asarray(lam_a, dtype=float)
    if np.any(lam_h <= 0) or np.any(lam_a <= 0):
        raise E9Error("Lambda values must be > 0")
    if not (np.all(np.isfinite(lam_h)) and np.all(np.isfinite(lam_a))):
        raise E9Error("Non-finite lambda values")

    ph, pa = _pmf_grid(lam_h, K), _pmf_grid(lam_a, K)
    Fa = np.cumsum(pa, axis=1)
    p_home = (ph[:, 1:] * Fa[:, :-1]).sum(axis=1)
    p_draw = (ph * pa).sum(axis=1)
    p_away = 1.0 - p_home - p_draw
    P = np.column_stack([p_home, p_draw, p_away])

    if np.any(P < -1e-12):
        raise E9Error("Negative probability produced")
    P = np.clip(P, 0.0, 1.0)
    s = P.sum(axis=1)
    if not np.all(np.abs(s - 1.0) <= ROWSUM_TOL):
        raise E9Error(
            f"Row sums deviate from 1 by up to {float(np.max(np.abs(s-1.0))):.3e}")
    return P


def validate_probs(P: np.ndarray, name: str = "probs") -> None:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise E9Error(f"{name}: expected (n, 3), got {P.shape}")
    if not np.all(np.isfinite(P)):
        raise E9Error(f"{name}: non-finite values")
    if np.any(P < 0) or np.any(P > 1):
        raise E9Error(f"{name}: outside [0, 1]")
    if not np.allclose(P.sum(axis=1), 1.0, atol=ROWSUM_TOL):
        raise E9Error(f"{name}: rows do not sum to 1")


# ---------------------------------------------------------------------------
# Fitting — the ONLY place sklearn / lightgbm are imported
# ---------------------------------------------------------------------------

def fit_v3_arm(X_tr_enc: np.ndarray, hg: np.ndarray, ag: np.ndarray,
               X_ev_enc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ARM A — V3's linear PoissonRegressor. Returns (lam_h, lam_a)."""
    from sklearn.linear_model import PoissonRegressor
    mh = PoissonRegressor(alpha=V3_ALPHA, max_iter=V3_MAX_ITER).fit(X_tr_enc, hg)
    ma = PoissonRegressor(alpha=V3_ALPHA, max_iter=V3_MAX_ITER).fit(X_tr_enc, ag)
    return mh.predict(X_ev_enc), ma.predict(X_ev_enc)


def fit_lgbm_arm(X_tr_enc: np.ndarray, hg: np.ndarray, ag: np.ndarray,
                 X_ev_enc: np.ndarray, params: dict,
                 return_models: bool = False):
    """ARM B — LightGBM Poisson. Returns (lam_h, lam_a[, models])."""
    from lightgbm import LGBMRegressor
    mh = LGBMRegressor(**params).fit(X_tr_enc, hg)
    ma = LGBMRegressor(**params).fit(X_tr_enc, ag)
    lam_h = mh.predict(X_ev_enc)
    lam_a = ma.predict(X_ev_enc)
    if return_models:
        return lam_h, lam_a, (mh, ma)
    return lam_h, lam_a


def lightgbm_version() -> str:
    import lightgbm
    return lightgbm.__version__


def sklearn_version() -> str:
    import sklearn
    return sklearn.__version__


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _onehot(y: np.ndarray) -> np.ndarray:
    o = np.zeros((len(y), 3), dtype=float)
    for i, c in enumerate(CLASS_ORDER):
        o[:, i] = (y == c)
    return o


def log_loss(y, P) -> float:
    Pc = np.clip(P, 1e-15, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(_onehot(y) * np.log(Pc), axis=1)))


def brier(y, P) -> float:
    return float(np.mean(np.sum((P - _onehot(y)) ** 2, axis=1)))


def rps(y, P) -> float:
    cp, co = np.cumsum(P, axis=1), np.cumsum(_onehot(y), axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


def accuracy(y, P) -> float:
    return float(np.mean(np.array([CLASS_ORDER[i]
                                   for i in np.asarray(P).argmax(1)]) == y))


def draw_recall(y, P) -> float:
    pred = np.array([CLASS_ORDER[i] for i in np.asarray(P).argmax(1)])
    m = (y == "D")
    return float(np.mean(pred[m] == "D")) if m.sum() else float("nan")


def reliability_bins(y, P, n_bins: int = 10) -> list[dict]:
    oh, pf = _onehot(y).ravel(), np.asarray(P, float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (pf >= lo) & (pf <= hi) if i == n_bins - 1 else (pf >= lo) & (pf < hi)
        n = int(m.sum())
        if not n:
            out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3),
                        "n": 0, "mean_predicted": None,
                        "observed_freq": None, "gap": None})
            continue
        mp, ob = float(pf[m].mean()), float(oh[m].mean())
        out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": n,
                    "mean_predicted": round(mp, 6),
                    "observed_freq": round(ob, 6), "gap": round(ob - mp, 6)})
    return out


def ece(y, P, n_bins: int = 10) -> float:
    b = reliability_bins(y, P, n_bins)
    tot = sum(x["n"] for x in b)
    return float(sum(x["n"] / tot * abs(x["gap"]) for x in b if x["n"])) if tot else float("nan")


def all_metrics(y, P, lam_h=None, lam_a=None, hg=None, ag=None) -> dict:
    m = {"n": int(len(y)),
         "log_loss": round(log_loss(y, P), 6),
         "brier": round(brier(y, P), 6),
         "rps": round(rps(y, P), 6),
         "accuracy": round(accuracy(y, P), 6),
         "draw_recall": round(draw_recall(y, P), 6),
         "draw_predicted_rate": round(float(np.mean(np.asarray(P).argmax(1) == 1)), 6),
         "ece": round(ece(y, P), 6)}
    if lam_h is not None and hg is not None:
        m["home_goal_mae"] = round(float(np.mean(np.abs(lam_h - hg))), 6)
        m["away_goal_mae"] = round(float(np.mean(np.abs(lam_a - ag))), 6)
    return m


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def lambda_diagnostics(lam_h, lam_a) -> dict:
    lh, la = np.asarray(lam_h, float), np.asarray(lam_a, float)
    d = {}
    for nm, a in (("lambda_home", lh), ("lambda_away", la)):
        d[nm] = {
            "min": round(float(a.min()), 6), "max": round(float(a.max()), 6),
            "mean": round(float(a.mean()), 6),
            "p01": round(float(np.percentile(a, 1)), 6),
            "p50": round(float(np.percentile(a, 50)), 6),
            "p99": round(float(np.percentile(a, 99)), 6),
        }
    d["all_finite"] = bool(np.all(np.isfinite(lh)) and np.all(np.isfinite(la)))
    d["all_positive"] = bool(np.all(lh > 0) and np.all(la > 0))
    d["within_plausible"] = bool(lh.max() <= MAX_PLAUSIBLE_LAMBDA
                                 and la.max() <= MAX_PLAUSIBLE_LAMBDA)
    d["any_nan"] = bool(np.any(np.isnan(lh)) or np.any(np.isnan(la)))
    d["any_inf"] = bool(np.any(np.isinf(lh)) or np.any(np.isinf(la)))
    d["passed"] = bool(d["all_finite"] and d["all_positive"]
                       and d["within_plausible"])
    d["clipping_applied"] = False   # E9 never clips; see spec section 19
    return d


def prediction_change(P_a: np.ndarray, P_b: np.ndarray,
                      lam_h_a, lam_a_a, lam_h_b, lam_a_b,
                      material: float = 0.01) -> dict:
    Pa, Pb = np.asarray(P_a, float), np.asarray(P_b, float)
    dP = np.abs(Pb - Pa)
    return {
        "mean_abs_prob_change": round(float(dP.mean()), 8),
        "max_prob_change": round(float(dP.max()), 8),
        "pct_materially_changed": round(
            100 * float(np.mean(dP.max(axis=1) > material)), 4),
        "pct_top_class_changed": round(
            100 * float(np.mean(Pa.argmax(1) != Pb.argmax(1))), 4),
        "mean_abs_lambda_home_change": round(
            float(np.mean(np.abs(np.asarray(lam_h_b) - np.asarray(lam_h_a)))), 8),
        "mean_abs_lambda_away_change": round(
            float(np.mean(np.abs(np.asarray(lam_a_b) - np.asarray(lam_a_a)))), 8),
        "material_threshold": material,
    }


def feature_importance(models, feature_names: list[str],
                       top_n: int = 20) -> dict:
    """Gain and split importance for the home and away LightGBM models."""
    mh, ma = models
    out = {}
    for nm, mdl in (("home", mh), ("away", ma)):
        gain = np.asarray(mdl.booster_.feature_importance(importance_type="gain"),
                          dtype=float)
        split = np.asarray(mdl.booster_.feature_importance(importance_type="split"),
                           dtype=float)
        order = np.argsort(gain)[::-1][:top_n]
        tot = gain.sum() if gain.sum() > 0 else 1.0
        out[nm] = [
            {"feature": feature_names[i] if i < len(feature_names) else f"col_{i}",
             "gain": round(float(gain[i]), 4),
             "gain_pct": round(100 * float(gain[i]) / tot, 4),
             "split": int(split[i])}
            for i in order
        ]
        out[f"{nm}_n_trees"] = int(mdl.booster_.num_trees())
    return out


def overfitting_diagnostics(y_tr, P_tr, y_va, P_va, y_te, P_te) -> dict:
    """Train / inner-validation / outer-test log loss and the gaps."""
    tr = log_loss(y_tr, P_tr)
    va = log_loss(y_va, P_va) if y_va is not None and len(y_va) else None
    te = log_loss(y_te, P_te)
    return {
        "train_log_loss": round(tr, 6),
        "inner_val_log_loss": round(va, 6) if va is not None else None,
        "test_log_loss": round(te, 6),
        "train_test_gap": round(te - tr, 6),
        "val_test_gap": round(te - va, 6) if va is not None else None,
    }


# ---------------------------------------------------------------------------
# Nested chronological selection — TRAINING PERIOD ONLY
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InnerSplit:
    train_seasons: tuple[str, ...]
    val_season: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def build_inner_splits(seasons: np.ndarray,
                       train_seasons: tuple[str, ...]) -> list[InnerSplit]:
    """Nested chronological splits inside the OUTER TRAINING seasons.

    Receives training seasons only; the outer test season is not a
    parameter and therefore cannot enter.
    """
    ordered = tuple(train_seasons)
    out: list[InnerSplit] = []
    for j in range(1, len(ordered)):
        tr, va = ordered[:j], ordered[j]
        ti = np.where(np.isin(seasons, tr))[0]
        vi = np.where(seasons == va)[0]
        if len(ti) and len(vi):
            out.append(InnerSplit(tr, va, ti, vi))
    return out


@dataclass(frozen=True)
class ConfigSelection:
    config_id: int
    params: dict
    mean_inner_log_loss: float
    curve: list[dict]
    n_inner_splits: int
    method: str


def select_config(inner_splits: list[InnerSplit], fit_eval,
                  configs: list[dict], fixed: dict) -> ConfigSelection:
    """Choose a configuration by mean inner-validation log loss.

    CAUSALITY: only inner splits built from outer-training seasons are
    supplied. No outer-test parameter exists in this signature.

    fit_eval: callable(split, params) -> inner-validation log loss.
    """
    if not inner_splits:
        raise E9Error("No inner splits available for configuration selection")
    if not configs:
        raise E9Error("No configurations supplied")

    curve, best, best_cfg = [], float("inf"), None
    for cfg in configs:
        params = config_params(cfg, fixed)
        per = [float(fit_eval(sp, params)) for sp in inner_splits]
        mean_ll = float(np.mean(per))
        curve.append({"config_id": cfg["id"], "note": cfg.get("note", ""),
                      "mean_inner_log_loss": round(mean_ll, 8),
                      "per_split_log_loss": [round(v, 8) for v in per]})
        if mean_ll < best - 1e-12:
            best, best_cfg = mean_ll, (cfg, params)

    cfg, params = best_cfg
    return ConfigSelection(config_id=cfg["id"], params=params,
                           mean_inner_log_loss=best, curve=curve,
                           n_inner_splits=len(inner_splits),
                           method="nested_inner_validation")


__all__ = [
    "CLASS_ORDER", "CATEGORICAL_COLUMN", "TAIL_TOL", "V3_ALPHA",
    "V3_MAX_ITER", "RANDOM_STATE", "MAX_PLAUSIBLE_LAMBDA", "E9Error",
    "load_grid", "config_params", "V3Preprocessor",
    "grid_size", "shared_grid_k", "hda_from_lambdas", "validate_probs",
    "fit_v3_arm", "fit_lgbm_arm", "lightgbm_version", "sklearn_version",
    "log_loss", "brier", "rps", "accuracy", "draw_recall",
    "reliability_bins", "ece", "all_metrics",
    "lambda_diagnostics", "prediction_change", "feature_importance",
    "overfitting_diagnostics",
    "InnerSplit", "build_inner_splits", "ConfigSelection", "select_config",
]
