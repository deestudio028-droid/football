"""E6 — Causal online Attack/Defense states.

APPROVED FORMULATION (Phase 1 audit, E6_AUDIT_REPORT.md)
--------------------------------------------------------
Each team carries two log-scale states, initialised at 0.0 and
persisting across seasons with no reset:

    A[team]  attack
    D[team]  defense

Pre-match rates for a fixture (home h, away a):

    lambda_home = mu_home * exp(A[h] - D[a])
    lambda_away = mu_away * exp(A[a] - D[h])

mu_home / mu_away are league baseline goal rates estimated from
TRAINING FIXTURES ONLY.

After the outcome is observed:

    error_home = goals_home - lambda_home
    error_away = goals_away - lambda_away

    A[h] += lr * error_home        D[a] -= lr * error_home
    A[a] += lr * error_away        D[h] -= lr * error_away

These are the exact Poisson log-likelihood score functions for a log
link, so the update is stochastic gradient ascent on the same
likelihood V3's goal models already optimise. It is opponent-adjusted
by construction: the error depends on the opponent's current state.

CAUSAL ORDER — the invariant that matters
-----------------------------------------
Two passes per distinct timestamp, mirroring the E1 Elo engine:

    PASS 1: read pre-match states for every fixture at T
    PASS 2: apply updates from outcomes at T

A fixture's own outcome therefore cannot reach its own features, and
simultaneous fixtures cannot contaminate one another.

FEATURE CONTRACT
----------------
    V3 = 87 columns
    E6 = 91 columns = V3[0:87] + (A_home, D_home, A_away, D_away)

Four raw states, not the differential: the audit measured
corr(net differential, elo_diff) = +0.9857, so a differential would be
near-duplication of Elo. The incremental information lives in the
separation of attack from defense.

NUMERICAL SAFETY
----------------
States are clipped to +-STATE_CLIP (1.5) on the log scale, the value
used and validated in the Phase 1 audit. The exponent argument is
clipped to +-EXP_CLIP (3.0) before exp() to prevent overflow.

DEPENDENCIES: numpy / pandas only. No scipy, no sklearn.
DETERMINISM: pure functions of ordered history. No RNG anywhere.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Initial state for every team, including unseen/promoted teams.
INIT_ATTACK: float = 0.0
INIT_DEFENSE: float = 0.0

#: Log-scale state clip (audit-approved; do not change silently).
STATE_CLIP: float = 1.5

#: Exponent argument clip, guards exp() overflow.
EXP_CLIP: float = 3.0

#: The four research columns E6 appends to the V3 contract.
AD_COLUMNS: tuple[str, ...] = ("A_home", "D_home", "A_away", "D_away")

#: Pre-registered learning-rate grid. DO NOT EXPAND.
LR_GRID: tuple[float, ...] = (0.005, 0.01, 0.02, 0.035, 0.05)

#: Completed-match statuses that trigger a state update.
COMPLETED_STATUS: frozenset[str] = frozenset({"FT", "AWARDED"})

CLASS_ORDER: tuple[str, str, str] = ("H", "D", "A")


class OnlineADError(ValueError):
    """Raised when an online A/D safety gate fails."""


@dataclass(frozen=True)
class BaselineRates:
    """League baseline goal rates fitted on TRAINING fixtures only."""
    mu_home: float
    mu_away: float
    n_train: int

    def as_dict(self) -> dict:
        return {"mu_home": self.mu_home, "mu_away": self.mu_away,
                "n_train": self.n_train}


def fit_baseline_rates(home_goals_train: np.ndarray,
                       away_goals_train: np.ndarray) -> BaselineRates:
    """Estimate mu_home / mu_away from training fixtures only.

    The signature accepts training goal arrays only — no test data and
    no fixture identifiers — so no test outcome can reach it.
    """
    h = np.asarray(home_goals_train, dtype=float)
    a = np.asarray(away_goals_train, dtype=float)
    if len(h) == 0 or len(a) == 0:
        raise OnlineADError("Cannot fit baseline rates on an empty training set")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(a))):
        raise OnlineADError("Non-finite goals in training set")
    mu_h, mu_a = float(h.mean()), float(a.mean())
    if mu_h <= 0 or mu_a <= 0:
        raise OnlineADError(f"Non-positive baseline rate: {mu_h}, {mu_a}")
    return BaselineRates(mu_home=mu_h, mu_away=mu_a, n_train=int(len(h)))


def compute_ad_states(
    fixtures: pd.DataFrame,
    lr: float,
    baseline: BaselineRates,
    enabled: bool = True,
) -> pd.DataFrame:
    """Compute causal pre-match A/D states for every fixture.

    Args:
        fixtures: must contain fixture_id, unix, home_id, away_id,
            home_goals, away_goals, status. Order is irrelevant — the
            function sorts internally — but the returned frame is keyed
            on fixture_id.
        lr: learning rate. lr = 0.0 freezes every state at its initial
            value (states stay identically 0.0).
        baseline: mu_home / mu_away fitted on training rows only.
        enabled: False returns all-zero states — the zero-online
            control. The full two-pass walk still executes so timing
            behaviour is identical; only the output is neutralised.

    Returns:
        DataFrame with fixture_id + AD_COLUMNS, one row per fixture.
    """
    if lr < 0:
        raise OnlineADError(f"lr must be >= 0, got {lr}")

    required = {"fixture_id", "unix", "home_id", "away_id",
                "home_goals", "away_goals", "status"}
    missing = required - set(fixtures.columns)
    if missing:
        raise OnlineADError(f"fixtures missing columns: {sorted(missing)}")

    df = fixtures.sort_values(["unix", "fixture_id"]).reset_index(drop=True)

    A: dict = defaultdict(lambda: INIT_ATTACK)
    D: dict = defaultdict(lambda: INIT_DEFENSE)

    fid = df["fixture_id"].to_numpy()
    unix = df["unix"].to_numpy(dtype=float)
    hid = df["home_id"].to_numpy()
    aid = df["away_id"].to_numpy()
    hg = df["home_goals"].to_numpy(dtype=float)
    ag = df["away_goals"].to_numpy(dtype=float)
    st = df["status"].to_numpy()

    n = len(df)
    out = np.zeros((n, 4), dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and unix[j] == unix[i]:
            j += 1

        # ---- PASS 1: read pre-match states -------------------------
        for k in range(i, j):
            out[k, 0] = A[hid[k]]
            out[k, 1] = D[hid[k]]
            out[k, 2] = A[aid[k]]
            out[k, 3] = D[aid[k]]

        # ---- PASS 2: update from observed outcomes -----------------
        if lr > 0.0:
            for k in range(i, j):
                if st[k] not in COMPLETED_STATUS:
                    continue
                if not (np.isfinite(hg[k]) and np.isfinite(ag[k])):
                    continue
                h, a = hid[k], aid[k]
                lam_h = baseline.mu_home * np.exp(
                    np.clip(A[h] - D[a], -EXP_CLIP, EXP_CLIP))
                lam_a = baseline.mu_away * np.exp(
                    np.clip(A[a] - D[h], -EXP_CLIP, EXP_CLIP))
                e_h = hg[k] - lam_h
                e_a = ag[k] - lam_a
                A[h] = float(np.clip(A[h] + lr * e_h, -STATE_CLIP, STATE_CLIP))
                D[a] = float(np.clip(D[a] - lr * e_h, -STATE_CLIP, STATE_CLIP))
                A[a] = float(np.clip(A[a] + lr * e_a, -STATE_CLIP, STATE_CLIP))
                D[h] = float(np.clip(D[h] - lr * e_a, -STATE_CLIP, STATE_CLIP))
        i = j

    if not np.all(np.isfinite(out)):
        raise OnlineADError("Non-finite A/D states produced")
    if np.any(np.abs(out) > STATE_CLIP + 1e-9):
        raise OnlineADError("A/D state escaped the clip bound")

    if not enabled:
        out = np.zeros_like(out)

    return pd.DataFrame({
        "fixture_id": fid,
        "A_home": out[:, 0], "D_home": out[:, 1],
        "A_away": out[:, 2], "D_away": out[:, 3],
    })


def e6_feature_columns(v3_columns: tuple[str, ...]) -> tuple[str, ...]:
    """The 91-column E6 contract: V3's 87 columns then the four states."""
    overlap = set(v3_columns) & set(AD_COLUMNS)
    if overlap:
        raise OnlineADError(f"V3 contract already contains {sorted(overlap)}")
    if len(v3_columns) != 87:
        raise OnlineADError(f"Expected 87 V3 columns, got {len(v3_columns)}")
    return tuple(v3_columns) + AD_COLUMNS


def build_design(X_v3: pd.DataFrame, v3_columns: tuple[str, ...],
                 states: pd.DataFrame, fixture_ids: np.ndarray) -> pd.DataFrame:
    """Return the 91-column E6 design matrix.

    The 87 V3 columns pass through untouched and in order; the four
    state columns are joined on fixture_id and appended.
    """
    if list(X_v3.columns) != list(v3_columns):
        raise OnlineADError("Input columns do not match the V3 contract order")
    smap = states.set_index("fixture_id")
    missing = [f for f in fixture_ids if f not in smap.index]
    if missing:
        raise OnlineADError(f"{len(missing)} fixtures have no A/D state")
    out = X_v3.copy()
    for c in AD_COLUMNS:
        out[c] = smap.loc[fixture_ids, c].to_numpy()
    expected = e6_feature_columns(v3_columns)
    if tuple(out.columns) != expected:
        raise OnlineADError("E6 design column order is wrong")
    return out


# ---------------------------------------------------------------------------
# Nested chronological lr selection — TRAINING PERIOD ONLY
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InnerSplit:
    """A temporal split strictly inside the outer training period."""
    train_seasons: tuple[str, ...]
    val_season: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def build_inner_splits(seasons: np.ndarray,
                       train_seasons: tuple[str, ...]) -> list[InnerSplit]:
    """Nested temporal splits inside the outer training seasons.

    Receives the outer TRAINING seasons only — the outer test season is
    not a parameter, so it cannot enter.
    """
    ordered = tuple(train_seasons)
    out: list[InnerSplit] = []
    for j in range(1, len(ordered)):
        tr, va = ordered[:j], ordered[j]
        tr_idx = np.where(np.isin(seasons, tr))[0]
        va_idx = np.where(seasons == va)[0]
        if len(tr_idx) and len(va_idx):
            out.append(InnerSplit(tr, va, tr_idx, va_idx))
    return out


@dataclass(frozen=True)
class LrSelection:
    """Outcome of a training-only learning-rate search."""
    lr: float
    mean_inner_log_loss: float
    curve: list[dict]
    n_inner_splits: int
    grid: tuple[float, ...]


def select_lr(inner_splits: list[InnerSplit], fit_predict,
              grid: tuple[float, ...] = LR_GRID) -> LrSelection:
    """Select lr by mean inner-validation log loss.

    CAUSALITY: receives only inner splits built from the outer training
    seasons. No outer-test parameter exists in the signature.

    fit_predict: callable(split: InnerSplit, lr: float) -> float
        returns inner-validation log loss.
    """
    if not inner_splits:
        raise OnlineADError("No inner splits available for lr selection")

    curve, best_lr, best = [], None, float("inf")
    for lr in grid:
        per = [float(fit_predict(sp, lr)) for sp in inner_splits]
        mean_ll = float(np.mean(per))
        curve.append({"lr": lr, "mean_inner_log_loss": round(mean_ll, 8),
                      "per_split_log_loss": [round(v, 8) for v in per]})
        if mean_ll < best - 1e-12:
            best, best_lr = mean_ll, lr
    return LrSelection(lr=best_lr, mean_inner_log_loss=best, curve=curve,
                       n_inner_splits=len(inner_splits), grid=tuple(grid))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def state_diagnostics(states: pd.DataFrame) -> dict:
    """Mean / std / min / max per state column, plus stability flags."""
    d = {}
    for c in AD_COLUMNS:
        v = states[c].to_numpy(dtype=float)
        d[c] = {"mean": round(float(v.mean()), 6),
                "std": round(float(v.std()), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6)}
    allv = states[list(AD_COLUMNS)].to_numpy(dtype=float)
    d["all_finite"] = bool(np.all(np.isfinite(allv)))
    d["within_clip"] = bool(np.all(np.abs(allv) <= STATE_CLIP + 1e-9))
    # collapse = every state effectively identical
    d["collapsed"] = bool(np.all(allv.std(axis=0) < 1e-6))
    d["exploded"] = bool(np.any(np.abs(allv) >= STATE_CLIP - 1e-9))
    d["stable"] = bool(d["all_finite"] and d["within_clip"]
                       and not d["collapsed"])
    return d


def prediction_change_diagnostics(lam_h_a, lam_a_a, P_a,
                                  lam_h_b, lam_a_b, P_b,
                                  material_threshold: float = 0.01) -> dict:
    """Quantify how much Arm B actually moves predictions vs Arm A."""
    dlh = np.abs(np.asarray(lam_h_b) - np.asarray(lam_h_a))
    dla = np.abs(np.asarray(lam_a_b) - np.asarray(lam_a_a))
    dP = np.abs(np.asarray(P_b) - np.asarray(P_a))
    max_row_dP = dP.max(axis=1)
    return {
        "mean_abs_lambda_home_delta": round(float(dlh.mean()), 6),
        "mean_abs_lambda_away_delta": round(float(dla.mean()), 6),
        "max_lambda_delta": round(float(max(dlh.max(), dla.max())), 6),
        "mean_abs_probability_delta": round(float(dP.mean()), 6),
        "max_probability_delta": round(float(dP.max()), 6),
        "pct_fixtures_materially_changed": round(
            100 * float(np.mean(max_row_dP > material_threshold)), 4),
        "pct_top_prediction_changed": round(
            100 * float(np.mean(np.asarray(P_a).argmax(1)
                                != np.asarray(P_b).argmax(1))), 4),
        "material_threshold": material_threshold,
    }


def check_extreme_safety(lam_h, lam_a, P, max_lambda: float = 15.0) -> dict:
    """Verify lambdas positive/finite/bounded and probabilities valid."""
    lh, la = np.asarray(lam_h, float), np.asarray(lam_a, float)
    Pm = np.asarray(P, float)
    rowsum = Pm.sum(axis=1)
    r = {
        "lambda_home_range": [round(float(lh.min()), 6), round(float(lh.max()), 6)],
        "lambda_away_range": [round(float(la.min()), 6), round(float(la.max()), 6)],
        "lambda_finite": bool(np.all(np.isfinite(lh)) and np.all(np.isfinite(la))),
        "lambda_positive": bool(np.all(lh > 0) and np.all(la > 0)),
        "lambda_within_max": bool(lh.max() <= max_lambda and la.max() <= max_lambda),
        "prob_finite": bool(np.all(np.isfinite(Pm))),
        "prob_in_unit": bool(np.all(Pm >= 0) and np.all(Pm <= 1)),
        "prob_sums_to_one": bool(np.allclose(rowsum, 1.0, atol=1e-9)),
        "max_rowsum_deviation": round(float(np.max(np.abs(rowsum - 1.0))), 12),
    }
    r["passed"] = all(v for k, v in r.items()
                      if isinstance(v, bool))
    return r


__all__ = [
    "INIT_ATTACK", "INIT_DEFENSE", "STATE_CLIP", "EXP_CLIP", "AD_COLUMNS",
    "LR_GRID", "COMPLETED_STATUS", "CLASS_ORDER", "OnlineADError",
    "BaselineRates", "fit_baseline_rates", "compute_ad_states",
    "e6_feature_columns", "build_design",
    "InnerSplit", "build_inner_splits", "LrSelection", "select_lr",
    "state_diagnostics", "prediction_change_diagnostics",
    "check_extreme_safety",
]
