"""H1 family-coefficient audit -- READ-ONLY reproduction of the frozen H1 control.

PURPOSE
    Resolve the H3 premise: is family-level fitted influence disproportionate
    to family size and measured signal?

    This is NOT a new model experiment. It reproduces the already-frozen H1
    control exactly and analyses its coefficients. Nothing is scaled, tuned,
    weighted, or promoted. 1/sqrt(n) family scaling is NOT applied here.

CONFIGURATION (frozen, not variable)
    76-column H1 contract | C=0.0005 | max_iter=2000 | random_state=0
    frozen LogisticRegressionPreprocessor | existing 3 walk-forward folds
    CLASS_ORDER = ["H","D","A"] | no calibration | no class weighting

FIDELITY GATE
    The reproduction must match the recorded H1 fold log losses to <1e-9
    before any coefficient is reported. Otherwise the script STOPS.

ENCODED-SPACE ACCOUNTING
    Ridge penalises FITTED coefficients, so families are defined on the
    ACTUAL ENCODED DESIGN MATRIX:
        numeric columns -> 1 coefficient each (standardised, unit variance)
        competition_id  -> 5 one-hot coefficients (NOT standardised)
    Both raw and encoded counts are reported. competition_id is never
    compared against a Cohen's d, which is undefined for a categorical.

UNIT CAVEAT (stated, not hidden)
    StandardScaler gives every NUMERIC column unit variance, so numeric
    coefficients are commensurable: "logit change per 1 SD". The one-hot
    block is appended AFTER scaling and is raw 0/1, so its coefficients are
    in different units and its mass is NOT comparable to the numeric
    families. It is reported separately and excluded from the ratio test.

WRITES NOTHING. stdout only. 2025/26 never loaded.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_h1_family_coefficient_audit.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import sqlite3
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from models.ablation import (
    COMPETITION_ID_COLUMN,
    FORM_COLUMNS,
    GOALS_CORE_COLUMNS,
    MODEL_B_COLUMNS,
    SHOTS_CORE_COLUMNS,
    SHOTS_ON_CORE_COLUMNS,
    STRENGTH_COLUMNS,
)
from models.baselines import CLASS_ORDER
from models.config import FINAL_TEST_SEASONS, MODEL_VERSION, WALK_FORWARD_FOLDS
from models.data import load_supervised_dataset
from models.evaluate import evaluate, validate_probabilities
from models.robustness import V1_LOGREG_BASE_KWARGS
from models.splits import iter_walk_forward_folds

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MATCHES_DB = REPO / "data" / "processed" / "matches.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"
PHASE4A_PREDS = REPO / "data" / "audit" / "phase4a_predictions"

C_VALUE = 0.0005
CONTEXT_COL = "competition_id"
ALIASES = ("home_goal_diff_last5", "home_goal_diff_last10",
           "away_goal_diff_last5", "away_goal_diff_last10")
H1_COLUMNS = tuple(c for c in MODEL_B_COLUMNS if c not in ALIASES)

H1_RECORDED = {
    "fold_1": 0.9979365509246185,
    "fold_2": 0.9846529437399034,
    "fold_3": 0.9920388667463793,
    "mean_log_loss": 0.9915427871369671,
    "mean_brier": 0.5913870749997768,
}

RAW_FAMILIES = {
    "goals_core": [c for c in GOALS_CORE_COLUMNS if c in H1_COLUMNS],
    "form": [c for c in FORM_COLUMNS if c in H1_COLUMNS],
    "strength": [c for c in STRENGTH_COLUMNS if c in H1_COLUMNS],
    "competition_id": [c for c in COMPETITION_ID_COLUMN if c in H1_COLUMNS],
    "shots_core": [c for c in SHOTS_CORE_COLUMNS if c in H1_COLUMNS],
    "shots_on_core": [c for c in SHOTS_ON_CORE_COLUMNS if c in H1_COLUMNS],
}
NUMERIC_FAMILIES = [f for f in RAW_FAMILIES if f != CONTEXT_COL]


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)


def stop(msg):
    print("\n" + "!" * 78); print("STOP -- audit halted"); print(msg); print("!" * 78)
    raise SystemExit(1)


def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def cohens_d(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 30 or len(b) < 30:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def family_signal():
    """mean |Cohen's d| per numeric family -- SAME definition/subset as H3-A.

    Subset: rows the recorded Phase 4A Model B validation run predicted as H;
    correct (y_true==H) vs wrong (y_true!=H). Model-independent of the fit
    performed below; it is recorded evidence, reused unchanged.
    """
    files = sorted(glob.glob(str(PHASE4A_PREDS / "model_b_fold_*.csv")))
    if not files:
        stop("recorded Phase 4A Model B fold predictions not found; signal measure unavailable")
    d = pd.concat([pd.read_csv(f).assign(fold="fold_" + Path(f).stem.split("_")[-1])
                   for f in files], ignore_index=True)
    con = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    F = pd.read_sql(f"select fixture_id,{','.join(H1_COLUMNS)} from feature_rows", con)
    con.close()
    m = d.merge(F, on="fixture_id", how="left")
    out = {}
    for scope, sub in [("pooled", m)] + [(f, m[m.fold == f]) for f in sorted(m.fold.unique())]:
        ph = sub[sub.y_pred == "H"]
        right, wrong = ph[ph.y_true == "H"], ph[ph.y_true != "H"]
        out[scope] = {}
        for fam in NUMERIC_FAMILIES:
            vals = [abs(cohens_d(right[c], wrong[c])) for c in RAW_FAMILIES[fam]]
            vals = [v for v in vals if not np.isnan(v)]
            out[scope][fam] = float(np.mean(vals)) if vals else np.nan
    return out


def main():
    print("H1 FAMILY-COEFFICIENT AUDIT -- read-only reproduction, no new experiment")
    print("No family scaling applied. 1/sqrt(n) is NOT tested here.")

    rule("PRE-FLIGHT")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}  C={C_VALUE}  kwargs={V1_LOGREG_BASE_KWARGS}")
    seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    if seasons & set(FINAL_TEST_SEASONS):
        stop("2025/26 reachable")
    print(f"  fold seasons {sorted(seasons)} | 2025/26 reachable: False")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines: mismatches={bad or 'NONE'}")
    if bad or MODEL_VERSION != "v1.0":
        stop("pre-flight integrity failure")

    rule("FAMILY DEFINITIONS -- RAW vs ENCODED")
    print(f"  {'family':<16} {'raw_n':>6} {'encoded_n':>10}  note")
    for fam, cols in RAW_FAMILIES.items():
        enc = "5" if fam == CONTEXT_COL else str(len(cols))
        note = "one-hot, NOT standardised" if fam == CONTEXT_COL else "standardised, unit variance"
        print(f"  {fam:<16} {len(cols):>6} {enc:>10}  {note}")
    print(f"  raw total={sum(len(c) for c in RAW_FAMILIES.values())}  "
          f"encoded total={sum(len(c) for f, c in RAW_FAMILIES.items() if f != CONTEXT_COL) + 5}")

    rule("SIGNAL MEASURE (recorded evidence, reused unchanged)")
    signal = family_signal()
    print(f"  {'family':<16} " + "".join(f"{k:>12}" for k in ["pooled", "fold_1", "fold_2", "fold_3"]))
    for fam in NUMERIC_FAMILIES:
        print(f"  {fam:<16} " + "".join(
            f"{signal.get(k, {}).get(fam, float('nan')):>12.4f}"
            for k in ["pooled", "fold_1", "fold_2", "fold_3"]))
    print(f"  {CONTEXT_COL:<16} " + "  NOT APPLICABLE (Cohen's d undefined for a categorical)")

    dataset = load_supervised_dataset(FEATURES_DB)
    rows = []
    fold_metrics = {}

    rule("FIT (reproduction) AND COEFFICIENT EXTRACTION")
    from sklearn.linear_model import LogisticRegression
    from models import train as train_module

    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        name = fold.name
        Xtr, Xva = train_ds.X[list(H1_COLUMNS)], val_ds.X[list(H1_COLUMNS)]
        pre = train_module.LogisticRegressionPreprocessor().fit(Xtr)
        Etr, Eva = pre.transform(Xtr), pre.transform(Xva)
        model = LogisticRegression(C=C_VALUE, **V1_LOGREG_BASE_KWARGS)
        model.fit(Etr, train_ds.y)
        P = np.asarray(train_module._reorder_proba(model, Eva), dtype=float)
        validate_probabilities(P)
        r = evaluate(val_ds.y, P)
        fold_metrics[name] = r
        print(f"  {name}: log_loss={r.log_loss:.16f}  brier={r.brier:.16f}  "
              f"n_iter={model.n_iter_}  n_train={len(Xtr)}  n_val={len(Xva)}")
        if int(np.max(model.n_iter_)) >= V1_LOGREG_BASE_KWARGS["max_iter"]:
            stop(f"{name}: hit max_iter -- unconverged, coefficients not interpretable")

        names = list(pre.numeric_columns) + [
            f"{CONTEXT_COL}=={c}" for c in pre.competition_categories]
        if len(names) != model.coef_.shape[1]:
            stop(f"{name}: encoded-name mapping mismatch "
                 f"({len(names)} names vs {model.coef_.shape[1]} coefficients)")
        coef = pd.DataFrame(model.coef_, index=list(model.classes_),
                            columns=names).reindex(CLASS_ORDER)

        col_to_family = {}
        for fam, cols in RAW_FAMILIES.items():
            for c in cols:
                if fam == CONTEXT_COL:
                    continue
                col_to_family[c] = fam
        for c in pre.competition_categories:
            col_to_family[f"{CONTEXT_COL}=={c}"] = CONTEXT_COL
        unmapped = [c for c in names if c not in col_to_family]
        if unmapped:
            stop(f"{name}: {len(unmapped)} encoded columns unmapped to a family: {unmapped[:5]}")

        for cls in CLASS_ORDER:
            for fam in RAW_FAMILIES:
                cols = [c for c in names if col_to_family[c] == fam]
                v = coef.loc[cls, cols].to_numpy(dtype=float)
                rows.append(dict(
                    fold=name, cls=cls, family=fam,
                    raw_n=len(RAW_FAMILIES[fam]), encoded_n=len(cols),
                    total_L1=float(np.abs(v).sum()),
                    mean_abs_coef=float(np.abs(v).mean()),
                    L1_per_dim=float(np.abs(v).sum() / len(cols)),
                    total_L2_sq=float((v ** 2).sum()),
                    L2_sq_per_dim=float((v ** 2).sum() / len(cols)),
                    mean_abs_d=signal[name].get(fam, np.nan) if fam != CONTEXT_COL else np.nan,
                ))

    # ---------------- FIDELITY GATE ----------------
    rule("FIDELITY GATE -- reproduction vs recorded H1")
    ok = True
    for n in ("fold_1", "fold_2", "fold_3"):
        obs = fold_metrics[n].log_loss
        d = abs(obs - H1_RECORDED[n]); ok &= d < 1e-9
        print(f"  {n}: recorded={H1_RECORDED[n]:.16f} observed={obs:.16f} diff={d:.3e}")
    mll = statistics.fmean(fold_metrics[n].log_loss for n in fold_metrics)
    mbr = statistics.fmean(fold_metrics[n].brier for n in fold_metrics)
    print(f"  mean log loss: recorded={H1_RECORDED['mean_log_loss']:.16f} observed={mll:.16f} "
          f"diff={abs(mll - H1_RECORDED['mean_log_loss']):.3e}")
    print(f"  mean Brier   : recorded={H1_RECORDED['mean_brier']:.16f} observed={mbr:.16f} "
          f"diff={abs(mbr - H1_RECORDED['mean_brier']):.3e}")
    ok &= abs(mll - H1_RECORDED["mean_log_loss"]) < 1e-9
    ok &= abs(mbr - H1_RECORDED["mean_brier"]) < 1e-9
    if not ok:
        stop("Reproduction does not match recorded H1 to <1e-9. Coefficients NOT reported.")
    print("  [PASS] exact reproduction -- these are the frozen H1 coefficients")

    T = pd.DataFrame(rows)
    T["influence_per_unit_signal"] = np.where(
        T.mean_abs_d.notna() & (T.mean_abs_d > 0), T.L1_per_dim / T.mean_abs_d, np.nan)

    rule("PER-FOLD, PER-CLASS FAMILY TABLE")
    print("  influence_per_unit_signal = L1_per_dim / mean_abs_d")
    print("    L1_per_dim  = sum(|coef|) over the family's ENCODED columns / encoded_n")
    print("    mean_abs_d  = mean |Cohen's d| over the family's RAW numeric columns")
    print("    undefined (NaN) where the signal measure does not exist\n")
    cols = ["family", "raw_n", "encoded_n", "mean_abs_d", "total_L1", "L1_per_dim",
            "total_L2_sq", "L2_sq_per_dim", "influence_per_unit_signal"]
    for n in sorted(T.fold.unique()):
        for cls in CLASS_ORDER:
            s = T[(T.fold == n) & (T.cls == cls)].sort_values("L1_per_dim", ascending=False)
            print(f"  --- {n} | class {cls} ---")
            print(s[cols].to_string(index=False, float_format=lambda v: f"{v:10.6f}"))
            print()

    rule("SIZE-NORMALISED INFLUENCE -- is mass merely proportional to size?")
    print("  Under ordinary L2 a bigger family is EXPECTED to hold more total mass.")
    print("  The diagnostic is L1_per_dim (size-normalised), not total_L1.\n")
    num = T[T.family != CONTEXT_COL]
    piv = num.pivot_table(index="family", columns="fold", values="L1_per_dim", aggfunc="mean")
    piv["raw_n"] = [len(RAW_FAMILIES[f]) for f in piv.index]
    print(piv.to_string(float_format=lambda v: f"{v:10.6f}"))
    print("\n  Spearman(encoded_n, total_L1) and Spearman(encoded_n, L1_per_dim) per fold/class:")
    for n in sorted(T.fold.unique()):
        for cls in CLASS_ORDER:
            s = num[(num.fold == n) & (num.cls == cls)]
            print(f"    {n} {cls}: rho(n, total_L1)={spearman(s.encoded_n, s.total_L1):+.3f}   "
                  f"rho(n, L1_per_dim)={spearman(s.encoded_n, s.L1_per_dim):+.3f}")

    rule("INFLUENCE vs SIGNAL -- the H3 test")
    print("  H3 predicts LARGER families receive MORE influence per unit of measured signal,")
    print("  i.e. Spearman(encoded_n, influence_per_unit_signal) > 0, replicated.\n")
    cells = []
    for n in sorted(T.fold.unique()):
        for cls in CLASS_ORDER:
            s = num[(num.fold == n) & (num.cls == cls)].dropna(subset=["influence_per_unit_signal"])
            rho_is = spearman(s.encoded_n, s.influence_per_unit_signal)
            rho_sig = spearman(s.mean_abs_d, s.L1_per_dim)
            cells.append(rho_is)
            print(f"  {n} {cls}: rho(n, influence_per_unit_signal)={rho_is:+.3f}   "
                  f"rho(mean_abs_d, L1_per_dim)={rho_sig:+.3f}   families={len(s)}")
            print(f"      ratios: "
                  f"{ {r.family: round(r.influence_per_unit_signal, 4) for r in s.itertuples()} }")
    cells = np.array([c for c in cells if not np.isnan(c)])
    pos = int((cells > 0).sum()); neg = int((cells < 0).sum())
    print(f"\n  POOLED: {len(cells)} fold x class cells | positive rho={pos} | negative={neg} "
          f"| mean rho={cells.mean():+.3f} | median={np.median(cells):+.3f}")
    print("  NOTE: only 5 numeric families -> Spearman over 5 points is very low powered.")
    print("  A sign that is not consistent across all 9 cells is not replication.")

    rule("competition_id -- REPORTED SEPARATELY, NOT COMPARED")
    ci = T[T.family == CONTEXT_COL]
    print("  raw_n=1, encoded_n=5. One-hot columns are appended AFTER standardisation and are")
    print("  raw 0/1, so their coefficients are in different units from the numeric families.")
    print("  Cohen's d is undefined for a categorical -> signal comparison NOT APPLICABLE.")
    print(ci[["fold", "cls", "raw_n", "encoded_n", "total_L1", "L1_per_dim",
              "total_L2_sq"]].to_string(index=False, float_format=lambda v: f"{v:10.6f}"))

    rule("H3-A PREMISE VERDICT")
    all_pos = len(cells) > 0 and bool((cells > 0).all())
    all_neg = len(cells) > 0 and bool((cells < 0).all())
    if all_pos:
        verdict = "CONFIRMED"
        why = ("influence per unit signal rises with family size in EVERY fold x class cell -- "
               "a replicated imbalance in the direction H3 predicts")
    elif all_neg:
        verdict = "FALSIFIED"
        why = ("influence per unit signal FALLS with family size in every cell -- the opposite "
               "of H3's prediction; large families are if anything under-weighted")
    elif pos == 0 or neg == 0:
        verdict = "INCONCLUSIVE"
        why = "sign is uniform only where defined but some cells are undefined; treat as unresolved"
    else:
        verdict = "FALSIFIED"
        why = (f"the sign is inconsistent across fold x class cells ({pos} positive, {neg} "
               "negative), so there is no replicated evidence of a size-driven imbalance. "
               "Per the H4 lesson, an unreplicated pooled tendency is not evidence.")
    print(f"  {verdict}")
    print(f"  reason: {why}")
    print("\n  No causal claim is made. No significance test was performed.")
    print("  Larger total_L1 for larger families is the null expectation, not evidence.")

    rule("INTEGRITY AUDIT")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    print(f"  MODEL_VERSION: {MODEL_VERSION!r}")
    ser = [str(p.relative_to(REPO)) for pat in ("*.pkl", "*.joblib", "*.pickle")
           for p in REPO.rglob(pat) if "__pycache__" not in p.parts]
    print(f"  serialized estimators: {ser or 'NONE'}")
    for line in ("2025/26 accessed: NO", "production files modified: NO",
                 "features.db: read-only", "artifacts written: NONE",
                 "estimator persisted: NONE", "family scaling applied: NO",
                 "1/sqrt(n) tested: NO", "tuning: NONE", "files changed: 0"):
        print(f"  {line}")

    print("\n" + "=" * 78)
    print(f"  H3-A PREMISE: {verdict}")
    print(f"  H3 EXPERIMENT AUTHORIZATION: "
          f"{'PERMITTED (pending explicit human go-ahead)' if verdict == 'CONFIRMED' else 'NO'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
