"""DRAW-SEPARATION DIAGNOSTIC -- READ-ONLY.

PRIMARY QUESTION
    Does the frozen H1 76-column contract contain reproducible, out-of-sample
    information separating DRAW from NON-DRAW matches?

DIAGNOSTIC ONLY. No model, no features, no class weights, no thresholds,
no calibration, no C search, no solver change, no resampling, no feature
selection, no production change, no 2025/26, no artifact. Writes nothing.

FROZEN CONFIGURATION
    76-column H1 contract | C=0.0005 | max_iter=2000 | random_state=0
    frozen LogisticRegressionPreprocessor | 3 existing walk-forward folds

MANDATORY GATE
    Exact H1 reproduction to <1e-9 on all three folds and the mean, and no
    fit at max_iter. Otherwise STOP before any draw analysis.

EVIDENCE RULES ENFORCED IN CODE
    - REPLICATED requires >=2 of 3 folds with the SAME direction;
    - pooled data never upgrades a non-replicated pattern;
    - all regime boundaries come from the TRAINING partition only;
    - no significance testing, so no p-value selection;
    - chance-replication is quantified explicitly for the multi-feature scan.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_h1_draw_separation_diagnostic.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from models.ablation import (
    COMPETITION_ID_COLUMN, FORM_COLUMNS, GOALS_CORE_COLUMNS, MODEL_B_COLUMNS,
    SHOTS_CORE_COLUMNS, SHOTS_ON_CORE_COLUMNS, STRENGTH_COLUMNS,
)
from models.baselines import CLASS_ORDER
from models.config import FINAL_TEST_SEASONS, MODEL_VERSION, WALK_FORWARD_FOLDS
from models.data import load_supervised_dataset
from models.evaluate import evaluate, validate_probabilities
from models.robustness import V1_LOGREG_BASE_KWARGS
from models.splits import iter_walk_forward_folds

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"

C_VALUE = 0.0005
CONTEXT_COL = "competition_id"
ALIASES = ("home_goal_diff_last5", "home_goal_diff_last10",
           "away_goal_diff_last5", "away_goal_diff_last10")
H1_COLUMNS = tuple(c for c in MODEL_B_COLUMNS if c not in ALIASES)
NUMERIC = tuple(c for c in H1_COLUMNS if c != CONTEXT_COL)
H1_RECORDED = {"fold_1": 0.9979365509246185, "fold_2": 0.9846529437399034,
               "fold_3": 0.9920388667463793, "mean": 0.9915427871369671}
FAMILIES = {
    "goals_core": [c for c in GOALS_CORE_COLUMNS if c in H1_COLUMNS],
    "form": [c for c in FORM_COLUMNS if c in H1_COLUMNS],
    "strength": [c for c in STRENGTH_COLUMNS if c in H1_COLUMNS],
    "competition_id": [c for c in COMPETITION_ID_COLUMN if c in H1_COLUMNS],
    "shots_core": [c for c in SHOTS_CORE_COLUMNS if c in H1_COLUMNS],
    "shots_on_core": [c for c in SHOTS_ON_CORE_COLUMNS if c in H1_COLUMNS],
}
MIN_N = 60
EPS = 1e-15


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)


def stop(m):
    print("\n" + "!" * 78); print("STOP"); print(m); print("!" * 78); raise SystemExit(1)


def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()


def auc(score, label):
    """Rank-based AUC. label=1 marks the positive (Draw) class."""
    m = ~np.isnan(score)
    s, y = score[m], label[m]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_auc(score, label):
    """Average precision via the step-wise precision-recall sum."""
    m = ~np.isnan(score)
    s, y = score[m], label[m]
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def cohens_d(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 30 or len(b) < 30:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def run():
    results = {}

    rule("STEP 0 -- PRE-FLIGHT")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION == 'v1.0'      : {MODEL_VERSION == 'v1.0'} ({MODEL_VERSION!r})")
    print(f"  contract columns             : {len(H1_COLUMNS)} "
          f"({len(NUMERIC)} numeric + competition_id) -> "
          f"{len(H1_COLUMNS) == 76 and len(NUMERIC) == 75}")
    seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    if seasons & set(FINAL_TEST_SEASONS):
        stop("2025/26 reachable")
    print(f"  folds                        : {[f.name for f in WALK_FORWARD_FOLDS]}")
    print(f"  2025/26 unreachable          : True")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines          : mismatches={bad or 'NONE'}")
    if bad or MODEL_VERSION != "v1.0" or len(H1_COLUMNS) != 76:
        stop("pre-flight failure")
    print("  tuning / calibration / model change: NONE (single fixed configuration)")

    rule("STEP 1 -- EXACT FROZEN H1 REPRODUCTION GATE")
    from sklearn.linear_model import LogisticRegression
    from models import train as train_module

    dataset = load_supervised_dataset(FEATURES_DB)
    folds = {}
    for fold, tr, va in iter_walk_forward_folds(dataset):
        Xtr, Xva = tr.X[list(H1_COLUMNS)], va.X[list(H1_COLUMNS)]
        pre = train_module.LogisticRegressionPreprocessor().fit(Xtr)
        Etr, Eva = pre.transform(Xtr), pre.transform(Xva)
        model = LogisticRegression(C=C_VALUE, **V1_LOGREG_BASE_KWARGS).fit(Etr, tr.y)
        if int(np.max(model.n_iter_)) >= V1_LOGREG_BASE_KWARGS["max_iter"]:
            stop(f"{fold.name}: hit max_iter -- unconverged")
        P = np.asarray(train_module._reorder_proba(model, Eva), dtype=float)
        validate_probabilities(P)
        r = evaluate(va.y, P)
        d = abs(r.log_loss - H1_RECORDED[fold.name])
        print(f"  {fold.name}: log_loss={r.log_loss:.16f} recorded={H1_RECORDED[fold.name]:.16f} "
              f"diff={d:.3e} n_iter={model.n_iter_}")
        if d >= 1e-9:
            stop(f"{fold.name} differs by >=1e-9 -- diagnostic invalid, no draw analysis produced")
        folds[fold.name] = dict(P=P, y=va.y.to_numpy(), r=r, model=model, pre=pre,
                                Xtr=Xtr, Xva=Xva, Eva=Eva,
                                names=list(pre.numeric_columns)
                                + [f"{CONTEXT_COL}=={c}" for c in pre.competition_categories])
    mean_ll = statistics.fmean(folds[n]["r"].log_loss for n in folds)
    print(f"  mean={mean_ll:.16f} recorded={H1_RECORDED['mean']:.16f} "
          f"diff={abs(mean_ll - H1_RECORDED['mean']):.3e}")
    if abs(mean_ll - H1_RECORDED["mean"]) >= 1e-9:
        stop("mean differs by >=1e-9")
    print("  [PASS] exact reproduction")

    for n, f in folds.items():
        P, y = f["P"], f["y"]
        idx = {c: i for i, c in enumerate(CLASS_ORDER)}
        rank = np.argsort(-P, axis=1)
        f["df"] = pd.DataFrame({
            "y_true": y, "y_pred": np.array(CLASS_ORDER)[P.argmax(1)],
            "p_H": P[:, 0], "p_D": P[:, 1], "p_A": P[:, 2], "p_max": P.max(1),
            "is_draw": (y == "D").astype(int),
            "d_rank": [int(np.where(rank[i] == idx["D"])[0][0]) + 1 for i in range(len(y))],
            "p_true": [P[i, idx[c]] for i, c in enumerate(y)],
        })

    # ---------------- STEP 2 ----------------
    rule("STEP 2 -- DRAW / NON-DRAW BASELINE (descriptive; AUC is not proof)")
    for n, f in folds.items():
        d = f["df"]; td, nd = d[d.is_draw == 1], d[d.is_draw == 0]
        q = lambda s: np.percentile(s, [10, 25, 50, 75, 90])
        print(f"\n  {n}: n_draw={len(td)} n_non_draw={len(nd)} prevalence={d.is_draw.mean():.6f}")
        print(f"    mean p_draw  true-D={td.p_D.mean():.6f}  non-D={nd.p_D.mean():.6f}  "
              f"difference={td.p_D.mean() - nd.p_D.mean():+.6f}")
        print(f"    median       true-D={td.p_D.median():.6f}  non-D={nd.p_D.median():.6f}")
        print(f"    true-D  p10/p25/p50/p75/p90 = {np.round(q(td.p_D), 6)}")
        print(f"    non-D   p10/p25/p50/p75/p90 = {np.round(q(nd.p_D), 6)}")
        a = auc(d.p_D.to_numpy(), d.is_draw.to_numpy())
        p = pr_auc(d.p_D.to_numpy(), d.is_draw.to_numpy())
        print(f"    ROC-AUC(p_draw -> D) = {a:.6f}   PR-AUC = {p:.6f}   "
              f"(PR baseline = prevalence = {d.is_draw.mean():.6f})")
        results.setdefault("auc_pdraw", {})[n] = a

    # ---------------- STEP 3 ----------------
    rule("STEP 3 -- UNIVARIATE DRAW SIGNAL (validation rows only, no significance tests)")
    rows = []
    for n, f in folds.items():
        d, X = f["df"], f["Xva"].reset_index(drop=True)
        y = d.is_draw.to_numpy()
        for c in NUMERIC:
            x = X[c].to_numpy(float)
            a = auc(x, y)
            rows.append(dict(fold=n, feature=c, auc=a, dev=abs(a - 0.5) if a == a else np.nan,
                             d=cohens_d(x[y == 1], x[y == 0]),
                             mean_D=np.nanmean(x[y == 1]), mean_nonD=np.nanmean(x[y == 0]),
                             med_D=np.nanmedian(x[y == 1]), med_nonD=np.nanmedian(x[y == 0]),
                             rank_biserial=(2 * a - 1) if a == a else np.nan))
    R = pd.DataFrame(rows)
    for n in folds:
        s = R[R.fold == n].dropna(subset=["dev"])
        print(f"\n  {n}: mean|AUC-0.5|={s.dev.mean():.5f} max={s.dev.max():.5f} "
              f"mean|d|={s.d.abs().mean():.5f} max|d|={s.d.abs().max():.5f}")
        print("    top 20 by |AUC-0.5|:")
        for r in s.nlargest(20, "dev").itertuples():
            print(f"      {r.feature:<44} AUC={r.auc:.4f} rb={r.rank_biserial:+.4f} "
                  f"d={r.d:+.4f} meanD={r.mean_D:+.4f} mean_nonD={r.mean_nonD:+.4f}")
    piv = R.pivot(index="feature", columns="fold", values="auc")
    cons = ((piv > 0.5).sum(1) == 3) | ((piv < 0.5).sum(1) == 3)
    two = ((piv > 0.5).sum(1) >= 2) | ((piv < 0.5).sum(1) >= 2)
    piv["mean_dev"] = (piv[[c for c in piv.columns]] - 0.5).abs().mean(1)
    n_feat = len(piv)
    p_one = 2 * 0.5 ** 3
    p_ge = sum(comb(n_feat, k) * p_one ** k * (1 - p_one) ** (n_feat - k)
               for k in range(int(cons.sum()), n_feat + 1))
    print(f"\n  features with SAME direction in all 3 folds: {int(cons.sum())} of {n_feat}")
    print(f"  expected under pure noise: {n_feat * p_one:.1f} | P(>= observed | noise) = {p_ge:.4f}")
    print(f"  features with same direction in >=2 of 3 folds: {int(two.sum())}")
    print("  top 12 consistently-directed CANDIDATES (not findings):")
    print(piv[cons].nlargest(12, "mean_dev").to_string(float_format=lambda v: f"{v:9.4f}"))
    results["consistent_all3"] = int(cons.sum())
    results["p_chance"] = float(p_ge)
    results["max_dev"] = float(R.dev.max())

    # ---------------- STEP 4 ----------------
    rule("STEP 4 -- DRAW SIGNAL BY FEATURE FAMILY")
    R["family"] = [next((f for f, cols in FAMILIES.items() if x in cols), "?") for x in R.feature]
    fp = R.pivot_table(index="family", columns="fold", values="dev", aggfunc="mean")
    fp["n"] = [len(FAMILIES[f]) for f in fp.index]
    fp["mean_abs_effect"] = fp[[c for c in fp.columns if c.startswith("fold")]].mean(1)
    print(fp.sort_values("mean_abs_effect", ascending=False)
          .to_string(float_format=lambda v: f"{v:9.5f}"))
    print("\n  per-family strongest / weakest feature and consistent-direction count:")
    for fam, cols in FAMILIES.items():
        if fam == CONTEXT_COL:
            continue
        s = R[R.family == fam].groupby("feature").dev.mean().sort_values()
        cc = int(cons[[c for c in cols if c in cons.index]].sum())
        print(f"    {fam:<15} n={len(cols):2d} consistent3={cc:2d} "
              f"strongest={s.index[-1]} ({s.iloc[-1]:.4f}) weakest={s.index[0]} ({s.iloc[0]:.4f})")
    print("\n  competition_id: REPORTED SEPARATELY -- categorical, 1 raw / 5 encoded.")
    for n, f in folds.items():
        d = f["df"]; X = f["Xva"].reset_index(drop=True)
        t = pd.crosstab(X[CONTEXT_COL], d.is_draw, normalize="index")
        print(f"    {n} draw rate by league: "
              f"{ {int(k): round(float(v.get(1, 0)), 4) for k, v in t.iterrows()} }")
    print("    Cohen's d / AUC are NOT computed for it; not comparable to numeric effects.")

    # ---------------- STEPS 5-6 ----------------
    rule("STEPS 5-6 -- MATCH-BALANCE REGIMES (boundaries from TRAINING partition only)")
    print("  Regimes fixed before inspecting validation labels: |strength_diff| terciles.")
    regime = []
    for n, f in folds.items():
        tr_sd = f["Xtr"]["strength_diff"].abs().dropna()
        lo, hi = np.nanquantile(tr_sd, [1 / 3, 2 / 3])
        d = f["df"].copy()
        sd = f["Xva"]["strength_diff"].abs().to_numpy()
        d["regime"] = np.where(sd <= lo, "balanced", np.where(sd <= hi, "mid", "lopsided"))
        print(f"\n  {n}: TRAIN-derived edges |sd| = {lo:.6f}, {hi:.6f}")
        print(f"    {'regime':<11}{'n':>6}{'draw_rate':>11}{'non_draw':>10}"
              f"{'mean_p_draw':>13}{'draw_recall':>13}{'D_rank1':>9}{'D_rank2':>9}")
        for reg in ("balanced", "mid", "lopsided"):
            g = d[d.regime == reg]
            if len(g) < MIN_N:
                print(f"    {reg:<11}{len(g):>6}   (n<{MIN_N}, not reported)")
                continue
            td = g[g.is_draw == 1]
            rec = float((td.y_pred == "D").mean()) if len(td) else float("nan")
            print(f"    {reg:<11}{len(g):>6}{g.is_draw.mean():>11.6f}{1 - g.is_draw.mean():>10.6f}"
                  f"{g.p_D.mean():>13.6f}{rec:>13.6f}"
                  f"{float((td.d_rank == 1).mean()):>9.4f}{float((td.d_rank == 2).mean()):>9.4f}")
            regime.append(dict(fold=n, regime=reg, draw_rate=g.is_draw.mean(),
                               mean_p_draw=g.p_D.mean()))
    RG = pd.DataFrame(regime)
    if len(RG):
        pv = RG.pivot(index="regime", columns="fold", values="draw_rate")
        print("\n  draw rate by regime and fold:")
        print(pv.to_string(float_format=lambda v: f"{v:10.6f}"))
        if {"balanced", "lopsided"} <= set(pv.index):
            diff = pv.loc["balanced"] - pv.loc["lopsided"]
            pos = int((diff > 0).sum())
            print(f"  balanced-minus-lopsided draw rate per fold: "
                  f"{ {k: round(float(v), 6) for k, v in diff.items()} }")
            print(f"  folds with balanced > lopsided: {pos} of 3 -> "
                  f"{'REPLICATED' if pos >= 2 else 'NOT REPLICATED'}")
            results["balance_replicated"] = pos

    # ---------------- STEP 7 ----------------
    rule("STEP 7 -- PROBABILITY RANKING DIAGNOSTIC (decision rule unchanged)")
    for n, f in folds.items():
        d = f["df"]; td, nd = d[d.is_draw == 1], d[d.is_draw == 0]
        print(f"\n  {n}: true draws n={len(td)}")
        for lab, g in (("true-D", td), ("non-D", nd)):
            rk = Counter(g.d_rank)
            print(f"    {lab:<7} D rank1={rk.get(1,0)/len(g):.4f} rank2={rk.get(2,0)/len(g):.4f} "
                  f"rank3={rk.get(3,0)/len(g):.4f} mean_rank={g.d_rank.mean():.4f} "
                  f"median={int(g.d_rank.median())}")
            print(f"            p_D={g.p_D.mean():.6f} p_H={g.p_H.mean():.6f} p_A={g.p_A.mean():.6f}")
        lift = (td.d_rank <= 2).mean() - (nd.d_rank <= 2).mean()
        print(f"    P(D ranked in top 2): true-D={(td.d_rank<=2).mean():.4f} "
              f"non-D={(nd.d_rank<=2).mean():.4f} lift={lift:+.4f}")
        results.setdefault("rank_lift", {})[n] = float(lift)

    # ---------------- STEP 8 ----------------
    rule("STEP 8 -- DRAW SIGNAL vs MODEL CONFIDENCE (descriptive bins)")
    bins = [(-np.inf, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, np.inf)]
    for n, f in folds.items():
        d = f["df"]
        print(f"\n  {n}: {'p_draw bin':<14}{'n':>7}{'draw_rate':>11}{'mean_p_draw':>13}"
              f"{'mean_max_p':>12}{'D_rank1':>9}{'D_rank2':>9}")
        for lo, hi in bins:
            g = d[(d.p_D > lo) & (d.p_D <= hi)]
            if len(g) < 20:
                continue
            print(f"       {f'{lo:.2f}-{hi:.2f}':<14}{len(g):>7}{g.is_draw.mean():>11.6f}"
                  f"{g.p_D.mean():>13.6f}{g.p_max.mean():>12.6f}"
                  f"{float((g.d_rank==1).mean()):>9.4f}{float((g.d_rank==2).mean()):>9.4f}")
        tail = d[d.p_D >= 0.35]
        print(f"    elevated tail p_draw>=0.35: n={len(tail)} "
              f"draw_rate={tail.is_draw.mean() if len(tail) else float('nan'):.6f} "
              f"(fold prevalence={d.is_draw.mean():.6f})")

    # ---------------- STEP 9 ----------------
    rule("STEP 9 -- COUNTERFACTUAL (upper bound only; NOT a fixability estimate)")
    print("  Single-feature move needed to make D the argmax for TRUE DRAW rows, using the")
    print("  actual fitted H1 coefficients, holding all other features fixed. Because the")
    print("  contract is highly correlated this OVERSTATES how movable these rows are.\n")
    recur = defaultdict(set)
    for n, f in folds.items():
        d, model, names, Eva = f["df"], f["model"], f["names"], f["Eva"]
        coef = pd.DataFrame(model.coef_, index=list(model.classes_),
                            columns=names).reindex(CLASS_ORDER)
        inter = dict(zip(list(model.classes_), np.asarray(model.intercept_)))
        idxs = np.where((d.is_draw == 1).to_numpy())[0]
        dist, feats = [], []
        for i in idxs:
            top = d.y_pred.iloc[i]
            if top == "D":
                continue
            gap = float((coef.loc[top].to_numpy() - coef.loc["D"].to_numpy()) @ Eva[i]
                        + inter[top] - inter["D"])
            dc = coef.loc["D"].to_numpy() - coef.loc[top].to_numpy()
            j = int(np.argmax(np.abs(dc)))
            dist.append(abs(gap / dc[j]) if dc[j] != 0 else np.inf)
            feats.append(names[j])
        arr = np.array(dist, float); fin = arr[np.isfinite(arr)]
        print(f"  {n}: true draws not predicted D = {len(dist)}")
        print(f"    move needed (SD): median={np.median(fin):.4f} p90={np.percentile(fin,90):.4f}")
        print(f"    within 0.5 SD={float((fin<=0.5).mean()):.4f}  within 1.0 SD={float((fin<=1).mean()):.4f}")
        top = Counter(feats).most_common(8)
        print(f"    most recurring: {top}")
        for name, _ in top:
            recur[name].add(n)

    rule("STEP 10 -- REPLICATION SUMMARY")
    rec2 = {k: sorted(v) for k, v in recur.items() if len(v) >= 2}
    print(f"  counterfactual features recurring in >=2 folds: {len(rec2)}")
    for k, v in sorted(rec2.items(), key=lambda x: -len(x[1])):
        print(f"    {k:<44} folds={v}")
    print(f"\n  p_draw ROC-AUC per fold: "
          f"{ {k: round(v, 6) for k, v in results.get('auc_pdraw', {}).items()} }")
    print(f"  D-in-top-2 lift per fold: "
          f"{ {k: round(v, 6) for k, v in results.get('rank_lift', {}).items()} }")
    print(f"  balanced>lopsided draw rate: {results.get('balance_replicated')} of 3 folds")
    print(f"  consistently-directed features (3/3): {results.get('consistent_all3')} "
          f"(P under noise = {results.get('p_chance'):.4f}); max |AUC-0.5| = "
          f"{results.get('max_dev'):.4f}")

    rule("STEP 11 -- DRAW INFORMATION CEILING CLASSIFICATION")
    print("  Categories A-F are argued in the written report from the evidence above.")
    print("  The script does not auto-assign: the choice depends on replication judgements")
    print("  that must be visible and auditable, not hidden in a code branch.")
    print("  Key discriminators printed above:")
    print("    - p_draw ROC-AUC (is there ANY usable draw ranking?)")
    print("    - D-in-top-2 lift (ranking useful but argmax suppressed? -> C)")
    print("    - max univariate |AUC-0.5| and replication count (-> A or E)")
    print("    - balanced-vs-lopsided replication (-> structured signal)")
    print("    - Step 8 elevated tail (is there a usable high-p_draw subpopulation?)")

    rule("STEP 12 -- PRE-REGISTERED NEXT-ACTION RULE")
    print("  Exactly one of STOP / INVESTIGATE / CONDITIONAL EXPERIMENT / INCONCLUSIVE is")
    print("  stated in the written report. This script implements NO experiment.")

    return folds, results, before, pins


def main():
    print("DRAW-SEPARATION DIAGNOSTIC -- read-only, no experiment, no model change")
    folds, results, before, pins = run()

    rule("STEP 13 -- DETERMINISM (full repeat)")
    folds2, results2, _, _ = run()
    same = all(folds2[n]["P"].tobytes() == folds[n]["P"].tobytes() for n in folds)
    for n in folds:
        print(f"  {n}: predictions byte-identical={folds2[n]['P'].tobytes()==folds[n]['P'].tobytes()} "
              f"sha256={hashlib.sha256(folds[n]['P'].tobytes()).hexdigest()[:32]}...")
    print(f"  DETERMINISM: {'PASS' if same else 'FAIL'}")
    if not same:
        stop("non-deterministic")

    rule("STEP 14 -- FINAL INTEGRITY AUDIT")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}  C={C_VALUE}  folds={sorted(folds)}")
    for line in ("2025/26 accessed = NO", "production source modified = NO",
                 "features.db modified = NO (read-only)", "tuning = NONE",
                 "calibration = NONE", "feature selection = NONE",
                 "threshold change = NONE", "class weighting = NONE",
                 "resampling = NONE", "artifacts = NONE",
                 "estimator persisted = NONE", "files written = NONE"):
        print(f"  {line}")
    print("\n" + "=" * 78)
    print("  DRAW DIAGNOSTIC ONLY")
    print("  NO H5 EXPERIMENT RUN")
    print("  NO MODEL CHANGE")
    print("  NO 2025/26 ACCESS")
    print("=" * 78)


if __name__ == "__main__":
    main()
