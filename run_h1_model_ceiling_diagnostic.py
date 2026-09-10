"""FINAL MODEL-CEILING / ERROR-ATTRIBUTION DIAGNOSTIC -- READ-ONLY.

QUESTION
    Where does the remaining H1 validation error come from, and is there
    evidence the 76-column feature set contains exploitable signal the
    frozen linear-additive model is failing to use?

NOT AN IMPROVEMENT EXPERIMENT. Nothing is tuned, calibrated, reweighted,
selected, added, removed or reordered. The argmax rule is unchanged.
2025/26 is never loaded. Writes nothing; stdout only.

FROZEN CONFIGURATION
    76-column H1 contract | C=0.0005 | max_iter=2000 | random_state=0
    frozen LogisticRegressionPreprocessor | 3 existing walk-forward folds
    CLASS_ORDER = ["H","D","A"]

MANDATORY GATE
    The reproduction must match the recorded H1 fold log losses to <1e-9.
    Otherwise the script STOPS before any interpretation.

EVIDENCE RULES ENFORCED IN CODE
    - a pattern is reported as a FINDING only if it replicates in >=2 of 3
      folds with consistent direction and adequate n;
    - pooled-only and single-fold patterns are printed but labelled
      NOT A FINDING;
    - regime boundaries are derived from the TRAINING partition only, never
      from validation labels;
    - no significance testing, therefore no p-value selection.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_h1_model_ceiling_diagnostic.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
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
NUMERIC_COLUMNS = tuple(c for c in H1_COLUMNS if c != CONTEXT_COL)

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

MIN_BIN_N = 60          # adequate-sample threshold for a regime cell
MIN_FOLDS = 2           # replication bar
EPS = 1e-15


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)


def stop(msg):
    print("\n" + "!" * 78); print("STOP -- diagnostic halted"); print(msg); print("!" * 78)
    raise SystemExit(1)


def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()


def quantiles(v, qs):
    return [float(np.nanquantile(v, q)) for q in qs]


def main():
    print("FINAL MODEL-CEILING / ERROR-ATTRIBUTION DIAGNOSTIC (read-only)")
    print("Purpose: determine WHETHER another experiment is justified. Not an experiment.")

    # ================= GATE =================
    rule("0. PRE-FLIGHT AND FROZEN H1 REPRODUCTION GATE")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r} C={C_VALUE} kwargs={V1_LOGREG_BASE_KWARGS}")
    print(f"  contract columns={len(H1_COLUMNS)} (numeric={len(NUMERIC_COLUMNS)})")
    seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    if seasons & set(FINAL_TEST_SEASONS):
        stop("2025/26 reachable through the folds")
    print(f"  fold seasons {sorted(seasons)} | 2025/26 reachable: False")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines: mismatches={bad or 'NONE'}")
    if bad or MODEL_VERSION != "v1.0":
        stop("pre-flight integrity failure")

    from sklearn.linear_model import LogisticRegression
    from models import train as train_module

    dataset = load_supervised_dataset(FEATURES_DB)
    folds = {}
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        Xtr, Xva = train_ds.X[list(H1_COLUMNS)], val_ds.X[list(H1_COLUMNS)]
        pre = train_module.LogisticRegressionPreprocessor().fit(Xtr)
        Etr, Eva = pre.transform(Xtr), pre.transform(Xva)
        model = LogisticRegression(C=C_VALUE, **V1_LOGREG_BASE_KWARGS)
        model.fit(Etr, train_ds.y)
        P = np.asarray(train_module._reorder_proba(model, Eva), dtype=float)
        validate_probabilities(P)
        r = evaluate(val_ds.y, P)
        if int(np.max(model.n_iter_)) >= V1_LOGREG_BASE_KWARGS["max_iter"]:
            stop(f"{fold.name}: hit max_iter -- unconverged")
        folds[fold.name] = dict(
            P=P, y=val_ds.y.to_numpy(), r=r, model=model, pre=pre,
            Xtr=Xtr, Xva=Xva, Eva=Eva, ytr=train_ds.y.to_numpy(),
            names=list(pre.numeric_columns) + [f"{CONTEXT_COL}=={c}"
                                               for c in pre.competition_categories])
        print(f"  {fold.name}: log_loss={r.log_loss:.16f} recorded={H1_RECORDED[fold.name]:.16f} "
              f"diff={abs(r.log_loss - H1_RECORDED[fold.name]):.3e}")
        if abs(r.log_loss - H1_RECORDED[fold.name]) >= 1e-9:
            stop(f"{fold.name} reproduction differs by >=1e-9. No interpretation follows.")
    mean_ll = statistics.fmean(folds[n]["r"].log_loss for n in folds)
    print(f"  mean: {mean_ll:.16f} recorded={H1_RECORDED['mean']:.16f} "
          f"diff={abs(mean_ll - H1_RECORDED['mean']):.3e}")
    if abs(mean_ll - H1_RECORDED["mean"]) >= 1e-9:
        stop("mean reproduction differs by >=1e-9")
    print("  [PASS] frozen H1 reproduced exactly -- diagnostics below are on the frozen model")

    # per-row frame
    for n, f in folds.items():
        P, y = f["P"], f["y"]
        srt = np.sort(P, axis=1)
        idx = {c: i for i, c in enumerate(CLASS_ORDER)}
        f["df"] = pd.DataFrame({
            "y_true": y,
            "y_pred": np.array(CLASS_ORDER)[P.argmax(axis=1)],
            "p_pred": P.max(axis=1),
            "p_true": np.array([P[i, idx[c]] for i, c in enumerate(y)]),
            "margin": srt[:, 2] - srt[:, 1],
            "p_H": P[:, 0], "p_D": P[:, 1], "p_A": P[:, 2],
        })
        f["df"]["correct"] = f["df"].y_true == f["df"].y_pred
        f["df"]["nll"] = -np.log(np.clip(f["df"].p_true, EPS, 1.0))
        f["df"]["true_rank"] = [int(np.where(np.argsort(-P[i]) == idx[c])[0][0]) + 1
                                for i, c in enumerate(y)]

    # ================= STEP 1 =================
    rule("1. ERROR INVENTORY  (log loss is primary; accuracy is secondary)")
    for n, f in folds.items():
        d, r = f["df"], f["r"]
        print(f"\n  {n}: n={len(d)} correct={int(d.correct.sum())} wrong={int((~d.correct).sum())} "
              f"accuracy={r.accuracy:.6f}")
        print(f"    log_loss={r.log_loss:.16f} brier={r.brier:.8f} "
              f"macro_f1={r.macro_f1:.6f} balanced_acc={r.balanced_accuracy:.6f}")
        tab = pd.crosstab(d.y_true, d.y_pred).reindex(index=CLASS_ORDER, columns=CLASS_ORDER,
                                                      fill_value=0)
        print("    transition counts (rows=true, cols=pred) and % of fold:")
        for t in CLASS_ORDER:
            cells = "  ".join(f"{t}->{p}: {int(tab.loc[t, p]):5d} ({tab.loc[t, p] / len(d) * 100:5.2f}%)"
                              for p in CLASS_ORDER)
            print(f"      {cells}")

    # ================= STEP 2 =================
    rule("2. CONFIDENCE vs ERROR")
    for n, f in folds.items():
        d = f["df"]
        print(f"\n  {n}:")
        for lab, g in (("correct", d[d.correct]), ("wrong", d[~d.correct])):
            q = quantiles(g.p_pred, [0.5, 0.75, 0.90, 0.95])
            print(f"    {lab:<8} n={len(g):5d} mean_conf={g.p_pred.mean():.6f} median={q[0]:.6f} "
                  f"p75={q[1]:.6f} p90={q[2]:.6f} p95={q[3]:.6f} max={g.p_pred.max():.6f}")
            print(f"             mean_p_true={g.p_true.mean():.6f} mean_margin={g.margin.mean():.6f}")
        hi = d.p_pred >= 0.60      # pre-declared split point, not tuned
        print(f"    A confident-correct (p_pred>=0.60): {int((hi & d.correct).sum())}")
        print(f"    B uncertain-correct (p_pred< 0.60): {int((~hi & d.correct).sum())}")
        print(f"    C confident-wrong   (p_pred>=0.60): {int((hi & ~d.correct).sum())}")
        print(f"    D uncertain-wrong   (p_pred< 0.60): {int((~hi & ~d.correct).sum())}")
    print("\n  NOTE: confident-wrong predictions are NOT assumed fixable. They indicate")
    print("  uncertainty, misspecification, noise or label ambiguity -- undetermined here.")

    # ================= STEP 3 =================
    rule("3. LOG-LOSS DECOMPOSITION BY CONFIDENCE BAND (wrong predictions)")
    bands = [(0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.0)]
    for n, f in folds.items():
        d = f["df"]; total_nll = d.nll.sum()
        print(f"\n  {n}: total NLL={total_nll:.4f} (mean log loss={d.nll.mean():.16f})")
        print(f"    {'band':<14}{'n':>7}{'accuracy':>11}{'mean_p_true':>14}"
              f"{'mean_NLL':>11}{'share_of_total_NLL':>21}")
        for lo, hi in bands:
            m = (d.p_pred >= lo) & (d.p_pred < hi if hi < 1.0 else d.p_pred <= 1.0)
            g = d[m]
            if len(g) == 0:
                continue
            print(f"    {f'{lo:.2f}-{hi:.2f}':<14}{len(g):>7}{g.correct.mean():>11.6f}"
                  f"{g.p_true.mean():>14.6f}{g.nll.mean():>11.6f}"
                  f"{g.nll.sum() / total_nll:>21.6f}")
        w = d[~d.correct]
        print(f"    wrong-only NLL share of fold total: {w.nll.sum() / total_nll:.6f} "
              f"({len(w)} rows, {len(w) / len(d) * 100:.2f}% of fold)")
        top = w.nll.sort_values(ascending=False)
        for k in (10, 50, 100):
            print(f"      worst {k:3d} wrong rows contribute "
                  f"{top.head(k).sum() / total_nll:.6f} of total NLL")

    # ================= STEP 4 =================
    rule("4. CALIBRATION DIAGNOSTIC (no calibration fitted, no prediction changed)")
    buckets = [(0.33, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70),
               (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]
    for n, f in folds.items():
        d = f["df"]
        print(f"\n  {n}: {'bucket':<14}{'n':>7}{'mean_conf':>12}{'accuracy':>11}{'conf-acc':>11}")
        for lo, hi in buckets:
            m = (d.p_pred >= lo) & (d.p_pred < hi if hi < 1.0 else d.p_pred <= 1.0)
            g = d[m]
            if len(g) < 20:
                if len(g):
                    print(f"          {f'{lo:.2f}-{hi:.2f}':<14}{len(g):>7}"
                          f"{'':>12}{'':>11}   (n<20, not reported)")
                continue
            print(f"          {f'{lo:.2f}-{hi:.2f}':<14}{len(g):>7}{g.p_pred.mean():>12.6f}"
                  f"{g.correct.mean():>11.6f}{g.p_pred.mean() - g.correct.mean():>+11.6f}")
        overall = d.p_pred.mean() - d.correct.mean()
        print(f"          OVERALL mean_conf - accuracy = {overall:+.6f} "
              f"({'overconfident' if overall > 0 else 'underconfident'})")
    print("\n  Per-class buckets are reported only where n>=20 in the bucket.")
    for n, f in folds.items():
        d = f["df"]
        for cls in CLASS_ORDER:
            g = d[d.y_pred == cls]
            if len(g) < 50:
                print(f"  {n} predicted-{cls}: n={len(g)} -- too few for a meaningful split")
                continue
            print(f"  {n} predicted-{cls}: n={len(g)} mean_conf={g.p_pred.mean():.6f} "
                  f"accuracy={g.correct.mean():.6f} diff={g.p_pred.mean() - g.correct.mean():+.6f}")

    # ================= STEP 5 =================
    rule("5. CLASS-SPECIFIC ERROR STRUCTURE (Draw emphasised)")
    for n, f in folds.items():
        d = f["df"]
        print(f"\n  {n}:")
        for cls in CLASS_ORDER:
            g = d[d.y_true == cls]
            recall = float((g.y_pred == cls).mean()) if len(g) else float("nan")
            wrong = Counter(g[g.y_pred != cls].y_pred)
            print(f"    actual {cls}: n={len(g):5d} recall={recall:.6f} "
                  f"mean_p_true={g.p_true.mean():.6f} mean_max_p={g.p_pred.mean():.6f} "
                  f"mean_NLL={g.nll.mean():.6f} mispredicted_as={dict(wrong)}")
        pd_ = d.p_D
        print(f"    DRAW structure: predicted-D count={int((d.y_pred == 'D').sum())} "
              f"({float((d.y_pred == 'D').mean()):.6f}) actual rate={float((d.y_true == 'D').mean()):.6f}")
        print(f"      p_draw mean={pd_.mean():.6f} max={pd_.max():.6f} "
              f"p99={np.quantile(pd_, 0.99):.6f}")
        need = d.p_pred - pd_
        print(f"      gap D must close to become argmax: mean={need.mean():.6f} "
              f"median={need.median():.6f} | within 0.05: {int((need < 0.05).sum())}")
        td = d[d.y_true == "D"]; nd = d[d.y_true != "D"]
        print(f"      separation: mean p_draw | true-D={td.p_D.mean():.6f} "
              f"non-D={nd.p_D.mean():.6f} delta={td.p_D.mean() - nd.p_D.mean():+.6f}")
        rk = Counter(td.true_rank)
        print(f"      among true draws, rank of D: {dict(sorted(rk.items()))}")
    print("\n  A/B/C/D attribution for Draw is stated in STEP 12 from the numbers above.")
    print("  The argmax rule is NOT changed anywhere in this diagnostic.")

    # ================= STEP 6 =================
    rule("6. TEMPORAL DRIFT ACROSS THE THREE FOLDS")
    print(f"  {'fold':<9}{'log_loss':>20}{'brier':>12}{'accuracy':>11}"
          f"{'mean_p_true':>13}{'mean_conf':>11}{'wrong_rate':>12}")
    for n in sorted(folds):
        d, r = folds[n]["df"], folds[n]["r"]
        print(f"  {n:<9}{r.log_loss:>20.16f}{r.brier:>12.6f}{r.accuracy:>11.6f}"
              f"{d.p_true.mean():>13.6f}{d.p_pred.mean():>11.6f}{1 - d.correct.mean():>12.6f}")
    lls = [folds[n]["r"].log_loss for n in sorted(folds)]
    mono = (lls[0] < lls[1] < lls[2]) or (lls[0] > lls[1] > lls[2])
    print(f"\n  monotone across the three folds: {mono}")
    print("  THREE folds cannot establish a replicated temporal trend: a monotone ordering")
    print("  of three points occurs by chance with probability 1/3. NOT A FINDING either way.")

    # ================= STEP 7 =================
    rule("7. FEATURE-REGIME ERROR ANALYSIS (training-derived bins, replication required)")
    print(f"  Screening {len(NUMERIC_COLUMNS)} numeric features x 3 folds.")
    print("  Bin edges come from the TRAINING partition only -- validation labels never used.")
    print("  No significance testing is performed, so no p-value selection is possible.")
    print(f"  A pattern is a FINDING only if it appears in >= {MIN_FOLDS} of 3 folds with the")
    print(f"  same direction and >= {MIN_BIN_N} rows per extreme bin.\n")
    per_feature = defaultdict(list)
    for n, f in folds.items():
        d, Xva, Xtr = f["df"], f["Xva"], f["Xtr"]
        for col in NUMERIC_COLUMNS:
            tr = Xtr[col].dropna()
            if len(tr) < 200:
                continue
            edges = np.unique(np.nanquantile(tr, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
            if len(edges) < 3:
                continue
            b = pd.cut(Xva[col], bins=edges, include_lowest=True, labels=False)
            g = d.assign(bin=b.to_numpy()).dropna(subset=["bin"])
            lo, hi = g[g.bin == g.bin.min()], g[g.bin == g.bin.max()]
            if len(lo) < MIN_BIN_N or len(hi) < MIN_BIN_N:
                continue
            per_feature[col].append(dict(fold=n, n_lo=len(lo), n_hi=len(hi),
                                         nll_lo=lo.nll.mean(), nll_hi=hi.nll.mean(),
                                         acc_lo=lo.correct.mean(), acc_hi=hi.correct.mean(),
                                         ptrue_lo=lo.p_true.mean(), ptrue_hi=hi.p_true.mean(),
                                         d_nll=hi.nll.mean() - lo.nll.mean()))
    findings = []
    for col, recs in per_feature.items():
        if len(recs) < MIN_FOLDS:
            continue
        signs = {np.sign(r["d_nll"]) for r in recs}
        if len(signs) == 1 and 0 not in signs:
            spread = float(np.mean([abs(r["d_nll"]) for r in recs]))
            findings.append((col, len(recs), float(np.sign(recs[0]["d_nll"])), spread))
    findings.sort(key=lambda x: -x[3])
    print(f"  features screened with adequate bins in >=1 fold : {len(per_feature)}")
    print(f"  features with consistent NLL direction in >={MIN_FOLDS} folds: {len(findings)}")
    print(f"\n  top 15 by mean |NLL(high bin) - NLL(low bin)|:")
    print(f"    {'feature':<44}{'folds':>6}{'dir':>5}{'mean|dNLL|':>12}")
    for col, k, sgn, spread in findings[:15]:
        print(f"    {col:<44}{k:>6}{'+' if sgn > 0 else '-':>5}{spread:>12.6f}")
    print("\n  MULTIPLICITY WARNING: ~75 features were screened. With a 2-of-3 replication")
    print("  filter and no correction, some consistent directions are expected by chance.")
    print("  These are candidates for inspection, NOT findings, and NOT hypotheses.")

    # ================= STEP 8 =================
    rule("8. STRENGTH / MATCH-BALANCE REGIMES (pre-defined, training-derived)")
    print("  Regimes fixed BEFORE inspection: strong_away / balanced / strong_home,")
    print("  split at the training-partition 33rd and 67th percentiles of strength_diff.\n")
    regime_rows = []
    for n, f in folds.items():
        d, Xva, Xtr = f["df"], f["Xva"], f["Xtr"]
        lo, hi = np.nanquantile(Xtr["strength_diff"].dropna(), [1 / 3, 2 / 3])
        sd = Xva["strength_diff"].to_numpy()
        reg = np.where(sd < lo, "strong_away", np.where(sd > hi, "strong_home", "balanced"))
        g = d.assign(regime=reg)
        print(f"  {n}: boundaries from TRAINING only: lo={lo:+.6f} hi={hi:+.6f}")
        print(f"    {'regime':<14}{'n':>7}{'accuracy':>11}{'log_loss':>14}"
              f"{'mean_p_true':>13}{'wrong_rate':>12}")
        for r in ("strong_away", "balanced", "strong_home"):
            s = g[g.regime == r]
            if len(s) < MIN_BIN_N:
                print(f"    {r:<14}{len(s):>7}   (n<{MIN_BIN_N}, not reported)")
                continue
            print(f"    {r:<14}{len(s):>7}{s.correct.mean():>11.6f}{s.nll.mean():>14.6f}"
                  f"{s.p_true.mean():>13.6f}{1 - s.correct.mean():>12.6f}")
            regime_rows.append(dict(fold=n, regime=r, nll=s.nll.mean(), acc=s.correct.mean()))
    R = pd.DataFrame(regime_rows)
    if len(R):
        piv = R.pivot(index="regime", columns="fold", values="nll")
        print("\n  mean NLL by regime and fold:")
        print(piv.to_string(float_format=lambda v: f"{v:10.6f}"))
        worst = piv.idxmax()
        print(f"  worst regime per fold: {dict(worst)}")
        agree = len(set(worst.values))
        print(f"  same worst regime in all 3 folds: {agree == 1}  "
              f"({'FINDING candidate' if agree == 1 else 'NOT A FINDING -- not replicated'})")

    # ================= STEP 10 =================
    rule("10. COUNTERFACTUAL DISTANCE (existing H1 attribution methodology)")
    print("  For each wrong row: logit gap to the true class, and the single-feature move")
    print("  (in scaled SD units) that would close it, using the ACTUAL fitted coefficients.")
    print("  LIMITATION: this holds all other features fixed. Because the contract is highly")
    print("  correlated, real movement is never single-feature, so 'within 1 SD' OVERSTATES")
    print("  how fixable these rows are. Treat as an upper bound, not a repair estimate.\n")
    recurring = defaultdict(set)
    for n, f in folds.items():
        d, model, names, Eva = f["df"], f["model"], f["names"], f["Eva"]
        coef = pd.DataFrame(model.coef_, index=list(model.classes_),
                            columns=names).reindex(CLASS_ORDER)
        inter = dict(zip(list(model.classes_), np.asarray(model.intercept_)))
        wrong_idx = np.where(~d.correct.to_numpy())[0]
        dist, feats = [], []
        for i in wrong_idx:
            t, p = d.y_true.iloc[i], d.y_pred.iloc[i]
            gap = float((coef.loc[p].to_numpy() - coef.loc[t].to_numpy()) @ Eva[i]
                        + inter[p] - inter[t])
            dc = coef.loc[t].to_numpy() - coef.loc[p].to_numpy()
            j = int(np.argmax(np.abs(dc)))
            dist.append(abs(gap / dc[j]) if dc[j] != 0 else np.inf)
            feats.append(names[j])
        arr = np.array(dist, dtype=float); fin = arr[np.isfinite(arr)]
        gaps = []
        for i in wrong_idx:
            t, p = d.y_true.iloc[i], d.y_pred.iloc[i]
            gaps.append(float((coef.loc[p].to_numpy() - coef.loc[t].to_numpy()) @ Eva[i]
                              + inter[p] - inter[t]))
        print(f"  {n}: wrong={len(wrong_idx)} mean logit gap={np.mean(gaps):.6f} "
              f"median={np.median(gaps):.6f}")
        print(f"      single-feature move needed (SD): median={np.median(fin):.4f} "
              f"p90={np.percentile(fin, 90):.4f}")
        print(f"      within 1.0 SD: {float((fin <= 1).mean()):.4f} | "
              f"within 0.5 SD: {float((fin <= 0.5).mean()):.4f}")
        top = Counter(feats).most_common(8)
        print(f"      most implicated: {top}")
        for name, _ in top:
            recurring[name].add(n)

    # ================= STEP 11 =================
    rule("11. RECURRING ERROR FEATURES (>=2 folds required)")
    rec = sorted(((k, sorted(v)) for k, v in recurring.items() if len(v) >= MIN_FOLDS),
                 key=lambda x: -len(x[1]))
    if rec:
        for name, fs in rec:
            fam = next((f for f, cols in FAMILIES.items() if name in cols), "encoded/other")
            print(f"  {name:<44} folds={fs} family={fam}")
    else:
        print("  none recur in >=2 folds")
    print("\n  This is NOT permission to create a hypothesis. Recurrence in a counterfactual")
    print("  ranking reflects large |coef| difference, which is partly a property of the fit,")
    print("  not necessarily of exploitable signal.")

    # ================= STEP 12 =================
    rule("12. MODEL-CEILING CLASSIFICATION")
    print("  Categories are asserted from the numbers above, each with supporting and")
    print("  contradicting folds. Evidence rules: no single-fold or pooled-only finding.\n")
    lowconf = {}
    for n, f in folds.items():
        d = f["df"]; w = d[~d.correct]
        lowconf[n] = float((w.p_pred < 0.60).mean())
        share = w[w.p_pred >= 0.60].nll.sum() / d.nll.sum()
        print(f"  {n}: wrong rows with conf<0.60 = {lowconf[n]:.4f} | "
              f"confident-wrong share of total NLL = {share:.4f}")
    print(f"\n  A (information-limited)   : supported if most wrong rows are low-confidence")
    print(f"  B (model-form limited)    : requires a REPLICATED regime with high-confidence errors")
    print(f"  C (calibration limited)   : requires a consistent conf-accuracy gap across folds")
    print(f"  D (class-separation)      : Draw evidence from STEP 5")
    print(f"  E (temporal)              : NOT ESTABLISHABLE from three folds (see STEP 6)")
    print(f"  F (no actionable limit)   : default if nothing replicates")
    print("\n  Ranking is stated in the written report from these numbers; the script does not")
    print("  auto-assign a category, because category choice requires judgement about")
    print("  replication that should be visible and auditable rather than hidden in code.")

    rule("13-15. STATEMENTS AND INTEGRITY AUDIT")
    print("  No model or feature change is authorized by this diagnostic.")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r} | H1 C={C_VALUE} | folds={sorted(folds)}")
    for line in ("2025/26 accessed = NO", "production files modified = NO",
                 "features.db modified = NO (read-only)", "tuning = NONE",
                 "calibration = NONE", "feature selection = NONE",
                 "threshold change = NONE", "class weights = NONE",
                 "artifacts = NONE", "estimator persisted = NONE",
                 "files written by this script = NONE"):
        print(f"  {line}")


if __name__ == "__main__":
    main()
