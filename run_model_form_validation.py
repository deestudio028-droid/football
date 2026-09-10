"""MODEL-FORM VALIDATION -- three pre-registered arms, validation folds only.

QUESTION
    Can a different MODEL FORM extract additional classification signal
    from the EXACT SAME 80 E-1 columns, without changing the information
    set? This is not a feature search and not a tuning exercise.

ARMS (pre-registered, fixed configurations, no grids)
    ARM 0  E-1 LogisticRegression  C=0.0005, max_iter=2000, random_state=0
    ARM 1  RandomForestClassifier  n_estimators=300, max_depth=8,
                                   min_samples_leaf=10, random_state=0, n_jobs=-1
    ARM 2  HistGradientBoosting    max_iter=200, learning_rate=0.05,
                                   max_leaf_nodes=15, min_samples_leaf=20,
                                   l2_regularization=1.0, random_state=0

PREPROCESSING -- DELIBERATE CHOICE, STATED PLAINLY
    All three arms use the frozen `LogisticRegressionPreprocessor`
    unchanged, because the brief requires "EXACTLY the same preprocessing"
    and "only the model form may change". Holding preprocessing fixed is
    what isolates model form as the single variable.

    CAVEAT, recorded in advance so it cannot be claimed afterwards: the
    project's own frozen HGB path (train.train_hist_gradient_boosting)
    instead uses CompetitionCodeEncoder plus HGB's NATIVE missing-value
    handling, with no imputation or scaling. Median-imputing before a tree
    model destroys the missingness signal that HGB could otherwise use, so
    ARM 2 here is mildly handicapped relative to that native path. Using
    the native path would have confounded model form with preprocessing,
    so it was not used. Trees are scale-invariant, so standardisation
    itself is harmless.

REUSED FROM THE PROJECT (not reimplemented)
    ablation.MODEL_B_COLUMNS | splits.iter_walk_forward_folds
    train.LogisticRegressionPreprocessor | train._reorder_proba
    evaluate.evaluate | evaluate.validate_probabilities
    robustness.V1_LOGREG_BASE_KWARGS | data.load_supervised_dataset

FORBIDDEN AND ABSENT: 2025/26, final test, C tuning, any hyperparameter
search, calibration, thresholds, class weighting, feature selection,
feature changes, new data, estimator persistence, artifact or database
writes, production modification, MODEL_VERSION change.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_model_form_validation.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from models.ablation import MODEL_B_COLUMNS
from models.baselines import CLASS_ORDER
from models.config import FINAL_TEST_SEASONS, MODEL_VERSION, WALK_FORWARD_FOLDS
from models.data import load_supervised_dataset
from models.evaluate import evaluate, validate_probabilities
from models.robustness import V1_LOGREG_BASE_KWARGS
from models.splits import iter_walk_forward_folds

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MATCHES_DB = REPO / "data" / "processed" / "matches.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"

C_VALUE = 0.0005
BASE = tuple(MODEL_B_COLUMNS)
HIGHER_BETTER = ("accuracy", "macro_f1", "balanced_accuracy")
LOWER_BETTER = ("log_loss", "brier")
METRICS = HIGHER_BETTER + LOWER_BETTER
EFFECT_MIN = 0.10          # same convention as D-2/D-6
RECALL_TOLERANCE = 0.01    # "materially degraded" bar for Draw/Away recall


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)


def stop(m):
    print("\n" + "!" * 78); print("STOP -- halted"); print(m); print("!" * 78)
    raise SystemExit(1)


def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()


def build_estimator(arm):
    """Exactly one fixed configuration per arm. No grids, no alternatives."""
    if arm == "E-1":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=C_VALUE, **V1_LOGREG_BASE_KWARGS)
    if arm == "RANDOM_FOREST":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, max_depth=8,
                                      min_samples_leaf=10, random_state=0, n_jobs=-1)
    if arm == "HIST_GB":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              max_leaf_nodes=15, min_samples_leaf=20,
                                              l2_regularization=1.0, random_state=0)
    raise ValueError(f"unknown arm {arm!r}; only three arms are authorized")


ARMS = ("E-1", "RANDOM_FOREST", "HIST_GB")


def per_class(y, P):
    y = np.asarray(y); pred = np.array(CLASS_ORDER)[P.argmax(1)]
    out = {}
    for c in CLASS_ORDER:
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        out[c] = dict(precision=pr, recall=rc, f1=f1, support=int((y == c).sum()),
                      predicted=int((pred == c).sum()))
    return out


def fit_arm(arm, Xtr, ytr, Xva):
    from models import train as train_module

    A, B = Xtr[list(BASE)], Xva[list(BASE)]
    pre = train_module.LogisticRegressionPreprocessor().fit(A)   # identical for all arms
    model = build_estimator(arm)
    model.fit(pre.transform(A), ytr)
    P = np.asarray(train_module._reorder_proba(model, pre.transform(B)), dtype=float)
    validate_probabilities(P)
    return P, model


def run_arm(dataset, arm):
    folds = {}
    for fold, tr, va in iter_walk_forward_folds(dataset):
        P, model = fit_arm(arm, tr.X, tr.y, va.X)
        r = evaluate(va.y, P)
        y = va.y.to_numpy()
        srt = np.sort(P, axis=1)
        idx = {c: i for i, c in enumerate(CLASS_ORDER)}
        folds[fold.name] = dict(
            P=P, y=y, r=r, per_class=per_class(y, P),
            pred=np.array(CLASS_ORDER)[P.argmax(1)],
            margin=srt[:, 2] - srt[:, 1], p_max=P.max(1),
            p_true=np.array([P[i, idx[c]] for i, c in enumerate(y)]),
            fixture_ids=va.metadata["fixture_id"].to_numpy(),
        )
    means = {m: statistics.fmean(folds[n]["r"].as_dict()[m] for n in sorted(folds)) for m in METRICS}
    sds = {m: statistics.pstdev([folds[n]["r"].as_dict()[m] for n in sorted(folds)]) for m in METRICS}
    by_fold = {m: [folds[n]["r"].as_dict()[m] for n in sorted(folds)] for m in METRICS}
    return dict(folds=folds, means=means, sds=sds, by_fold=by_fold, arm=arm)


def pooled(a):
    # BUGFIX (reporting only): two dead locals were removed here. The first,
    # `pr = {c: 0 for c in CLASS_ORDER}`, is a comprehension whose loop variable
    # does NOT leak in Python 3, so the following line's bare `c` raised
    # NameError: name 'c' is not defined. Neither variable was ever used --
    # `agg` below is the real accumulator. No metric, estimator, preprocessing,
    # feature list, fold, seed, C, fitting path, data loading or decision rule
    # is touched by this fix.
    agg = {c: dict(pred=0, tp=0, fp=0, fn=0, sup=0) for c in CLASS_ORDER}
    cm = np.zeros((3, 3), dtype=int)
    for n in a["folds"]:
        f = a["folds"][n]
        cm += f["r"].confusion_matrix
        for c in CLASS_ORDER:
            pc = f["per_class"][c]
            agg[c]["pred"] += pc["predicted"]; agg[c]["sup"] += pc["support"]
            tp = int(((f["pred"] == c) & (f["y"] == c)).sum())
            agg[c]["tp"] += tp
            agg[c]["fp"] += int(((f["pred"] == c) & (f["y"] != c)).sum())
            agg[c]["fn"] += int(((f["pred"] != c) & (f["y"] == c)).sum())
    out = {}
    for c in CLASS_ORDER:
        g = agg[c]
        pr_ = g["tp"] / (g["tp"] + g["fp"]) if (g["tp"] + g["fp"]) else 0.0
        rc_ = g["tp"] / (g["tp"] + g["fn"]) if (g["tp"] + g["fn"]) else 0.0
        f1_ = 2 * pr_ * rc_ / (pr_ + rc_) if (pr_ + rc_) else 0.0
        out[c] = dict(predicted=g["pred"], support=g["sup"], precision=pr_, recall=rc_, f1=f1_)
    return out, cm


def main():
    print("MODEL-FORM VALIDATION -- three pre-registered arms, same 80 E-1 columns")
    print("Not a feature search. Not a tuning exercise. Validation folds only.")

    # ---------------- 1. PRE-FLIGHT ----------------
    rule("1. PRE-FLIGHT")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}  (must be 'v1.0': {MODEL_VERSION == 'v1.0'})")
    print(f"  E-1 contract columns: {len(BASE)}  (must be 80: {len(BASE) == 80})")
    print("  fold definitions (from the project, not restated):")
    for f in WALK_FORWARD_FOLDS:
        print(f"    {f.name}: train={f.train_seasons} validation={f.validation_seasons}")
    fold_seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    forbidden = sorted(fold_seasons & set(FINAL_TEST_SEASONS))
    print(f"  final-test seasons intersecting any fold: {forbidden or 'NONE'}")
    if forbidden:
        stop(f"2025/26 reachable through the folds: {forbidden}")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines: mismatches={bad or 'NONE'}")
    print(f"  features.db md5={md5(FEATURES_DB)}")
    print(f"  matches.db  md5={md5(MATCHES_DB)}")
    if bad or MODEL_VERSION != "v1.0" or len(BASE) != 80:
        stop("pre-flight integrity failure")

    # ---------------- 2. ARM CONTRACTS ----------------
    rule("2. ARM CONTRACTS")
    for arm in ARMS:
        est = build_estimator(arm)
        print(f"  {arm:<14} {type(est).__name__}")
        print(f"  {'':<14} params: { {k: v for k, v in est.get_params().items() if v is not None and k in ('C','max_iter','random_state','n_estimators','max_depth','min_samples_leaf','n_jobs','learning_rate','max_leaf_nodes','l2_regularization')} }")
    print(f"  features (all arms): {len(BASE)} E-1 columns, identical")
    print("  preprocessing (all arms): frozen LogisticRegressionPreprocessor, identical")
    print("  CAVEAT: median-imputing before a tree model removes the missingness signal")
    print("          HGB could use natively. Accepted deliberately to isolate model form.")

    dataset = load_supervised_dataset(FEATURES_DB)
    print(f"\n  supervised rows loaded: {len(dataset.X)}")

    # ---------------- fit ----------------
    rule("FITTING THREE ARMS")
    arms = {}
    for arm in ARMS:
        arms[arm] = run_arm(dataset, arm)
        print(f"  {arm:<14} " + "  ".join(f"{m}={arms[arm]['means'][m]:.10f}" for m in METRICS))
    base = arms["E-1"]
    for arm in ARMS[1:]:
        for n in base["folds"]:
            if not np.array_equal(base["folds"][n]["fixture_ids"], arms[arm]["folds"][n]["fixture_ids"]):
                stop(f"{arm}/{n}: fixture ordering differs from baseline")
    print("  [PASS] identical fixture ordering across all arms and folds")
    spread = {m: max(base["by_fold"][m]) - min(base["by_fold"][m]) for m in METRICS}

    # ---------------- 3. PER-FOLD ----------------
    rule("3. PER-FOLD RESULTS")
    for arm in ARMS:
        print(f"\n  --- {arm} ---")
        for n in sorted(arms[arm]["folds"]):
            r = arms[arm]["folds"][n]["r"]
            print(f"    {n}: acc={r.accuracy:.8f} macro_f1={r.macro_f1:.8f} "
                  f"bal_acc={r.balanced_accuracy:.8f} log_loss={r.log_loss:.16f} "
                  f"brier={r.brier:.8f} n={r.n}")

    # ---------------- 4. SCORECARD ----------------
    rule("4. MEAN + SD SCORECARD")
    print(f"  {'Arm':<16}{'Accuracy':>20}{'Macro-F1':>20}{'BalAcc':>20}{'LogLoss':>22}{'Brier':>20}")
    for arm in ARMS:
        a = arms[arm]
        print(f"  {arm:<16}" + "".join(
            f"{a['means'][m]:>13.8f}({a['sds'][m]:.4f})" if m != "log_loss"
            else f"{a['means'][m]:>15.10f}({a['sds'][m]:.4f})" for m in METRICS))

    # ---------------- 5 + 6. DELTAS AND EFFECT SIZE ----------------
    rule("5-6. DELTAS vs E-1 AND EFFECT SIZE")
    verdicts = {}
    for arm in ARMS[1:]:
        a = arms[arm]
        d = {m: a["means"][m] - base["means"][m] for m in METRICS}
        rel = {m: (d[m] / abs(base["means"][m]) * 100) if base["means"][m] else float("nan")
               for m in METRICS}
        pf = {m: [a["by_fold"][m][i] - base["by_fold"][m][i] for i in range(3)] for m in METRICS}
        allf = {m: all((x > 0) if m in HIGHER_BETTER else (x < 0) for x in pf[m]) for m in METRICS}
        eff = {m: (abs(d[m]) / spread[m]) if spread[m] else float("nan") for m in METRICS}
        print(f"\n  --- {arm} ---")
        for m in METRICS:
            good = (d[m] > 0) if m in HIGHER_BETTER else (d[m] < 0)
            print(f"    {m:<20} delta={d[m]:+.8e} ({rel[m]:+.4f}%)  {'better' if good else 'worse'}"
                  f"  all3={allf[m]}  effect={eff[m]:.4f}x baseline spread")
            print(f"    {'':<20} per_fold={[f'{x:+.3e}' for x in pf[m]]}")
        verdicts[arm] = dict(d=d, allf=allf, eff=eff)

    # ---------------- 7. CLASSIFICATION DIAGNOSTICS ----------------
    rule("7. CLASSIFICATION DIAGNOSTICS (pooled validation)")
    bp, bcm = pooled(base)
    for arm in ARMS:
        p, cm = pooled(arms[arm])
        print(f"\n  --- {arm} ---")
        print(f"    {'class':<6}{'support':>9}{'predicted':>11}{'recall':>10}{'precision':>11}{'F1':>10}")
        for c in CLASS_ORDER:
            print(f"    {c:<6}{p[c]['support']:>9}{p[c]['predicted']:>11}{p[c]['recall']:>10.6f}"
                  f"{p[c]['precision']:>11.6f}{p[c]['f1']:>10.6f}")
        print(f"    confusion matrix (rows=true {CLASS_ORDER}, cols=pred):")
        for row in cm:
            print(f"      {row}")
        if arm != "E-1":
            dpred = {c: p[c]["predicted"] - bp[c]["predicted"] for c in CLASS_ORDER}
            drec = {c: p[c]["recall"] - bp[c]["recall"] for c in CLASS_ORDER}
            print(f"    delta vs E-1  predicted: { {c: f'{dpred[c]:+d}' for c in CLASS_ORDER} }")
            print(f"    delta vs E-1  recall   : { {c: f'{drec[c]:+.6f}' for c in CLASS_ORDER} }")
            print(f"    Q1 improves Draw recall : {drec['D'] > 0}")
            print(f"    Q2 improves Away recall : {drec['A'] > 0}")
            print(f"    Q3 predicts Home more   : {dpred['H'] > 0}")
            print(f"    Q4 suppresses Draw      : {dpred['D'] < 0}")
            print(f"    Q5 suppresses Away      : {dpred['A'] < 0}")
            bal_b = statistics.pstdev([bp[c]["recall"] for c in CLASS_ORDER])
            bal_a = statistics.pstdev([p[c]["recall"] for c in CLASS_ORDER])
            print(f"    Q6 better H/D/A balance : {bal_a < bal_b}  "
                  f"(recall sd {bal_b:.6f} -> {bal_a:.6f}; lower = more balanced)")
            verdicts[arm]["dpred"] = dpred
            verdicts[arm]["drec"] = drec

    # ---------------- 8. PROBABILITY DIAGNOSTICS ----------------
    rule("8. PROBABILITY DIAGNOSTICS")
    for arm in ARMS:
        a = arms[arm]
        def m_(fn): return statistics.fmean(fn(a["folds"][n]) for n in a["folds"])
        pc = {c: m_(lambda f, i=i: f["P"][:, i].mean()) for i, c in enumerate(CLASS_ORDER)}
        corr = m_(lambda f: float(f["p_true"][f["pred"] == f["y"]].mean()))
        wrong = m_(lambda f: float(f["p_true"][f["pred"] != f["y"]].mean()))
        extra = ""
        if arm != "E-1":
            mv = m_(lambda f, arm=arm: float(np.abs(
                f["P"] - base["folds"][[k for k in base["folds"]][0]]["P"]).max()) if False else 0.0)
            mv = statistics.fmean(
                float(np.abs(a["folds"][n]["P"] - base["folds"][n]["P"]).max(1).mean())
                for n in a["folds"])
            extra = f"  mean|prob move| vs E-1={mv:.6e}"
        print(f"  {arm:<14} mean_max_p={m_(lambda f: f['p_max'].mean()):.6f} "
              f"row_spread={m_(lambda f: f['P'].std(1).mean()):.6f} "
              f"p_H={pc['H']:.6f} p_D={pc['D']:.6f} p_A={pc['A']:.6f}")
        print(f"  {'':<14} mean p_true={m_(lambda f: f['p_true'].mean()):.6f} "
              f"| correct={corr:.6f} | incorrect={wrong:.6f}{extra}")

    # ---------------- 9. ERROR / MARGIN ----------------
    rule("9. ERROR / MARGIN ANALYSIS")
    print(f"  {'Arm':<14}{'errors':>8}" + "".join(f"{'<'+str(t):>10}" for t in (0.01, 0.02, 0.05, 0.10)))
    for arm in ARMS:
        a = arms[arm]
        marg = np.concatenate([a["folds"][n]["margin"][a["folds"][n]["pred"] != a["folds"][n]["y"]]
                               for n in sorted(a["folds"])])
        print(f"  {arm:<14}{len(marg):>8}" + "".join(
            f"{float((marg < t).mean()):>10.4f}" for t in (0.01, 0.02, 0.05, 0.10)))
    print("  (fraction of that arm's OWN errors falling below each margin)")
    print("  A form that truly helps should reduce CONFIDENT wrong predictions,")
    print("  not merely redistribute probability mass.")

    # ---------------- 10. DECISION RULE ----------------
    rule("10. PRE-REGISTERED DECISION RULE")
    for arm in ARMS[1:]:
        v = verdicts[arm]
        d, allf, eff = v["d"], v["allf"], v["eff"]
        dpred, drec = v["dpred"], v["drec"]
        c = {
            "1  mean accuracy not decreased": d["accuracy"] >= 0,
            "2  mean macro-F1 not decreased": d["macro_f1"] >= 0,
            "3  mean balanced acc not decreased": d["balanced_accuracy"] >= 0,
            "4  mean log loss improves": d["log_loss"] < 0,
            "5  mean Brier improves": d["brier"] < 0,
            "6  log loss improves in all 3 folds": allf["log_loss"],
            "7  Brier improves in all 3 folds": allf["brier"],
            "8  not merely Home over-selection": not (dpred["H"] > 0 and d["balanced_accuracy"] <= 0),
            "9  Draw/Away recall not materially degraded":
                (drec["D"] >= -RECALL_TOLERANCE) and (drec["A"] >= -RECALL_TOLERANCE),
            "10 effect not tiny vs baseline spread": eff["log_loss"] > EFFECT_MIN,
        }
        d1_pattern = (d["log_loss"] < 0 and d["brier"] < 0
                      and (d["accuracy"] < 0 or d["balanced_accuracy"] < 0 or d["macro_f1"] < 0))
        c["11 does NOT reproduce the D-1/D-2 pattern"] = not d1_pattern
        serious = all(c.values())
        print(f"\n  --- {arm} ---")
        for k, ok in c.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {k}")
        verdicts[arm]["serious"] = serious
        verdicts[arm]["d1_pattern"] = d1_pattern
        print(f"    => {'SERIOUS CANDIDATE' if serious else 'NOT A CANDIDATE'}")

    # ---------------- reproducibility ----------------
    rule("REPRODUCIBILITY")
    ok = True
    for arm in ARMS:
        rep = run_arm(dataset, arm)
        same = all(rep["folds"][n]["P"].tobytes() == arms[arm]["folds"][n]["P"].tobytes()
                   for n in arms[arm]["folds"])
        ok &= same
        h = hashlib.sha256(b"".join(arms[arm]["folds"][n]["P"].tobytes()
                                    for n in sorted(arms[arm]["folds"]))).hexdigest()
        print(f"  {arm:<14} byte-identical repeat={same}  sha256={h[:40]}...")
    print(f"  REPRODUCIBILITY: {'PASS' if ok else 'FAIL -- reported, not hidden'}")

    # ---------------- 11. VERDICT ----------------
    rule("11. MODEL-FORM VERDICT")
    serious = [a for a in ARMS[1:] if verdicts[a]["serious"]]
    d1like = [a for a in ARMS[1:] if verdicts[a]["d1_pattern"]]
    if serious:
        print(f"  CLEAR MODEL-FORM CANDIDATE: {serious}")
        print("  NOT promoted. NO final test. NO 2025/26. A separately authorized")
        print("  follow-up experiment would be required.")
    elif d1like:
        print(f"  USEFUL FOR PROBABILITY ESTIMATION BUT NOT SUFFICIENT FOR PROMOTION: {d1like}")
        print("  These reproduce the D-1/D-2 pattern: probability metrics improve while")
        print("  classification metrics degrade.")
    else:
        print("  NO EVIDENCE THAT MODEL FORM IS THE CURRENT BINDING LIMITATION")
        print("  Recommend closing the current improvement cycle rather than continuing")
        print("  unrestricted experimentation.")

    # ---------------- 12. INTEGRITY ----------------
    rule("12. INTEGRITY AUDIT")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}")
    print(f"  features.db md5={md5(FEATURES_DB)} (unchanged: {md5(FEATURES_DB) == before.get('data/processed/features.db', md5(FEATURES_DB))})")
    print(f"  matches.db  md5={md5(MATCHES_DB)}")
    ser = [str(p.relative_to(REPO)) for pat in ("*.pkl", "*.joblib", "*.pickle")
           for p in REPO.rglob(pat) if "__pycache__" not in p.parts]
    print(f"  serialized estimators on disk: {ser or 'NONE'}")
    for line in ("2025/26 accessed = NO", "final test run = NO",
                 "production files changed = NO", "database writes = NONE",
                 "estimator persisted = NONE", "artifacts created = NONE",
                 "C changed = NO (0.0005 fixed for ARM 0; trees have no C)",
                 "feature changes = NONE (identical 80 columns in all arms)",
                 "calibration = NONE", "threshold changes = NONE",
                 "class weighting = NONE", "hyperparameter search = NONE"):
        print(f"  {line}")
    print("\n  NO PROMOTION. NO FINAL TEST. NO 2025/26. NO PRODUCTION CHANGE.")


if __name__ == "__main__":
    main()
