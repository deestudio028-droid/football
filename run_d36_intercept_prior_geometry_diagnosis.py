"""D-36 -- multiclass intercept / class-prior geometry investigation.

    python run_d36_intercept_prior_geometry_diagnosis.py

READ-ONLY DIAGNOSTIC. NOTHING IS FIXED, TUNED, REBALANCED OR PERSISTED.

D-35 found the intercept term accounts for 76-113% of the pairwise score
gaps. D-36 asks WHY, and whether that actually supports CASE B.

TWO MATHEMATICAL CAVEATS THAT GOVERN EVERY NUMBER BELOW
--------------------------------------------------------
(1) ABSOLUTE INTERCEPTS ARE NOT IDENTIFIABLE. Multinomial softmax is
    invariant to adding the same constant to all three class scores:
    softmax(s + c) == softmax(s). Only the CONTRASTS (H-D, H-A, D-A)
    are identified. Any statement about intercept_H on its own is
    meaningless, and this script never makes one.

(2) THE INTERCEPT DOES NOT SIT AT A REAL POINT IN FEATURE SPACE.
    `LogisticRegressionPreprocessor` standardises the 79 numeric columns
    (so their zero IS the training mean) but the 5 competition_id
    one-hot columns are appended UNSCALED as 0/1. The model's zero point
    is therefore "numeric features at their training mean AND belonging
    to no league at all" -- a fixture that cannot exist, since every row
    has exactly one league flag set.

    Consequently the bare intercept is the wrong quantity to compare
    against a class prior. The identified per-league offset is
        intercept_c + coef_c[that league]
    and this script reports BOTH so the difference is visible rather
    than assumed. D-35's "intercept share" figures inherit this caveat.

WHAT V1 ACTUALLY IS (verified from source, not memory)
    LogisticRegression(max_iter=2000, C=1.0, random_state=0)
    Every other parameter is the sklearn default. Confirmed by AST
    inspection of src/models/train.py: no class_weight, no sample_weight,
    no solver override, no penalty override, no fit_intercept override,
    and no post-fit manipulation of coef_ or intercept_ anywhere in
    src/models/. STEP 6 prints the resolved values from the fitted
    estimator rather than trusting any of this prose.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LEAGUES = {423: "Premier League", 477: "Bundesliga", 419: "La Liga",
           200: "Ligue 1", 499: "Serie A"}
POOLED_REF_LOG_LOSS = 0.9993791056968738
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
FORBIDDEN_SEASONS = [602681, 667780, 725788, 725793, 762170]


def rule(t): print("\n" + "=" * 112); print(t); print("=" * 112)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 112)
    print("HARD FAIL -- D-36 halted. No diagnosis is produced.")
    print(msg)
    print("!" * 112)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        stop(f"stop condition: {label}")


def snapshot():
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    lg = REPO / "data/processed/leagues"
    if lg.exists():
        for p in sorted(lg.glob("*.db")):
            snap[f"leagues/{p.name}"] = md5(p)
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["artifact"] = md5(art)
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def main():
    rule("STEP 1 -- ENVIRONMENT + INTEGRITY")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")
    try:
        import sklearn
        print(f"  sklearn {sklearn.__version__}")
    except ImportError as exc:
        stop(f"scikit-learn is required: {exc}. Do not substitute another estimator.")

    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        FINAL_TEST_SEASONS, MODEL_VERSION, REQUIRED_FEATURE_VERSION,
        RECOMMENDED_CONTEXT_FEATURE, SEASON_NAME_TO_IDS,
    )
    from models.data import load_supervised_dataset
    from models.evaluate import evaluate
    from models.splits import iter_walk_forward_folds
    from models.train import train_logistic_regression

    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(rel, before[rel] == exp, before[rel])
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    check("13/13 LOCKED_INPUTS at expected values",
          all(before[f"pin:{r}"] == v["expected"] for r, v in pins.items()))
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")
    check("contract is exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])

    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    check("forbidden season set matches the documented list",
          sorted(test_ids) == sorted(FORBIDDEN_SEASONS), str(sorted(test_ids)))
    folds = list(iter_walk_forward_folds(ds))
    check("exactly three walk-forward folds", len(folds) == 3)
    for fold, tr, va in folds:
        seen = set(tr.metadata["season_id"]) | set(va.metadata["season_id"])
        check(f"{fold.name}: no final-test season_id", not (seen & test_ids))

    rule("STEP 2 -- REPRODUCE V1")
    fits, per_fold = {}, []
    S_l, P_l, y_l, c_l, f_l = [], [], [], [], []
    for fold, tr, va in folds:
        Xtr, Xva = tr.X[list(MODEL_B_COLUMNS)], va.X[list(MODEL_B_COLUMNS)]
        model, pre, P = train_logistic_regression(Xtr, tr.y, Xva)
        r = evaluate(va.y, P)
        per_fold.append((fold.name, len(va), r))
        cls = list(model.classes_)
        order = [cls.index(c) for c in CLASS_ORDER]
        enc = pre.transform(Xva)
        S = model.decision_function(enc)[:, order]
        fits[fold.name] = dict(model=model, pre=pre, order=order, cls=cls,
                               y_train=tr.y, X_train=Xtr)
        S_l.append(S); P_l.append(P); y_l.append(va.y.to_numpy())
        c_l.append(va.X[RECOMMENDED_CONTEXT_FEATURE].to_numpy())
        f_l.append(np.repeat(fold.name, len(va)))
        print(f"  {fold.name}: N={len(va)} log_loss={r.log_loss:.16f} brier={r.brier:.16f}")
    S = np.vstack(S_l); P = np.vstack(P_l); Y = np.concatenate(y_l)
    CID = np.concatenate(c_l); FOLD = np.concatenate(f_l)
    pooled_ll = float(np.mean([r.log_loss for _, _, r in per_fold]))
    print(f"\n  pooled mean log loss: {pooled_ll:.16f}  (reference {POOLED_REF_LOG_LOSS:.16f})")
    check("pooled log loss reproduces documented V1",
          abs(pooled_ll - POOLED_REF_LOG_LOSS) < 1e-9,
          f"diff={pooled_ll - POOLED_REF_LOG_LOSS:.3e}")
    iH, iD, iA = 0, 1, 2
    pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]

    rule("STEP 3 -- TRAIN / VALIDATION CLASS PRIORS BY FOLD (from labels, not predictions)")
    print("| Fold | part | N | H | D | A | P(H) | P(D) | P(A) | log(H/D) | log(H/A) | log(D/A) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    priors = {}
    for (fold, tr, va) in folds:
        for lab, yy in (("train", tr.y), ("valid", va.y)):
            n = len(yy)
            c = {k: int((yy == k).sum()) for k in CLASS_ORDER}
            p = {k: c[k] / n for k in CLASS_ORDER}
            lr = {f"{a}{b}": np.log(c[a] / c[b]) for a, b in (("H", "D"), ("H", "A"), ("D", "A"))}
            if lab == "train":
                priors[fold.name] = lr
            print(f"| {fold.name} | {lab} | {n} | {c['H']} | {c['D']} | {c['A']} | "
                  f"{p['H']:.4f} | {p['D']:.4f} | {p['A']:.4f} | "
                  f"{lr['HD']:+.4f} | {lr['HA']:+.4f} | {lr['DA']:+.4f} |")
    print("\n  log(n_a/n_b) is the intercept contrast an INTERCEPT-ONLY multinomial model")
    print("  would fit exactly. It is the natural reference for STEP 5.")

    rule("STEP 4 -- FITTED INTERCEPTS (contrasts only -- absolutes are not identifiable)")
    print("| Fold | b_H | b_D | b_A | sum | H-D | H-A | D-A |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for fname, _, _ in per_fold:
        f = fits[fname]
        b = f["model"].intercept_[f["order"]]
        f["b"] = b
        print(f"| {fname} | {b[0]:+.4f} | {b[1]:+.4f} | {b[2]:+.4f} | {b.sum():+.4f} | "
              f"{b[0]-b[1]:+.4f} | {b[0]-b[2]:+.4f} | {b[1]-b[2]:+.4f} |")
    print("\n  The 'sum' column is diagnostic of the solver's identifiability handling only.")
    print("  softmax(s + c) == softmax(s), so shifting all three intercepts changes nothing.")

    rule("STEP 5 -- EMPIRICAL PRIOR CONTRAST vs FITTED CONTRAST (bare AND league-adjusted)")
    print("  The bare intercept sits at an IMPOSSIBLE point: numeric features at their")
    print("  training mean AND all five league flags zero. Every real fixture has exactly")
    print("  one flag set, so the identified offset for a league is b_c + w_c[league].")
    print("  Both comparisons are shown; the league-adjusted one is the meaningful one.\n")
    print("| Fold | Pair | empirical log-prior | bare intercept contrast | diff | "
          "league-adj contrast (mean over leagues, train-weighted) | diff |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for fname, _, _ in per_fold:
        f = fits[fname]
        b, model, pre = f["b"], f["model"], f["pre"]
        W = model.coef_[f["order"]]
        n_num = len(pre.numeric_columns)
        cats = list(pre.competition_categories)
        share = np.array([float((f["X_train"][RECOMMENDED_CONTEXT_FEATURE] == c).mean())
                          for c in cats])
        eff = b[:, None] + W[:, n_num:n_num + len(cats)]        # (3, n_leagues)
        eff_mean = eff @ share
        f["eff"], f["cats"], f["share"] = eff, cats, share
        for a, bb, ia, ib in (("H", "D", 0, 1), ("H", "A", 0, 2), ("D", "A", 1, 2)):
            emp = priors[fname][f"{a}{bb}"]
            bare = b[ia] - b[ib]
            adj = eff_mean[ia] - eff_mean[ib]
            print(f"| {fname} | {a}-{bb} | {emp:+.4f} | {bare:+.4f} | {bare-emp:+.4f} | "
                  f"{adj:+.4f} | {adj-emp:+.4f} |")
    print("\n  If the league-adjusted contrast tracks the empirical log-prior closely, the")
    print("  class geometry IS essentially the base rate -- which is correct behaviour for")
    print("  a probabilistic model, not a defect.")

    rule("STEP 6 -- REGULARIZATION / SOLVER / MULTICLASS CONFIGURATION AUDIT (resolved values)")
    m0 = fits[per_fold[0][0]]["model"]
    params = m0.get_params()
    for k in sorted(params):
        print(f"    {k:20s} = {params[k]!r}")
    print(f"\n  resolved classes_          : {list(m0.classes_)}")
    print(f"  coef_ shape                : {m0.coef_.shape}")
    print(f"  intercept_ shape           : {m0.intercept_.shape}")
    print(f"  n_iter_                    : {m0.n_iter_}  (max_iter={m0.max_iter})")
    check("converged well inside max_iter", int(np.max(m0.n_iter_)) < m0.max_iter,
          f"{int(np.max(m0.n_iter_))}/{m0.max_iter}")
    print("\n  Whether the L2 penalty reaches the intercept is decided by sklearn's")
    print("  implementation, not by this project. Tested empirically below rather than")
    print("  asserted: an unpenalised intercept in a multinomial fit reproduces the")
    print("  training log-prior contrast almost exactly once the league offset is")
    print("  accounted for; a penalised one shrinks it toward zero. STEP 5's final")
    print("  column is that test.")

    rule("STEP 7 -- PREPROCESSING / FEATURE-CENTRING AUDIT")
    for fname, _, _ in per_fold:
        pre = fits[fname]["pre"]
        sc = pre._scaler
        print(f"\n  {fname}: numeric columns scaled = {len(pre.numeric_columns)}   "
              f"one-hot league columns (UNSCALED) = {len(pre.competition_categories)}")
        print(f"    scaler mean_  : min={sc.mean_.min():+.4f} max={sc.mean_.max():+.4f}")
        print(f"    scaler scale_ : min={sc.scale_.min():.4f} max={sc.scale_.max():.4f}")
        print(f"    imputer strategy: {pre._imputer.strategy}  (fitted on the TRAIN fold only)")
    print("\n  Numeric zero == training mean. League one-hot zero == 'no league', which no")
    print("  real fixture satisfies. This is exactly why STEP 5 reports a league-adjusted")
    print("  contrast: the bare intercept is anchored at an unobservable point.")

    rule("STEP 8 -- INTERCEPT vs FEATURE CONTRIBUTION (extends D-35)")
    print("| Fold | Class | intercept | mean feature contrib | mean total score |")
    print("|---|---|---:|---:|---:|")
    for fname, _, _ in per_fold:
        f = fits[fname]
        m = FOLD == fname
        feat = S[m] - f["b"]
        f["feat"] = feat
        for i, c in enumerate(CLASS_ORDER):
            print(f"| {fname} | {c} | {f['b'][i]:+.4f} | {feat[:,i].mean():+.4f} | "
                  f"{S[m,i].mean():+.4f} |")
    print("\n| Fold | Pair | intercept diff | feature-contrib diff | total | intercept share |")
    print("|---|---|---:|---:|---:|---:|")
    for fname, _, _ in per_fold:
        f = fits[fname]
        for a, bb, ia, ib in (("H", "D", 0, 1), ("H", "A", 0, 2), ("D", "A", 1, 2)):
            di = f["b"][ia] - f["b"][ib]
            df = float((f["feat"][:, ia] - f["feat"][:, ib]).mean())
            tot = di + df
            sh = f"{100*di/tot:.1f}%" if abs(tot) > 1e-12 else "n/a"
            print(f"| {fname} | {a}-{bb} | {di:+.4f} | {df:+.4f} | {tot:+.4f} | {sh} |")
    print("\n  A share above 100% (D-35 saw 112.8%) means the FEATURE contribution pushes")
    print("  the opposite way to the intercept -- not that the intercept explains 'more")
    print("  than everything'. Reported as measured.")

    rule("STEP 9 -- LEAGUE ONE-HOT PRIOR-CORRECTION AUDIT")
    print("| Fold | League | train P(H) | train P(D) | train P(A) | w_H | w_D | w_A | "
          "b+w H-D | b+w H-A | league log(H/D) | league log(H/A) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fname, _, _ in per_fold:
        f = fits[fname]
        pre, model = f["pre"], f["model"]
        n_num = len(pre.numeric_columns)
        W = model.coef_[f["order"]]
        ytr, Xtr = f["y_train"], f["X_train"]
        for k, c in enumerate(f["cats"]):
            sel = (Xtr[RECOMMENDED_CONTEXT_FEATURE] == c).values
            yy = ytr[sel]
            cnt = {q: int((yy == q).sum()) for q in CLASS_ORDER}
            p = {q: cnt[q] / len(yy) for q in CLASS_ORDER}
            w = W[:, n_num + k]
            e = f["eff"][:, k]
            print(f"| {fname} | {LEAGUES.get(int(c), c)} | {p['H']:.4f} | {p['D']:.4f} | "
                  f"{p['A']:.4f} | {w[0]:+.4f} | {w[1]:+.4f} | {w[2]:+.4f} | "
                  f"{e[0]-e[1]:+.4f} | {e[0]-e[2]:+.4f} | "
                  f"{np.log(cnt['H']/cnt['D']):+.4f} | {np.log(cnt['H']/cnt['A']):+.4f} |")
    print("\n  These are NOT importance scores. The question is only whether each league's")
    print("  offset moves toward that league's own training base rate.")

    rule("STEP 10 -- INTERCEPT-ONLY DESCRIPTIVE GEOMETRY")
    print("  DIAGNOSTIC ONLY -- NOT A MODEL -- NOT DEPLOYED -- NOT PERSISTED -- NOT TUNED.")
    print("  softmax of the fitted intercepts alone, i.e. the class geometry present")
    print("  BEFORE any feature contribution.\n")
    print("| Fold | intercept-only P(H) | P(D) | P(A) | train P(H) | P(D) | P(A) | "
          "full-model mean P(H) | P(D) | P(A) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fname, _, _ in per_fold:
        f = fits[fname]
        e = np.exp(f["b"] - f["b"].max()); q = e / e.sum()
        ytr = f["y_train"]
        tp = [float((ytr == c).mean()) for c in CLASS_ORDER]
        m = FOLD == fname
        mp = [float(P[m, i].mean()) for i in range(3)]
        print(f"| {fname} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | "
              f"{tp[0]:.4f} | {tp[1]:.4f} | {tp[2]:.4f} | "
              f"{mp[0]:.4f} | {mp[1]:.4f} | {mp[2]:.4f} |")
    print("\n  Caveat repeated: the intercept-only point corresponds to 'no league', so")
    print("  this is an approximation of the pre-feature geometry, not a model.")

    rule("STEP 11 -- CROSS-FOLD INTERCEPT STABILITY (ranges only; 3 folds cannot support SD)")
    B = np.vstack([fits[f]["b"] for f, _, _ in per_fold])
    labs = ["b_H", "b_D", "b_A", "H-D", "H-A", "D-A"]
    vals = np.column_stack([B[:, 0], B[:, 1], B[:, 2],
                            B[:, 0] - B[:, 1], B[:, 0] - B[:, 2], B[:, 1] - B[:, 2]])
    print("| Quantity | fold_1 | fold_2 | fold_3 | min | max | range |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for i, lab in enumerate(labs):
        v = vals[:, i]
        print(f"| {lab} | {v[0]:+.4f} | {v[1]:+.4f} | {v[2]:+.4f} | {v.min():+.4f} | "
              f"{v.max():+.4f} | {v.max()-v.min():.4f} |")
    print("\n  Only the last three rows are identifiable quantities. NO standard deviation")
    print("  is reported: three folds cannot support one.")

    rule("STEP 12 -- CASE EVIDENCE (measurements only)")
    dsub = Y == "D"
    Pd = P[dsub]
    print(f"  actual draws                         : {int(dsub.sum())}")
    print(f"  Draw argmax share (all fixtures)     : {100*(pred=='D').mean():.2f}%")
    print(f"  fixtures with P(D) > 1/3             : {int((P[:,iD]>1/3).sum())} "
          f"({100*(P[:,iD]>1/3).mean():.2f}%)   <- structural ceiling on Draw argmax")
    print(f"  fixtures with P(H) > 1/3             : {int((P[:,iH]>1/3).sum())} "
          f"({100*(P[:,iH]>1/3).mean():.2f}%)")
    print(f"  fixtures with P(A) > 1/3             : {int((P[:,iA]>1/3).sum())} "
          f"({100*(P[:,iA]>1/3).mean():.2f}%)")
    print(f"  SD of P(H)/P(D)/P(A)                 : {P[:,iH].std():.4f} / "
          f"{P[:,iD].std():.4f} / {P[:,iA].std():.4f}")
    print(f"  mean |predicted - actual| class share : " + "  ".join(
        f"{c}={abs(P[:,i].mean()-(Y==c).mean()):.4f}" for i, c in enumerate(CLASS_ORDER)))
    print("\n  Per-league prior spread (train), the CASE D discriminator:")
    for cid, nm in LEAGUES.items():
        tr_y = pd.concat([tr.y[(tr.X[RECOMMENDED_CONTEXT_FEATURE] == cid).values]
                          for _, tr, _ in folds])
        print(f"    {nm:16s} P(H)={float((tr_y=='H').mean()):.4f} "
              f"P(D)={float((tr_y=='D').mean()):.4f} P(A)={float((tr_y=='A').mean()):.4f}")

    rule("STEP 13 -- CONCLUSION SCAFFOLD (fill from the measurements above)")
    print("  Decision guide, stated in advance so the reading is not chosen to fit:")
    print("   * STEP 5 league-adjusted contrast ~= empirical log-prior  -> the intercept")
    print("     geometry is the BASE RATE, which is correct behaviour. That weakens CASE B")
    print("     as a 'defect' and strengthens CASE A or CASE C.")
    print("   * STEP 5 contrast far from the log-prior                  -> CASE B genuine.")
    print("   * P(D) > 1/3 share ~= observed Draw argmax share          -> Draw is at its")
    print("     structural ceiling; no decision rule recovers it -> CASE C, not CASE A.")
    print("   * Per-league log-priors differ materially AND league offsets fail to track")
    print("     them                                                    -> CASE D.")

    rule("STEP 14 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    print("\nFILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
