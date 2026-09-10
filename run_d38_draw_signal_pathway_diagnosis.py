"""D-38 -- Draw signal pathway / contribution attribution investigation.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_d38_draw_signal_pathway_diagnosis.py

READ-ONLY DIAGNOSTIC. NOTHING IS FIXED, TUNED, CALIBRATED OR PERSISTED.

THE QUESTION
D-37 found substantial, stable Draw-vs-Home / Draw-vs-Away separation in
raw feature space, yet the fitted linear feature contribution on actual
draws is approximately neutral (H-D ~ +0.028/+0.038/-0.013;
D-A ~ -0.079/-0.011/-0.008), while aggregate P(D)=0.2486 tracks the
0.2519 base rate and Draw wins argmax on only 2.40% of fixtures.

D-38 measures WHERE the signal disappears along:
    raw feature -> preprocessing -> coefficient -> per-feature
    contribution -> class-pair contribution -> margin -> probability
    -> argmax

It does not assume collinearity, regularization or the intercept as the
cause. Each is a measurable component and each is measured.

EXACT ATTRIBUTION IDENTITY USED THROUGHOUT
`LogisticRegressionPreprocessor.transform` returns
    np.hstack([scaled_numeric, league_onehot])
so for any class pair (a,b) the score margin decomposes EXACTLY as
    M_ab = (b_a - b_b)                                   [intercept]
         + sum_{j in league}  x_j (w_aj - w_bj)          [league]
         + sum_{j in numeric} x_j (w_aj - w_bj)          [numeric]
STEP 7 verifies this identity to numerical precision and hard-fails if
it does not hold, so every attribution number is traceable to the
production preprocessing and the in-memory fitted model.

STANDING CAVEATS (restated in the output)
  - Coefficient magnitude is NOT feature importance (D-34: 47/84 sign
    flips across folds; the 80 columns are strongly collinear).
  - SMD is an association, never a cause.
  - Pairwise margins are diagnostics, never deployed classifiers.
  - Three folds => ranges only, never a standard deviation.
  - Interactions are not inferred unless measured.
"""
from __future__ import annotations

import hashlib
import inspect
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

#: EXACTLY the D-34 taxonomy. Not redefined, not extended.
FAMILIES = {
    "goals_season": lambda c: "goals" in c and "season" in c and "venue" not in c,
    "goals_last5": lambda c: "goals" in c and "last5" in c,
    "goals_last10": lambda c: "goals" in c and "last10" in c,
    "shots_season": lambda c: "shots" in c and "shots_on" not in c and "season" in c,
    "shots_last": lambda c: "shots" in c and "shots_on" not in c and "last" in c,
    "shots_on_season": lambda c: "shots_on" in c and "season" in c,
    "shots_on_last": lambda c: "shots_on" in c and "last" in c,
    "form": lambda c: "form" in c or "points" in c or "streak" in c or "result" in c,
    "strength": lambda c: "strength" in c or "attack" in c or "defence" in c or "defense" in c,
    "context": lambda c: c == "competition_id",
}


def rule(t): print("\n" + "=" * 116); print(t); print("=" * 116)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 116)
    print("HARD FAIL -- D-38 halted. No diagnosis is produced.")
    print(msg)
    print("!" * 116)
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


def smd(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def cancel_ratio(contrib):
    """|net| / sum|components|, per row then averaged, plus the aggregate."""
    net = np.abs(contrib.sum(axis=1))
    tot = np.abs(contrib).sum(axis=1)
    per = np.divide(net, tot, out=np.full_like(net, np.nan), where=tot > 0)
    return float(np.nanmean(per)), float(net.sum() / tot.sum()) if tot.sum() else float("nan")


def main():
    rule("STEP 1 -- ENVIRONMENT + INTEGRITY")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")
    try:
        import sklearn
        print(f"  sklearn {sklearn.__version__}")
    except ImportError as exc:
        stop(f"scikit-learn is required: {exc}. Do not substitute another estimator.")

    from models import train as train_mod
    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        FINAL_TEST_SEASONS, MODEL_VERSION, RECOMMENDED_CONTEXT_FEATURE,
        REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS,
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

    rule("RULE 1 -- PRODUCTION CODE TRACE (read from source, not memory)")
    src = inspect.getsource(train_mod.LogisticRegressionPreprocessor)
    for token, why in (("SimpleImputer", "imputation"), ("StandardScaler", "standardisation"),
                       ("np.hstack([scaled, onehot])", "final column layout"),
                       ("RECOMMENDED_CONTEXT_FEATURE", "categorical column")):
        check(f"preprocessor source contains {token!r} ({why})", token in src)
    tsrc = inspect.getsource(train_logistic_regression)
    check("training path constructs LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
          "LogisticRegression(max_iter=2000, C=1.0, random_state=0)" in tsrc)
    check("training path calls _reorder_proba", "_reorder_proba" in tsrc)
    print("    transform() = hstack([ scaler(imputer(numeric)) , league_onehot ])")
    print("    numeric_columns = contract order minus competition_id (order preserved)")
    print("    league one-hot appended UNSCALED as 0/1, in sorted category order")

    rule("STEP 2 -- REPRODUCE V1 (in-memory refits; production artifact NOT loaded)")
    fits, per_fold = {}, []
    S_l, P_l, T_l, y_l, c_l, f_l, X_l = [], [], [], [], [], [], []
    for fold, tr, va in folds:
        Xtr, Xva = tr.X[list(MODEL_B_COLUMNS)], va.X[list(MODEL_B_COLUMNS)]
        model, pre, P = train_logistic_regression(Xtr, tr.y, Xva)
        r = evaluate(va.y, P)
        per_fold.append((fold.name, len(va), r))
        cls = list(model.classes_)
        order = [cls.index(c) for c in CLASS_ORDER]
        T = pre.transform(Xva)                      # transformed design matrix
        S = model.decision_function(T)[:, order]
        fits[fold.name] = dict(model=model, pre=pre, order=order,
                               W=model.coef_[order], b=model.intercept_[order])
        S_l.append(S); P_l.append(P); T_l.append(T); y_l.append(va.y.to_numpy())
        c_l.append(va.X[RECOMMENDED_CONTEXT_FEATURE].to_numpy())
        f_l.append(np.repeat(fold.name, len(va)))
        X_l.append(Xva.reset_index(drop=True))
        print(f"  {fold.name}: N={len(va)} log_loss={r.log_loss:.16f} brier={r.brier:.16f}")
    S = np.vstack(S_l); P = np.vstack(P_l); T = np.vstack(T_l)
    Y = np.concatenate(y_l); CID = np.concatenate(c_l); FOLD = np.concatenate(f_l)
    X = pd.concat(X_l, ignore_index=True)
    pooled_ll = float(np.mean([r.log_loss for _, _, r in per_fold]))
    print(f"\n  pooled mean log loss: {pooled_ll:.16f}  (reference {POOLED_REF_LOG_LOSS:.16f})")
    check("pooled log loss reproduces documented V1",
          abs(pooled_ll - POOLED_REF_LOG_LOSS) < 1e-9,
          f"diff={pooled_ll - POOLED_REF_LOG_LOSS:.3e}")

    iH, iD, iA = 0, 1, 2
    pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    isD, isH, isA = Y == "D", Y == "H", Y == "A"

    rule("STEP 3 -- SCORE / PROBABILITY CONSISTENCY GATE")
    e = np.exp(S - S.max(axis=1, keepdims=True))
    soft = e / e.sum(axis=1, keepdims=True)
    dev = float(np.abs(soft - P).max())
    mism = int((soft.argmax(axis=1) != P.argmax(axis=1)).sum())
    print(f"  max |softmax(scores) - predict_proba| : {dev:.3e}   (D-35 saw ~2.22e-16)")
    print(f"  argmax disagreements                  : {mism} / {len(Y)}   (D-35 saw 0)")
    check("softmax(scores) reproduces predict_proba to 1e-9", dev <= 1e-9, f"{dev:.3e}")
    check("argmax(scores) == argmax(probabilities) everywhere", mism == 0)

    rule("STEP 4 -- FEATURE-MAPPING AUDIT")
    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    f0 = fits[per_fold[0][0]]
    pre0 = f0["pre"]
    cats = list(pre0.competition_categories)
    n_num, n_cat = len(pre0.numeric_columns), len(cats)
    check("preprocessor numeric_columns == contract minus competition_id",
          list(pre0.numeric_columns) == numeric, f"{len(pre0.numeric_columns)} vs {len(numeric)}")
    check("model column count == numeric + leagues", f0["W"].shape[1] == n_num + n_cat,
          f"{f0['W'].shape[1]} vs {n_num + n_cat}")
    check("transformed matrix width matches coefficient width", T.shape[1] == n_num + n_cat)
    names = list(pre0.numeric_columns) + [f"competition_id={c}" for c in cats]
    check("mapping is unique (no duplicate model column names)",
          len(set(names)) == len(names))
    check("every contract feature appears exactly once in the mapping",
          set(numeric) | {RECOMMENDED_CONTEXT_FEATURE} == set(MODEL_B_COLUMNS))
    print(f"  numeric (imputed+scaled) : model cols 0..{n_num-1}   ({n_num})")
    print(f"  league one-hot (UNSCALED): model cols {n_num}..{n_num+n_cat-1}  ({n_cat})")
    print(f"  total model columns      : {len(names)}")
    print("\n  first 5 and last 8 mappings:")
    for j in list(range(5)) + list(range(len(names) - 8, len(names))):
        kind = "numeric/scaled" if j < n_num else "league/unscaled"
        print(f"    col {j:>2} -> {names[j][:52]:52s} [{kind}]")

    rule("STEP 5 -- RAW SIGNAL -> COEFFICIENT ALIGNMENT (association, NOT importance)")
    Wm = np.mean([fits[f]["W"] for f, _, _ in per_fold], axis=0)
    Td = T[isD]
    mean_td = Td.mean(axis=0)
    raw_dh = np.array([smd(X[c].to_numpy(float)[isD], X[c].to_numpy(float)[isH])
                       for c in numeric])
    raw_da = np.array([smd(X[c].to_numpy(float)[isD], X[c].to_numpy(float)[isA])
                       for c in numeric])
    md_dh = np.array([np.nanmean(X[c].to_numpy(float)[isD]) - np.nanmean(X[c].to_numpy(float)[isH])
                      for c in numeric])
    md_da = np.array([np.nanmean(X[c].to_numpy(float)[isD]) - np.nanmean(X[c].to_numpy(float)[isA])
                      for c in numeric])
    wdh = Wm[iD, :n_num] - Wm[iH, :n_num]
    wda = Wm[iD, :n_num] - Wm[iA, :n_num]
    exp_dh = mean_td[:n_num] * wdh
    exp_da = mean_td[:n_num] * wda
    tab = pd.DataFrame({"feature": numeric, "raw_smd_DH": raw_dh, "raw_smd_DA": raw_da,
                        "raw_md_DH": md_dh, "raw_md_DA": md_da,
                        "w_H": Wm[iH, :n_num], "w_D": Wm[iD, :n_num], "w_A": Wm[iA, :n_num],
                        "wD_minus_wH": wdh, "wD_minus_wA": wda,
                        "exp_contrib_DH": exp_dh, "exp_contrib_DA": exp_da})
    for lab, rs, ws in (("D vs H", "raw_smd_DH", "wD_minus_wH"),
                        ("D vs A", "raw_smd_DA", "wD_minus_wA")):
        ok = tab[[rs, ws]].dropna()
        agree = float((np.sign(ok[rs]) == np.sign(ok[ws])).mean())
        corr = float(np.corrcoef(ok[rs], ok[ws])[0, 1])
        print(f"  {lab}: sign agreement raw-SMD vs coefficient difference = {100*agree:.1f}%"
              f"   Pearson r = {corr:+.4f}   (n={len(ok)})")
    print("\n  A low sign-agreement means the fitted coefficients do NOT point the way the")
    print("  raw class separation does -- measured, not assumed. This is an association")
    print("  statement about the fitted linear map, not a claim of importance.")

    rule("STEP 6 -- CONTRIBUTION WATERFALL (top features, per condition)")
    cd_ = Td[:, :n_num] * wdh          # per-fixture D-H numeric contribution
    ca_ = Td[:, :n_num] * wda
    tab["contrib_DH"] = cd_.mean(axis=0)
    tab["contrib_DA"] = ca_.mean(axis=0)
    tab["raw_mean_D"] = [np.nanmean(X[c].to_numpy(float)[isD]) for c in numeric]
    tab["raw_mean_H"] = [np.nanmean(X[c].to_numpy(float)[isH]) for c in numeric]
    tab["raw_mean_A"] = [np.nanmean(X[c].to_numpy(float)[isA]) for c in numeric]

    def show(df, title, cols):
        print(f"\n### {title}  (n reported = {len(df)})")
        print("| Feature | rawD | rawH | rawA | SMD DH | SMD DA | w_H | w_D | w_A "
              "| contrib DH | contrib DA | raw sign | fit sign |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for _, r in df.head(20).iterrows():
            rs = r["raw_smd_DH"] if "DH" in cols else r["raw_smd_DA"]
            fs = r["contrib_DH"] if "DH" in cols else r["contrib_DA"]
            print(f"| {r['feature'][:38]} | {r['raw_mean_D']:+.3f} | {r['raw_mean_H']:+.3f} | "
                  f"{r['raw_mean_A']:+.3f} | {r['raw_smd_DH']:+.3f} | {r['raw_smd_DA']:+.3f} | "
                  f"{r['w_H']:+.4f} | {r['w_D']:+.4f} | {r['w_A']:+.4f} | "
                  f"{r['contrib_DH']:+.5f} | {r['contrib_DA']:+.5f} | "
                  f"{'+' if rs > 0 else '-'} | {'+' if fs > 0 else '-'} |")

    show(tab.sort_values("contrib_DH", ascending=False), "TOP 20 favouring Draw over Home", "DH")
    show(tab.sort_values("contrib_DA", ascending=False), "TOP 20 favouring Draw over Away", "DA")
    weak_dh = tab[(tab.raw_smd_DH.abs() >= .10) &
                  ((np.sign(tab.raw_smd_DH) != np.sign(tab.contrib_DH)) |
                   (tab.contrib_DH.abs() < .001))].reindex(
        tab.raw_smd_DH.abs().sort_values(ascending=False).index).dropna(subset=["feature"])
    weak_da = tab[(tab.raw_smd_DA.abs() >= .10) &
                  ((np.sign(tab.raw_smd_DA) != np.sign(tab.contrib_DA)) |
                   (tab.contrib_DA.abs() < .001))].reindex(
        tab.raw_smd_DA.abs().sort_values(ascending=False).index).dropna(subset=["feature"])
    show(weak_dh, "STRONG raw D-vs-H signal but WEAK/OPPOSITE fitted D-H contribution", "DH")
    show(weak_da, "STRONG raw D-vs-A signal but WEAK/OPPOSITE fitted D-A contribution", "DA")

    rule("STEP 7 -- MARGIN DECOMPOSITION AND EXACTNESS CHECK")
    comp = {}
    for a, b, ia, ib in (("H", "D", iH, iD), ("H", "A", iH, iA), ("D", "A", iD, iA)):
        inter = np.empty(len(Y)); leag = np.empty(len(Y)); num = np.empty(len(Y))
        numeric_contrib = np.empty((len(Y), n_num))
        for fname, _, _ in per_fold:
            m = FOLD == fname
            f = fits[fname]
            dw = f["W"][ia] - f["W"][ib]
            inter[m] = f["b"][ia] - f["b"][ib]
            leag[m] = T[m][:, n_num:] @ dw[n_num:]
            nc = T[m][:, :n_num] * dw[:n_num]
            numeric_contrib[m] = nc
            num[m] = nc.sum(axis=1)
        total = S[:, ia] - S[:, ib]
        err = float(np.abs(inter + leag + num - total).max())
        check(f"{a}-{b}: intercept + league + numeric == margin", err < 1e-9, f"max err {err:.3e}")
        comp[(a, b)] = dict(inter=inter, leag=leag, num=num, total=total, nc=numeric_contrib)

    print("\n  ACTUAL DRAWS -- component decomposition (negative favours Draw for H-D/H-A;")
    print("  positive favours Draw for D-A)")
    print("| Pair | component | mean | median | p05 | p95 | frac favouring Draw |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for (a, b), c in comp.items():
        fav_sign = -1 if b == "D" else (1 if a == "D" else None)
        for lab in ("inter", "leag", "num", "total"):
            v = c[lab][isD]
            if fav_sign is None:
                frac = "n/a"
            else:
                frac = f"{100*float((v*fav_sign > 0).mean()):.1f}%"
            print(f"| {a}-{b} | {lab} | {v.mean():+.5f} | {np.median(v):+.5f} | "
                  f"{np.percentile(v,5):+.5f} | {np.percentile(v,95):+.5f} | {frac} |")

    rule("STEP 8 -- CANCELLATION CONCENTRATION (actual draws)")
    print("| Pair | mean sum\\|contrib\\| | mean \\|net\\| | per-row cancel ratio | aggregate ratio |")
    print("|---|---:|---:|---:|---:|")
    for (a, b), c in comp.items():
        nc = c["nc"][isD]
        per, agg = cancel_ratio(nc)
        print(f"| {a}-{b} | {np.abs(nc).sum(axis=1).mean():.4f} | "
              f"{np.abs(nc.sum(axis=1)).mean():.4f} | {per:.4f} | {agg:.4f} |")
    print("\n  ratio near 0 with LARGE sum|contrib| -> strong cancellation among features")
    print("  ratio near 0 with SMALL sum|contrib| -> weak effective linear signal")
    print("  ratio large                          -> features genuinely move the boundary")
    print("  No case is selected here.")

    rule("STEP 9 -- FAMILY CONTRIBUTION (D-34 taxonomy, unchanged)")
    fam_cols, assigned = {}, set()
    for fam, fn in FAMILIES.items():
        cs = [c for c in numeric if fn(c)]
        fam_cols[fam] = cs
        assigned |= set(cs)
    unmatched = [c for c in numeric if c not in assigned]
    print(f"  taxonomy covers {len(assigned)}/{len(numeric)} numeric features")
    if unmatched:
        print(f"  UNMATCHED ({len(unmatched)}): {unmatched[:12]}{' ...' if len(unmatched)>12 else ''}")
    idx = {c: i for i, c in enumerate(numeric)}
    ncDH, ncDA = comp[("H", "D")]["nc"][isD], comp[("D", "A")]["nc"][isD]
    tot_dh = np.abs(ncDH).sum(); tot_da = np.abs(ncDA).sum()
    print("\n| Family | n | sum\\|DH\\| | net DH | share \\|DH\\| | sum\\|DA\\| | net DA | share \\|DA\\| "
          "| DH sign consistent |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for fam, cs in fam_cols.items():
        if not cs:
            print(f"| {fam} | 0 | - | - | - | - | - | - | - |")
            continue
        jj = [idx[c] for c in cs if c in idx]
        sdh_, ndh_ = float(np.abs(ncDH[:, jj]).sum()), float(ncDH[:, jj].sum())
        sda_, nda_ = float(np.abs(ncDA[:, jj]).sum()), float(ncDA[:, jj].sum())
        signs = []
        for fname, _, _ in per_fold:
            m = (FOLD == fname) & isD
            f = fits[fname]
            dw = (f["W"][iH] - f["W"][iD])[:n_num]
            signs.append(np.sign((T[m][:, :n_num] * dw)[:, jj].sum()))
        print(f"| {fam} | {len(cs)} | {sdh_:.3f} | {ndh_:+.3f} | {100*sdh_/tot_dh:.1f}% | "
              f"{sda_:.3f} | {nda_:+.3f} | {100*sda_/tot_da:.1f}% | "
              f"{bool(len(set(signs)) == 1)} |")

    rule("STEP 10 -- CONTRIBUTION BY ACTUAL CLASS (reproduces D-37 STEP 7, then extends)")
    print("| Fold | subset | N | feat H | feat D | feat A | H-D | H-A | D-A "
          "| sum\\|DH\\| | net DH | cancel DH |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fname, _, _ in per_fold:
        f = fits[fname]
        m = FOLD == fname
        feat = S[m] - f["b"]
        yy = Y[m]
        dw = (f["W"][iH] - f["W"][iD])[:n_num]
        nc_all = T[m][:, :n_num] * dw
        for lab, sub in (("actual D", yy == "D"), ("actual H", yy == "H"), ("actual A", yy == "A")):
            fc = feat[sub]; nc = nc_all[sub]
            per, _ = cancel_ratio(nc)
            print(f"| {fname} | {lab} | {int(sub.sum())} | {fc[:,iH].mean():+.4f} | "
                  f"{fc[:,iD].mean():+.4f} | {fc[:,iA].mean():+.4f} | "
                  f"{(fc[:,iH]-fc[:,iD]).mean():+.4f} | {(fc[:,iH]-fc[:,iA]).mean():+.4f} | "
                  f"{(fc[:,iD]-fc[:,iA]).mean():+.4f} | {np.abs(nc).sum(axis=1).mean():.4f} | "
                  f"{nc.sum(axis=1).mean():+.4f} | {per:.4f} |")

    rule("STEP 11 -- CROSS-FOLD PATHWAY STABILITY (ranges only; no SD)")
    print("| Metric | fold_1 | fold_2 | fold_3 | min | max | range |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    rowsm = {}
    for fname, _, _ in per_fold:
        m = (FOLD == fname) & isD
        f = fits[fname]
        dwH = (f["W"][iH] - f["W"][iD])[:n_num]
        dwA = (f["W"][iD] - f["W"][iA])[:n_num]
        ncH = T[m][:, :n_num] * dwH
        ncA = T[m][:, :n_num] * dwA
        rdh = np.array([smd(X[c].to_numpy(float)[(FOLD == fname) & isD],
                            X[c].to_numpy(float)[(FOLD == fname) & isH]) for c in numeric])
        opp = int(np.sum(np.sign(rdh) != np.sign(dwH)))
        rowsm[fname] = [-ncH.sum(axis=1).mean(), ncA.sum(axis=1).mean(),
                        np.abs(ncH).sum(axis=1).mean(), np.abs(ncA).sum(axis=1).mean(),
                        cancel_ratio(ncH)[0], cancel_ratio(ncA)[0], float(opp)]
    labs = ["mean D-H feat contrib", "mean D-A feat contrib", "mean |D-H| contrib",
            "mean |D-A| contrib", "cancel ratio D-H", "cancel ratio D-A",
            "features raw/fit opposite sign"]
    for i, lab in enumerate(labs):
        v = [rowsm[f][i] for f, _, _ in per_fold]
        print(f"| {lab} | {v[0]:+.4f} | {v[1]:+.4f} | {v[2]:+.4f} | {min(v):+.4f} | "
              f"{max(v):+.4f} | {max(v)-min(v):.4f} |")

    rule("STEP 12 -- LEAGUE-SPECIFIC PATHWAY (no causal interpretation)")
    print("| League | N | draws | mean numeric D-H | mean numeric D-A | mean league D-H "
          "| mean league D-A | Draw argmax | mean P(D) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cid, nm in LEAGUES.items():
        m = (CID == cid) & isD
        allm = CID == cid
        print(f"| {nm} | {int(allm.sum())} | {int(m.sum())} | "
              f"{-comp[('H','D')]['num'][m].mean():+.4f} | {comp[('D','A')]['num'][m].mean():+.4f} | "
              f"{-comp[('H','D')]['leag'][m].mean():+.4f} | {comp[('D','A')]['leag'][m].mean():+.4f} | "
              f"{100*(pred[allm]=='D').mean():.2f}% | {P[allm,iD].mean():.4f} |")

    rule("STEP 13 -- PRE-REGISTERED CASE DISCRIMINATION INPUTS (measurements only)")
    perH, aggH = cancel_ratio(comp[("H", "D")]["nc"][isD])
    perA, aggA = cancel_ratio(comp[("D", "A")]["nc"][isD])
    print(f"  raw separation      : mean|SMD| D-H = {np.nanmean(np.abs(raw_dh)):.4f}  "
          f"D-A = {np.nanmean(np.abs(raw_da)):.4f}  max = "
          f"{np.nanmax(np.abs(raw_dh)):.4f}/{np.nanmax(np.abs(raw_da)):.4f}")
    print(f"  net numeric contrib : D-H = {-comp[('H','D')]['num'][isD].mean():+.5f}  "
          f"D-A = {comp[('D','A')]['num'][isD].mean():+.5f}")
    print(f"  |contrib| magnitude : D-H = {np.abs(comp[('H','D')]['nc'][isD]).sum(axis=1).mean():.4f}  "
          f"D-A = {np.abs(comp[('D','A')]['nc'][isD]).sum(axis=1).mean():.4f}")
    print(f"  cancellation ratio  : D-H = {perH:.4f} (agg {aggH:.4f})  "
          f"D-A = {perA:.4f} (agg {aggA:.4f})")
    print(f"  intercept component : H-D = {comp[('H','D')]['inter'][isD].mean():+.5f}  "
          f"D-A = {comp[('D','A')]['inter'][isD].mean():+.5f}")
    print(f"  league component    : H-D = {comp[('H','D')]['leag'][isD].mean():+.5f}  "
          f"D-A = {comp[('D','A')]['leag'][isD].mean():+.5f}")
    print(f"  aggregate P(D) gap  : {abs(P[:,iD].mean()-isD.mean()):.4f}  "
          f"(mean {P[:,iD].mean():.4f} vs actual {float(isD.mean()):.4f})")
    print(f"  Draw argmax share   : {100*(pred=='D').mean():.2f}%   "
          f"P(D)>1/3 = {100*(P[:,iD]>1/3).mean():.2f}%")
    print("\n  CASE A/B/C/D/E rules are applied in the written diagnosis, not here.")
    print("  CASE E retains its existing D-34 definition and is not redefined.")

    rule("STEP 15 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    print("\nD-38 COMPLETE -- DIAGNOSIS ONLY, NOTHING CHANGED")
    print("FILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
