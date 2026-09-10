"""D-39 -- feature collinearity / coefficient geometry diagnosis.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_d39_feature_collinearity_coefficient_diagnosis.py

READ-ONLY DIAGNOSTIC. NOTHING IS FIXED, TUNED, REMOVED OR PERSISTED.

THE QUESTION (STEP 5)
D-38 closed Step 4 with: raw Draw signal exists (mean |SMD| 0.2457 /
0.2281) but the fitted linear map does not transmit it directionally
(sign agreement 57.0% / 55.7%; Pearson r +0.0692 / +0.0537; cancellation
ratio 0.0607 / 0.0463). Two mechanisms remain:

    (A) multicollinearity redistributing coefficient mass across
        correlated columns  -> CASE C mechanism
    (B) Draw-relevant structure not representable in the current linear
        multinomial feature space -> CASE E

D-39 measures the collinearity structure and the coefficient geometry
inside it. It does not select the case; the written diagnosis does.

THE DISCRIMINATING LOGIC, PRE-REGISTERED
    individual coefficients unstable BUT cluster-level contribution
    stable                                   -> redistribution (A)
    both individual AND cluster-level weak/unstable
                                             -> representation limit (B)

STANDING CAVEATS RESTATED IN THE OUTPUT
  - Coefficient magnitude is NOT feature importance. Neither is
    correlation. These are geometry and contribution decompositions.
  - No causal claims from correlation, SMD or ablation.
  - Three folds => ranges only, never a standard deviation.
  - The |r| >= 0.90 clustering threshold is PRE-REGISTERED, applied for
    diagnosis only, and is not feature selection. Nothing is removed,
    merged or altered.
  - STEP 10 zeroes columns in STANDARDISED space. Because the production
    scaler centres the numeric block on the training mean, zeroing a
    column sets it to that mean -- it does NOT delete the feature and is
    NOT a retrain. It is a mathematical attribution, not an
    intervention.
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
CLUSTER_THRESHOLD = 0.90          # pre-registered, diagnosis only

#: Production source strings this script refuses to run without.
PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)

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


def rule(t): print("\n" + "=" * 118); print(t); print("=" * 118)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 118)
    print("HARD FAIL -- D-39 halted. No diagnosis is produced.")
    print(msg)
    print("!" * 118)
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


def components(adj):
    """Single-linkage connected components via union-find (no scipy)."""
    n = adj.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in np.where(adj[i])[0]:
            if j > i:
                ri, rj = find(i), find(int(j))
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


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
        FINAL_TEST_SEASONS, MODEL_VERSION, RECOMMENDED_CONTEXT_FEATURE,
        REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS,
    )
    from models.data import load_supervised_dataset
    from models.evaluate import evaluate
    from models.splits import iter_walk_forward_folds
    from models.train import train_logistic_regression

    rule("RULE 1 -- PRODUCTION CODE TRACE (source text, never memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:64]}", tok in tsrc)
    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("numeric block == contract minus competition_id (79)", len(numeric) == 79)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])

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

    rule("STEP 1b -- REPRODUCE V1 (in-memory refits; artifact NOT substituted)")
    fits, per_fold = {}, []
    S_l, P_l, T_l, y_l, c_l, f_l, X_l = [], [], [], [], [], [], []
    for fold, tr, va in folds:
        Xtr, Xva = tr.X[list(MODEL_B_COLUMNS)], va.X[list(MODEL_B_COLUMNS)]
        model, pre, P = train_logistic_regression(Xtr, tr.y, Xva)
        r = evaluate(va.y, P)
        per_fold.append((fold.name, len(va), r))
        cls = list(model.classes_)
        order = [cls.index(c) for c in CLASS_ORDER]
        fits[fold.name] = dict(model=model, pre=pre, W=model.coef_[order],
                               b=model.intercept_[order],
                               Ttr=pre.transform(Xtr), ytr=tr.y.to_numpy())
        T = pre.transform(Xva)
        S_l.append(model.decision_function(T)[:, order]); P_l.append(P); T_l.append(T)
        y_l.append(va.y.to_numpy()); c_l.append(va.X[RECOMMENDED_CONTEXT_FEATURE].to_numpy())
        f_l.append(np.repeat(fold.name, len(va))); X_l.append(Xva.reset_index(drop=True))
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
    n_num = 79

    rule("STEP 2 -- EXACT DESIGN-MATRIX AUDIT")
    for fname, _, _ in per_fold:
        f = fits[fname]
        pre, sc = f["pre"], f["pre"]._scaler
        check(f"{fname}: preprocessor numeric_columns == contract order minus competition_id",
              list(pre.numeric_columns) == numeric)
        check(f"{fname}: transformed width == 84",
              f["Ttr"].shape[1] == n_num + len(pre.competition_categories))
        check(f"{fname}: coefficient width matches transformed width",
              f["W"].shape[1] == f["Ttr"].shape[1])
        print(f"  {fname}: train {f['Ttr'].shape}  numeric=79  league="
              f"{len(pre.competition_categories)}  imputer={pre._imputer.strategy}")
        print(f"    scaler mean_  min={sc.mean_.min():+.4f} max={sc.mean_.max():+.4f}")
        print(f"    scaler scale_ min={sc.scale_.min():.4f} max={sc.scale_.max():.4f}")
        print(f"    numeric block standardised: col means ~0 "
              f"(max |mean|={np.abs(f['Ttr'][:, :n_num].mean(axis=0)).max():.2e})")
    print("\n  The matrix used below IS pre.transform(X) from the production preprocessor.")
    print("  It is never altered in place and never written to disk.")

    rule("STEP 3 -- NUMERIC CORRELATION STRUCTURE (TRAINING data only)")
    corrs = {}
    for fname, _, _ in per_fold:
        A = fits[fname]["Ttr"][:, :n_num]
        R = np.corrcoef(A, rowvar=False)
        R = np.nan_to_num(R, nan=0.0)
        corrs[fname] = R
        off = R[~np.eye(n_num, dtype=bool)]
        print(f"  {fname}: pairs |r|>=0.80 {int((np.abs(off)>=.80).sum()//2):5d}   "
              f">=0.90 {int((np.abs(off)>=.90).sum()//2):5d}   "
              f">=0.95 {int((np.abs(off)>=.95).sum()//2):5d}   "
              f"max|r| {np.abs(off).max():.4f}   median|r| {np.median(np.abs(off)):.4f}")
    print(f"\n  total unique pairs among 79 features: {79*78//2}")
    print("\n  per-family correlation density (D-34 taxonomy, mean |r| within family):")
    fam_cols, assigned = {}, set()
    for fam, fn in FAMILIES.items():
        cs = [c for c in numeric if fn(c)]
        fam_cols[fam] = cs
        assigned |= set(cs)
    unmatched = [c for c in numeric if c not in assigned]
    idx = {c: i for i, c in enumerate(numeric)}
    R0 = corrs[per_fold[0][0]]
    print("| Family | n | mean \\|r\\| within | max \\|r\\| within |")
    print("|---|---:|---:|---:|")
    for fam, cs in fam_cols.items():
        jj = [idx[c] for c in cs if c in idx]
        if len(jj) < 2:
            print(f"| {fam} | {len(jj)} | n/a | n/a |")
            continue
        sub = np.abs(R0[np.ix_(jj, jj)])
        m = ~np.eye(len(jj), dtype=bool)
        print(f"| {fam} | {len(jj)} | {sub[m].mean():.4f} | {sub[m].max():.4f} |")
    print(f"\n  UNMATCHED by the taxonomy ({len(unmatched)}): "
          f"{unmatched[:12]}{' ...' if len(unmatched) > 12 else ''}")

    rule(f"STEP 4 -- COLLINEARITY CLUSTERS (pre-registered |r| >= {CLUSTER_THRESHOLD}; "
         "diagnosis only, nothing removed)")
    adj = np.abs(R0) >= CLUSTER_THRESHOLD
    np.fill_diagonal(adj, False)
    comps = [c for c in components(adj) if len(c) > 1]
    singles = [c[0] for c in components(adj) if len(c) == 1]
    print(f"  clusters with >=2 members: {len(comps)}   singleton features: {len(singles)}")
    print("\n| Cluster | n | mean \\|r\\| | max \\|r\\| | near-duplicate (max\\|r\\|>=0.99) | families |")
    print("|---:|---:|---:|---:|---|---|")
    for k, cl in enumerate(comps, 1):
        sub = np.abs(R0[np.ix_(cl, cl)])
        m = ~np.eye(len(cl), dtype=bool)
        fams = sorted({fam for fam, cs in fam_cols.items()
                       for c in cs if c in idx and idx[c] in cl})
        print(f"| {k} | {len(cl)} | {sub[m].mean():.4f} | {sub[m].max():.4f} | "
              f"{bool(sub[m].max() >= .99)} | {','.join(fams) or 'unmatched'} |")
    for k, cl in enumerate(comps[:8], 1):
        print(f"\n  cluster {k} members ({len(cl)}):")
        for j in cl:
            print(f"    {numeric[j]}")

    rule("STEP 5 -- COEFFICIENT CONTRAST WITHIN CLUSTERS (NOT importance)")
    Wf = {f: fits[f]["W"] for f, _, _ in per_fold}
    dDH = {f: Wf[f][iD, :n_num] - Wf[f][iH, :n_num] for f in Wf}
    dDA = {f: Wf[f][iD, :n_num] - Wf[f][iA, :n_num] for f in Wf}
    tot_dh = np.mean([np.abs(dDH[f]).sum() for f in dDH])
    tot_da = np.mean([np.abs(dDA[f]).sum() for f in dDA])
    print("| Cluster | n | sum\\|D-H\\| | max\\|D-H\\| | share | sum\\|D-A\\| | max\\|D-A\\| | share "
          "| coefs flipping sign across folds |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, cl in enumerate(comps, 1):
        s_dh = np.mean([np.abs(dDH[f][cl]).sum() for f in dDH])
        s_da = np.mean([np.abs(dDA[f][cl]).sum() for f in dDA])
        mx_dh = np.mean([np.abs(dDH[f][cl]).max() for f in dDH])
        mx_da = np.mean([np.abs(dDA[f][cl]).max() for f in dDA])
        flips = sum(1 for j in cl
                    if len({np.sign(dDH[f][j]) for f in dDH}) > 1
                    or len({np.sign(dDA[f][j]) for f in dDA}) > 1)
        print(f"| {k} | {len(cl)} | {s_dh:.4f} | {mx_dh:.4f} | {100*s_dh/tot_dh:.1f}% | "
              f"{s_da:.4f} | {mx_da:.4f} | {100*s_da/tot_da:.1f}% | {flips}/{len(cl)} |")

    rule("STEP 6 -- RAW SIGNAL vs COEFFICIENT DIRECTION INSIDE CLUSTERS  [CRITICAL]")
    raw_dh = np.array([smd(X[c].to_numpy(float)[isD], X[c].to_numpy(float)[isH]) for c in numeric])
    raw_da = np.array([smd(X[c].to_numpy(float)[isD], X[c].to_numpy(float)[isA]) for c in numeric])
    mDH = np.mean([dDH[f] for f in dDH], axis=0)
    mDA = np.mean([dDA[f] for f in dDA], axis=0)
    print("| Cluster | n | mean raw SMD D-H | raw sign coherent | sign agree raw vs coef "
          "| within-cluster r(raw,coef) | opposing coef signs inside cluster |")
    print("|---:|---:|---:|---|---:|---:|---|")
    for k, cl in enumerate(comps, 1):
        r_ = raw_dh[cl]; c_ = mDH[cl]
        ok = ~np.isnan(r_)
        coherent = len({np.sign(x) for x in r_[ok]}) == 1 if ok.any() else False
        agree = float((np.sign(r_[ok]) == np.sign(c_[ok])).mean()) if ok.any() else float("nan")
        rr = (float(np.corrcoef(r_[ok], c_[ok])[0, 1])
              if ok.sum() >= 3 and np.std(r_[ok]) > 0 and np.std(c_[ok]) > 0 else float("nan"))
        opposing = len({np.sign(x) for x in c_ if x != 0}) > 1
        print(f"| {k} | {len(cl)} | {np.nanmean(r_):+.4f} | {coherent} | "
              f"{100*agree:.1f}% | {rr:+.4f} | {opposing} |")
    print("\n  Strong coherent raw separation + opposing coefficient signs inside the SAME")
    print("  correlated cluster is the signature of redistribution. Coherent raw separation")
    print("  with weak coefficients in LOW-collinearity features points the other way.")

    rule("STEP 7 -- CONDITION NUMBER / RANK / VIF (TRAINING data only)")
    for fname, _, _ in per_fold:
        A = fits[fname]["Ttr"][:, :n_num]
        sv = np.linalg.svd(A, compute_uv=False)
        tol = sv.max() * max(A.shape) * np.finfo(float).eps
        rank = int((sv > tol).sum())
        energy = np.cumsum(sv**2) / np.sum(sv**2)
        r99 = int(np.searchsorted(energy, .99) + 1)
        cond = float(sv.max() / sv.min()) if sv.min() > 0 else float("inf")
        print(f"\n  {fname}: singular values max={sv.max():.4f} min={sv.min():.4e}")
        print(f"    condition number      : {cond:.4e}")
        print(f"    numerical rank        : {rank}/{n_num}  (tol={tol:.3e})")
        print(f"    components for 99% var: {r99}/{n_num}")
        R = np.nan_to_num(np.corrcoef(A, rowvar=False), nan=0.0)
        try:
            Rinv = np.linalg.inv(R)
            vif = np.diag(Rinv)
            valid = np.isfinite(vif) & (vif > 0)
            print(f"    VIF: computed via exact inverse. max={vif[valid].max():.2f}  "
                  f"median={np.median(vif[valid]):.2f}  "
                  f">10: {int((vif[valid]>10).sum())}/{n_num}  "
                  f">100: {int((vif[valid]>100).sum())}/{n_num}")
        except np.linalg.LinAlgError:
            vif = np.diag(np.linalg.pinv(R))
            print("    VIF: correlation matrix is SINGULAR -- exact inverse undefined.")
            print(f"    Reported from pseudo-inverse and NOT reliable: "
                  f"max={np.nanmax(vif):.2f}, median={np.nanmedian(vif):.2f}")
            print("    Reported honestly rather than forced into a clean number.")
    print("\n  High condition number / VIF is evidence about IDENTIFIABILITY and stability.")
    print("  It is NOT proof that the model is wrong.")

    rule("STEP 8 -- COEFFICIENT STABILITY ACROSS FOLDS (ranges only; no SD)")
    stab = []
    for j, c in enumerate(numeric):
        v = [dDH[f][j] for f, _, _ in per_fold]
        w = [dDA[f][j] for f, _, _ in per_fold]
        maxr = np.abs(R0[j]).copy(); maxr[j] = 0
        stab.append(dict(feature=c, dh=v, da=w, rng_dh=max(v) - min(v), rng_da=max(w) - min(w),
                         sign_dh=len(set(np.sign(v))) == 1, sign_da=len(set(np.sign(w))) == 1,
                         maxr=maxr.max()))
    sdf = pd.DataFrame(stab)
    print(f"  D-H contrast: stable sign {int(sdf.sign_dh.sum())}/79   "
          f"sign flips {int((~sdf.sign_dh).sum())}/79")
    print(f"  D-A contrast: stable sign {int(sdf.sign_da.sum())}/79   "
          f"sign flips {int((~sdf.sign_da).sum())}/79")
    hi = sdf[sdf.maxr >= CLUSTER_THRESHOLD]
    lo = sdf[sdf.maxr < CLUSTER_THRESHOLD]
    print(f"\n  features with max|r| >= {CLUSTER_THRESHOLD}: {len(hi)}   "
          f"of these, D-H sign flips: {int((~hi.sign_dh).sum())} "
          f"({100*float((~hi.sign_dh).mean()) if len(hi) else float('nan'):.1f}%)")
    print(f"  features with max|r| <  {CLUSTER_THRESHOLD}: {len(lo)}   "
          f"of these, D-H sign flips: {int((~lo.sign_dh).sum())} "
          f"({100*float((~lo.sign_dh).mean()) if len(lo) else float('nan'):.1f}%)")
    print(f"  mean D-H range, high-collinearity: "
          f"{hi.rng_dh.mean() if len(hi) else float('nan'):.4f}   "
          f"low-collinearity: {lo.rng_dh.mean() if len(lo) else float('nan'):.4f}")
    print("\n  15 largest D-H contrast ranges:")
    print("| Feature | f1 | f2 | f3 | range | max\\|r\\| | sign stable |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for _, r in sdf.sort_values("rng_dh", ascending=False).head(15).iterrows():
        print(f"| {r.feature[:40]} | {r.dh[0]:+.4f} | {r.dh[1]:+.4f} | {r.dh[2]:+.4f} | "
              f"{r.rng_dh:.4f} | {r.maxr:.4f} | {r.sign_dh} |")

    rule("STEP 9 -- CLUSTER-LEVEL CONTRIBUTION STABILITY  [DECISIVE COMPARISON]")
    print("  Contribution convention: positive favours DRAW.")
    print("| Cluster | n | D-H f1 | f2 | f3 | mean | range | sign stable "
          "| D-A f1 | f2 | f3 | mean | range | sign stable |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|")
    for k, cl in enumerate(comps, 1):
        ch, ca = [], []
        for fname, _, _ in per_fold:
            m = (FOLD == fname) & isD
            ch.append(float((T[m][:, cl] * dDH[fname][cl]).sum(axis=1).mean()))
            ca.append(float((T[m][:, cl] * dDA[fname][cl]).sum(axis=1).mean()))
        print(f"| {k} | {len(cl)} | {ch[0]:+.4f} | {ch[1]:+.4f} | {ch[2]:+.4f} | "
              f"{np.mean(ch):+.4f} | {max(ch)-min(ch):.4f} | {len(set(np.sign(ch)))==1} | "
              f"{ca[0]:+.4f} | {ca[1]:+.4f} | {ca[2]:+.4f} | {np.mean(ca):+.4f} | "
              f"{max(ca)-min(ca):.4f} | {len(set(np.sign(ca)))==1} |")
    print("\n  Compare against STEP 8: individual coefficients unstable BUT cluster totals")
    print("  stable => redistribution. Both unstable/weak => representation limitation.")

    rule("STEP 10 -- CLUSTER ABLATION (mathematical attribution; NOT a retrain)")
    print("  Zeroing a standardised column sets it to the TRAINING MEAN. It does not")
    print("  delete the feature, is not an intervention, and no model is refitted.")
    print("\n| Cluster | n | d mean M(D-H) | d mean M(D-A) | d mean P(D) | d Draw argmax share |")
    print("|---:|---:|---:|---:|---:|---:|")
    base_mhd = (S[:, iD] - S[:, iH]).mean()
    base_mda = (S[:, iD] - S[:, iA]).mean()
    base_pd = P[:, iD].mean()
    base_arg = float((pred == "D").mean())
    for k, cl in enumerate(comps, 1):
        S2 = np.empty_like(S)
        for fname, _, _ in per_fold:
            m = FOLD == fname
            f = fits[fname]
            Tm = T[m].copy()
            Tm[:, cl] = 0.0
            S2[m] = f["model"].decision_function(Tm)[:, [list(f["model"].classes_).index(c)
                                                         for c in CLASS_ORDER]]
        e2 = np.exp(S2 - S2.max(axis=1, keepdims=True))
        P2 = e2 / e2.sum(axis=1, keepdims=True)
        a2 = float((np.array(CLASS_ORDER)[P2.argmax(axis=1)] == "D").mean())
        print(f"| {k} | {len(cl)} | {(S2[:,iD]-S2[:,iH]).mean()-base_mhd:+.5f} | "
              f"{(S2[:,iD]-S2[:,iA]).mean()-base_mda:+.5f} | {P2[:,iD].mean()-base_pd:+.5f} | "
              f"{100*(a2-base_arg):+.3f}pp |")
        del S2, P2
    print(f"\n  baseline: mean M(D-H)={base_mhd:+.4f}  M(D-A)={base_mda:+.4f}  "
          f"mean P(D)={base_pd:.4f}  Draw argmax={100*base_arg:.2f}%")

    rule("STEP 11 -- CROSS-FOLD RANGES FOR THE STRONGEST CLUSTERS (no SD)")
    print("| Cluster | n | mean\\|r\\| | coef contrast range D-H | contribution range D-H "
          "| sign stable (coef) | sign stable (contribution) |")
    print("|---:|---:|---:|---:|---:|---|---|")
    for k, cl in enumerate(comps[:10], 1):
        sub = np.abs(R0[np.ix_(cl, cl)]); mm = ~np.eye(len(cl), dtype=bool)
        cvals = [float(np.abs(dDH[f][cl]).sum()) for f, _, _ in per_fold]
        ch = []
        for fname, _, _ in per_fold:
            m = (FOLD == fname) & isD
            ch.append(float((T[m][:, cl] * dDH[fname][cl]).sum(axis=1).mean()))
        flips = sum(1 for j in cl if len({np.sign(dDH[f][j]) for f in dDH}) > 1)
        print(f"| {k} | {len(cl)} | {sub[mm].mean():.4f} | {max(cvals)-min(cvals):.4f} | "
              f"{max(ch)-min(ch):.4f} | {flips == 0} | {len(set(np.sign(ch)))==1} |")

    rule("STEP 12 -- CASE C vs CASE E DISCRIMINATION INPUTS (measurements only)")
    off = R0[~np.eye(n_num, dtype=bool)]
    print(f"  collinearity      : pairs |r|>=0.90 = {int((np.abs(off)>=.90).sum()//2)}   "
          f"clusters = {len(comps)}   clustered features = {sum(len(c) for c in comps)}/79")
    print(f"  raw separation    : mean|SMD| D-H = {np.nanmean(np.abs(raw_dh)):.4f}   "
          f"D-A = {np.nanmean(np.abs(raw_da)):.4f}")
    print(f"  coef stability    : D-H sign flips = {int((~sdf.sign_dh).sum())}/79   "
          f"D-A = {int((~sdf.sign_da).sum())}/79")
    print(f"  flips by collinearity: high-|r| {int((~hi.sign_dh).sum())}/{len(hi)}   "
          f"low-|r| {int((~lo.sign_dh).sum())}/{len(lo)}")
    print("  cluster contribution stability and ablation magnitudes: see STEPS 9 and 10.")
    print("\n  CASE C and CASE E remain distinct and are NOT selected here. No winner is")
    print("  forced if the measurements are ambiguous.")

    rule("STEP 13 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    print("\nFILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")
    print("\nD-39 COMPLETE -- DIAGNOSIS ONLY, NOTHING CHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
