"""D-40 -- conditional / grouped signal recovery diagnosis.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_d40_conditional_group_signal_diagnosis.py

READ-ONLY DIAGNOSTIC. NOTHING IS FIXED, TUNED, REMOVED OR PERSISTED.

THE QUESTION (final measurement inside STEP 5)
D-39 left CASE C and CASE E ambiguous:
  raw Draw separation exists (mean |SMD| 0.2457 / 0.2281)
  coefficient alignment is weak (57.0% / 55.7%; r +0.0692 / +0.0537)
  coefficients are unstable (D-H flips 35/79, D-A 24/79)
  collinearity is real (20/79 features in 8 clusters at |r|>=0.90;
  numerical rank ~59-60/79) and flips concentrate in it (60.0% vs 39.0%)
  but cluster-level contribution stability was MIXED.

D-40 asks whether treating each correlated cluster as ONE information
group recovers a coherent, stable Draw direction, and how much
conditional Draw signal survives inside a cluster once its shared
component is removed.

    CASE C: signal exists in the representation but the fitted
            coefficient geometry transmits it poorly
    CASE E: Draw-relevant structure is not recoverable as a stable
            linear function of the existing contract

The script measures. It does NOT select a case.

TWO IMPLEMENTATION NOTES THAT MATTER FOR READING STEP 8
  1. Group collapsing is EXACTLY additive: for a cluster k,
     C_k = sum_{j in k} x_j (w_Dj - w_Hj). Summing the C_k over clusters
     plus singletons therefore reproduces the feature-level net EXACTLY
     -- the signed net is INVARIANT by construction, and STEP 8 verifies
     that as an identity rather than reporting it as a finding.
     What genuinely changes is the CANCELLATION RATIO: |net| / sum|.|
     computed over 79 feature terms versus over (clusters + singletons)
     terms. Any increase measures within-cluster cancellation.
  2. STEP 9 residualisation is fitted on TRAINING rows only (least
     squares of each member on the other members of its own cluster) and
     then applied to validation rows, so no validation information
     enters the residualiser. It is diagnostic; V1 is never refitted on
     residualised features and nothing is persisted.

STANDING CAVEATS RESTATED IN THE OUTPUT
  - No "feature importance" terminology. These are geometry, contribution
    decompositions and separation measures.
  - No causal claims from correlation, SMD, residualisation or grouping.
  - Three folds => ranges only, never a standard deviation.
  - The |r| >= 0.90 single-linkage clustering is D-39's, reused verbatim
    and not re-thresholded after seeing any result.
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
CLUSTER_THRESHOLD = 0.90          # D-39's pre-registered threshold, unchanged

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


def rule(t): print("\n" + "=" * 120); print(t); print("=" * 120)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 120)
    print("HARD FAIL -- D-40 halted. No diagnosis is produced.")
    print(msg)
    print("!" * 120)
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
    """Single-linkage connected components via union-find (D-39's method)."""
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


def ratio(terms):
    """|net| / sum|terms| per row, averaged; plus the aggregate form."""
    net = np.abs(terms.sum(axis=1))
    tot = np.abs(terms).sum(axis=1)
    per = np.divide(net, tot, out=np.full_like(net, np.nan), where=tot > 0)
    return float(np.nanmean(per)), (float(net.sum() / tot.sum()) if tot.sum() else float("nan"))


def main():
    rule("RULE 1 -- PRODUCTION CODE TRACE (source text, never memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:66]}", tok in tsrc)

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

    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    n_num = len(numeric)
    check("numeric block == contract minus competition_id (79)", n_num == 79)
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

    rule("STEP 2 -- REPRODUCE THE THREE WALK-FORWARD FITS (in memory only)")
    fits, per_fold = {}, []
    S_l, P_l, T_l, y_l, c_l, f_l, X_l = [], [], [], [], [], [], []
    for fold, tr, va in folds:
        Xtr, Xva = tr.X[list(MODEL_B_COLUMNS)], va.X[list(MODEL_B_COLUMNS)]
        model, pre, P = train_logistic_regression(Xtr, tr.y, Xva)
        r = evaluate(va.y, P)
        per_fold.append((fold.name, len(va), r))
        cls = list(model.classes_)
        order = [cls.index(c) for c in CLASS_ORDER]
        Ttr = pre.transform(Xtr)
        Tva = pre.transform(Xva)
        check(f"{fold.name}: transformed width == 84", Tva.shape[1] == n_num + 5)
        check(f"{fold.name}: numeric_columns == contract minus competition_id",
              list(pre.numeric_columns) == numeric)
        fits[fold.name] = dict(model=model, pre=pre, order=order,
                               W=model.coef_[order], b=model.intercept_[order],
                               Ttr=Ttr, ytr=tr.y.to_numpy())
        S_l.append(model.decision_function(Tva)[:, order]); P_l.append(P); T_l.append(Tva)
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
    isD, isH, isA = Y == "D", Y == "H", Y == "A"

    rule(f"STEP 3 -- PRE-REGISTERED COLLINEARITY GROUPS (D-39 method, |r| >= "
         f"{CLUSTER_THRESHOLD}, TRAINING data only)")
    R0 = np.nan_to_num(np.corrcoef(fits[per_fold[0][0]]["Ttr"][:, :n_num], rowvar=False), nan=0.0)
    adj = np.abs(R0) >= CLUSTER_THRESHOLD
    np.fill_diagonal(adj, False)
    allc = components(adj)
    clusters = [c for c in allc if len(c) > 1]
    singles = [c[0] for c in allc if len(c) == 1]
    print(f"  clusters (>=2 members): {len(clusters)}   clustered features: "
          f"{sum(len(c) for c in clusters)}   singletons: {len(singles)}")
    fam_cols, assigned = {}, set()
    for fam, fn in FAMILIES.items():
        cs = [c for c in numeric if fn(c)]
        fam_cols[fam] = cs
        assigned |= set(cs)
    unmatched = [c for c in numeric if c not in assigned]
    idx = {c: i for i, c in enumerate(numeric)}
    for k, cl in enumerate(clusters, 1):
        fams = sorted({fam for fam, cs in fam_cols.items()
                       for c in cs if c in idx and idx[c] in cl})
        sub = np.abs(R0[np.ix_(cl, cl)]); m = ~np.eye(len(cl), dtype=bool)
        print(f"\n  cluster {k}  n={len(cl)}  mean|r|={sub[m].mean():.4f}  "
              f"max|r|={sub[m].max():.4f}  families={','.join(fams) or 'unmatched'}")
        for j in cl:
            print(f"    {numeric[j]}")
    print(f"\n  UNMATCHED by the D-34 taxonomy ({len(unmatched)}): "
          f"{unmatched[:12]}{' ...' if len(unmatched) > 12 else ''}")

    rule("STEP 4 -- GROUP-COLLAPSED RAW SIGNAL (diagnostic; no feature is created)")
    raw_dh = np.array([smd(X[c].to_numpy(float)[isD], X[c].to_numpy(float)[isH]) for c in numeric])
    raw_da = np.array([smd(X[c].to_numpy(float)[isD], X[c].to_numpy(float)[isA]) for c in numeric])
    print("| Cluster | n | mean signed SMD D-H | mean signed SMD D-A | mean \\|SMD\\| "
          "| max \\|SMD\\| | sign coherent (D-H) |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    for k, cl in enumerate(clusters, 1):
        rd, ra = raw_dh[cl], raw_da[cl]
        ok = ~np.isnan(rd)
        print(f"| {k} | {len(cl)} | {np.nanmean(rd):+.4f} | {np.nanmean(ra):+.4f} | "
              f"{np.nanmean(np.abs(rd)):.4f} | {np.nanmax(np.abs(rd)):.4f} | "
              f"{len({np.sign(x) for x in rd[ok]}) == 1} |")
    sg = np.array(singles)
    print(f"| SINGLETONS | {len(sg)} | {np.nanmean(raw_dh[sg]):+.4f} | "
          f"{np.nanmean(raw_da[sg]):+.4f} | {np.nanmean(np.abs(raw_dh[sg])):.4f} | "
          f"{np.nanmax(np.abs(raw_dh[sg])):.4f} | n/a |")

    rule("STEP 5 -- GROUP-LEVEL COEFFICIENT CONTRAST (NOT importance)")
    dDH = {f: fits[f]["W"][iD, :n_num] - fits[f]["W"][iH, :n_num] for f in fits}
    dDA = {f: fits[f]["W"][iD, :n_num] - fits[f]["W"][iA, :n_num] for f in fits}
    print("| Cluster | n | G_DH f1 | f2 | f3 | range | sign stable | G_DA f1 | f2 | f3 "
          "| range | sign stable | member coefs flipping |")
    print("|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|")
    for k, cl in enumerate(clusters, 1):
        gh = [float(dDH[f][cl].sum()) for f, _, _ in per_fold]
        ga = [float(dDA[f][cl].sum()) for f, _, _ in per_fold]
        flips = sum(1 for j in cl if len({np.sign(dDH[f][j]) for f in dDH}) > 1)
        print(f"| {k} | {len(cl)} | {gh[0]:+.4f} | {gh[1]:+.4f} | {gh[2]:+.4f} | "
              f"{max(gh)-min(gh):.4f} | {len(set(np.sign(gh)))==1} | {ga[0]:+.4f} | "
              f"{ga[1]:+.4f} | {ga[2]:+.4f} | {max(ga)-min(ga):.4f} | "
              f"{len(set(np.sign(ga)))==1} | {flips}/{len(cl)} |")
    ind_stable = sum(1 for j in range(n_num) if len({np.sign(dDH[f][j]) for f in dDH}) == 1)
    grp_stable = sum(1 for cl in clusters
                     if len({np.sign(float(dDH[f][cl].sum())) for f, _, _ in per_fold}) == 1)
    print(f"\n  individual D-H coefficients with stable sign : {ind_stable}/{n_num}")
    print(f"  cluster D-H group contrasts with stable sign : {grp_stable}/{len(clusters)}")

    rule("STEP 6 -- GROUP RAW-SIGNAL / FITTED-DIRECTION ALIGNMENT (measurement pattern only)")
    print("| Cluster | n | raw group D-H | fitted G_DH (mean) | pattern D-H "
          "| raw group D-A | fitted G_DA (mean) | pattern D-A |")
    print("|---:|---:|---:|---:|---|---:|---:|---|")

    def pattern(raw, fit):
        if not np.isfinite(raw) or not np.isfinite(fit):
            return "undefined"
        if abs(fit) < 1e-3:
            return "near-zero fitted"
        return "aligned" if np.sign(raw) == np.sign(fit) else "opposed"

    for k, cl in enumerate(clusters, 1):
        rh, ra = float(np.nanmean(raw_dh[cl])), float(np.nanmean(raw_da[cl]))
        gh = float(np.mean([dDH[f][cl].sum() for f in dDH]))
        ga = float(np.mean([dDA[f][cl].sum() for f in dDA]))
        print(f"| {k} | {len(cl)} | {rh:+.4f} | {gh:+.4f} | {pattern(rh, gh)} | "
              f"{ra:+.4f} | {ga:+.4f} | {pattern(ra, ga)} |")

    rule("STEP 7 -- GROUP CONTRIBUTION ON ACTUAL DRAWS + EXACT IDENTITY CHECK")
    print("  contribution convention: positive favours DRAW")
    for a, b, ia, ib in (("D", "H", iD, iH), ("D", "A", iD, iA)):
        inter = np.empty(len(Y)); leag = np.empty(len(Y)); num = np.empty(len(Y))
        for fname, _, _ in per_fold:
            m = FOLD == fname
            f = fits[fname]
            dw = f["W"][ia] - f["W"][ib]
            inter[m] = f["b"][ia] - f["b"][ib]
            leag[m] = T[m][:, n_num:] @ dw[n_num:]
            num[m] = (T[m][:, :n_num] * dw[:n_num]).sum(axis=1)
        total = S[:, ia] - S[:, ib]
        err = float(np.abs(inter + leag + num - total).max())
        check(f"{a}-{b}: intercept + league + numeric == margin", err < 1e-9,
              f"max err {err:.3e}")
        # clusters + singletons must reconstruct the numeric part exactly
        recon = np.zeros(len(Y))
        for fname, _, _ in per_fold:
            m = FOLD == fname
            dw = (fits[fname]["W"][ia] - fits[fname]["W"][ib])[:n_num]
            for cl in clusters:
                recon[m] += (T[m][:, cl] * dw[cl]).sum(axis=1)
            recon[m] += (T[m][:, sg] * dw[sg]).sum(axis=1)
        err2 = float(np.abs(recon - num).max())
        check(f"{a}-{b}: sum(clusters) + sum(singletons) == numeric term", err2 < 1e-9,
              f"max err {err2:.3e}")
        print(f"    {a}-{b} on actual draws: intercept {inter[isD].mean():+.5f}  "
              f"league {leag[isD].mean():+.5f}  numeric {num[isD].mean():+.5f}  "
              f"margin {total[isD].mean():+.5f}")

    rule("STEP 8 -- DECISIVE RECOVERY TEST: feature-level vs group-collapsed")
    print("  NOTE: the SIGNED net is invariant under grouping by construction (verified as")
    print("  an identity in STEP 7). What changes is the CANCELLATION RATIO. An increase")
    print("  from feature-level to group-level measures within-cluster cancellation.\n")
    print("| Pair | level | mean signed | mean \\|.\\| | per-row cancel ratio | aggregate ratio |")
    print("|---|---|---:|---:|---:|---:|")
    grpstats = {}
    for a, b, ia, ib in (("D", "H", iD, iH), ("D", "A", iD, iA)):
        feat_terms = np.zeros((int(isD.sum()), n_num))
        grp_terms = np.zeros((int(isD.sum()), len(clusters) + len(sg)))
        pos = 0
        for fname, _, _ in per_fold:
            m = (FOLD == fname) & isD
            k_ = int(m.sum())
            dw = (fits[fname]["W"][ia] - fits[fname]["W"][ib])[:n_num]
            ft = T[m][:, :n_num] * dw
            feat_terms[pos:pos + k_] = ft
            for gi, cl in enumerate(clusters):
                grp_terms[pos:pos + k_, gi] = ft[:, cl].sum(axis=1)
            grp_terms[pos:pos + k_, len(clusters):] = ft[:, sg]
            pos += k_
        fp, fa = ratio(feat_terms)
        gp, ga_ = ratio(grp_terms)
        grpstats[(a, b)] = (fp, gp, fa, ga_)
        for lab, terms, (pr, ag) in (("feature (79)", feat_terms, (fp, fa)),
                                     (f"grouped ({len(clusters)}+{len(sg)})", grp_terms, (gp, ga_))):
            print(f"| {a}-{b} | {lab} | {terms.sum(axis=1).mean():+.5f} | "
                  f"{np.abs(terms).sum(axis=1).mean():.4f} | {pr:.4f} | {ag:.4f} |")
    print("\n  cluster-only vs singleton-only contribution on actual draws:")
    print("| Pair | block | mean signed f1 | f2 | f3 | range | sign stable |")
    print("|---|---|---:|---:|---:|---:|---|")
    for a, b, ia, ib in (("D", "H", iD, iH), ("D", "A", iD, iA)):
        for lab, cols in (("clusters", np.concatenate(clusters) if clusters else np.array([], int)),
                          ("singletons", sg)):
            v = []
            for fname, _, _ in per_fold:
                m = (FOLD == fname) & isD
                dw = (fits[fname]["W"][ia] - fits[fname]["W"][ib])[:n_num]
                v.append(float((T[m][:, cols] * dw[cols]).sum(axis=1).mean()) if len(cols) else 0.0)
            print(f"| {a}-{b} | {lab} | {v[0]:+.5f} | {v[1]:+.5f} | {v[2]:+.5f} | "
                  f"{max(v)-min(v):.5f} | {len(set(np.sign(v)))==1} |")

    rule("STEP 9 -- CONDITIONAL RESIDUAL SIGNAL (residualiser fitted on TRAINING only)")
    print("  Each cluster member is regressed on the OTHER members of its own cluster using")
    print("  TRAINING rows; the fitted map is applied to VALIDATION rows and the residual's")
    print("  Draw separation is measured. Diagnostic only: V1 is never refitted on residuals")
    print("  and nothing is persisted.\n")
    print("| Cluster | n | original mean \\|SMD\\| D-H | residual mean \\|SMD\\| D-H | retention "
          "| orig sign | resid sign | D-A retention |")
    print("|---:|---:|---:|---:|---:|---|---|---:|")
    for k, cl in enumerate(clusters, 1):
        res_dh, res_da = [], []
        for j_pos, j in enumerate(cl):
            others = [q for q in cl if q != j]
            resid = np.empty(len(Y))
            for fname, _, _ in per_fold:
                m = FOLD == fname
                Atr = fits[fname]["Ttr"][:, others]
                ytr = fits[fname]["Ttr"][:, j]
                Atr1 = np.hstack([Atr, np.ones((len(Atr), 1))])
                coef, *_ = np.linalg.lstsq(Atr1, ytr, rcond=None)
                Ava1 = np.hstack([T[m][:, others], np.ones((int(m.sum()), 1))])
                resid[m] = T[m][:, j] - Ava1 @ coef
            res_dh.append(smd(resid[isD], resid[isH]))
            res_da.append(smd(resid[isD], resid[isA]))
        o_dh = float(np.nanmean(np.abs(raw_dh[cl])))
        r_dh = float(np.nanmean(np.abs(res_dh)))
        o_da = float(np.nanmean(np.abs(raw_da[cl])))
        r_da = float(np.nanmean(np.abs(res_da)))
        print(f"| {k} | {len(cl)} | {o_dh:.4f} | {r_dh:.4f} | "
              f"{(r_dh/o_dh if o_dh > 0 else float('nan')):.4f} | "
              f"{'+' if np.nanmean(raw_dh[cl]) > 0 else '-'} | "
              f"{'+' if np.nanmean(res_dh) > 0 else '-'} | "
              f"{(r_da/o_da if o_da > 0 else float('nan')):.4f} |")
    print("\n  high original + very low residual  -> signal is shared/redundant in the group")
    print("  high original + substantial residual -> members carry distinct conditional info")

    rule("STEP 10 -- FINAL C vs E EVIDENCE TABLE (measurements only; no case selected)")
    fpH, gpH, _, _ = grpstats[("D", "H")]
    fpA, gpA, _, _ = grpstats[("D", "A")]
    hi = [j for j in range(n_num) if (np.abs(R0[j]).max() if np.abs(R0[j]).max() < 1
                                      else np.sort(np.abs(R0[j]))[-2]) >= CLUSTER_THRESHOLD]
    lo = [j for j in range(n_num) if j not in hi]
    fl_hi = sum(1 for j in hi if len({np.sign(dDH[f][j]) for f in dDH}) > 1)
    fl_lo = sum(1 for j in lo if len({np.sign(dDH[f][j]) for f in dDH}) > 1)
    print("| Row | Measurement | Value |")
    print("|---|---|---|")
    print(f"| 1 | Raw Draw separation survives grouping | cluster mean\\|SMD\\| vs singleton "
          f"mean\\|SMD\\| -- see STEP 4 |")
    print(f"| 2 | Group coefficient direction stability | {grp_stable}/{len(clusters)} clusters "
          f"sign-stable vs {ind_stable}/{n_num} individual features |")
    print(f"| 3 | Group contribution stability | see STEP 8 cluster/singleton blocks |")
    print(f"| 4 | Group/raw directional alignment | see STEP 6 pattern column |")
    print(f"| 5 | Conditional residual retention | see STEP 9 retention ratios |")
    print(f"| 6 | High- vs low-collinearity stability | D-H flips {fl_hi}/{len(hi)} high-\\|r\\| "
          f"vs {fl_lo}/{len(lo)} low-\\|r\\| |")
    print(f"| 7 | Cancellation ratio, feature -> grouped | D-H {fpH:.4f} -> {gpH:.4f}   "
          f"D-A {fpA:.4f} -> {gpA:.4f} |")

    rule("STEP 11 -- CROSS-FOLD STABILITY (ranges only; no SD; fold disagreement preserved)")
    print("| Metric | fold_1 | fold_2 | fold_3 | min | max | range |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    mets = {}
    for fname, _, _ in per_fold:
        m = (FOLD == fname) & isD
        dwH = (fits[fname]["W"][iD] - fits[fname]["W"][iH])[:n_num]
        dwA = (fits[fname]["W"][iD] - fits[fname]["W"][iA])[:n_num]
        ftH = T[m][:, :n_num] * dwH
        gtH = np.column_stack([ftH[:, cl].sum(axis=1) for cl in clusters] + [ftH[:, sg]])
        mets[fname] = [float(ftH.sum(axis=1).mean()),
                       float((T[m][:, :n_num] * dwA).sum(axis=1).mean()),
                       ratio(ftH)[0], ratio(gtH)[0],
                       float(np.mean([dDH[fname][cl].sum() for cl in clusters])),
                       float(np.abs(ftH).sum(axis=1).mean())]
    labs = ["net D-H contribution", "net D-A contribution", "cancel ratio (feature)",
            "cancel ratio (grouped)", "mean cluster G_DH", "mean |D-H| contribution"]
    for i, lab in enumerate(labs):
        v = [mets[f][i] for f, _, _ in per_fold]
        print(f"| {lab} | {v[0]:+.4f} | {v[1]:+.4f} | {v[2]:+.4f} | {min(v):+.4f} | "
              f"{max(v):+.4f} | {max(v)-min(v):.4f} |")

    rule("STEP 12 -- PRE-REGISTERED DIAGNOSIS INPUTS")
    print(f"  collinear clusters                 : {len(clusters)}")
    print(f"  clustered features                 : {sum(len(c) for c in clusters)}/{n_num}  "
          f"(singletons {len(sg)})")
    print(f"  raw group signal (cluster mean|SMD| D-H): "
          f"{np.nanmean([np.nanmean(np.abs(raw_dh[cl])) for cl in clusters]):.4f}   "
          f"singleton mean|SMD|: {np.nanmean(np.abs(raw_dh[sg])):.4f}")
    print(f"  group coefficient stability        : {grp_stable}/{len(clusters)} clusters "
          f"sign-stable (individual: {ind_stable}/{n_num})")
    print(f"  cancellation ratio feature->grouped: D-H {fpH:.4f} -> {gpH:.4f}   "
          f"D-A {fpA:.4f} -> {gpA:.4f}")
    print(f"  high vs low collinearity D-H flips : {fl_hi}/{len(hi)} vs {fl_lo}/{len(lo)}")
    print("  conditional residual retention     : see STEP 9")
    print("\n  CASE C and CASE E remain distinct and are NOT selected by this script.")

    rule("STEP 13 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    print("\nFILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")
    print("\nD-40 COMPLETE -- DIAGNOSIS ONLY, NOTHING CHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
