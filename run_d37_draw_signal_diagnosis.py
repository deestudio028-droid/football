"""D-37 -- Draw signal / identifiability investigation.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_d37_draw_signal_diagnosis.py

READ-ONLY DIAGNOSTIC. NOTHING IS FIXED, TUNED, CALIBRATED OR PERSISTED.

CORE QUESTION
Does the existing 80-feature representation contain usable, separable
Draw signal, and if so where is it lost between raw feature space and
the final argmax?

The four layers are kept strictly separate throughout, per rule 13:
    (1) raw feature-space signal      -> STEPS 3, 4, 5, 6
    (2) linear score contribution     -> STEP 7
    (3) probability geometry          -> STEPS 9, 10
    (4) argmax behaviour              -> STEPS 8, 10

METHODOLOGY PRESERVED FROM D-33/D-34/D-35/D-36
Same integrity gates, same in-memory refit through
`train.train_logistic_regression`, same folds, same contract, same
CLASS_ORDER, same 1e-9 reproduction requirement, same feature-family
taxonomy as D-34 (not redefined here), same "three folds => ranges, not
SD" rule, same read-only production contract.

STANDING CAVEATS RESTATED IN THE OUTPUT
  - Coefficient magnitude is NOT feature importance (D-34 found 47/84
    sign flips across folds; the 80 columns are strongly collinear).
  - SMD and AUC are ASSOCIATIONS, never causes.
  - P(D) > 1/3 is a NECESSARY GEOMETRIC CONDITION for Draw to be
    argmax, not evidence of learnability.
  - Univariate AUC is computed on non-null rows only; coverage is
    reported alongside every value so a high AUC on thin coverage
    cannot be mistaken for a strong signal.
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

#: D-35 reproducibility checks for STEP 10.
D35_DRAW_GT_H = 0.220
D35_DRAW_GT_A = 0.451
D35_DRAW_GT_BOTH = 0.028


def rule(t): print("\n" + "=" * 114); print(t); print("=" * 114)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 114)
    print("HARD FAIL -- D-37 halted. No diagnosis is produced.")
    print(msg)
    print("!" * 114)
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
    """Standardised mean difference (Cohen's d, pooled SD), NaN-aware."""
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def qtiles(a):
    a = a[~np.isnan(a)]
    return (f"{a.mean():+.4f} | {np.median(a):+.4f} | {a.std():.4f} | "
            f"{np.percentile(a,5):+.4f} | {np.percentile(a,25):+.4f} | "
            f"{np.percentile(a,75):+.4f} | {np.percentile(a,95):+.4f} | {a.max():+.4f}")


def main():
    rule("STEP 1 -- ENVIRONMENT + INTEGRITY")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")
    try:
        import sklearn
        from sklearn.metrics import roc_auc_score
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

    rule("STEP 2 -- REPRODUCE V1 (in-memory refit; no artifact substituted)")
    fits, per_fold = {}, []
    S_l, P_l, y_l, c_l, f_l, X_l = [], [], [], [], [], []
    for fold, tr, va in folds:
        Xtr, Xva = tr.X[list(MODEL_B_COLUMNS)], va.X[list(MODEL_B_COLUMNS)]
        model, pre, P = train_logistic_regression(Xtr, tr.y, Xva)
        r = evaluate(va.y, P)
        per_fold.append((fold.name, len(va), r))
        cls = list(model.classes_)
        order = [cls.index(c) for c in CLASS_ORDER]
        S = model.decision_function(pre.transform(Xva))[:, order]
        fits[fold.name] = dict(model=model, pre=pre, b=model.intercept_[order])
        S_l.append(S); P_l.append(P); y_l.append(va.y.to_numpy())
        c_l.append(va.X[RECOMMENDED_CONTEXT_FEATURE].to_numpy())
        f_l.append(np.repeat(fold.name, len(va)))
        X_l.append(Xva.reset_index(drop=True))
        print(f"  {fold.name}: N={len(va)} log_loss={r.log_loss:.16f} brier={r.brier:.16f}")
    S = np.vstack(S_l); P = np.vstack(P_l); Y = np.concatenate(y_l)
    CID = np.concatenate(c_l); FOLD = np.concatenate(f_l)
    X = pd.concat(X_l, ignore_index=True)
    pooled_ll = float(np.mean([r.log_loss for _, _, r in per_fold]))
    print(f"\n  pooled mean log loss: {pooled_ll:.16f}  (reference {POOLED_REF_LOG_LOSS:.16f})")
    check("pooled log loss reproduces documented V1",
          abs(pooled_ll - POOLED_REF_LOG_LOSS) < 1e-9,
          f"diff={pooled_ll - POOLED_REF_LOG_LOSS:.3e}")

    iH, iD, iA = 0, 1, 2
    pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    M_HD = S[:, iH] - S[:, iD]
    M_DA = S[:, iD] - S[:, iA]
    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    isD, isH, isA = Y == "D", Y == "H", Y == "A"

    rule("STEP 3 -- LAYER 1: RAW FEATURE SPACE, DRAW vs HOME AND DRAW vs AWAY")
    print("  Association only. NOT importance. NOT causal. Compare against STEP 7,")
    print("  which measures the fitted linear contribution -- a different layer.\n")
    dh = {c: smd(X[c].to_numpy(float)[isD], X[c].to_numpy(float)[isH]) for c in numeric}
    da = {c: smd(X[c].to_numpy(float)[isD], X[c].to_numpy(float)[isA]) for c in numeric}
    sdh = pd.Series(dh).dropna(); sda = pd.Series(da).dropna()
    print(f"  Draw vs Home : max|SMD|={sdh.abs().max():.4f}  mean|SMD|={sdh.abs().mean():.4f}  "
          f"features with |SMD|>=0.2: {int((sdh.abs()>=.2).sum())}/{len(sdh)}")
    print(f"  Draw vs Away : max|SMD|={sda.abs().max():.4f}  mean|SMD|={sda.abs().mean():.4f}  "
          f"features with |SMD|>=0.2: {int((sda.abs()>=.2).sum())}/{len(sda)}")
    for lab, s in (("Draw vs Home", sdh), ("Draw vs Away", sda)):
        print(f"\n  top 15 by |SMD| -- {lab}")
        print("| Rank | Feature | mean(D) | mean(other) | SMD | direction |")
        print("|---:|---|---:|---:|---:|---|")
        other = isH if lab.endswith("Home") else isA
        for i, (c, v) in enumerate(s.reindex(s.abs().sort_values(ascending=False).index)
                                   .head(15).items(), 1):
            print(f"| {i} | {c[:44]} | {np.nanmean(X[c].to_numpy(float)[isD]):+.4f} | "
                  f"{np.nanmean(X[c].to_numpy(float)[other]):+.4f} | {v:+.4f} | "
                  f"{'D higher' if v > 0 else 'D lower'} |")
    print("\n  per-fold |SMD| summary:")
    print("| Fold | Draw-vs-Home mean\\|SMD\\| | max | Draw-vs-Away mean\\|SMD\\| | max |")
    print("|---|---:|---:|---:|---:|")
    fold_smd = {}
    for fname, _, _ in per_fold:
        m = FOLD == fname
        a = pd.Series({c: smd(X[c].to_numpy(float)[m & isD], X[c].to_numpy(float)[m & isH])
                       for c in numeric}).dropna()
        b = pd.Series({c: smd(X[c].to_numpy(float)[m & isD], X[c].to_numpy(float)[m & isA])
                       for c in numeric}).dropna()
        fold_smd[fname] = (a, b)
        print(f"| {fname} | {a.abs().mean():.4f} | {a.abs().max():.4f} | "
              f"{b.abs().mean():.4f} | {b.abs().max():.4f} |")

    rule("STEP 4 -- FEATURE-FAMILY DRAW SEPARATION (D-34 taxonomy, unchanged)")
    assigned = set()
    fam_cols = {}
    for fam, fn in FAMILIES.items():
        cs = [c for c in numeric if fn(c)]
        fam_cols[fam] = cs
        assigned |= set(cs)
    unmatched = [c for c in numeric if c not in assigned]
    print(f"  features covered by the existing taxonomy: {len(assigned)}/{len(numeric)}")
    if unmatched:
        print(f"  NOT REPRESENTABLE by the D-34 taxonomy ({len(unmatched)}): "
              f"{unmatched[:12]}{' ...' if len(unmatched) > 12 else ''}")
        print("  Reported explicitly rather than inventing a new family, per the brief.")
    print("\n| Family | n | mean\\|SMD\\| D-vs-H | mean\\|SMD\\| D-vs-A | f1 D-H | f2 D-H | f3 D-H "
          "| direction consistent (D-H) |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for fam, cs in fam_cols.items():
        if not cs:
            print(f"| {fam} | 0 | - | - | - | - | - | - |")
            continue
        pv = [np.nanmean([abs(fold_smd[f][0].get(c, np.nan)) for c in cs])
              for f, _, _ in per_fold]
        signs = [np.sign(np.nanmean([fold_smd[f][0].get(c, np.nan) for c in cs]))
                 for f, _, _ in per_fold]
        print(f"| {fam} | {len(cs)} | {np.nanmean([abs(dh[c]) for c in cs]):.4f} | "
              f"{np.nanmean([abs(da[c]) for c in cs]):.4f} | "
              f"{pv[0]:.4f} | {pv[1]:.4f} | {pv[2]:.4f} | "
              f"{bool(len(set(signs)) == 1)} |")

    rule("STEP 5 -- SIGN STABILITY ACROSS FOLDS (ranges and sign consistency; no SD)")
    rows = []
    for c in numeric:
        v = [fold_smd[f][0].get(c, np.nan) for f, _, _ in per_fold]
        w = [fold_smd[f][1].get(c, np.nan) for f, _, _ in per_fold]
        rows.append((c, v, w))
    cons_dh = [c for c, v, _ in rows if not np.isnan(v).any() and len(set(np.sign(v))) == 1]
    cons_da = [c for c, _, w in rows if not np.isnan(w).any() and len(set(np.sign(w))) == 1]
    print(f"  Draw-vs-Home: consistent sign across all 3 folds: {len(cons_dh)}/{len(numeric)}")
    print(f"  Draw-vs-Away: consistent sign across all 3 folds: {len(cons_da)}/{len(numeric)}")
    print("\n  features that are BOTH sign-consistent AND |pooled SMD| >= 0.10 (D vs H):")
    print("| Feature | f1 | f2 | f3 | pooled SMD | range |")
    print("|---|---:|---:|---:|---:|---:|")
    shown = 0
    for c, v, _ in rows:
        if c in cons_dh and abs(dh[c]) >= .10:
            print(f"| {c[:44]} | {v[0]:+.4f} | {v[1]:+.4f} | {v[2]:+.4f} | {dh[c]:+.4f} "
                  f"| {max(v)-min(v):.4f} |")
            shown += 1
    if not shown:
        print("| (none) | - | - | - | - | - |")
    print(f"\n  count: {shown}. Near-zero separation is reported as such, not as absence")
    print("  of signal in some other representation.")

    rule("STEP 6 -- LAYER 1: UNIVARIATE DRAW DISCRIMINATION (ROC-AUC, diagnostic only)")
    print("  AUC computed on non-null rows ONLY; coverage reported so a high AUC on thin")
    print("  coverage cannot be read as a strong signal. 0.5 == chance.\n")

    def auc_pair(col, pos_mask, neg_mask):
        v = X[col].to_numpy(float)
        m = (pos_mask | neg_mask) & ~np.isnan(v)
        if m.sum() < 30 or len(set((pos_mask[m]).astype(int))) < 2:
            return float("nan"), float(m.mean())
        return float(roc_auc_score(pos_mask[m].astype(int), v[m])), float(m.mean())

    auc_dh, auc_da, cov = {}, {}, {}
    for c in numeric:
        auc_dh[c], cov[c] = auc_pair(c, isD, isH)
        auc_da[c], _ = auc_pair(c, isD, isA)
    for lab, d in (("Draw vs Home", auc_dh), ("Draw vs Away", auc_da)):
        s = pd.Series(d).dropna()
        dev = (s - .5).abs().sort_values(ascending=False)
        print(f"\n  {lab}: mean |AUC-0.5| = {(s-.5).abs().mean():.4f}   "
              f"max |AUC-0.5| = {(s-.5).abs().max():.4f}   "
              f"features with |AUC-0.5| >= 0.05: {int(((s-.5).abs()>=.05).sum())}/{len(s)}")
        print("| Rank | Feature | AUC | AUC-0.5 | coverage | f1 | f2 | f3 | consistent side |")
        print("|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for i, c in enumerate(dev.head(15).index, 1):
            fa = []
            for fname, _, _ in per_fold:
                m = FOLD == fname
                pm = isD & m
                nm = (isH if lab.endswith("Home") else isA) & m
                fa.append(auc_pair(c, pm, nm)[0])
            side = len({np.sign(x - .5) for x in fa if not np.isnan(x)}) == 1
            print(f"| {i} | {c[:40]} | {d[c]:.4f} | {d[c]-.5:+.4f} | {cov[c]:.3f} | " +
                  " | ".join(f"{x:.4f}" if not np.isnan(x) else "n/a" for x in fa) +
                  f" | {side} |")

    rule("STEP 7 -- LAYER 2: LINEAR SCORE CONTRIBUTION ON ACTUAL DRAWS")
    print("| Fold | subset | N | feat contrib H | feat contrib D | feat contrib A "
          "| H-D | H-A | D-A |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for fname, _, _ in per_fold:
        f = fits[fname]
        m = FOLD == fname
        feat = S[m] - f["b"]
        yy = Y[m]
        for lab, sub in (("actual D", yy == "D"), ("actual H", yy == "H"),
                         ("actual A", yy == "A")):
            fc = feat[sub]
            print(f"| {fname} | {lab} | {int(sub.sum())} | {fc[:,iH].mean():+.4f} | "
                  f"{fc[:,iD].mean():+.4f} | {fc[:,iA].mean():+.4f} | "
                  f"{(fc[:,iH]-fc[:,iD]).mean():+.4f} | {(fc[:,iH]-fc[:,iA]).mean():+.4f} | "
                  f"{(fc[:,iD]-fc[:,iA]).mean():+.4f} |")
    print("\n  Question this answers: on a real draw, does the FEATURE part of the score")
    print("  move Draw up relative to Home/Away, or does it leave Draw stranded between")
    print("  them? Compare the 'actual D' rows against 'actual H' and 'actual A'.")

    rule("STEP 8 -- LAYER 4: ORDERING REGIONS AMONG ACTUAL DRAWS")
    Pd = P[isD]
    regions = {
        "1 H>D>A": (Pd[:, iH] > Pd[:, iD]) & (Pd[:, iD] > Pd[:, iA]),
        "2 A>D>H": (Pd[:, iA] > Pd[:, iD]) & (Pd[:, iD] > Pd[:, iH]),
        "3 H>A>D": (Pd[:, iH] > Pd[:, iA]) & (Pd[:, iA] > Pd[:, iD]),
        "4 A>H>D": (Pd[:, iA] > Pd[:, iH]) & (Pd[:, iH] > Pd[:, iD]),
        "5 D>H>A": (Pd[:, iD] > Pd[:, iH]) & (Pd[:, iH] > Pd[:, iA]),
        "6 D>A>H": (Pd[:, iD] > Pd[:, iA]) & (Pd[:, iA] > Pd[:, iH]),
    }
    MHD_d, MDA_d = M_HD[isD], M_DA[isD]
    Sd = S[isD]
    print("| Region | N | share | mean P(H) | mean P(D) | mean P(A) | mean M_HD | mean M_DA "
          "| mean score_H | mean score_D | mean score_A |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab, m in regions.items():
        if not m.any():
            print(f"| {lab} | 0 | 0.0% | - | - | - | - | - | - | - | - |")
            continue
        print(f"| {lab} | {int(m.sum())} | {100*m.mean():.1f}% | {Pd[m,iH].mean():.4f} | "
              f"{Pd[m,iD].mean():.4f} | {Pd[m,iA].mean():.4f} | {MHD_d[m].mean():+.4f} | "
              f"{MDA_d[m].mean():+.4f} | {Sd[m,iH].mean():+.4f} | {Sd[m,iD].mean():+.4f} | "
              f"{Sd[m,iA].mean():+.4f} |")
    print("\n  strongest associated families per region (mean |SMD| vs all other draws):")
    Xd = X[isD].reset_index(drop=True)
    for lab, m in regions.items():
        if m.sum() < 30 or (~m).sum() < 30:
            print(f"    {lab}: N too small for a family comparison ({int(m.sum())})")
            continue
        fam_scores = {}
        for fam, cs in fam_cols.items():
            if cs:
                fam_scores[fam] = np.nanmean(
                    [abs(smd(Xd[c].to_numpy(float)[m], Xd[c].to_numpy(float)[~m])) for c in cs])
        top = sorted(fam_scores.items(), key=lambda kv: -kv[1])[:3]
        print(f"    {lab}: " + ", ".join(f"{k}={v:.4f}" for k, v in top))

    rule("STEP 9 -- LAYER 3: P(D) DISPERSION AND STRUCTURAL CEILING")
    print("  CAVEAT: P(D) > 1/3 is a NECESSARY GEOMETRIC CONDITION for Draw to become")
    print("  argmax. It is NOT evidence that Draw is learnable, and must not be read as")
    print("  such.\n")
    print("| Subset | N | mean | median | SD | p05 | p25 | p75 | p95 | max | P(D)>1/3 | share |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab, m in (("all fixtures", np.ones(len(Y), bool)), ("actual Draw", isD),
                   ("actual Home", isH), ("actual Away", isA)):
        a = P[m, iD]
        n = int((a > 1 / 3).sum())
        print(f"| {lab} | {int(m.sum())} | {a.mean():.4f} | {np.median(a):.4f} | {a.std():.4f} | "
              + " | ".join(f"{np.percentile(a,p):.4f}" for p in (5, 25, 75, 95)) +
              f" | {a.max():.4f} | {n} | {100*n/len(a):.2f}% |")

    rule("STEP 10 -- DRAW PAIRWISE SEPARATION (+ D-35 reproducibility checks)")
    print("| Subset | quantity | mean | median | SD | p05 | p25 | p75 | p95 | max |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab, m in (("actual Draw", isD), ("actual Home", isH), ("actual Away", isA)):
        print(f"| {lab} | M_HD | {qtiles(M_HD[m])} |")
        print(f"| {lab} | M_DA | {qtiles(M_DA[m])} |")
    gh = float((P[isD, iD] > P[isD, iH]).mean())
    ga = float((P[isD, iD] > P[isD, iA]).mean())
    gb = float(((P[isD, iD] > P[isD, iH]) & (P[isD, iD] > P[isD, iA])).mean())
    print(f"\n  D > H  : {100*gh:.1f}%   (D-35 recorded {100*D35_DRAW_GT_H:.1f}%)")
    print(f"  D > A  : {100*ga:.1f}%   (D-35 recorded {100*D35_DRAW_GT_A:.1f}%)")
    print(f"  D > both: {100*gb:.1f}%   (D-35 recorded {100*D35_DRAW_GT_BOTH:.1f}%)")
    for lab, got, want in (("D>H", gh, D35_DRAW_GT_H), ("D>A", ga, D35_DRAW_GT_A),
                           ("D>both", gb, D35_DRAW_GT_BOTH)):
        check(f"reproduces D-35 {lab} within 1pp", abs(got - want) <= .01,
              f"got {100*got:.1f}% vs {100*want:.1f}%")

    rule("STEP 11 -- LEAGUE-SPECIFIC DRAW SIGNAL")
    print("| League | N | draws | mean P(D) | SD P(D) | Draw argmax | rank1 | rank2 | rank3 "
          "| mean M_HD | mean M_DA | Draw AUC | mean\\|SMD\\| D-H | mean\\|SMD\\| D-A |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cid, nm in LEAGUES.items():
        m = CID == cid
        d = m & isD
        Pm = P[d]
        rk = (Pm > Pm[:, iD][:, None]).sum(axis=1) + 1
        yD = isD[m].astype(int)
        auc = roc_auc_score(yD, P[m, iD]) if 0 < yD.sum() < len(yD) else float("nan")
        a = np.nanmean([abs(smd(X[c].to_numpy(float)[m & isD], X[c].to_numpy(float)[m & isH]))
                        for c in numeric])
        b = np.nanmean([abs(smd(X[c].to_numpy(float)[m & isD], X[c].to_numpy(float)[m & isA]))
                        for c in numeric])
        print(f"| {nm} | {int(m.sum())} | {int(d.sum())} | {P[m,iD].mean():.4f} | "
              f"{P[m,iD].std():.4f} | {100*(pred[m]=='D').mean():.2f}% | "
              f"{100*(rk==1).mean():.1f}% | {100*(rk==2).mean():.1f}% | {100*(rk==3).mean():.1f}% | "
              f"{M_HD[m].mean():+.4f} | {M_DA[m].mean():+.4f} | {auc:.4f} | {a:.4f} | {b:.4f} |")

    rule("STEP 12 -- CROSS-FOLD STABILITY (ranges only; three folds cannot support SD)")
    print("| Metric | fold_1 | fold_2 | fold_3 | min | max | range |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    mets = {
        "Draw argmax share": lambda m: (pred[m] == "D").mean(),
        "mean P(D)": lambda m: P[m, iD].mean(),
        "SD P(D)": lambda m: P[m, iD].std(),
        "P(D)>1/3 share": lambda m: (P[m, iD] > 1 / 3).mean(),
        "Draw AUC (P(D))": lambda m: roc_auc_score(isD[m].astype(int), P[m, iD]),
        "mean M_HD": lambda m: M_HD[m].mean(),
        "mean M_DA": lambda m: M_DA[m].mean(),
        "mean|SMD| D-vs-H": lambda m: fold_smd[FOLD[m][0]][0].abs().mean(),
        "mean|SMD| D-vs-A": lambda m: fold_smd[FOLD[m][0]][1].abs().mean(),
    }
    for lab, fn in mets.items():
        v = [fn(FOLD == f) for f, _, _ in per_fold]
        print(f"| {lab} | {v[0]:+.4f} | {v[1]:+.4f} | {v[2]:+.4f} | {min(v):+.4f} | "
              f"{max(v):+.4f} | {max(v)-min(v):.4f} |")

    rule("STEP 13 -- EVIDENCE MATRIX INPUTS (measurements only; no case selected here)")
    print("  The following are the discriminating quantities. The written diagnosis is")
    print("  produced separately from this output, not inside the script.\n")
    print(f"  L1 raw signal   : mean|SMD| D-vs-H = {sdh.abs().mean():.4f}, "
          f"D-vs-A = {sda.abs().mean():.4f}; max = {sdh.abs().max():.4f}/{sda.abs().max():.4f}")
    print(f"  L1 univariate   : mean |AUC-0.5| D-vs-H = "
          f"{(pd.Series(auc_dh).dropna()-.5).abs().mean():.4f}, D-vs-A = "
          f"{(pd.Series(auc_da).dropna()-.5).abs().mean():.4f}")
    print(f"  L2 contribution : see STEP 7 'actual D' rows")
    print(f"  L3 geometry     : SD P(D) = {P[:,iD].std():.4f}; "
          f"P(D)>1/3 = {100*(P[:,iD]>1/3).mean():.2f}%")
    print(f"  L4 argmax       : Draw argmax = {100*(pred=='D').mean():.2f}%; "
          f"D>both = {100*gb:.1f}%")
    print(f"  aggregate check : mean P(D) = {P[:,iD].mean():.4f} vs actual "
          f"{float(isD.mean()):.4f}  (|gap| = {abs(P[:,iD].mean()-isD.mean()):.4f})")

    rule("STEP 15 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    print("\nFILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")
    print("\nD-37 COMPLETE -- DIAGNOSIS ONLY, NOTHING CHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
