"""D-34 -- mathematical diagnosis of V1's Home/Away directional failures.

    python run_d34_home_away_diagnosis.py

READ-ONLY DIAGNOSTIC. NOT A FIX. NOT PROMOTION EVIDENCE.

Refits the EXISTING V1 production path in memory on the EXISTING three
walk-forward folds and dissects directional (Home vs Away) behaviour.
Nothing is tuned, weighted, resampled, calibrated or thresholded; no
estimator or prediction is persisted; 2025/26 hard-fails the run if it
appears anywhere.

Aborts unless pooled log loss reproduces 0.9993791056968738, so every
figure is provably measured on the real V1 baseline.

INTERPRETATION LIMITS, STATED UP FRONT
  - A large coefficient does NOT mean a feature is important. Inputs are
    standardised but strongly collinear, so coefficient mass is split
    arbitrarily among correlated columns.
  - Feature distribution differences between failure groups are
    ASSOCIATIONS, not causes.
  - Margin buckets are descriptive. They are not decision rules and must
    not be turned into one here.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
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
PROB_BINS = [(i / 10, (i + 1) / 10) for i in range(10)]
MARGIN_BUCKETS = [
    (0.30, 9.0, ">= +0.30"), (0.20, 0.30, "+0.20..+0.30"), (0.10, 0.20, "+0.10..+0.20"),
    (0.0, 0.10, "0..+0.10"), (-0.10, 0.0, "-0.10..0"), (-0.20, -0.10, "-0.20..-0.10"),
    (-0.30, -0.20, "-0.30..-0.20"), (-9.0, -0.30, "<= -0.30"),
]
#: Semantic families over the 80-column contract, matched on name only.
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


def rule(t): print("\n" + "=" * 112); print(t); print("=" * 112)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 112)
    print("HARD FAIL -- D-34 halted. No diagnosis is produced.")
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


def smd(a, b):
    """Standardised mean difference (Cohen's d, pooled SD), NaN-aware."""
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


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
        FINAL_TEST_SEASONS, MODEL_VERSION, REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS,
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
    print(f"  forbidden 2025/26 season_ids: {sorted(test_ids)}")
    folds = list(iter_walk_forward_folds(ds))
    check("exactly three walk-forward folds", len(folds) == 3)
    for fold, tr, va in folds:
        seen = set(tr.metadata["season_id"]) | set(va.metadata["season_id"])
        check(f"{fold.name}: no final-test season_id", not (seen & test_ids))

    rule("STEP 2 -- REPRODUCE THE V1 BASELINE (in-memory refit)")
    P_l, y_l, c_l, f_l, X_l, fid_l = [], [], [], [], [], []
    models, per_fold = {}, []
    for fold, tr, va in folds:
        Xtr, Xva = tr.X[list(MODEL_B_COLUMNS)], va.X[list(MODEL_B_COLUMNS)]
        model, pre, P = train_logistic_regression(Xtr, tr.y, Xva)
        r = evaluate(va.y, P)
        per_fold.append((fold.name, len(va), r))
        models[fold.name] = (model, pre)
        P_l.append(P); y_l.append(va.y.to_numpy())
        c_l.append(va.X["competition_id"].to_numpy())
        f_l.append(np.repeat(fold.name, len(va)))
        X_l.append(Xva.reset_index(drop=True))
        fid_l.append(va.metadata["fixture_id"].to_numpy())
        print(f"  {fold.name}: N={len(va)} log_loss={r.log_loss:.16f} brier={r.brier:.16f}")
    P = np.vstack(P_l); Y = np.concatenate(y_l); CID = np.concatenate(c_l)
    FOLD = np.concatenate(f_l); FID = np.concatenate(fid_l)
    X = pd.concat(X_l, ignore_index=True)
    pooled_ll = float(np.mean([r.log_loss for _, _, r in per_fold]))
    print(f"\n  pooled mean log loss: {pooled_ll:.16f}  (reference {POOLED_REF_LOG_LOSS:.16f})")
    check("pooled log loss reproduces documented V1", abs(pooled_ll - POOLED_REF_LOG_LOSS) < 1e-9,
          f"diff={pooled_ll - POOLED_REF_LOG_LOSS:.3e}")

    iH, iD, iA = 0, 1, 2
    pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    margin = P[:, iH] - P[:, iA]

    def scopes():
        yield "POOLED", np.ones(len(Y), bool)
        for cid, nm in LEAGUES.items():
            yield nm, CID == cid

    rule("STEP 3 -- CONFUSION MATRICES AND DIRECTIONAL ERRORS")
    for nm, m in scopes():
        n_err = int((pred[m] != Y[m]).sum())
        print(f"\n### {nm}   N={int(m.sum())}   errors={n_err}")
        print("| actual \\ pred | H | D | A | row N | H % | D % | A % |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for a in CLASS_ORDER:
            r = [int((m & (Y == a) & (pred == p)).sum()) for p in CLASS_ORDER]
            t = sum(r)
            print(f"| {a} | {r[0]} | {r[1]} | {r[2]} | {t} | " +
                  " | ".join(f"{100*x/t:.1f}%" if t else "n/a" for x in r) + " |")
        print("\n| Error type | count | % of actual class | % of all errors |")
        print("|---|---:|---:|---:|")
        for a, p in (("H", "A"), ("A", "H"), ("H", "D"), ("A", "D"), ("D", "H"), ("D", "A")):
            c = int((m & (Y == a) & (pred == p)).sum())
            base = int((m & (Y == a)).sum())
            print(f"| {a} -> {p} | {c} | {100*c/base:.1f}% | "
                  f"{100*c/n_err if n_err else float('nan'):.1f}% |")

    rule("STEP 4 -- DIRECTIONAL BIAS: margin_HA = P(H) - P(A)")
    print("| Scope | subset | N | mean | p05 | p25 | median | p75 | p95 | min | max |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for nm, m in scopes():
        for lab, sub in (("all", m), ("actual H", m & (Y == "H")),
                         ("actual D", m & (Y == "D")), ("actual A", m & (Y == "A"))):
            a = margin[sub]
            print(f"| {nm} | {lab} | {len(a)} | {a.mean():+.4f} | " +
                  " | ".join(f"{np.percentile(a,p):+.4f}" for p in (5, 25, 50, 75, 95)) +
                  f" | {a.min():+.4f} | {a.max():+.4f} |")

    rule("STEP 5 -- HOME AND AWAY CALIBRATION (one-vs-rest, 10 bins)")
    for cls, idx in (("HOME", iH), ("AWAY", iA)):
        for nm, m in scopes():
            print(f"\n### {cls} -- {nm}")
            print("| Bin | N | actual freq | mean pred | gap (actual-pred) |")
            print("|---|---:|---:|---:|---:|")
            gaps = []
            for lo, hi in PROB_BINS:
                b = m & (P[:, idx] >= lo) & (P[:, idx] < (hi if hi < 1 else 1.01))
                n = int(b.sum())
                if not n:
                    print(f"| {int(lo*100)}-{int(hi*100)}% | 0 | - | - | - |")
                    continue
                af = float((Y[b] == cls[0]).mean()); mp = float(P[b, idx].mean())
                gaps.append(abs(af - mp))
                print(f"| {int(lo*100)}-{int(hi*100)}% | {n} | {af:.4f} | {mp:.4f} | {af-mp:+.4f} |")
            yc = (Y[m] == cls[0]).astype(float)
            print(f"  one-vs-rest Brier: {float(np.mean((P[m,idx]-yc)**2)):.6f}   "
                  f"mean |gap| over populated bins: {np.mean(gaps):.4f}")

    rule("STEP 6 -- ACTUAL HOME vs ACTUAL AWAY PROBABILITY SEPARATION")
    print("| Scope | P(H\\|H) | P(D\\|H) | P(A\\|H) | P(H\\|A) | P(D\\|A) | P(A\\|A) "
          "| Home sep | Away sep |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for nm, m in scopes():
        h, a = m & (Y == "H"), m & (Y == "A")
        hs = P[h, iH].mean() - P[a, iH].mean()
        as_ = P[a, iA].mean() - P[h, iA].mean()
        print(f"| {nm} | {P[h,iH].mean():.4f} | {P[h,iD].mean():.4f} | {P[h,iA].mean():.4f} "
              f"| {P[a,iH].mean():.4f} | {P[a,iD].mean():.4f} | {P[a,iA].mean():.4f} "
              f"| {hs:+.4f} | {as_:+.4f} |")

    rule("STEP 7 -- HIGH-CONFIDENCE DIRECTIONAL FAILURES")
    for truth, wrongcls, idx in (("A", "H", iH), ("H", "A", iA)):
        print(f"\n### actual {truth} predicted {wrongcls}")
        print("| Scope | thr | actual N | failures | rate | mean P(H) | mean P(D) | mean P(A) "
              "| mean H-A margin |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for nm, m in scopes():
            base = m & (Y == truth)
            for thr in (.5, .6, .7, .8):
                f = base & (pred == wrongcls) & (P[:, idx] >= thr)
                n = int(f.sum())
                print(f"| {nm} | >={int(thr*100)}% | {int(base.sum())} | {n} | "
                      f"{100*n/base.sum():.2f}% | " +
                      (f"{P[f,iH].mean():.4f} | {P[f,iD].mean():.4f} | {P[f,iA].mean():.4f} | "
                       f"{margin[f].mean():+.4f} |" if n else "- | - | - | - |"))

    rule("STEP 8 -- MARGIN BUCKETS (descriptive; NOT decision rules)")
    print("| Bucket | N | actual H % | actual D % | actual A % | argmax accuracy |")
    print("|---|---:|---:|---:|---:|---:|")
    for lo, hi, lab in MARGIN_BUCKETS:
        b = (margin >= lo) & (margin < hi)
        n = int(b.sum())
        if not n:
            print(f"| {lab} | 0 | - | - | - | - |")
            continue
        print(f"| {lab} | {n} | {100*(Y[b]=='H').mean():.1f}% | {100*(Y[b]=='D').mean():.1f}% "
              f"| {100*(Y[b]=='A').mean():.1f}% | {100*(pred[b]==Y[b]).mean():.1f}% |")

    rule("STEP 9 -- PER-LEAGUE DIRECTIONAL SUMMARY (per fold and mean)")
    print("| League | Fold | H recall | A recall | H precision | A precision | H->A rate "
          "| A->H rate | mean P(H) | mean P(A) | mean margin |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cid, nm in LEAGUES.items():
        rows = []
        for fname, _, _ in per_fold:
            m = (CID == cid) & (FOLD == fname)
            hr = (pred[m & (Y == "H")] == "H").mean()
            ar = (pred[m & (Y == "A")] == "A").mean()
            hp = (Y[m & (pred == "H")] == "H").mean() if (m & (pred == "H")).any() else np.nan
            ap = (Y[m & (pred == "A")] == "A").mean() if (m & (pred == "A")).any() else np.nan
            ha = (pred[m & (Y == "H")] == "A").mean()
            ah = (pred[m & (Y == "A")] == "H").mean()
            rows.append((hr, ar, hp, ap, ha, ah, P[m, iH].mean(), P[m, iA].mean(), margin[m].mean()))
            print(f"| {nm} | {fname} | " + " | ".join(f"{v:.4f}" for v in rows[-1]) + " |")
        mean = np.nanmean(np.array(rows), axis=0)
        print(f"| **{nm}** | **mean** | " + " | ".join(f"{v:.4f}" for v in mean) + " |")

    rule("STEP 10/12 -- FEATURE ASSOCIATION WITH DIRECTIONAL FAILURES (association only)")
    numeric = [c for c in MODEL_B_COLUMNS if c != "competition_id"]
    groups = {
        "A_away_pred_home": (Y == "A") & (pred == "H"),
        "B_home_pred_away": (Y == "H") & (pred == "A"),
        "C_correct_home": (Y == "H") & (pred == "H"),
        "D_correct_away": (Y == "A") & (pred == "A"),
    }
    for g, m in groups.items():
        print(f"  {g}: N={int(m.sum())}")
    for lab, (ga, gb) in (("Away->Home failures vs correct Away",
                           ("A_away_pred_home", "D_correct_away")),
                          ("Home->Away failures vs correct Home",
                           ("B_home_pred_away", "C_correct_home"))):
        print(f"\n### {lab}  (standardised mean difference, Cohen's d)")
        d = {c: smd(X[c].to_numpy(dtype=float)[groups[ga]],
                    X[c].to_numpy(dtype=float)[groups[gb]]) for c in numeric}
        s = pd.Series(d).dropna().sort_values(key=np.abs, ascending=False)
        print("| Rank | Feature | SMD |")
        print("|---:|---|---:|")
        for i, (c, v) in enumerate(s.head(15).items(), 1):
            print(f"| {i} | {c} | {v:+.4f} |")
        print("\n  family-level mean |SMD|:")
        for fam, pred_fn in FAMILIES.items():
            cs = [c for c in numeric if pred_fn(c)]
            if cs:
                print(f"    {fam:18s} n={len(cs):3d}  mean|SMD|={np.nanmean([abs(d[c]) for c in cs]):.4f}")
    print("\n  These are ASSOCIATIONS between existing feature values and failure groups.")
    print("  No causal claim is made and no feature is added, removed or reweighted.")

    rule("STEP 11 -- COEFFICIENT DIRECTION AND STABILITY ACROSS FOLDS")
    tabs = {}
    for fname, (model, pre) in models.items():
        names = list(pre.numeric_columns) + [f"competition_id={c}" for c in pre.competition_categories]
        cls = list(model.classes_)
        tabs[fname] = (names, cls, model.coef_)
    names0, cls0, _ = tabs[per_fold[0][0]]
    hi_, ai_ = cls0.index("H"), cls0.index("A")
    H = np.vstack([tabs[f][2][hi_] for f, _, _ in per_fold])
    A = np.vstack([tabs[f][2][ai_] for f, _, _ in per_fold])
    for lab, M in (("HOME", H), ("AWAY", A)):
        mean = M.mean(axis=0)
        order = np.argsort(mean)[::-1][:15]
        print(f"\n### top 15 features pushing toward {lab} (by mean coefficient)")
        print("| Feature | fold_1 | fold_2 | fold_3 | mean | sign consistent |")
        print("|---|---:|---:|---:|---:|---|")
        for j in order:
            signs = np.sign(M[:, j])
            print(f"| {names0[j][:46]} | {M[0,j]:+.4f} | {M[1,j]:+.4f} | {M[2,j]:+.4f} "
                  f"| {mean[j]:+.4f} | {bool(np.all(signs == signs[0]))} |")
    opp = np.argsort(-(np.abs(H.mean(0) - A.mean(0))))[:15]
    print("\n### features where Home and Away coefficients most strongly oppose")
    print("| Feature | mean H coef | mean A coef | gap |")
    print("|---|---:|---:|---:|")
    for j in opp:
        print(f"| {names0[j][:46]} | {H.mean(0)[j]:+.4f} | {A.mean(0)[j]:+.4f} "
              f"| {H.mean(0)[j]-A.mean(0)[j]:+.4f} |")
    unstable = [j for j in range(len(names0))
                if not np.all(np.sign(H[:, j]) == np.sign(H[0, j]))
                or not np.all(np.sign(A[:, j]) == np.sign(A[0, j]))]
    print(f"\n  features with a sign flip across folds (H or A): {len(unstable)} of {len(names0)}")
    for j in unstable[:15]:
        print(f"    {names0[j][:46]:46s} H={H[:,j]}  A={A[:,j]}")
    print("\n  A large coefficient does NOT mean a feature is important: the 80 contract")
    print("  columns are strongly collinear, so mass is split arbitrarily among them.")

    rule("STEP 13 -- TOP 20 HIGHEST-CONFIDENCE DIRECTIONAL FAILURES (validation folds only)")
    mcon = sqlite3.connect(f"file:{(REPO/'data/processed/matches.db').resolve()}?mode=ro", uri=True)
    wrong_dir = (((Y == "A") & (pred == "H")) | ((Y == "H") & (pred == "A")))
    conf = P.max(axis=1)
    idxs = np.argsort(-np.where(wrong_dir, conf, -1))[:20]
    print("| League | Fold | Home team | Away team | actual | pred | P(H) | P(D) | P(A) | H-A margin |")
    print("|---|---|---|---|---|---|---:|---:|---:|---:|")
    for i in idxs:
        if not wrong_dir[i]:
            continue
        row = mcon.execute("SELECT home_name, away_name, season_id FROM fixtures WHERE fixture_id=?",
                           (int(FID[i]),)).fetchone()
        if row and row[2] in test_ids:
            stop("a final-test fixture reached the failure table -- firewall breach")
        hn, an = (row[0], row[1]) if row else ("?", "?")
        print(f"| {LEAGUES.get(int(CID[i]),'?')} | {FOLD[i]} | {hn[:20]} | {an[:20]} | {Y[i]} "
              f"| {pred[i]} | {P[i,iH]:.4f} | {P[i,iD]:.4f} | {P[i,iA]:.4f} | {margin[i]:+.4f} |")
    mcon.close()

    rule("STEP 14 -- COMPARISON AGAINST SIMPLE BASELINES (descriptive)")
    print("| Scope | strategy | accuracy | H recall | A recall | balanced acc |")
    print("|---|---|---:|---:|---:|---:|")
    for nm, m in scopes():
        yy = Y[m]
        def bal(pp):
            rs = [(pp[yy == c] == c).mean() if (yy == c).any() else np.nan for c in CLASS_ORDER]
            return np.nanmean(rs)
        cands = {
            "V1 argmax": pred[m],
            "always Home": np.full(len(yy), "H"),
            "always Away": np.full(len(yy), "A"),
            "league prior argmax": np.full(len(yy), pd.Series(yy).value_counts().idxmax()),
        }
        for lab, pp in cands.items():
            print(f"| {nm} | {lab} | {(pp==yy).mean():.4f} | {(pp[yy=='H']=='H').mean():.4f} "
                  f"| {(pp[yy=='A']=='A').mean():.4f} | {bal(pp):.4f} |")

    rule("STEP 15 -- CROSS-FOLD STABILITY (three folds: ranges only)")
    print("| Metric | fold_1 | fold_2 | fold_3 | range |")
    print("|---|---:|---:|---:|---:|")
    mets = {
        "Home recall": lambda m: (pred[m & (Y == "H")] == "H").mean(),
        "Away recall": lambda m: (pred[m & (Y == "A")] == "A").mean(),
        "H->A error rate": lambda m: (pred[m & (Y == "H")] == "A").mean(),
        "A->H error rate": lambda m: (pred[m & (Y == "A")] == "H").mean(),
        "mean H-A margin": lambda m: margin[m].mean(),
        "mean P(H)": lambda m: P[m, iH].mean(),
        "mean P(A)": lambda m: P[m, iA].mean(),
    }
    for lab, fn in mets.items():
        vs = [fn(FOLD == f) for f, _, _ in per_fold]
        print(f"| {lab} | {vs[0]:.4f} | {vs[1]:.4f} | {vs[2]:.4f} | {max(vs)-min(vs):.4f} |")
    print("\n  THREE folds cannot support a standard deviation. Ranges only.")

    rule("STEP 16 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    print("\nFILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")

    rule("D-34 COMPLETE -- DIAGNOSIS ONLY, NOTHING CHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
