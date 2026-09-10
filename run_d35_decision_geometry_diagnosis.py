"""D-35 -- multiclass decision-geometry diagnosis of V1.

    python run_d35_decision_geometry_diagnosis.py

READ-ONLY DIAGNOSTIC. NOT A FIX. NOT PROMOTION EVIDENCE.

THE QUESTION
Aggregate class probabilities are close to the observed frequencies
(mean P(H)=43.25% vs actual 43.63%; P(D)=24.86% vs 25.19%; P(A)=31.90%
vs 31.18%), yet argmax picks Home far more often than the base rate and
Draw almost never (2.4% argmax share, 2.83% recall). This script asks
why, by inspecting the raw multiclass decision scores rather than the
probabilities.

METHOD
Same in-memory V1 refit as D-33/D-34, same folds, same contract, same
CLASS_ORDER, same integrity gates, same 1e-9 reproduction requirement.
Raw scores come from `model.decision_function(pre.transform(X_val))`
using the fitted objects `train_logistic_regression` returns -- no
preprocessing is reimplemented, and STEP 9 proves softmax of those
scores reproduces predict_proba, so the scores are provably the same
object the probabilities came from.

INTERPRETATION RULES ENFORCED IN THE OUTPUT
  - Coefficient magnitude is NOT importance (47/84 coefficients flip
    sign across folds; the 80 columns are strongly collinear).
  - Feature/score associations are not causes.
  - Pairwise competitions are diagnostics, never deployed classifiers.
  - Margin cut-points are descriptive, never thresholds.
  - Conflicting evidence is reported as a conflict, not resolved by
    assumption.
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


def rule(t): print("\n" + "=" * 112); print(t); print("=" * 112)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 112)
    print("HARD FAIL -- D-35 halted. No diagnosis is produced.")
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


def dist(a):
    return (f"{a.mean():+.4f} | {np.percentile(a,5):+.4f} | {np.percentile(a,25):+.4f} | "
            f"{np.median(a):+.4f} | {np.percentile(a,75):+.4f} | {np.percentile(a,95):+.4f} | "
            f"{a.min():+.4f} | {a.max():+.4f}")


def main():
    rule("STEP 1 -- ENVIRONMENT + INTEGRITY (identical gates to D-33/D-34)")
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

    rule("STEP 2 -- REPRODUCE V1 (in-memory refit, configuration unchanged)")
    P_l, S_l, y_l, c_l, f_l = [], [], [], [], []
    parts, per_fold = {}, []
    for fold, tr, va in folds:
        Xtr, Xva = tr.X[list(MODEL_B_COLUMNS)], va.X[list(MODEL_B_COLUMNS)]
        model, pre, P = train_logistic_regression(Xtr, tr.y, Xva)
        r = evaluate(va.y, P)
        per_fold.append((fold.name, len(va), r))

        # Raw scores from the SAME fitted objects. classes_ is alphabetical
        # (A, D, H); reorder to CLASS_ORDER exactly as train._reorder_proba does.
        enc = pre.transform(Xva)
        raw = model.decision_function(enc)
        cls = list(model.classes_)
        order = [cls.index(c) for c in CLASS_ORDER]
        S = raw[:, order]
        inter = model.intercept_[order]
        feat = S - inter                      # linear feature contribution
        parts[fold.name] = (inter, feat, model, pre, cls)

        P_l.append(P); S_l.append(S); y_l.append(va.y.to_numpy())
        c_l.append(va.X["competition_id"].to_numpy())
        f_l.append(np.repeat(fold.name, len(va)))
        print(f"  {fold.name}: N={len(va)} log_loss={r.log_loss:.16f} brier={r.brier:.16f}")

    P = np.vstack(P_l); S = np.vstack(S_l); Y = np.concatenate(y_l)
    CID = np.concatenate(c_l); FOLD = np.concatenate(f_l)
    pooled_ll = float(np.mean([r.log_loss for _, _, r in per_fold]))
    print(f"\n  pooled mean log loss: {pooled_ll:.16f}  (reference {POOLED_REF_LOG_LOSS:.16f})")
    check("pooled log loss reproduces documented V1",
          abs(pooled_ll - POOLED_REF_LOG_LOSS) < 1e-9,
          f"diff={pooled_ll - POOLED_REF_LOG_LOSS:.3e}")

    iH, iD, iA = 0, 1, 2
    pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    M_HD = S[:, iH] - S[:, iD]
    M_HA = S[:, iH] - S[:, iA]
    M_DA = S[:, iD] - S[:, iA]

    def scopes():
        yield "POOLED", np.ones(len(Y), bool)
        for cid, nm in LEAGUES.items():
            yield nm, CID == cid

    rule("STEP 9 (run early -- validates every later score claim) "
         "PROBABILITY vs SCORE CONSISTENCY")
    e = np.exp(S - S.max(axis=1, keepdims=True))
    soft = e / e.sum(axis=1, keepdims=True)
    max_dev = float(np.abs(soft - P).max())
    mismatch = int((soft.argmax(axis=1) != P.argmax(axis=1)).sum())
    print(f"  max |softmax(scores) - predict_proba| : {max_dev:.3e}")
    print(f"  argmax(scores) != argmax(probabilities): {mismatch} of {len(Y)}")
    check("argmax(scores) == argmax(probabilities) for every fixture", mismatch == 0)
    check("softmax of the raw scores reproduces predict_proba", max_dev < 1e-9,
          f"{max_dev:.3e}")
    print("  => any argmax behaviour is a property of the SCORES, not of the softmax.")

    rule("STEP 3 -- SCORE AND PAIRWISE-MARGIN DISTRIBUTIONS")
    print("| Subset | Quantity | N | mean | p05 | p25 | median | p75 | p95 | min | max |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab, m in (("all", np.ones(len(Y), bool)), ("actual H", Y == "H"),
                   ("actual D", Y == "D"), ("actual A", Y == "A")):
        for qn, arr in (("score_H", S[:, iH]), ("score_D", S[:, iD]), ("score_A", S[:, iA]),
                        ("M_HD", M_HD), ("M_HA", M_HA), ("M_DA", M_DA)):
            print(f"| {lab} | {qn} | {int(m.sum())} | {dist(arr[m])} |")

    rule("STEP 3b -- PROBABILITY DISPERSION (why argmax can collapse)")
    print("| Class | mean P | sd P | p05 | p95 | fixtures with P > 1/3 | share |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for i, c in enumerate(CLASS_ORDER):
        a = P[:, i]
        n = int((a > 1 / 3).sum())
        print(f"| {c} | {a.mean():.4f} | {a.std():.4f} | {np.percentile(a,5):.4f} | "
              f"{np.percentile(a,95):.4f} | {n} | {100*n/len(a):.2f}% |")
    print("\n  STRUCTURAL CEILING: with three classes summing to 1, a class can only be")
    print("  argmax if its probability exceeds 1/3 at minimum. The 'P > 1/3' column is")
    print("  therefore an upper bound on how often each class can ever win the argmax,")
    print("  independent of calibration.")

    rule("STEP 4 -- CLASSWISE DECISION-SCORE CENTRES")
    print("| Actual | N | mean score_H | mean score_D | mean score_A | H-D | H-A | D-A |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in CLASS_ORDER:
        m = Y == c
        print(f"| {c} | {int(m.sum())} | {S[m,iH].mean():+.4f} | {S[m,iD].mean():+.4f} | "
              f"{S[m,iA].mean():+.4f} | {M_HD[m].mean():+.4f} | {M_HA[m].mean():+.4f} | "
              f"{M_DA[m].mean():+.4f} |")
    print("\n  Reads directly on the brief's two questions:")
    print(f"    actual draws, score_H above score_D?  mean M_HD = {M_HD[Y=='D'].mean():+.4f}")
    print(f"    actual aways, score_H above score_A?  mean M_HA = {M_HA[Y=='A'].mean():+.4f}")

    rule("STEP 5 -- ARGMAX REGION GEOMETRY")
    print("| Region | N | share | actual H | actual D | actual A | mean P(H) | mean P(D) "
          "| mean P(A) | mean M_HD | mean M_HA | mean M_DA | top-2nd score gap |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    srt = np.sort(S, axis=1)
    gap = srt[:, 2] - srt[:, 1]
    for c in CLASS_ORDER:
        m = pred == c
        if not m.any():
            print(f"| pred {c} | 0 | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        print(f"| pred {c} | {int(m.sum())} | {100*m.mean():.1f}% | "
              f"{100*(Y[m]=='H').mean():.1f}% | {100*(Y[m]=='D').mean():.1f}% | "
              f"{100*(Y[m]=='A').mean():.1f}% | {P[m,iH].mean():.4f} | {P[m,iD].mean():.4f} | "
              f"{P[m,iA].mean():.4f} | {M_HD[m].mean():+.4f} | {M_HA[m].mean():+.4f} | "
              f"{M_DA[m].mean():+.4f} | {gap[m].mean():.4f} |")

    rule("STEP 6 -- PAIRWISE CLASS COMPETITION (diagnostic, never a deployed classifier)")
    for a, b, marg in (("H", "D", M_HD), ("H", "A", M_HA), ("D", "A", M_DA)):
        wa = marg > 0
        rel = np.isin(Y, [a, b])
        acc = (np.where(wa, a, b)[rel] == Y[rel]).mean()
        print(f"\n### {a} vs {b}")
        print(f"  {a} wins: {int(wa.sum())} ({100*wa.mean():.1f}%)   "
              f"{b} wins: {int((~wa).sum())} ({100*(~wa).mean():.1f}%)")
        print(f"  restricted to actual {a}/{b} fixtures (N={int(rel.sum())}): "
              f"pairwise winner correct {100*acc:.1f}%")
        print(f"  margin distribution: mean {marg.mean():+.4f}  median {np.median(marg):+.4f}"
              f"  p05 {np.percentile(marg,5):+.4f}  p95 {np.percentile(marg,95):+.4f}")
        print("  by actual class: " + "  ".join(
            f"{c}: mean={marg[Y==c].mean():+.4f} {a}-wins={100*(marg[Y==c]>0).mean():.1f}%"
            for c in CLASS_ORDER))

    rule("STEP 7 -- DRAW'S POSITION RELATIVE TO H/A (actual draws only)")
    d = Y == "D"
    nd = int(d.sum())
    print(f"  actual draws: {nd}")
    for lab, cond in (("P(D) > P(H)", P[d, iD] > P[d, iH]),
                      ("P(D) > P(A)", P[d, iD] > P[d, iA]),
                      ("P(D) > both", (P[d, iD] > P[d, iH]) & (P[d, iD] > P[d, iA]))):
        print(f"  {lab:16s} {int(cond.sum()):5d}  ({100*cond.mean():.1f}%)")
    print("\n| Ordering | N | % | mean gap to top | mean gap to next |")
    print("|---|---:|---:|---:|---:|")
    orders = {
        "H > D > A": (P[d, iH] > P[d, iD]) & (P[d, iD] > P[d, iA]),
        "A > D > H": (P[d, iA] > P[d, iD]) & (P[d, iD] > P[d, iH]),
        "H > A > D": (P[d, iH] > P[d, iA]) & (P[d, iA] > P[d, iD]),
        "A > H > D": (P[d, iA] > P[d, iH]) & (P[d, iH] > P[d, iD]),
        "D > H > A": (P[d, iD] > P[d, iH]) & (P[d, iH] > P[d, iA]),
        "D > A > H": (P[d, iD] > P[d, iA]) & (P[d, iA] > P[d, iH]),
    }
    Pd = P[d]
    top = Pd.max(axis=1)
    for lab, m in orders.items():
        if not m.any():
            print(f"| {lab} | 0 | 0.0% | - | - |")
            continue
        print(f"| {lab} | {int(m.sum())} | {100*m.mean():.1f}% | "
              f"{(top[m]-Pd[m,iD]).mean():.4f} | "
              f"{(np.sort(Pd[m],axis=1)[:,2]-np.sort(Pd[m],axis=1)[:,1]).mean():.4f} |")
    rank = (Pd > Pd[:, iD][:, None]).sum(axis=1) + 1
    print(f"\n  Draw rank among the three (reconciles with D-33): "
          f"1st={100*(rank==1).mean():.1f}%  2nd={100*(rank==2).mean():.1f}%  "
          f"3rd={100*(rank==3).mean():.1f}%")

    rule("STEP 8 -- HOME/AWAY COMPETITION GEOMETRY ON ACTUAL DRAWS")
    hbd = P[d, iH] > P[d, iD]; abd = P[d, iA] > P[d, iD]; hba = P[d, iH] > P[d, iA]
    cases = {
        "A: Home beats Draw, Away close behind Home (|H-A| < 0.05)":
            hbd & (np.abs(P[d, iH] - P[d, iA]) < .05),
        "B: Home beats Away and Draw is third":
            hba & hbd & abd,
        "C: Home narrowly beats both (margin to 2nd < 0.05)":
            (Pd.argmax(axis=1) == iH) & ((top - np.sort(Pd, axis=1)[:, 1]) < .05),
        "D: Away beats Home but Draw still loses":
            (~hba) & abd,
        "E: Draw wins":
            (Pd.argmax(axis=1) == iD),
    }
    print("| Geometry | N | % of actual draws |")
    print("|---|---:|---:|")
    for lab, m in cases.items():
        print(f"| {lab} | {int(m.sum())} | {100*m.mean():.1f}% |")
    print("  (categories are diagnostic descriptions and may overlap by construction)")

    rule("STEP 10 -- INTERCEPT vs FEATURE CONTRIBUTION")
    print("| Fold | Class | intercept | mean feature contrib | mean total score |")
    print("|---|---|---:|---:|---:|")
    for fname, _, _ in per_fold:
        inter, feat, *_ = parts[fname]
        m = FOLD == fname
        for i, c in enumerate(CLASS_ORDER):
            print(f"| {fname} | {c} | {inter[i]:+.4f} | {feat[:,i].mean():+.4f} | "
                  f"{S[m,i].mean():+.4f} |")
    print("\n| Fold | Pair | intercept diff | mean feature-contrib diff | mean score diff "
          "| intercept share of the gap |")
    print("|---|---|---:|---:|---:|---:|")
    for fname, _, _ in per_fold:
        inter, feat, *_ = parts[fname]
        m = FOLD == fname
        for a, b, ia, ib in (("H", "D", iH, iD), ("H", "A", iH, iA), ("D", "A", iD, iA)):
            di = inter[ia] - inter[ib]
            df = float((feat[:, ia] - feat[:, ib]).mean())
            tot = di + df
            share = f"{100*di/tot:.1f}%" if abs(tot) > 1e-12 else "n/a"
            print(f"| {fname} | {a}-{b} | {di:+.4f} | {df:+.4f} | {tot:+.4f} | {share} |")
    print("\n  This separates 'the class geometry is offset' from 'the features drive it'.")
    print("  No intercept or coefficient is modified.")

    rule("STEP 11 -- PER-LEAGUE DECISION GEOMETRY")
    print("| League | N | mean M_HD | mean M_HA | mean M_DA | pred H | pred D | pred A "
          "| actual H | actual D | actual A |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for nm, m in scopes():
        print(f"| {nm} | {int(m.sum())} | {M_HD[m].mean():+.4f} | {M_HA[m].mean():+.4f} | "
              f"{M_DA[m].mean():+.4f} | {100*(pred[m]=='H').mean():.1f}% | "
              f"{100*(pred[m]=='D').mean():.1f}% | {100*(pred[m]=='A').mean():.1f}% | "
              f"{100*(Y[m]=='H').mean():.1f}% | {100*(Y[m]=='D').mean():.1f}% | "
              f"{100*(Y[m]=='A').mean():.1f}% |")

    rule("STEP 12 -- CROSS-FOLD STABILITY (three folds: ranges only, no SD)")
    print("| Fold | N | mean M_HD | mean M_HA | mean M_DA | pred H | pred D | pred A "
          "| actual H | actual D | actual A | draw 1st | draw 2nd | draw 3rd |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for fname, n, _ in per_fold:
        m = FOLD == fname
        dm = m & (Y == "D")
        Pm = P[dm]
        rk = (Pm > Pm[:, iD][:, None]).sum(axis=1) + 1
        r = (M_HD[m].mean(), M_HA[m].mean(), M_DA[m].mean(),
             (pred[m] == "H").mean(), (pred[m] == "D").mean(), (pred[m] == "A").mean(),
             (Y[m] == "H").mean(), (Y[m] == "D").mean(), (Y[m] == "A").mean(),
             (rk == 1).mean(), (rk == 2).mean(), (rk == 3).mean())
        rows.append(r)
        print(f"| {fname} | {n} | " + " | ".join(f"{v:+.4f}" if i < 3 else f"{100*v:.1f}%"
                                                 for i, v in enumerate(r)) + " |")
    arr = np.array(rows)
    labels = ["M_HD", "M_HA", "M_DA", "pred H", "pred D", "pred A",
              "actual H", "actual D", "actual A", "draw 1st", "draw 2nd", "draw 3rd"]
    print("\n  ranges across the three folds:")
    for i, lab in enumerate(labels):
        print(f"    {lab:10s} min={arr[:,i].min():+.4f}  max={arr[:,i].max():+.4f}  "
              f"range={arr[:,i].max()-arr[:,i].min():.4f}")
    print("\n  THREE folds cannot support a standard deviation. Ranges only.")

    rule("STEP 15 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))

    print("\nD-35 COMPLETE -- DIAGNOSIS ONLY, NOTHING CHANGED")
    print("FILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
