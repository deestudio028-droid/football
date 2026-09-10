"""D-33 -- mathematical diagnosis of V1's Draw behaviour.

    python run_d33_draw_diagnosis.py

READ-ONLY DIAGNOSTIC. NOT A FIX. NOT PROMOTION EVIDENCE.

Refits the EXISTING V1 production path in memory on the EXISTING three
walk-forward folds, then dissects the Draw probability. Nothing is
tuned, weighted, resampled, calibrated, thresholded or persisted; no
artifact is created or touched; 2025/26 is firewalled out and hard-fails
the run if it appears anywhere.

The run aborts unless the pooled log loss reproduces
0.9993791056968738, so every number below is provably measured on the
real V1 baseline rather than something subtly different.

ONE NOTE ON METHOD: ROC-AUC and PR-AUC come from sklearn.metrics rather
than a hand-rolled implementation. They are descriptive diagnostics, not
production metrics -- the project's own log loss / Brier / confusion
matrix continue to come from models.evaluate, which is untouched.
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
POOLED_REF_BRIER = 0.5965016957578898
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
BINS = [(0, .05), (.05, .10), (.10, .15), (.15, .20), (.20, .25),
        (.25, .30), (.30, .40), (.40, .50), (.50, 1.01)]
BIN_LABEL = ["0-5%", "5-10%", "10-15%", "15-20%", "20-25%",
             "25-30%", "30-40%", "40-50%", "50%+"]


def rule(t): print("\n" + "=" * 110); print(t); print("=" * 110)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 110)
    print("HARD FAIL -- D-33 halted. No diagnosis is produced.")
    print(msg)
    print("!" * 110)
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
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    for rel in pins:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    rule("STEP 1 -- ENVIRONMENT + INTEGRITY")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")
    try:
        import sklearn
        from sklearn.metrics import average_precision_score, roc_auc_score
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
    P_all, y_all, cid_all, fold_all = [], [], [], []
    coefs = {}
    per_fold = []
    for fold, tr, va in folds:
        Xtr = tr.X[list(MODEL_B_COLUMNS)]
        Xva = va.X[list(MODEL_B_COLUMNS)]
        model, pre, P = train_logistic_regression(Xtr, tr.y, Xva)
        r = evaluate(va.y, P)
        per_fold.append((fold.name, len(va), r))
        coefs[fold.name] = (model, pre)
        P_all.append(P)
        y_all.append(va.y.to_numpy())
        cid_all.append(va.X["competition_id"].to_numpy())
        fold_all.append(np.repeat(fold.name, len(va)))
        print(f"  {fold.name}: N={len(va)}  log_loss={r.log_loss:.16f}  brier={r.brier:.16f}")
    P = np.vstack(P_all); Y = np.concatenate(y_all)
    CID = np.concatenate(cid_all); FOLD = np.concatenate(fold_all)
    pooled_ll = float(np.mean([r.log_loss for _, _, r in per_fold]))
    pooled_br = float(np.mean([r.brier for _, _, r in per_fold]))
    print(f"\n  pooled mean log loss : {pooled_ll:.16f}  (reference {POOLED_REF_LOG_LOSS:.16f})")
    print(f"  pooled mean Brier    : {pooled_br:.16f}  (reference {POOLED_REF_BRIER:.16f})")
    check("pooled log loss reproduces the documented V1 value",
          abs(pooled_ll - POOLED_REF_LOG_LOSS) < 1e-9,
          f"diff={pooled_ll - POOLED_REF_LOG_LOSS:.3e}")

    iH, iD, iA = 0, 1, 2
    pd_ = P[:, iD]
    pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]

    def subsets():
        yield "POOLED", np.ones(len(Y), bool)
        for cid, name in LEAGUES.items():
            yield name, CID == cid

    rule("STEP 3 -- P(Draw) DISTRIBUTION")
    print("| Scope | N | mean | min | p05 | p10 | p25 | p50 | p75 | p90 | p95 | max |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in subsets():
        a = pd_[m]
        print(f"| {name} | {m.sum()} | {a.mean():.4f} | {a.min():.4f} | " +
              " | ".join(f"{q(a,p):.4f}" for p in (5, 10, 25, 50, 75, 90, 95)) +
              f" | {a.max():.4f} |")
    print("\n  by ACTUAL class (pooled):")
    print("| Actual | N | mean | p25 | p50 | p75 | p90 | max |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in CLASS_ORDER:
        a = pd_[Y == c]
        print(f"| {c} | {len(a)} | {a.mean():.4f} | {q(a,25):.4f} | {q(a,50):.4f} | "
              f"{q(a,75):.4f} | {q(a,90):.4f} | {a.max():.4f} |")

    rule("STEP 4 -- P(Draw) BY ACTUAL OUTCOME + SEPARATION")
    print("| Scope | mean P(D\\|D) | mean P(D\\|H) | mean P(D\\|A) | sep vs H | sep vs A |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, m in subsets():
        d = pd_[m & (Y == "D")].mean()
        h = pd_[m & (Y == "H")].mean()
        a = pd_[m & (Y == "A")].mean()
        print(f"| {name} | {d:.4f} | {h:.4f} | {a:.4f} | {d-h:+.4f} | {d-a:+.4f} |")
    print("\n  Positive separation => the model DOES rank real draws higher, even if Draw")
    print("  rarely wins the argmax. Near-zero separation => a representation problem.")

    rule("STEP 5 -- HOW OFTEN IS DRAW THE ARGMAX?")
    print("| Scope | N | actual D | pred D | actual D % | pred D % | ratio | recall | precision |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in subsets():
        ad = int((Y[m] == "D").sum()); pdct = int((pred[m] == "D").sum()); n = int(m.sum())
        rec = (pred[m & (Y == "D")] == "D").mean() if ad else float("nan")
        prec = (Y[m & (pred == "D")] == "D").mean() if pdct else float("nan")
        ratio = (ad / pdct) if pdct else float("inf")
        print(f"| {name} | {n} | {ad} | {pdct} | {100*ad/n:.1f}% | {100*pdct/n:.1f}% | "
              f"{ratio:.1f}x | {rec:.4f} | {prec:.4f} |")

    rule("STEP 6/7 -- P(Draw) BINS, CALIBRATION AND RELIABILITY")
    for name, m in subsets():
        print(f"\n### {name}")
        print("| Bin | N | actual draws | actual rate | mean P(D) | gap (actual-pred) |")
        print("|---|---:|---:|---:|---:|---:|")
        gaps = []
        for (lo, hi), lab in zip(BINS, BIN_LABEL):
            b = m & (pd_ >= lo) & (pd_ < hi)
            n = int(b.sum())
            if not n:
                print(f"| {lab} | 0 | - | - | - | - |")
                continue
            ar = float((Y[b] == "D").mean()); mp = float(pd_[b].mean())
            gaps.append(abs(ar - mp))
            print(f"| {lab} | {n} | {int((Y[b]=='D').sum())} | {ar:.4f} | {mp:.4f} | {ar-mp:+.4f} |")
        yD = (Y[m] == "D").astype(float)
        brier_d = float(np.mean((pd_[m] - yD) ** 2))
        eps = 1e-15
        p = np.clip(pd_[m], eps, 1 - eps)
        ll_d = float(-np.mean(yD * np.log(p) + (1 - yD) * np.log(1 - p)))
        print(f"\n  Draw one-vs-rest Brier: {brier_d:.6f}   log loss: {ll_d:.6f}   "
              f"mean |calibration gap| over populated bins: {np.mean(gaps):.4f}")

    rule("STEP 8 -- PROBABILITY MARGIN / DECISION-RULE SENSITIVITY (diagnostic only)")
    srt = np.sort(P, axis=1)
    margin = srt[:, 2] - srt[:, 1]
    isD = Y == "D"
    print(f"  actual draws: {int(isD.sum())}")
    print(f"  P(D) > P(H)                     : {int((pd_[isD] > P[isD,iH]).sum())}")
    print(f"  P(D) > P(A)                     : {int((pd_[isD] > P[isD,iA]).sum())}")
    rank = (P[isD] > pd_[isD][:, None]).sum(axis=1) + 1
    for r in (1, 2, 3):
        print(f"  Draw ranked {r}{'st' if r==1 else 'nd' if r==2 else 'rd'} among the three"
              f"    : {int((rank==r).sum())} ({100*(rank==r).mean():.1f}%)")
    print(f"  mean top-second margin, actual draws : {margin[isD].mean():.4f}")
    print(f"  mean top-second margin, all fixtures : {margin.mean():.4f}")
    print("\n  Draws recovered under alternative decision rules (NOT deployed, NOT tuned):")
    other_max = np.maximum(P[:, iH], P[:, iA])
    for lab, mask in (
            ("argmax (current)", pred == "D"),
            ("P(D) >= P(H)", pd_ >= P[:, iH]),
            ("P(D) >= P(A)", pd_ >= P[:, iA]),
            ("P(D) >= max(H,A) - 0.01", pd_ >= other_max - .01),
            ("P(D) >= max(H,A) - 0.02", pd_ >= other_max - .02),
            ("P(D) >= max(H,A) - 0.05", pd_ >= other_max - .05)):
        rec = int((mask & isD).sum()); tot = int(mask.sum())
        print(f"    {lab:28s} draws caught {rec:5d}/{int(isD.sum())} "
              f"({100*rec/isD.sum():5.1f}%)  fixtures flagged {tot:5d}  "
              f"precision {100*rec/tot if tot else float('nan'):5.1f}%")
    print("\n  These are sensitivity diagnostics. No threshold is adopted, tuned or deployed.")

    rule("STEP 9 -- FITTED COEFFICIENT DIAGNOSTIC (in memory only)")
    for fname, (model, pre) in coefs.items():
        names = list(pre.numeric_columns) + [f"competition_id={c}" for c in pre.competition_categories]
        classes = list(model.classes_)
        di = classes.index("D")
        C = model.coef_
        print(f"\n### {fname}  coef shape={C.shape}  classes={classes}  "
              f"features={len(names)}")
        print("  intercepts: " + "  ".join(f"{c}={model.intercept_[i]:+.4f}"
                                           for i, c in enumerate(classes)))
        print("  |coef| mean by class: " + "  ".join(
            f"{c}={np.abs(C[i]).mean():.4f}" for i, c in enumerate(classes)))
        print("  |coef| max  by class: " + "  ".join(
            f"{c}={np.abs(C[i]).max():.4f}" for i, c in enumerate(classes)))
        row = C[di]
        order = np.argsort(row)
        print("  top 10 features pushing TOWARD Draw:")
        for j in order[::-1][:10]:
            print(f"    {names[j][:52]:52s} {row[j]:+.5f}")
        print("  top 10 features pushing AWAY from Draw:")
        for j in order[:10]:
            print(f"    {names[j][:52]:52s} {row[j]:+.5f}")
        print("  league one-hot coefficients for Draw:")
        for k, c in enumerate(pre.competition_categories):
            j = len(pre.numeric_columns) + k
            nm = LEAGUES.get(int(c), str(c))
            print(f"    {nm:18s} D={C[di][j]:+.5f}  " + "  ".join(
                f"{cl}={C[i][j]:+.5f}" for i, cl in enumerate(classes) if cl != "D"))

    rule("STEP 10 -- LEAGUE-SPECIFIC DRAW SUMMARY")
    print("| League | actual D% | pred D% | recall | precision | mean P(D) | P(D\\|D) | "
          "P(D\\|H) | P(D\\|A) | Draw Brier | mean gap |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in subsets():
        n = int(m.sum()); ad = int((Y[m] == "D").sum()); pdct = int((pred[m] == "D").sum())
        rec = (pred[m & (Y == "D")] == "D").mean() if ad else float("nan")
        prec = (Y[m & (pred == "D")] == "D").mean() if pdct else float("nan")
        yD = (Y[m] == "D").astype(float)
        bd = float(np.mean((pd_[m] - yD) ** 2))
        gaps = []
        for lo, hi in BINS:
            b = m & (pd_ >= lo) & (pd_ < hi)
            if b.sum():
                gaps.append(abs(float((Y[b] == "D").mean()) - float(pd_[b].mean())))
        print(f"| {name} | {100*ad/n:.1f}% | {100*pdct/n:.1f}% | {rec:.4f} | {prec:.4f} | "
              f"{pd_[m].mean():.4f} | {pd_[m&(Y=='D')].mean():.4f} | {pd_[m&(Y=='H')].mean():.4f} | "
              f"{pd_[m&(Y=='A')].mean():.4f} | {bd:.4f} | {np.mean(gaps):.4f} |")

    rule("STEP 11 -- HOME/AWAY PROBABILITY COMPETITION ON ACTUAL DRAWS")
    print("| Scope | mean P(H) on D | mean P(D) on D | mean P(A) on D | mean P(H) on H | "
          "mean P(A) on A |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, m in subsets():
        print(f"| {name} | {P[m&(Y=='D'),iH].mean():.4f} | {P[m&(Y=='D'),iD].mean():.4f} | "
              f"{P[m&(Y=='D'),iA].mean():.4f} | {P[m&(Y=='H'),iH].mean():.4f} | "
              f"{P[m&(Y=='A'),iA].mean():.4f} |")

    rule("STEP 12 -- CLASS PRIORS vs MEAN PREDICTED PROBABILITY")
    print("| Scope | train P(H) | train P(D) | train P(A) | val P(H) | val P(D) | val P(A) | "
          "mean pH | mean pD | mean pA | dH | dD | dA |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in subsets():
        if name == "POOLED":
            tr_y = pd.concat([tr.y for _, tr, _ in folds])
        else:
            cid = [k for k, v in LEAGUES.items() if v == name][0]
            tr_y = pd.concat([tr.y[(tr.X["competition_id"] == cid).values] for _, tr, _ in folds])
        tp = {c: float((tr_y == c).mean()) for c in CLASS_ORDER}
        vp = {c: float((Y[m] == c).mean()) for c in CLASS_ORDER}
        mp = {c: float(P[m, i].mean()) for i, c in enumerate(CLASS_ORDER)}
        print(f"| {name} | {tp['H']:.4f} | {tp['D']:.4f} | {tp['A']:.4f} | "
              f"{vp['H']:.4f} | {vp['D']:.4f} | {vp['A']:.4f} | "
              f"{mp['H']:.4f} | {mp['D']:.4f} | {mp['A']:.4f} | "
              f"{mp['H']-vp['H']:+.4f} | {mp['D']-vp['D']:+.4f} | {mp['A']-vp['A']:+.4f} |")
    print("\n  d* = mean predicted probability - actual validation frequency.")
    print("  A well-behaved probabilistic model keeps these near zero in aggregate.")

    rule("STEP 13 -- DRAW INFORMATION CONTENT (ranking quality of P(Draw))")
    print("| Scope | N | draws | ROC-AUC | PR-AUC | base rate | PR lift over base |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, m in subsets():
        yD = (Y[m] == "D").astype(int)
        if yD.sum() == 0 or yD.sum() == len(yD):
            print(f"| {name} | {m.sum()} | {yD.sum()} | n/a | n/a | n/a | n/a |")
            continue
        auc = roc_auc_score(yD, pd_[m])
        ap = average_precision_score(yD, pd_[m])
        br = yD.mean()
        print(f"| {name} | {int(m.sum())} | {int(yD.sum())} | {auc:.4f} | {ap:.4f} | "
              f"{br:.4f} | {ap/br:.3f}x |")
    print("\n  AUC meaningfully > 0.5 with near-zero Draw recall => the model holds Draw")
    print("  information that argmax is not exploiting. AUC near 0.5 => the existing 80")
    print("  features do not separate draws.")

    rule("STEP 14 -- CROSS-FOLD STABILITY (three folds: ranges only, no SD)")
    print("| Fold | N | Draw recall | pred D % | mean P(D) | mean P(D\\|D) | Draw Brier | Draw AUC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    vals = {}
    for fname, n, _ in per_fold:
        m = FOLD == fname
        yD = (Y[m] == "D").astype(int)
        rec = (pred[m & (Y == "D")] == "D").mean()
        auc = roc_auc_score(yD, pd_[m])
        bd = float(np.mean((pd_[m] - yD) ** 2))
        vals[fname] = (rec, float((pred[m] == "D").mean()), float(pd_[m].mean()),
                       float(pd_[m & (Y == "D")].mean()), bd, auc)
        print(f"| {fname} | {int(m.sum())} | {rec:.4f} | {100*vals[fname][1]:.1f}% | "
              f"{vals[fname][2]:.4f} | {vals[fname][3]:.4f} | {bd:.4f} | {auc:.4f} |")
    labels = ["Draw recall", "pred D %", "mean P(D)", "mean P(D|D)", "Draw Brier", "Draw AUC"]
    print("\n  ranges across the three folds:")
    for i, lab in enumerate(labels):
        xs = [vals[f][i] for f in vals]
        print(f"    {lab:14s} min={min(xs):.4f}  max={max(xs):.4f}  range={max(xs)-min(xs):.4f}")
    print("\n  THREE folds cannot support a standard deviation. Ranges only; a single-fold")
    print("  outlier is not a trend.")

    rule("STEP 15 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified (databases, league partitions, artifact, pins)", not changed,
          str(changed))
    print("  no estimator persisted; no prediction dataset saved; no 2025/26 accessed")

    rule("D-33 COMPLETE -- DIAGNOSIS ONLY, NO FIX APPLIED")
    print("Descriptive diagnostics on validation folds. No threshold, weight, calibration")
    print("or feature change was made, and none is authorized by this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
