"""D-32 -- per-league diagnostic decomposition of pooled V1 behaviour.

    python run_d32_per_league_diagnostic.py

DIAGNOSTIC ONLY. NOT PROMOTION EVIDENCE. NOT A MODEL CHANGE.

These are descriptive walk-forward diagnostics. They do not establish
that any league is production-reliable, and they do not authorize a model
change.

WHAT THIS DOES
Refits the EXISTING V1 production path -- `train.train_logistic_regression`,
so LogisticRegression(C=1.0, max_iter=2000, random_state=0) with the
existing median-imputer / StandardScaler / competition_id one-hot -- on
the existing three walk-forward folds, then decomposes the validation
predictions by competition_id. The pooled model is unchanged; only the
reporting is split by league. Nothing is tuned, selected or calibrated.

WHAT IT DOES NOT DO
No 2025/26 anywhere: a hard firewall aborts the run if any final-test
season_id reaches a fold. No new folds, no shuffling, no sampling, no
class weighting, no threshold tuning, no POSSESSION, no feature change,
no calibration. Writes nothing, saves nothing, persists no estimator or
prediction; all scratch lives outside the repository and is removed.

READ THE SAMPLE SIZES. Per-league per-fold N is 306-381. Three folds is
too few for a meaningful standard deviation, and this script reports
ranges rather than pretending otherwise.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LEAGUES = {
    423: "Premier League",
    477: "Bundesliga",
    419: "La Liga",
    200: "Ligue 1",
    499: "Serie A",
}

#: Documented pooled V1 reference (Phase 4A/4B/5A). Comparison only.
POOLED_REF_LOG_LOSS = 0.9993791056968738
POOLED_REF_BRIER = 0.5965016957578898

EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}


def rule(t): print("\n" + "=" * 108); print(t); print("=" * 108)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 108)
    print("HARD FAIL -- D-32 halted. No diagnostic is produced.")
    print(msg)
    print("!" * 108)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        stop(f"stop condition: {label}")


def snapshot():
    """Checksums of everything that must not change."""
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    lg = REPO / "data/processed/leagues"
    if lg.exists():
        for p in sorted(lg.glob("*.db")):
            snap[f"leagues/{p.name}"] = md5(p)
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["v1_logreg.pkl"] = md5(art)
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    for rel in pins:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def main():
    rule("STEP 1 -- ENVIRONMENT AND SOURCE INTEGRITY")
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
    check("13/13 LOCKED_INPUTS at expected values",
          all(before[f"pin:{r}"] == v["expected"] for r, v in
              json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                         .read_text())["locked_input_checksums"].items()))
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")
    check("contract is exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("CLASS_ORDER is H/D/A", list(CLASS_ORDER) == ["H", "D", "A"])

    rule("STEP 2 -- LOAD + FINAL-TEST FIREWALL (exact season_id membership)")
    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    print(f"  forbidden 2025/26 season_ids: {sorted(test_ids)}")
    folds = list(iter_walk_forward_folds(ds))
    check("exactly three walk-forward folds", len(folds) == 3)
    for fold, tr, va in folds:
        seen = set(tr.metadata["season_id"]) | set(va.metadata["season_id"])
        bad = seen & test_ids
        check(f"{fold.name}: no final-test season_id", not bad, str(sorted(bad)))
        print(f"    {fold.name}: train={len(tr)} valid={len(va)}")

    rule("STEP 3 -- COMPETITION CATEGORY COVERAGE PER TRAINING FOLD")
    print("| Fold | " + " | ".join(LEAGUES.values()) + " | all 5 one-hot categories |")
    print("|---|" + "---:|" * len(LEAGUES) + "---|")
    for fold, tr, va in folds:
        counts = [int((tr.X["competition_id"] == c).sum()) for c in LEAGUES]
        cats = set(tr.X["competition_id"].dropna().unique().tolist())
        check(f"{fold.name}: all five leagues present in training", cats == set(LEAGUES))
        print(f"| {fold.name} | " + " | ".join(str(c) for c in counts) +
              f" | {len(cats)} |")

    rule("STEP 4 -- DIAGNOSTIC REFIT (existing V1 path, unchanged configuration)")
    print("  LogisticRegression(C=1.0, max_iter=2000, random_state=0) via "
          "train.train_logistic_regression")
    print("  preprocessing fitted on the training fold only; no tuning, no calibration\n")

    rows = []          # per league x fold
    pooled_rows = []   # per fold, pooled
    store = {}         # (league, fold) -> (y, P)
    for fold, tr, va in folds:
        Xtr = tr.X[list(MODEL_B_COLUMNS)]
        Xva = va.X[list(MODEL_B_COLUMNS)]
        model, pre, P = train_logistic_regression(Xtr, tr.y, Xva)
        r = evaluate(va.y, P)
        pooled_rows.append((fold.name, len(va), r))
        print(f"  {fold.name}: pooled N={len(va)} log_loss={r.log_loss:.16f} "
              f"brier={r.brier:.16f}")
        comp = va.X["competition_id"].to_numpy()
        for cid, name in LEAGUES.items():
            m = comp == cid
            if not m.any():
                stop(f"{fold.name}: no validation rows for {name}")
            y_l = va.y[m]
            P_l = P[m]
            rl = evaluate(y_l, P_l)
            rows.append({"league": name, "cid": cid, "fold": fold.name,
                         "n": int(m.sum()), "r": rl})
            store[(name, fold.name)] = (y_l.to_numpy(), P_l)

    def recall(res, cls):
        i = CLASS_ORDER.index(cls)
        cm = res.confusion_matrix
        s = cm[i, :].sum()
        return cm[i, i] / s if s else float("nan")

    rule("STEP 5 -- PER-LEAGUE x FOLD METRICS")
    print("| League | Fold | N | Log Loss | Brier | Accuracy | Macro-F1 | Balanced Acc "
          "| Home Recall | Draw Recall | Away Recall |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in rows:
        r = d["r"]
        print(f"| {d['league']} | {d['fold']} | {d['n']} | {r.log_loss:.6f} | {r.brier:.6f} "
              f"| {r.accuracy:.6f} | {r.macro_f1:.6f} | {r.balanced_accuracy:.6f} "
              f"| {recall(r,'H'):.4f} | {recall(r,'D'):.4f} | {recall(r,'A'):.4f} |")

    rule("STEP 6 -- LEAGUE MEANS ACROSS THE THREE FOLDS (unweighted mean of folds)")
    print("| League | Total N | Mean LL | Mean Brier | Mean Accuracy | Mean Macro-F1 "
          "| Mean Balanced Acc | Home Recall | Draw Recall | Away Recall | vs pooled LL |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    league_mean = {}
    for cid, name in LEAGUES.items():
        rs = [d for d in rows if d["league"] == name]
        mean = lambda f: float(np.mean([f(d["r"]) for d in rs]))  # noqa: E731
        ll = mean(lambda r: r.log_loss)
        league_mean[name] = ll
        print(f"| {name} | {sum(d['n'] for d in rs)} | {ll:.6f} | "
              f"{mean(lambda r: r.brier):.6f} | {mean(lambda r: r.accuracy):.6f} | "
              f"{mean(lambda r: r.macro_f1):.6f} | {mean(lambda r: r.balanced_accuracy):.6f} | "
              f"{mean(lambda r: recall(r,'H')):.4f} | {mean(lambda r: recall(r,'D')):.4f} | "
              f"{mean(lambda r: recall(r,'A')):.4f} | "
              f"{'better' if ll < POOLED_REF_LOG_LOSS else 'worse'} |")
    pooled_ll = float(np.mean([r.log_loss for _, _, r in pooled_rows]))
    pooled_br = float(np.mean([r.brier for _, _, r in pooled_rows]))
    print(f"| **POOLED (this run)** | {sum(n for _, n, _ in pooled_rows)} | {pooled_ll:.16f} "
          f"| {pooled_br:.16f} | | | | | | | reference |")
    print(f"| **POOLED (documented)** | | {POOLED_REF_LOG_LOSS:.16f} | {POOLED_REF_BRIER:.16f} "
          f"| | | | | | | Phase 4A/4B/5A |")
    check("this run reproduces the documented pooled V1 log loss",
          abs(pooled_ll - POOLED_REF_LOG_LOSS) < 1e-9, f"diff={pooled_ll - POOLED_REF_LOG_LOSS:.3e}")
    print("\n  'better'/'worse' is DESCRIPTIVE ONLY: league composition, base rates and")
    print("  sample sizes differ, so a lower raw metric does not mean a league is more")
    print("  reliable in production.")

    rule("STEP 7 -- LEAGUE CONTRIBUTION TO POOLED ERROR")
    print("| League | Rows | Row share | Sum log loss | LL share | Sum Brier | Brier share "
          "| Mean LL / fixture | Mean Brier / fixture |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot_n = sum(d["n"] for d in rows)
    tot_ll = sum(d["r"].log_loss * d["n"] for d in rows)
    tot_br = sum(d["r"].brier * d["n"] for d in rows)
    for cid, name in LEAGUES.items():
        rs = [d for d in rows if d["league"] == name]
        n = sum(d["n"] for d in rs)
        sll = sum(d["r"].log_loss * d["n"] for d in rs)
        sbr = sum(d["r"].brier * d["n"] for d in rs)
        print(f"| {name} | {n} | {100*n/tot_n:.1f}% | {sll:.2f} | {100*sll/tot_ll:.1f}% "
              f"| {sbr:.2f} | {100*sbr/tot_br:.1f}% | {sll/n:.6f} | {sbr/n:.6f} |")
    print("\n  Contribution share tracks sample size closely; do not read a large share")
    print("  as intrinsic difficulty without comparing mean loss per fixture.")

    rule("STEP 8 -- ACTUAL vs PREDICTED CLASS DISTRIBUTION (the Draw diagnostic)")
    print("| League | N | Actual H | Actual D | Actual A | Pred H | Pred D | Pred A "
          "| Draw gap (pred-actual) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cid, name in LEAGUES.items():
        ys = np.concatenate([store[(name, f.name)][0] for f, _, _ in folds])
        Ps = np.vstack([store[(name, f.name)][1] for f, _, _ in folds])
        pred = np.array(CLASS_ORDER)[Ps.argmax(axis=1)]
        n = len(ys)
        a = {c: float((ys == c).mean()) for c in CLASS_ORDER}
        p = {c: float((pred == c).mean()) for c in CLASS_ORDER}
        print(f"| {name} | {n} | {100*a['H']:.1f}% | {100*a['D']:.1f}% | {100*a['A']:.1f}% "
              f"| {100*p['H']:.1f}% | {100*p['D']:.1f}% | {100*p['A']:.1f}% "
              f"| {100*(p['D']-a['D']):+.1f}pp |")

    rule("STEP 9 -- CONFIDENCE PROFILE (max predicted probability)")
    print("| League | Mean Conf | Median | p25 | p75 | >=50% | >=60% | >=70% |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cid, name in LEAGUES.items():
        Ps = np.vstack([store[(name, f.name)][1] for f, _, _ in folds])
        conf = Ps.max(axis=1)
        print(f"| {name} | {conf.mean():.4f} | {np.median(conf):.4f} | "
              f"{np.percentile(conf,25):.4f} | {np.percentile(conf,75):.4f} | "
              f"{100*(conf>=.5).mean():.1f}% | {100*(conf>=.6).mean():.1f}% | "
              f"{100*(conf>=.7).mean():.1f}% |")

    rule("STEP 10 -- HIGH-CONFIDENCE FAILURES")
    print("| League | >=50 total | >=50 wrong | err rate | >=60 total | >=60 wrong | err rate "
          "| >=70 total | >=70 wrong | err rate | overall err |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cid, name in LEAGUES.items():
        ys = np.concatenate([store[(name, f.name)][0] for f, _, _ in folds])
        Ps = np.vstack([store[(name, f.name)][1] for f, _, _ in folds])
        pred = np.array(CLASS_ORDER)[Ps.argmax(axis=1)]
        conf = Ps.max(axis=1)
        wrong = pred != ys
        cells = []
        for th in (.5, .6, .7):
            m = conf >= th
            t = int(m.sum()); w = int(wrong[m].sum())
            cells += [str(t), str(w), (f"{100*w/t:.1f}%" if t else "n/a")]
        print(f"| {name} | " + " | ".join(cells) + f" | {100*wrong.mean():.1f}% |")
    print("\n  Compare each band's error rate with the league's overall error rate:")
    print("  a band that is NOT lower means confidence is not buying reliability there.")

    rule("STEP 11 -- CONFUSION MATRICES (rows = actual, cols = predicted, H/D/A)")
    for cid, name in LEAGUES.items():
        ys = np.concatenate([store[(name, f.name)][0] for f, _, _ in folds])
        Ps = np.vstack([store[(name, f.name)][1] for f, _, _ in folds])
        pred = np.array(CLASS_ORDER)[Ps.argmax(axis=1)]
        print(f"\n### {name}")
        print("| actual \\ pred | H | D | A | row total | H % | D % | A % |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for a in CLASS_ORDER:
            r = [int(((ys == a) & (pred == p)).sum()) for p in CLASS_ORDER]
            tot = sum(r)
            pct = [f"{100*x/tot:.1f}%" if tot else "n/a" for x in r]
            print(f"| {a} | {r[0]} | {r[1]} | {r[2]} | {tot} | {pct[0]} | {pct[1]} | {pct[2]} |")

    rule("STEP 12 -- FOLD STABILITY (three folds only -- range, not inference)")
    print("| League | LL fold_1 | LL fold_2 | LL fold_3 | best | worst | range "
          "| Brier range |")
    print("|---|---:|---:|---:|---|---|---:|---:|")
    for cid, name in LEAGUES.items():
        rs = {d["fold"]: d["r"] for d in rows if d["league"] == name}
        lls = {k: v.log_loss for k, v in rs.items()}
        brs = [v.brier for v in rs.values()]
        best = min(lls, key=lls.get); worst = max(lls, key=lls.get)
        print(f"| {name} | {lls['fold_1']:.6f} | {lls['fold_2']:.6f} | {lls['fold_3']:.6f} "
              f"| {best} | {worst} | {max(lls.values())-min(lls.values()):.6f} "
              f"| {max(brs)-min(brs):.6f} |")
    print("\n  THREE folds is far too few for a standard deviation to mean anything.")
    print("  Ranges are reported instead, and a one-fold outlier is not a trend.")

    rule("STEP 13 -- FEATURE MISSINGNESS BY LEAGUE (validation partitions, pre-imputation)")
    print("| League | Rows | Cells | NULL | NULL % | 0 NULLs | 1-5 | >5 | Worst feature |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for cid, name in LEAGUES.items():
        X = pd.concat([va.X[(va.X["competition_id"] == cid).values][list(MODEL_B_COLUMNS)]
                       for _, _, va in folds])
        nulls = X.isna(); per = nulls.sum(axis=1)
        nn = int(nulls.values.sum()); cells = len(X) * 80
        nr = (nulls.mean() * 100).sort_values(ascending=False)
        print(f"| {name} | {len(X)} | {cells} | {nn} | {100*nn/cells:.2f}% | "
              f"{int((per==0).sum())} | {int(((per>=1)&(per<=5)).sum())} | "
              f"{int((per>5).sum())} | {nr.index[0]} ({nr.iloc[0]:.2f}%) |")

    rule("STEP 14 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified (databases, league partitions, artifact, pins)", not changed,
          str(changed))
    check("MODEL_VERSION still v1.0", MODEL_VERSION == "v1.0")
    print("  no estimator persisted; no prediction saved; no repository file written")

    rule("D-32 COMPLETE -- DESCRIPTIVE DIAGNOSTIC ONLY")
    print("These are descriptive walk-forward diagnostics. They do not establish that")
    print("any league is production-reliable, and they do not authorize a model change.")
    return 0


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="d32_")
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
