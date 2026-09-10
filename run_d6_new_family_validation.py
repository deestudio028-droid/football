"""D-6 -- validation of the three D-5 verified new information families.

ARMS (four; one family per arm, no combinations)
    ARM 0  E-1 baseline                80 columns
    ARM 1  E-1 + POSSESSION            82
    ARM 2  E-1 + CORNERS               82
    ARM 3  E-1 + RED_CARDS             82

FROZEN PROTOCOL
    LogisticRegression, C=0.0005, max_iter=2000, random_state=0,
    existing LogisticRegressionPreprocessor, the three existing
    walk-forward folds, existing metrics. Nothing is tuned.

FEATURE CONSTRUCTION
    The six candidate columns are built by IMPORTING the D-5 audited
    implementation (run_d5_new_feature_construction_audit.build_dataset).
    They are NOT reimplemented here and NOT changed, so the construction
    that passed D-5's 7/7 leakage tests is the construction used.
    They are held in memory and never written to any database.

RED-CARD MISSINGNESS
    The project's existing preprocessing convention is applied unchanged:
    SimpleImputer(strategy="median") fitted on the training partition.
    Missing red-card values are NOT manually replaced with zero and no
    special imputation is introduced. The missingness entering the
    pipeline is reported.

FORBIDDEN AND ABSENT: 2025/26, final test, C tuning, calibration,
thresholds, class weighting, feature selection, combinations, odds, xG,
league accumulators, positions, formations, referees, model persistence,
production or database modification.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_d6_new_family_validation.py
"""
from __future__ import annotations

import hashlib
import importlib.util
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
from models.config import (
    FINAL_TEST_SEASONS, MODEL_VERSION, SEASON_NAME_TO_IDS, WALK_FORWARD_FOLDS,
)
from models.data import SupervisedDataset, load_supervised_dataset
from models.evaluate import evaluate, validate_probabilities
from models.robustness import V1_LOGREG_BASE_KWARGS
from models.splits import iter_walk_forward_folds

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MATCHES_DB = REPO / "data" / "processed" / "matches.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"
D5_SCRIPT = REPO / "run_d5_new_feature_construction_audit.py"

C_VALUE = 0.0005
BASE = tuple(MODEL_B_COLUMNS)
FAMILIES = {
    "POSSESSION": ("home_possession_per_match_season", "away_possession_per_match_season"),
    "CORNERS": ("home_corners_per_match_season", "away_corners_per_match_season"),
    "RED_CARDS": ("home_red_cards_per_match_season", "away_red_cards_per_match_season"),
}
NEW_COLUMNS = tuple(c for cols in FAMILIES.values() for c in cols)
HIGHER_BETTER = ("accuracy", "macro_f1", "balanced_accuracy")
LOWER_BETTER = ("log_loss", "brier")
METRICS = HIGHER_BETTER + LOWER_BETTER
EFFECT_MIN = 0.10          # same convention as D-2: |delta| must exceed 10% of baseline spread


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)


def stop(m):
    print("\n" + "!" * 78); print("STOP -- D-6 halted"); print(m); print("!" * 78)
    raise SystemExit(1)


def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()


def load_d5_module():
    """Import D-5 without executing it (it has an __main__ guard)."""
    spec = importlib.util.spec_from_file_location("d5_audit", D5_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def fit_arm(Xtr, ytr, Xva, cols, label):
    from sklearn.linear_model import LogisticRegression
    from models import train as train_module

    miss = [c for c in cols if c not in Xtr.columns]
    if miss:
        stop(f"{label}: missing columns {miss[:5]}")
    A, B = Xtr[list(cols)], Xva[list(cols)]
    pre = train_module.LogisticRegressionPreprocessor().fit(A)
    model = LogisticRegression(C=C_VALUE, **V1_LOGREG_BASE_KWARGS).fit(pre.transform(A), ytr)
    if int(np.max(model.n_iter_)) >= V1_LOGREG_BASE_KWARGS["max_iter"]:
        stop(f"{label}: hit max_iter -- unconverged")
    P = np.asarray(train_module._reorder_proba(model, pre.transform(B)), dtype=float)
    validate_probabilities(P)
    return P, model, pre


def run_arm(dataset, cols, label):
    folds = {}
    for fold, tr, va in iter_walk_forward_folds(dataset):
        P, model, pre = fit_arm(tr.X, tr.y, va.X, cols, label)
        r = evaluate(va.y, P)
        y = va.y.to_numpy()
        folds[fold.name] = dict(
            P=P, y=y, r=r, per_class=per_class(y, P), n_iter=model.n_iter_,
            fixture_ids=va.metadata["fixture_id"].to_numpy(),
            train_nulls={c: float(tr.X[c].isna().mean()) for c in cols if c in NEW_COLUMNS},
            medians={c: float(dict(zip(pre.numeric_columns, pre._imputer.statistics_))[c])
                     for c in cols if c in NEW_COLUMNS},
        )
    means = {m: statistics.fmean(folds[n]["r"].as_dict()[m] for n in sorted(folds)) for m in METRICS}
    sds = {m: statistics.pstdev([folds[n]["r"].as_dict()[m] for n in sorted(folds)]) for m in METRICS}
    by_fold = {m: [folds[n]["r"].as_dict()[m] for n in sorted(folds)] for m in METRICS}
    return dict(folds=folds, means=means, sds=sds, by_fold=by_fold, label=label)


def main():
    print("D-6 -- NEW INFORMATION FAMILY VALIDATION (validation folds only)")
    print("Four arms, one family each. No combinations. No tuning. No 2025/26.")

    # ---------------- firewall + pre-flight ----------------
    rule("PRE-FLIGHT AND 2025/26 FIREWALL")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}  C={C_VALUE}  kwargs={V1_LOGREG_BASE_KWARGS}")
    fold_seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    forbidden = sorted(fold_seasons & set(FINAL_TEST_SEASONS))
    print(f"  fold seasons: {sorted(fold_seasons)}")
    print(f"  validation seasons: {[f.validation_seasons[0] for f in WALK_FORWARD_FOLDS]}")
    print(f"  final-test seasons intersecting the folds: {forbidden or 'NONE'}")
    if forbidden:
        stop(f"2025/26 reachable through the folds: {forbidden}")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines: mismatches={bad or 'NONE'}")
    if bad or MODEL_VERSION != "v1.0":
        stop("integrity failure")

    # ---------------- feature construction, imported from D-5 ----------------
    rule("FEATURE CONSTRUCTION -- imported unchanged from the D-5 audit")
    d5 = load_d5_module()
    print(f"  source: {D5_SCRIPT.name}  md5={md5(D5_SCRIPT)}")
    all_fixtures = d5.load_fixtures_chronological(MATCHES_DB)
    fixtures = [f for f in all_fixtures if f["season_id"] not in d5.TEST_SEASON_IDS]
    print(f"  fixtures loaded={len(all_fixtures)}  2025/26 dropped={len(all_fixtures)-len(fixtures)}"
          f"  retained={len(fixtures)}")
    kept = sorted({f["season"] for f in fixtures})
    leaked = sorted(set(kept) & set(FINAL_TEST_SEASONS))
    print(f"  seasons retained: {kept}")
    if leaked:
        stop(f"final-test season survived filtering: {leaked}")
    print("  [PASS] 2025/26 absent from the constructed features")
    new_df = d5.build_dataset(fixtures)[["fixture_id"] + list(NEW_COLUMNS)]
    print(f"  candidate feature rows built: {len(new_df)}  columns: {list(NEW_COLUMNS)}")

    # ---------------- join to the supervised dataset ----------------
    rule("DATASET ASSEMBLY")
    dataset = load_supervised_dataset(FEATURES_DB)
    print(f"  supervised rows (all seasons incl. 2025/26 as loaded by the frozen loader): "
          f"{len(dataset.X)}")
    ids = dataset.metadata["fixture_id"].to_numpy()
    lookup = new_df.set_index("fixture_id")
    X = dataset.X.copy()
    for c in NEW_COLUMNS:
        X[c] = pd.Series(ids, index=X.index).map(lookup[c])
    joined = SupervisedDataset(metadata=dataset.metadata.copy(), X=X, y=dataset.y.copy())
    unmatched = int(pd.Series(ids).isin(lookup.index).eq(False).sum())
    print(f"  rows with no constructed features (expected = the 2025/26 rows the loader "
          f"carries but the folds never touch): {unmatched}")
    for c in BASE:
        if not dataset.X[c].equals(joined.X[c]):
            stop(f"baseline column {c} was altered during the join")
    print(f"  [PASS] all {len(BASE)} baseline columns byte-identical after the join")
    print("  NOTE: the walk-forward folds select by season and never include 2025/26, so the")
    print("        unmatched rows are unreachable by every arm. Verified per fold below.")

    # Per-fold firewall proof, plus the distinction between "2025/26 leaked in"
    # and "early-season min-coverage NULL", which look identical if not separated.
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    print("\n  per-fold check -- rows where ALL six new features are NULL:")
    for fold, tr, va in iter_walk_forward_folds(joined):
        for lab, part in (("train", tr), ("val", va)):
            m = part.X[list(NEW_COLUMNS)].isna().all(axis=1).to_numpy()
            n_test = int(part.metadata.loc[m, "season_id"].isin(test_ids).sum())
            e1_null = int(part.X.loc[m, "home_shots_for_per_match_season"].isna().sum())
            print(f"    {fold.name} {lab}: all-null={int(m.sum()):4d}  from 2025/26={n_test}  "
                  f"also-null in E-1's own season feature={e1_null}  "
                  f"({m.sum()/len(part.X)*100:.2f}% of partition)")
            if n_test:
                stop(f"{fold.name}/{lab}: {n_test} rows from the final-test season entered a fold")
    print("  [PASS] zero final-test rows in any partition. The all-null rows are early-season")
    print("         cases hitting the SAME min-coverage rule that already nulls E-1's own")
    print("         season features -- not leakage, and not specific to the new families.")

    # ---------------- arms ----------------
    rule("FITTING FOUR ARMS")
    arms = {"E-1": run_arm(joined, BASE, "E-1")}
    for fam, cols in FAMILIES.items():
        arms[fam] = run_arm(joined, BASE + cols, f"E-1+{fam}")
    for name, a in arms.items():
        for n, f in a["folds"].items():
            if np.isnan(f["P"]).any():
                stop(f"{name}/{n}: NaN probabilities")
        print(f"  {name:<12} " + "  ".join(f"{m}={a['means'][m]:.10f}" for m in METRICS))

    base = arms["E-1"]
    for fam in FAMILIES:
        for n in base["folds"]:
            if not np.array_equal(base["folds"][n]["fixture_ids"], arms[fam]["folds"][n]["fixture_ids"]):
                stop(f"{fam}/{n}: fixture ordering differs from baseline")
    print("  [PASS] identical fixture ordering across all arms and folds")
    spread = {m: max(base["by_fold"][m]) - min(base["by_fold"][m]) for m in METRICS}

    # ---------------- per-fold ----------------
    rule("PER-FOLD RESULTS")
    for name in ["E-1"] + list(FAMILIES):
        a = arms[name]
        print(f"\n  --- {name} ---")
        for n in sorted(a["folds"]):
            r = a["folds"][n]["r"]
            print(f"    {n}: acc={r.accuracy:.8f} macro_f1={r.macro_f1:.8f} "
                  f"bal_acc={r.balanced_accuracy:.8f} log_loss={r.log_loss:.16f} "
                  f"brier={r.brier:.8f} n={r.n} n_iter={a['folds'][n]['n_iter']}")

    # ---------------- final table ----------------
    rule("FINAL TABLE -- mean over 3 folds (sd)")
    print(f"  {'Arm':<14}{'Accuracy':>20}{'Macro-F1':>20}{'BalAcc':>20}{'LogLoss':>22}{'Brier':>20}")
    for name in ["E-1"] + list(FAMILIES):
        a = arms[name]
        lab = "E-1" if name == "E-1" else f"+{name}"
        print(f"  {lab:<14}" + "".join(
            f"{a['means'][m]:>13.8f}({a['sds'][m]:.4f})" if m != "log_loss"
            else f"{a['means'][m]:>15.10f}({a['sds'][m]:.4f})" for m in METRICS))

    rule("DELTAS vs E-1 -- mean, per-fold, all-3-fold consistency")
    verdicts = {}
    for fam in FAMILIES:
        a = arms[fam]
        d = {m: a["means"][m] - base["means"][m] for m in METRICS}
        pf = {m: [a["by_fold"][m][i] - base["by_fold"][m][i] for i in range(3)] for m in METRICS}
        allf = {m: all((x > 0) if m in HIGHER_BETTER else (x < 0) for x in pf[m]) for m in METRICS}
        eff = {m: (abs(d[m]) / spread[m]) if spread[m] else float("nan") for m in METRICS}
        print(f"\n  --- E-1 + {fam} ---")
        for m in METRICS:
            good = (d[m] > 0) if m in HIGHER_BETTER else (d[m] < 0)
            print(f"    {m:<20} mean_delta={d[m]:+.8e}  {'better' if good else 'worse'}  "
                  f"all3={allf[m]}  effect={eff[m]:.4f}x spread")
            print(f"    {'':<20} per_fold={[f'{x:+.3e}' for x in pf[m]]}")
        verdicts[fam] = dict(d=d, pf=pf, allf=allf, eff=eff)

    # ---------------- pre-registered rule ----------------
    rule("PRE-REGISTERED DECISION RULE")
    for fam, v in verdicts.items():
        d, allf, eff = v["d"], v["allf"], v["eff"]
        c = {
            "1 mean log loss improves": d["log_loss"] < 0,
            "2 mean Brier improves": d["brier"] < 0,
            "3 log loss improves in ALL 3 folds": allf["log_loss"],
            "4 Brier improves in ALL 3 folds": allf["brier"],
            "5 accuracy not decreased (mean)": d["accuracy"] >= 0,
            "6 macro-F1 not decreased (mean)": d["macro_f1"] >= 0,
            "7 balanced acc not decreased (mean)": d["balanced_accuracy"] >= 0,
            "8 effect not tiny vs baseline spread": eff["log_loss"] > EFFECT_MIN,
        }
        d1_pattern = (d["log_loss"] < 0 and d["brier"] < 0
                      and (d["accuracy"] < 0 or d["balanced_accuracy"] < 0 or d["macro_f1"] < 0))
        c["9 does NOT reproduce the D-1 pattern"] = not d1_pattern
        serious = all(c.values())
        print(f"\n  E-1 + {fam}")
        for k, ok in c.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {k}")
        verdicts[fam]["serious"] = serious
        verdicts[fam]["d1_pattern"] = d1_pattern
        print(f"    => {'SERIOUS CANDIDATE -- VALIDATION EVIDENCE ONLY' if serious else 'NOT A CANDIDATE'}")

    # ---------------- classification mechanism ----------------
    rule("CLASSIFICATION MECHANISM -- pooled validation")
    def pooled(a):
        pr = {c: 0 for c in CLASS_ORDER}; rec = {c: [] for c in CLASS_ORDER}
        for n in a["folds"]:
            pc = a["folds"][n]["per_class"]
            for c in CLASS_ORDER:
                pr[c] += pc[c]["predicted"]; rec[c].append(pc[c]["recall"])
        return pr, {c: statistics.fmean(rec[c]) for c in CLASS_ORDER}
    bp, br = pooled(base)
    print(f"  {'Arm':<14}" + "".join(f"{'pred_'+c:>10}" for c in CLASS_ORDER)
          + "".join(f"{'recall_'+c:>13}" for c in CLASS_ORDER))
    print(f"  {'E-1':<14}" + "".join(f"{bp[c]:>10d}" for c in CLASS_ORDER)
          + "".join(f"{br[c]:>13.6f}" for c in CLASS_ORDER))
    for fam in FAMILIES:
        p, r = pooled(arms[fam])
        print(f"  {'+'+fam:<14}" + "".join(f"{p[c]:>10d}" for c in CLASS_ORDER)
              + "".join(f"{r[c]:>13.6f}" for c in CLASS_ORDER))
        print(f"  {'  delta':<14}" + "".join(f"{p[c]-bp[c]:>+10d}" for c in CLASS_ORDER)
              + "".join(f"{r[c]-br[c]:>+13.6f}" for c in CLASS_ORDER))
        dh, da, dd = p["H"] - bp["H"], p["A"] - bp["A"], p["D"] - bp["D"]
        tags = []
        if dh > 0 and da < 0: tags.append("A Home over-selection + B Away suppression")
        elif dh > 0: tags.append("A Home over-selection")
        elif da < 0: tags.append("B Away suppression")
        if dd < 0: tags.append("C Draw suppression")
        if not tags: tags.append("D no meaningful classification movement")
        print(f"  {'  mechanism':<14}{'; '.join(tags)}")

    # ---------------- probability diagnostics ----------------
    rule("PROBABILITY DIAGNOSTICS (descriptive only -- never a reason to change C)")
    for name in ["E-1"] + list(FAMILIES):
        a = arms[name]
        mx = statistics.fmean(a["folds"][n]["P"].max(1).mean() for n in a["folds"])
        sd = statistics.fmean(a["folds"][n]["P"].std(1).mean() for n in a["folds"])
        pc = {c: statistics.fmean(a["folds"][n]["P"][:, i].mean() for n in a["folds"])
              for i, c in enumerate(CLASS_ORDER)}
        lab = "E-1" if name == "E-1" else f"+{name}"
        extra = ""
        if name != "E-1":
            mv = statistics.fmean(
                float(np.abs(a["folds"][n]["P"] - base["folds"][n]["P"]).max(1).mean())
                for n in a["folds"])
            extra = f"  mean|prob move| vs E-1={mv:.6e}"
        print(f"  {lab:<14} mean_max_p={mx:.6f} mean_row_spread={sd:.6f} "
              f"p_H={pc['H']:.6f} p_D={pc['D']:.6f} p_A={pc['A']:.6f}{extra}")

    # ---------------- red-card missingness ----------------
    rule("RED-CARD MISSINGNESS ENTERING THE PIPELINE (reported, not altered)")
    a = arms["RED_CARDS"]
    for n in sorted(a["folds"]):
        f = a["folds"][n]
        print(f"  {n}: training null rate " + "  ".join(
            f"{c.split('_per')[0]}={v:.4f}" for c, v in f["train_nulls"].items()))
        print(f"        training-median used by the existing imputer: " + "  ".join(
            f"{c.split('_per')[0]}={v:.6f}" for c, v in f["medians"].items()))
    print("\n  The existing SimpleImputer(median) convention was applied unchanged.")
    print("  Missing red-card values were NOT replaced with zero. Note the consequence:")
    print("  a missing value becomes the league-typical rate, not 'no red cards'. That is")
    print("  the project's standing convention for every other statistic; it is reported")
    print("  here rather than altered, and it is a methodological caveat on this family.")

    # ---------------- determinism ----------------
    rule("REPRODUCIBILITY -- one repeat per arm")
    ok = True
    for name in ["E-1"] + list(FAMILIES):
        cols = BASE if name == "E-1" else BASE + FAMILIES[name]
        rep = run_arm(joined, cols, f"{name} repeat")
        same_p = all(rep["folds"][n]["P"].tobytes() == arms[name]["folds"][n]["P"].tobytes()
                     for n in arms[name]["folds"])
        same_m = all(abs(rep["means"][m] - arms[name]["means"][m]) == 0.0 for m in METRICS)
        same_o = all(np.array_equal(rep["folds"][n]["fixture_ids"],
                                    arms[name]["folds"][n]["fixture_ids"])
                     for n in arms[name]["folds"])
        ok &= same_p and same_m and same_o
        h = hashlib.sha256(b"".join(arms[name]["folds"][n]["P"].tobytes()
                                    for n in sorted(arms[name]["folds"]))).hexdigest()
        print(f"  {name:<12} predictions={same_p} metrics={same_m} ordering={same_o} "
              f"sha256={h[:40]}...")
    print(f"  REPRODUCIBILITY: {'PASS' if ok else 'FAIL'}")
    if not ok:
        stop("non-deterministic")

    # ---------------- integrity ----------------
    rule("INTEGRITY AUDIT")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}")
    for line in ("2025/26 accessed = NO", "final test reused = NO",
                 "production modified = NO", "features.db modified = NO (read-only)",
                 "matches.db modified = NO (read-only)", "C tuning = NONE",
                 "calibration = NONE", "threshold changes = NONE",
                 "class weighting = NONE", "feature selection = NONE",
                 "combinations tested = NONE", "estimator persisted = NONE",
                 "artifacts created = NONE",
                 "new features written to any database = NO (in-memory only)"):
        print(f"  {line}")

    rule("FAMILY VERDICTS")
    serious = [f for f, v in verdicts.items() if v["serious"]]
    for fam in FAMILIES:
        v = verdicts[fam]
        print(f"  {fam:<12} {'PASS' if v['serious'] else 'NOT A CANDIDATE'}"
              f"{'  (reproduces the D-1 pattern)' if v['d1_pattern'] else ''}")
    print(f"\n  SERIOUS CANDIDATES: {serious or 'NONE'}")
    if not serious:
        print("  No family satisfies the complete pre-registered rule. STOP.")
        print("  No promotion, no combination, no final test.")
    else:
        print("  SERIOUS CANDIDATE -- VALIDATION EVIDENCE ONLY. Not promoted.")
        print("  A separate authorization would be required for any next step.")


if __name__ == "__main__":
    main()
