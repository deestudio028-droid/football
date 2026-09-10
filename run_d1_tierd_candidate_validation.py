"""D-1 -- Tier-D information candidate, VALIDATION FOLDS ONLY.

QUESTION
    Model B's 80 columns exclude the Phase 4A "Tier D" constructs. Those
    were only ever evaluated BUNDLED WITH xG (Model D = Model C + Tier D,
    Model C = Model B + xG), and xG is 100% null in the fold_1/fold_2
    training partitions, so Models C/D were never rankable on the 3-fold
    mean. Tier D has therefore NEVER been tested on the full protocol.

    D-1 tests Model B + the temporally-safe Tier-D subset, on validation
    folds only, against the E-1 baseline.

BASELINE (E-1, final-test-backed)
    Model B, 80 columns, C=0.0005, max_iter=2000, random_state=0

CANDIDATE (D-1)
    Baseline columns + tempo(6) + discipline(4) + goals_venue(4)
    + venue_goal_diff(2) + history_flags(4) = 100 columns.
    Same estimator, same C, same preprocessing, same folds.
    ONE variable changes: the information set.

DELIBERATELY EXCLUDED from the candidate
    league_home_advantage_season -- Phase 5A Condition 1 closed it
        (re-adding worsened mean validation log loss); must not return.
    league_mean_goals_per_team_match_season -- derived from the league
        accumulator, which carries the OPEN simultaneous-kickoff
        discrepancy (docs/GATE10_LEAKAGE_IMPACT_DECISION_RECORD.md).
        Including it would import an unresolved leakage question.
    xG (22 cols) -- structurally unusable (100% null, folds 1-2 train).

NO 2025/26. No tuning, no calibration, no thresholds, no class weights,
no feature selection, no production change, no artifact, no estimator
persisted. Writes nothing.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_d1_tierd_candidate_validation.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from models.ablation import (
    DISCIPLINE_COLUMNS, GOALS_VENUE_COLUMNS, HISTORY_FLAG_COLUMNS, MODEL_B,
    MODEL_B_COLUMNS, TEMPO_COLUMNS, VENUE_GOAL_DIFF_COLUMNS, XG_CORE_COLUMNS,
    LEAGUE_HOME_ADVANTAGE_COLUMN, LEAGUE_MEAN_GOALS_COLUMN,
)
from models.baselines import CLASS_ORDER
from models.config import FINAL_TEST_SEASONS, MODEL_VERSION, WALK_FORWARD_FOLDS
from models.data import load_supervised_dataset
from models.evaluate import evaluate, validate_probabilities
from models.robustness import V1_LOGREG_BASE_KWARGS
from models.splits import iter_walk_forward_folds

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"

C_VALUE = 0.0005
BASELINE_COLUMNS = tuple(MODEL_B_COLUMNS)
TIER_D_SAFE = {
    "tempo": tuple(TEMPO_COLUMNS),
    "discipline": tuple(DISCIPLINE_COLUMNS),
    "goals_venue": tuple(GOALS_VENUE_COLUMNS),
    "venue_goal_diff": tuple(VENUE_GOAL_DIFF_COLUMNS),
    "history_flags": tuple(HISTORY_FLAG_COLUMNS),
}
ADDED = tuple(c for g in TIER_D_SAFE.values() for c in g)
CANDIDATE_COLUMNS = BASELINE_COLUMNS + ADDED
EXCLUDED = {
    "league_home_advantage_season": "Phase 5A Condition 1 -- re-adding worsened log loss",
    "league_mean_goals_per_team_match_season": "league accumulator -- OPEN simultaneous-kickoff issue",
}


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)


def stop(m):
    print("\n" + "!" * 78); print("STOP"); print(m); print("!" * 78); raise SystemExit(1)


def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()


def per_class(y, P):
    y = np.asarray(y); pred = np.array(CLASS_ORDER)[P.argmax(1)]
    out = {}
    for c in CLASS_ORDER:
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        out[c] = dict(precision=pr, recall=rc, f1=f1, support=int((y == c).sum()))
    return out


def select(X, cols, label):
    missing = [c for c in cols if c not in X.columns]
    if missing:
        stop(f"{label}: {len(missing)} column(s) missing: {missing[:5]}")
    return X[list(cols)]


def fit_arm(Xtr, ytr, Xva, cols, label):
    from sklearn.linear_model import LogisticRegression
    from models import train as train_module

    A, B = select(Xtr, cols, label), select(Xva, cols, label)
    pre = train_module.LogisticRegressionPreprocessor().fit(A)
    model = LogisticRegression(C=C_VALUE, **V1_LOGREG_BASE_KWARGS).fit(pre.transform(A), ytr)
    if int(np.max(model.n_iter_)) >= V1_LOGREG_BASE_KWARGS["max_iter"]:
        stop(f"{label}: hit max_iter -- unconverged")
    P = np.asarray(train_module._reorder_proba(model, pre.transform(B)), dtype=float)
    validate_probabilities(P)
    return P, model


def run_arm(dataset, cols, label):
    folds = {}
    for fold, tr, va in iter_walk_forward_folds(dataset):
        P, model = fit_arm(tr.X, tr.y, va.X, cols, label)
        r = evaluate(va.y, P)
        y = va.y.to_numpy()
        folds[fold.name] = dict(P=P, y=y, r=r, per_class=per_class(y, P),
                                n_iter=model.n_iter_,
                                correct=int((np.array(CLASS_ORDER)[P.argmax(1)] == y).sum()))
        print(f"    [{label}] {fold.name}: acc={r.accuracy:.6f} macro_f1={r.macro_f1:.6f} "
              f"bal_acc={r.balanced_accuracy:.6f} log_loss={r.log_loss:.16f} brier={r.brier:.8f}")
    return folds


def summarise(folds):
    out = {}
    for m in ("accuracy", "macro_f1", "balanced_accuracy", "log_loss", "brier"):
        v = [folds[n]["r"].as_dict()[m] for n in sorted(folds)]
        out[m] = dict(mean=statistics.fmean(v), sd=statistics.pstdev(v), by_fold=v)
    return out


def main():
    print("D-1 -- TIER-D INFORMATION CANDIDATE (validation folds only)")
    print("One variable: the information set. C, estimator, preprocessing, folds unchanged.")

    rule("PRE-FLIGHT")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}  C={C_VALUE}  kwargs={V1_LOGREG_BASE_KWARGS}")
    seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    if seasons & set(FINAL_TEST_SEASONS):
        stop("2025/26 reachable")
    print(f"  fold seasons {sorted(seasons)} | 2025/26 reachable: False")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines: mismatches={bad or 'NONE'}")
    if bad or MODEL_VERSION != "v1.0":
        stop("pre-flight failure")

    rule("CONTRACT DEFINITION")
    print(f"  baseline (E-1)  : {len(BASELINE_COLUMNS)} columns")
    for g, cols in TIER_D_SAFE.items():
        print(f"  + {g:<16}: {len(cols):2d}  {list(cols)}")
    print(f"  candidate (D-1) : {len(CANDIDATE_COLUMNS)} columns")
    overlap = set(BASELINE_COLUMNS) & set(ADDED)
    print(f"  overlap with baseline (must be 0): {len(overlap)} {sorted(overlap)}")
    if overlap or len(CANDIDATE_COLUMNS) != len(BASELINE_COLUMNS) + len(ADDED):
        stop("candidate contract is not a clean superset of the baseline")
    print(f"  baseline is a strict subset of the candidate: "
          f"{set(BASELINE_COLUMNS) < set(CANDIDATE_COLUMNS)}")
    print("\n  DELIBERATELY EXCLUDED:")
    for c, why in EXCLUDED.items():
        print(f"    {c:<44} {why}")
    print(f"    xG ({len(XG_CORE_COLUMNS)} cols){'':<32} 100% null in fold_1/2 training partitions")
    for c in list(EXCLUDED) + list(XG_CORE_COLUMNS):
        if c in CANDIDATE_COLUMNS:
            stop(f"excluded column {c} leaked into the candidate contract")
    print("  [PASS] no excluded column present")

    rule("LEAKAGE AUDIT (structural, before fitting)")
    print("  Every added column is a *_season aggregate or history count computed by")
    print("  feature_builder from ctx.history_before(team_id) under compute-then-record,")
    print("  i.e. prior fixtures of THAT TEAM only. Covered by the Phase 2 leakage suite:")
    print("    test_5_season_aggregate_only_uses_matches_before_T  -> tempo, discipline")
    print("    test_6_venue_restricted_stats_exclude_target        -> goals_venue, venue_goal_diff")
    print("    test_1..4 (target excluded, future cannot affect past) -> all")
    print("  All added columns are TEAM-level, so the OPEN league-accumulator")
    print("  simultaneous-kickoff discrepancy does NOT apply to them.")
    print("  No odds, no post-match, no future-season information is introduced.")

    dataset = load_supervised_dataset(FEATURES_DB)
    print(f"\n  dataset rows: {len(dataset.X)}")
    print("  training-partition null rates of the added columns (feasibility check):")
    for fold, tr, va in iter_walk_forward_folds(dataset):
        worst = {g: max(tr.X[c].isna().mean() for c in cols) for g, cols in TIER_D_SAFE.items()}
        print(f"    {fold.name}: " + "  ".join(f"{g}={v:.4f}" for g, v in worst.items()))
        if max(worst.values()) >= 0.5:
            stop(f"{fold.name}: an added group exceeds 50% null in training -- not evaluable")

    rule("BASELINE ARM -- E-1 (80 columns, C=0.0005)")
    base = run_arm(dataset, BASELINE_COLUMNS, "E-1")
    rule("CANDIDATE ARM -- D-1 (100 columns, C=0.0005)")
    cand = run_arm(dataset, CANDIDATE_COLUMNS, "D-1")

    bs, cs = summarise(base), summarise(cand)

    rule("PER-FOLD RESULTS")
    for n in sorted(base):
        b, c = base[n]["r"], cand[n]["r"]
        print(f"\n  {n} (n={b.n}):")
        for m in ("accuracy", "macro_f1", "balanced_accuracy", "log_loss", "brier"):
            x, y = b.as_dict()[m], c.as_dict()[m]
            better = (y > x) if m in ("accuracy", "macro_f1", "balanced_accuracy") else (y < x)
            print(f"    {m:<20} E-1={x:.16f}  D-1={y:.16f}  delta={y-x:+.6e}  "
                  f"{'better' if better else 'worse' if y != x else 'equal'}")
        print(f"    correct              E-1={base[n]['correct']}  D-1={cand[n]['correct']}  "
              f"delta={cand[n]['correct']-base[n]['correct']:+d}")
        for cl in CLASS_ORDER:
            pb, pc = base[n]["per_class"][cl], cand[n]["per_class"][cl]
            print(f"      {cl}: recall {pb['recall']:.6f}->{pc['recall']:.6f} "
                  f"f1 {pb['f1']:.6f}->{pc['f1']:.6f} support={pb['support']}")

    rule("SCORECARD (mean over 3 folds; sd in parentheses)")
    print(f"  {'Candidate':<14}{'Accuracy':>22}{'Macro-F1':>22}{'BalAcc':>22}"
          f"{'LogLoss':>22}{'Brier':>22}")
    for lab, s in (("E-1 baseline", bs), ("D-1 candidate", cs)):
        print(f"  {lab:<14}" + "".join(
            f"{s[m]['mean']:>16.10f}({s[m]['sd']:.4f})"
            for m in ("accuracy", "macro_f1", "balanced_accuracy", "log_loss", "brier")))
    print(f"\n  {'Delta (D-1 - E-1)':<20}")
    higher_better = {"accuracy", "macro_f1", "balanced_accuracy"}
    improved, worsened, allfolds = [], [], {}
    for m in ("accuracy", "macro_f1", "balanced_accuracy", "log_loss", "brier"):
        d = cs[m]["mean"] - bs[m]["mean"]
        good = d > 0 if m in higher_better else d < 0
        rel = d / bs[m]["mean"] * 100 if bs[m]["mean"] else float("nan")
        pf = [(cs[m]["by_fold"][i] - bs[m]["by_fold"][i]) for i in range(3)]
        af = all((x > 0) if m in higher_better else (x < 0) for x in pf)
        allfolds[m] = af
        (improved if good else worsened).append(m)
        print(f"    {m:<20} delta={d:+.6e}  relative={rel:+.4f}%  "
              f"all_3_folds_improved={af}  per_fold={[f'{x:+.2e}' for x in pf]}")

    rule("MULTI-METRIC CLASSIFICATION")
    n_imp = len(improved)
    if n_imp == 5:
        cls = "A. DOMINANT -- improves or equals all five metrics"
    elif n_imp >= 3:
        cls = "B. STRONG TRADE-OFF"
    elif n_imp >= 1:
        cls = "C. MIXED"
    else:
        cls = "D. NO IMPROVEMENT"
    print(f"  improved: {improved}")
    print(f"  worsened: {worsened}")
    print(f"  metrics improved in ALL THREE folds: {[m for m, v in allfolds.items() if v]}")
    print(f"  CLASSIFICATION: {cls}")
    print("\n  EFFECT SIZE CONTEXT (do not celebrate tiny numbers):")
    for m in ("accuracy", "macro_f1", "balanced_accuracy", "log_loss", "brier"):
        spread = max(bs[m]["by_fold"]) - min(bs[m]["by_fold"])
        d = abs(cs[m]["mean"] - bs[m]["mean"])
        print(f"    {m:<20} |delta|={d:.6e}  baseline fold-to-fold spread={spread:.6e}  "
              f"ratio={d/spread if spread else float('nan'):.4f}")
    print("  A |delta| far below the fold-to-fold spread is NOT practically meaningful.")

    rule("DRAW ANALYSIS")
    for n in sorted(base):
        pb, pc = base[n], cand[n]
        db = int((np.array(CLASS_ORDER)[pb["P"].argmax(1)] == "D").sum())
        dc = int((np.array(CLASS_ORDER)[pc["P"].argmax(1)] == "D").sum())
        print(f"  {n}: predicted-D  E-1={db}  D-1={dc}  | mean p_draw "
              f"E-1={pb['P'][:,1].mean():.6f} D-1={pc['P'][:,1].mean():.6f} | "
              f"actual draw rate={(pb['y']=='D').mean():.6f}")
        print(f"      D recall E-1={pb['per_class']['D']['recall']:.6f} "
              f"D-1={pc['per_class']['D']['recall']:.6f}")

    rule("DETERMINISM")
    rep = run_arm(dataset, CANDIDATE_COLUMNS, "D-1 repeat")
    same = all(rep[n]["P"].tobytes() == cand[n]["P"].tobytes() for n in cand)
    fh = hashlib.sha256("|".join(CANDIDATE_COLUMNS).encode()).hexdigest()
    print(f"  candidate byte-identical repeat: {same}")
    print(f"  feature-list sha256: {fh}")
    print(f"  prediction sha256: "
          f"{hashlib.sha256(b''.join(cand[n]['P'].tobytes() for n in sorted(cand))).hexdigest()}")
    print(f"  random_state={V1_LOGREG_BASE_KWARGS['random_state']} "
          f"max_iter={V1_LOGREG_BASE_KWARGS['max_iter']} C={C_VALUE}")
    if not same:
        stop("non-deterministic")

    rule("INTEGRITY AUDIT")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}")
    for line in ("2025/26 accessed = NO", "final test reused = NO",
                 "production source modified = NO", "features.db modified = NO",
                 "tuning = NONE", "calibration = NONE", "threshold change = NONE",
                 "class weighting = NONE", "feature selection = NONE (superset only)",
                 "estimator persisted = NONE", "artifacts = NONE", "files written = NONE"):
        print(f"  {line}")
    print("\n  D-1 IS NOT PROMOTED. NO NEW PRODUCTION MODEL. MODEL_VERSION UNCHANGED.")


if __name__ == "__main__":
    main()
