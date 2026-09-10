"""D-2 -- Tier-D FAMILY ABLATION (validation folds only).

QUESTION
    D-1 (E-1 + all 20 Tier-D columns) improved log loss and Brier in all
    three folds while degrading accuracy and balanced accuracy in all
    three. Which of the five pre-registered families drives which side of
    that trade-off?

ARMS (six; one variable each -- the added family)
    E-1 baseline        80 columns
    + TEMPO              6
    + DISCIPLINE         4
    + GOALS_VENUE        4
    + VENUE_GOAL_DIFF    2
    + HISTORY_FLAGS      4
    Everything else identical: LogisticRegression, C=0.0005, max_iter=2000,
    random_state=0, frozen preprocessing, the same three walk-forward folds.

NOT PERFORMED
    No C tuning, no calibration, no thresholds, no class weighting, no
    feature-by-feature selection, no combination arms (a combination is
    permitted only if the family-level results give a pre-existing reason,
    which this script does not assume), no 2025/26, no final-test reuse,
    no production change, no estimator persistence, no artifact. Writes
    nothing; stdout only.

PRE-REGISTERED DECISION RULE (applied in code, fixed before the run)
    A family is a SERIOUS CANDIDATE only if:
      (1) log loss improves in all 3 folds, AND
      (2) Brier improves in all 3 folds, AND
      (3) accuracy does NOT degrade on the mean, AND
      (4) balanced accuracy does NOT degrade on the mean, AND
      (5) macro-F1 does NOT degrade on the mean, AND
      (6) the log-loss gain is not negligible against the baseline's own
          fold-to-fold spread.
    A family that reproduces D-1's pattern -- probability metrics up,
    classification metrics down -- is NOT a candidate.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_d2_tierd_family_ablation.py
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
    DISCIPLINE_COLUMNS, GOALS_VENUE_COLUMNS, HISTORY_FLAG_COLUMNS,
    MODEL_B_COLUMNS, TEMPO_COLUMNS, VENUE_GOAL_DIFF_COLUMNS, XG_CORE_COLUMNS,
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
BASE = tuple(MODEL_B_COLUMNS)
FAMILIES = {
    "TEMPO": tuple(TEMPO_COLUMNS),
    "DISCIPLINE": tuple(DISCIPLINE_COLUMNS),
    "GOALS_VENUE": tuple(GOALS_VENUE_COLUMNS),
    "VENUE_GOAL_DIFF": tuple(VENUE_GOAL_DIFF_COLUMNS),
    "HISTORY_FLAGS": tuple(HISTORY_FLAG_COLUMNS),
}
FORBIDDEN = ("league_home_advantage_season", "league_mean_goals_per_team_match_season")
HIGHER_BETTER = ("accuracy", "macro_f1", "balanced_accuracy")
LOWER_BETTER = ("log_loss", "brier")
METRICS = HIGHER_BETTER + LOWER_BETTER


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
    names = list(pre.numeric_columns) + [f"competition_id=={c}" for c in pre.competition_categories]
    return P, model, names


def run_arm(dataset, cols, label):
    folds = {}
    for fold, tr, va in iter_walk_forward_folds(dataset):
        P, model, names = fit_arm(tr.X, tr.y, va.X, cols, label)
        r = evaluate(va.y, P)
        y = va.y.to_numpy()
        folds[fold.name] = dict(P=P, y=y, r=r, per_class=per_class(y, P),
                                model=model, names=names)
    means = {m: statistics.fmean(folds[n]["r"].as_dict()[m] for n in sorted(folds))
             for m in METRICS}
    by_fold = {m: [folds[n]["r"].as_dict()[m] for n in sorted(folds)] for m in METRICS}
    return dict(folds=folds, means=means, by_fold=by_fold, cols=cols, label=label)


def main():
    print("D-2 -- TIER-D FAMILY ABLATION (validation folds only, one family per arm)")

    rule("PRE-FLIGHT")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}  C={C_VALUE}  kwargs={V1_LOGREG_BASE_KWARGS}")
    seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    if seasons & set(FINAL_TEST_SEASONS):
        stop("2025/26 reachable")
    print(f"  fold seasons {sorted(seasons)} (2020/21-2024/25) | 2025/26 reachable: False")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines: mismatches={bad or 'NONE'}")
    if bad or MODEL_VERSION != "v1.0":
        stop("pre-flight failure")

    rule("ARM CONTRACTS")
    total = 0
    for fam, cols in FAMILIES.items():
        overlap = set(BASE) & set(cols)
        bad_cols = [c for c in cols if c in FORBIDDEN or c in XG_CORE_COLUMNS]
        print(f"  E-1 + {fam:<16} = {len(BASE)} + {len(cols):2d} = {len(BASE)+len(cols)} columns "
              f"| overlap={len(overlap)} | forbidden={bad_cols or 'none'}")
        if overlap or bad_cols:
            stop(f"{fam}: contract violation")
        total += len(cols)
    print(f"  families total = {total} columns (must equal D-1's 20): {total == 20}")
    if total != 20:
        stop("family decomposition does not reconstruct D-1's 20 columns")

    dataset = load_supervised_dataset(FEATURES_DB)
    print(f"  dataset rows: {len(dataset.X)}")

    rule("FITTING SIX ARMS")
    arms = {"E-1": run_arm(dataset, BASE, "E-1")}
    print(f"  E-1 baseline: " + "  ".join(f"{m}={arms['E-1']['means'][m]:.10f}" for m in METRICS))
    for fam, cols in FAMILIES.items():
        arms[fam] = run_arm(dataset, BASE + cols, f"E-1+{fam}")
        print(f"  E-1+{fam:<16}: " + "  ".join(
            f"{m}={arms[fam]['means'][m]:.10f}" for m in METRICS))

    base = arms["E-1"]
    spread = {m: max(base["by_fold"][m]) - min(base["by_fold"][m]) for m in METRICS}

    rule("PER-FOLD RESULTS")
    for fam in FAMILIES:
        print(f"\n  --- E-1 + {fam} ---")
        for i, n in enumerate(sorted(base["folds"])):
            b, c = base["folds"][n]["r"], arms[fam]["folds"][n]["r"]
            line = "  ".join(
                f"{m}: {b.as_dict()[m]:.8f}->{c.as_dict()[m]:.8f} ({c.as_dict()[m]-b.as_dict()[m]:+.2e})"
                for m in METRICS)
            print(f"    {n}: {line}")

    rule("SCORECARD -- mean over 3 folds, delta vs E-1")
    print(f"  {'Arm':<18}{'Accuracy':>14}{'Macro-F1':>14}{'BalAcc':>14}{'LogLoss':>16}{'Brier':>14}")
    print(f"  {'E-1 baseline':<18}" + "".join(f"{base['means'][m]:>14.8f}" if m != 'log_loss'
                                              else f"{base['means'][m]:>16.10f}" for m in METRICS))
    print("  " + "-" * 90)
    verdicts = {}
    for fam in FAMILIES:
        a = arms[fam]
        d = {m: a["means"][m] - base["means"][m] for m in METRICS}
        allf = {m: all((a["by_fold"][m][i] > base["by_fold"][m][i]) if m in HIGHER_BETTER
                       else (a["by_fold"][m][i] < base["by_fold"][m][i]) for i in range(3))
                for m in METRICS}
        print(f"  {'+'+fam:<18}" + "".join(
            f"{d[m]:>+14.8f}" if m != 'log_loss' else f"{d[m]:>+16.10f}" for m in METRICS))
        print(f"  {'  all-3-folds':<18}" + "".join(
            f"{str(allf[m]):>14}" if m != 'log_loss' else f"{str(allf[m]):>16}" for m in METRICS))
        # pre-registered rule
        c1 = allf["log_loss"]
        c2 = allf["brier"]
        c3 = d["accuracy"] >= 0
        c4 = d["balanced_accuracy"] >= 0
        c5 = d["macro_f1"] >= 0
        c6 = abs(d["log_loss"]) > 0.10 * spread["log_loss"]
        serious = c1 and c2 and c3 and c4 and c5 and c6
        d1_pattern = c1 and c2 and (d["accuracy"] < 0 or d["balanced_accuracy"] < 0)
        verdicts[fam] = dict(delta=d, allf=allf, serious=serious, d1_pattern=d1_pattern,
                             checks=dict(ll_all3=c1, brier_all3=c2, acc_ok=c3,
                                         balacc_ok=c4, f1_ok=c5, effect_ok=c6))

    rule("PRE-REGISTERED RULE EVALUATION")
    for fam, v in verdicts.items():
        print(f"\n  E-1 + {fam}")
        for k, ok in v["checks"].items():
            print(f"    {k:<12} {ok}")
        print(f"    => {'SERIOUS CANDIDATE' if v['serious'] else 'NOT A CANDIDATE'}"
              f"{'  (reproduces D-1 pattern: probability up, classification down)' if v['d1_pattern'] else ''}")

    rule("EFFECT-SIZE CONTEXT vs BASELINE FOLD-TO-FOLD SPREAD")
    print(f"  {'metric':<20}{'baseline spread':>18}" + "".join(f"{f'+{f}':>18}" for f in FAMILIES))
    for m in METRICS:
        row = "".join(f"{abs(verdicts[f]['delta'][m])/spread[m] if spread[m] else float('nan'):>18.4f}"
                      for f in FAMILIES)
        print(f"  {m:<20}{spread[m]:>18.8f}{row}")
    print("  values are |delta| / baseline spread. Below ~0.10 is not practically meaningful.")

    rule("MECHANISM -- PREDICTED CLASS DISTRIBUTION AND PER-CLASS RECALL")
    print("  Testing whether classification degradation comes from Home over-selection,")
    print("  Away suppression, or Draw suppression.\n")
    for fam in ["E-1"] + list(FAMILIES):
        a = arms[fam]
        agg = {c: dict(pred=0, rec=[], sup=0) for c in CLASS_ORDER}
        for n in sorted(a["folds"]):
            pc = a["folds"][n]["per_class"]
            for c in CLASS_ORDER:
                agg[c]["pred"] += pc[c]["predicted"]; agg[c]["rec"].append(pc[c]["recall"])
                agg[c]["sup"] += pc[c]["support"]
        lab = "E-1 baseline" if fam == "E-1" else f"E-1 + {fam}"
        print(f"  {lab:<24} " + "  ".join(
            f"{c}: pred={agg[c]['pred']:5d} recall={statistics.fmean(agg[c]['rec']):.6f} "
            f"support={agg[c]['sup']}" for c in CLASS_ORDER))
    print("\n  deltas vs E-1 (pooled predicted counts, mean recall):")
    b = arms["E-1"]
    bp = {c: sum(b["folds"][n]["per_class"][c]["predicted"] for n in b["folds"]) for c in CLASS_ORDER}
    br = {c: statistics.fmean(b["folds"][n]["per_class"][c]["recall"] for n in b["folds"])
          for c in CLASS_ORDER}
    for fam in FAMILIES:
        a = arms[fam]
        ap = {c: sum(a["folds"][n]["per_class"][c]["predicted"] for n in a["folds"])
              for c in CLASS_ORDER}
        ar = {c: statistics.fmean(a["folds"][n]["per_class"][c]["recall"] for n in a["folds"])
              for c in CLASS_ORDER}
        print(f"    +{fam:<16} " + "  ".join(
            f"{c}: pred{ap[c]-bp[c]:+5d} recall{ar[c]-br[c]:+.6f}" for c in CLASS_ORDER))

    rule("DIAGNOSTIC -- PROBABILITY MOVEMENT AND SHRINKAGE (not justification)")
    print("  Under C=0.0005 the ridge is strong (1/C = 2000). Adding columns can flatten")
    print("  probabilities toward the base rate: better log loss, weaker argmax separation.")
    print("  Reported descriptively; no action is taken on it.\n")
    for fam in ["E-1"] + list(FAMILIES):
        a = arms[fam]
        mm = statistics.fmean(a["folds"][n]["P"].max(1).mean() for n in a["folds"])
        sd = statistics.fmean(a["folds"][n]["P"].std(1).mean() for n in a["folds"])
        cm = statistics.fmean(float(np.abs(a["folds"][n]["model"].coef_).sum())
                              for n in a["folds"])
        lab = "E-1 baseline" if fam == "E-1" else f"E-1 + {fam}"
        print(f"  {lab:<24} mean max_p={mm:.6f}  mean row spread={sd:.6f}  "
              f"total |coef|={cm:.6f}")

    rule("DETERMINISM")
    det = True
    for fam in FAMILIES:
        rep = run_arm(dataset, BASE + FAMILIES[fam], f"repeat {fam}")
        ok = all(rep["folds"][n]["P"].tobytes() == arms[fam]["folds"][n]["P"].tobytes()
                 for n in arms[fam]["folds"])
        det &= ok
        h = hashlib.sha256(b"".join(arms[fam]["folds"][n]["P"].tobytes()
                                    for n in sorted(arms[fam]["folds"]))).hexdigest()
        print(f"  +{fam:<16} byte-identical={ok}  sha256={h[:32]}...")
    print(f"  DETERMINISM: {'PASS' if det else 'FAIL'}")
    if not det:
        stop("non-deterministic")

    rule("VERDICT")
    serious = [f for f, v in verdicts.items() if v["serious"]]
    d1like = [f for f, v in verdicts.items() if v["d1_pattern"]]
    print(f"  serious candidates      : {serious or 'NONE'}")
    print(f"  reproduce the D-1 pattern: {d1like or 'NONE'}")
    if not serious:
        print("\n  *** NO FAMILY SATISFIES THE PRE-REGISTERED RULE ***")
        print("  Classification of the Tier-D information:")
        print('  "Useful for probability estimation but not sufficient to justify promotion')
        print('   under the current multi-metric objective."')
        print("  No family is proposed. No promotion. STOP.")
    elif len(serious) == 1:
        print(f"\n  *** {serious[0]} satisfies every pre-registered condition ***")
        print("  Reported as a CANDIDATE for a separately authorized experiment.")
        print("  NOT promoted, NOT frozen, no final test, MODEL_VERSION unchanged.")
    else:
        print(f"\n  *** {len(serious)} families satisfy the rule: {serious} ***")
        print("  No combination arm is run here -- combining would require its own")
        print("  authorization and a pre-registered reason. Reported as-is.")

    rule("INTEGRITY AUDIT")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}")
    for line in ("2025/26 accessed = NO", "final test reused = NO",
                 "production source modified = NO", "features.db modified = NO",
                 "C tuning = NONE", "calibration = NONE", "threshold change = NONE",
                 "class weighting = NONE", "feature-by-feature selection = NONE",
                 "combination arms = NONE", "estimator persisted = NONE",
                 "artifacts = NONE", "files written = NONE"):
        print(f"  {line}")


if __name__ == "__main__":
    main()
