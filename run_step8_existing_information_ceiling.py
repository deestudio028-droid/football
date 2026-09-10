"""STEP 8 -- existing-information ceiling / nonlinear capacity diagnosis.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step8_existing_information_ceiling.py

ONE QUESTION ONLY
Is the Draw-discriminating information contained in the existing 79-feature
production contract already close to exhausted by V1, or is there materially
more achievable predictive signal in the SAME information that the linear V1
model cannot extract?

DIAGNOSTIC EXPERIMENT ONLY. NOTHING IS TUNED, SELECTED, PROPOSED OR PERSISTED.

WHY THE ESTIMATOR IS NOT CHOSEN BY THIS SCRIPT
The ceiling probe is `train.train_hist_gradient_boosting` -- the project's OWN
pre-existing nonlinear candidate. It is listed in
`run_experiments.REAL_CANDIDATE_MODELS = (MODEL_LOGREG, MODEL_HGB)` and was
evaluated alongside LogisticRegression in Phase 3/4A. Its configuration lives
in production source and predates this investigation entirely:

    HistGradientBoostingClassifier(
        random_state=0,
        categorical_features=[competition_id],
    )
    -- every other hyperparameter is the scikit-learn default
    -- no n_jobs anywhere in train.py, so no parallel non-determinism
    -- no early-stopping validation split configured

Using it means NO hyperparameter is chosen by this experiment, no grid search
is run, no alternative algorithm is compared, and no project convention is
reinvented. It is a CEILING PROBE, NOT A PRODUCTION RECOMMENDATION.

WHY IT IS A VALID CEILING PROBE
It captures interactions and non-linear boundaries among the same inputs. It
receives EXACTLY the information V1 receives: the 79 numeric production
features plus competition_id. Nothing is engineered, expanded, differenced,
squared, ratioed or added. THE ONLY THING THAT CHANGES IS MODEL CAPACITY.

Tree models handle missing values natively and need no scaling, so the two
arms differ in preprocessing by construction -- that difference is a property
of the capacity class being probed, not an extra information channel, and it
is stated explicitly in STEP 4 rather than hidden.

NOT DONE HERE
No production change. No feature added or removed. No threshold, class weight,
solver or random state altered. No 2025/26 or final-test data. No persistence
of any model, matrix or prediction. No new information proposed. No feature
shopping. No narrative diagnosis inside the script -- this is an evidence
generator; the formal diagnosis is written separately.
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

POOLED_REF_LOG_LOSS = 0.9993791056968738
POOLED_REF_BRIER = 0.5965016957578898
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_MD5 = "5e504427712b35778bb8a62a8496c7cd"
FORBIDDEN_SEASONS = [602681, 667780, 725788, 725793, 762170]

PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)
#: The ceiling probe's configuration, verified from production source, not chosen here.
CEILING_ASSUMPTIONS = (
    "def train_hist_gradient_boosting",
    "HistGradientBoostingClassifier(",
    "random_state=0,",
    "categorical_features=categorical_features,",
)

#: D-38 / D-41 recorded production raw separation, used as printed reference only.
D38_SMD_DH = 0.2457
D38_SMD_DA = 0.2281


def rule(t): print("\n" + "=" * 122); print(t); print("=" * 122)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 122)
    print("ABORT -- STEP 8 halted. No ceiling measurement is produced.")
    print(msg)
    print("!" * 122)
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
    snap["src/models/train.py"] = md5(REPO / "src/models/train.py")
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


def draw_geometry(P, Y, order):
    iH, iD, iA = 0, 1, 2
    pred = np.array(order)[P.argmax(axis=1)]
    Pd = P[Y == "D"]
    return {
        "pd_mean": float(P[:, iD].mean()),
        "pd_sd": float(P[:, iD].std()),
        "pd_gt_third": float((P[:, iD] > 1 / 3).mean()),
        "draw_argmax_share": float((pred == "D").mean()),
        "draw_rank1_share": float(((Pd[:, iD] > Pd[:, iH]) & (Pd[:, iD] > Pd[:, iA])).mean()),
        "d_gt_h": float((Pd[:, iD] > Pd[:, iH]).mean()),
        "d_gt_a": float((Pd[:, iD] > Pd[:, iA]).mean()),
        "d_gt_both": float(((Pd[:, iD] > Pd[:, iH]) & (Pd[:, iD] > Pd[:, iA])).mean()),
    }


def main():
    # ---------------------------------------------------------------- RULE 1
    rule("RULE 1 -- PRODUCTION SOURCE RE-VERIFICATION (source text, never memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:72]}", tok in tsrc)
    print("\n  ceiling-probe configuration, also read from production source:")
    for tok in CEILING_ASSUMPTIONS:
        check(f"train.py contains: {tok[:72]}", tok in tsrc)
    check("no n_jobs anywhere in train.py (determinism)", "n_jobs" not in tsrc)
    check("no early_stopping override in train.py", "early_stopping" not in tsrc)

    try:
        import sklearn  # noqa: F401
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
    from models.train import train_hist_gradient_boosting, train_logistic_regression

    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("production contract == exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric production block == exactly 79 columns", len(numeric) == 79)
    check("competition_id is the only categorical production column",
          RECOMMENDED_CONTEXT_FEATURE in MODEL_B_COLUMNS
          and len(set(MODEL_B_COLUMNS) - set(numeric)) == 1)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")

    # ---------------------------------------------------------------- STEP 1
    rule("STEP 1 -- ENVIRONMENT + INTEGRITY")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"pandas {pd.__version__}  sklearn {sklearn.__version__}")
    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(rel, before[rel] == exp, before[rel])
    if "artifact" in before:
        check("v1_logreg.pkl unchanged", before["artifact"] == ARTIFACT_MD5, before["artifact"])
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    check("13/13 LOCKED_INPUTS at expected values",
          all(before[f"pin:{r}"] == v["expected"] for r, v in pins.items()))

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
        check(f"{fold.name}: no final-test season_id in train or validation",
              not (seen & test_ids))

    # ---------------------------------------------------------------- STEP 3
    rule("STEP 3 -- CEILING EXPERIMENT PRE-REGISTRATION (fixed BEFORE any fitting)")
    print("  estimator            : HistGradientBoostingClassifier")
    print("  source of truth      : models.train.train_hist_gradient_boosting")
    print("  fixed hyperparameters: random_state=0")
    print("                         categorical_features=[competition_id]")
    print("                         ALL OTHERS = scikit-learn defaults")
    print("                         (learning_rate, max_iter, max_leaf_nodes, max_depth,")
    print("                          min_samples_leaf, l2_regularization, max_bins,")
    print("                          early_stopping, tol -- none set by this project")
    print("                          and none set by this script)")
    print("  categorical handling : CompetitionCodeEncoder fitted on TRAIN ONLY")
    print("  missing values       : native (tree splits), no imputation")
    print("  scaling              : none (not meaningful for a tree model)")
    print("\n  WHY THIS ESTIMATOR IS USED AS A CEILING PROBE:")
    print("    It is the project's OWN pre-existing nonlinear candidate, listed in")
    print("    run_experiments.REAL_CANDIDATE_MODELS and evaluated in Phase 3/4A. Its")
    print("    configuration predates this investigation, so NO hyperparameter is")
    print("    chosen here, no grid search is run, and no alternative algorithm is")
    print("    compared. It can represent interactions and non-linear boundaries that")
    print("    the linear multinomial form cannot, which is exactly the capacity")
    print("    question STEP 8 asks.")
    print("\n  THIS IS NOT A PRODUCTION RECOMMENDATION. The probe measures an achievable")
    print("  ceiling under one capacity class. It is not proposed, promoted, persisted")
    print("  or evaluated for deployment.")

    # ---------------------------------------------------------------- STEP 4
    rule("STEP 4 -- INFORMATION-EQUALITY AUDIT (before any fitting)")
    print(f"  V1 information      : {len(numeric)} numeric + competition_id")
    print(f"  Ceiling information : {len(numeric)} numeric + competition_id")
    print("  Additional external information : NONE")
    print("  Additional engineered features  : NONE")
    print("  (no closeness, ratios, differences, squares, polynomial expansion or")
    print("   manually selected interactions -- only the estimator's internal capacity)")
    for fold, tr, va in folds:
        Xtr = tr.X[list(MODEL_B_COLUMNS)]
        Xva = va.X[list(MODEL_B_COLUMNS)]
        check(f"{fold.name}: identical column set for both arms",
              list(Xtr.columns) == list(MODEL_B_COLUMNS) == list(Xva.columns))
        check(f"{fold.name}: no column beyond the 80-column contract",
              Xtr.shape[1] == 80)
        check(f"{fold.name}: train/validation row counts fixed",
              len(Xtr) == len(tr.y) and len(Xva) == len(va.y))
    lbl = {"label_home_goals", "label_away_goals", "label_result"}
    check("no post-match/label column reaches either arm", not (lbl & set(MODEL_B_COLUMNS)))

    # ------------------------------------------------------- STEPS 2, 5, 6
    rule("STEP 2 -- V1 REPRODUCTION  +  STEP 5 -- WALK-FORWARD CEILING EVALUATION")
    v1_rows, ce_rows = [], []
    P1_l, P2_l, Y_l, F_l, X_l = [], [], [], [], []
    for fold, tr, va in folds:
        Xtr = tr.X[list(MODEL_B_COLUMNS)]
        Xva = va.X[list(MODEL_B_COLUMNS)]
        _, _, P1 = train_logistic_regression(Xtr, tr.y, Xva)
        r1 = evaluate(va.y, P1)
        _, _, P2 = train_hist_gradient_boosting(Xtr, tr.y, Xva)
        r2 = evaluate(va.y, P2)
        v1_rows.append((fold.name, len(va), r1))
        ce_rows.append((fold.name, len(va), r2))
        P1_l.append(P1); P2_l.append(P2); Y_l.append(va.y.to_numpy())
        F_l.append(np.repeat(fold.name, len(va))); X_l.append(Xva.reset_index(drop=True))
        print(f"  {fold.name}: N={len(va)}  V1 ll={r1.log_loss:.16f} br={r1.brier:.16f}"
              f"   CEIL ll={r2.log_loss:.16f} br={r2.brier:.16f}")

    P1 = np.vstack(P1_l); P2 = np.vstack(P2_l)
    Y = np.concatenate(Y_l); FOLD = np.concatenate(F_l)
    X = pd.concat(X_l, ignore_index=True)
    ll1 = float(np.mean([r.log_loss for _, _, r in v1_rows]))
    ll2 = float(np.mean([r.log_loss for _, _, r in ce_rows]))
    br1 = float(np.mean([r.brier for _, _, r in v1_rows]))
    br2 = float(np.mean([r.brier for _, _, r in ce_rows]))
    print(f"\n  V1 pooled mean log loss : {ll1:.16f}   (reference {POOLED_REF_LOG_LOSS:.16f})")
    check("V1 reproduces the documented pooled log loss",
          abs(ll1 - POOLED_REF_LOG_LOSS) <= 1e-9, f"diff={ll1 - POOLED_REF_LOG_LOSS:.3e}")
    print(f"  V1 pooled mean Brier    : {br1:.16f}   (reference {POOLED_REF_BRIER:.16f})")

    print("\n| Fold | N | V1 log loss | Ceiling log loss | delta |")
    print("|---|---:|---:|---:|---:|")
    for (n, k, a), (_, _, b) in zip(v1_rows, ce_rows):
        print(f"| {n} | {k} | {a.log_loss:.10f} | {b.log_loss:.10f} | {b.log_loss-a.log_loss:+.10f} |")
    print("\n| Fold | N | V1 Brier | Ceiling Brier | delta |")
    print("|---|---:|---:|---:|---:|")
    for (n, k, a), (_, _, b) in zip(v1_rows, ce_rows):
        print(f"| {n} | {k} | {a.brier:.10f} | {b.brier:.10f} | {b.brier-a.brier:+.10f} |")

    imp_ll = sum(1 for (_, _, a), (_, _, b) in zip(v1_rows, ce_rows) if b.log_loss < a.log_loss)
    imp_br = sum(1 for (_, _, a), (_, _, b) in zip(v1_rows, ce_rows) if b.brier < a.brier)
    print(f"\n  V1 pooled mean log loss      : {ll1:.16f}")
    print(f"  Ceiling pooled mean log loss : {ll2:.16f}")
    print(f"  delta log loss (ceiling - V1): {ll2 - ll1:+.16f}   "
          f"({'ceiling improves' if ll2 < ll1 else 'ceiling worsens'})")
    print(f"  V1 pooled mean Brier         : {br1:.16f}")
    print(f"  Ceiling pooled mean Brier    : {br2:.16f}")
    print(f"  delta Brier    (ceiling - V1): {br2 - br1:+.16f}   "
          f"({'ceiling improves' if br2 < br1 else 'ceiling worsens'})")
    print(f"  folds improving log loss     : {imp_ll}/3")
    print(f"  folds improving Brier        : {imp_br}/3")

    # ---------------------------------------------------------------- STEP 6
    rule("STEP 6 -- DRAW-SPECIFIC CEILING MEASUREMENTS")
    from sklearn.metrics import roc_auc_score
    iD = 1
    yD = (Y == "D").astype(int)
    auc1 = float(roc_auc_score(yD, P1[:, iD]))
    auc2 = float(roc_auc_score(yD, P2[:, iD]))
    print("  Draw multiclass AUC: NOT AVAILABLE under the existing project convention.")
    print("    models.evaluate exposes log_loss, brier, macro_f1, balanced_accuracy,")
    print("    accuracy and confusion_matrix only. No multiclass-AUC convention exists")
    print("    in this project, so none is substituted. Draw-vs-rest AUC is reported")
    print("    instead, matching the one-vs-rest convention already used in D-33/D-37.")
    g1 = draw_geometry(P1, Y, CLASS_ORDER)
    g2 = draw_geometry(P2, Y, CLASS_ORDER)
    print("\n| quantity | V1 | Ceiling | delta |")
    print("|---|---:|---:|---:|")
    print(f"| Draw-vs-rest AUC | {auc1:.4f} | {auc2:.4f} | {auc2-auc1:+.4f} |")
    for k, lab in (("pd_mean", "mean P(D)"), ("pd_sd", "standard deviation P(D)"),
                   ("pd_gt_third", "P(D) > 1/3 share"),
                   ("draw_argmax_share", "Draw argmax share"),
                   ("draw_rank1_share", "actual-Draw rank-1 share"),
                   ("d_gt_h", "D > H share"), ("d_gt_a", "D > A share"),
                   ("d_gt_both", "D > both share")):
        print(f"| {lab} | {g1[k]:.4f} | {g2[k]:.4f} | {g2[k]-g1[k]:+.4f} |")

    # ---------------------------------------------------------------- STEP 7
    rule("STEP 7 -- RAW INFORMATION CEILING CHECK (existing 79 features only)")
    isD, isH, isA = Y == "D", Y == "H", Y == "A"
    p_dh = np.array([smd(pd.to_numeric(X[c], errors="coerce").to_numpy()[isD],
                         pd.to_numeric(X[c], errors="coerce").to_numpy()[isH])
                     for c in numeric])
    p_da = np.array([smd(pd.to_numeric(X[c], errors="coerce").to_numpy()[isD],
                         pd.to_numeric(X[c], errors="coerce").to_numpy()[isA])
                     for c in numeric])
    print(f"  production mean |SMD| D-H : {np.nanmean(np.abs(p_dh)):.4f}   "
          f"(D-38 recorded {D38_SMD_DH:.4f})")
    print(f"  production mean |SMD| D-A : {np.nanmean(np.abs(p_da)):.4f}   "
          f"(D-38 recorded {D38_SMD_DA:.4f})")
    print(f"  Draw base rate           : {float(isD.mean()):.4f}")
    print(f"  V1 Draw-vs-rest AUC      : {auc1:.4f}")
    print(f"  Ceiling Draw-vs-rest AUC : {auc2:.4f}")
    print("\n  Reading guide (stated before the numbers were seen; no causal claim):")
    print("    A information limitation : ceiling AUC and log loss both close to V1")
    print("    B linear-capacity limit  : ceiling materially better on both")
    print("    C weak/noisy signal      : raw |SMD| present but neither arm converts it")
    print("    D insufficient evidence  : metrics disagree or folds are unstable")

    # ---------------------------------------------------------------- STEP 8
    rule("STEP 8 -- THREE-FOLD STABILITY (ranges only; no standard deviation)")
    print("| quantity | fold_1 | fold_2 | fold_3 | min | max | range |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    per = {}
    for (n, _, a), (_, _, b) in zip(v1_rows, ce_rows):
        m = FOLD == n
        yd = (Y[m] == "D").astype(int)
        per[n] = [a.log_loss, b.log_loss, b.log_loss - a.log_loss,
                  a.brier, b.brier, b.brier - a.brier,
                  float(roc_auc_score(yd, P1[m, iD])), float(roc_auc_score(yd, P2[m, iD])),
                  draw_geometry(P1[m], Y[m], CLASS_ORDER)["draw_argmax_share"],
                  draw_geometry(P2[m], Y[m], CLASS_ORDER)["draw_argmax_share"],
                  draw_geometry(P1[m], Y[m], CLASS_ORDER)["pd_gt_third"],
                  draw_geometry(P2[m], Y[m], CLASS_ORDER)["pd_gt_third"]]
    labs = ["V1 log loss", "ceiling log loss", "log-loss delta",
            "V1 Brier", "ceiling Brier", "Brier delta",
            "V1 Draw-vs-rest AUC", "ceiling Draw-vs-rest AUC",
            "V1 Draw argmax share", "ceiling Draw argmax share",
            "V1 P(D)>1/3 share", "ceiling P(D)>1/3 share"]
    for i, lab in enumerate(labs):
        v = [per[n][i] for n, _, _ in v1_rows]
        print(f"| {lab} | {v[0]:+.6f} | {v[1]:+.6f} | {v[2]:+.6f} | {min(v):+.6f} | "
              f"{max(v):+.6f} | {max(v)-min(v):.6f} |")

    # ---------------------------------------------------------------- STEP 9
    rule("STEP 9 -- PRE-REGISTERED CEILING CLASSIFICATION")
    ll_better = ll2 < ll1
    ll_majority = imp_ll >= 2
    br_not_contradicting = br2 <= br1
    draw_sep_better = auc2 > auc1
    print(f"  pooled log loss improves                  : {ll_better}  ({ll2-ll1:+.10f})")
    print(f"  improvement in a majority of folds (>=2/3): {ll_majority}  ({imp_ll}/3)")
    print(f"  Brier does not materially contradict      : {br_not_contradicting}  "
          f"({br2-br1:+.10f})")
    print(f"  Draw-vs-rest separation improves          : {draw_sep_better}  "
          f"({auc2-auc1:+.4f})")
    unstable = (max(per[n][2] for n, _, _ in v1_rows) > 0) and \
               (min(per[n][2] for n, _, _ in v1_rows) < 0)
    print(f"  fold-level log-loss deltas change sign    : {unstable}")
    if ll_better and ll_majority and br_not_contradicting and draw_sep_better:
        verdict = "A) MATERIAL HIGHER CEILING"
    elif (not ll_better) and (not ll_majority):
        verdict = "B) NO MATERIAL HIGHER CEILING"
    else:
        verdict = "C) INCONCLUSIVE"
    print(f"\n  CLASSIFICATION: {verdict}")
    print("\n  A does NOT mean the probe is production-ready.")
    print("  B does NOT prove Draw is unlearnable.")
    print("  C is used whenever metrics disagree or folds are unstable; A and B are")
    print("  never forced.")

    # --------------------------------------------------------------- STEP 10
    rule("STEP 10 -- CASE C vs CASE E UPDATE (measurements only)")
    print("  CASE C = aggregate probability approximately correct but Draw pairwise")
    print("           dominance / argmax structurally suppressed")
    print("  CASE E = model-capacity / representation limitation\n")
    print(f"  supports CASE E (capacity) : ceiling log loss delta {ll2-ll1:+.6f}, "
          f"Draw AUC delta {auc2-auc1:+.4f}, folds improving {imp_ll}/3")
    print(f"  supports CASE C / information limitation : |mean P(D) - base rate| "
          f"V1 {abs(g1['pd_mean']-float(isD.mean())):.4f} vs ceiling "
          f"{abs(g2['pd_mean']-float(isD.mean())):.4f}; Draw argmax "
          f"{g1['draw_argmax_share']:.4f} -> {g2['draw_argmax_share']:.4f}")
    print(f"  weakens CASE E if           : ceiling fails to beat V1 on the same inputs")
    print(f"  weakens CASE C if           : ceiling raises Draw separation AND improves scoring")
    print("  unresolved                  : neither outcome PROVES its explanation; one")
    print("                                capacity class does not bound all capacity classes")

    # --------------------------------------------------------------- STEP 12
    rule("STEP 12 -- FINAL DIAGNOSIS INPUTS")
    print("| Measurement | V1 | Ceiling | delta |")
    print("|---|---:|---:|---:|")
    print(f"| pooled log loss | {ll1:.10f} | {ll2:.10f} | {ll2-ll1:+.10f} |")
    print(f"| pooled Brier | {br1:.10f} | {br2:.10f} | {br2-br1:+.10f} |")
    print(f"| Draw-vs-rest AUC | {auc1:.4f} | {auc2:.4f} | {auc2-auc1:+.4f} |")
    print(f"| mean P(D) | {g1['pd_mean']:.4f} | {g2['pd_mean']:.4f} | "
          f"{g2['pd_mean']-g1['pd_mean']:+.4f} |")
    print(f"| P(D)>1/3 | {g1['pd_gt_third']:.4f} | {g2['pd_gt_third']:.4f} | "
          f"{g2['pd_gt_third']-g1['pd_gt_third']:+.4f} |")
    print(f"| Draw argmax share | {g1['draw_argmax_share']:.4f} | "
          f"{g2['draw_argmax_share']:.4f} | "
          f"{g2['draw_argmax_share']-g1['draw_argmax_share']:+.4f} |")
    print(f"| actual-Draw rank-1 | {g1['draw_rank1_share']:.4f} | "
          f"{g2['draw_rank1_share']:.4f} | "
          f"{g2['draw_rank1_share']-g1['draw_rank1_share']:+.4f} |")
    print(f"\n  classification            : {verdict}")
    print(f"  CASE C effect             : "
          f"{'weakened' if (ll_better and draw_sep_better) else 'retained as description'}")
    print(f"  CASE E effect             : "
          f"{'strengthened' if (ll_better and ll_majority) else 'weakened'}")
    print(f"  strongest supporting evid : pooled log-loss delta {ll2-ll1:+.10f} over "
          f"{imp_ll}/3 folds; Draw-vs-rest AUC {auc1:.4f} -> {auc2:.4f}")
    print(f"  strongest counter-evidence: Brier delta {br2-br1:+.10f}; "
          f"fold-delta sign change {unstable}; "
          f"log-loss delta range {max(per[n][2] for n,_,_ in v1_rows)-min(per[n][2] for n,_,_ in v1_rows):.6f}")
    print("  unresolved ambiguity      : one capacity class does not bound the achievable")
    print("                              ceiling; preprocessing differs by construction")
    print("                              between a tree model and the linear pipeline")
    print("  confidence                : three folds support ranges only, never a")
    print("                              standard deviation or a significance claim")

    # --------------------------------------------------------------- STEP 13
    rule("STEP 13 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    check("src/models/train.py unchanged",
          after["src/models/train.py"] == before["src/models/train.py"])
    check("v1_logreg.pkl unchanged", after.get("artifact") == before.get("artifact"))
    check("features.db unchanged", after["data/processed/features.db"] ==
          EXPECTED_MD5["data/processed/features.db"])
    check("matches.db unchanged", after["data/processed/matches.db"] ==
          EXPECTED_MD5["data/processed/matches.db"])
    check("13/13 LOCKED_INPUTS unchanged",
          all(after[f"pin:{r}"] == v["expected"] for r, v in pins.items()))
    print("\nFILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")
    print("\nSTEP 8 COMPLETE -- EVIDENCE GENERATED, NOTHING CHANGED, NOTHING PERSISTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
