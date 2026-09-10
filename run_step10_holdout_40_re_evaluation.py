"""STEP 10 -- 40-MATCH HOLDOUT RE-EVALUATION.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step10_holdout_40_re_evaluation.py

THIS IS NOT A BLIND TEST. It is a HOLDOUT RE-EVALUATION.
The 2025/26 holdout has already been used in this investigation, and the
40-match set REUSES the previous 20-match sample. It is descriptive evidence
about whether the recorded 9/20 behaviour persists in a larger slice of the
same holdout. It cannot establish generalisation to unseen future data and
cannot replace a future 2026/27 blind test.

MODEL FROZEN. The existing data/models/v1_logreg.pkl is loaded and used
as-is. Nothing is retrained, refitted, tuned, recalibrated, thresholded,
reweighted or engineered. `fit()` is never called on the evaluation path;
the only fits performed are the three walk-forward refits required by the
STEP 2 baseline-reproduction gate, which touch neither the artifact nor the
holdout.

SELECTION RULE -- RECOVERED FROM SOURCE, NOT FROM MEMORY
predict_blind_2025_26.py persists nothing ("No prediction is saved -- stdout
only"), so no record of the 20 fixture ids exists. What DOES exist, and is
recoverable exactly, is the deterministic rule in its source:

    universe : fixtures with season_id in FINAL_TEST_SEASONS and status='FT',
               ordered by (unix ASC, fixture_id ASC)
    selection: step = len(universe) / count
               idx  = sorted({int(i * step) for i in range(count)})

Re-applying that rule with count=20 reproduces the previous universe exactly.
The script does NOT trust that reproduction: it recomputes correctness and
HARD-FAILS unless the result is exactly 9/20, which is the only available
independent confirmation that the recovered universe is the right one.

WHY count=40 NESTS THE 20
For j = 2k, int(2k*L/40) == int(k*L/20) for every k, so the count=40 index
set contains the count=20 index set by construction. Overlap is therefore
exactly 20 and the additional block exactly 20 -- verified, not assumed. No
new selection rule is invented to reach 40.

BLINDNESS-OF-ARTIFACT CHECK
The artifact must not have been fitted on 2025/26, or the evaluation is
in-sample. The SimpleImputer stores the medians it was fitted on, so the
script recomputes those medians for both candidate training partitions and
identifies which one the artifact carries -- the same probe used earlier in
this investigation. It hard-fails if the artifact saw the holdout.
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
PREVIOUS_CORRECT = 9
PREVIOUS_N = 20
CURRENT_N = 40
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
#: The recovered selection rule, verified verbatim in the earlier script.
SELECTION_RULE_TOKENS = (
    "step = len(ordered) / args.count",
    "idx = sorted({int(i * step) for i in range(args.count)})",
    "ORDER BY unix ASC, fixture_id ASC",
    "status = 'FT'",
)
PREV_SCRIPT = "predict_blind_2025_26.py"


def rule(t): print("\n" + "=" * 124); print(t); print("=" * 124)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 124)
    print("ABORT -- STEP 10 halted. No re-evaluation is produced.")
    print(msg)
    print("!" * 124)
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


def prf(cm, i):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return float(p), float(r), float(f)


def draw_block(P, y, order):
    iH, iD, iA = 0, 1, 2
    pred = np.array(order)[P.argmax(axis=1)]
    d = y == "D"
    Pd = P[d]
    out = {
        "actual_draws": int(d.sum()),
        "pred_draws": int((pred == "D").sum()),
        "correct_draws": int(((pred == "D") & d).sum()),
        "draw_argmax_share": float((pred == "D").mean()),
        "mean_pd": float(P[:, iD].mean()),
        "median_pd": float(np.median(P[:, iD])),
        "sd_pd": float(P[:, iD].std()),
        "pd_gt_third": float((P[:, iD] > 1 / 3).mean()),
    }
    if len(Pd):
        out["rank1_share"] = float(((Pd[:, iD] > Pd[:, iH]) & (Pd[:, iD] > Pd[:, iA])).mean())
        out["d_gt_h"] = float((Pd[:, iD] > Pd[:, iH]).mean())
        out["d_gt_a"] = float((Pd[:, iD] > Pd[:, iA]).mean())
        out["d_gt_both"] = out["rank1_share"]
    else:
        out.update({k: float("nan") for k in ("rank1_share", "d_gt_h", "d_gt_a", "d_gt_both")})
    return out


def main():
    # ---------------------------------------------------------------- RULE 1
    rule("RULE 1 -- PRODUCTION CODE TRACE (source text, before any selection or prediction)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:74]}", tok in tsrc)

    try:
        import sklearn
    except ImportError as exc:
        stop(f"scikit-learn is required: {exc}. Do not substitute another estimator.")

    from models import artifact as artifact_mod
    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.candidate_contract import predict_candidate
    from models.config import (
        FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, MODEL_VERSION,
        RECOMMENDED_CONTEXT_FEATURE, REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS,
    )
    from models.data import load_supervised_dataset
    from models.evaluate import evaluate
    from models.splits import iter_walk_forward_folds
    from models.train import train_logistic_regression

    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("production contract == exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric production features == 79", len(numeric) == 79)
    check("competition_id is the only categorical production column",
          len(set(MODEL_B_COLUMNS) - set(numeric)) == 1)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")
    check("C == 1.0 in source", "C=1.0" in tsrc)
    check("random_state == 0 in source", "random_state=0" in tsrc)
    check("max_iter == 2000 in source", "max_iter=2000" in tsrc)

    # ---------------------------------------------------------------- STEP 1
    rule("STEP 1 -- INTEGRITY GATE")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"pandas {pd.__version__}  sklearn {sklearn.__version__}")
    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(rel, before[rel] == exp, before[rel])
    check("v1_logreg.pkl present and unchanged",
          before.get("artifact") == ARTIFACT_MD5, str(before.get("artifact")))
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    check("13/13 LOCKED_INPUTS unchanged",
          all(before[f"pin:{r}"] == v["expected"] for r, v in pins.items()))
    check("src/models/train.py at its pinned checksum",
          before["src/models/train.py"] == pins["src/models/train.py"]["expected"])

    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    check("forbidden season set matches the documented list",
          sorted(test_ids) == sorted(FORBIDDEN_SEASONS), str(sorted(test_ids)))

    # ---------------------------------------------------------------- STEP 2
    rule("STEP 2 -- V1 BASELINE REPRODUCTION")
    folds = list(iter_walk_forward_folds(ds))
    check("exactly three walk-forward folds", len(folds) == 3)
    lls = []
    for fold, tr, va in folds:
        _, _, P = train_logistic_regression(tr.X[list(MODEL_B_COLUMNS)], tr.y,
                                            va.X[list(MODEL_B_COLUMNS)])
        r = evaluate(va.y, P)
        lls.append(r.log_loss)
        print(f"  {fold.name}: N={len(va)}  log_loss={r.log_loss:.16f}  brier={r.brier:.16f}")
    ll_ref = float(np.mean(lls))
    print(f"\n  pooled mean log loss: {ll_ref:.16f}  (expected {POOLED_REF_LOG_LOSS:.16f})")
    check("V1 baseline reproduces exactly",
          abs(ll_ref - POOLED_REF_LOG_LOSS) <= 1e-9, f"diff={ll_ref - POOLED_REF_LOG_LOSS:.3e}")
    print("  These three refits exist ONLY to prove V1 is unchanged. They do not touch")
    print("  the artifact and do not touch the holdout.")

    # ---------------------------------------------------------------- STEP 3
    rule("STEP 3 -- RECOVER THE PREVIOUS 20-MATCH EVALUATION")
    prev = REPO / PREV_SCRIPT
    check(f"{PREV_SCRIPT} exists as a project artifact", prev.exists())
    psrc = prev.read_text(encoding="utf-8")
    for tok in SELECTION_RULE_TOKENS:
        check(f"selection rule token recovered from source: {tok[:60]}", tok in psrc)
    check("the previous script persisted no prediction record",
          "No prediction is saved" in psrc)
    print("\n  RECOVERED SELECTION RULE (verbatim from source, not from memory):")
    print("    universe  : season_id IN FINAL_TEST_SEASONS AND status = 'FT'")
    print("    ordering  : ORDER BY unix ASC, fixture_id ASC")
    print("    step      : len(universe) / count")
    print("    indices   : sorted({int(i * step) for i in range(count)})")
    print("  No prediction artifact was ever written, so the 20 fixture ids are")
    print("  reproduced by re-applying this deterministic rule -- and the reproduction")
    print("  is then CHECKED against the recorded 9/20 rather than assumed correct.")

    import sqlite3
    mdb = REPO / "data/processed/matches.db"
    con = sqlite3.connect(f"file:{mdb.resolve()}?mode=ro", uri=True)
    try:
        q = ",".join(str(i) for i in sorted(test_ids))
        rows = con.execute(
            f"SELECT fixture_id, home_name, away_name, unix, season_id, competition_id "
            f"FROM fixtures WHERE season_id IN ({q}) AND status = 'FT' "
            f"ORDER BY unix ASC, fixture_id ASC").fetchall()
    finally:
        con.close()
    ordered = [r[0] for r in rows]
    meta = {r[0]: r for r in rows}
    L = len(ordered)
    print(f"\n  holdout universe size (2025/26, status FT): {L}")

    def select(count):
        step = L / count
        return [ordered[i] for i in sorted({int(i * step) for i in range(count)})]

    sel20 = select(PREVIOUS_N)
    sel40 = select(CURRENT_N)
    check(f"count={PREVIOUS_N} yields exactly {PREVIOUS_N} unique ids",
          len(sel20) == len(set(sel20)) == PREVIOUS_N)
    check(f"count={CURRENT_N} yields exactly {CURRENT_N} unique ids",
          len(sel40) == len(set(sel40)) == CURRENT_N)
    check("the previous 20 are a subset of the 40 (same rule, nested by construction)",
          set(sel20) <= set(sel40))
    check("nesting identity int(2k*L/40) == int(k*L/20) holds for every k",
          all(int(2 * k * L / CURRENT_N) == int(k * L / PREVIOUS_N) for k in range(PREVIOUS_N)))

    # ------------------------------------------------- artifact + blindness probe
    art = artifact_mod.load(REPO / "data/models/v1_logreg.pkl")
    check("artifact contract == frozen 80 columns, in order",
          tuple(art.feature_columns) == tuple(MODEL_B_COLUMNS))
    check("artifact model_version == v1.0", art.model_version == "v1.0")
    check("artifact class_order == ['H','D','A']", art.class_order == ["H", "D", "A"])
    check("artifact estimator is the frozen configuration",
          art.model.C == 1.0 and art.model.max_iter == 2000 and art.model.random_state == 0,
          f"C={art.model.C} max_iter={art.model.max_iter} rs={art.model.random_state}")

    train_season_ids = set()
    for s in FINAL_TRAIN_SEASONS:
        train_season_ids.update(SEASON_NAME_TO_IDS[s])
    num_cols = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    stats = np.asarray(art.preprocessor._imputer.statistics_, dtype=float)
    mask_excl = ds.metadata["season_id"].isin(train_season_ids).values
    cand = {
        "EXCLUDES 2025/26 (blind to the holdout)":
            ds.X[mask_excl][num_cols].median(skipna=True).to_numpy(dtype=float),
        "INCLUDES 2025/26 (holdout was in training)":
            ds.X[num_cols].median(skipna=True).to_numpy(dtype=float),
    }
    dists = {k: float(np.nanmax(np.abs(stats - v))) for k, v in cand.items()}
    print("\n  artifact training-partition probe (imputer medians):")
    for k, v in dists.items():
        print(f"    max|diff| vs partition that {k:44s} = {v:.6e}")
    nearest = min(dists, key=dists.get)
    print(f"    identified partition: {nearest}")
    check("the artifact was NOT fitted on the 2025/26 holdout",
          nearest.startswith("EXCLUDES") and dists[nearest] < 1e-9,
          f"nearest-distance={dists[nearest]:.3e}")

    # ---------------------------------------------------------------- STEP 6
    rule("STEP 6 -- PRODUCTION CONTRACT AUDIT (all 40 matches)")
    idx = ds.metadata["fixture_id"].isin(sel40).values
    Xh = ds.X[idx][list(art.feature_columns)]
    meta_h = ds.metadata[idx].reset_index(drop=True)
    yh = ds.y[idx].to_numpy()
    check("all 40 selected fixtures have a v1.0 feature row", len(Xh) == CURRENT_N,
          f"{len(Xh)}/{CURRENT_N}")
    check("frame carries exactly the 80 production columns, in contract order",
          list(Xh.columns) == list(MODEL_B_COLUMNS))
    print(f"    production columns   : {Xh.shape[1]}")
    print(f"    additional columns   : {Xh.shape[1] - 80}")
    print(f"    closeness columns    : {sum(1 for c in Xh.columns if c.startswith('closeness_'))}")
    print(f"    new information srcs : 0")
    print(f"    model changes        : 0")
    check("no closeness column present",
          not any(c.startswith("closeness_") for c in Xh.columns))
    check("no label / post-match column present",
          not ({"label_result", "label_home_goals", "label_away_goals"} & set(Xh.columns)))

    # ---------------------------------------------------------------- STEP 5
    rule("STEP 5 -- 40-MATCH UNIVERSE AUDIT")
    add20 = [f for f in sel40 if f not in set(sel20)]
    print(f"  original 20 : {len(sel20)}   additional : {len(add20)}   "
          f"overlap : {len(set(sel20) & set(sel40))}   total : {len(sel40)}")
    check("overlap is exactly 20", len(set(sel20) & set(sel40)) == PREVIOUS_N)
    check("additional block is exactly 20", len(add20) == PREVIOUS_N)
    import datetime
    print("\n| # | match_id | date | home | away | season_id | competition_id | block |")
    print("|---:|---:|---|---|---|---:|---:|---|")
    for i, f in enumerate(sel40, 1):
        r = meta[f]
        d = datetime.datetime.utcfromtimestamp(r[3]).strftime("%Y-%m-%d")
        blk = "ORIGINAL-20" if f in set(sel20) else "ADDITIONAL-20"
        print(f"| {i} | {f} | {d} | {r[1][:22]} | {r[2][:22]} | {r[4]} | {r[5]} | {blk} |")
    check("all 40 belong to the 2025/26 holdout universe",
          set(meta_h["season_id"]) <= test_ids)
    check("no duplicate fixture ids", len(set(sel40)) == CURRENT_N)

    # ---------------------------------------------------------------- STEP 7
    rule("STEP 7 -- V1 PREDICTIONS (existing artifact only; fit() never called here)")
    out = predict_candidate(art.model, art.preprocessor, Xh, meta_h["fixture_id"].tolist())
    order = list(out.class_order)
    check("prediction class order == ['H','D','A']", order == ["H", "D", "A"])
    P = np.asarray(out.probabilities, dtype=float)
    pos = {f: i for i, f in enumerate(out.fixture_ids)}
    rowsP = np.array([P[pos[f]] for f in sel40])
    y40 = np.array([yh[list(meta_h["fixture_id"]).index(f)] for f in sel40])
    pred40 = np.array(order)[rowsP.argmax(axis=1)]
    lab = {"H": "HOME WIN", "D": "DRAW", "A": "AWAY WIN"}
    print("| # | match_id | date | home | away | P(H) | P(D) | P(A) | predicted | conf "
          "| actual | correct | block |")
    print("|---:|---:|---|---|---|---:|---:|---:|---|---:|---|---|---|")
    for i, f in enumerate(sel40):
        r = meta[f]
        d = datetime.datetime.utcfromtimestamp(r[3]).strftime("%Y-%m-%d")
        p = rowsP[i]
        blk = "ORIG" if f in set(sel20) else "ADD"
        print(f"| {i+1} | {f} | {d} | {r[1][:18]} | {r[2][:18]} | {p[0]:.4f} | {p[1]:.4f} "
              f"| {p[2]:.4f} | {lab[pred40[i]]} | {p.max():.4f} | {lab[y40[i]]} "
              f"| {'YES' if pred40[i]==y40[i] else 'no'} | {blk} |")

    # ---------------------------------------------------------------- STEP 3 verify
    rule("STEP 3 (verification) -- DOES THE RECOVERED 20 REPRODUCE THE RECORDED 9/20?")
    is_orig = np.array([f in set(sel20) for f in sel40])
    c20 = int((pred40[is_orig] == y40[is_orig]).sum())
    print(f"  recovered original-20 correct predictions: {c20}/{PREVIOUS_N}")
    print(f"  previously recorded                      : {PREVIOUS_CORRECT}/{PREVIOUS_N}")
    check("recovered 20-match universe reproduces the recorded result exactly",
          c20 == PREVIOUS_CORRECT,
          f"got {c20}, expected {PREVIOUS_CORRECT} -- the recovered universe is NOT the "
          f"previous one; per the FINAL RULE this aborts rather than substituting a guess")

    # ---------------------------------------------------------------- STEP 8
    rule("STEP 8 -- 40-MATCH RESULTS")
    res = evaluate(pd.Series(y40), rowsP)
    cm = res.confusion_matrix
    print(f"  correct predictions : {int((pred40 == y40).sum())}/{CURRENT_N}")
    print(f"  accuracy            : {res.accuracy:.4f}")
    print(f"  log loss            : {res.log_loss:.10f}")
    print(f"  multiclass Brier    : {res.brier:.10f}")
    print(f"  macro-F1            : {res.macro_f1:.4f}")
    print(f"  balanced accuracy   : {res.balanced_accuracy:.4f}")
    print(f"  average max probability: {rowsP.max(axis=1).mean():.4f}")
    print("\n| class | precision | recall | F1 |")
    print("|---|---:|---:|---:|")
    for i, c in enumerate(order):
        p, r_, f_ = prf(cm, i)
        print(f"| {c} | {p:.4f} | {r_:.4f} | {f_:.4f} |")
    db = draw_block(rowsP, y40, order)
    print("\n  DRAW-SPECIFIC")
    for k, v in db.items():
        print(f"    {k:22s} {v if isinstance(v,int) else f'{v:.4f}'}")
    print("\n  CONFUSION MATRIX (rows = predicted, cols = actual, H/D/A)")
    print("        Actual")
    print("        H    D    A")
    for i, c in enumerate(order):
        print(f"  Pred {c}  " + "  ".join(f"{int(cm[j, i]):3d}" for j in range(3)))
    print("\n  (models.evaluate returns rows=true, cols=pred; transposed above to match")
    print("   the requested Pred-by-Actual layout)")
    print("\n  calibration by confidence bucket: NOT REPORTED -- no established project")
    print("  convention exists for it, and 40 matches cannot populate buckets meaningfully.")

    # ---------------------------------------------------------------- STEP 9
    rule("STEP 9 -- 20 vs 40 COMPARISON (nested samples, NOT independent experiments)")
    add = ~is_orig
    res20 = evaluate(pd.Series(y40[is_orig]), rowsP[is_orig])
    resA = evaluate(pd.Series(y40[add]), rowsP[add])
    d20, dA = draw_block(rowsP[is_orig], y40[is_orig], order), draw_block(rowsP[add], y40[add], order)
    cA = int((pred40[add] == y40[add]).sum())
    c40 = int((pred40 == y40).sum())
    print(f"  correct in ORIGINAL 20   : {c20}/20   ({100*c20/20:.1f}%)")
    print(f"  correct in ADDITIONAL 20 : {cA}/20   ({100*cA/20:.1f}%)")
    print(f"  correct in COMBINED 40   : {c40}/40   ({100*c40/40:.1f}%)")
    print("\n| Metric | Previous 20 | Additional 20 | Combined 40 |")
    print("|---|---:|---:|---:|")
    print(f"| Correct | {c20} | {cA} | {c40} |")
    print(f"| Accuracy | {res20.accuracy:.4f} | {resA.accuracy:.4f} | {res.accuracy:.4f} |")
    print(f"| Log loss | {res20.log_loss:.6f} | {resA.log_loss:.6f} | {res.log_loss:.6f} |")
    print(f"| Brier | {res20.brier:.6f} | {resA.brier:.6f} | {res.brier:.6f} |")
    print(f"| Actual Draws | {d20['actual_draws']} | {dA['actual_draws']} | {db['actual_draws']} |")
    print(f"| Predicted Draws | {d20['pred_draws']} | {dA['pred_draws']} | {db['pred_draws']} |")
    print(f"| Correct Draws | {d20['correct_draws']} | {dA['correct_draws']} | {db['correct_draws']} |")
    for i, c in enumerate(order):
        if c != "D":
            continue
        p20, r20, f20 = prf(res20.confusion_matrix, i)
        pA, rA, fA = prf(resA.confusion_matrix, i)
        p40, r40, f40 = prf(cm, i)
        print(f"| Draw precision | {p20:.4f} | {pA:.4f} | {p40:.4f} |")
        print(f"| Draw recall | {r20:.4f} | {rA:.4f} | {r40:.4f} |")
        print(f"| Draw F1 | {f20:.4f} | {fA:.4f} | {f40:.4f} |")
    print("\n  The 40 CONTAINS the 20. The three columns are not independent samples, and")
    print("  the combined figure is not a replication of the original.")

    # --------------------------------------------------------------- STEP 10
    rule("STEP 10 -- PROBABILITY QUALITY vs HISTORICAL WALK-FORWARD REFERENCE")
    print("| Measurement | 40-match holdout | historical walk-forward V1 | delta |")
    print("|---|---:|---:|---:|")
    print(f"| log loss | {res.log_loss:.10f} | {POOLED_REF_LOG_LOSS:.10f} | "
          f"{res.log_loss - POOLED_REF_LOG_LOSS:+.10f} |")
    print(f"| Brier | {res.brier:.10f} | {POOLED_REF_BRIER:.10f} | "
          f"{res.brier - POOLED_REF_BRIER:+.10f} |")
    print("\n  Priority order for reading this: 1 log loss, 2 Brier, 3 accuracy,")
    print("  4 Draw-specific. More predicted Draws with worse probability scores is NOT")
    print("  an improvement. The historical reference is a 5,331-row walk-forward mean;")
    print("  40 matches cannot resolve a difference of the size seen across these runs.")
    print("  No significance test is performed -- no project convention defines one.")

    # --------------------------------------------------------------- STEP 11
    rule("STEP 11 -- HONEST INTERPRETATION (mandatory statements)")
    print("  This is NOT an independent blind test.")
    print("  The 40-match set REUSES the previous 20-match holdout sample.")
    print("  Therefore it:")
    print("    - is useful descriptive evidence")
    print("    - can show whether the previous 9/20 behaviour persists")
    print("    - CANNOT establish generalisation to unseen future data")
    print("    - CANNOT replace the future 2026/27 blind test")
    print("    - MUST NOT be used as proof of model improvement")
    print("  No significance claim is made from 40 matches, and the model is not")
    print("  'validated' by this test.")

    # --------------------------------------------------------------- STEP 12
    rule("STEP 12 -- FINAL CLASSIFICATION (descriptive; no invented threshold)")
    better = (res.log_loss < POOLED_REF_LOG_LOSS) and (res.brier < POOLED_REF_BRIER)
    worse = (res.log_loss > POOLED_REF_LOG_LOSS) and (res.brier > POOLED_REF_BRIER)
    acc_gap = abs(res.accuracy - c20 / PREVIOUS_N)
    print(f"  40-match log loss vs reference : {res.log_loss - POOLED_REF_LOG_LOSS:+.10f}")
    print(f"  40-match Brier vs reference    : {res.brier - POOLED_REF_BRIER:+.10f}")
    print(f"  original-20 accuracy {c20/PREVIOUS_N:.4f} vs combined-40 {res.accuracy:.4f}   "
          f"|gap| {acc_gap:.4f}")
    if better and not worse:
        verdict = "B) BETTER DESCRIPTIVE PERFORMANCE"
    elif worse and not better:
        verdict = "C) WORSE DESCRIPTIVE PERFORMANCE"
    elif not better and not worse:
        verdict = "D) INCONCLUSIVE -- log loss and Brier disagree in direction"
    else:
        verdict = "A) CONSISTENT WITH PREVIOUS 20-MATCH RESULT"
    print(f"\n  CLASSIFICATION: {verdict}")
    print("  Descriptive only. No production change is recommended. The model is unchanged.")

    # --------------------------------------------------------------- STEP 13
    rule("STEP 13 -- FINAL INTEGRITY AUDIT")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    print(f"\n  MODEL")
    print(f"    v1_logreg.pkl hash : {after.get('artifact')}")
    print(f"    MODEL_VERSION      : {MODEL_VERSION}")
    print(f"    contract           : {len(MODEL_B_COLUMNS)} columns "
          f"({len(numeric)} numeric + competition_id)")
    print(f"  DATA")
    print(f"    features.db        : {after['data/processed/features.db']}")
    print(f"    matches.db         : {after['data/processed/matches.db']}")
    print(f"    40 match ids       : {sel40}")
    print(f"  TRAINING")
    print(f"    retrained          : NO")
    print(f"    fit() on holdout   : NO (artifact loaded; only STEP 2 baseline refits)")
    print(f"    tuning             : NO")
    print(f"  FEATURES")
    print(f"    contract modified  : NO")
    print(f"    closeness features : NO")
    print(f"    new information    : NO")
    print(f"    HGB                : NO")
    print(f"  FILES")
    print(f"    files created      : NONE")
    print(f"    files modified     : NONE")
    print(f"    databases modified : NONE")
    print("\nSTEP 10 COMPLETE -- HOLDOUT RE-EVALUATION, MODEL FROZEN, NOTHING CHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
