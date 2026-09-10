"""FINAL HOLDOUT EVALUATION -- Two-Poisson + Venue, one-time 2025/26 test.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_final_holdout_poisson_venue.py

Trains the FIXED champion on 2020/21-2024/25 and evaluates it ONCE on 2025/26.
No feature is added, no hyperparameter is tuned, no calibration is applied, and
no model choice is revisited. If the result is acceptable under the criteria
pre-registered below, a NEW artifact is written; v1_logreg.pkl is never
overwritten.

------------------------------------------------------------------------------
AN ACCURACY CORRECTION ABOUT THE WORD "HOLDOUT" -- READ BEFORE THE RESULT
------------------------------------------------------------------------------
2025/26 is NOT a pristine, never-observed holdout for this project. Its
outcomes have already been read twice in this investigation:

    * a 20-match blind evaluation of frozen V1 (recorded ~9/20)
    * a 40-match holdout re-evaluation of frozen V1 (19/40 = 47.5%), which
      printed the full per-fixture table including actual results

What IS true, and it is the thing that matters here:

    NO selection decision in the Poisson chain used 2025/26. The model form
    (STEP 3), the venue-feature choice (STEP 4/5) and the decision rules were
    all fixed on the three walk-forward folds alone. For THIS model, 2025/26
    is genuinely unseen data.

So this run is an honest first evaluation of the champion, but it is not a
pristine holdout in the strict sense, and the report must not claim otherwise.
Two consequences are carried through to the end of this script:

    1. the season is fully spent after this run, for every purpose;
    2. a residual optimism of unknown size exists because the analyst has seen
       V1's behaviour on this season. It cannot be quantified, so it is
       disclosed rather than estimated away.

------------------------------------------------------------------------------
THE OUTCOME BARRIER -- WHY THIS SCRIPT IS ORDERED THE WAY IT IS
------------------------------------------------------------------------------
Every technical gate runs on FEATURES AND PREDICTIONS ONLY, before a single
2025/26 outcome is loaded. Lambdas, probabilities, row sums, tail mass and the
complement control are all verifiable without knowing who won.

That ordering implements the standing rule directly: if something fails
technically, the script aborts BEFORE any outcome is read, so the
implementation can be fixed and rerun with the holdout still intact. Only
after every gate passes does `_load_holdout_outcomes()` execute, once, behind
a printed barrier.
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import importlib.util
import json
import pickle
import shutil
import sqlite3
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------- model spec
MODEL_NAME = "poisson_venue"
MODEL_VERSION_V2 = "v2.0-poisson-venue"
POISSON_ALPHA = 1.0
POISSON_MAX_ITER = 2000
VENUE_VALUE_COLUMNS = (
    "home_goals_for_home_venue_season",
    "home_goals_against_home_venue_season",
    "away_goals_for_away_venue_season",
    "away_goals_against_away_venue_season",
)
VENUE_N_COLUMNS = tuple(c + "_n" for c in VENUE_VALUE_COLUMNS)

# ------------------------------------------------- walk-forward research refs
WF_CHAMP = {"accuracy": 0.5281, "log loss": 0.992603, "Brier": 0.591782}
WF_POISSON_BASE = {"accuracy": 0.5272, "log loss": 0.992823, "Brier": 0.591968}
WF_V1 = {"accuracy": 0.5184, "log loss": 0.999379, "Brier": 0.596502}

# =============================================================================
# PRE-REGISTERED ACCEPTANCE CRITERIA -- fixed here, in source, BEFORE the
# outcome barrier. Not re-openable after the result is seen.
#
# The two reference points are principled rather than arbitrary:
#   * PRIOR baseline: constant probabilities equal to the TRAINING class
#     frequencies. Beating it is the minimum meaning of "the model knows
#     something". Its parameters come from training only.
#   * CONSTANT-LAMBDA baseline: Poisson at TRAINING league mean goals. Beating
#     it shows the 84 features contribute beyond league-level scoring rates.
#
# CATASTROPHIC_LL_RATIO is a judgement call and is labelled as one. It is set
# before any holdout number exists precisely so it cannot be fitted to the
# answer. A season-to-season degradation beyond 15% in log loss is treated as
# a failure of generalisation rather than noise.
# =============================================================================
CATASTROPHIC_LL_RATIO = 1.15
ACCEPTANCE_CRITERIA = (
    "A. every technical gate passes (lambdas, probabilities, tail mass, sums)",
    "B. holdout log loss  <  training-prior baseline log loss  [knows something]",
    "C. holdout Brier     <  training-prior baseline Brier",
    "D. holdout log loss  <  constant-lambda baseline log loss [features help]",
    "E. holdout log loss  <= walk-forward log loss * 1.15       [no catastrophe]",
    "F. holdout accuracy  >= training-prior baseline accuracy",
)

# ---------------------------------------------------------------- pinned inputs
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_V1_MD5 = "5e504427712b35778bb8a62a8496c7cd"
TRAIN_PY_MD5 = "21425459195311492f49e73f5ae38fe0"
STEP3_SCRIPT = "run_step_poisson_3_first_experiment.py"
STEP3_MD5 = "24827e92bc0acb04b9fc1dbe5a9958ae"
OUT_ARTIFACT = Path("data/models/v2_poisson_venue.pkl")

MASS_TOL, COMPLEMENT_TOL, ROWSUM_TOL = 1e-12, 1e-10, 1e-9

PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)


def rule(t): print("\n" + "=" * 128); print(t); print("=" * 128)
def sub(t): print("\n" + "-" * 128); print(t); print("-" * 128)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def abort(msg):
    print("\n" + "!" * 128)
    print("FINAL HOLDOUT ABORTED -- no result, no artifact")
    print(msg)
    print("!" * 128)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        abort(f"stop condition: {label}")


def snapshot():
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    for rel in ("src/models/train.py", "src/models/config.py",
                "src/features/feature_builder.py", "src/features/rolling.py",
                "src/features/history.py", "src/models/artifact.py", STEP3_SCRIPT):
        snap[rel] = md5(REPO / rel)
    v1 = REPO / "data/models/v1_logreg.pkl"
    if v1.exists():
        snap["data/models/v1_logreg.pkl"] = md5(v1)
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def draw_stats(P, y):
    return {
        "Draw argmax": float((P.argmax(axis=1) == 1).mean()),
        "mean P(D)": float(P[:, 1].mean()),
        "SD P(D)": float(P[:, 1].std()),
        "P(D)>1/3": float((P[:, 1] > 1 / 3).mean()),
        "actual draw rate": float((y == "D").mean()),
    }


def main():
    # ======================================================== PHASE A: INTEGRITY
    rule("PHASE A / STEP 1 -- SOURCE AND INTEGRITY AUDIT")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:78]}", tok in tsrc)

    try:
        import sklearn
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import PoissonRegressor
        from sklearn.metrics import mean_poisson_deviance, roc_auc_score
    except ImportError as exc:
        abort(f"scikit-learn is required: {exc}. No estimator is substituted.")

    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, RECOMMENDED_CONTEXT_FEATURE,
        REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS, X_EXCLUDED_COLUMNS,
    )
    from models.data import load_supervised_dataset
    from models.evaluate import evaluate
    from models.train import LogisticRegressionPreprocessor

    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"pandas {pd.__version__}  sklearn {sklearn.__version__}")
    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(f"(1) {rel} unchanged", before[rel] == exp, before[rel])
    check("(1) src/models/train.py unchanged", before["src/models/train.py"] == TRAIN_PY_MD5)
    check("(1) v1_logreg.pkl unchanged",
          before.get("data/models/v1_logreg.pkl") == ARTIFACT_V1_MD5)
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if before[f"pin:{r}"] != v["expected"]]
    check(f"(1) {len(pins)}/{len(pins)} LOCKED_INPUTS unchanged", not bad, str(bad))

    # reuse the pinned, already-verified conversion
    check(f"STEP 3 conversion source at its pinned checksum",
          before[STEP3_SCRIPT] == STEP3_MD5, before[STEP3_SCRIPT])
    spec = importlib.util.spec_from_file_location("_step3", REPO / STEP3_SCRIPT)
    step3 = importlib.util.module_from_spec(spec)
    sys.modules["_step3"] = step3
    spec.loader.exec_module(step3)
    hda_tail_safe, modal_scoreline = step3.hda_tail_safe, step3.modal_scoreline
    check("tail-safe conversion imported (not reimplemented)", callable(hda_tail_safe))
    check("STEP 3 tail tolerance unchanged", step3.TAIL_TOL == 1e-15)

    fdb = REPO / "data/processed/features.db"
    con = sqlite3.connect(f"file:{fdb.resolve()}?mode=ro", uri=True)
    try:
        all_cols = [r[1] for r in con.execute("PRAGMA table_info(feature_rows)")]
    finally:
        con.close()
    for c in VENUE_VALUE_COLUMNS:
        check(f"(2) venue column exists: {c}", c in all_cols)
    check("(2) venue columns are not in the frozen 80-column contract",
          not (set(VENUE_VALUE_COLUMNS) & set(MODEL_B_COLUMNS)))
    check("(2) venue columns are not label/excluded columns",
          not (set(VENUE_VALUE_COLUMNS) & set(X_EXCLUDED_COLUMNS)))

    rl = (REPO / "src/features/rolling.py").read_text(encoding="utf-8")
    fb = (REPO / "src/features/feature_builder.py").read_text(encoding="utf-8")
    for tok in ('season_matches = [m for m in history if m["season_id"] == target_season_id]',
                'venue_matches = [m for m in season_matches if m["is_home"] == home_only]',
                "value = mean if coverage_n >= VENUE_MIN_COVERAGE else None"):
        check(f"(3) venue built from prior fixtures only: {tok[:56]}", tok in rl)

    tree = ast.parse(fb)
    walker = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "_walk_and_build"), None)
    loop = next((n for n in ast.walk(walker) if isinstance(n, ast.For)), None)
    bl = rc = None
    for node in ast.walk(loop):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "build_feature_row":
                bl = node.lineno
            if isinstance(fn, ast.Attribute) and fn.attr == "record":
                rc = node.lineno
    check("(4) build_feature_row executes before ctx.record", bl is not None
          and rc is not None and bl < rc, f"build@{bl} < record@{rc}")
    check("(4) exactly one ctx.record call site",
          sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "record") == 1)

    vfun = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "_venue_features"), None)
    vargs = [a.arg for a in vfun.args.args]
    check("(5) _venue_features takes no fixture / no goals argument",
          "fixture" not in vargs and not any("goal" in a for a in vargs), str(vargs))
    vsrc = ast.get_source_segment(fb, vfun) or ""
    for banned in ("fixture", "label_", "home_goals", "away_goals"):
        check(f"(5) _venue_features body never references `{banned}`", banned not in vsrc)

    FEATURES = list(MODEL_B_COLUMNS) + list(VENUE_VALUE_COLUMNS)
    check("(6) the four *_n venue columns are NOT in the model input set",
          not (set(VENUE_N_COLUMNS) & set(FEATURES)))
    check("input contract is exactly 84 columns", len(FEATURES) == 84, str(len(FEATURES)))
    check("input contract == frozen 80 + exactly the 4 venue columns",
          set(FEATURES) - set(MODEL_B_COLUMNS) == set(VENUE_VALUE_COLUMNS))
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])

    train_ids, test_ids = set(), set()
    for s in FINAL_TRAIN_SEASONS:
        train_ids.update(SEASON_NAME_TO_IDS[s])
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    check("(9) training seasons are exactly 2020/21-2024/25",
          tuple(FINAL_TRAIN_SEASONS) == ("2020/2021", "2021/2022", "2022/2023",
                                         "2023/2024", "2024/2025"),
          str(FINAL_TRAIN_SEASONS))
    check("(10) holdout season is exactly 2025/26",
          tuple(FINAL_TEST_SEASONS) == ("2025/2026",), str(FINAL_TEST_SEASONS))
    check("(7) training and holdout season ids are disjoint", not (train_ids & test_ids))

    ds = load_supervised_dataset(fdb)
    m_tr = ds.metadata["season_id"].isin(train_ids).values
    m_ho = ds.metadata["season_id"].isin(test_ids).values
    check("(7) every labelled row is either training or holdout, never both",
          not np.any(m_tr & m_ho) and bool(np.all(m_tr | m_ho)))
    fid_tr = set(ds.metadata["fixture_id"][m_tr])
    fid_ho = set(ds.metadata["fixture_id"][m_ho])
    check("(8) training and holdout fixture ids are disjoint", not (fid_tr & fid_ho))
    print(f"\n  training fixtures : {len(fid_tr)}")
    print(f"  holdout  fixtures : {len(fid_ho)}")

    sub("HONEST STATUS OF THIS HOLDOUT -- printed before any result exists")
    print("  2025/26 outcomes have previously been read in this project, for the")
    print("  frozen V1 model only (a 20-match sample and a 40-match sample).")
    print("  NO selection decision in the Poisson chain used this season: the model")
    print("  form and the venue features were chosen on the three walk-forward folds")
    print("  alone. For THIS champion the season is genuinely unseen, so this is an")
    print("  honest first evaluation -- but it is NOT a pristine holdout, and the")
    print("  report will not claim it is. After this run the season is fully spent.")

    sub("PRE-REGISTERED ACCEPTANCE CRITERIA (fixed in source, before any outcome)")
    for c in ACCEPTANCE_CRITERIA:
        print(f"    {c}")
    print(f"\n  CATASTROPHIC_LL_RATIO = {CATASTROPHIC_LL_RATIO} -- a judgement call,")
    print("  labelled as one, set before any holdout number exists so it cannot be")
    print("  fitted to the answer.")

    # ================================================ PHASE B: FINAL TRAINING
    rule("PHASE B / STEP 2 -- FINAL TRAINING ON 2020/21-2024/25 ONLY")
    Xtr = ds.X[m_tr][FEATURES]
    Xho = ds.X[m_ho][FEATURES]
    meta_ho = ds.metadata[m_ho].reset_index(drop=True)
    check("training frame carries exactly the 84 columns, in order",
          list(Xtr.columns) == FEATURES)
    check("holdout frame carries exactly the 84 columns, in order",
          list(Xho.columns) == FEATURES)
    check("no 2025/26 row is in the training frame",
          not set(ds.metadata["season_id"][m_tr]) & test_ids)

    con = sqlite3.connect(f"file:{fdb.resolve()}?mode=ro", uri=True)
    try:
        gq = ",".join(str(i) for i in sorted(train_ids))
        gtr = pd.read_sql_query(
            f"SELECT fixture_id, label_home_goals, label_away_goals FROM feature_rows "
            f"WHERE season_id IN ({gq}) AND label_result IS NOT NULL", con)
    finally:
        con.close()
    gh = dict(zip(gtr["fixture_id"], gtr["label_home_goals"]))
    ga = dict(zip(gtr["fixture_id"], gtr["label_away_goals"]))
    fids_tr = ds.metadata["fixture_id"][m_tr].to_numpy()
    htr = np.array([gh[f] for f in fids_tr], dtype=float)
    atr = np.array([ga[f] for f in fids_tr], dtype=float)
    check("training goal targets loaded for every training row",
          len(htr) == len(Xtr) and len(atr) == len(Xtr))
    check("training goal targets contain no 2025/26 fixture",
          not (set(gtr["fixture_id"]) & fid_ho))
    print(f"  training goals: mean home {htr.mean():.6f}  mean away {atr.mean():.6f}")

    prep = LogisticRegressionPreprocessor()
    prep.fit(Xtr)
    check("preprocessing fitted on training data only "
          f"({len(prep._imputer.statistics_)} numeric columns)",
          len(prep._imputer.statistics_) == len(FEATURES) - 1,
          f"{len(prep._imputer.statistics_)} vs {len(FEATURES) - 1}")
    Etr, Eho = prep.transform(Xtr), prep.transform(Xho)
    check("encoded training/holdout widths match", Etr.shape[1] == Eho.shape[1],
          f"{Etr.shape} vs {Eho.shape}")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        model_h = PoissonRegressor(alpha=POISSON_ALPHA,
                                   max_iter=POISSON_MAX_ITER).fit(Etr, htr)
        model_a = PoissonRegressor(alpha=POISSON_ALPHA,
                                   max_iter=POISSON_MAX_ITER).fit(Etr, atr)
        conv = [x for x in w if issubclass(x.category, ConvergenceWarning)]
    print(f"  PoissonRegressor n_iter_: home={getattr(model_h,'n_iter_','?')}  "
          f"away={getattr(model_a,'n_iter_','?')}")
    check("both Poisson models converged", not conv,
          "; ".join(str(x.message)[:90] for x in conv))

    # ============================== PHASE C: PREDICT -- NO OUTCOMES READ YET
    rule("PHASE C / STEP 3 -- PREDICTION AND TECHNICAL GATES (NO OUTCOME IS READ)")
    print("  Everything in this phase is verifiable without knowing any result.")
    print("  If any gate fails the script aborts here, leaving the holdout intact.")
    lam_h, lam_a = model_h.predict(Eho), model_a.predict(Eho)
    check("lambdas finite and strictly positive",
          bool(np.all(np.isfinite(lam_h)) and np.all(np.isfinite(lam_a))
               and np.all(lam_h > 0) and np.all(lam_a > 0)),
          f"home[{lam_h.min():.4f},{lam_h.max():.4f}] away[{lam_a.min():.4f},{lam_a.max():.4f}]")
    P, K, resid, ctl = hda_tail_safe(lam_h, lam_a)
    print(f"  adaptive scoreline grid K = 0..{K}")
    check(f"scoreline residual mass < {MASS_TOL:.0e}", resid < MASS_TOL, f"{resid:.3e}")
    check(f"P(A) complement vs direct < {COMPLEMENT_TOL:.0e}", ctl < COMPLEMENT_TOL,
          f"{ctl:.3e}")
    check("every probability row sums to 1",
          bool(np.all(np.abs(P.sum(axis=1) - 1.0) < ROWSUM_TOL)),
          f"max|sum-1|={float(np.max(np.abs(P.sum(axis=1)-1.0))):.3e}")
    check("all probabilities finite", bool(np.all(np.isfinite(P))))
    check("all probabilities in [0,1]", bool(np.all(P >= 0) and np.all(P <= 1)))
    mi, mj, mp = modal_scoreline(lam_h, lam_a, K)
    check("modal scoreline produced for every fixture", len(mi) == len(P))
    print(f"  predictions produced : {len(P)}")
    print(f"  mean lambda_home {lam_h.mean():.6f}   mean lambda_away {lam_a.mean():.6f}")
    print(f"  mean P(H) {P[:,0].mean():.6f}  P(D) {P[:,1].mean():.6f}  "
          f"P(A) {P[:,2].mean():.6f}")

    cov = Xho[list(VENUE_VALUE_COLUMNS)].notna().all(axis=1).mean()
    print(f"  holdout venue coverage (all four present): {cov:.4f}")
    print(f"  prediction coverage: {len(P)}/{len(fid_ho)} holdout fixtures "
          f"({len(P) / len(fid_ho):.4f})")
    check("a prediction exists for every labelled holdout fixture", len(P) == len(fid_ho))

    # ---- baselines, built from TRAINING ONLY (still no outcome read) --------
    prior = np.array([(ds.y[m_tr].to_numpy() == c).mean() for c in ["H", "D", "A"]])
    P_prior = np.tile(prior, (len(P), 1))
    check("training-prior baseline sums to 1", abs(P_prior[0].sum() - 1) < 1e-12)
    P_const, Kc, rc_, cc_ = hda_tail_safe(np.full(len(P), htr.mean()),
                                          np.full(len(P), atr.mean()))
    check("constant-lambda baseline tail-safe", rc_ < MASS_TOL and cc_ < COMPLEMENT_TOL)
    print(f"  training class prior H/D/A : {prior.round(6)}")

    # ================================================= THE OUTCOME BARRIER
    rule(">>> OUTCOME BARRIER -- 2025/26 RESULTS ARE READ NOW, EXACTLY ONCE <<<")
    print("  All technical gates above passed without any outcome. Reading results.")

    def _load_holdout_outcomes():
        con = sqlite3.connect(f"file:{fdb.resolve()}?mode=ro", uri=True)
        try:
            q = ",".join(str(i) for i in sorted(test_ids))
            g = pd.read_sql_query(
                f"SELECT fixture_id, label_home_goals, label_away_goals FROM feature_rows "
                f"WHERE season_id IN ({q}) AND label_result IS NOT NULL", con)
        finally:
            con.close()
        mh = dict(zip(g["fixture_id"], g["label_home_goals"]))
        ma = dict(zip(g["fixture_id"], g["label_away_goals"]))
        f = meta_ho["fixture_id"].to_numpy()
        return (np.array([mh[x] for x in f], dtype=float),
                np.array([ma[x] for x in f], dtype=float))

    h_true, a_true = _load_holdout_outcomes()
    y = ds.y[m_ho].to_numpy()
    derived = np.where(h_true > a_true, "H", np.where(h_true < a_true, "A", "D"))
    check("holdout goals agree with label_result on every fixture",
          np.array_equal(derived, y))

    # ================================================ PHASE D: THE EVALUATION
    rule("PHASE D / STEP 4 -- ONE-TIME 2025/26 EVALUATION")
    ys = pd.Series(y)
    r = evaluate(ys, P)
    dstat = draw_stats(P, y)
    auc = float(roc_auc_score((y == "D").astype(int), P[:, 1]))
    print("  PRIMARY")
    print(f"    fixtures evaluated       : {len(y)}")
    print(f"    accuracy                 : {r.accuracy:.10f}   ({100 * r.accuracy:.2f}%)")
    print(f"    multiclass log loss      : {r.log_loss:.10f}")
    print(f"    Brier score              : {r.brier:.10f}")
    print("  SECONDARY")
    print(f"    Draw-vs-rest AUC         : {auc:.10f}")
    print(f"    actual draw rate         : {dstat['actual draw rate']:.10f}")
    print(f"    mean P(D)                : {dstat['mean P(D)']:.10f}")
    print(f"    SD P(D)                  : {dstat['SD P(D)']:.10f}")
    print(f"    Draw argmax frequency    : {dstat['Draw argmax']:.10f}")
    print(f"    P(D) > 1/3 frequency     : {dstat['P(D)>1/3']:.10f}")
    print("  GOAL MODEL DIAGNOSTICS")
    print(f"    home-goal MAE            : {float(np.abs(lam_h - h_true).mean()):.10f}")
    print(f"    away-goal MAE            : {float(np.abs(lam_a - a_true).mean()):.10f}")
    print(f"    home Poisson deviance    : {float(mean_poisson_deviance(h_true, lam_h)):.10f}")
    print(f"    away Poisson deviance    : {float(mean_poisson_deviance(a_true, lam_a)):.10f}")
    print(f"    mean predicted home goals: {float(lam_h.mean()):.10f}   "
          f"actual {float(h_true.mean()):.10f}")
    print(f"    mean predicted away goals: {float(lam_a.mean()):.10f}   "
          f"actual {float(a_true.mean()):.10f}")

    pred = np.array(["H", "D", "A"])[P.argmax(axis=1)]
    print("\n  confusion matrix (rows = predicted, cols = actual)")
    print("            H      D      A")
    for i, pc in enumerate(["H", "D", "A"]):
        print(f"    Pred {pc}  " + "  ".join(
            f"{int(((pred == pc) & (y == ac)).sum()):5d}" for ac in ["H", "D", "A"]))

    sub("SAMPLE OF THE PRODUCTION OUTPUT (first 10 holdout fixtures, descriptive)")
    print("| fixture_id | E[home] | E[away] | P(H) | P(D) | P(A) | top score | P(score) | actual |")
    print("|---:|---:|---:|---:|---:|---:|---|---:|---|")
    for i in range(min(10, len(P))):
        print(f"| {meta_ho['fixture_id'][i]} | {lam_h[i]:.3f} | {lam_a[i]:.3f} "
              f"| {P[i,0]:.4f} | {P[i,1]:.4f} | {P[i,2]:.4f} | {mi[i]}-{mj[i]} "
              f"| {mp[i]:.4f} | {int(h_true[i])}-{int(a_true[i])} |")

    # ============================================== PHASE E: COMPARISONS
    rule("PHASE E / STEP 5 -- CONTEXT (research vs holdout are DIFFERENT things)")
    r_prior, r_const = evaluate(ys, P_prior), evaluate(ys, P_const)
    print("  A. BASELINES ON THIS SAME HOLDOUT (parameters from TRAINING only)")
    print("| baseline | accuracy | log loss | Brier |")
    print("|---|---:|---:|---:|")
    print(f"| training class prior | {r_prior.accuracy:.6f} | {r_prior.log_loss:.6f} "
          f"| {r_prior.brier:.6f} |")
    print(f"| constant-lambda Poisson | {r_const.accuracy:.6f} | {r_const.log_loss:.6f} "
          f"| {r_const.brier:.6f} |")
    print(f"| **Poisson + Venue** | **{r.accuracy:.6f}** | **{r.log_loss:.6f}** "
          f"| **{r.brier:.6f}** |")

    print("\n  B. WALK-FORWARD RESEARCH RESULTS (2020/21-2024/25) -- NOT a target")
    print("| model | accuracy | log loss | Brier |")
    print("|---|---:|---:|---:|")
    for nm, d in (("V1 logistic regression", WF_V1),
                  ("Poisson + 80 columns", WF_POISSON_BASE),
                  ("Poisson + Venue (champion)", WF_CHAMP)):
        print(f"| {nm} | {d['accuracy']:.6f} | {d['log loss']:.6f} | {d['Brier']:.6f} |")
    print(f"\n  holdout minus walk-forward, champion:")
    print(f"    accuracy  {r.accuracy - WF_CHAMP['accuracy']:+.6f}  "
          f"({100 * (r.accuracy - WF_CHAMP['accuracy']):+.2f} percentage points)")
    print(f"    log loss  {r.log_loss - WF_CHAMP['log loss']:+.6f}   "
          f"ratio {r.log_loss / WF_CHAMP['log loss']:.6f}")
    print(f"    Brier     {r.brier - WF_CHAMP['Brier']:+.6f}")
    print("\n  These are DIFFERENT QUANTITIES. Walk-forward averages three")
    print("  validation seasons with progressively more training data; the holdout")
    print("  is one unseen season with the full training set. A gap in either")
    print("  direction is expected and is not by itself evidence of a problem.")
    print("  Beating 52.81% on 2025/26 was never a requirement.")

    # ============================================ PHASE F: ACCEPTANCE DECISION
    rule("PHASE F / STEP 6 -- ACCEPTANCE DECISION (pre-registered criteria only)")
    A = True
    B = r.log_loss < r_prior.log_loss
    C = r.brier < r_prior.brier
    D = r.log_loss < r_const.log_loss
    E = r.log_loss <= WF_CHAMP["log loss"] * CATASTROPHIC_LL_RATIO
    F = r.accuracy >= r_prior.accuracy
    for lab, ok, detail in (
        ("A technical validity", A, "all gates in phases A-C passed"),
        ("B log loss < prior baseline", B, f"{r.log_loss:.6f} vs {r_prior.log_loss:.6f}"),
        ("C Brier < prior baseline", C, f"{r.brier:.6f} vs {r_prior.brier:.6f}"),
        ("D log loss < constant-lambda", D, f"{r.log_loss:.6f} vs {r_const.log_loss:.6f}"),
        ("E no catastrophic degradation", E,
         f"{r.log_loss:.6f} <= {WF_CHAMP['log loss'] * CATASTROPHIC_LL_RATIO:.6f}"),
        ("F accuracy >= prior baseline", F, f"{r.accuracy:.6f} vs {r_prior.accuracy:.6f}"),
    ):
        print(f"  {'PASS' if ok else 'FAIL'}  {lab:34s} {detail}")
    all_ok = all((A, B, C, D, E, F))
    hard_fail = (not B) or (not D) or (not E)
    decision = ("ACCEPTABLE FOR PRODUCTION" if all_ok else
                ("NOT ACCEPTABLE FOR PRODUCTION" if hard_fail else "INCONCLUSIVE"))
    print(f"\n  DECISION: {decision}")

    acc_up = r.accuracy > WF_CHAMP["accuracy"]
    prob_worse = r.log_loss > WF_CHAMP["log loss"] or r.brier > WF_CHAMP["Brier"]
    if acc_up and prob_worse:
        print("\n  FLAG: accuracy is higher than the research figure while log loss")
        print("  and/or Brier are worse. Accuracy alone is not being used to judge.")
    if (not acc_up) and not prob_worse:
        print("\n  NOTE: accuracy is below the research figure while probability")
        print("  quality holds up or improves. Probability quality is the more")
        print("  reliable signal; accuracy is an argmax step function and moves")
        print("  more on small samples.")
    if dstat["Draw argmax"] > 0 and auc <= 0.5:
        print("\n  FLAG: Draw predictions are being made with AUC at or below chance.")

    # ============================================ PHASE G: ARTIFACT (IF ACCEPTED)
    rule("PHASE G / STEP 7 -- PRODUCTION ARTIFACT")
    artifact_path = REPO / OUT_ARTIFACT
    backup = None
    if decision != "ACCEPTABLE FOR PRODUCTION":
        print(f"  Decision is {decision}. NO artifact is written and nothing is")
        print("  replaced. The model is not retrained and no criterion is revisited.")
    else:
        payload = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION_V2,
            "feature_version": REQUIRED_FEATURE_VERSION,
            "feature_columns": tuple(FEATURES),
            "n_features": len(FEATURES),
            "base_contract_columns": tuple(MODEL_B_COLUMNS),
            "venue_columns": tuple(VENUE_VALUE_COLUMNS),
            "class_order": ["H", "D", "A"],
            "model_home_goals": model_h,
            "model_away_goals": model_a,
            "preprocessor": prep,
            "estimator_config": {"estimator": "PoissonRegressor",
                                 "alpha": POISSON_ALPHA, "max_iter": POISSON_MAX_ITER},
            "conversion": {"method": "independent Poisson, tail-safe",
                           "tail_tolerance": step3.TAIL_TOL,
                           "grid": "adaptive from max lambda",
                           "p_away": "by complement (rows sum to exactly 1)",
                           "source": STEP3_SCRIPT, "source_md5": STEP3_MD5},
            "training_seasons": tuple(FINAL_TRAIN_SEASONS),
            "training_season_ids": sorted(train_ids),
            "n_training_fixtures": int(len(fid_tr)),
            "holdout_season": tuple(FINAL_TEST_SEASONS),
            "holdout_evaluated_once": True,
            "holdout_used_in_training": False,
            "holdout_metrics": {"n": int(len(y)), "accuracy": float(r.accuracy),
                                "log_loss": float(r.log_loss), "brier": float(r.brier),
                                "draw_auc": auc},
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path.exists():
            backup = artifact_path.with_suffix(
                f".bak-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl")
            shutil.copyfile(artifact_path, backup)
            print(f"  previous artifact backed up to: {backup.name}")
        tmp = artifact_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(artifact_path)
        print(f"  artifact written : {artifact_path}")
        print(f"  artifact md5     : {md5(artifact_path)}")
        print("  v1_logreg.pkl is NOT replaced and NOT modified.")

        # ==================================== PHASE H: PRODUCTION SAFETY CHECKS
        rule("PHASE H / STEP 8 -- PRODUCTION SAFETY (reload + smoke test)")
        with open(artifact_path, "rb") as fh:
            back = pickle.load(fh)
        check("artifact deserialises", isinstance(back, dict))
        for k in ("model_home_goals", "model_away_goals", "preprocessor",
                  "feature_columns", "model_version", "class_order",
                  "training_seasons", "feature_version", "conversion"):
            check(f"artifact contains `{k}`", k in back)
        check("artifact contract is exactly 84 columns", len(back["feature_columns"]) == 84)
        check("artifact contract matches the trained input set",
              tuple(back["feature_columns"]) == tuple(FEATURES))
        check("artifact class order == ['H','D','A']", back["class_order"] == ["H", "D", "A"])
        check("artifact records that 2025/26 was NOT trained on",
              back["holdout_used_in_training"] is False)
        check("artifact estimator config preserved",
              back["estimator_config"]["alpha"] == POISSON_ALPHA
              and back["estimator_config"]["max_iter"] == POISSON_MAX_ITER)

        smoke = Xho.iloc[:25][list(back["feature_columns"])]
        Es = back["preprocessor"].transform(smoke)
        sh = back["model_home_goals"].predict(Es)
        sa = back["model_away_goals"].predict(Es)
        check("smoke test: 25 rows accepted, 84 columns", smoke.shape == (25, 84),
              str(smoke.shape))
        check("smoke test: lambdas finite and strictly positive",
              bool(np.all(np.isfinite(sh)) and np.all(np.isfinite(sa))
                   and np.all(sh > 0) and np.all(sa > 0)))
        Ps, Ks, rs, cs = hda_tail_safe(sh, sa)
        check("smoke test: rows sum to 1",
              bool(np.all(np.abs(Ps.sum(axis=1) - 1.0) < ROWSUM_TOL)))
        check("smoke test: tail residual within tolerance", rs < MASS_TOL)
        check("smoke test: complement control within tolerance", cs < COMPLEMENT_TOL)
        si, sj, sp = modal_scoreline(sh, sa, Ks)
        check("smoke test: scoreline output produced", len(si) == 25)
        check("smoke test: reload reproduces the original predictions exactly",
              bool(np.max(np.abs(Ps - P[:25])) == 0.0),
              f"max|diff|={float(np.max(np.abs(Ps - P[:25]))):.3e}")
        print(f"  smoke sample: E[home] {sh[0]:.4f}  E[away] {sa[0]:.4f}  "
              f"P(H/D/A) {Ps[0].round(4)}  top score {si[0]}-{sj[0]} p={sp[0]:.4f}")

    # ================================================================= REPORT
    rule("FINAL REPORT")
    print(f"   1. training seasons          : {', '.join(FINAL_TRAIN_SEASONS)}")
    print(f"   2. holdout season            : {', '.join(FINAL_TEST_SEASONS)}")
    print(f"   3. training fixtures         : {len(fid_tr)}")
    print(f"   4. holdout fixtures          : {len(y)}")
    print(f"   5. final accuracy            : {r.accuracy:.10f}  ({100 * r.accuracy:.2f}%)")
    print(f"   6. final log loss            : {r.log_loss:.10f}")
    print(f"   7. final Brier               : {r.brier:.10f}")
    print(f"   8. Draw-vs-rest AUC          : {auc:.10f}")
    print(f"   9. expected-goal metrics     : MAE home {float(np.abs(lam_h - h_true).mean()):.6f}"
          f"   MAE away {float(np.abs(lam_a - a_true).mean()):.6f}")
    print(f"                                  deviance home "
          f"{float(mean_poisson_deviance(h_true, lam_h)):.6f}"
          f"   away {float(mean_poisson_deviance(a_true, lam_a)):.6f}")
    print(f"  10. vs walk-forward champion  : accuracy "
          f"{100 * (r.accuracy - WF_CHAMP['accuracy']):+.2f} pp, "
          f"log loss {r.log_loss - WF_CHAMP['log loss']:+.6f}, "
          f"Brier {r.brier - WF_CHAMP['Brier']:+.6f}")
    print(f"  11. production decision       : {decision}")
    print(f"  12. artifact                  : "
          f"{artifact_path if decision == 'ACCEPTABLE FOR PRODUCTION' else 'NONE WRITTEN'}")
    if backup:
        print(f"      previous artifact backup  : {backup.name}")
    print("  13. 2025/26 evaluated exactly once: YES -- a single read, behind the")
    print("      outcome barrier, after all technical gates had passed")
    print("  14. tuning after seeing holdout  : NONE. No feature, hyperparameter,")
    print("      threshold, calibration or criterion was changed at any point.")

    sub("STANDING CAVEATS")
    print("  * 2025/26 is now fully spent. It cannot be used again for evaluation,")
    print("    and no further experiment should be judged on it.")
    print("  * This season was previously observed for frozen V1 (20- and 40-match")
    print("    samples), so a residual optimism of unknown size exists. It is")
    print("    disclosed rather than estimated away.")
    print("  * One season is one sample. No significance test is computed or implied.")
    print("  * The next genuinely clean evaluation is 2026/27, once ingested.")

    # ============================================================== INTEGRITY
    rule("POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("no pinned input, production source or existing artifact changed",
          not changed, str(changed))
    print("    production source changes : NONE")
    print("    database writes           : NONE")
    print("    v1_logreg.pkl             : UNCHANGED (never overwritten)")
    print("    feature writes            : NONE")
    print("    V1 retrained              : NO")
    print("    hyperparameters tuned     : NO")
    print(f"    new artifact written      : "
          f"{'YES -> ' + OUT_ARTIFACT.name if decision == 'ACCEPTABLE FOR PRODUCTION' else 'NO'}")
    for k in ("data/processed/features.db", "data/processed/matches.db",
              "data/models/v1_logreg.pkl", "src/models/train.py"):
        print(f"    {k:40s} {after[k]}")
    print(f"    LOCKED_INPUTS                            {len(pins)}/{len(pins)} unchanged")

    print("\n" + "=" * 128)
    print(f"FINAL HOLDOUT COMPLETE -- {decision}")
    print("=" * 128)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
