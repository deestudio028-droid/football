"""POISSON V2 -- STEP 5: VENUE-FEATURE CONTROLLED EXPERIMENT.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step_poisson_5_venue_experiment.py

THE ONE QUESTION
With the estimator, preprocessing, targets, conversion and folds all held
identical, do these four already-existing venue columns improve the current
two-Poisson model?

    BASELINE  two-Poisson on the frozen 80-column contract
    CANDIDATE two-Poisson on the 80 columns + exactly four venue columns

The four columns are the ONLY difference between the arms.

EXPERIMENT ONLY. Even if the candidate wins: production is not modified, V1 is
not replaced, no artifact is created, the feature contract is not changed.
Nothing is written anywhere. 2025/26 is never read.

------------------------------------------------------------------------------
THE CONVERSION IS IMPORTED, NOT COPIED
------------------------------------------------------------------------------
`hda_tail_safe` / `_grid_size` / `_pmf_grid` are loaded from
run_step_poisson_3_first_experiment.py and that file is PINNED BY CHECKSUM
below. Copying the functions would let the two experiments silently diverge;
importing them makes it provable that STEP 3 and STEP 5 ran the same maths.
The STEP 3 module is import-safe (its main() is guarded by __name__), so
importing executes no experiment.

------------------------------------------------------------------------------
ACCURACY vs THE PREREGISTERED RULE -- HOW THIS SCRIPT HANDLES THE TENSION
------------------------------------------------------------------------------
Accuracy is the practical objective and is reported first and prominently.
It is NOT the promotion criterion, and the criterion is not being changed
after the fact -- the STEP 4 rule was fixed before this script existed and is
reproduced here verbatim.

The reason accuracy alone cannot promote a model is mechanical, not
philosophical: argmax accuracy is a step function of the probabilities, so a
model can win extra argmax calls by nudging borderline cases while getting
materially worse at everything else. STEP 8 is the worked example from this
very project -- its probe raised the Draw argmax share 2.4% -> 13.1% while
Draw-vs-rest AUC FELL and log loss worsened by +0.0897.

So all three outcomes are reported distinctly and none is renamed:
  * rule passes AND accuracy improves -> "VENUE FEATURES IMPROVED THE MODEL"
  * accuracy improves, rule fails     -> "Accuracy improved, but the
                                          preregistered probabilistic guardrail
                                          did not pass."
  * accuracy does not improve         -> stated plainly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---- STEP 3 reference values (gate) ----------------------------------------
BASE_REF_LOG_LOSS = 0.9928234559
BASE_REF_BRIER = 0.5919683
BASE_REF_ACCURACY = 0.5272
TOL_SCORE = 1e-6          # log loss / Brier, quoted to 10 and 7 dp
TOL_ACC = 1e-3            # accuracy, quoted to 4 dp
V1_LOG_LOSS = 0.9993791056968738
V1_BRIER = 0.5965016957578898

# ---- pinned inputs ---------------------------------------------------------
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_MD5 = "5e504427712b35778bb8a62a8496c7cd"
TRAIN_PY_MD5 = "21425459195311492f49e73f5ae38fe0"
STEP3_SCRIPT = "run_step_poisson_3_first_experiment.py"
STEP3_MD5 = "24827e92bc0acb04b9fc1dbe5a9958ae"

POISSON_ALPHA = 1.0
POISSON_MAX_ITER = 2000
MASS_TOL = 1e-12
COMPLEMENT_TOL = 1e-10
ROWSUM_TOL = 1e-9

VENUE_VALUE_COLUMNS = (
    "home_goals_for_home_venue_season",
    "home_goals_against_home_venue_season",
    "away_goals_for_away_venue_season",
    "away_goals_against_away_venue_season",
)
VENUE_N_COLUMNS = tuple(c + "_n" for c in VENUE_VALUE_COLUMNS)

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
    print("STEP 5 ABORTED -- no result is reported")
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
                "src/features/history.py", STEP3_SCRIPT):
        snap[rel] = md5(REPO / rel)
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["data/models/v1_logreg.pkl"] = md5(art)
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
    # ================================================================= RULE 1
    rule("RULE 1 -- PRODUCTION SOURCE TRACE (source text, never memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:78]}", tok in tsrc)

    try:
        import sklearn
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import PoissonRegressor
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        abort(f"scikit-learn is required: {exc}. No estimator is substituted.")

    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        FINAL_TEST_SEASONS, MODEL_VERSION, RECOMMENDED_CONTEXT_FEATURE,
        REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS, WALK_FORWARD_FOLDS,
        X_EXCLUDED_COLUMNS,
    )
    from models.data import load_supervised_dataset
    from models.evaluate import evaluate
    from models.splits import iter_walk_forward_folds
    from models.train import LogisticRegressionPreprocessor

    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("production contract == exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric production features == exactly 79", len(numeric) == 79)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")

    rule("STEP 0 -- INTEGRITY GATE")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"pandas {pd.__version__}  sklearn {sklearn.__version__}")
    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(f"{rel} unchanged", before[rel] == exp, before[rel])
    check("src/models/train.py unchanged", before["src/models/train.py"] == TRAIN_PY_MD5)
    check("v1_logreg.pkl unchanged", before.get("data/models/v1_logreg.pkl") == ARTIFACT_MD5)
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if before[f"pin:{r}"] != v["expected"]]
    check(f"{len(pins)}/{len(pins)} LOCKED_INPUTS unchanged", not bad, str(bad))

    # ------------------------------------------------ import STEP 3 conversion
    sub("REUSED CONVERSION -- imported and checksum-pinned, not copied")
    s3 = REPO / STEP3_SCRIPT
    check(f"{STEP3_SCRIPT} present", s3.exists())
    print(f"  {STEP3_SCRIPT} md5 : {before[STEP3_SCRIPT]}")
    check("STEP 3 script is at its pinned checksum (conversion cannot have drifted)",
          before[STEP3_SCRIPT] == STEP3_MD5, before[STEP3_SCRIPT])
    spec = importlib.util.spec_from_file_location("_step3", s3)
    step3 = importlib.util.module_from_spec(spec)
    sys.modules["_step3"] = step3
    spec.loader.exec_module(step3)          # main() is __name__-guarded: no run
    hda_tail_safe = step3.hda_tail_safe
    check("hda_tail_safe imported from the pinned STEP 3 module", callable(hda_tail_safe))
    check("STEP 3's own tail tolerance is unchanged", step3.TAIL_TOL == 1e-15,
          str(step3.TAIL_TOL))

    # =============================================================== SOURCE TRACE
    rule("SOURCE TRACE -- seven required verifications")
    fdb = REPO / "data/processed/features.db"
    con = sqlite3.connect(f"file:{fdb.resolve()}?mode=ro", uri=True)
    try:
        all_cols = [r[1] for r in con.execute("PRAGMA table_info(feature_rows)")]
    finally:
        con.close()

    print("  (1) the four exact venue columns exist in features.db")
    for c in VENUE_VALUE_COLUMNS:
        check(f"      {c}", c in all_cols)
    print("  (2) they are NOT part of the frozen 80-column contract")
    check("      no overlap with MODEL_B_COLUMNS",
          not (set(VENUE_VALUE_COLUMNS) & set(MODEL_B_COLUMNS)))
    check("      none is an excluded/label column",
          not (set(VENUE_VALUE_COLUMNS) & set(X_EXCLUDED_COLUMNS)))

    fb = (REPO / "src/features/feature_builder.py").read_text(encoding="utf-8")
    rl = (REPO / "src/features/rolling.py").read_text(encoding="utf-8")
    print("  (3) construction uses historical data only")
    for tok, why in (
        ('season_matches = [m for m in history if m["season_id"] == target_season_id]',
         "reads `history`, i.e. history_before() output"),
        ('venue_matches = [m for m in season_matches if m["is_home"] == home_only]',
         "venue restriction over that same history"),
        ("value = mean if coverage_n >= VENUE_MIN_COVERAGE else None",
         "NULL below the minimum-coverage gate"),
    ):
        check(f"      rolling.py: {why}", tok in rl)

    print("  (4) build_feature_row executes before ctx.record")
    import ast
    tree = ast.parse(fb)
    walker = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "_walk_and_build"), None)
    check("      single chronological walker exists", walker is not None)
    loop = next((n for n in ast.walk(walker) if isinstance(n, ast.For)), None)
    bl = rl_ = None
    for node in ast.walk(loop):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "build_feature_row":
                bl = node.lineno
            if isinstance(fn, ast.Attribute) and fn.attr == "record":
                rl_ = node.lineno
    check("      build_feature_row precedes ctx.record in the same iteration",
          bl is not None and rl_ is not None and bl < rl_, f"build@{bl} < record@{rl_}")
    check("      exactly one ctx.record call site",
          sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "record") == 1)

    print("  (5) no current-fixture goals enter _venue_features")
    vfun = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "_venue_features"), None)
    check("      _venue_features exists", vfun is not None)
    vargs = [a.arg for a in vfun.args.args]
    check("      receives no fixture object and no goals argument",
          "fixture" not in vargs and not any("goal" in a for a in vargs), str(vargs))
    vsrc = ast.get_source_segment(fb, vfun) or ""
    for banned in ("fixture", "label_", "home_goals", "away_goals"):
        check(f"      body never references `{banned}`", banned not in vsrc)

    print("  (6) the four *_n columns are excluded from the model")
    for c in VENUE_N_COLUMNS:
        check(f"      {c} exists but is NOT used", c in all_cols)

    print("  (7) 2025/26 is excluded")
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    for f in WALK_FORWARD_FOLDS:
        check(f"      {f.name}: 2025/26 in neither partition",
              not (set(f.train_seasons + f.validation_seasons) & set(FINAL_TEST_SEASONS)))
    check("      exactly three folds, read from config (not hard-coded)",
          len(WALK_FORWARD_FOLDS) == 3)

    BASE_COLS = list(MODEL_B_COLUMNS)
    CAND_COLS = list(MODEL_B_COLUMNS) + list(VENUE_VALUE_COLUMNS)
    check("candidate input set is exactly baseline + 4", len(CAND_COLS) == 84)
    check("the only difference between arms is the four venue columns",
          set(CAND_COLS) - set(BASE_COLS) == set(VENUE_VALUE_COLUMNS))

    # ==================================================================== DATA
    ds = load_supervised_dataset(fdb)
    con = sqlite3.connect(f"file:{fdb.resolve()}?mode=ro", uri=True)
    try:
        goals = pd.read_sql_query(
            "SELECT fixture_id, label_home_goals, label_away_goals "
            "FROM feature_rows WHERE label_result IS NOT NULL", con)
    finally:
        con.close()
    gh = dict(zip(goals["fixture_id"], goals["label_home_goals"]))
    ga = dict(zip(goals["fixture_id"], goals["label_away_goals"]))

    def goals_for(subset):
        fids = subset.metadata["fixture_id"].to_numpy()
        h = np.array([gh[f] for f in fids], dtype=float)
        a = np.array([ga[f] for f in fids], dtype=float)
        derived = np.where(h > a, "H", np.where(h < a, "A", "D"))
        if not np.array_equal(derived, subset.y.to_numpy()):
            abort("goal targets disagree with label_result on the modelling rows.")
        return h, a

    def fit_arm(cols, tr, va, htr, atr, tag, fold_name):
        Xtr, Xva = tr.X[cols], va.X[cols]
        prep = LogisticRegressionPreprocessor()
        prep.fit(Xtr)                                  # TRAIN partition only
        Etr, Eva = prep.transform(Xtr), prep.transform(Xva)
        n_num = len(cols) - 1                          # competition_id is categorical
        check(f"{fold_name}/{tag}: preprocessing fitted on train only, "
              f"{len(prep._imputer.statistics_)} numeric columns",
              len(prep._imputer.statistics_) == n_num,
              f"{len(prep._imputer.statistics_)} vs {n_num}")
        check(f"{fold_name}/{tag}: encoded train/val widths match",
              Etr.shape[1] == Eva.shape[1], f"{Etr.shape} vs {Eva.shape}")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            mh = PoissonRegressor(alpha=POISSON_ALPHA,
                                  max_iter=POISSON_MAX_ITER).fit(Etr, htr)
            ma = PoissonRegressor(alpha=POISSON_ALPHA,
                                  max_iter=POISSON_MAX_ITER).fit(Etr, atr)
            conv = [x for x in w if issubclass(x.category, ConvergenceWarning)]
        check(f"{fold_name}/{tag}: both Poisson models converged", not conv,
              "; ".join(str(x.message)[:80] for x in conv))
        lam_h, lam_a = mh.predict(Eva), ma.predict(Eva)
        check(f"{fold_name}/{tag}: lambdas finite and strictly positive",
              bool(np.all(np.isfinite(lam_h)) and np.all(np.isfinite(lam_a))
                   and np.all(lam_h > 0) and np.all(lam_a > 0)),
              f"h[{lam_h.min():.4f},{lam_h.max():.4f}] a[{lam_a.min():.4f},{lam_a.max():.4f}]")
        P, K, resid, ctl = hda_tail_safe(lam_h, lam_a)
        check(f"{fold_name}/{tag}: tail residual < {MASS_TOL:.0e} (grid K={K})",
              resid < MASS_TOL, f"{resid:.3e}")
        check(f"{fold_name}/{tag}: P(A) complement vs direct < {COMPLEMENT_TOL:.0e}",
              ctl < COMPLEMENT_TOL, f"{ctl:.3e}")
        check(f"{fold_name}/{tag}: probability rows sum to 1",
              bool(np.all(np.abs(P.sum(axis=1) - 1.0) < ROWSUM_TOL)),
              f"max|sum-1|={float(np.max(np.abs(P.sum(axis=1)-1.0))):.3e}")
        check(f"{fold_name}/{tag}: all probabilities finite", bool(np.all(np.isfinite(P))))
        return P

    # ================================================================ FOLD LOOP
    rule("VENUE COVERAGE AND FOLD-BY-FOLD EXPERIMENT")
    per_fold, fold_names = {}, []
    for fold, tr, va in iter_walk_forward_folds(ds):
        fold_names.append(fold.name)
        sub(f"{fold.name}   train n={len(tr)}   validation n={len(va)}")
        check(f"{fold.name}: no 2025/26 in train",
              not set(tr.metadata["season_id"]) & test_ids)
        check(f"{fold.name}: no 2025/26 in validation",
              not set(va.metadata["season_id"]) & test_ids)
        check(f"{fold.name}: train/validation fixtures disjoint",
              not (set(tr.metadata["fixture_id"]) & set(va.metadata["fixture_id"])))
        check(f"{fold.name}: validation strictly later than train",
              float(va.metadata["unix"].min()) > float(tr.metadata["unix"].max()))

        print("  venue coverage (no *_n column is used by either arm):")
        print("  | column | train coverage | validation coverage |")
        print("  |---|---:|---:|")
        for c in VENUE_VALUE_COLUMNS:
            print(f"  | {c} | {tr.X[c].notna().mean():.4f} | {va.X[c].notna().mean():.4f} |")
        print(f"  all four present: train {tr.X[list(VENUE_VALUE_COLUMNS)].notna().all(axis=1).mean():.4f}"
              f"   validation {va.X[list(VENUE_VALUE_COLUMNS)].notna().all(axis=1).mean():.4f}")
        print("  Missing values are imputed by the production training-partition")
        print("  median, unchanged. Missingness is early-season by construction.")

        htr, atr = goals_for(tr)
        P_base = fit_arm(BASE_COLS, tr, va, htr, atr, "baseline", fold.name)
        P_cand = fit_arm(CAND_COLS, tr, va, htr, atr, "candidate", fold.name)

        y = va.y.to_numpy()
        rec = {}
        for tag, P in (("baseline", P_base), ("candidate", P_cand)):
            r = evaluate(va.y, P)
            rec[tag] = {
                "n": len(y), "accuracy": r.accuracy, "log loss": r.log_loss,
                "Brier": r.brier,
                "Draw AUC": float(roc_auc_score((y == "D").astype(int), P[:, 1])),
                **draw_stats(P, y),
            }
        per_fold[fold.name] = rec

        print(f"\n  | metric | baseline | candidate | delta |")
        print("  |---|---:|---:|---:|")
        for m in ("accuracy", "log loss", "Brier", "Draw AUC", "Draw argmax",
                  "mean P(D)", "SD P(D)", "P(D)>1/3"):
            b, c = rec["baseline"][m], rec["candidate"][m]
            print(f"  | {m} | {b:.10f} | {c:.10f} | {c - b:+.10f} |")
        print(f"  | actual draw rate | {rec['baseline']['actual draw rate']:.4f} | "
              f"{rec['candidate']['actual draw rate']:.4f} | (same rows) |")
        d_acc = rec["candidate"]["accuracy"] - rec["baseline"]["accuracy"]
        print(f"\n  ACCURACY on {fold.name}: "
              f"{'IMPROVED' if d_acc > 0 else ('UNCHANGED' if d_acc == 0 else 'WORSE')}"
              f"  ({d_acc:+.10f} = {100 * d_acc:+.4f} percentage points)")

    # ======================================================= POOLED + GATE
    def pool(tag, m):
        return float(np.mean([per_fold[f][tag][m] for f in fold_names]))

    rule("BASELINE REPRODUCTION GATE (before the candidate result is accepted)")
    b_ll, b_br, b_ac = pool("baseline", "log loss"), pool("baseline", "Brier"), \
        pool("baseline", "accuracy")
    print(f"  reproduced pooled log loss : {b_ll:.16f}   reference "
          f"{BASE_REF_LOG_LOSS:.10f}   diff {b_ll - BASE_REF_LOG_LOSS:+.3e}")
    print(f"  reproduced pooled Brier    : {b_br:.16f}   reference "
          f"{BASE_REF_BRIER:.7f}   diff {b_br - BASE_REF_BRIER:+.3e}")
    print(f"  reproduced pooled accuracy : {b_ac:.16f}   reference "
          f"{BASE_REF_ACCURACY:.4f}   diff {b_ac - BASE_REF_ACCURACY:+.3e}")
    print("  (pooled = unweighted mean of the three fold values, the STEP 3 convention)")
    check("baseline reproduces the STEP 3 pooled log loss",
          abs(b_ll - BASE_REF_LOG_LOSS) <= TOL_SCORE, f"tol {TOL_SCORE:.0e}")
    check("baseline reproduces the STEP 3 pooled Brier",
          abs(b_br - BASE_REF_BRIER) <= TOL_SCORE, f"tol {TOL_SCORE:.0e}")
    check("baseline reproduces the STEP 3 pooled accuracy",
          abs(b_ac - BASE_REF_ACCURACY) <= TOL_ACC, f"tol {TOL_ACC:.0e}")
    print(f"\n  V1 walk-forward reference (secondary, cannot promote anything):")
    print(f"    log loss {V1_LOG_LOSS:.10f}   Brier {V1_BRIER:.10f}")

    rule("POOLED COMPARISON")
    c_ll, c_br, c_ac = pool("candidate", "log loss"), pool("candidate", "Brier"), \
        pool("candidate", "accuracy")
    print("| metric | Poisson 80 | Poisson + Venue | delta |")
    print("|---|---:|---:|---:|")
    for m in ("accuracy", "log loss", "Brier", "Draw AUC", "mean P(D)", "SD P(D)",
              "Draw argmax", "P(D)>1/3"):
        b, c = pool("baseline", m), pool("candidate", m)
        print(f"| {m} | {b:.10f} | {c:.10f} | {c - b:+.10f} |")
    d_ll, d_br, d_ac = c_ll - b_ll, c_br - b_br, c_ac - b_ac
    print("\n  sign convention: accuracy positive = better; log loss / Brier negative = better")
    print(f"\n  ACCURACY: {100 * b_ac:.2f}% -> {100 * c_ac:.2f}% "
          f"= {100 * d_ac:+.2f} percentage points")

    rule("FOLD CONSISTENCY (not hidden behind the pooled mean)")
    imp_ac = sum(per_fold[f]["candidate"]["accuracy"] > per_fold[f]["baseline"]["accuracy"]
                 for f in fold_names)
    imp_ll = sum(per_fold[f]["candidate"]["log loss"] < per_fold[f]["baseline"]["log loss"]
                 for f in fold_names)
    imp_br = sum(per_fold[f]["candidate"]["Brier"] < per_fold[f]["baseline"]["Brier"]
                 for f in fold_names)
    print("| fold | acc delta | log-loss delta | Brier delta |")
    print("|---|---:|---:|---:|")
    for f in fold_names:
        print(f"| {f} | "
              f"{per_fold[f]['candidate']['accuracy'] - per_fold[f]['baseline']['accuracy']:+.10f} | "
              f"{per_fold[f]['candidate']['log loss'] - per_fold[f]['baseline']['log loss']:+.10f} | "
              f"{per_fold[f]['candidate']['Brier'] - per_fold[f]['baseline']['Brier']:+.10f} |")
    print(f"\n  accuracy improved on {imp_ac}/3 folds")
    print(f"  log loss improved on {imp_ll}/3 folds")
    print(f"  Brier    improved on {imp_br}/3 folds")

    # ================================================================ SIGNATURE
    rule("STEP 8 DRAW FAILURE SIGNATURE TEST")
    d_argmax = pool("candidate", "Draw argmax") - pool("baseline", "Draw argmax")
    d_auc = pool("candidate", "Draw AUC") - pool("baseline", "Draw AUC")
    print(f"  Draw argmax share : {pool('baseline','Draw argmax'):.4f} -> "
          f"{pool('candidate','Draw argmax'):.4f}   delta {d_argmax:+.4f}")
    print(f"  Draw-vs-rest AUC  : {pool('baseline','Draw AUC'):.4f} -> "
          f"{pool('candidate','Draw AUC'):.4f}   delta {d_auc:+.4f}")
    signature = (d_argmax > 0) and (d_auc <= 0)
    if signature:
        print("\n  >>> STEP 8 FAILURE SIGNATURE RECURRED <<<")
        print("  More Draw predictions WITHOUT better Draw discrimination.")
        print("  This is not balance and not an improvement.")
    else:
        print("\n  Signature NOT present.")

    # ================================================================= DECISION
    rule("PREREGISTERED DECISION RULE (fixed in STEP 4, reproduced verbatim, unchanged)")
    c1 = c_ll < b_ll
    c2 = imp_ll >= 2
    c3 = c_br <= b_br
    print(f"  1. candidate pooled log loss < baseline   : {c1}   "
          f"({c_ll:.10f} vs {b_ll:.10f})")
    print(f"  2. log loss improves on >= 2 of 3 folds   : {c2}   ({imp_ll}/3)")
    print(f"  3. candidate pooled Brier <= baseline     : {c3}   (delta {d_br:+.10f})")
    both_worse = (c_ll > b_ll) and (c_br > b_br)
    if c1 and c2 and c3:
        verdict = "BETTER"
    elif both_worse:
        verdict = "WORSE"
    else:
        verdict = "INCONCLUSIVE"
    acc_improved = d_ac > 0

    rule("FINAL VERDICT")
    print(f"  1. baseline accuracy                 : {b_ac:.10f}  ({100 * b_ac:.2f}%)")
    print(f"  2. venue candidate accuracy          : {c_ac:.10f}  ({100 * c_ac:.2f}%)")
    print(f"  3. accuracy delta                    : {100 * d_ac:+.2f} percentage points")
    print(f"  4. accuracy improved on              : {imp_ac}/3 folds")
    print(f"  5. log-loss delta                    : {d_ll:+.10f}")
    print(f"  6. Brier delta                       : {d_br:+.10f}")
    print(f"  7. PREREGISTERED VERDICT             : {verdict}")
    print(f"  8. accuracy improved                 : {'YES' if acc_improved else 'NO'}")

    print()
    if verdict == "BETTER" and acc_improved:
        print("  VENUE FEATURES IMPROVED THE MODEL")
        print(f"  Accuracy improvement: {100 * b_ac:.2f}% -> {100 * c_ac:.2f}% "
              f"= {100 * d_ac:+.2f} percentage points")
        carry = ("YES -- carry forward as the new baseline for the next experiment, "
                 "subject to the multiple-comparison caveat below.")
    elif acc_improved:
        print("  Accuracy improved, but the preregistered probabilistic guardrail")
        print("  did not pass.")
        print(f"  Accuracy {100 * b_ac:.2f}% -> {100 * c_ac:.2f}% "
              f"({100 * d_ac:+.2f} pp), but verdict is {verdict}.")
        print("  This is the pattern the guardrail exists to catch: better argmax")
        print("  calls without better probabilities. NOT an overall improvement.")
        carry = "NO -- accuracy alone does not promote a model under the fixed rule."
    else:
        print(f"  Accuracy did NOT improve ({100 * d_ac:+.2f} percentage points).")
        print(f"  Preregistered verdict: {verdict}.")
        carry = ("NO" if verdict != "BETTER" else
                 "PARTIAL -- probabilistic rule passed while accuracy did not; "
                 "report both, promote nothing on this evidence alone.")
    if signature:
        print("  Additionally: STEP 8 FAILURE SIGNATURE RECURRED.")
    print(f"\n  9. worth carrying forward: {carry}")

    sub("STANDING CAVEATS -- these do not change with the result")
    print("  * Three folds support RANGES only. No standard deviation, no p-value,")
    print("    no significance claim is computed or implied.")
    print("  * This is the SECOND candidate tried against these same three folds.")
    print("    Sequential testing with a stop-at-first-win inflates spurious-win")
    print("    risk; a second BETTER verdict is weaker evidence than the first.")
    print("  * 2025/26 remains untouched and is the only remaining honest check.")
    print("  * Nothing here is a production recommendation.")

    # ================================================================ INTEGRITY
    rule("POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified anywhere", not changed, str(changed))
    print("    production changes: NONE")
    print("    database writes: NONE")
    print("    artifact writes: NONE")
    print("    feature writes: NONE")
    print("    2025/26 used: NO")
    print("    V1 retrained: NO      hyperparameters tuned: NO")
    print("    files created: NONE")
    for k in ("data/processed/features.db", "data/processed/matches.db",
              "data/models/v1_logreg.pkl", "src/models/train.py",
              "src/features/rolling.py", STEP3_SCRIPT):
        print(f"    {k:42s} {after[k]}")
    print(f"    LOCKED_INPUTS                              {len(pins)}/{len(pins)} unchanged")

    print("\n" + "=" * 128)
    print("STEP 5 COMPLETE -- VENUE EXPERIMENT")
    print("=" * 128)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
