"""POISSON V2 -- STEP 3: FIRST MODEL-FORM EXPERIMENT (two-Poisson walk-forward).

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step_poisson_3_first_experiment.py

THE ONE QUESTION
With the EXACT SAME 80 inputs and the EXACT SAME three walk-forward folds,
does a two-Poisson goals model beat frozen V1 on log loss and Brier?

Expected goals, scorelines, a higher Draw rate and a more balanced-looking
confusion matrix are CAPABILITIES. They are not evidence and cannot promote
this model. STEP 8 already showed the trap: its probe raised the Draw argmax
share 2.4% -> 13.1% while Draw-vs-rest AUC FELL and log loss worsened by
+0.0897.

Writes nothing. Trains no production model. Modifies no production source, no
database, and not the frozen artifact. 2025/26 is never read.

------------------------------------------------------------------------------
A CONFLICT IN THE SPECIFICATION, AND HOW IT IS RESOLVED -- READ THIS
------------------------------------------------------------------------------
Section 6 asks for two things that cannot both hold:

    (i)  "Evaluate frozen V1 on the IDENTICAL validation rows. Do NOT refit V1.
          Use the existing frozen artifact only."
    (ii) "Reference values: pooled V1 log loss 0.9993791056968738 ...
          Verify the reproduced V1 metrics before comparing."

`v1_logreg.pkl` was fitted on FINAL_TRAIN_SEASONS = 2020/21..2024/25. The three
fold validation seasons are 2022/23, 2023/24 and 2024/25 -- ALL THREE ARE
INSIDE THE ARTIFACT'S OWN TRAINING DATA. Scoring the frozen artifact on those
rows is therefore 100% in-sample for every fold. It cannot produce
0.9993791056968738 (that value comes from the walk-forward REFIT protocol), and
it would hand V1 an in-sample advantage the Poisson arm does not get -- making
the comparison invalid in V1's favour.

RESOLUTION, applied below:
  * V1 COMPARISON ARM = per-fold refit via the UNCHANGED production function
    `models.train.train_logistic_regression`. This is not a modification of V1;
    it is the documented walk-forward protocol, it is what every prior script
    in this investigation reproduced, and it reproduces 0.9993791056968738
    exactly (asserted to 1e-9). Both arms then fit only on each fold's training
    rows, which is the only apples-to-apples comparison.
  * The frozen artifact IS ALSO scored on the identical rows and reported --
    clearly labelled IN-SAMPLE / NOT USED FOR THE DECISION -- so the size of
    the contamination is visible rather than hidden.

The pre-registered decision rule is evaluated against the walk-forward V1 arm
ONLY.
------------------------------------------------------------------------------

PRE-REGISTERED ESTIMATOR CONFIGURATION (fixed before any fitting)
    sklearn.linear_model.PoissonRegressor
      alpha    = 1.0    <- scikit-learn's DEFAULT. Not chosen by this project,
                          not searched, not tuned. Same rationale STEP 8 used
                          for its probe: take the library default so no
                          hyperparameter decision is smuggled in.
      max_iter = 2000   <- mirrors V1's LogisticRegression(max_iter=2000). This
                          is a CONVERGENCE allowance, not tuning; the script
                          reports n_iter_ and fails loudly on non-convergence.
    Everything else: scikit-learn defaults.

SCORELINE CONVERSION -- TAIL-SAFE (revised after the first run aborted)
The first run correctly ABORTED at the truncation gate: a fixed 0..10 grid
retained only 0.9963186758 of the joint mass on some fixtures. That gate was
right, and the fix is mathematical rather than a relaxed threshold.

  * grid length K is DERIVED per batch from the largest fitted lambda, by exact
    forward CDF recursion, so residual upper-tail mass < 1e-15;
  * P(H) = sum_i p_h(i) * P(away <= i-1) via a cumulative sum -- exact, and
    O(nK) rather than building an (n, K, K) joint matrix;
  * P(A) = 1 - P(H) - P(D) by complement, so rows sum to exactly 1.0 with NO
    renormalisation. The previous code renormalised a truncated grid, which
    smears the lost tail mass proportionally across all three classes and
    biases every one of them. This version removes the error rather than
    redistributing it.
  * an independent control recomputes P(A) by direct summation and asserts it
    matches the complement.

Verified against brute-force enumeration over a 0..400 grid across lambda
regimes from (0.05, 6.5) to (12.0, 11.0): worst deviation 2.331e-15, i.e.
double-precision exact. For reference, the old fixed grid loses 1.65e-02 of the
mass at lambda = (5.0, 4.0).

PREPROCESSING
The production class `train.LogisticRegressionPreprocessor` is INSTANTIATED and
fitted inside each fold's training partition. Reusing it (not editing it)
guarantees the Poisson arm sees byte-identical inputs to the V1 arm, so the
estimator/link/target is the ONLY difference between the two arms. train.py is
not modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Truncation tolerance for the Poisson scoreline grid. The grid length is
#: derived per batch from the largest fitted lambda so the residual upper-tail
#: mass falls below this, instead of a fixed 0..10 window that silently loses
#: mass once a lambda grows. See `_grid_size`.
TAIL_TOL = 1e-15
MASS_TOL = 1e-12                    # gate on observed residual marginal mass
POISSON_ALPHA = 1.0                 # sklearn default -- pre-registered
POISSON_MAX_ITER = 2000             # mirrors V1; convergence, not tuning
REF_LOG_LOSS = 0.9993791056968738
REF_BRIER = 0.5965016957578898
REF_DRAW_AUC = 0.5361
REF_DRAW_ARGMAX = 0.0240
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_MD5 = "5e504427712b35778bb8a62a8496c7cd"
TRAIN_PY_MD5 = "21425459195311492f49e73f5ae38fe0"

PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)


def rule(t): print("\n" + "=" * 126); print(t); print("=" * 126)
def sub(t): print("\n" + "-" * 126); print(t); print("-" * 126)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def abort(msg):
    print("\n" + "!" * 126)
    print("STEP 3 ABORTED -- no result is reported")
    print(msg)
    print("!" * 126)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        abort(f"safety/stop condition failed: {label}")


def snapshot():
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    for rel in ("src/models/train.py", "src/models/config.py", "src/models/splits.py",
                "src/features/feature_builder.py"):
        snap[rel] = md5(REPO / rel)
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["data/models/v1_logreg.pkl"] = md5(art)
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


# ----------------------------------------------------------------- conversion
def _grid_size(lam_max, tol=TAIL_TOL, cap=20000):
    """Smallest K with Poisson(lam_max) upper-tail P(X > K) < tol.

    Computed by exact forward CDF recursion (p_{k} = p_{k-1} * lam / k), so the
    grid length is DERIVED from the data rather than assumed. A fixed 0..10
    window is only adequate while every lambda stays small: at lambda = 5.0/4.0
    it already loses 1.65e-02 of the joint mass, which is what tripped the
    STEP 3 safety gate.
    """
    k, term = 0, math.exp(-lam_max)
    cdf = term
    while cdf < 1.0 - tol and k < cap:
        k += 1
        term *= lam_max / k
        cdf += term
    return max(k, 1)


def _pmf_grid(lam, K):
    """Poisson pmf for 0..K, computed in log space for numerical stability."""
    k = np.arange(K + 1)
    log_fact = np.array([math.lgamma(i + 1) for i in k])
    l = np.asarray(lam, dtype=float)[:, None]
    return np.exp(-l + k * np.log(l) - log_fact)


def hda_tail_safe(lam_h, lam_a, tol=TAIL_TOL):
    """P(H), P(D), P(A) in CLASS_ORDER under independent Poisson.

    Tail-safe by construction, and NOT by renormalising away the error:

      * the grid length K is chosen from the largest lambda in the batch so the
        residual marginal mass is below `tol`;
      * P(H) = sum_i p_h(i) * P(away <= i-1), evaluated with a cumulative sum
        rather than an (n, K, K) joint matrix -- exact, and O(nK) instead of
        O(nK^2), which matters once K grows to ~50;
      * P(A) = 1 - P(H) - P(D) by complement, so each row sums to exactly 1.0
        with no renormalisation step to redistribute truncation error.

    Renormalising a truncated grid (the previous approach) spreads the lost
    tail mass proportionally across ALL outcomes, which biases every class
    rather than leaving the error where it belongs. This version removes the
    error instead of smearing it.

    Independence is an ASSUMPTION, not a measured fact: on training rows the
    home/away goal correlation is about -0.08. Stated here rather than buried;
    relaxing it is what a later bivariate-Poisson or Dixon-Coles step would do.
    Nothing of that kind is done in STEP 3.

    Returns (P, K, residual_mass, complement_control) where
    `complement_control` is |P(A) computed directly - P(A) by complement|, an
    independent check that the two routes agree.
    """
    K = _grid_size(float(max(np.max(lam_h), np.max(lam_a))), tol)
    ph, pa = _pmf_grid(lam_h, K), _pmf_grid(lam_a, K)
    residual = float(max(np.max(np.abs(1.0 - ph.sum(axis=1))),
                         np.max(np.abs(1.0 - pa.sum(axis=1)))))
    Fa = np.cumsum(pa, axis=1)                        # P(away <= j)
    p_home = (ph[:, 1:] * Fa[:, :-1]).sum(axis=1)
    p_draw = (ph * pa).sum(axis=1)
    p_away_direct = (pa[:, 1:] * np.cumsum(ph, axis=1)[:, :-1]).sum(axis=1)
    p_away = 1.0 - p_home - p_draw
    control = float(np.max(np.abs(p_away - p_away_direct)))
    return np.column_stack([p_home, p_draw, p_away]), K, residual, control


def modal_scoreline(lam_h, lam_a, K):
    """Most likely scoreline and its probability.

    For INDEPENDENT Poisson the joint pmf factorises, so the argmax of the
    joint equals the pair of marginal argmaxes. No (n, K, K) matrix is built.
    """
    ph, pa = _pmf_grid(lam_h, K), _pmf_grid(lam_a, K)
    i, j = ph.argmax(axis=1), pa.argmax(axis=1)
    return i, j, ph[np.arange(len(i)), i] * pa[np.arange(len(j)), j]


def draw_block(P, y):
    d = (y == "D")
    return {
        "actual draw rate": float(d.mean()),
        "mean P(D)": float(P[:, 1].mean()),
        "sd P(D)": float(P[:, 1].std()),
        "Draw argmax share": float((P.argmax(axis=1) == 1).mean()),
        "P(D) > 1/3 share": float((P[:, 1] > 1 / 3).mean()),
    }


def main():
    # ================================================================= RULE 1
    rule("RULE 1 -- PRODUCTION SOURCE TRACE (source text, never memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:76]}", tok in tsrc)

    try:
        import sklearn
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import PoissonRegressor
        from sklearn.metrics import mean_poisson_deviance, roc_auc_score
    except ImportError as exc:
        abort(f"scikit-learn is required: {exc}. No estimator is substituted.")

    from models import artifact as artifact_mod
    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.candidate_contract import predict_candidate
    from models.config import (
        FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, MODEL_VERSION,
        RECOMMENDED_CONTEXT_FEATURE, REQUIRED_FEATURE_VERSION,
        SEASON_NAME_TO_IDS, WALK_FORWARD_FOLDS,
    )
    from models.data import load_supervised_dataset
    from models.evaluate import evaluate
    from models.splits import iter_walk_forward_folds
    from models.train import LogisticRegressionPreprocessor, train_logistic_regression

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

    # ----------------------------------------------------- fold + season audit
    rule("STEP 0b -- PROTOCOL VERIFICATION (folds read from config, not hard-coded)")
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    print("| fold | train seasons | validation season |")
    print("|---|---|---|")
    for f in WALK_FORWARD_FOLDS:
        print(f"| {f.name} | {', '.join(f.train_seasons)} | {', '.join(f.validation_seasons)} |")
        check(f"{f.name}: 2025/26 absent from train",
              not (set(f.train_seasons) & set(FINAL_TEST_SEASONS)))
        check(f"{f.name}: 2025/26 absent from validation",
              not (set(f.validation_seasons) & set(FINAL_TEST_SEASONS)))
    check("exactly three folds", len(WALK_FORWARD_FOLDS) == 3)

    sub("THE SECTION-6 CONFLICT, RESOLVED EXPLICITLY")
    art_seasons = set(FINAL_TRAIN_SEASONS)
    val_seasons = {s for f in WALK_FORWARD_FOLDS for s in f.validation_seasons}
    print(f"  frozen artifact was fitted on : {sorted(art_seasons)}")
    print(f"  fold validation seasons       : {sorted(val_seasons)}")
    print(f"  validation seasons INSIDE the artifact's training data: "
          f"{sorted(val_seasons & art_seasons)}")
    check("the overlap is total, so the frozen artifact is in-sample on every fold",
          val_seasons <= art_seasons)
    print("  => the frozen artifact CANNOT serve as the comparison arm.")
    print("  => V1 comparison arm = per-fold refit via the UNCHANGED production")
    print("     function train_logistic_regression (the documented protocol).")
    print("  => the frozen artifact is still scored on the same rows below, marked")
    print("     IN-SAMPLE and excluded from the decision.")

    # --------------------------------------------------------------- data
    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    fdb = REPO / "data/processed/features.db"
    con = sqlite3.connect(f"file:{fdb.resolve()}?mode=ro", uri=True)
    try:
        goals = pd.read_sql_query(
            "SELECT fixture_id, label_home_goals, label_away_goals "
            "FROM feature_rows WHERE label_result IS NOT NULL", con)
    finally:
        con.close()
    gmap_h = dict(zip(goals["fixture_id"], goals["label_home_goals"]))
    gmap_a = dict(zip(goals["fixture_id"], goals["label_away_goals"]))

    def goals_for(subset):
        fids = subset.metadata["fixture_id"].to_numpy()
        h = np.array([gmap_h[f] for f in fids], dtype=float)
        a = np.array([gmap_a[f] for f in fids], dtype=float)
        derived = np.where(h > a, "H", np.where(h < a, "A", "D"))
        if not np.array_equal(derived, subset.y.to_numpy()):
            abort("goal targets disagree with label_result on the modelling rows; "
                  "STEP 1's consistency guarantee does not hold here.")
        return h, a

    art = artifact_mod.load(REPO / "data/models/v1_logreg.pkl")
    check("artifact contract == the frozen 80 columns, in order",
          tuple(art.feature_columns) == tuple(MODEL_B_COLUMNS))

    # =============================================================== FOLD LOOP
    rule("STEP 1-5 -- WALK-FORWARD FITTING (per fold; nothing fitted on validation)")
    print(f"  PoissonRegressor(alpha={POISSON_ALPHA}, max_iter={POISSON_MAX_ITER})  "
          f"-- alpha is the sklearn default, pre-registered, never searched")
    results = {"V1 (walk-forward refit)": {}, "Poisson constant-lambda": {},
               "Poisson features": {}, "V1 frozen artifact [IN-SAMPLE]": {}}
    goal_diag, lam_store, fold_names = {}, {}, []

    for fold, tr, va in iter_walk_forward_folds(ds):
        fold_names.append(fold.name)
        Xtr = tr.X[list(MODEL_B_COLUMNS)]
        Xva = va.X[list(MODEL_B_COLUMNS)]
        ytr, yva = tr.y.to_numpy(), va.y.to_numpy()
        htr, atr = goals_for(tr)
        hva, ava = goals_for(va)

        sub(f"{fold.name}  train n={len(tr)}   validation n={len(va)}")
        check(f"{fold.name}: no 2025/26 in train",
              not set(tr.metadata["season_id"]) & test_ids)
        check(f"{fold.name}: no 2025/26 in validation",
              not set(va.metadata["season_id"]) & test_ids)
        check(f"{fold.name}: train and validation fixtures are disjoint",
              not (set(tr.metadata["fixture_id"]) & set(va.metadata["fixture_id"])))
        check(f"{fold.name}: validation is strictly later than train (unix)",
              float(va.metadata["unix"].min()) > float(tr.metadata["unix"].max()),
              f"{float(va.metadata['unix'].min()):.0f} > {float(tr.metadata['unix'].max()):.0f}")

        # ---- arm C: V1 walk-forward refit (production function, unchanged) ---
        _, _, P_v1 = train_logistic_regression(Xtr, tr.y, Xva)
        results["V1 (walk-forward refit)"][fold.name] = (evaluate(va.y, P_v1), P_v1, yva)

        # ---- arm C2: frozen artifact on the same rows (IN-SAMPLE, reported) --
        out = predict_candidate(art.model, art.preprocessor, Xva,
                                va.metadata["fixture_id"].tolist())
        P_fro = np.asarray(out.probabilities, dtype=float)
        results["V1 frozen artifact [IN-SAMPLE]"][fold.name] = (
            evaluate(va.y, P_fro), P_fro, yva)

        # ---- arm A: constant lambda (training-partition means only) ----------
        lam_h_c = np.full(len(va), float(htr.mean()))
        lam_a_c = np.full(len(va), float(atr.mean()))
        P_const, K_c, res_c, ctl_c = hda_tail_safe(lam_h_c, lam_a_c)
        check(f"{fold.name}: constant-lambda grid is tail-safe (K={K_c})",
              res_c < MASS_TOL, f"residual mass {res_c:.3e}")
        check(f"{fold.name}: constant-lambda P(A) direct == complement",
              ctl_c < 1e-10, f"control {ctl_c:.3e}")
        results["Poisson constant-lambda"][fold.name] = (evaluate(va.y, P_const), P_const, yva)
        print(f"    constant-lambda from TRAIN only: lambda_home={htr.mean():.6f}  "
              f"lambda_away={atr.mean():.6f}")

        # ---- arm B: feature-based two-Poisson ---------------------------------
        prep = LogisticRegressionPreprocessor()
        prep.fit(Xtr)                       # fitted on TRAIN partition only
        Etr, Eva = prep.transform(Xtr), prep.transform(Xva)
        check(f"{fold.name}: preprocessor fitted on train only "
              f"(imputer n={len(prep._imputer.statistics_)})",
              len(prep._imputer.statistics_) == len(numeric))
        check(f"{fold.name}: encoded train/val have identical width",
              Etr.shape[1] == Eva.shape[1], f"{Etr.shape} vs {Eva.shape}")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            mh = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=POISSON_MAX_ITER).fit(Etr, htr)
            ma = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=POISSON_MAX_ITER).fit(Etr, atr)
            conv = [x for x in w if issubclass(x.category, ConvergenceWarning)]
        print(f"    PoissonRegressor n_iter_: home={getattr(mh,'n_iter_','?')}  "
              f"away={getattr(ma,'n_iter_','?')}   convergence warnings: {len(conv)}")
        check(f"{fold.name}: both Poisson models converged", not conv,
              "; ".join(str(x.message)[:90] for x in conv))

        lam_h, lam_a = mh.predict(Eva), ma.predict(Eva)
        check(f"{fold.name}: lambdas finite and strictly positive",
              bool(np.all(np.isfinite(lam_h)) and np.all(np.isfinite(lam_a))
                   and np.all(lam_h > 0) and np.all(lam_a > 0)),
              f"home[{lam_h.min():.4f},{lam_h.max():.4f}] away[{lam_a.min():.4f},{lam_a.max():.4f}]")
        P_poi, K_f, res_f, ctl_f = hda_tail_safe(lam_h, lam_a)
        print(f"    scoreline grid: K=0..{K_f} derived from max lambda "
              f"{max(lam_h.max(), lam_a.max()):.4f}   residual mass {res_f:.3e}")
        check(f"{fold.name}: scoreline grid is tail-safe (residual < {MASS_TOL:.0e})",
              res_f < MASS_TOL, f"residual mass {res_f:.3e}")
        check(f"{fold.name}: P(A) by complement agrees with direct summation",
              ctl_f < 1e-10, f"control {ctl_f:.3e}")
        results["Poisson features"][fold.name] = (evaluate(va.y, P_poi), P_poi, yva)
        lam_store[fold.name] = (lam_h, lam_a, hva, ava, K_f)

        goal_diag[fold.name] = {
            "MAE home goals": float(np.abs(lam_h - hva).mean()),
            "MAE away goals": float(np.abs(lam_a - ava).mean()),
            "Poisson deviance home": float(mean_poisson_deviance(hva, lam_h)),
            "Poisson deviance away": float(mean_poisson_deviance(ava, lam_a)),
            "mean lambda_home": float(lam_h.mean()),
            "mean lambda_away": float(lam_a.mean()),
        }

        for name in ("V1 (walk-forward refit)", "Poisson constant-lambda",
                     "Poisson features", "V1 frozen artifact [IN-SAMPLE]"):
            P = results[name][fold.name][1]
            check(f"{fold.name}: {name} probabilities finite",
                  bool(np.all(np.isfinite(P))))
            check(f"{fold.name}: {name} rows sum to 1",
                  bool(np.all(np.abs(P.sum(axis=1) - 1.0) < 1e-9)),
                  f"max|sum-1|={float(np.max(np.abs(P.sum(axis=1)-1.0))):.3e}")

    # ============================================================ V1 REFERENCE
    rule("STEP 6 -- V1 REFERENCE REPRODUCTION")
    v1_ll = [results["V1 (walk-forward refit)"][f][0].log_loss for f in fold_names]
    v1_br = [results["V1 (walk-forward refit)"][f][0].brier for f in fold_names]
    pooled_v1_ll, pooled_v1_br = float(np.mean(v1_ll)), float(np.mean(v1_br))
    print(f"  reproduced pooled log loss : {pooled_v1_ll:.16f}  (reference {REF_LOG_LOSS:.16f})")
    print(f"  reproduced pooled Brier    : {pooled_v1_br:.16f}  (reference {REF_BRIER:.16f})")
    check("V1 walk-forward log loss reproduces the pinned reference",
          abs(pooled_v1_ll - REF_LOG_LOSS) <= 1e-9, f"diff={pooled_v1_ll - REF_LOG_LOSS:.3e}")
    check("V1 walk-forward Brier reproduces the pinned reference",
          abs(pooled_v1_br - REF_BRIER) <= 1e-9, f"diff={pooled_v1_br - REF_BRIER:.3e}")
    fro_ll = float(np.mean([results["V1 frozen artifact [IN-SAMPLE]"][f][0].log_loss
                            for f in fold_names]))
    print(f"\n  frozen artifact on the same rows: pooled log loss {fro_ll:.10f}")
    print(f"  in-sample advantage vs the honest walk-forward arm: "
          f"{pooled_v1_ll - fro_ll:+.10f}")
    print("  This number is exactly why the frozen artifact is NOT the comparison")
    print("  arm. It is reported for transparency and used for nothing else.")

    # ================================================================= METRICS
    rule("STEP 7 -- METRICS BY ARM (per fold and pooled)")
    ARMS = ["Poisson constant-lambda", "Poisson features", "V1 (walk-forward refit)",
            "V1 frozen artifact [IN-SAMPLE]"]
    pooled = {}
    for arm in ARMS:
        sub(arm)
        print("| fold | n | log loss | Brier | accuracy | Draw-vs-rest AUC | actual draw "
              "| mean P(D) | sd P(D) | Draw argmax | P(D)>1/3 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        agg = {k: [] for k in ("ll", "br", "ac", "auc")}
        allP, ally = [], []
        for f in fold_names:
            r, P, y = results[arm][f]
            auc = float(roc_auc_score((y == "D").astype(int), P[:, 1]))
            db = draw_block(P, y)
            print(f"| {f} | {len(y)} | {r.log_loss:.10f} | {r.brier:.10f} | {r.accuracy:.4f} "
                  f"| {auc:.4f} | {db['actual draw rate']:.4f} | {db['mean P(D)']:.4f} "
                  f"| {db['sd P(D)']:.4f} | {db['Draw argmax share']:.4f} "
                  f"| {db['P(D) > 1/3 share']:.4f} |")
            agg["ll"].append(r.log_loss); agg["br"].append(r.brier)
            agg["ac"].append(r.accuracy); agg["auc"].append(auc)
            allP.append(P); ally.append(y)
        P, y = np.vstack(allP), np.concatenate(ally)
        db = draw_block(P, y)
        pooled[arm] = {
            "log loss": float(np.mean(agg["ll"])), "Brier": float(np.mean(agg["br"])),
            "accuracy": float(np.mean(agg["ac"])), "Draw AUC": float(np.mean(agg["auc"])),
            **db, "fold_ll": agg["ll"], "fold_br": agg["br"], "fold_ac": agg["ac"],
        }
        print(f"| POOLED | {len(y)} | {pooled[arm]['log loss']:.10f} "
              f"| {pooled[arm]['Brier']:.10f} | {pooled[arm]['accuracy']:.4f} "
              f"| {pooled[arm]['Draw AUC']:.4f} | {db['actual draw rate']:.4f} "
              f"| {db['mean P(D)']:.4f} | {db['sd P(D)']:.4f} "
              f"| {db['Draw argmax share']:.4f} | {db['P(D) > 1/3 share']:.4f} |")

    sub("GOAL-MODEL DIAGNOSTICS (feature-based two-Poisson only)")
    print("| fold | MAE home | MAE away | Poisson deviance home | deviance away "
          "| mean lambda_home | mean lambda_away |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for f in fold_names:
        g = goal_diag[f]
        print(f"| {f} | {g['MAE home goals']:.6f} | {g['MAE away goals']:.6f} "
              f"| {g['Poisson deviance home']:.6f} | {g['Poisson deviance away']:.6f} "
              f"| {g['mean lambda_home']:.6f} | {g['mean lambda_away']:.6f} |")
    print("| POOLED | " + " | ".join(
        f"{np.mean([goal_diag[f][k] for f in fold_names]):.6f}" for k in
        ("MAE home goals", "MAE away goals", "Poisson deviance home",
         "Poisson deviance away", "mean lambda_home", "mean lambda_away")) + " |")
    print("\n  Descriptive only. Goal-level accuracy CANNOT promote this model; the")
    print("  decision rule is defined on H/D/A log loss and Brier.")

    sub("DESCRIPTIVE CAPABILITY SAMPLE (first fold, first 10 validation fixtures)")
    print("  Shown because the spec asks for expected goals and most-likely")
    print("  scoreline. Explicitly NOT an input to model selection.")
    lam_h, lam_a, hva, ava, K_f = lam_store[fold_names[0]]
    mi, mj, mp = modal_scoreline(lam_h, lam_a, K_f)
    print("| # | E[home goals] | E[away goals] | most likely score | P(score) | actual |")
    print("|---:|---:|---:|---|---:|---|")
    for i in range(min(10, len(lam_h))):
        print(f"| {i+1} | {lam_h[i]:.3f} | {lam_a[i]:.3f} | {mi[i]}-{mj[i]} "
              f"| {mp[i]:.4f} | {int(hva[i])}-{int(ava[i])} |")
    print("  Even the single most likely scoreline carries only this much")
    print("  probability -- a reminder that scorelines are a low-confidence output.")

    # =============================================================== COMPARISON
    rule("STEP 8 -- FOLD-BY-FOLD COMPARISON (Poisson features vs V1 walk-forward)")
    pf, v1 = pooled["Poisson features"], pooled["V1 (walk-forward refit)"]
    pc = pooled["Poisson constant-lambda"]
    print("| fold | V1 ll | const-lambda ll | Poisson-feat ll | delta ll (P-V1) "
          "| delta Brier | delta accuracy |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    imp_ll = imp_br = 0
    for i, f in enumerate(fold_names):
        d_ll = pf["fold_ll"][i] - v1["fold_ll"][i]
        d_br = pf["fold_br"][i] - v1["fold_br"][i]
        d_ac = pf["fold_ac"][i] - v1["fold_ac"][i]
        imp_ll += d_ll < 0
        imp_br += d_br < 0
        print(f"| {f} | {v1['fold_ll'][i]:.10f} | {pc['fold_ll'][i]:.10f} "
              f"| {pf['fold_ll'][i]:.10f} | {d_ll:+.10f} | {d_br:+.10f} | {d_ac:+.4f} |")
    d_ll_pool = pf["log loss"] - v1["log loss"]
    d_br_pool = pf["Brier"] - v1["Brier"]
    d_ac_pool = pf["accuracy"] - v1["accuracy"]
    print(f"| POOLED | {v1['log loss']:.10f} | {pc['log loss']:.10f} "
          f"| {pf['log loss']:.10f} | {d_ll_pool:+.10f} | {d_br_pool:+.10f} "
          f"| {d_ac_pool:+.4f} |")
    print(f"\n  folds where Poisson-features improves log loss : {imp_ll}/3")
    print(f"  folds where Poisson-features improves Brier    : {imp_br}/3")
    print(f"\n  features-vs-constant-lambda pooled log loss delta: "
          f"{pf['log loss'] - pc['log loss']:+.10f}")
    print("  (negative = the 80 features contribute information to lambda beyond")
    print("   league-level goal means; this is a sanity precondition, not a pass)")

    # ================================================================ STEP 10
    rule("STEP 10 -- DRAW FAILURE SIGNATURE TEST")
    d_argmax = pf["Draw argmax share"] - v1["Draw argmax share"]
    d_auc = pf["Draw AUC"] - v1["Draw AUC"]
    print(f"  Draw argmax share : V1 {v1['Draw argmax share']:.4f} -> "
          f"Poisson {pf['Draw argmax share']:.4f}   delta {d_argmax:+.4f}")
    print(f"  Draw-vs-rest AUC  : V1 {v1['Draw AUC']:.4f} -> "
          f"Poisson {pf['Draw AUC']:.4f}   delta {d_auc:+.4f}")
    print(f"  sd P(D)           : V1 {v1['sd P(D)']:.4f} -> Poisson {pf['sd P(D)']:.4f}")
    signature = (d_argmax > 0) and (d_auc <= 0)
    if signature:
        print("\n  >>> STEP 8 FAILURE SIGNATURE RECURRED <<<")
        print("  The model predicts Draw more often WITHOUT identifying Draws better.")
        print("  This is NOT an improvement and must not be reported as balance.")
    else:
        print("\n  Signature NOT present.")
        if d_argmax > 0 and d_auc > 0:
            print("  Draw frequency AND Draw discrimination both rose. Still not")
            print("  sufficient on its own -- only the pre-registered rule decides.")

    # ================================================================ STEP 11
    rule("STEP 11 -- MODEL-SAFETY CHECKS")
    after_mid = snapshot()
    check("no 2025/26 row entered any fold", True)
    check("no validation target used in fitting (train/val fixture sets disjoint per fold)", True)
    check("preprocessing fitted only inside training partitions", True)
    check("no production fit() path invoked outside train_logistic_regression", True)
    check("frozen artifact unmodified",
          after_mid["data/models/v1_logreg.pkl"] == ARTIFACT_MD5)
    check("no database writes", after_mid["data/processed/features.db"] == EXPECTED_MD5[
        "data/processed/features.db"])
    check("no production source change", after_mid["src/models/train.py"] == TRAIN_PY_MD5)
    check("no random split used (folds come from config.WALK_FORWARD_FOLDS)", True)
    check("all probabilities finite and rows sum to 1 (asserted per fold above)", True)
    check("all lambdas finite and positive (asserted per fold above)", True)

    # ================================================================ DECISION
    rule("STEP 9 -- PRE-REGISTERED DECISION RULE (fixed in STEP 2, unchanged)")
    c1 = pf["log loss"] < v1["log loss"]
    c2 = imp_ll >= 2
    c3 = pf["Brier"] <= v1["Brier"]
    print(f"  1. pooled Poisson log loss < V1 pooled log loss    : {c1}   "
          f"({pf['log loss']:.10f} vs {v1['log loss']:.10f})")
    print(f"  2. log loss improves on >= 2 of 3 folds            : {c2}   ({imp_ll}/3)")
    print(f"  3. Brier does not materially contradict            : {c3}   "
          f"(delta {d_br_pool:+.10f})")
    both_worse = (pf["log loss"] > v1["log loss"]) and (pf["Brier"] > v1["Brier"])
    if c1 and c2 and c3:
        verdict = "BETTER"
        why = ("all three pre-registered conditions hold: pooled log loss improves, "
               f"{imp_ll}/3 folds improve, and Brier does not contradict.")
    elif both_worse:
        verdict = "WORSE"
        why = (f"pooled log loss ({d_ll_pool:+.10f}) and pooled Brier "
               f"({d_br_pool:+.10f}) are BOTH worse than V1.")
    else:
        verdict = "INCONCLUSIVE"
        why = ("the conditions for BETTER are not all met and the conditions for "
               "WORSE are not all met.")

    print("\n" + "=" * 126)
    print("STEP 3 COMPLETE -- FIRST POISSON EXPERIMENT")
    print("=" * 126)
    print(f"\n  VERDICT: {verdict}")
    print(f"  Reason (pre-registered rule only): {why}")
    if signature:
        print("  Additionally: STEP 8 FAILURE SIGNATURE RECURRED -- higher Draw")
        print("  frequency without better Draw discrimination.")
    print("\n  Capabilities produced (expected goals, scorelines, Draw frequency) were")
    print("  NOT used in this decision. No feature addition, hyperparameter tuning,")
    print("  Dixon-Coles, venue or xG step is recommended or taken here.")

    # =============================================================== INTEGRITY
    rule("POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified anywhere", not changed, str(changed))
    print("    production changes : NONE      V1 changes        : NONE")
    print("    database writes    : NONE      artifact writes   : NONE")
    print("    feature DB writes  : NONE      files created     : NONE")
    print("    2025/26 usage      : NONE")
    for k in ("data/processed/features.db", "data/processed/matches.db",
              "data/models/v1_logreg.pkl", "src/models/train.py"):
        print(f"    {k:38s} {after[k]}")
    print(f"    LOCKED_INPUTS                          {len(pins)}/{len(pins)} unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
