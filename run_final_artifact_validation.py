"""FINAL ARTIFACT VALIDATION -- v2_poisson_venue.pkl (read-only, no evaluation).

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_final_artifact_validation.py

Validates the ALREADY-TRAINED, ALREADY-WRITTEN artifact at
data/models/v2_poisson_venue.pkl.

WHAT THIS DOES NOT DO
  * does not re-run the final holdout evaluation
  * does not read a single 2025/26 OUTCOME (no label_result, no goals)
  * does not recompute, restate or alter any holdout metric -- the previously
    obtained figures are read back OUT of the artifact and reprinted verbatim
  * does not retrain, refit, tune or calibrate anything
  * does not modify v1_logreg.pkl, either database, any production source, or
    the v2 artifact itself
Only 2025/26 FEATURE rows are used, as model input for a smoke test. Producing
a prediction is not an evaluation: nothing is compared to an outcome.

------------------------------------------------------------------------------
THE ONE CHANGE FROM THE PREVIOUS SCRIPT, AND WHY IT WAS WRONG
------------------------------------------------------------------------------
The previous run asserted bitwise equality:

    max|Ps - P[:25]| == 0.0        <- mathematically inappropriate

and failed at 2.220e-16. That is one unit in the last place of a float64 near
1.0 -- the smallest non-zero difference representable there. It is replaced by
an explicit, pre-defined numerical tolerance:

    REPRODUCIBILITY_TOL = 1e-12

Nothing else changes. No model, feature, training partition, holdout, decision
criterion or metric is touched.

------------------------------------------------------------------------------
WHAT ACTUALLY CAUSED THE 2.220e-16 -- IT WAS NOT SERIALIZATION
------------------------------------------------------------------------------
In the previous script the reference `P[:25]` was a SLICE of a prediction
computed over all 1751 holdout rows at once, while `Ps` was computed as a
standalone 25-row batch. BLAS chooses different blocking and summation orders
for different matrix shapes, so the two routes can differ by an ULP even with
identical inputs, identical weights and no serialization involved at all.

This script separates the two candidate causes instead of assuming either:

    CHECK 1  serialization determinism
             predictions from the loaded artifact vs predictions from a fresh
             pickle round-trip of that same artifact, at the SAME batch size.
             If pickling were lossy this would differ.

    CHECK 2  batch-shape sensitivity
             the same 25 rows computed standalone vs extracted from the full
             1751-row batch. This is the exact condition that produced the
             2.220e-16, reproduced deliberately and measured.

Both are reported with their observed maxima, so the origin of the difference
is demonstrated rather than asserted.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Pre-defined, stated before any measurement. ~4 orders of magnitude above
#: float64 epsilon (2.22e-16) and far below any difference that could change a
#: predicted class, a probability at reporting precision, or any metric.
REPRODUCIBILITY_TOL = 1e-12

ARTIFACT = Path("data/models/v2_poisson_venue.pkl")
V1_ARTIFACT = Path("data/models/v1_logreg.pkl")
SMOKE_ROWS = 25
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_V1_MD5 = "5e504427712b35778bb8a62a8496c7cd"
TRAIN_PY_MD5 = "21425459195311492f49e73f5ae38fe0"
STEP3_SCRIPT = "run_step_poisson_3_first_experiment.py"
STEP3_MD5 = "24827e92bc0acb04b9fc1dbe5a9958ae"
MASS_TOL, COMPLEMENT_TOL, ROWSUM_TOL = 1e-12, 1e-10, 1e-9

VENUE_VALUE_COLUMNS = (
    "home_goals_for_home_venue_season",
    "home_goals_against_home_venue_season",
    "away_goals_for_away_venue_season",
    "away_goals_against_away_venue_season",
)
REQUIRED_KEYS = (
    "model_name", "model_version", "feature_version", "feature_columns",
    "n_features", "base_contract_columns", "venue_columns", "class_order",
    "model_home_goals", "model_away_goals", "preprocessor", "estimator_config",
    "conversion", "training_seasons", "training_season_ids",
    "n_training_fixtures", "holdout_season", "holdout_evaluated_once",
    "holdout_used_in_training", "holdout_metrics", "created_utc",
)
#: Columns this script is permitted to touch. Outcome columns are absent by
#: construction, and their absence is asserted at runtime.
FORBIDDEN_COLUMNS = ("label_result", "label_home_goals", "label_away_goals")

_failures = []


def rule(t): print("\n" + "=" * 126); print(t); print("=" * 126)
def sub(t): print("\n" + "-" * 126); print(t); print("-" * 126)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def abort(msg):
    print("\n" + "!" * 126)
    print("FINAL ARTIFACT VALIDATION: FAIL")
    print(msg)
    print("!" * 126)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        _failures.append(label)
    return bool(ok)


def hard(label, ok, extra=""):
    if not check(label, ok, extra):
        abort(f"stop condition: {label}")


def snapshot():
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    for rel in ("src/models/train.py", "src/models/config.py",
                "src/features/feature_builder.py", "src/features/rolling.py",
                STEP3_SCRIPT):
        snap[rel] = md5(REPO / rel)
    for rel in (V1_ARTIFACT, ARTIFACT):
        if (REPO / rel).exists():
            snap[str(rel).replace("\\", "/")] = md5(REPO / rel)
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def main():
    rule("STEP 0 -- INTEGRITY GATE (nothing here is retrained or re-evaluated)")
    try:
        import sklearn  # noqa: F401
        from sklearn.linear_model import PoissonRegressor  # noqa: F401
    except ImportError as exc:
        abort(f"scikit-learn is required to deserialise the artifact: {exc}")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"pandas {pd.__version__}  sklearn {sklearn.__version__}")
    print(f"\n  PRE-DEFINED REPRODUCIBILITY TOLERANCE : {REPRODUCIBILITY_TOL:.0e}")
    print(f"  float64 machine epsilon               : {np.finfo(np.float64).eps:.3e}")
    print("  (stated before any measurement is taken)")

    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        hard(f"{rel} unchanged", before[rel] == exp, before[rel])
    hard("src/models/train.py unchanged", before["src/models/train.py"] == TRAIN_PY_MD5)
    hard("v1_logreg.pkl unchanged",
         before.get("data/models/v1_logreg.pkl") == ARTIFACT_V1_MD5)
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if before[f"pin:{r}"] != v["expected"]]
    hard(f"{len(pins)}/{len(pins)} LOCKED_INPUTS unchanged", not bad, str(bad))
    hard("STEP 3 conversion source at its pinned checksum",
         before[STEP3_SCRIPT] == STEP3_MD5, before[STEP3_SCRIPT])

    spec = importlib.util.spec_from_file_location("_step3", REPO / STEP3_SCRIPT)
    step3 = importlib.util.module_from_spec(spec)
    sys.modules["_step3"] = step3
    spec.loader.exec_module(step3)
    hda_tail_safe, modal_scoreline = step3.hda_tail_safe, step3.modal_scoreline
    hard("tail-safe conversion imported from the pinned source", callable(hda_tail_safe))

    # ================================================================ STEP 1
    rule("STEP 1 -- LOAD THE EXISTING ARTIFACT (it was written before the failed check)")
    ap = REPO / ARTIFACT
    hard("data/models/v2_poisson_venue.pkl exists", ap.exists(), str(ap))
    print(f"  path : {ap}")
    print(f"  size : {ap.stat().st_size} bytes")
    print(f"  md5  : {md5(ap)}")
    with open(ap, "rb") as fh:
        art = pickle.load(fh)
    hard("artifact deserialises", isinstance(art, dict), type(art).__name__)
    print(f"  keys : {len(art)}")

    # ============================================================= STEPS 6-8
    rule("STEPS 6-8 -- STRUCTURAL CHECKS")
    for k in REQUIRED_KEYS:
        hard(f"artifact contains `{k}`", k in art)
    from models.ablation import MODEL_B_COLUMNS
    from models.config import FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS

    cols = list(art["feature_columns"])
    hard("(7) contract is exactly 84 columns", len(cols) == 84, str(len(cols)))
    hard("(7) n_features metadata agrees", art["n_features"] == 84, str(art["n_features"]))
    hard("(7) contract == frozen 80 + exactly the 4 venue columns",
         set(cols) - set(MODEL_B_COLUMNS) == set(VENUE_VALUE_COLUMNS))
    hard("(7) the 80 base columns are present in contract order",
         cols[:80] == list(MODEL_B_COLUMNS))
    hard("(8) class order == ['H','D','A']", art["class_order"] == ["H", "D", "A"],
         str(art["class_order"]))
    hard("no outcome column is in the contract",
         not (set(FORBIDDEN_COLUMNS) & set(cols)))
    hard("estimator config preserved (alpha=1.0, max_iter=2000)",
         art["estimator_config"]["alpha"] == 1.0
         and art["estimator_config"]["max_iter"] == 2000,
         str(art["estimator_config"]))
    hard("conversion metadata points at the pinned STEP 3 source",
         art["conversion"]["source_md5"] == STEP3_MD5, str(art["conversion"]["source_md5"]))
    hard("training seasons are 2020/21-2024/25",
         tuple(art["training_seasons"]) == tuple(FINAL_TRAIN_SEASONS),
         str(art["training_seasons"]))
    hard("artifact records 2025/26 was NOT used in training",
         art["holdout_used_in_training"] is False)
    hard("artifact records the holdout was evaluated once",
         art["holdout_evaluated_once"] is True)
    hard("holdout season recorded as 2025/26",
         tuple(art["holdout_season"]) == tuple(FINAL_TEST_SEASONS))
    for nm, val in (("model_home_goals", art["model_home_goals"]),
                    ("model_away_goals", art["model_away_goals"])):
        hard(f"{nm} is a fitted PoissonRegressor",
             type(val).__name__ == "PoissonRegressor" and hasattr(val, "coef_"),
             type(val).__name__)
        hard(f"{nm} coefficient vector matches the encoded width",
             len(val.coef_) > 0, str(len(val.coef_)))
    hard("preprocessor is fitted",
         getattr(art["preprocessor"], "_imputer", None) is not None)
    hard("preprocessor numeric block is 83 columns (84 minus competition_id)",
         len(art["preprocessor"]._imputer.statistics_) == 83,
         str(len(art["preprocessor"]._imputer.statistics_)))

    sub("METADATA (reprinted from the artifact; no value is recomputed)")
    for k in ("model_name", "model_version", "feature_version", "n_features",
              "n_training_fixtures", "created_utc"):
        print(f"    {k:24s} {art[k]}")
    print(f"    training_seasons         {', '.join(art['training_seasons'])}")
    print(f"    venue_columns            {len(art['venue_columns'])} columns")

    sub("PREVIOUSLY OBTAINED HOLDOUT METRICS -- read from the artifact, UNCHANGED")
    hm = art["holdout_metrics"]
    for k, v in hm.items():
        print(f"    {k:12s} {v}")
    print("\n  These were produced by the completed holdout run. They are NOT")
    print("  recomputed here, and no 2025/26 outcome is read by this script.")

    # ================================================================ STEP 2
    rule("STEP 2 -- REBUILD THE SAME 25-ROW SMOKE INPUTS (FEATURES ONLY)")
    from models.data import load_supervised_dataset
    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    m_ho = ds.metadata["season_id"].isin(test_ids).values
    Xho = ds.X[m_ho][cols]
    hard("holdout feature frame carries exactly the 84 contract columns, in order",
         list(Xho.columns) == cols)
    hard("no outcome column reached the feature frame",
         not (set(FORBIDDEN_COLUMNS) & set(Xho.columns)))
    smoke = Xho.iloc[:SMOKE_ROWS]
    hard(f"smoke input is {SMOKE_ROWS} rows x 84 columns",
         smoke.shape == (SMOKE_ROWS, 84), str(smoke.shape))
    print(f"  holdout rows available : {len(Xho)}")
    print(f"  smoke rows used        : {len(smoke)} (the same first {SMOKE_ROWS} rows)")
    print("  Only FEATURE values are read. Producing a prediction is not an")
    print("  evaluation: nothing is compared against an outcome anywhere below.")

    # ================================================================ STEP 3
    rule("STEP 3 -- RECOMPUTE PREDICTIONS AFTER DESERIALIZATION")
    prep, mh, ma = art["preprocessor"], art["model_home_goals"], art["model_away_goals"]
    E = prep.transform(smoke)
    lam_h, lam_a = mh.predict(E), ma.predict(E)
    P, K, resid, ctl = hda_tail_safe(lam_h, lam_a)
    print(f"  encoded shape {E.shape}   adaptive grid K = 0..{K}")

    # ---------------------------------------------------------- STEPS 9-12
    rule("STEPS 9-12 -- PREDICTION VALIDITY")
    hard("(9) lambdas finite and strictly positive",
         bool(np.all(np.isfinite(lam_h)) and np.all(np.isfinite(lam_a))
              and np.all(lam_h > 0) and np.all(lam_a > 0)),
         f"home[{lam_h.min():.4f},{lam_h.max():.4f}] away[{lam_a.min():.4f},{lam_a.max():.4f}]")
    hard("(10) every probability row sums to 1",
         bool(np.all(np.abs(P.sum(axis=1) - 1.0) < ROWSUM_TOL)),
         f"max|sum-1| = {float(np.max(np.abs(P.sum(axis=1) - 1.0))):.3e}")
    hard("(10) all probabilities finite and in [0,1]",
         bool(np.all(np.isfinite(P)) and np.all(P >= 0) and np.all(P <= 1)))
    hard(f"(11) tail-safe conversion: residual mass < {MASS_TOL:.0e}",
         resid < MASS_TOL, f"{resid:.3e}")
    hard(f"(11) tail-safe conversion: P(A) complement vs direct < {COMPLEMENT_TOL:.0e}",
         ctl < COMPLEMENT_TOL, f"{ctl:.3e}")
    mi, mj, mp = modal_scoreline(lam_h, lam_a, K)
    hard("(12) scoreline output produced for every row",
         len(mi) == SMOKE_ROWS and len(mj) == SMOKE_ROWS and len(mp) == SMOKE_ROWS)
    hard("(12) scoreline probabilities are valid",
         bool(np.all(np.isfinite(mp)) and np.all(mp > 0) and np.all(mp <= 1)))

    sub("SAMPLE OUTPUT (features only -- no actual result is shown or read)")
    print("| # | E[home] | E[away] | P(H) | P(D) | P(A) | top score | P(score) |")
    print("|---:|---:|---:|---:|---:|---:|---|---:|")
    for i in range(min(5, SMOKE_ROWS)):
        print(f"| {i+1} | {lam_h[i]:.4f} | {lam_a[i]:.4f} | {P[i,0]:.4f} | {P[i,1]:.4f} "
              f"| {P[i,2]:.4f} | {mi[i]}-{mj[i]} | {mp[i]:.4f} |")

    # ================================================================ STEPS 4-5
    rule("STEPS 4-5 -- REPRODUCIBILITY, TOLERANCE-BASED (the corrected check)")
    print(f"  tolerance : max|diff| <= {REPRODUCIBILITY_TOL:.0e}   (NOT bitwise equality)")

    sub("CHECK 1 -- serialization determinism (same batch size, fresh round-trip)")
    buf = io.BytesIO()
    pickle.dump({"p": prep, "h": mh, "a": ma}, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    rt = pickle.load(buf)
    E2 = rt["p"].transform(smoke)
    P2, _, _, _ = hda_tail_safe(rt["h"].predict(E2), rt["a"].predict(E2))
    d_ser = float(np.max(np.abs(P2 - P)))
    print(f"    max|diff| = {d_ser:.6e}")
    hard("serialization round-trip reproduces predictions within tolerance",
         d_ser <= REPRODUCIBILITY_TOL, f"tol {REPRODUCIBILITY_TOL:.0e}")
    if d_ser == 0.0:
        print("    -> exactly 0.0. Pickling is lossless; serialization was never the")
        print("       cause of the 2.220e-16 seen in the previous run.")

    sub("CHECK 2 -- batch-shape sensitivity (the actual cause, reproduced)")
    print("    The previous script compared a standalone 25-row batch against a")
    print("    SLICE of the full 1751-row batch. Reproducing that comparison here,")
    print("    using features only:")
    E_full = prep.transform(Xho)
    P_full, _, r_full, c_full = hda_tail_safe(mh.predict(E_full), ma.predict(E_full))
    d_batch = float(np.max(np.abs(P_full[:SMOKE_ROWS] - P)))
    print(f"    max|diff| (25-row batch vs first 25 of the {len(Xho)}-row batch)"
          f" = {d_batch:.6e}")
    print(f"    float64 eps for reference                                     "
          f"= {np.finfo(np.float64).eps:.6e}")
    hard("batch-shape difference is within tolerance",
         d_batch <= REPRODUCIBILITY_TOL, f"tol {REPRODUCIBILITY_TOL:.0e}")
    hard("full-batch conversion is also tail-safe",
         r_full < MASS_TOL and c_full < COMPLEMENT_TOL,
         f"residual {r_full:.3e}, control {c_full:.3e}")

    sub("CONSEQUENCE FOR DECISIONS -- does any difference change an output?")
    pred_a = np.array(["H", "D", "A"])[P.argmax(axis=1)]
    pred_b = np.array(["H", "D", "A"])[P_full[:SMOKE_ROWS].argmax(axis=1)]
    hard("predicted class is identical for every smoke row",
         bool(np.array_equal(pred_a, pred_b)))
    hard("probabilities agree to 12 decimal places",
         bool(np.array_equal(np.round(P, 12), np.round(P_full[:SMOKE_ROWS], 12))))
    print(f"    largest observed difference {max(d_ser, d_batch):.3e} is "
          f"{max(d_ser, d_batch) / np.finfo(np.float64).eps:.1f} x float64 epsilon.")
    print("    It cannot change a predicted class, a probability at any sane")
    print("    reporting precision, or any metric. The previous exact-equality")
    print("    assertion was the defect; the model was never in question.")

    # ================================================================= RESULT
    rule("RESULT")
    print(f"  observed max absolute difference (serialization) : {d_ser:.6e}")
    print(f"  observed max absolute difference (batch shape)   : {d_batch:.6e}")
    print(f"  pre-defined tolerance                            : {REPRODUCIBILITY_TOL:.0e}")
    print(f"  previously reported failure value                : 2.220e-16")
    print(f"  structural / validity checks failed              : {len(_failures)}")
    if _failures:
        for f in _failures:
            print(f"    FAILED: {f}")
        abort("One or more artifact validations failed.")

    print("\n" + "=" * 126)
    print("FINAL ARTIFACT VALIDATION: PASS")
    print("=" * 126)
    print(f"  artifact : {ARTIFACT.as_posix()}")
    print(f"  md5      : {md5(ap)}")
    print(f"  version  : {art['model_version']}   contract {art['n_features']} columns")
    print(f"  trained on {art['n_training_fixtures']} fixtures, "
          f"{', '.join(art['training_seasons'])}")
    print("\n  The holdout evaluation was NOT re-run. 2025/26 outcomes were NOT read.")
    print("  The previously obtained holdout metrics stand exactly as recorded:")
    for k, v in hm.items():
        print(f"    {k:12s} {v}")

    # ============================================================== INTEGRITY
    rule("POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    hard("nothing modified anywhere, including the artifact itself",
         not changed, str(changed))
    print("    artifact modified     : NO (opened read-only)")
    print("    v1_logreg.pkl         : UNCHANGED")
    print("    databases             : UNCHANGED")
    print("    production source     : UNCHANGED")
    print("    retraining            : NONE")
    print("    holdout re-evaluation : NONE")
    print("    2025/26 outcomes read : NONE")
    print("    holdout metrics       : UNCHANGED (read from artifact only)")
    for k in ("data/processed/features.db", "data/models/v1_logreg.pkl",
              "data/models/v2_poisson_venue.pkl", "src/models/train.py"):
        print(f"    {k:42s} {after[k]}")
    print(f"    LOCKED_INPUTS                              {len(pins)}/{len(pins)} unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
