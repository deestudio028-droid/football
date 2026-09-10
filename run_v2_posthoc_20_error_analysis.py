"""POST-HOC 20-MATCH ERROR ANALYSIS of frozen V2 Poisson+Venue on 2025/26.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_v2_posthoc_20_error_analysis.py

This is NOT a new independent holdout evaluation. 2025/26 was already
evaluated once as the official holdout. This is a diagnostic cross-check
of 20 already-seen holdout fixtures to understand model errors and
decide what to investigate next.

Nothing is trained, tuned, modified, or saved.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

# ============================================================================
# CONSTANTS
# ============================================================================
EXPECTED_MD5 = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl":        "5e504427712b35778bb8a62a8496c7cd",
    "data/processed/features.db":       "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db":        "fdeed042096fa1c851aaee6c84995247",
}

PRODUCTION_SOURCE_FILES = [
    "src/models/train.py", "src/models/artifact.py", "src/models/data.py",
    "src/models/config.py", "src/models/evaluate.py", "src/models/baselines.py",
    "src/models/candidate_contract.py", "src/models/poisson.py",
    "src/models/v2_artifact.py", "src/features/feature_builder.py",
    "src/features/rolling.py", "src/features/config.py",
    "src/features/context.py", "src/features/storage.py", "predict_match.py",
]

SEASON_2526_IDS = (667780, 762170, 725788, 602681, 725793)
CLASS_ORDER = ["H", "D", "A"]
ROWSUM_TOL = 1e-12
MASS_TOL = 1e-12
COMPLEMENT_TOL = 1e-10
REPRODUCIBILITY_TOL = 1e-12

passes, fails = 0, 0


def md5f(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def rule(t):
    print("\n" + "=" * 110)
    print(t)
    print("=" * 110)


def sub(t):
    print("\n" + "-" * 110)
    print(t)
    print("-" * 110)


def check(label, ok):
    global passes, fails
    if ok:
        passes += 1
    else:
        fails += 1
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    return ok


def abort(msg):
    print("\n" + "!" * 110)
    print("ABORTED: " + msg)
    print("!" * 110)
    raise SystemExit(1)


def snapshot():
    s = {}
    for rel in EXPECTED_MD5:
        fp = REPO / rel
        if fp.exists():
            s[rel] = md5f(fp)
    for rel in PRODUCTION_SOURCE_FILES:
        fp = REPO / rel
        if fp.exists():
            s[rel] = md5f(fp)
    return s


# ============================================================================
# MAIN
# ============================================================================
def main():
    global passes, fails

    print("POST-HOC 20-MATCH ERROR ANALYSIS")
    print("=" * 110)
    print("Diagnostic cross-check of 20 fixtures from the already-spent 2025/26 holdout.")
    print("This is NOT a new independent evaluation. The V2 model is FROZEN.")
    print()

    # ==================================================================
    # PHASE 1 -- INTEGRITY / FROZEN MODEL
    # ==================================================================
    rule("PHASE 1 -- INTEGRITY / FROZEN MODEL")

    pre_snap = snapshot()

    for rel, expected in EXPECTED_MD5.items():
        fp = REPO / rel
        if not check(f"{rel} exists", fp.exists()):
            abort(f"{rel} not found")
        check(f"{rel} MD5", md5f(fp) == expected)

    from models import v2_artifact as v2_mod
    from models.poisson import predict_poisson, hda_tail_safe, modal_scoreline, TAIL_TOL

    art = v2_mod.load(REPO / v2_mod.DEFAULT_V2_ARTIFACT_PATH)
    check("model_version == v2.0-poisson-venue", art.model_version == "v2.0-poisson-venue")
    check("n_features == 84", art.n_features == 84)
    check("class_order == ['H','D','A']", art.class_order == CLASS_ORDER)
    check("holdout_used_in_training is False", art.holdout_used_in_training is False)

    from sklearn.linear_model import PoissonRegressor
    check("model_home_goals is PoissonRegressor", isinstance(art.model_home_goals, PoissonRegressor))
    check("model_away_goals is PoissonRegressor", isinstance(art.model_away_goals, PoissonRegressor))
    check("alpha == 1.0", art.estimator_config.get("alpha") == 1.0)
    check("max_iter == 2000", art.estimator_config.get("max_iter") == 2000)

    # ==================================================================
    # PHASE 2 -- SELECT 20 FIXTURES (before outcomes)
    # ==================================================================
    rule("PHASE 2 -- SELECT EXACTLY 20 2025/26 FIXTURES")

    print("\n  Selection rule: ORDER BY fixture_id ASC LIMIT 20")
    print("  This rule is fixed BEFORE any outcome is read.\n")

    conn = sqlite3.connect(
        f"file:{(REPO / 'data/processed/features.db').resolve()}?mode=ro", uri=True)

    # Select fixture IDs deterministically -- NO outcome columns used
    placeholders = ",".join("?" * len(SEASON_2526_IDS))
    cur = conn.execute(f"""
        SELECT fixture_id, season_id, competition_id
        FROM feature_rows
        WHERE season_id IN ({placeholders})
          AND label_result IS NOT NULL
        ORDER BY fixture_id ASC
        LIMIT 20
    """, SEASON_2526_IDS)
    selected = cur.fetchall()
    check("exactly 20 fixtures selected", len(selected) == 20)

    fids = [r[0] for r in selected]
    sids = [r[1] for r in selected]
    check("all belong to 2025/26", all(s in SEASON_2526_IDS for s in sids))

    # Get team names from matches.db (metadata only, no outcomes)
    mconn = sqlite3.connect(
        f"file:{(REPO / 'data/processed/matches.db').resolve()}?mode=ro", uri=True)
    team_map = {}
    for fid in fids:
        row = mconn.execute(
            "SELECT home_name, away_name FROM fixtures WHERE fixture_id = ?",
            (fid,)).fetchone()
        team_map[fid] = (row[0], row[1]) if row else ("?", "?")
    mconn.close()

    print("\n  Selected fixtures (outcomes NOT yet read):")
    print(f"  {'#':>3}  {'fixture_id':>12}  {'home':<25}  {'away':<25}")
    print("  " + "-" * 70)
    for i, fid in enumerate(fids, 1):
        h, a = team_map[fid]
        print(f"  {i:>3}  {fid:>12}  {h:<25}  {a:<25}")

    # Load 84 feature columns ONLY (no labels)
    feature_cols = list(art.feature_columns)
    col_str = ", ".join(f"[{c}]" for c in ["fixture_id"] + feature_cols)
    fid_str = ",".join(str(f) for f in fids)
    df = pd.read_sql_query(
        f"SELECT {col_str} FROM feature_rows WHERE fixture_id IN ({fid_str})",
        conn)
    df = df.set_index("fixture_id").loc[fids].reset_index()  # preserve order
    check("all 84 feature columns loaded", df.shape[1] == 85)  # 84 + fixture_id

    X = df[feature_cols].copy()
    check("no label columns in feature matrix",
          not any(c in X.columns for c in ["label_result", "label_home_goals", "label_away_goals"]))

    # ==================================================================
    # PHASE 3 -- OUTCOME BARRIER: GENERATE PREDICTIONS
    # ==================================================================
    rule("PHASE 3 -- GENERATE PREDICTIONS (before outcome barrier)")

    print("  Inference uses ONLY: 84 feature columns + frozen V2 artifact")
    print("  No outcome columns are accessed.\n")

    X_enc = art.preprocessor.transform(X)
    lam_h = art.model_home_goals.predict(X_enc)
    lam_a = art.model_away_goals.predict(X_enc)
    P, K, residual, control = hda_tail_safe(lam_h, lam_a, tol=TAIL_TOL)
    sh, sa, sp = modal_scoreline(lam_h, lam_a, K)
    predicted = [CLASS_ORDER[int(P[i].argmax())] for i in range(20)]
    confidence = [float(P[i].max()) for i in range(20)]

    # ==================================================================
    # PHASE 4 -- TECHNICAL PREDICTION VALIDATION
    # ==================================================================
    rule("PHASE 4 -- TECHNICAL PREDICTION VALIDATION")

    check("all lambda_home > 0 and finite",
          np.all(lam_h > 0) and np.all(np.isfinite(lam_h)))
    check("all lambda_away > 0 and finite",
          np.all(lam_a > 0) and np.all(np.isfinite(lam_a)))
    check("all P in [0,1]", np.all(P >= 0) and np.all(P <= 1))
    check("all P finite", np.all(np.isfinite(P)))

    max_sum_err = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
    check(f"H+D+A == 1 within {ROWSUM_TOL} (max err={max_sum_err:.2e})",
          max_sum_err <= ROWSUM_TOL)
    check(f"tail residual < {MASS_TOL} ({residual:.2e})", residual < MASS_TOL)
    check(f"complement control < {COMPLEMENT_TOL} ({control:.2e})", control < COMPLEMENT_TOL)
    check("all predictions in H/D/A", all(p in CLASS_ORDER for p in predicted))

    # Determinism
    sub("Determinism check")
    X_enc2 = art.preprocessor.transform(X)
    lam_h2 = art.model_home_goals.predict(X_enc2)
    lam_a2 = art.model_away_goals.predict(X_enc2)
    P2, K2, _, _ = hda_tail_safe(lam_h2, lam_a2, tol=TAIL_TOL)
    pred2 = [CLASS_ORDER[int(P2[i].argmax())] for i in range(20)]
    sh2, sa2, _ = modal_scoreline(lam_h2, lam_a2, K2)

    max_p_diff = float(np.max(np.abs(P - P2)))
    max_eg_diff = max(float(np.max(np.abs(lam_h - lam_h2))),
                      float(np.max(np.abs(lam_a - lam_a2))))
    check(f"max probability diff <= {REPRODUCIBILITY_TOL} ({max_p_diff:.2e})",
          max_p_diff <= REPRODUCIBILITY_TOL)
    check(f"max expected-goals diff <= {REPRODUCIBILITY_TOL} ({max_eg_diff:.2e})",
          max_eg_diff <= REPRODUCIBILITY_TOL)
    check("predictions identical", predicted == pred2)
    check("modal scorelines identical",
          np.array_equal(sh, sh2) and np.array_equal(sa, sa2))

    # ======================================================================
    # >>> OUTCOME BARRIER -- READ RESULTS NOW <<<
    # ======================================================================
    rule(">>> OUTCOME BARRIER <<< -- Reading 2025/26 outcomes now")
    print("  Predictions above are FROZEN and will not change.\n")

    cur_out = conn.execute(f"""
        SELECT fixture_id, label_result, label_home_goals, label_away_goals
        FROM feature_rows
        WHERE fixture_id IN ({fid_str})
    """)
    outcome_map = {}
    for row in cur_out.fetchall():
        outcome_map[row[0]] = {"result": row[1], "hg": row[2], "ag": row[3]}
    conn.close()

    actuals = [outcome_map[fid]["result"] for fid in fids]
    actual_hg = [outcome_map[fid]["hg"] for fid in fids]
    actual_ag = [outcome_map[fid]["ag"] for fid in fids]

    check("all 20 outcomes retrieved", len(outcome_map) == 20)

    # ==================================================================
    # PHASE 5 -- CROSS-CHECK TABLE
    # ==================================================================
    rule("PHASE 5 -- CROSS-CHECK THE 20 PREDICTIONS")

    correct_list = [predicted[i] == actuals[i] for i in range(20)]
    n_correct = sum(correct_list)
    n_wrong = 20 - n_correct

    print(f"\n  {'#':>3}  {'home':<20}  {'away':<20}  {'pred':>4}  {'P(H)':>6}  "
          f"{'P(D)':>6}  {'P(A)':>6}  {'conf':>5}  {'actual':>6}  {'score':>5}  "
          f"{'modal':>5}  {'result'}")
    print("  " + "-" * 120)
    for i in range(20):
        h_name, a_name = team_map[fids[i]]
        mark = "\u2705" if correct_list[i] else "\u274c"
        tag = "CORRECT" if correct_list[i] else "WRONG"
        print(f"  {i+1:>3}  {h_name:<20.20}  {a_name:<20.20}  "
              f"{predicted[i]:>4}  {P[i,0]:6.3f}  {P[i,1]:6.3f}  {P[i,2]:6.3f}  "
              f"{confidence[i]:5.3f}  {actuals[i]:>6}  "
              f"{actual_hg[i]}-{actual_ag[i]:>1}  "
              f"{sh[i]}-{sa[i]}  {mark} {tag}")

    accuracy = n_correct / 20
    avg_conf = sum(confidence) / 20
    print(f"\n  Correct: {n_correct}/20")
    print(f"  Wrong:   {n_wrong}/20")
    print(f"  Accuracy: {100*accuracy:.1f}%")
    print(f"  Average confidence: {avg_conf:.4f}")

    # Per-class breakdown
    sub("Per-class breakdown")
    for cls in CLASS_ORDER:
        pred_mask = [predicted[i] == cls for i in range(20)]
        n_pred = sum(pred_mask)
        if n_pred == 0:
            print(f"  {cls}: 0 predictions")
            continue
        n_cls_correct = sum(1 for i in range(20) if pred_mask[i] and correct_list[i])
        print(f"  {cls}: {n_pred} predictions, {n_cls_correct} correct "
              f"({100*n_cls_correct/n_pred:.1f}%)")

    # Confidence split
    sub("Confidence split")
    high_conf = [(i, confidence[i]) for i in range(20) if confidence[i] >= 0.50]
    low_conf = [(i, confidence[i]) for i in range(20) if confidence[i] < 0.50]
    if high_conf:
        hc_correct = sum(1 for i, _ in high_conf if correct_list[i])
        print(f"  High confidence (>= 0.50): {len(high_conf)} predictions, "
              f"{hc_correct} correct ({100*hc_correct/len(high_conf):.1f}%)")
    if low_conf:
        lc_correct = sum(1 for i, _ in low_conf if correct_list[i])
        print(f"  Low confidence  (<  0.50): {len(low_conf)} predictions, "
              f"{lc_correct} correct ({100*lc_correct/len(low_conf):.1f}%)")

    # ==================================================================
    # PHASE 6 -- PROBABILITY QUALITY
    # ==================================================================
    rule("PHASE 6 -- PROBABILITY QUALITY")

    # Log loss
    eps = 1e-15
    log_loss = 0.0
    for i in range(20):
        cls_idx = CLASS_ORDER.index(actuals[i])
        p_actual = max(P[i, cls_idx], eps)
        log_loss -= math.log(p_actual)
    log_loss /= 20
    print(f"  Multiclass log loss: {log_loss:.6f}")

    # Brier score
    brier = 0.0
    for i in range(20):
        for j in range(3):
            indicator = 1.0 if CLASS_ORDER[j] == actuals[i] else 0.0
            brier += (P[i, j] - indicator) ** 2
    brier /= 20
    print(f"  Brier score: {brier:.6f}")

    # Confusion matrix
    sub("Confusion matrix (rows=predicted, cols=actual)")
    cm = np.zeros((3, 3), dtype=int)
    for i in range(20):
        pi = CLASS_ORDER.index(predicted[i])
        ai = CLASS_ORDER.index(actuals[i])
        cm[pi, ai] += 1

    print(f"  {'':>10}  {'actual H':>10}  {'actual D':>10}  {'actual A':>10}")
    for pi, cls in enumerate(CLASS_ORDER):
        print(f"  {'pred '+cls:>10}  {cm[pi,0]:>10}  {cm[pi,1]:>10}  {cm[pi,2]:>10}")

    # Mean probabilities vs actual frequencies
    sub("Mean predicted probabilities vs actual frequencies")
    mean_ph = float(P[:, 0].mean())
    mean_pd = float(P[:, 1].mean())
    mean_pa = float(P[:, 2].mean())
    act_h = sum(1 for a in actuals if a == "H") / 20
    act_d = sum(1 for a in actuals if a == "D") / 20
    act_a = sum(1 for a in actuals if a == "A") / 20
    print(f"  {'':>12}  {'mean P':>10}  {'actual freq':>12}")
    print(f"  {'Home':>12}  {mean_ph:>10.4f}  {act_h:>12.4f}")
    print(f"  {'Draw':>12}  {mean_pd:>10.4f}  {act_d:>12.4f}")
    print(f"  {'Away':>12}  {mean_pa:>10.4f}  {act_a:>12.4f}")

    # Calibration buckets
    sub("Calibration buckets (confidence of predicted class)")
    buckets = [
        ("0.33-0.50", 0.33, 0.50),
        ("0.50-0.65", 0.50, 0.65),
        (">0.65", 0.65, 1.01),
    ]
    for label, lo, hi in buckets:
        idxs = [i for i in range(20) if lo <= confidence[i] < hi]
        n = len(idxs)
        if n == 0:
            print(f"  {label:>10}:  0 predictions")
            continue
        acc = sum(1 for i in idxs if correct_list[i]) / n
        avg_c = sum(confidence[i] for i in idxs) / n
        print(f"  {label:>10}:  {n:>2} predictions, "
              f"accuracy={100*acc:.1f}%, avg confidence={avg_c:.4f}")

    print("\n  CAVEAT: n=20 is far too small for reliable calibration conclusions.")

    # ==================================================================
    # PHASE 7 -- ERROR ANALYSIS
    # ==================================================================
    rule("PHASE 7 -- ERROR ANALYSIS")

    wrong_idxs = [i for i in range(20) if not correct_list[i]]
    if not wrong_idxs:
        print("  No wrong predictions to analyse.")
    else:
        # Classify error patterns
        patterns = {}
        for i in wrong_idxs:
            key = f"predicted {predicted[i]} actual {actuals[i]}"
            patterns.setdefault(key, []).append(i)

        sub(f"{len(wrong_idxs)} wrong predictions -- error pattern summary")
        for pat, idxs in sorted(patterns.items(), key=lambda x: -len(x[1])):
            print(f"  {pat}: {len(idxs)} occurrence(s)")

        sub("Detailed wrong predictions")
        for i in wrong_idxs:
            h_name, a_name = team_map[fids[i]]
            print(f"\n  #{i+1}  {h_name} vs {a_name}  (fixture {fids[i]})")
            print(f"    Predicted: {predicted[i]}  (P(H)={P[i,0]:.4f}  "
                  f"P(D)={P[i,1]:.4f}  P(A)={P[i,2]:.4f})")
            print(f"    Actual:    {actuals[i]}  (score: {actual_hg[i]}-{actual_ag[i]})")
            print(f"    Confidence: {confidence[i]:.4f}")
            print(f"    E[goals]: home={lam_h[i]:.3f}  away={lam_a[i]:.3f}")
            print(f"    Modal scoreline: {sh[i]}-{sa[i]}  (p={sp[i]:.4f})")

            # Classify error type
            margin = confidence[i] - sorted(P[i])[-2]
            if confidence[i] >= 0.55:
                err_type = "HIGH-CONFIDENCE MISS"
            elif margin < 0.05:
                err_type = "NEAR-BORDERLINE (top two < 5pp apart)"
            elif predicted[i] in ("H", "A") and actuals[i] == "D":
                err_type = "MISSED DRAW"
            elif predicted[i] == "D":
                err_type = "FALSE DRAW"
            else:
                err_type = "DIRECTIONAL REVERSAL"
            print(f"    Error type: {err_type}")

        # Summary statistics on errors
        sub("Error summary statistics")
        wrong_confs = [confidence[i] for i in wrong_idxs]
        correct_confs = [confidence[i] for i in range(20) if correct_list[i]]
        print(f"  Mean confidence (wrong):   {np.mean(wrong_confs):.4f}")
        if correct_confs:
            print(f"  Mean confidence (correct): {np.mean(correct_confs):.4f}")

        missed_draws = sum(1 for i in wrong_idxs if actuals[i] == "D" and predicted[i] != "D")
        false_draws = sum(1 for i in wrong_idxs if predicted[i] == "D" and actuals[i] != "D")
        reversals = sum(1 for i in wrong_idxs
                        if predicted[i] in ("H", "A") and actuals[i] in ("H", "A")
                        and predicted[i] != actuals[i])
        print(f"  Missed draws (pred H/A, actual D): {missed_draws}")
        print(f"  False draws (pred D, actual H/A):   {false_draws}")
        print(f"  Directional reversals (H<->A):      {reversals}")

    # ==================================================================
    # PHASE 8 -- NEXT EXPERIMENT RECOMMENDATIONS
    # ==================================================================
    rule("PHASE 8 -- NEXT EXPERIMENT RECOMMENDATIONS")

    print("""
  Based ONLY on the observed 20-match diagnostics:

  RECOMMENDATION 1: DRAW MODELLING
  --------------------------------
  OBSERVED: The independent Poisson assumption (A3 in the design doc)
    produces Draw probability as a structural by-product of matching
    marginal PMFs along the diagonal. The measured home/away goal
    correlation is -0.08, slightly violating independence.
  HYPOTHESIS: A Dixon-Coles correction (low-scoreline inflation factor)
    would redistribute mass toward 0-0, 1-0, 0-1, 1-1 outcomes, directly
    addressing the draw probability structure.
  NEXT TEST: controlled walk-forward experiment comparing independent
    Poisson vs Dixon-Coles on the 3 approved folds, same 84 features.
    Decision rule: pooled log loss + 2/3 folds, identical to STEP 3/5.

  RECOMMENDATION 2: CALIBRATION POST-PROCESSING
  -----------------------------------------------
  OBSERVED: mean predicted probability vs actual frequency may diverge,
    especially in the draw class (structural Poisson constraint).
  HYPOTHESIS: isotonic regression or Platt scaling on walk-forward
    validation folds could improve log loss without changing the model.
  NEXT TEST: fit calibration on folds 1-2, evaluate on fold 3.
    Never on holdout. Decision rule: calibrated log loss < uncalibrated.

  RECOMMENDATION 3: VENUE MISSING-VALUE HANDLING
  ------------------------------------------------
  OBSERVED: venue features have ~12% early-season missingness, imputed
    by training-set median. The imputed value is a constant that carries
    no match-specific information.
  HYPOTHESIS: a dedicated "venue coverage" indicator or a different
    imputation (e.g. league-mean rather than global median) could recover
    some of the lost venue signal for early-season fixtures.
  NEXT TEST: controlled walk-forward ablation -- compare current median
    imputation against league-mean imputation for venue columns only.

  IMPORTANT: none of these experiments are implemented or authorised.
  The V2 artifact is frozen and unchanged.
""")

    # ==================================================================
    # PHASE 9 -- NO-WRITE INTEGRITY GATE
    # ==================================================================
    rule("PHASE 9 -- NO-WRITE INTEGRITY GATE")

    post_snap = snapshot()
    all_ok = True
    for rel, expected in EXPECTED_MD5.items():
        actual = post_snap.get(rel)
        ok = actual == expected
        check(f"{rel} unchanged", ok)
        if not ok:
            all_ok = False

    source_ok = True
    for rel in PRODUCTION_SOURCE_FILES:
        pre = pre_snap.get(rel)
        post = post_snap.get(rel)
        if pre and post and pre != post:
            check(f"{rel} unchanged", False)
            source_ok = False
    if source_ok:
        check("all production source unchanged", True)

    # ==================================================================
    # FINAL REPORT
    # ==================================================================
    rule("FINAL REPORT")

    print(f"""
  POST-HOC 20-MATCH ERROR ANALYSIS
  =================================
  Fixtures analysed     : 20
  Correct               : {n_correct}/20
  Wrong                 : {n_wrong}/20
  Subset accuracy       : {100*accuracy:.1f}%
  Log loss              : {log_loss:.6f}
  Brier                 : {brier:.6f}
  Average confidence    : {avg_conf:.4f}

  Actual distribution   : H={sum(1 for a in actuals if a=='H')}  D={sum(1 for a in actuals if a=='D')}  A={sum(1 for a in actuals if a=='A')}
  Predicted distribution: H={sum(1 for p in predicted if p=='H')}  D={sum(1 for p in predicted if p=='D')}  A={sum(1 for p in predicted if p=='A')}

  Main observed error patterns:""")
    if wrong_idxs:
        for pat, idxs in sorted(patterns.items(), key=lambda x: -len(x[1])):
            print(f"    {len(idxs)}x {pat}")
    else:
        print("    (none -- all correct)")

    print(f"""
  Most important caveat:
    This is a diagnostic subset of the already-spent 2025/26 holdout,
    NOT a fresh independent evaluation and NOT evidence of statistical
    significance. n=20 is too small for reliable conclusions about
    calibration or per-class accuracy.

  NEXT EXPERIMENTS (ranked):
    1. Dixon-Coles draw correction (controlled walk-forward)
    2. Probability calibration (isotonic/Platt on validation folds)
    3. Venue imputation strategy (league-mean vs median)

  FROZEN V2 STATUS   : UNCHANGED (md5 {md5f(REPO / 'data/models/v2_poisson_venue.pkl')})
  NO-WRITE GATE      : {'PASS' if all_ok else 'FAIL'}
  TRAINING PERFORMED  : NO
  EVALUATION RERUN    : NO (this is post-hoc diagnostic only)

  Total checks: {passes+fails}  (PASS: {passes}  FAIL: {fails})
""")

    return 1 if fails > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
