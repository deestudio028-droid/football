"""H1 -- coefficient / counterfactual attribution (validation folds ONLY).

QUESTION
    Does removing the four alias columns change meaningful LEARNED
    COEFFICIENTS, or is the observed -2.915e-05 mean validation log-loss
    gain numerical noise?

METHOD
    Fit both arms (80-col control and 76-col H1, both C=0.0005) on each
    walk-forward fold, extract the actual fitted `coef_`/`intercept_`, and
    test four predictions derived ALGEBRAICALLY BEFORE any fit:

      P1  in the 80-col fit the two members of each alias pair have nearly
          EQUAL coefficients (ridge splits a duplicated signal evenly)
      P2  the 76-col coefficient approximates the SUM of the 80-col pair,
          but is SMALLER in magnitude (it now pays the full penalty)
      P3  the other 72 columns move only slightly
      P4  the four affected signals carry LESS total weight at 76 columns

    Ridge algebra behind them: for a duplicated column, penalty is
    2*(s/2)^2 = s^2/2 versus s^2 for a single column carrying the same
    signal s. De-duplication therefore DOUBLES the effective penalty on
    that signal. If P1/P2 fail, that mechanism is not what happened.

NOISE-FLOOR CONTROL (the decisive test)
    scikit-learn's LogisticRegression stops at tol=1e-4 by default. The
    observed gain is 2.915e-05 -- at or below that scale. Both arms are
    therefore refitted at tol=1e-10. If the gain vanishes, flips sign, or
    changes materially, it was optimiser noise rather than a mechanism.
    The tightened tol is a DIAGNOSTIC PROBE ONLY and is never proposed as
    a model change.

Nothing is modified. No 2025/26 access. Writes nothing; stdout only.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_h1_coefficient_attribution.py
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

from models.ablation import MODEL_B_COLUMNS
from models.baselines import CLASS_ORDER
from models.config import FINAL_TEST_SEASONS, MODEL_VERSION, WALK_FORWARD_FOLDS
from models.data import load_supervised_dataset
from models.evaluate import evaluate, validate_probabilities
from models.robustness import V1_LOGREG_BASE_KWARGS
from models.splits import iter_walk_forward_folds

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"

C_VALUE = 0.0005
ALIASES = ("home_goal_diff_last5", "home_goal_diff_last10",
           "away_goal_diff_last5", "away_goal_diff_last10")
TWIN = {"home_goal_diff_last5": "home_goals_diff_last5",
        "home_goal_diff_last10": "home_goals_diff_last10",
        "away_goal_diff_last5": "away_goals_diff_last5",
        "away_goal_diff_last10": "away_goals_diff_last10"}
H1_COLUMNS = tuple(c for c in MODEL_B_COLUMNS if c not in ALIASES)
OBSERVED_H1_GAIN = -2.915e-05          # quoted from the completed H1 run


def rule(t: str) -> None:
    print("\n" + "=" * 78); print(t); print("=" * 78)


def stop(msg: str) -> None:
    print("\n" + "!" * 78); print("STOP"); print(msg); print("!" * 78)
    raise SystemExit(1)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def fit_arm(X_train, y_train, X_val, columns, tol=None):
    """Returns (P_val, model, preprocessor, encoded_feature_names)."""
    from sklearn.linear_model import LogisticRegression
    from models import train as train_module

    Xtr = X_train[list(columns)]
    Xva = X_val[list(columns)]
    pre = train_module.LogisticRegressionPreprocessor().fit(Xtr)
    Etr, Eva = pre.transform(Xtr), pre.transform(Xva)
    kwargs = dict(V1_LOGREG_BASE_KWARGS)
    if tol is not None:
        kwargs["tol"] = tol
    model = LogisticRegression(C=C_VALUE, **kwargs)
    model.fit(Etr, y_train)
    P = np.asarray(train_module._reorder_proba(model, Eva), dtype=float)
    validate_probabilities(P)

    # Exact encoded-column names, read from the preprocessor's own state.
    # transform() returns hstack([scaled(numeric_columns), onehot(competition)]),
    # so the mapping is exact rather than assumed (see train.py transform()).
    names = list(pre.numeric_columns) + [
        f"competition_id=={cat}" for cat in pre.competition_categories
    ]
    if len(names) != Etr.shape[1]:
        raise RuntimeError(
            f"encoded-name mapping mismatch: derived {len(names)} names for "
            f"{Etr.shape[1]} encoded columns; refusing to guess."
        )
    return P, model, pre, names, Eva


def coef_series(model, names) -> pd.DataFrame:
    """coef_ rows follow model.classes_ (alphabetical A, D, H); relabel explicitly."""
    df = pd.DataFrame(model.coef_, index=list(model.classes_), columns=names)
    return df.reindex(CLASS_ORDER)


def main() -> None:
    print("H1 COEFFICIENT / COUNTERFACTUAL ATTRIBUTION -- validation folds only")
    print(f"Control: 80 columns, C={C_VALUE}   |   H1: 76 columns, C={C_VALUE}")

    rule("PRE-FLIGHT")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION {MODEL_VERSION!r} | ridge strength 1/C = {1/C_VALUE:.0f}")
    seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    if seasons & set(FINAL_TEST_SEASONS):
        stop("2025/26 reachable")
    print(f"  fold seasons {sorted(seasons)} | 2025/26 reachable: False")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines: mismatches={bad or 'NONE'}")
    if bad:
        stop(f"pinned mismatch {bad}")

    rule("ALGEBRAIC PREDICTIONS (fixed before any fit)")
    print("  P1 80-col pair members have nearly EQUAL coefficients")
    print("  P2 76-col coefficient ~ SUM of the 80-col pair, but SMALLER in magnitude")
    print("  P3 the other 72 columns move only slightly")
    print("  P4 the four affected signals carry LESS total weight at 76 columns")
    print("  Basis: duplicated-column ridge penalty s^2/2 vs single-column s^2,")
    print("         so de-duplication doubles the effective penalty on that signal.")

    dataset = load_supervised_dataset(FEATURES_DB)
    results = {}

    rule("FIT BOTH ARMS PER FOLD (default tol)")
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        P80, m80, _, n80, _ = fit_arm(train_ds.X, train_ds.y, val_ds.X, MODEL_B_COLUMNS)
        P76, m76, _, n76, _ = fit_arm(train_ds.X, train_ds.y, val_ds.X, H1_COLUMNS)
        r80, r76 = evaluate(val_ds.y, P80), evaluate(val_ds.y, P76)
        results[fold.name] = dict(P80=P80, P76=P76, m80=m80, m76=m76,
                                  c80=coef_series(m80, n80), c76=coef_series(m76, n76),
                                  r80=r80, r76=r76, y=val_ds.y.to_numpy())
        print(f"  {fold.name}: log_loss 80col={r80.log_loss:.16f}  76col={r76.log_loss:.16f}  "
              f"delta={r76.log_loss - r80.log_loss:+.3e}")
        print(f"           n_iter_ 80col={m80.n_iter_}  76col={m76.n_iter_}  "
              f"(max_iter={V1_LOGREG_BASE_KWARGS['max_iter']})")

    mean80 = statistics.fmean(v["r80"].log_loss for v in results.values())
    mean76 = statistics.fmean(v["r76"].log_loss for v in results.values())
    print(f"\n  MEAN log loss  80col={mean80:.16f}  76col={mean76:.16f}  "
          f"delta={mean76 - mean80:+.6e}")
    print(f"  reported H1 gain from the earlier run: {OBSERVED_H1_GAIN:+.6e}  "
          f"| reproduced here: {abs((mean76 - mean80) - OBSERVED_H1_GAIN) < 1e-8}")

    conv = [(n, v["m80"].n_iter_, v["m76"].n_iter_) for n, v in results.items()]
    hit_max = any(int(np.max(a)) >= V1_LOGREG_BASE_KWARGS["max_iter"]
                  or int(np.max(b)) >= V1_LOGREG_BASE_KWARGS["max_iter"] for _, a, b in conv)
    print(f"  any arm hit max_iter (=> unconverged, differences meaningless): {hit_max}")

    rule("P1 / P2 -- ALIAS PAIR COEFFICIENTS")
    for name, v in results.items():
        print(f"\n  {name}:")
        for alias in ALIASES:
            twin = TWIN[alias]
            print(f"    signal '{twin}'")
            for cls in CLASS_ORDER:
                a = float(v["c80"].loc[cls, alias]) if alias in v["c80"].columns else np.nan
                t = float(v["c80"].loc[cls, twin])
                s = float(v["c76"].loc[cls, twin])
                total80 = a + t
                p1 = abs(a - t)
                print(f"      {cls}: 80col alias={a:+.8e} twin={t:+.8e} "
                      f"|diff|={p1:.2e} sum={total80:+.8e} | 76col={s:+.8e} "
                      f"| 76col-sum80={s - total80:+.2e} | |76col|<|sum80|="
                      f"{abs(s) < abs(total80)}")

    rule("P1 / P2 SUMMARY ACROSS FOLDS AND CLASSES")
    p1_diffs, p2_ratio, shrunk = [], [], []
    for v in results.values():
        for alias in ALIASES:
            twin = TWIN[alias]
            for cls in CLASS_ORDER:
                a = float(v["c80"].loc[cls, alias]); t = float(v["c80"].loc[cls, twin])
                s = float(v["c76"].loc[cls, twin]); tot = a + t
                p1_diffs.append(abs(a - t) / (abs(tot) + 1e-30))
                if abs(tot) > 1e-30:
                    p2_ratio.append(s / tot)
                shrunk.append(abs(s) < abs(tot))
    print(f"  P1 relative |alias-twin| gap: mean={np.mean(p1_diffs):.4e} "
          f"max={np.max(p1_diffs):.4e}   (near 0 => ridge split the signal evenly)")
    print(f"  P2 ratio 76col / (sum of 80col pair): mean={np.mean(p2_ratio):.6f} "
          f"min={np.min(p2_ratio):.6f} max={np.max(p2_ratio):.6f}")
    print(f"     (a ratio below 1 is the predicted extra shrinkage; ~1 means no extra penalty)")
    print(f"  P4 fraction of cases where |76col| < |sum of 80col pair|: {np.mean(shrunk):.4f}")

    rule("P3 -- MOVEMENT OF THE OTHER COLUMNS")
    for name, v in results.items():
        shared = [c for c in v["c76"].columns if c in v["c80"].columns and c not in ALIASES]
        d = (v["c76"][shared] - v["c80"][shared]).abs()
        scale = v["c80"][shared].abs().to_numpy().max()
        print(f"  {name}: shared columns={len(shared)}  max|delta|={d.to_numpy().max():.3e}  "
              f"mean|delta|={d.to_numpy().mean():.3e}  (largest 80col |coef| = {scale:.3e})")
        worst = d.max().sort_values(ascending=False).head(5)
        print(f"    largest movers: {[(c, f'{worst[c]:.2e}') for c in worst.index]}")
        print(f"    intercepts: 80col={np.asarray(v['m80'].intercept_)} "
              f"76col={np.asarray(v['m76'].intercept_)}")

    rule("EFFECTIVE TOTAL WEIGHT ON THE FOUR AFFECTED SIGNALS")
    for name, v in results.items():
        t80 = sum(abs(float(v["c80"].loc[cls, a])) + abs(float(v["c80"].loc[cls, TWIN[a]]))
                  for a in ALIASES for cls in CLASS_ORDER)
        t76 = sum(abs(float(v["c76"].loc[cls, TWIN[a]])) for a in ALIASES for cls in CLASS_ORDER)
        print(f"  {name}: sum|coef| on those signals  80col={t80:.6e}  76col={t76:.6e}  "
              f"ratio={t76 / t80 if t80 else float('nan'):.6f}")

    rule("NOISE-FLOOR CONTROL -- refit both arms at tol=1e-10 (DIAGNOSTIC ONLY)")
    print("  Default tol=1e-4 is larger than the observed gain, so the gain must be")
    print("  shown to survive tighter convergence before it can be called a mechanism.")
    tight = {}
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        P80, m80, _, _, _ = fit_arm(train_ds.X, train_ds.y, val_ds.X, MODEL_B_COLUMNS, tol=1e-10)
        P76, m76, _, _, _ = fit_arm(train_ds.X, train_ds.y, val_ds.X, H1_COLUMNS, tol=1e-10)
        a, b = evaluate(val_ds.y, P80).log_loss, evaluate(val_ds.y, P76).log_loss
        tight[fold.name] = (a, b)
        print(f"  {fold.name}: 80col={a:.16f}  76col={b:.16f}  delta={b - a:+.6e}  "
              f"n_iter 80={m80.n_iter_} 76={m76.n_iter_}")
    t80m = statistics.fmean(a for a, _ in tight.values())
    t76m = statistics.fmean(b for _, b in tight.values())
    tight_gain = t76m - t80m
    print(f"\n  MEAN at tol=1e-10: 80col={t80m:.16f}  76col={t76m:.16f}  delta={tight_gain:+.6e}")
    print(f"  gain at default tol : {mean76 - mean80:+.6e}")
    print(f"  gain at tol=1e-10   : {tight_gain:+.6e}")
    same_sign = np.sign(tight_gain) == np.sign(mean76 - mean80)
    print(f"  sign preserved: {same_sign} | magnitude ratio: "
          f"{abs(tight_gain) / abs(mean76 - mean80) if (mean76 - mean80) else float('nan'):.4f}")
    print(f"  |shift caused by tightening tol alone| (80col): {abs(t80m - mean80):.6e}")
    print(f"  |shift caused by tightening tol alone| (76col): {abs(t76m - mean76):.6e}")
    print("\n  READ THIS CAREFULLY: if the tol-induced shift is of the same order as the")
    print("  H1 gain, the gain is not distinguishable from optimiser convergence noise.")

    rule("COUNTERFACTUAL -- what would have to change to fix a wrong prediction")
    print("  Exact logit decomposition on the encoded design matrix, 80-col arm,")
    print("  using the actual fitted coefficients. Read-only.")
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        P80, m80, pre, names, Eva = fit_arm(train_ds.X, train_ds.y, val_ds.X, MODEL_B_COLUMNS)
        y = val_ds.y.to_numpy()
        pred = np.array(CLASS_ORDER)[P80.argmax(axis=1)]
        wrong = np.where(pred != y)[0]
        coef = coef_series(m80, names)
        idx = {c: i for i, c in enumerate(CLASS_ORDER)}
        needed = []
        for i in wrong[:2000]:
            true_c, pred_c = y[i], pred[i]
            z_gap = float((coef.loc[pred_c].to_numpy() - coef.loc[true_c].to_numpy()) @ Eva[i])
            z_gap += float(np.asarray(m80.intercept_)[list(m80.classes_).index(pred_c)]
                           - np.asarray(m80.intercept_)[list(m80.classes_).index(true_c)])
            dcoef = coef.loc[true_c].to_numpy() - coef.loc[pred_c].to_numpy()
            j = int(np.argmax(np.abs(dcoef)))
            delta_x = z_gap / dcoef[j] if dcoef[j] != 0 else np.inf
            needed.append((names[j], abs(delta_x), z_gap))
        if needed:
            arr = np.array([n[1] for n in needed], dtype=float)
            finite = arr[np.isfinite(arr)]
            from collections import Counter
            top = Counter(n[0] for n in needed).most_common(5)
            print(f"\n  {fold.name}: wrong={len(wrong)}  analysed={len(needed)}")
            print(f"    logit gap to close: mean={np.mean([n[2] for n in needed]):.4f}")
            print(f"    single-feature move needed (in scaled SD units): "
                  f"median={np.median(finite):.4f}  p90={np.percentile(finite, 90):.4f}")
            print(f"    within 1 SD (plausible): "
                  f"{float((finite <= 1).mean()):.4f}  within 0.5 SD: {float((finite <= 0.5).mean()):.4f}")
            print(f"    most frequently implicated single feature: {top}")

    rule("VERDICT")
    print(f"  H1 mean validation gain            : {mean76 - mean80:+.6e}")
    print(f"  H1 gain under tol=1e-10            : {tight_gain:+.6e}")
    print(f"  P1 (even split)  relative gap mean : {np.mean(p1_diffs):.4e}")
    print(f"  P2 ratio 76col/sum80 mean          : {np.mean(p2_ratio):.6f}")
    print(f"  P4 shrinkage observed in           : {np.mean(shrunk):.4f} of cases")
    print("\n  Interpretation is left to the reader. Stated plainly:")
    print("   - if P1/P2/P4 hold AND the gain survives tol=1e-10 with the same sign and")
    print("     order of magnitude, the mechanism is real though very small;")
    print("   - if the tol-induced shift is comparable to the gain, the gain is")
    print("     numerical noise regardless of whether P1/P2 hold.")
    print("  No claim of significance. No promotion. 2025/26 untouched.")

    rule("INTEGRITY")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    print(f"  MODEL_VERSION: {MODEL_VERSION!r}")
    print("  2025/26 accessed: NO | model modified: NO | artifact written: NONE")
    print("  tol=1e-10 used ONLY as a diagnostic probe, never as a proposed configuration")


if __name__ == "__main__":
    main()
