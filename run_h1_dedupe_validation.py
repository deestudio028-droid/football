"""H1 -- de-duplicated 76-column candidate, validation folds ONLY.

HYPOTHESIS UNDER TEST
    The frozen 80-column contract contains four alias pairs that carry the
    same signal. Under L2 a duplicated signal splits its weight (w/2 each)
    and pays 2*(w/2)^2 = w^2/2 -- half the penalty a single feature with
    the same total influence would pay. Duplicated signals are therefore
    systematically under-penalised. H1 removes the four aliases.

EXACTLY ONE VARIABLE CHANGES
    CONTROL   : 80 columns (MODEL_B_COLUMNS), C=0.0005
    H1        : 76 columns (same, minus 4 aliases), C=0.0005
    Everything else identical: same LogisticRegression, same C, same V1
    preprocessing, same three walk-forward folds, same metrics, same seed.
    H2 and H3 are NOT tested here -- testing them together would confound.

2025/26
    Structurally unreachable: imports `iter_walk_forward_folds`, never
    `final_split`. The three folds span 2020/21-2024/25 only.

WRITES NOTHING. stdout only. No artifact, no estimator, no file.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_h1_dedupe_validation.py
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
    FORM_COLUMNS,
    GOALS_CORE_COLUMNS,
    MODEL_B,
    MODEL_B_COLUMNS,
    select_feature_columns,
)
from models.baselines import CLASS_ORDER
from models.config import FINAL_TEST_SEASONS, MODEL_VERSION, WALK_FORWARD_FOLDS
from models.data import load_supervised_dataset
from models.evaluate import evaluate, validate_probabilities
from models.robustness import V1_LOGREG_BASE_KWARGS
from models.splits import iter_walk_forward_folds

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"

CANDIDATE_C: float = 0.0005          # frozen E-1 value; the ONLY C used here

#: The four form-family aliases removed by H1. Each is numerically equal
#: (to floating-point residue) to a retained goals_core twin.
H1_REMOVED_ALIASES: tuple[str, ...] = (
    "home_goal_diff_last5", "home_goal_diff_last10",
    "away_goal_diff_last5", "away_goal_diff_last10",
)
ALIAS_TWIN: dict[str, str] = {
    "home_goal_diff_last5": "home_goals_diff_last5",
    "home_goal_diff_last10": "home_goals_diff_last10",
    "away_goal_diff_last5": "away_goals_diff_last5",
    "away_goal_diff_last10": "away_goals_diff_last10",
}

H1_COLUMNS: tuple[str, ...] = tuple(c for c in MODEL_B_COLUMNS if c not in H1_REMOVED_ALIASES)

#: E-1 recorded results for the 80-column arm at C=0.0005. The control must
#: reproduce these before any H1 number is trusted.
E1_CONTROL = {
    "mean_log_loss": 0.9915719393130367,
    "mean_brier": 0.5914091026848851,
    "fold_1": 0.9980114078007254,
    "fold_2": 0.9846581646671166,
    "fold_3": 0.9920462454712682,
}

FROZEN_V1_SOURCES = (
    "src/models/config.py", "src/models/train.py", "src/models/splits.py",
    "src/models/evaluate.py", "src/models/data.py", "src/models/baselines.py",
    "src/models/run_experiments.py",
)


class Stop(SystemExit):
    pass


def stop(msg: str) -> None:
    print("\n" + "!" * 78)
    print("STOP -- experiment halted")
    print(msg)
    print("!" * 78)
    raise Stop(1)


def rule(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def build_estimator(C: float):
    if C != CANDIDATE_C:
        raise ValueError(f"C={C!r} is not the frozen candidate value {CANDIDATE_C}; "
                         "this experiment varies the feature set, never C.")
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, **V1_LOGREG_BASE_KWARGS)


def select_columns(X: pd.DataFrame, columns: tuple[str, ...], label: str) -> pd.DataFrame:
    """Fail loudly on any missing column; never substitute or silently drop."""
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise RuntimeError(f"{label}: {len(missing)} column(s) missing: {missing[:5]}")
    return X[list(columns)]


def fit_predict(X_train, y_train, X_val, columns, label) -> np.ndarray:
    from models import train as train_module

    Xtr = select_columns(X_train, columns, label)
    Xva = select_columns(X_val, columns, label)
    pre = train_module.LogisticRegressionPreprocessor().fit(Xtr)
    model = build_estimator(CANDIDATE_C)
    model.fit(pre.transform(Xtr), y_train)
    P = train_module._reorder_proba(model, pre.transform(Xva))
    P = np.asarray(P, dtype=float)
    assert P.ndim == 2 and P.shape[1] == 3 and P.shape[0] == len(Xva)
    validate_probabilities(P)
    return P


def per_class(y_true, P):
    y_true = np.asarray(y_true)
    pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    out = {}
    for c in CLASS_ORDER:
        tp = int(((pred == c) & (y_true == c)).sum())
        fp = int(((pred == c) & (y_true != c)).sum())
        fn = int(((pred != c) & (y_true == c)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        out[c] = dict(precision=pr, recall=rc, f1=f1, support=int((y_true == c).sum()))
    return out


def run_arm(dataset, columns, label):
    folds = {}
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        P = fit_predict(train_ds.X, train_ds.y, val_ds.X, columns, label)
        r = evaluate(val_ds.y, P)
        y = val_ds.y.to_numpy()
        pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]
        folds[fold.name] = dict(
            metrics=r.as_dict(), per_class=per_class(y, P), P=P, y=y, pred=pred,
            confusion=r.confusion_matrix, n=r.n,
            draw={"argmax_count": int((pred == "D").sum()),
                  "argmax_share": float((pred == "D").mean()),
                  "mean_p_draw": float(P[:, 1].mean()),
                  "max_p_draw": float(P[:, 1].max()),
                  "actual_rate": float((y == "D").mean())},
        )
        print(f"    [{label}] {fold.name:<8} log_loss={r.log_loss:.16f} brier={r.brier:.8f} "
              f"acc={r.accuracy:.6f} n={r.n}")
    means = {m: statistics.fmean(f["metrics"][m] for f in folds.values())
             for m in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy")}
    print(f"    [{label}] MEAN log_loss={means['log_loss']:.16f} brier={means['brier']:.16f}")
    return dict(folds=folds, means=means, columns=columns, label=label)


def main() -> None:
    print("H1 -- DE-DUPLICATED 76-COLUMN CANDIDATE (validation folds only)")
    print("One variable: the feature contract. C stays at the frozen 0.0005.")

    rule("PRE-FLIGHT")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION: {MODEL_VERSION!r} | C (both arms): {CANDIDATE_C}")
    print(f"  estimator kwargs: {V1_LOGREG_BASE_KWARGS}")

    seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    if seasons & set(FINAL_TEST_SEASONS):
        stop(f"2025/26 reachable via folds: {seasons & set(FINAL_TEST_SEASONS)}")
    print(f"  fold seasons: {sorted(seasons)} | 2025/26 reachable: False  [PASS]")

    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, rec in pins.items() if before[r] != rec["expected"]]
    print(f"  13 pinned baselines: mismatches={bad or 'NONE'}  [{'PASS' if not bad else 'FAIL'}]")
    if bad:
        stop(f"pinned mismatch: {bad}")
    for rel in FROZEN_V1_SOURCES:
        print(f"    {md5(REPO / rel)}  {rel}")

    # -----------------------------------------------------------------
    rule("CONTRACT PROOF -- the ONLY difference is the four alias removals")
    removed = [c for c in MODEL_B_COLUMNS if c not in H1_COLUMNS]
    added = [c for c in H1_COLUMNS if c not in MODEL_B_COLUMNS]
    order_ok = H1_COLUMNS == tuple(c for c in MODEL_B_COLUMNS if c in set(H1_COLUMNS))
    print(f"  control columns : {len(MODEL_B_COLUMNS)}")
    print(f"  H1 columns      : {len(H1_COLUMNS)}")
    print(f"  removed ({len(removed)})    : {removed}")
    print(f"  added   ({len(added)})    : {added or 'NONE'}")
    print(f"  removed set == the 4 declared aliases : {set(removed) == set(H1_REMOVED_ALIASES)}")
    print(f"  relative order of retained columns kept: {order_ok}")
    print(f"  H1 subset of the frozen contract       : {set(H1_COLUMNS) <= set(MODEL_B_COLUMNS)}")
    if not (set(removed) == set(H1_REMOVED_ALIASES) and not added and order_ok
            and len(H1_COLUMNS) == 76):
        stop("The H1 contract differs by more than the four alias removals.")

    dataset = load_supervised_dataset(FEATURES_DB)
    print(f"\n  dataset rows: {len(dataset.X)}")
    print("  numeric equivalence of each removed alias to its retained twin "
          "(whole dataset, not just folds):")
    for alias, twin in ALIAS_TWIN.items():
        a, t = dataset.X[alias], dataset.X[twin]
        nan_same = bool((a.isna() == t.isna()).all())
        both = a.notna() & t.notna()
        maxdiff = float(np.abs(a[both].to_numpy() - t[both].to_numpy()).max()) if both.any() else 0.0
        bitwise = bool(np.array_equal(a[both].to_numpy(), t[both].to_numpy()))
        r = float(a.corr(t))
        print(f"    {alias:<24} vs {twin:<25} corr={r:.10f} max|diff|={maxdiff:.3e} "
              f"nan_pattern_same={nan_same} bitwise_identical={bitwise}")
        if not nan_same or maxdiff > 1e-12:
            stop(f"{alias} is NOT numerically equivalent to {twin} "
                 f"(max|diff|={maxdiff:.3e}); H1's premise does not hold.")
    print("  [PASS] every removed alias is numerically equivalent to a retained twin "
          "(to <1e-12); removal loses no measurable information")
    print(f"  aliases belonged to form={all(c in FORM_COLUMNS for c in H1_REMOVED_ALIASES)}, "
          f"twins retained in goals_core={all(t in GOALS_CORE_COLUMNS for t in ALIAS_TWIN.values())}")

    # -----------------------------------------------------------------
    rule("CONTROL -- 80 columns, C=0.0005")
    control = run_arm(dataset, MODEL_B_COLUMNS, "80-col")

    rule("FIDELITY -- control must reproduce the recorded E-1 result")
    ok = True
    print(f"  mean log loss  recorded={E1_CONTROL['mean_log_loss']:.16f}  "
          f"observed={control['means']['log_loss']:.16f}  "
          f"diff={abs(control['means']['log_loss'] - E1_CONTROL['mean_log_loss']):.3e}")
    ok &= abs(control["means"]["log_loss"] - E1_CONTROL["mean_log_loss"]) < 1e-9
    print(f"  mean Brier     recorded={E1_CONTROL['mean_brier']:.16f}  "
          f"observed={control['means']['brier']:.16f}  "
          f"diff={abs(control['means']['brier'] - E1_CONTROL['mean_brier']):.3e}")
    ok &= abs(control["means"]["brier"] - E1_CONTROL["mean_brier"]) < 1e-9
    for name in ("fold_1", "fold_2", "fold_3"):
        obs = control["folds"][name]["metrics"]["log_loss"]
        d = abs(obs - E1_CONTROL[name])
        ok &= d < 1e-9
        print(f"  {name} recorded={E1_CONTROL[name]:.16f} observed={obs:.16f} diff={d:.3e}")
    if not ok:
        stop("Control did not reproduce the recorded E-1 80-column C=0.0005 result to <1e-9. "
             "No H1 number may be trusted.")
    print("  [PASS] control reproduces E-1 exactly")

    # -----------------------------------------------------------------
    rule("H1 ARM -- 76 columns, C=0.0005")
    h1 = run_arm(dataset, H1_COLUMNS, "76-col")

    rule("PER-FOLD COMPARISON")
    for name in control["folds"]:
        c, h = control["folds"][name], h1["folds"][name]
        print(f"\n  {name} (n={c['n']}):")
        for m in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy"):
            a, b = c["metrics"][m], h["metrics"][m]
            print(f"    {m:<20} 80col={a:.16f}  76col={b:.16f}  delta={b - a:+.3e}")
        print("    per-class (80col -> 76col):")
        for cl in CLASS_ORDER:
            pa, pb = c["per_class"][cl], h["per_class"][cl]
            print(f"      {cl}: precision {pa['precision']:.6f}->{pb['precision']:.6f} "
                  f"recall {pa['recall']:.6f}->{pb['recall']:.6f} "
                  f"f1 {pa['f1']:.6f}->{pb['f1']:.6f} support={pa['support']}")

    rule("MEAN METRICS -- 80col vs 76col")
    print(f"  {'Metric':<22} {'80col (control)':>22} {'76col (H1)':>22} {'Delta':>16}")
    print("  " + "-" * 84)
    for m in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy"):
        a, b = control["means"][m], h1["means"][m]
        print(f"  {m:<22} {a:>22.16f} {b:>22.16f} {b - a:>+16.3e}")

    rule("DRAW BEHAVIOUR")
    for name in control["folds"]:
        c, h = control["folds"][name]["draw"], h1["folds"][name]["draw"]
        print(f"  {name}: actual draw rate {c['actual_rate']:.6f}")
        print(f"    argmax D count   80col={c['argmax_count']:4d}  76col={h['argmax_count']:4d}  "
              f"({c['argmax_share']:.6f} -> {h['argmax_share']:.6f})")
        print(f"    mean p_draw      80col={c['mean_p_draw']:.6f}  76col={h['mean_p_draw']:.6f}  "
              f"delta={h['mean_p_draw'] - c['mean_p_draw']:+.3e}")
        print(f"    max  p_draw      80col={c['max_p_draw']:.6f}  76col={h['max_p_draw']:.6f}")

    rule("PREDICTION MOVEMENT")
    for name in control["folds"]:
        Pc, Ph = control["folds"][name]["P"], h1["folds"][name]["P"]
        d = np.abs(Ph - Pc).max(axis=1)
        changed = int((control["folds"][name]["pred"] != h1["folds"][name]["pred"]).sum())
        print(f"  {name}: max|delta| mean={d.mean():.6e} median={np.median(d):.6e} "
              f"max={d.max():.6e} | predicted class changed for {changed} of {len(d)}")

    rule("DETERMINISM -- double run")
    det = {}
    for label, arm, cols in (("80col", control, MODEL_B_COLUMNS), ("76col", h1, H1_COLUMNS)):
        repeat = run_arm(dataset, cols, f"repeat {label}")
        same = all(repeat["folds"][n]["P"].tobytes() == arm["folds"][n]["P"].tobytes()
                   for n in arm["folds"])
        det[label] = same
        digest = hashlib.sha256(
            b"".join(arm["folds"][n]["P"].tobytes() for n in sorted(arm["folds"]))).hexdigest()
        print(f"  {label}: byte-identical repeat={same}  sha256={digest[:40]}...")
    if not all(det.values()):
        stop(f"Determinism failed: {det}")
    print("  [PASS] both arms deterministic")

    rule("VERDICT (validation evidence only)")
    d_ll = h1["means"]["log_loss"] - control["means"]["log_loss"]
    per_fold_better = {n: h1["folds"][n]["metrics"]["log_loss"]
                       < control["folds"][n]["metrics"]["log_loss"] for n in control["folds"]}
    d_br = h1["means"]["brier"] - control["means"]["brier"]
    print(f"  primary: mean validation log loss delta (76col - 80col) = {d_ll:+.6e}")
    print(f"  improves in each fold: {per_fold_better}")
    print(f"  mean Brier delta: {d_br:+.6e}")
    if d_ll < 0:
        print("\n  *** H1 IMPROVES mean validation log loss ***")
        print("  NOT frozen. NOT promoted. 2025/26 NOT touched.")
        print("  Reporting evidence only, per instruction. Awaiting review.")
        if not all(per_fold_better.values()):
            print("  NOTE: the improvement is not present in every fold -- weaker evidence "
                  "than E-1's all-folds result.")
    elif d_ll > 0:
        print("\n  *** H1 WORSENS mean validation log loss -- H1 REJECTED ***")
        print("  The 80-column contract is retained. No further H1 work.")
    else:
        print("\n  *** H1 is numerically identical on the primary metric ***")
    print("\n  No significance claim is made (no statistical test performed).")
    print("  No claim of generalization beyond these validation folds.")

    rule("INTEGRITY / FORBIDDEN-OPERATION AUDIT")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines after run: mismatches={drift or 'NONE'}")
    if drift:
        stop(f"experiment altered pinned files: {drift}")
    print(f"  MODEL_VERSION: {MODEL_VERSION!r}")
    ser = [str(p.relative_to(REPO)) for pat in ("*.pkl", "*.joblib", "*.pickle")
           for p in REPO.rglob(pat) if "__pycache__" not in p.parts]
    print(f"  serialized estimators: {ser or 'NONE'}")
    for line in ("2025/26 accessed: NO", "C tuned: NO (fixed 0.0005, other C rejected)",
                 "calibration: NONE", "class weighting: NONE", "H2/H3 tested: NO",
                 "production source modified: NO", "features.db regenerated: NO",
                 "artifact written: NONE", "file created by this script: NONE"):
        print(f"  {line}")

    rule("END OF H1 EXPERIMENT")


if __name__ == "__main__":
    main()
