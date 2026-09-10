"""READ-ONLY LEAKAGE-IMPACT MODEL COMPARISON  (CURRENT vs STRICT)

PURPOSE
    Quantify whether the discovered simultaneous-kickoff contamination in
    the league-season accumulator materially changes the audited Model B
    result. Measurement only.

WHAT THIS IS NOT
    Not a V2 exercise. Not model selection. Not tuning. Not calibration.
    Not an upcoming-match prediction. No estimator, prediction file, or
    report file is ever written -- this script prints to stdout and
    creates nothing on disk.

FROZEN IDENTITY (imported, never redefined)
    MODEL_B_COLUMNS (80, in contract order)
    LogisticRegression(C=1.0, max_iter=2000, random_state=0)
        via models.train.train_logistic_regression
    V1 preprocessing (LogisticRegressionPreprocessor, inside that call)
    CLASS_ORDER = ["H", "D", "A"]
    no calibration
    SHRINKAGE_K and strength_features imported from features/

THE ONLY DIFFERENCE between the two conditions is the historical cutoff
used to build the five strength columns:
    CURRENT : the league accumulator as the existing pipeline builds it
              (includes same-unix fixtures ordered earlier by fixture_id)
    STRICT  : the league accumulator over history.unix < target.unix only

2025/26 IS USED ONLY in the existing Gate 7 / Phase 4C final-test
verification role. It is never used to fit, tune, calibrate, select a
model, select features, or choose between CURRENT and STRICT.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python leakage_impact_diagnostic.py
"""
from __future__ import annotations

import bisect
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from features.config import PLAYED_STATUSES, SHRINKAGE_K
from features.history import FeatureContext, LeagueSeasonAccumulator, load_fixtures_chronological
from features.strength import strength_features
from models.ablation import MODEL_B_COLUMNS, STRENGTH_COLUMNS, select_feature_columns
from models.baselines import CLASS_ORDER
from models.config import MODEL_VERSION
from models.data import SupervisedDataset, load_supervised_dataset
from models.evaluate import evaluate, validate_probabilities
from models.splits import final_split, iter_walk_forward_folds
from models.train import train_logistic_regression

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MATCHES_DB = REPO / "data" / "processed" / "matches.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"

STRENGTH = list(STRENGTH_COLUMNS)
LA_LIGA_COMPETITION_ID = 419
PHASE4C_MODEL_B_LOG_LOSS = 0.9960487649063601   # locked reference, read-only
EXPECTED_UNAFFECTED = 7858                       # from the prior impact quantification

EXTRA_GUARDED_FILES = (
    "src/models/calibration.py",
    "src/models/candidate_contract.py",
    "src/models/ablation.py",
    "data/audit/phase3_probability_diagnostics.json",
    "data/audit/phase4c_final_comparison.json",
    "data/audit/phase5a_dropped_feature_comparison.json",
)


class DiagnosticStop(SystemExit):
    """Raised to halt the diagnostic rather than proceed on bad evidence."""


def stop(message: str) -> None:
    print("\n" + "!" * 78)
    print("STOP -- diagnostic halted, nothing further executed")
    print(message)
    print("!" * 78)
    raise DiagnosticStop(1)


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def repo_snapshot() -> set[str]:
    """Every file in the repository, for the newly-created-file check."""
    return {
        str(p.relative_to(REPO))
        for p in REPO.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


# =====================================================================
# SECTION 1 -- leakage reconstruction + validation
# =====================================================================
def build_strict_strength() -> pd.DataFrame:
    """Recompute the five strength columns under history.unix < target.unix.

    Reuses the frozen formula (strength_features -> shrunk_rate ->
    SHRINKAGE_K) and the frozen LeagueSeasonAccumulator class. Only the
    accumulator CONTENTS differ; team histories are the same objects
    passed to both computations.
    """
    fixtures = load_fixtures_chronological(MATCHES_DB)

    # Sorted per-(competition, season) timeline of played fixtures.
    timeline: dict[tuple[int, int], list[tuple[int, float]]] = collections.defaultdict(list)
    for f in fixtures:
        if f.get("status") in PLAYED_STATUSES and f.get("home_goals") is not None:
            timeline[(f["competition_id"], f["season_id"])].append(
                (f["unix"], float(f["home_goals"] + f["away_goals"]))
            )
    index: dict[tuple[int, int], tuple[list[int], list[float]]] = {}
    for key, values in timeline.items():
        values.sort()
        unixes = [v[0] for v in values]
        cumulative = [0.0]
        for v in values:
            cumulative.append(cumulative[-1] + v[1])
        index[key] = (unixes, cumulative)

    def strict_totals(key, target_unix):
        """Totals over played fixtures with unix STRICTLY < target_unix.

        bisect_left on the sorted unix list is the assertion: it can never
        include an equal timestamp, so a same-kickoff fixture is
        structurally unreachable here (no fixture_id tie-break exists).
        """
        if key not in index:
            return 0, 0.0
        unixes, cumulative = index[key]
        i = bisect.bisect_left(unixes, target_unix)
        return i, float(cumulative[i])

    def accumulator(matches: int, goals: float) -> LeagueSeasonAccumulator:
        acc = LeagueSeasonAccumulator()
        acc.matches = matches
        acc.goals_for_sum = goals
        return acc

    ctx = FeatureContext()
    rows: list[dict] = []
    affected: set[int] = set()
    unaffected_identical = 0
    unaffected_total = 0

    violations = {
        "team_history_at_or_after_target": [],
        "target_in_own_history": [],
        "unaffected_value_differs": [],
        "history_after_target_unix": [],
    }

    for f in fixtures:
        key = (f["competition_id"], f["season_id"])
        current_acc = ctx.league_accumulator(*key)
        cur_m, cur_g = current_acc.matches, current_acc.goals_for_sum
        st_m, st_g = strict_totals(key, f["unix"])

        if cur_m < st_m:
            violations["history_after_target_unix"].append(f["fixture_id"])

        home_history = ctx.history_before(f["home_id"]) if f["home_id"] is not None else []
        away_history = ctx.history_before(f["away_id"]) if f["away_id"] is not None else []

        for match in list(home_history) + list(away_history):
            unix = match.get("unix")
            if unix is not None and unix >= f["unix"]:
                violations["team_history_at_or_after_target"].append(f["fixture_id"])
                break
            if match.get("fixture_id") == f["fixture_id"]:
                violations["target_in_own_history"].append(f["fixture_id"])
                break

        strict = strength_features(home_history, away_history, f["season_id"],
                                   accumulator(st_m, st_g))

        if cur_m != st_m:
            affected.add(f["fixture_id"])
        else:
            unaffected_total += 1
            current = strength_features(home_history, away_history, f["season_id"],
                                        accumulator(cur_m, cur_g))
            same = all(
                (current[c] is None and strict[c] is None) or current[c] == strict[c]
                for c in STRENGTH
            )
            if same:
                unaffected_identical += 1
            else:
                violations["unaffected_value_differs"].append(f["fixture_id"])

        row = {"fixture_id": f["fixture_id"]}
        row.update({c: strict[c] for c in STRENGTH})
        rows.append(row)

        ctx.record(f)   # compute-then-record, exactly as feature_builder does

    frame = pd.DataFrame(rows)
    frame.attrs.update(
        affected=affected,
        violations=violations,
        unaffected_total=unaffected_total,
        unaffected_identical=unaffected_identical,
        n_fixtures=len(fixtures),
    )
    return frame


def make_conditions():
    print("  Reconstructing strict-temporal strength values (in memory)...")
    strict_values = build_strict_strength()
    violations = strict_values.attrs["violations"]

    print(f"\n  fixtures walked                        : {strict_values.attrs['n_fixtures']}")
    print(f"  affected (same-kickoff contamination)  : {len(strict_values.attrs['affected'])}")
    print(f"  unaffected fixtures                    : {strict_values.attrs['unaffected_total']}")
    print(f"  unaffected identical across 5 columns  : {strict_values.attrs['unaffected_identical']}")

    print("\n  Pre-fit assertions:")
    checks = [
        ("strict history contains only unix < target.unix",
         not violations["history_after_target_unix"]),
        ("no same-unix fixture in strict history (bisect_left, no fixture_id tie-break)", True),
        ("target fixture never in its own history",
         not violations["target_in_own_history"]),
        ("team history has no unix >= target.unix",
         not violations["team_history_at_or_after_target"]),
        ("unaffected fixtures identical in both conditions",
         not violations["unaffected_value_differs"]),
        (f"unaffected identical count == expected {EXPECTED_UNAFFECTED}",
         strict_values.attrs["unaffected_identical"] == EXPECTED_UNAFFECTED),
    ]
    for label, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
    if not all(ok for _, ok in checks):
        stop("A pre-fit leakage assertion failed. Details:\n"
             + json.dumps({k: v[:10] for k, v in violations.items()}, indent=2)
             + f"\nunaffected_identical={strict_values.attrs['unaffected_identical']} "
               f"expected={EXPECTED_UNAFFECTED}")

    current = load_supervised_dataset(FEATURES_DB)
    lookup = strict_values.set_index("fixture_id")
    ids = current.metadata["fixture_id"].to_numpy()
    missing = [int(i) for i in ids if i not in lookup.index]
    if missing:
        stop(f"{len(missing)} fixture_id(s) in features.db absent from the reconstruction, "
             f"e.g. {missing[:5]}")

    strict_X = current.X.copy()
    for column in STRENGTH:
        strict_X[column] = lookup.loc[ids, column].to_numpy()

    others = [c for c in current.X.columns if c not in STRENGTH]
    if not current.X[others].equals(strict_X[others]):
        differing = [c for c in others if not current.X[c].equals(strict_X[c])]
        stop(f"Non-strength columns differ between conditions: {differing[:10]} "
             "-- the comparison would not be single-variable.")
    print(f"\n    [PASS] only league-season aggregation differs "
          f"({len(others)} non-strength columns byte-identical)")

    # `.equals()` is exact, so a column differing only by floating-point
    # residue is correctly reported as "differing". That is accurate but
    # can read as misleading, so the magnitude is printed alongside: the
    # earlier algebraic audit established that the league mean cancels in
    # (attack - defence), making `strength_diff` invariant, and residue at
    # ~1e-16 is the expected signature of that invariance. This is
    # REPORTING only -- no comparison or tolerance is altered here.
    changed = [c for c in current.X.columns if not current.X[c].equals(strict_X[c])]
    print(f"    columns differing between conditions   : {changed}")
    print("    per-column magnitude (exact .equals(), residue shown for context):")
    for column in changed:
        a = pd.to_numeric(current.X[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(strict_X[column], errors="coerce").to_numpy(dtype=float)
        both = ~(np.isnan(a) | np.isnan(b))
        delta = np.abs(a[both] - b[both])
        max_delta = delta.max() if delta.size else 0.0
        kind = "FLOAT RESIDUE ONLY" if max_delta < 1e-12 else "MATERIAL"
        print(f"      {column:<34} max|delta|={max_delta:.6e}  -> {kind}")

    strict = SupervisedDataset(metadata=current.metadata.copy(), X=strict_X, y=current.y.copy())
    return current, strict, strict_values.attrs["affected"]


# =====================================================================
# SECTION 2 -- frozen fit + evaluation
# =====================================================================
def fit_predict(train_ds: SupervisedDataset, eval_ds: SupervisedDataset) -> np.ndarray:
    """The audited Model B path, unchanged. In-memory only; nothing saved.

    API NOTE (authoritative source: src/models/train.py:139-153):
    `train_logistic_regression` returns a 3-TUPLE
        (model, preprocessor, P_val)
    not a bare probability matrix. `P_val` is already `_reorder_proba`'d
    into CLASS_ORDER (H, D, A) column order. Every existing caller in the
    project unpacks it the same way -- calibration.py:104/712,
    run_ablation.py:156, run_dropped_feature.py:156,
    run_experiments.py:170/293, run_final_comparison.py:202 -- so this
    follows the established convention rather than inventing one.

    The model and preprocessor are deliberately discarded: this diagnostic
    persists no estimator.
    """
    X_train = select_feature_columns(train_ds.X, MODEL_B_COLUMNS, "model_b")
    X_eval = select_feature_columns(eval_ds.X, MODEL_B_COLUMNS, "model_b")

    _model, _preprocessor, P = train_logistic_regression(X_train, train_ds.y, X_eval)

    # The API genuinely returns an (n, 3) float ndarray in CLASS_ORDER, so
    # this asserts that contract rather than coercing anything into shape.
    P = np.asarray(P, dtype=float)
    assert P.ndim == 2, f"expected a 2-D probability matrix, got ndim={P.ndim}"
    assert P.shape[1] == 3, f"expected 3 columns (H, D, A), got {P.shape[1]}"
    assert P.shape[0] == len(eval_ds.X), (
        f"row count {P.shape[0]} does not match the {len(eval_ds.X)} evaluated fixtures"
    )

    validate_probabilities(P)   # authoritative validator, unmodified
    return P


def per_class_metrics(y_true, P: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    predicted = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    out = {}
    for c in CLASS_ORDER:
        tp = int(((predicted == c) & (y_true == c)).sum())
        fp = int(((predicted == c) & (y_true != c)).sum())
        fn = int(((predicted != c) & (y_true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = dict(precision=precision, recall=recall, f1=f1,
                      support=int((y_true == c).sum()))
    return out


def run_walk_forward(dataset: SupervisedDataset, label: str):
    folds = {}
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        P = fit_predict(train_ds, val_ds)
        result = evaluate(val_ds.y, P)
        folds[fold.name] = dict(
            metrics=result.as_dict(),
            P=P,
            fixture_ids=val_ds.metadata["fixture_id"].to_numpy(),
            y=val_ds.y.to_numpy(),
        )
        print(f"    [{label}] {fold.name:<8} log_loss={result.log_loss:.16f}  "
              f"acc={result.accuracy:.6f}  n={result.n}")
    mean_log_loss = statistics.fmean(f["metrics"]["log_loss"] for f in folds.values())
    print(f"    [{label}] MEAN validation log loss = {mean_log_loss:.16f}")
    return folds, mean_log_loss


# =====================================================================
# SECTION 3 -- probability-delta analysis
# =====================================================================
def delta_stats(P_current: np.ndarray, P_strict: np.ndarray, name: str) -> np.ndarray:
    diff = np.abs(np.asarray(P_current) - np.asarray(P_strict))
    per_fixture_max = diff.max(axis=1)
    ordered = np.sort(per_fixture_max)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

    print(f"\n    {name}  (n = {len(per_fixture_max)} fixtures)")
    print(f"      identical (max delta == 0)         : {int((per_fixture_max == 0).sum())}")
    print(f"      non-identical                      : {int((per_fixture_max > 0).sum())}")
    print(f"      mean   |delta|  (all H/D/A cells)  : {diff.mean():.6e}")
    print(f"      median |delta|  (per fixture max)  : {np.median(per_fixture_max):.6e}")
    print(f"      p95    |delta|  (per fixture max)  : {p95:.6e}")
    print(f"      max    |delta|  (per fixture max)  : {per_fixture_max.max():.6e}")
    print("      descriptive bins (NOT acceptance thresholds):")
    for threshold in (1e-6, 1e-4, 1e-3, 1e-2):
        print(f"        fixtures with max delta > {threshold:<7g}: "
              f"{int((per_fixture_max > threshold).sum())}")
    return per_fixture_max


# =====================================================================
# SECTION 6 -- integrity
# =====================================================================
def integrity_report(stage: str) -> dict[str, str]:
    print(f"\n  --- integrity: {stage} ---")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    observed = {}
    mismatches = []
    for rel, record in sorted(pins.items()):
        actual = md5(REPO / rel)
        observed[rel] = actual
        if actual != record["expected"]:
            mismatches.append(rel)
    print(f"    13 pinned baselines: {len(pins)} checked | "
          f"mismatches: {mismatches if mismatches else 'NONE'}")
    print(f"    features.db : {observed.get('data/processed/features.db')}")
    print(f"    matches.db  : {observed.get('data/processed/matches.db')}")
    for rel in EXTRA_GUARDED_FILES:
        actual = md5(REPO / rel)
        observed[rel] = actual
        print(f"    {actual}  {rel}")
    print(f"    MODEL_VERSION: {MODEL_VERSION!r}")

    serialized = [str(p.relative_to(REPO)) for pattern in ("*.pkl", "*.joblib", "*.pickle", "*.onnx", "*.sav")
                  for p in REPO.rglob(pattern) if "__pycache__" not in p.parts]
    v2 = [str(p.relative_to(REPO)) for p in REPO.rglob("*v2*") if p.is_file() and "__pycache__" not in p.parts]
    print(f"    serialized estimators: {serialized or 'NONE'}")
    print(f"    V2 artifacts         : {v2 or 'NONE'}")
    if mismatches:
        stop(f"Pinned baseline mismatch at stage {stage}: {mismatches}")
    return observed


# =====================================================================
def main() -> None:
    print("READ-ONLY LEAKAGE-IMPACT MODEL COMPARISON  (CURRENT vs STRICT)")
    print("Measurement only. No pipeline decision, no V1/V2 judgement, no match prediction.")

    rule("0. ENVIRONMENT AND FROZEN IDENTITY")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
          f"numpy {np.__version__} | pandas {pd.__version__}")
    print(f"  MODEL_VERSION           : {MODEL_VERSION!r}  (read only, never written)")
    print(f"  MODEL_B_COLUMNS         : {len(MODEL_B_COLUMNS)} columns, contract order preserved")
    print(f"  CLASS_ORDER             : {CLASS_ORDER}")
    print(f"  SHRINKAGE_K             : {SHRINKAGE_K}")
    print("  estimator               : LogisticRegression(C=1.0, max_iter=2000, random_state=0)")
    print("                            via models.train.train_logistic_regression (V1 preprocessing)")
    print("  calibration             : NONE")

    files_before = repo_snapshot()
    checks_before = integrity_report("BEFORE")

    rule("1. LEAKAGE RECONSTRUCTION VALIDATION")
    current, strict, affected = make_conditions()

    rule("2. CURRENT vs STRICT -- METRICS (existing walk-forward protocol)")
    print("  CURRENT:")
    current_folds, current_mean = run_walk_forward(current, "CURRENT")
    print("\n  STRICT:")
    strict_folds, strict_mean = run_walk_forward(strict, "STRICT")

    print(f"\n  {'Metric':<26} {'CURRENT':>22} {'STRICT':>22} {'Delta':>14}")
    print(f"  {'-' * 26} {'-' * 22} {'-' * 22} {'-' * 14}")
    print(f"  {'mean_log_loss':<26} {current_mean:>22.16f} {strict_mean:>22.16f} "
          f"{strict_mean - current_mean:>14.3e}")
    for metric in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy"):
        a = statistics.fmean(f["metrics"][metric] for f in current_folds.values())
        b = statistics.fmean(f["metrics"][metric] for f in strict_folds.values())
        print(f"  {'mean_' + metric:<26} {a:>22.16f} {b:>22.16f} {b - a:>14.3e}")

    print("\n  Per-fold log loss:")
    for name in current_folds:
        a = current_folds[name]["metrics"]["log_loss"]
        b = strict_folds[name]["metrics"]["log_loss"]
        print(f"    {name:<10} CURRENT={a:.16f}  STRICT={b:.16f}  delta={b - a:+.6e}")

    order_current = sorted(current_folds, key=lambda k: current_folds[k]["metrics"]["log_loss"])
    order_strict = sorted(strict_folds, key=lambda k: strict_folds[k]["metrics"]["log_loss"])
    print(f"\n  fold ordering by log loss : CURRENT={order_current}  STRICT={order_strict}")
    print(f"  fold ordering              : {'UNCHANGED' if order_current == order_strict else 'CHANGED'}")
    lower = ("CURRENT" if current_mean < strict_mean
             else "STRICT" if strict_mean < current_mean else "IDENTICAL")
    print(f"  lower mean validation log loss: {lower}")

    rule("2b. 2025/26 FINAL PARTITION -- Gate 7 / Phase 4C VERIFICATION ROLE ONLY")
    print("  Used strictly as the locked final regression-verification partition.")
    print("  NOT used to fit, tune, calibrate, select a model, select features,")
    print("  choose thresholds, or choose between CURRENT and STRICT.")
    print("  The existing Phase 4C artifact is read for reference only and never written.")

    current_train, current_test = final_split(current)
    strict_train, strict_test = final_split(strict)
    P_current = fit_predict(current_train, current_test)
    P_strict = fit_predict(strict_train, strict_test)
    result_current = evaluate(current_test.y, P_current)
    result_strict = evaluate(strict_test.y, P_strict)

    print(f"\n  {'Metric':<26} {'CURRENT':>22} {'STRICT':>22} {'Delta':>14}")
    print(f"  {'-' * 26} {'-' * 22} {'-' * 22} {'-' * 14}")
    for metric in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy"):
        a = result_current.as_dict()[metric]
        b = result_strict.as_dict()[metric]
        print(f"  {metric:<26} {a:>22.16f} {b:>22.16f} {b - a:>14.3e}")
    print(f"  {'n':<26} {result_current.n:>22} {result_strict.n:>22}")

    print(f"\n  Locked Phase 4C Model B reference log loss : {PHASE4C_MODEL_B_LOG_LOSS:.16f}")
    print(f"  CURRENT   absolute difference from it      : "
          f"{abs(result_current.log_loss - PHASE4C_MODEL_B_LOG_LOSS):.6e}")
    print(f"  STRICT    absolute difference from it      : "
          f"{abs(result_strict.log_loss - PHASE4C_MODEL_B_LOG_LOSS):.6e}")

    print("\n  Per-class metrics (CURRENT -> STRICT):")
    pc_current = per_class_metrics(current_test.y, P_current)
    pc_strict = per_class_metrics(strict_test.y, P_strict)
    for c in CLASS_ORDER:
        a, b = pc_current[c], pc_strict[c]
        print(f"    {c}: precision {a['precision']:.6f} -> {b['precision']:.6f} | "
              f"recall {a['recall']:.6f} -> {b['recall']:.6f} | "
              f"f1 {a['f1']:.6f} -> {b['f1']:.6f} | support={a['support']}")

    print("\n  Confusion matrices:")
    print(f"    CURRENT (rows=true {CLASS_ORDER}, cols=pred):\n{result_current.confusion_matrix}")
    print(f"    STRICT  (rows=true {CLASS_ORDER}, cols=pred):\n{result_strict.confusion_matrix}")

    rule("3. PROBABILITY-DELTA ANALYSIS")
    print("  Validation folds:")
    for name in current_folds:
        if not np.array_equal(current_folds[name]["fixture_ids"], strict_folds[name]["fixture_ids"]):
            stop(f"fixture_id alignment differs between conditions in {name}")
        delta_stats(current_folds[name]["P"], strict_folds[name]["P"], name)

    print("\n  Final partition (2025/26):")
    if not np.array_equal(current_test.metadata["fixture_id"].to_numpy(),
                          strict_test.metadata["fixture_id"].to_numpy()):
        stop("fixture_id alignment differs between conditions on the final partition")
    delta_stats(P_current, P_strict, "final partition")

    rule("4. 2025/26 AFFECTED-SUBSET DELTA ANALYSIS")
    print("  Probability-change measurement ONLY. No accuracy or performance")
    print("  claim is made for this subset.")
    test_ids = current_test.metadata["fixture_id"].to_numpy()
    competition = (current_test.X["competition_id"].to_numpy()
                   if "competition_id" in current_test.X.columns else None)
    mask = np.array([
        (int(fid) in affected) and (competition is None or int(competition[i]) == LA_LIGA_COMPETITION_ID)
        for i, fid in enumerate(test_ids)
    ])
    print(f"\n  affected 2025/26 La Liga fixtures matched in the final partition: {int(mask.sum())}")
    if mask.any():
        delta_stats(P_current[mask], P_strict[mask], "affected La Liga subset")
        print("\n  affected fixture_ids:")
        print(f"    {sorted(int(f) for f in test_ids[mask])}")
        if (~mask).any():
            delta_stats(P_current[~mask], P_strict[~mask], "all other 2025/26 fixtures")
    else:
        print("  (no affected fixture present in the final partition)")

    rule("5. DETERMINISM (repeat runs under the existing deterministic protocol)")
    P_current_repeat = fit_predict(current_train, current_test)
    P_strict_repeat = fit_predict(strict_train, strict_test)
    current_deterministic = P_current.tobytes() == P_current_repeat.tobytes()
    strict_deterministic = P_strict.tobytes() == P_strict_repeat.tobytes()
    print(f"  CURRENT repeated run byte-identical : {current_deterministic}")
    print(f"  STRICT  repeated run byte-identical : {strict_deterministic}")
    print(f"  CURRENT sha256 : {hashlib.sha256(P_current.tobytes()).hexdigest()}")
    print(f"  STRICT  sha256 : {hashlib.sha256(P_strict.tobytes()).hexdigest()}")
    if not (current_deterministic and strict_deterministic):
        stop("Non-determinism detected under a protocol that requires byte-identical repeats.")

    rule("6. INTEGRITY / CHECKSUM RESULT")
    checks_after = integrity_report("AFTER")
    drifted = [k for k in checks_before if checks_before[k] != checks_after.get(k)]
    print(f"\n  files whose checksum changed during the diagnostic: {drifted if drifted else 'NONE'}")
    if drifted:
        stop(f"The diagnostic itself altered: {drifted}")

    files_after = repo_snapshot()
    created = sorted(files_after - files_before)
    removed = sorted(files_before - files_after)
    print(f"  newly created files (excluding this script): "
          f"{[f for f in created if f != Path(__file__).name] or 'NONE'}")
    print(f"  removed files: {removed or 'NONE'}")

    rule("7. FORBIDDEN-OPERATION AUDIT")
    for line in (
        "hyperparameter tuning        : NONE -- C, max_iter, random_state imported unchanged",
        "calibration fitting          : NONE -- no calibrator constructed or applied",
        "model selection decision     : NONE -- no condition chosen, no ranking acted upon",
        "feature selection            : NONE -- MODEL_B_COLUMNS used as frozen",
        "threshold selection          : NONE",
        "estimator persisted          : NONE -- both fits in memory only",
        "prediction artifact written  : NONE -- no CSV/JSON/JSONL produced",
        "report file written          : NONE -- stdout only",
        "V2 creation                  : NONE",
        "deployment / API / UI        : NONE",
        "production exposure          : NONE",
        "frozen V1 code modified      : NONE -- all V1 modules imported read-only",
        "databases modified           : NONE -- opened read-only by existing loaders",
        "2025/26 usage                : Gate 7 / Phase 4C verification role ONLY",
    ):
        print(f"  {line}")

    rule("8. FINAL FACTUAL CONCLUSION")
    validation_delta = strict_mean - current_mean
    final_delta = result_strict.log_loss - result_current.log_loss
    all_max = np.concatenate([
        np.abs(current_folds[n]["P"] - strict_folds[n]["P"]).max(axis=1) for n in current_folds
    ] + [np.abs(P_current - P_strict).max(axis=1)])
    print(f"  mean validation log loss delta (STRICT - CURRENT) : {validation_delta:+.6e}")
    print(f"  final-partition log loss delta (STRICT - CURRENT) : {final_delta:+.6e}")
    print(f"  largest single-fixture probability change         : {all_max.max():.6e}")
    print(f"  fixtures with any probability change              : "
          f"{int((all_max > 0).sum())} of {len(all_max)}")
    print(f"  fixtures with probability change > 1e-2           : {int((all_max > 1e-2).sum())}")
    print(f"  fold ordering                                     : "
          f"{'UNCHANGED' if order_current == order_strict else 'CHANGED'}")
    print("\n  These are measurements. This diagnostic does NOT decide whether the")
    print("  feature pipeline should be changed, does NOT declare V1 invalid, does")
    print("  NOT approve V2, and makes no claim that the leakage is acceptable.")

    rule("END OF DIAGNOSTIC")


if __name__ == "__main__":
    main()
