"""Blind out-of-sample batch prediction on 2025/26 fixtures (read-only).

    python predict_blind_2025_26.py --count 20
    python predict_blind_2025_26.py --fixture-ids 123456 234567 ...

PREDICTION PHASE ONLY. This tool never reads, queries, prints or derives
`label_result`, `label_home_goals` or `label_away_goals`. Ground truth is
revealed by a SEPARATE command, run afterwards, so prediction and outcome
stay temporally separated:

    python reveal_results_2025_26.py --fixture-ids <the same ids>

WHY THIS IS BLIND, AND HOW THAT IS VERIFIED
The artifact must have been trained with --exclude-final-test, i.e. on
2020/21-2024/25 only, so 2025/26 was never seen. That claim is not taken
on trust: STEP 2 recomputes the median-imputer statistics for BOTH
candidate partitions from features.db and checks which one the artifact
actually carries. If the artifact matches the all-rows partition (i.e. it
saw 2025/26), this tool HARD-FAILS rather than presenting in-sample
predictions as blind ones.

WHAT IT DOES NOT DO
No training, no tuning, no refitting, no writes of any kind. The model
artifact, features.db, matches.db and every pinned input are untouched.
No prediction is saved -- stdout only. No aggregate metric is computed
here; accuracy cannot be assessed until after the reveal step, and doing
it in this process would defeat the separation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

RESULT_LABEL = {"H": "HOME WIN", "D": "DRAW", "A": "AWAY WIN"}
REVEAL_TOOL = "reveal_results_2025_26.py"


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 78)
    print("STOP -- blind prediction halted. No prediction is produced.")
    print(msg)
    print("!" * 78)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        stop(f"stop condition: {label}")


def _non_docstring_strings(tree):
    """String constants excluding docstrings -- docstrings legitimately
    name the label columns they promise never to read, and a plain grep
    would flag that prose as a violation."""
    doc_ids = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                doc_ids.add(id(n.body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_ids]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Blind out-of-sample batch prediction on 2025/26 fixtures.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", type=int, help="number of deterministic 2025/26 FT fixtures")
    g.add_argument("--fixture-ids", nargs="+", help="explicit 2025/26 fixture ids")
    ap.add_argument("--artifact", default=None, help="model artifact path")
    args = ap.parse_args(argv)

    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        stop(f"scikit-learn is required to load the model artifact: {exc}")

    import numpy as np

    from models import artifact as artifact_mod
    from models.ablation import MODEL_B_COLUMNS
    from models.candidate_contract import predict_candidate
    from models.config import (
        FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS, X_EXCLUDED_COLUMNS,
    )
    from models.data import load_supervised_dataset

    # ---------------- STEP 1: leakage trace ----------------
    rule("STEP 1 -- LEAKAGE TRACE (static, before anything is loaded)")
    # Label names are imported from the single source of truth rather than
    # restated here, so this tool contains no label string of its own --
    # otherwise it would have to name the columns in order to prove it
    # never names them.
    from features.storage import LABEL_COLUMNS
    for label in LABEL_COLUMNS:
        check(f"{label} is in X_EXCLUDED_COLUMNS (stripped before X exists)",
              label in X_EXCLUDED_COLUMNS)
    this_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    consts = _non_docstring_strings(this_tree)
    for label in LABEL_COLUMNS:
        check(f"this tool contains no executable reference to {label}",
              label not in consts)
    check("no label name appears anywhere in this tool's own code",
          not any(lbl in consts for lbl in LABEL_COLUMNS))
    called = {n.func.attr for n in ast.walk(this_tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for banned in ("fit", "fit_transform", "partial_fit"):
        check(f"this tool never calls .{banned}()", banned not in called)

    # ---------------- artifact ----------------
    rule("STEP 2 -- ARTIFACT + BLINDNESS VERIFICATION")
    path = Path(args.artifact) if args.artifact else artifact_mod.DEFAULT_ARTIFACT_PATH
    if not path.is_absolute():
        path = REPO / path
    art_md5 = md5(path)
    art = artifact_mod.load(path)
    print(f"  artifact         : {path}")
    print(f"  artifact md5     : {art_md5}")
    check("contract is the frozen 80 columns, in order",
          tuple(art.feature_columns) == tuple(MODEL_B_COLUMNS))
    check("model_version == v1.0", art.model_version == "v1.0")
    check("class_order == H/D/A", art.class_order == ["H", "D", "A"])

    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    test_ids, train_ids = set(), set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    for s in FINAL_TRAIN_SEASONS:
        train_ids.update(SEASON_NAME_TO_IDS[s])

    # Which partition did the median imputer actually see? SimpleImputer
    # stores the per-column medians it was fitted on, so recomputing both
    # candidates and comparing identifies the training partition without
    # trusting anyone's recollection.
    numeric = [c for c in MODEL_B_COLUMNS if c != "competition_id"]
    stats = np.asarray(art.preprocessor._imputer.statistics_, dtype=float)
    blind_mask = ds.metadata["season_id"].isin(train_ids).values
    cand = {
        "blind (FINAL_TRAIN_SEASONS, excludes 2025/26)":
            ds.X[blind_mask][numeric].median(skipna=True).to_numpy(dtype=float),
        "all rows (INCLUDES 2025/26)":
            ds.X[numeric].median(skipna=True).to_numpy(dtype=float),
    }
    print(f"\n  imputer statistics_ length: {len(stats)}  (expected {len(numeric)})")
    check("imputer statistics_ length matches the numeric contract",
          len(stats) == len(numeric))
    dists = {k: float(np.nanmax(np.abs(stats - v))) for k, v in cand.items()}
    for k, d in dists.items():
        print(f"    max|diff| vs {k:46s} = {d:.6e}")
    nearest = min(dists, key=dists.get)
    print(f"\n  artifact partition identified as: {nearest}")
    check("artifact was trained WITHOUT 2025/26 (blind)",
          nearest.startswith("blind") and dists[nearest] < 1e-9,
          f"nearest-distance={dists[nearest]:.3e}")

    # ---------------- fixture selection ----------------
    rule("STEP 3 -- FIXTURE SELECTION (2025/26 only, from matches.db)")
    mdb = REPO / "data/processed/matches.db"
    con = sqlite3.connect(f"file:{mdb.resolve()}?mode=ro", uri=True)
    try:
        q = ",".join(str(i) for i in sorted(test_ids))
        # status is fixture metadata, not an outcome: it says the match was
        # played, never who won. No label column is read here.
        rows = con.execute(
            f"SELECT fixture_id, home_name, away_name, unix FROM fixtures "
            f"WHERE season_id IN ({q}) AND status = 'FT' "
            f"ORDER BY unix ASC, fixture_id ASC").fetchall()
    finally:
        con.close()
    universe = {r[0]: (r[1], r[2]) for r in rows}
    ordered = [r[0] for r in rows]
    print(f"  2025/26 FT fixtures in matches.db: {len(ordered)}")

    have_features = set(ds.metadata["fixture_id"].tolist())
    if args.fixture_ids:
        try:
            selected = [int(x) for x in args.fixture_ids]
        except ValueError:
            stop(f"--fixture-ids must all be integers, got {args.fixture_ids}")
        bad = [f for f in selected if f not in universe]
        if bad:
            stop(f"not 2025/26 FT fixtures in matches.db: {bad[:10]}. "
                 "This tool accepts 2025/26 fixtures only.")
    else:
        if args.count < 1 or args.count > len(ordered):
            stop(f"--count must be between 1 and {len(ordered)}")
        # Deterministic and reproducible: evenly spaced across the season in
        # (unix, fixture_id) order. Taking the first N would concentrate on
        # opening matchdays, where season-window features are still empty.
        step = len(ordered) / args.count
        idx = sorted({int(i * step) for i in range(args.count)})
        selected = [ordered[i] for i in idx]
        print(f"  selection rule   : {args.count} evenly spaced across the season "
              f"in (unix, fixture_id) order -- deterministic")

    missing = [f for f in selected if f not in have_features]
    if missing:
        stop(f"no feature row in features.db for: {missing[:10]}")
    check(f"all {len(selected)} selected fixtures are 2025/26",
          all(f in universe for f in selected))
    check("all selected fixtures have a v1.0 feature row", not missing)

    # ---------------- predict ----------------
    rule("STEP 4 -- BLIND PREDICTION")
    mask = ds.metadata["fixture_id"].isin(selected).values
    meta = ds.metadata[mask]
    X = ds.X[mask][list(art.feature_columns)]
    check("X carries exactly the 80 contract columns", list(X.columns) == list(MODEL_B_COLUMNS))
    check("no excluded/label column reached X",
          not (set(X.columns) & set(X_EXCLUDED_COLUMNS)))
    out = predict_candidate(art.model, art.preprocessor, X, meta["fixture_id"].tolist())
    check("class order is H/D/A", list(out.class_order) == ["H", "D", "A"])

    order = {f: i for i, f in enumerate(out.fixture_ids)}
    print(f"\n{'fixture_id':>11}  {'home':22} {'away':22}  {'Home':>7} {'Draw':>7} "
          f"{'Away':>7}  predicted")
    print("-" * 96)
    for fid in selected:
        p = out.probabilities[order[fid]]
        home, away = universe[fid]
        pred = out.class_order[int(p.argmax())]
        print(f"{fid:>11}  {home[:22]:22} {away[:22]:22}  "
              f"{100 * p[0]:6.2f}% {100 * p[1]:6.2f}% {100 * p[2]:6.2f}%  {RESULT_LABEL[pred]}")

    rule("PREDICTION PHASE COMPLETE -- GROUND TRUTH NOT CONSULTED")
    print("These are BLIND OUT-OF-SAMPLE predictions: STEP 2 verified the artifact")
    print("was fitted on FINAL_TRAIN_SEASONS only, so none of these fixtures was in")
    print("its training data. No label column was read, queried or derived. Nothing")
    print("was saved and no aggregate metric was computed.")
    print("\nReveal the actual results with the SEPARATE command:")
    print(f"  python {REVEAL_TOOL} --fixture-ids " + " ".join(str(f) for f in selected))

    print("\n[integrity] nothing modified")
    for rel, exp in (("data/processed/features.db", "e7ebe7fc07040a5927683c35b6371e63"),
                     ("data/processed/matches.db", "fdeed042096fa1c851aaee6c84995247")):
        check(rel, md5(REPO / rel) == exp, md5(REPO / rel))
    check("model artifact unchanged", md5(path) == art_md5, md5(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
