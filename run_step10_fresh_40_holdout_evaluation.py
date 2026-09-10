"""STEP 10 -- FRESH 40-MATCH HOLDOUT RE-EVALUATION of the frozen V1 artifact.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step10_fresh_40_holdout_evaluation.py

This is a fresh 40-match holdout re-evaluation of the existing frozen V1
artifact. The previous 20-match universe was not reconstructed and is not
assumed to be the same set.

EVALUATION ONLY. The existing data/models/v1_logreg.pkl is loaded and used
as-is. fit() is never called. Nothing is retrained, refitted, tuned,
recalibrated, thresholded or reweighted. No production file, database, model
or locked input is modified, and this script writes nothing at all.

SELECTION RULE -- FIXED AND PRINTED BEFORE ANY PREDICTION
    universe : matches.db, season_id IN FINAL_TEST_SEASONS AND status = 'FT'
    ordering : ORDER BY unix ASC, fixture_id ASC        (total; ties broken by id)
    step     : L / 40                                    (L = universe size)
    indices  : sorted({int(i * step) for i in range(40)})
    selected : [ordered[i] for i in indices]
Deterministic and reproducible: no randomness, no seed, no sampling, no manual
list, no date or competition filter, no outcome-dependent choice. Evenly spaced
rather than the first 40, because the first 40 would concentrate on opening
matchdays where season-window features are still empty.

`status` is fixture metadata -- it records that the match was played, never who
won -- so using it to define the universe consults no outcome. Ground truth is
read only in the evaluation section, after every probability is fixed.

WHAT THIS CANNOT DO
40 matches cannot support a significance test, a calibration claim or a
generalisation claim, and none is computed or made here. The 2025/26 season has
already been used in this investigation, so this is a holdout re-evaluation and
NOT a blind test of unseen data.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

N_MATCHES = 40
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_MD5 = "5e504427712b35778bb8a62a8496c7cd"
TRAIN_PY_MD5 = "21425459195311492f49e73f5ae38fe0"
FORBIDDEN_SEASONS = [602681, 667780, 725788, 725793, 762170]

#: Production source strings this script refuses to run without.
PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)
LABEL = {"H": "HOME WIN", "D": "DRAW", "A": "AWAY WIN"}

DISCLAIMER = (
    "This is a fresh 40-match holdout re-evaluation of the existing frozen V1 "
    "artifact.\nThe previous 20-match universe was not reconstructed and is not "
    "assumed to be the same set."
)


def rule(t): print("\n" + "=" * 126); print(t); print("=" * 126)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 126)
    print("ABORT -- STEP 10 halted. No evaluation is produced.")
    print(msg)
    print("!" * 126)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        stop(f"stop condition: {label}")


def snapshot():
    """Every artifact whose integrity is asserted, hashed before and after."""
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    snap["src/models/train.py"] = md5(REPO / "src/models/train.py")
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["data/models/v1_logreg.pkl"] = md5(art)
    lg = REPO / "data/processed/leagues"
    if lg.exists():
        for p in sorted(lg.glob("*.db")):
            snap[f"leagues/{p.name}"] = md5(p)
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def block_report(title, sel, meta, P, pred, actual, order):
    """Descriptive summary for a set of fixtures. No inference, no test."""
    correct = pred == actual
    n = len(sel)
    print(f"\n  --- {title} (n={n}) ---")
    print(f"    correct            : {int(correct.sum())}")
    print(f"    incorrect          : {int(n - correct.sum())}")
    print(f"    accuracy           : {correct.sum() / n:.4f}  ({100 * correct.sum() / n:.1f}%)")
    for c in order:
        pm = pred == c
        am = actual == c
        print(f"    {LABEL[c]:9s} predicted {int(pm.sum()):3d}   correct {int((pm & am).sum()):3d}"
              f"   actual {int(am.sum()):3d}")
    print(f"    mean confidence    : {P.max(axis=1).mean():.4f}")


def main():
    # ================================================================= RULE 1
    rule("RULE 1 -- PRODUCTION SOURCE TRACE (read from source; nothing from memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:78]}", tok in tsrc)
    check("train.py: C == 1.0", "C=1.0" in tsrc)
    check("train.py: random_state == 0", "random_state=0" in tsrc)
    check("train.py: max_iter == 2000", "max_iter=2000" in tsrc)

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
        X_EXCLUDED_COLUMNS,
    )
    from models.data import load_supervised_dataset

    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("production feature contract == exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric production features == exactly 79", len(numeric) == 79)
    check("competition_id is the only categorical production column",
          set(MODEL_B_COLUMNS) - set(numeric) == {RECOMMENDED_CONTEXT_FEATURE})
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")

    # ================================================================= STEP 1
    rule("STEP 1 -- INTEGRITY GATE (before anything is read or predicted)")
    print(f"  Python {sys.version.split()[0]}   numpy {np.__version__}   "
          f"pandas {pd.__version__}   sklearn {sklearn.__version__}")
    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(f"{rel} unchanged", before[rel] == exp, before[rel])
    check("src/models/train.py unchanged",
          before["src/models/train.py"] == TRAIN_PY_MD5, before["src/models/train.py"])
    check("data/models/v1_logreg.pkl unchanged",
          before.get("data/models/v1_logreg.pkl") == ARTIFACT_MD5,
          str(before.get("data/models/v1_logreg.pkl")))
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if before[f"pin:{r}"] != v["expected"]]
    check(f"{len(pins)}/{len(pins)} LOCKED_INPUTS unchanged", not bad, str(bad))

    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    print(f"  2025/26 final-test season ids : {sorted(test_ids)}")
    check("final-test season ids match the documented forbidden set",
          sorted(test_ids) == sorted(FORBIDDEN_SEASONS))

    # ================================================================= STEP 2
    rule("STEP 2 -- ARTIFACT LOAD + CONFIGURATION VERIFICATION (no fit)")
    art_path = REPO / "data/models/v1_logreg.pkl"
    art = artifact_mod.load(art_path)
    print(f"  artifact      : {art_path}")
    print(f"  artifact md5  : {before['data/models/v1_logreg.pkl']}")
    check("artifact contract == the frozen 80 columns, in contract order",
          tuple(art.feature_columns) == tuple(MODEL_B_COLUMNS))
    check("artifact model_version == v1.0", art.model_version == "v1.0")
    check("artifact class_order == ['H','D','A']", art.class_order == ["H", "D", "A"])
    check("artifact estimator C == 1.0", art.model.C == 1.0, str(art.model.C))
    check("artifact estimator max_iter == 2000", art.model.max_iter == 2000,
          str(art.model.max_iter))
    check("artifact estimator random_state == 0", art.model.random_state == 0,
          str(art.model.random_state))
    # sklearn orders predict_proba's columns by model.classes_, which for string
    # labels is ALPHABETICAL -- A, D, H -- NOT the project's CLASS_ORDER H, D, A.
    # Asserting classes_ == CLASS_ORDER would therefore be an invalid assertion
    # about an implementation detail sklearn owns. What must hold is that the
    # estimator was fitted on exactly the three allowed labels; the column
    # mapping itself is verified empirically in STEP 5, not assumed here.
    print(f"  artifact estimator native classes_ : {list(art.model.classes_)}  "
          f"(sklearn alphabetical order)")
    print(f"  project CLASS_ORDER                : {list(CLASS_ORDER)}  "
          f"(probability column order)")
    check("artifact estimator was fitted on exactly the three allowed labels",
          set(art.model.classes_) == set(CLASS_ORDER),
          str(sorted(art.model.classes_)))
    check("every CLASS_ORDER label is present in the estimator's classes_",
          all(c in list(art.model.classes_) for c in CLASS_ORDER))

    ds = load_supervised_dataset(REPO / "data/processed/features.db")

    # The imputer stores the medians it was fitted on, so recomputing both
    # candidate partitions identifies which one the artifact carries. Reported
    # for the record: it establishes what "holdout" means for this artifact.
    train_ids = set()
    for s in FINAL_TRAIN_SEASONS:
        train_ids.update(SEASON_NAME_TO_IDS[s])
    stats = np.asarray(art.preprocessor._imputer.statistics_, dtype=float)
    check("imputer statistics_ length matches the numeric contract",
          len(stats) == len(numeric), f"{len(stats)} vs {len(numeric)}")
    cand = {
        "trained WITHOUT 2025/26 (holdout genuinely held out)":
            ds.X[ds.metadata["season_id"].isin(train_ids).values][numeric]
            .median(skipna=True).to_numpy(dtype=float),
        "trained WITH 2025/26 (would make this in-sample)":
            ds.X[numeric].median(skipna=True).to_numpy(dtype=float),
    }
    dists = {k: float(np.nanmax(np.abs(stats - v))) for k, v in cand.items()}
    print()
    for k, v in dists.items():
        print(f"    imputer max|diff| vs partition {k:52s} = {v:.6e}")
    nearest = min(dists, key=dists.get)
    print(f"    identified training partition: {nearest}")
    check("the artifact was NOT fitted on the 2025/26 holdout",
          nearest.startswith("trained WITHOUT") and dists[nearest] < 1e-9,
          f"nearest-distance={dists[nearest]:.3e}")

    # ================================================================= STEP 3
    rule("STEP 3 -- 40-MATCH SELECTION RULE (fixed and printed BEFORE any prediction)")
    print("    universe : matches.db, season_id IN FINAL_TEST_SEASONS AND status = 'FT'")
    print("    ordering : ORDER BY unix ASC, fixture_id ASC")
    print(f"    step     : L / {N_MATCHES}")
    print(f"    indices  : sorted({{int(i * step) for i in range({N_MATCHES})}})")
    print("    selected : [ordered[i] for i in indices]")
    print("  Deterministic. No randomness, no seed, no sampling, no manual list, no date")
    print("  or competition filter, and no outcome-dependent choice of any kind.")

    mdb = REPO / "data/processed/matches.db"
    con = sqlite3.connect(f"file:{mdb.resolve()}?mode=ro", uri=True)
    try:
        q = ",".join(str(i) for i in sorted(test_ids))
        rows = con.execute(
            f"SELECT fixture_id, home_name, away_name, unix, season_id, competition_id, "
            f"home_goals, away_goals FROM fixtures "
            f"WHERE season_id IN ({q}) AND status = 'FT' "
            f"ORDER BY unix ASC, fixture_id ASC").fetchall()
    finally:
        con.close()
    ordered = [r[0] for r in rows]
    meta = {r[0]: r for r in rows}
    L = len(ordered)
    print(f"\n  ordered 2025/26 FT universe size L : {L}")
    check("universe has no duplicate fixture_id", len(set(ordered)) == L)
    check("universe has no NULL unix (ordering is total)",
          all(r[3] is not None for r in rows))
    check(f"universe is large enough for {N_MATCHES}", L >= N_MATCHES, f"L={L}")

    step = L / N_MATCHES
    idx = sorted({int(i * step) for i in range(N_MATCHES)})
    selected = [ordered[i] for i in idx]
    print(f"  step = {L}/{N_MATCHES} = {step}")
    check(f"selection yields exactly {N_MATCHES} unique fixtures",
          len(selected) == len(set(selected)) == N_MATCHES,
          f"{len(set(selected))} unique from {len(idx)} indices")

    # ================================================================= STEP 4
    rule("STEP 4 -- 40-MATCH UNIVERSE AUDIT (printed before prediction)")
    print("| # | fixture_id | date | home | away | season_id | competition_id |")
    print("|---:|---:|---|---|---|---:|---:|")
    for i, f in enumerate(selected, 1):
        r = meta[f]
        d = datetime.datetime.utcfromtimestamp(r[3]).strftime("%Y-%m-%d")
        print(f"| {i} | {f} | {d} | {r[1][:24]} | {r[2][:24]} | {r[4]} | {r[5]} |")

    print()
    check(f"exactly {N_MATCHES} fixtures", len(selected) == N_MATCHES)
    check("no duplicate fixture ids", len(set(selected)) == N_MATCHES)
    check("all belong to the 2025/26 final-test universe",
          all(meta[f][4] in test_ids for f in selected))
    have_features = set(ds.metadata["fixture_id"].tolist())
    missing = [f for f in selected if f not in have_features]
    check("all have a v1.0 feature row in features.db", not missing, str(missing[:10]))

    mask = ds.metadata["fixture_id"].isin(selected).values
    X = ds.X[mask][list(art.feature_columns)]
    m = ds.metadata[mask].reset_index(drop=True)
    check(f"feature frame has exactly {N_MATCHES} rows", len(X) == N_MATCHES, str(len(X)))
    check("frame carries exactly the frozen 80-column contract, in order",
          list(X.columns) == list(MODEL_B_COLUMNS))
    n_extra = X.shape[1] - 80
    n_close = sum(1 for c in X.columns if c.startswith("closeness_"))
    n_label = len(set(X.columns) & set(X_EXCLUDED_COLUMNS))
    print(f"\n    production columns        : {X.shape[1]}")
    print(f"    additional columns        : {n_extra}")
    print(f"    closeness columns         : {n_close}")
    print(f"    post-match/label columns  : {n_label}")
    print(f"    new information sources   : 0")
    print(f"    model changes             : 0")
    check("zero additional features", n_extra == 0)
    check("zero closeness features", n_close == 0)
    check("zero post-match / label columns reached X", n_label == 0)

    # ================================================================= STEP 5
    rule("STEP 5 -- PREDICTION (frozen artifact only; fit() is never called)")
    out = predict_candidate(art.model, art.preprocessor, X, m["fixture_id"].tolist())
    order = list(out.class_order)
    check("prediction class order == ['H','D','A']", order == ["H", "D", "A"])
    pos = {f: i for i, f in enumerate(out.fixture_ids)}
    P = np.array([np.asarray(out.probabilities, dtype=float)[pos[f]] for f in selected])
    check(f"exactly {N_MATCHES} probability rows", P.shape == (N_MATCHES, 3), str(P.shape))
    check("every probability row sums to 1",
          bool(np.all(np.abs(P.sum(axis=1) - 1.0) < 1e-9)),
          f"max|sum-1| = {float(np.max(np.abs(P.sum(axis=1) - 1.0))):.3e}")

    # ---- EXPLICIT PROBABILITY-COLUMN MAPPING VERIFICATION -------------------
    # Production reorders predict_proba's columns from the estimator's native
    # alphabetical order into CLASS_ORDER (train._reorder_proba). That mapping is
    # proved here empirically rather than assumed: the raw estimator output is
    # recomputed and the expected permutation applied by hand, then compared
    # against what the production path returned. predict_proba is inference only
    # -- fit() is not called and the preprocessor is transform-only.
    print("\n  probability-column mapping verification")
    encoded = art.preprocessor.transform(X)
    raw = np.asarray(art.model.predict_proba(encoded), dtype=float)
    class_to_col = {c: i for i, c in enumerate(art.model.classes_)}
    permutation = [class_to_col[c] for c in order]
    print(f"    estimator native column order : {list(art.model.classes_)}")
    print(f"    required CLASS_ORDER          : {order}")
    print(f"    permutation applied           : {permutation}")
    for k, c in enumerate(order):
        print(f"      CLASS_ORDER[{k}] = '{c}'  <-  raw predict_proba column "
              f"{class_to_col[c]} ('{art.model.classes_[class_to_col[c]]}')")
    check("permutation is a true bijection over the three classes",
          sorted(permutation) == [0, 1, 2], str(permutation))
    expected = raw[:, permutation]
    produced = np.asarray(out.probabilities, dtype=float)
    max_dev = float(np.max(np.abs(produced - expected)))
    check("production probabilities equal the explicitly re-permuted raw output",
          max_dev == 0.0, f"max|deviation| = {max_dev:.3e}")
    # A permutation is only proved correct if a WRONG one is measurably different;
    # otherwise the equality above could hold for a degenerate reason.
    wrong = raw[:, [permutation[1], permutation[2], permutation[0]]]
    check("control: a deliberately rotated permutation does NOT match",
          float(np.max(np.abs(produced - wrong))) > 1e-6,
          f"max|deviation| = {float(np.max(np.abs(produced - wrong))):.3e}")
    print(f"    column 0 = P(H), column 1 = P(D), column 2 = P(A)  -- VERIFIED")
    # ------------------------------------------------------------------------

    pred = np.array(order)[P.argmax(axis=1)]

    # Ground truth is read only now, after every probability is fixed.
    label_by_fixture = dict(zip(ds.metadata["fixture_id"][mask].to_numpy(),
                                ds.y[mask].to_numpy()))
    actual, disagree = [], []
    for f in selected:
        hg, ag = meta[f][6], meta[f][7]
        g = "H" if hg > ag else ("A" if ag > hg else "D")
        if g != label_by_fixture[f]:
            disagree.append(f)
        actual.append(g)
    actual = np.array(actual)
    check("matches.db goals and features.db label_result agree on every fixture",
          not disagree, str(disagree))
    correct = pred == actual

    print("\n| # | fixture_id | date | home | away | P(H) | P(D) | P(A) | predicted "
          "| confidence | actual | correct |")
    print("|---:|---:|---|---|---|---:|---:|---:|---|---:|---|---|")
    for i, f in enumerate(selected):
        r = meta[f]
        d = datetime.datetime.utcfromtimestamp(r[3]).strftime("%Y-%m-%d")
        print(f"| {i+1} | {f} | {d} | {r[1][:20]} | {r[2][:20]} | {P[i,0]:.4f} | "
              f"{P[i,1]:.4f} | {P[i,2]:.4f} | {LABEL[pred[i]]} | {P[i].max():.4f} | "
              f"{LABEL[actual[i]]} | {'YES' if correct[i] else 'NO'} |")

    # ================================================================= STEP 6
    rule("STEP 6 -- 40-MATCH RESULTS")
    n_ok = int(correct.sum())
    print(f"  1. correct predictions   : {n_ok}")
    print(f"  2. accuracy              : {n_ok}/{N_MATCHES} = {n_ok / N_MATCHES:.4f} "
          f"({100 * n_ok / N_MATCHES:.1f}%)")
    print(f"  3. incorrect predictions : {N_MATCHES - n_ok}")
    for i, c in enumerate(order, 4):
        pm = pred == c
        print(f"  {i}. {LABEL[c]:9s} predictions correct : {int((pm & (actual == c)).sum())} "
              f"of {int(pm.sum())} predicted")
    print("\n  7. actual result distribution:")
    for c in order:
        n = int((actual == c).sum())
        print(f"       {LABEL[c]:9s} {n:3d}  ({100 * n / N_MATCHES:5.1f}%)")
    print("  8. predicted result distribution:")
    for c in order:
        n = int((pred == c).sum())
        print(f"       {LABEL[c]:9s} {n:3d}  ({100 * n / N_MATCHES:5.1f}%)")

    print("\n  confusion matrix (rows = predicted, cols = actual)")
    print("            Actual")
    print("            H    D    A")
    for pc in order:
        cells = "  ".join(f"{int(((pred == pc) & (actual == ac)).sum()):3d}" for ac in order)
        print(f"    Pred {pc}   {cells}")

    # ================================================================= STEP 7
    rule("STEP 7 -- DESCRIPTIVE HALVES OF THIS NEWLY SELECTED 40")
    print("  These are the first and second chronological halves of the 40 fixtures")
    print("  selected above. Neither is the historical 20-match test, and neither is")
    print("  claimed to correspond to it in any way.")
    h = N_MATCHES // 2
    block_report("FIRST 20 of this 40 (chronologically earlier)", selected[:h],
                 meta, P[:h], pred[:h], actual[:h], order)
    block_report("SECOND 20 of this 40 (chronologically later)", selected[h:],
                 meta, P[h:], pred[h:], actual[h:], order)
    block_report("COMBINED 40", selected, meta, P, pred, actual, order)
    print("\n  The two halves differ in calendar position, so any difference between")
    print("  them also reflects how much season history each fixture's features carry.")
    print("  No significance test, calibration claim or statistical conclusion is drawn")
    print("  from 40 matches, or from 20-match halves of them.")

    # ================================================================= STEP 8
    rule("STEP 8 -- INTERPRETATION")
    print("  " + DISCLAIMER.replace("\n", "\n  "))
    print("\n  2025/26 has already been used in this investigation, so this is a HOLDOUT")
    print("  RE-EVALUATION, not a blind test, not an unseen test, not an independent")
    print("  test. It is descriptive evidence about the frozen V1 artifact on this")
    print("  holdout. It does not establish generalisation to unseen future data, does")
    print("  not replace a future 2026/27 blind test, and is not proof of any model")
    print("  improvement. The model was not changed, so no production change follows.")

    # ================================================================= STEP 9
    rule("STEP 9 -- POST-RUN INTEGRITY CHECK")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified anywhere", not changed, str(changed))
    print(f"    no production files modified : CONFIRMED")
    print(f"    no database modified         : CONFIRMED")
    print(f"    no model modified            : CONFIRMED")
    print(f"    no artifact modified         : CONFIRMED")
    print(f"    no locked input modified     : CONFIRMED ({len(pins)}/{len(pins)})")
    print(f"    retrained                    : NO")
    print(f"    fit() called                 : NO")
    print(f"    tuning / recalibration       : NO")
    print(f"    files created by this script : NONE")
    print()
    for k in ("data/processed/features.db", "data/processed/matches.db",
              "data/models/v1_logreg.pkl", "src/models/train.py"):
        print(f"    {k:34s} {after[k]}")
    print(f"\n    evaluated fixture ids : {selected}")

    print("\n" + "=" * 126)
    print(f"STEP 10 COMPLETE -- FRESH 40-MATCH HOLDOUT RE-EVALUATION: "
          f"{n_ok}/{N_MATCHES} correct ({100 * n_ok / N_MATCHES:.1f}%)")
    print(DISCLAIMER)
    print("=" * 126)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
