"""D-29 -- train the production V1 model and persist it.

    python train_v1_model.py [--out data/models/v1_logreg.pkl]

Uses the existing production path unchanged: `data.load_supervised_dataset`
on features.db, the frozen 80-column `ablation.MODEL_B_COLUMNS`, the
existing `train.LogisticRegressionPreprocessor` (median imputation,
StandardScaler, competition_id one-hot) and
`LogisticRegression(C=1.0, max_iter=2000, random_state=0)`. Nothing is
duplicated, tuned, or reimplemented here.

TRAINING PARTITION. All labelled rows in features.db, which includes
2025/26. That season is the spent final test, and D-29 forbids using it
for *evaluation or tuning* -- neither happens here. This script fits a
deployable artifact and computes no metric of any kind. Use
--exclude-final-test if you would rather the artifact never see 2025/26;
the season breakdown is printed either way so the choice is always
visible in the log.

Writes exactly one file: the model artifact. features.db, matches.db and
every pinned input are verified byte-identical before and after.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}


def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 66)
    print("STOP -- training halted. No artifact written.")
    print(msg)
    print("!" * 66)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        stop(f"stop condition: {label}")


def verify_integrity(when: str):
    print(f"\n[{when}] artifact integrity")
    for rel, exp in EXPECTED_MD5.items():
        check(rel, md5(REPO / rel) == exp, md5(REPO / rel))
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if md5(REPO / r) != v["expected"]]
    check("13/13 LOCKED_INPUTS unchanged", not bad, str(bad))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train and persist the V1 production model.")
    ap.add_argument("--out", default=None, help="artifact path (default: data/models/v1_logreg.pkl)")
    ap.add_argument("--exclude-final-test", action="store_true",
                    help="train only on FINAL_TRAIN_SEASONS (omit 2025/26)")
    args = ap.parse_args(argv)

    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        stop(f"scikit-learn is required and not installed: {exc}. Do not substitute "
             "another estimator and do not hand-roll one.")

    from models import artifact as artifact_mod
    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        FINAL_TRAIN_SEASONS, MODEL_VERSION, REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS,
    )
    from models.data import load_supervised_dataset
    from models.train import train_logistic_regression

    out = Path(args.out) if args.out else artifact_mod.DEFAULT_ARTIFACT_PATH
    if not out.is_absolute():
        out = REPO / out

    verify_integrity("before")
    print("\n[before] contract and version")
    check("MODEL_B_COLUMNS == 80", len(MODEL_B_COLUMNS) == 80)
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0", MODEL_VERSION)
    check("REQUIRED_FEATURE_VERSION == v1.0",
          REQUIRED_FEATURE_VERSION == "v1.0", REQUIRED_FEATURE_VERSION)

    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    versions = set(ds.metadata["feature_version"].unique())
    check("exactly one feature_version", len(versions) == 1, str(versions))
    check("feature_version == v1.0", versions == {"v1.0"})
    check("labels are exactly H/D/A", set(ds.y.unique()) <= set(CLASS_ORDER),
          str(sorted(ds.y.unique())))
    missing = [c for c in MODEL_B_COLUMNS if c not in ds.X.columns]
    check("all 80 contract columns present in the dataset", not missing, str(missing[:5]))

    if args.exclude_final_test:
        keep_ids = set()
        for name in FINAL_TRAIN_SEASONS:
            keep_ids.update(SEASON_NAME_TO_IDS[name])
        mask = ds.metadata["season_id"].isin(keep_ids).values
        X, y, meta = ds.X[mask], ds.y[mask], ds.metadata[mask]
        partition = f"FINAL_TRAIN_SEASONS {list(FINAL_TRAIN_SEASONS)}"
    else:
        X, y, meta = ds.X, ds.y, ds.metadata
        partition = "all labelled rows in features.db"

    X_train = X[list(MODEL_B_COLUMNS)]

    print("\n[training partition]", partition)
    counts = meta.groupby("season_id").size()
    for sid, n in counts.items():
        print(f"    season_id {sid:<8} rows={n}")

    # train_logistic_regression fits preprocessing + estimator on the
    # training frame and returns validation probabilities. There is no
    # held-out set here (this is artifact training, not evaluation), so
    # the training frame is passed as the third argument and the returned
    # probabilities are discarded -- deliberately unused, never reported.
    model, preprocessor, _unused_in_sample_probabilities = train_logistic_regression(
        X_train, y, X_train)

    path = artifact_mod.save(model, preprocessor, out)

    print("\n" + "=" * 50)
    print("V1 MODEL TRAINING")
    print("=" * 50)
    print(f"Feature version : {sorted(versions)[0]}")
    print(f"Training rows   : {len(y)}")
    print(f"Feature count   : {len(MODEL_B_COLUMNS)}")
    print(f"Estimator       : {type(model).__name__}")
    print(f"C               : {model.C}")
    print(f"max_iter        : {model.max_iter}")
    print(f"random_state    : {model.random_state}")
    print(f"Classes         : {' / '.join(CLASS_ORDER)}")
    print(f"Model saved     : {path}")
    print("=" * 50)

    verify_integrity("after")

    print("\n[reload] verifying the artifact in a FRESH Python process")
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from models.artifact import load\n"
        "from models.ablation import MODEL_B_COLUMNS\n"
        "a = load(r'%s')\n"
        "assert tuple(a.feature_columns) == tuple(MODEL_B_COLUMNS)\n"
        "assert a.model_version == 'v1.0'\n"
        "assert a.class_order == ['H', 'D', 'A']\n"
        "print('  reload OK  features=%%d  version=%%s  classes=%%s'\n"
        "      %% (len(a.feature_columns), a.model_version, '/'.join(a.class_order)))\n"
    ) % (REPO / "src", path)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(res.stdout.rstrip() or res.stderr.rstrip())
    check("fresh-process reload succeeded", res.returncode == 0)

    print("\nTraining complete. Predict a fixture with:")
    print("  python predict_match.py --fixture-id <id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
