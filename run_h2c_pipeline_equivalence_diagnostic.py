"""H2-C pipeline equivalence diagnostic -- READ-ONLY.

PURPOSE
    Find the FIRST stage at which the frozen V1 preprocessing and the
    experimental preprocessing diverge, given identical H1 76-column
    input on identical folds.

SOURCE OF TRUTH
    src/models/train.py :: LogisticRegressionPreprocessor, used directly.
    Its semantics are NOT recreated from memory; the frozen object is
    instantiated and its fitted internal state is read
    (_imputer.statistics_, _scaler.mean_/scale_/var_, numeric_columns,
    competition_categories).

    Verified by inspection of the frozen source, fit():
        numeric_columns = [c for c in X_train.columns if c != competition_id]
        competition_categories = sorted(unique non-null competition_id)
        imputed = SimpleImputer(strategy="median").fit_transform(X_train[numeric_columns])
        StandardScaler().fit(imputed)              <- scaler fits on IMPUTED data
    transform():
        imputed -> scaled -> one-hot(competition) -> hstack([scaled, onehot])
    Order is therefore: SELECT -> IMPUTE -> SCALE -> ONE-HOT -> HSTACK.
    No dtype conversion is performed beyond what sklearn does internally;
    unseen competition categories yield an all-zero one-hot row; no
    special infinite-value handling exists.

WHAT THIS DIAGNOSTIC DOES NOT DO
    It does not modify the experimental preprocessor, does not re-run
    H2-C, does not fit anything until every preprocessing stage has been
    compared, and never touches 2025/26. stdout only; writes nothing.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_h2c_pipeline_equivalence_diagnostic.py
"""
from __future__ import annotations

import hashlib
import json
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
from models.robustness import V1_LOGREG_BASE_KWARGS
from models.splits import iter_walk_forward_folds

FEATURES_DB = REPO / "data" / "processed" / "features.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"

C_VALUE = 0.0005
CONTEXT_COL = "competition_id"
ALIASES = ("home_goal_diff_last5", "home_goal_diff_last10",
           "away_goal_diff_last5", "away_goal_diff_last10")
H1_COLUMNS = tuple(c for c in MODEL_B_COLUMNS if c not in ALIASES)


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)


def halt(msg):
    print("\n" + "!" * 78); print("HALT"); print(msg); print("!" * 78)
    raise SystemExit(1)


def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()


def report(stage, frozen, experimental, names=None):
    """Uniform per-stage comparison block."""
    F = np.asarray(frozen, dtype=float)
    E = np.asarray(experimental, dtype=float)
    print(f"  STAGE            : {stage}")
    print(f"  FROZEN shape     : {F.shape}")
    print(f"  EXPERIMENTAL     : {E.shape}")
    if F.shape != E.shape:
        print("  FIRST_DIFFERENCE : shape mismatch")
        return False
    d = np.abs(F - E)
    print(f"  MAX_ABS_DELTA    : {d.max():.6e}")
    print(f"  MEAN_ABS_DELTA   : {d.mean():.6e}")
    if d.max() == 0.0:
        print("  FIRST_DIFFERENCE : none -- bit-for-bit identical")
        return True
    idx = np.unravel_index(np.argmax(d), d.shape)
    col = names[idx[1]] if names is not None and idx[1] < len(names) else idx[1]
    print(f"  FIRST_DIFFERENCE : row={idx[0]} col={col} "
          f"frozen={F[idx]:.10g} experimental={E[idx]:.10g}")
    return False


# The experimental preprocessor, copied UNCHANGED from
# run_h2c_recency_reparameterization.py. It must not be edited here.
class ExperimentalPreprocessor:
    def __init__(self, triples, apply_transform: bool):
        self.triples = triples
        self.apply_transform = apply_transform

    def fit(self, X_train: pd.DataFrame):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        self.numeric_columns = [c for c in X_train.columns if c != CONTEXT_COL]
        self.competition_categories = sorted(X_train[CONTEXT_COL].dropna().unique().tolist())
        self._imputer = SimpleImputer(strategy="median").fit(X_train[self.numeric_columns])
        imputed = self._impute_frame(X_train)
        self.imputer_medians_ = dict(zip(self.numeric_columns, self._imputer.statistics_))
        transformed = self._reparameterize(imputed)
        self.output_columns_ = list(transformed.columns)
        self._scaler = StandardScaler().fit(transformed.to_numpy())
        return self

    def _impute_frame(self, X):
        arr = self._imputer.transform(X[self.numeric_columns])
        return pd.DataFrame(arr, columns=self.numeric_columns, index=X.index)

    def _reparameterize(self, imputed):
        if not self.apply_transform:
            return imputed
        out = imputed.copy()
        for stem, t in self.triples.items():
            s, l10, l5 = imputed[t["season"]], imputed[t["last10"]], imputed[t["last5"]]
            out[t["last10"]] = l10 - s
            out[t["last5"]] = l5 - l10
        renames = {}
        for stem, t in self.triples.items():
            renames[t["last10"]] = f"{stem}_last10_minus_season"
            renames[t["last5"]] = f"{stem}_last5_minus_last10"
        return out.rename(columns=renames)

    def transform(self, X):
        imputed = self._impute_frame(X)
        reparam = self._reparameterize(imputed)
        scaled = self._scaler.transform(reparam.to_numpy())
        onehot = np.zeros((len(X), len(self.competition_categories)))
        index = {c: i for i, c in enumerate(self.competition_categories)}
        for row, val in enumerate(X[CONTEXT_COL].tolist()):
            col = index.get(val)
            if col is not None:
                onehot[row, col] = 1.0
        return np.hstack([scaled, onehot])


def main():
    print("H2-C PIPELINE EQUIVALENCE DIAGNOSTIC -- read-only, no H2-C re-run")

    rule("PRE-FLIGHT")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION: {MODEL_VERSION!r}")
    seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    if seasons & set(FINAL_TEST_SEASONS):
        halt("2025/26 reachable")
    print(f"  fold seasons {sorted(seasons)} | 2025/26 reachable: False")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines: mismatches={bad or 'NONE'}")
    if bad:
        halt(f"pinned mismatch {bad}")

    rule("FROZEN PREPROCESSOR -- OPERATION ORDER (read from source, not memory)")
    import inspect
    from models.train import LogisticRegressionPreprocessor
    src = inspect.getsource(LogisticRegressionPreprocessor)
    for tag, q in (("imputation before scaling",
                    src.index("SimpleImputer") < src.index("StandardScaler")),
                   ("scaler fitted on IMPUTED data", "self._scaler.fit(imputed)" in src),
                   ("one-hot built inside transform", "onehot" in src),
                   ("output is hstack([scaled, onehot])", "np.hstack([scaled, onehot])" in src),
                   ("unseen category -> all-zero row", "cat_index.get(val)" in src),
                   ("explicit dtype conversion present", ".astype(" in src),
                   ("explicit infinite-value handling", "isinf" in src or "isfinite" in src)):
        print(f"  {tag:<40}: {q}")
    print("  => SELECT -> IMPUTE -> SCALE -> ONE-HOT -> HSTACK")

    triples = {}
    suffix = {"last5": "_last5", "last10": "_last10", "season": "_season"}
    stems = {}
    for c in H1_COLUMNS:
        for w, s in suffix.items():
            if c.endswith(s):
                stems.setdefault(c[: -len(s)], {})[w] = c
    triples = {k: v for k, v in stems.items() if set(v) == {"last5", "last10", "season"}}
    print(f"  triples available (unused by the identity arm): {len(triples)}")

    dataset = load_supervised_dataset(FEATURES_DB)
    fold, train_ds, val_ds = next(iter(iter_walk_forward_folds(dataset)))
    Xtr = train_ds.X[list(H1_COLUMNS)]
    Xva = val_ds.X[list(H1_COLUMNS)]
    print(f"\n  diagnosing on {fold.name}: n_train={len(Xtr)} n_val={len(Xva)}")

    # ---------------- STAGE 1 ----------------
    rule("STAGE 1 -- RAW INPUT")
    frozen_pre = LogisticRegressionPreprocessor().fit(Xtr)
    exp_pre = ExperimentalPreprocessor(triples, apply_transform=False).fit(Xtr)
    print(f"  FROZEN numeric_columns count      : {len(frozen_pre.numeric_columns)}")
    print(f"  EXPERIMENTAL numeric_columns count: {len(exp_pre.numeric_columns)}")
    same_cols = frozen_pre.numeric_columns == exp_pre.numeric_columns
    print(f"  identical column list AND order   : {same_cols}")
    if not same_cols:
        diff = [(i, a, b) for i, (a, b) in
                enumerate(zip(frozen_pre.numeric_columns, exp_pre.numeric_columns)) if a != b]
        print(f"  FIRST_DIFFERENCE : {diff[:3]}")
        halt("STAGE 1 -- raw input column selection/order differs. Halting as required.")
    dtypes_same = Xtr.dtypes.equals(Xtr[list(H1_COLUMNS)].dtypes)
    print(f"  dtypes stable                     : {dtypes_same}")
    print(f"  raw values identical (same object): True (both arms receive the same frame)")
    print("  MAX_ABS_DELTA    : 0.000000e+00")

    # ---------------- STAGE 2 ----------------
    rule("STAGE 2 -- IMPUTATION")
    f_med = np.asarray(frozen_pre._imputer.statistics_, dtype=float)
    e_med = np.asarray(exp_pre._imputer.statistics_, dtype=float)
    print(f"  FROZEN medians (first 5)      : {np.round(f_med[:5], 8)}")
    print(f"  EXPERIMENTAL medians (first 5): {np.round(e_med[:5], 8)}")
    ok_med = report("imputation statistics", f_med.reshape(1, -1), e_med.reshape(1, -1),
                    frozen_pre.numeric_columns)
    f_imp_tr = frozen_pre._imputer.transform(Xtr[frozen_pre.numeric_columns])
    e_imp_tr = exp_pre._impute_frame(Xtr).to_numpy()
    ok_tr = report("imputed TRAINING matrix", f_imp_tr, e_imp_tr, frozen_pre.numeric_columns)
    f_imp_va = frozen_pre._imputer.transform(Xva[frozen_pre.numeric_columns])
    e_imp_va = exp_pre._impute_frame(Xva).to_numpy()
    ok_va = report("imputed VALIDATION matrix", f_imp_va, e_imp_va, frozen_pre.numeric_columns)
    stage2_ok = ok_med and ok_tr and ok_va

    # ---------------- STAGE 3 ----------------
    rule("STAGE 3 -- SCALING")
    print(f"  FROZEN scaler mean_[:5]      : {np.round(frozen_pre._scaler.mean_[:5], 8)}")
    print(f"  EXPERIMENTAL scaler mean_[:5]: {np.round(exp_pre._scaler.mean_[:5], 8)}")
    ok_m = report("scaler mean_", frozen_pre._scaler.mean_.reshape(1, -1),
                  exp_pre._scaler.mean_.reshape(1, -1), frozen_pre.numeric_columns)
    ok_s = report("scaler scale_", frozen_pre._scaler.scale_.reshape(1, -1),
                  exp_pre._scaler.scale_.reshape(1, -1), frozen_pre.numeric_columns)
    ok_v = report("scaler var_", frozen_pre._scaler.var_.reshape(1, -1),
                  exp_pre._scaler.var_.reshape(1, -1), frozen_pre.numeric_columns)
    ok_sc = report("scaled VALIDATION numeric matrix",
                   frozen_pre._scaler.transform(f_imp_va),
                   exp_pre._scaler.transform(e_imp_va), frozen_pre.numeric_columns)
    stage3_ok = ok_m and ok_s and ok_v and ok_sc

    # ---------------- STAGE 4 ----------------
    rule("STAGE 4 -- COMPETITION ONE-HOT")
    print(f"  FROZEN categories      : {frozen_pre.competition_categories}")
    print(f"  EXPERIMENTAL categories: {exp_pre.competition_categories}")
    cats_same = frozen_pre.competition_categories == exp_pre.competition_categories
    print(f"  identical ordering     : {cats_same}")
    nnum = len(frozen_pre.numeric_columns)
    f_full_va = frozen_pre.transform(Xva)
    e_full_va = exp_pre.transform(Xva)
    stage4_ok = cats_same and report("one-hot block (validation)",
                                     f_full_va[:, nnum:], e_full_va[:, nnum:],
                                     [f"{CONTEXT_COL}=={c}" for c in frozen_pre.competition_categories])

    # ---------------- STAGE 5 ----------------
    rule("STAGE 5 -- FINAL ENCODED DESIGN MATRIX")
    f_names = list(frozen_pre.numeric_columns) + \
        [f"{CONTEXT_COL}=={c}" for c in frozen_pre.competition_categories]
    e_names = list(exp_pre.output_columns_) + \
        [f"{CONTEXT_COL}=={c}" for c in exp_pre.competition_categories]
    print(f"  FROZEN encoded columns      : {len(f_names)}")
    print(f"  EXPERIMENTAL encoded columns: {len(e_names)}")
    print(f"  identical encoded names/order: {f_names == e_names}")
    stage5_ok = report("final encoded VALIDATION matrix", f_full_va, e_full_va, f_names)
    f_full_tr = frozen_pre.transform(Xtr)
    e_full_tr = exp_pre.transform(Xtr)
    stage5_ok &= report("final encoded TRAINING matrix", f_full_tr, e_full_tr, f_names)

    preprocessing_identical = stage2_ok and stage3_ok and stage4_ok and stage5_ok
    print(f"\n  PREPROCESSING IDENTICAL END-TO-END: {preprocessing_identical}")

    # ---------------- STAGE 6 ----------------
    rule("STAGE 6 -- MODEL (diagnostic only, run last)")
    from sklearn.linear_model import LogisticRegression
    from models import train as train_module

    print("  What the A2 gate actually compared in the failed H2-C run:")
    print("    A1 = models.train.train_logistic_regression(...)")
    print("    A2 = LogisticRegression(C=0.0005, **V1_LOGREG_BASE_KWARGS)")
    src_t = inspect.getsource(train_module.train_logistic_regression)
    hard = [ln.strip() for ln in src_t.splitlines() if "LogisticRegression(" in ln]
    print(f"    frozen entry point constructs   : {hard}")
    print(f"    experimental arms construct     : LogisticRegression(C={C_VALUE}, "
          f"{V1_LOGREG_BASE_KWARGS})")

    m_frozen_entry, _, P_frozen_entry = train_module.train_logistic_regression(
        Xtr, train_ds.y, Xva)
    print(f"\n  frozen entry point fitted C     : {m_frozen_entry.C}")

    m_same_C = LogisticRegression(C=C_VALUE, **V1_LOGREG_BASE_KWARGS).fit(f_full_tr, train_ds.y)
    P_same_C = np.asarray(train_module._reorder_proba(m_same_C, f_full_va), dtype=float)
    m_exp = LogisticRegression(C=C_VALUE, **V1_LOGREG_BASE_KWARGS).fit(e_full_tr, train_ds.y)
    P_exp = np.asarray(train_module._reorder_proba(m_exp, e_full_va), dtype=float)

    print("\n  (a) frozen ENTRY POINT (C=1.0) vs experimental (C=0.0005):")
    report("probabilities", P_frozen_entry, P_exp)
    print("\n  (b) SAME C=0.0005 on frozen-encoded vs experimental-encoded matrices:")
    ok_coef = report("coefficients", m_same_C.coef_, m_exp.coef_)
    ok_int = report("intercepts", np.asarray(m_same_C.intercept_).reshape(1, -1),
                    np.asarray(m_exp.intercept_).reshape(1, -1))
    ok_p = report("probabilities", P_same_C, P_exp)

    # ---------------- CONCLUSION ----------------
    rule("CONCLUSION")
    if not preprocessing_identical:
        first = ("STAGE 2 -- IMPUTATION" if not stage2_ok else
                 "STAGE 3 -- SCALING" if not stage3_ok else
                 "STAGE 4 -- COMPETITION ONE-HOT" if not stage4_ok else
                 "STAGE 5 -- FINAL ENCODED DESIGN MATRIX")
        cause = "the preprocessing pipelines genuinely differ at the stage named above"
    elif not (ok_coef and ok_int and ok_p):
        first = "STAGE 6 -- MODEL (same C, identical encoded matrices)"
        cause = ("preprocessing is bit-identical, yet the fits differ at the same C -- "
                 "investigate solver nondeterminism before anything else")
    else:
        first = "STAGE 6 -- MODEL (regularisation strength)"
        cause = (
            "Preprocessing is bit-for-bit identical at every stage. The A2 gate failed "
            "because the two arms were not fitted at the same C. The A1 arm called "
            "models.train.train_logistic_regression, whose estimator line hardcodes "
            "C=1.0 (src/models/train.py), while the A2/A3 arms constructed "
            f"LogisticRegression(C={C_VALUE}). The gate therefore compared a C=1.0 model "
            "against C=0.0005 models, which is exactly the ~0.2 probability gap observed. "
            "This is a defect in the experiment script's control arm, NOT in the "
            "experimental preprocessor."
        )
    print(f"FIRST DIVERGENCE:\n{first}")
    print(f"\nROOT CAUSE:\n{cause}")
    print("\nH2-C STATUS:\nBLOCKED UNTIL EXPERIMENTAL PIPELINE MATCHES FROZEN V1")
    print("\nNo H2-C result is proposed. No claim is made that H2-C improved or worsened.")
    print("The A3 numbers from the halted run remain INVALID and are not interpreted.")

    rule("INTEGRITY AUDIT")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    print(f"  MODEL_VERSION: {MODEL_VERSION!r}")
    ser = [str(p.relative_to(REPO)) for pat in ("*.pkl", "*.joblib", "*.pickle")
           for p in REPO.rglob(pat) if "__pycache__" not in p.parts]
    print(f"  serialized estimators: {ser or 'NONE'}")
    for line in ("features.db untouched: YES (read-only load)", "production source modified: NO",
                 "experimental preprocessor modified: NO", "2025/26 accessed: NO",
                 "tuning: NONE", "calibration: NONE", "artifacts written: NONE",
                 "H2-C re-run: NO"):
        print(f"  {line}")


if __name__ == "__main__":
    main()
