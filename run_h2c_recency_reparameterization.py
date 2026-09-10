"""H2-C -- recency reparameterization with TRAINING-ONLY IMPUTATION FIRST.

CORE QUESTION
    Does replacing correlated absolute recency windows
        [season, last10, last5]
    with
        [season, last10-season, last5-last10]
    improve the H1 (76-column, C=0.0005) model on the validation folds?

WHY "-C"
    The naive transform creates NaNs whenever one operand is missing
    (+10,548 cells measured), which would confound representation with
    missing-data treatment. H2-C therefore IMPUTES FIRST (training-fold
    medians only) and TRANSFORMS SECOND, so the transform introduces
    exactly zero new NaNs.

ISOLATION OF THE VARIABLE
    Three arms are fitted, not two:
      A1  H1 control via the FROZEN LogisticRegressionPreprocessor
          -> must reproduce the recorded H1 numbers to <1e-9
      A2  H1 via the EXPERIMENTAL pipeline with the identity transform
          -> must be BYTE-IDENTICAL to A1
      A3  H2-C via the EXPERIMENTAL pipeline with T applied
    A2 exists solely to prove the experimental pipeline is faithful, so
    any A3-vs-A1 difference is attributable to the transform alone and
    not to the reimplemented plumbing.

NO PRODUCTION CHANGE
    src/models/* is imported read-only. Nothing is modified, nothing is
    persisted, no file is written. 2025/26 is never loaded.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_h2c_recency_reparameterization.py
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
CONTEXT_COL = "competition_id"
ALIASES = ("home_goal_diff_last5", "home_goal_diff_last10",
           "away_goal_diff_last5", "away_goal_diff_last10")
H1_COLUMNS = tuple(c for c in MODEL_B_COLUMNS if c not in ALIASES)

EXPECTED_TRIPLES = 18
RECON_TOL = 1e-12

H1_RECORDED = {
    "mean_log_loss": 0.9915427871369671,
    "mean_brier": 0.5913870749997768,
    "fold_1": 0.9979365509246185,
    "fold_2": 0.9846529437399034,
    "fold_3": 0.9920388667463793,
}


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)


def stop(msg):
    print("\n" + "!" * 78); print("STOP -- H2-C halted"); print(msg); print("!" * 78)
    raise SystemExit(1)


def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------
# STEP 2 -- derive triples mechanically
# ---------------------------------------------------------------------
def derive_triples(columns):
    suffix = {"last5": "_last5", "last10": "_last10", "season": "_season"}
    stems = {}
    for c in columns:
        for w, s in suffix.items():
            if c.endswith(s):
                stems.setdefault(c[: -len(s)], {})[w] = c
    complete = {k: v for k, v in stems.items() if set(v) == {"last5", "last10", "season"}}
    partial = {k: v for k, v in stems.items() if set(v) != {"last5", "last10", "season"}}
    return complete, partial


# ---------------------------------------------------------------------
# Experimental pipeline: impute (train-only) -> [transform] -> scale -> onehot
# Mirrors train.LogisticRegressionPreprocessor exactly when transform=False.
# ---------------------------------------------------------------------
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

    def _reparameterize(self, imputed: pd.DataFrame) -> pd.DataFrame:
        if not self.apply_transform:
            return imputed
        out = imputed.copy()
        for stem, t in self.triples.items():
            s, l10, l5 = imputed[t["season"]], imputed[t["last10"]], imputed[t["last5"]]
            out[t["last10"]] = l10 - s          # u2 : medium-term deviation
            out[t["last5"]] = l5 - l10          # u3 : recent acceleration
        renames = {}
        for stem, t in self.triples.items():
            renames[t["last10"]] = f"{stem}_last10_minus_season"
            renames[t["last5"]] = f"{stem}_last5_minus_last10"
        return out.rename(columns=renames)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        imputed = self._impute_frame(X)
        reparam = self._reparameterize(imputed)
        if list(reparam.columns) != self.output_columns_:
            raise RuntimeError("column order drift between fit and transform")
        scaled = self._scaler.transform(reparam.to_numpy())
        onehot = np.zeros((len(X), len(self.competition_categories)))
        index = {c: i for i, c in enumerate(self.competition_categories)}
        for row, val in enumerate(X[CONTEXT_COL].tolist()):
            col = index.get(val)
            if col is not None:
                onehot[row, col] = 1.0
        return np.hstack([scaled, onehot])

    def encoded_names(self):
        return list(self.output_columns_) + [
            f"{CONTEXT_COL}=={c}" for c in self.competition_categories]

    # diagnostics
    def imputed_only(self, X):
        return self._impute_frame(X)

    def reparam_only(self, X):
        return self._reparameterize(self._impute_frame(X))


def fit_arm(X_train, y_train, X_val, pre):
    from sklearn.linear_model import LogisticRegression
    from models import train as train_module

    pre.fit(X_train)
    Etr, Eva = pre.transform(X_train), pre.transform(X_val)
    model = LogisticRegression(C=C_VALUE, **V1_LOGREG_BASE_KWARGS)
    model.fit(Etr, y_train)
    P = np.asarray(train_module._reorder_proba(model, Eva), dtype=float)
    validate_probabilities(P)
    return P, model, pre


def fit_frozen_arm(X_train, y_train, X_val):
    """A1 control: FROZEN V1 preprocessing + the H1 estimator at C=0.0005.

    `models.train.train_logistic_regression()` is deliberately NOT used.
    Its estimator line (src/models/train.py:149) hardcodes C=1.0, which in
    the previous run silently made A1 a C=1.0 model while A2/A3 were
    C=0.0005 -- producing the ~0.2 probability gap that failed the A2 gate.
    The pipeline-equivalence diagnostic proved the preprocessing on both
    sides is bit-identical, so the estimator construction was the only
    fault. It is corrected here by building the estimator directly from the
    frozen kwargs, exactly as the validated H1 experiment did.

    The FROZEN LogisticRegressionPreprocessor is still used, unmodified, so
    A1 remains the genuine frozen-pipeline control.
    """
    from sklearn.linear_model import LogisticRegression
    from models import train as train_module

    pre = train_module.LogisticRegressionPreprocessor().fit(X_train)
    Etr, Eva = pre.transform(X_train), pre.transform(X_val)
    model = LogisticRegression(C=C_VALUE, **V1_LOGREG_BASE_KWARGS)
    model.fit(Etr, y_train)
    P = np.asarray(train_module._reorder_proba(model, Eva), dtype=float)
    validate_probabilities(P)
    return P, model, pre


def per_class(y, P):
    y = np.asarray(y); pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    out = {}
    for c in CLASS_ORDER:
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        out[c] = dict(precision=pr, recall=rc, f1=f1, support=int((y == c).sum()))
    return out


def draw_stats(y, P):
    pred = np.array(CLASS_ORDER)[P.argmax(axis=1)]
    return dict(count=int((pred == "D").sum()), pct=float((pred == "D").mean()),
                mean_p=float(P[:, 1].mean()), max_p=float(P[:, 1].max()),
                actual=float((np.asarray(y) == "D").mean()))


def main():
    print("H2-C -- RECENCY REPARAMETERIZATION, TRAINING-ONLY IMPUTATION FIRST")
    print("Validation folds only. One variable: the recency basis.")

    # ---------------- STEP 1 ----------------
    rule("STEP 1 -- PRE-FLIGHT")
    import sklearn
    print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"  MODEL_VERSION == 'v1.0'          : {MODEL_VERSION == 'v1.0'}  ({MODEL_VERSION!r})")
    print(f"  C == 0.0005                      : {C_VALUE == 0.0005}")
    print(f"  max_iter == 2000                 : {V1_LOGREG_BASE_KWARGS['max_iter'] == 2000}")
    print(f"  random_state == 0                : {V1_LOGREG_BASE_KWARGS['random_state'] == 0}")
    seasons = {s for f in WALK_FORWARD_FOLDS for s in (f.train_seasons + f.validation_seasons)}
    reachable = bool(seasons & set(FINAL_TEST_SEASONS))
    print(f"  2025/26 unreachable              : {not reachable}  (folds: {sorted(seasons)})")
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines unchanged    : {not bad}  (mismatches={bad or 'NONE'})")
    if MODEL_VERSION != "v1.0" or reachable or bad or C_VALUE != 0.0005:
        stop("pre-flight integrity failure")

    # ---------------- STEP 2 ----------------
    rule("STEP 2 -- DERIVE TRIPLES FROM THE H1 CONTRACT")
    triples, partial = derive_triples(H1_COLUMNS)
    print(f"  H1 contract columns: {len(H1_COLUMNS)}")
    print(f"  complete triples derived: {len(triples)}")
    for stem in sorted(triples):
        t = triples[stem]
        print(f"    {stem:<34} season={t['season']:<42} last10={t['last10']:<42} last5={t['last5']}")
    print(f"\n  incomplete stems EXCLUDED ({len(partial)}): "
          f"{ {k: sorted(v) for k, v in sorted(partial.items())} }")
    covered = {c for t in triples.values() for c in t.values()}
    print(f"  columns covered by triples: {len(covered)} | untouched: {len(H1_COLUMNS) - len(covered)}")
    if len(triples) != EXPECTED_TRIPLES:
        stop(f"derived {len(triples)} triples, expected {EXPECTED_TRIPLES}. Explain before proceeding.")
    print(f"  [PASS] derived count == {EXPECTED_TRIPLES}")

    # ---------------- STEP 3 ----------------
    rule("STEP 3 -- TRANSFORMATION PROOF")
    T = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1]], dtype=float)
    print(f"  T =\n{T}")
    print(f"  det(T) = {np.linalg.det(T):+.10f}   rank = {np.linalg.matrix_rank(T)}")
    print(f"  T^-1 =\n{np.linalg.inv(T)}")
    print("  inverse: season = u1 ; last10 = u1 + u2 ; last5 = u1 + u2 + u3")
    if abs(np.linalg.det(T) - 1.0) > 1e-12 or np.linalg.matrix_rank(T) != 3:
        stop("transformation matrix is not unit-determinant / full rank")
    print("  [PASS] invertible, determinant exactly 1")

    dataset = load_supervised_dataset(FEATURES_DB)
    print(f"\n  dataset rows: {len(dataset.X)}")

    results = {"A1": {}, "A2": {}, "A3": {}}
    coefs = {}
    folds_data = {}

    # ---------------- STEPS 4-6 ----------------
    rule("STEPS 4-6 -- PER-FOLD: IMPUTE FIRST, TRANSFORM SECOND, THEN FIT")
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        name = fold.name
        Xtr = train_ds.X[list(H1_COLUMNS)]
        Xva = val_ds.X[list(H1_COLUMNS)]
        print(f"\n  --- {name} ---")
        print(f"    STEP 5 isolation: n_train={len(Xtr)}  n_val={len(Xva)}  "
              f"train seasons={fold.train_seasons}  val seasons={fold.validation_seasons}")

        pre_id = ExperimentalPreprocessor(triples, apply_transform=False)
        pre_h2 = ExperimentalPreprocessor(triples, apply_transform=True)

        # STEP 4 -- NaN accounting, on training and validation separately
        nan_before_tr = int(Xtr[sorted(covered)].isna().sum().sum())
        nan_before_va = int(Xva[sorted(covered)].isna().sum().sum())
        pre_h2.fit(Xtr)
        imp_tr = pre_h2.imputed_only(Xtr); imp_va = pre_h2.imputed_only(Xva)
        rep_tr = pre_h2.reparam_only(Xtr); rep_va = pre_h2.reparam_only(Xva)
        nan_after_imp = int(imp_tr[sorted(covered)].isna().sum().sum()
                            + imp_va[sorted(covered)].isna().sum().sum())
        nan_after_rep = int(rep_tr.isna().sum().sum() + rep_va.isna().sum().sum())
        print(f"    STEP 4 NaNs in triple columns  before impute: train={nan_before_tr} "
              f"val={nan_before_va}")
        print(f"    STEP 4 NaNs after imputation   : {nan_after_imp}  (must be 0)")
        print(f"    STEP 4 NaNs after transform    : {nan_after_rep}  (must be 0 -- the point of H2-C)")
        if nan_after_imp != 0 or nan_after_rep != 0:
            stop(f"{name}: imputation/transform left NaNs "
                 f"(after_impute={nan_after_imp}, after_transform={nan_after_rep})")

        # STEP 4 -- reconstruction against the IMPUTED originals
        worst = 0.0
        for stem, t in triples.items():
            s = imp_va[t["season"]].to_numpy()
            u2 = rep_va[f"{stem}_last10_minus_season"].to_numpy()
            u3 = rep_va[f"{stem}_last5_minus_last10"].to_numpy()
            worst = max(worst,
                        float(np.abs((s + u2) - imp_va[t["last10"]].to_numpy()).max()),
                        float(np.abs((s + u2 + u3) - imp_va[t["last5"]].to_numpy()).max()))
        print(f"    STEP 4 max reconstruction error vs imputed originals: {worst:.3e} "
              f"(tol {RECON_TOL:g})")
        if worst > RECON_TOL:
            stop(f"{name}: reconstruction error {worst:.3e} exceeds {RECON_TOL:g}")

        med = pre_h2.imputer_medians_
        shown = [t["season"] for t in list(triples.values())[:3]]
        print(f"    STEP 5 training-only medians (first 3 season cols): "
              f"{ {k: round(float(med[k]), 6) for k in shown} }")
        assert "label_result" not in Xtr.columns and "label_result" not in Xva.columns, \
            "label column leaked into the design matrix"

        # PASS 1 -- control arms ONLY. A3 is not fitted until the A1==A2
        # gate has passed, per the corrected protocol.
        P1, m1, _ = fit_frozen_arm(Xtr, train_ds.y, Xva)             # A1 frozen preproc, C=0.0005
        P2, m2, _ = fit_arm(Xtr, train_ds.y, Xva, pre_id)            # A2 experimental identity
        folds_data[name] = dict(Xtr=Xtr, Xva=Xva, y_train=train_ds.y, y_series=val_ds.y)

        y = val_ds.y.to_numpy()
        for key, P, model in (("A1", P1, m1), ("A2", P2, m2)):
            r = evaluate(val_ds.y, P)
            results[key][name] = dict(P=P, y=y, r=r, per_class=per_class(y, P),
                                      draw=draw_stats(y, P),
                                      pred=np.array(CLASS_ORDER)[P.argmax(axis=1)],
                                      n_iter=model.n_iter_, model=model)
        print(f"    A1 frozen  C={m1.C}  log_loss={results['A1'][name]['r'].log_loss:.16f}")
        print(f"    A2 exp-id  C={m2.C}  log_loss={results['A2'][name]['r'].log_loss:.16f}")
        if m1.C != C_VALUE or m2.C != C_VALUE:
            stop(f"{name}: an arm was fitted at C={m1.C}/{m2.C}, expected {C_VALUE}")
        print(f"    A3 H2-C    NOT YET FITTED -- gated on A1 fidelity and A1==A2")

    # ---------------- GATE 1 (STEP 7) ----------------
    rule("GATE 1 (STEP 7) -- A1 FIDELITY vs RECORDED H1")
    m_ll = statistics.fmean(results["A1"][n]["r"].log_loss for n in results["A1"])
    m_br = statistics.fmean(results["A1"][n]["r"].brier for n in results["A1"])
    ok = True
    for n in ("fold_1", "fold_2", "fold_3"):
        obs = results["A1"][n]["r"].log_loss
        d = abs(obs - H1_RECORDED[n]); ok &= d < 1e-9
        print(f"  {n}: recorded={H1_RECORDED[n]:.16f} observed={obs:.16f} diff={d:.3e}")
    print(f"  mean log loss: recorded={H1_RECORDED['mean_log_loss']:.16f} observed={m_ll:.16f} "
          f"diff={abs(m_ll - H1_RECORDED['mean_log_loss']):.3e}")
    print(f"  mean Brier   : recorded={H1_RECORDED['mean_brier']:.16f} observed={m_br:.16f} "
          f"diff={abs(m_br - H1_RECORDED['mean_brier']):.3e}")
    ok &= abs(m_ll - H1_RECORDED["mean_log_loss"]) < 1e-9
    ok &= abs(m_br - H1_RECORDED["mean_brier"]) < 1e-9
    if not ok:
        stop("A1 did not reproduce the recorded H1 results to <1e-9. H2-C not fitted, not reported.")
    print("  [PASS] GATE 1 -- A1 reproduces recorded H1")

    # ---------------- GATE 2 ----------------
    rule("GATE 2 -- A2 (experimental, identity) MUST REPRODUCE A1 EXACTLY")
    gate2 = True
    for n in sorted(results["A1"]):
        a, b = results["A1"][n], results["A2"][n]
        p_same = a["P"].tobytes() == b["P"].tobytes()
        c_same = a["model"].coef_.tobytes() == b["model"].coef_.tobytes()
        i_same = (np.asarray(a["model"].intercept_).tobytes()
                  == np.asarray(b["model"].intercept_).tobytes())
        gate2 &= p_same and c_same and i_same
        print(f"  {n}: probabilities byte-identical={p_same} "
              f"max|delta|={float(np.abs(b['P'] - a['P']).max()):.3e}")
        print(f"        coefficients byte-identical={c_same} "
              f"max|delta|={float(np.abs(b['model'].coef_ - a['model'].coef_).max()):.3e}")
        print(f"        intercepts   byte-identical={i_same} "
              f"max|delta|={float(np.abs(np.asarray(b['model'].intercept_) - np.asarray(a['model'].intercept_)).max()):.3e}")
    if not gate2:
        stop("A2 does not reproduce A1 bit-for-bit. The experimental pipeline is not proven "
             "faithful, so any A3 result would be confounded. A3 was NOT fitted.")
    print("  [PASS] GATE 2 -- experimental pipeline reproduces the frozen pipeline exactly;")
    print("         any A3 difference is therefore attributable to the transform alone")

    # ---------------- PASS 2 -- A3 fitted ONLY after both gates pass ----------
    rule("PASS 2 -- FIT A3 (H2-C) -- reached only because GATE 1 and GATE 2 passed")
    for name, fd in folds_data.items():
        pre_h2 = ExperimentalPreprocessor(triples, apply_transform=True)
        P3, m3, p3 = fit_arm(fd["Xtr"], fd["y_train"], fd["Xva"], pre_h2)
        if m3.C != C_VALUE:
            stop(f"{name}: A3 fitted at C={m3.C}, expected {C_VALUE}")
        y = fd["y_series"].to_numpy()
        r = evaluate(fd["y_series"], P3)
        results["A3"][name] = dict(P=P3, y=y, r=r, per_class=per_class(y, P3),
                                   draw=draw_stats(y, P3),
                                   pred=np.array(CLASS_ORDER)[P3.argmax(axis=1)],
                                   n_iter=m3.n_iter_, model=m3)
        coefs[name] = dict(h2_model=m3, h2_names=p3.encoded_names(),
                           h1_model=results["A2"][name]["model"])
        print(f"  {name}: A3 H2-C  C={m3.C}  log_loss={r.log_loss:.16f}")

    # ---------------- STEPS 8-9 ----------------
    rule("STEPS 8-9 -- PER-FOLD METRICS")
    for n in results["A1"]:
        a, b = results["A1"][n], results["A3"][n]
        print(f"\n  {n} (n={a['r'].n}):")
        for m in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy"):
            x, y_ = a["r"].as_dict()[m], b["r"].as_dict()[m]
            print(f"    {m:<20} H1={x:.16f}  H2C={y_:.16f}  delta={y_ - x:+.6e}")
        ca = int((a["pred"] == a["y"]).sum()); cb = int((b["pred"] == b["y"]).sum())
        print(f"    correct              H1={ca}  H2C={cb}  delta={cb - ca:+d}")
        for c in CLASS_ORDER:
            pa, pb = a["per_class"][c], b["per_class"][c]
            print(f"      {c}: precision {pa['precision']:.6f}->{pb['precision']:.6f} "
                  f"recall {pa['recall']:.6f}->{pb['recall']:.6f} "
                  f"f1 {pa['f1']:.6f}->{pb['f1']:.6f} support={pa['support']}")

    means = {}
    for key in ("A1", "A3"):
        means[key] = {m: statistics.fmean(results[key][n]["r"].as_dict()[m] for n in results[key])
                      for m in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy")}

    rule("FINAL TABLE")
    tot_c1 = sum(int((results["A1"][n]["pred"] == results["A1"][n]["y"]).sum()) for n in results["A1"])
    tot_c3 = sum(int((results["A3"][n]["pred"] == results["A3"][n]["y"]).sum()) for n in results["A3"])
    tot_n = sum(results["A1"][n]["r"].n for n in results["A1"])
    print(f"  {'Metric':<20} {'H1 CONTROL':>22} {'H2-C':>22} {'Delta':>16}")
    print("  " + "-" * 82)
    print(f"  {'Correct':<20} {tot_c1:>22} {tot_c3:>22} {tot_c3 - tot_c1:>+16d}")
    print(f"  {'Wrong':<20} {tot_n - tot_c1:>22} {tot_n - tot_c3:>22} "
          f"{(tot_n - tot_c3) - (tot_n - tot_c1):>+16d}")
    for m in ("accuracy", "log_loss", "brier", "macro_f1", "balanced_accuracy"):
        print(f"  {m:<20} {means['A1'][m]:>22.16f} {means['A3'][m]:>22.16f} "
              f"{means['A3'][m] - means['A1'][m]:>+16.6e}")

    rule("FOLD-BY-FOLD DELTAS")
    improved = {}
    for n in results["A1"]:
        dll = results["A3"][n]["r"].log_loss - results["A1"][n]["r"].log_loss
        dbr = results["A3"][n]["r"].brier - results["A1"][n]["r"].brier
        improved[n] = dll < 0
        print(f"  {n}: log-loss delta={dll:+.6e} ({'improved' if dll < 0 else 'worse'})  "
              f"Brier delta={dbr:+.6e}")
    print(f"  improves log loss in: {[n for n, v in improved.items() if v]} "
          f"({sum(improved.values())} of {len(improved)} folds)")

    # ---------------- STEP 10 ----------------
    rule("STEP 10 -- ACCURACY MOVEMENT AND TRANSITIONS")
    tot = {}
    for n in results["A1"]:
        a, b = results["A1"][n], results["A3"][n]
        nc = int(((a["pred"] != a["y"]) & (b["pred"] == b["y"])).sum())
        nw = int(((a["pred"] == a["y"]) & (b["pred"] != b["y"])).sum())
        print(f"  {n}: newly correct={nc}  newly wrong={nw}  net={nc - nw:+d}")
        for x in CLASS_ORDER:
            for y_ in CLASS_ORDER:
                k = f"{x}->{y_}"
                c = int(((a["pred"] == x) & (b["pred"] == y_)).sum())
                tot[k] = tot.get(k, 0) + c
    print("\n  pooled transition matrix (H1 predicted -> H2-C predicted):")
    for x in CLASS_ORDER:
        print("    " + "  ".join(f"{x}->{y_}: {tot.get(f'{x}->{y_}', 0):5d}" for y_ in CLASS_ORDER))
    NC = sum(int(((results['A1'][n]['pred'] != results['A1'][n]['y'])
                  & (results['A3'][n]['pred'] == results['A3'][n]['y'])).sum()) for n in results['A1'])
    NW = sum(int(((results['A1'][n]['pred'] == results['A1'][n]['y'])
                  & (results['A3'][n]['pred'] != results['A3'][n]['y'])).sum()) for n in results['A1'])
    print(f"  POOLED newly correct={NC}  newly wrong={NW}  net={NC - NW:+d}")

    # ---------------- STEP 11 ----------------
    rule("STEP 11 -- DRAW SAFETY CHECK")
    for n in results["A1"]:
        a, b = results["A1"][n]["draw"], results["A3"][n]["draw"]
        print(f"  {n}: actual draw rate={a['actual']:.6f}")
        print(f"    predicted D count  H1={a['count']:4d} ({a['pct']:.6f})  "
              f"H2C={b['count']:4d} ({b['pct']:.6f})  delta={b['count'] - a['count']:+d}")
        print(f"    mean p_draw        H1={a['mean_p']:.6f}  H2C={b['mean_p']:.6f}  "
              f"delta={b['mean_p'] - a['mean_p']:+.6e}")
        print(f"    max  p_draw        H1={a['max_p']:.6f}  H2C={b['max_p']:.6f}")
    dd = statistics.fmean(results["A3"][n]["draw"]["mean_p"] - results["A1"][n]["draw"]["mean_p"]
                          for n in results["A1"])
    dcount = sum(results["A3"][n]["draw"]["count"] - results["A1"][n]["draw"]["count"]
                 for n in results["A1"])
    if means["A3"]["log_loss"] < means["A1"]["log_loss"] and (dcount < 0 or dd < -0.005):
        print("\n  *** FLAG: log loss improves while Draw behaviour degrades "
              f"(predicted-D delta={dcount:+d}, mean p_draw delta={dd:+.3e}) ***")

    # ---------------- STEP 12 ----------------
    rule("STEP 12 -- PROBABILITY MOVEMENT")
    allmax = []
    for n in results["A1"]:
        A, B = results["A1"][n]["P"], results["A3"][n]["P"]
        d = np.abs(B - A); pm = d.max(axis=1); allmax.append(pm)
        o = np.sort(pm)
        print(f"  {n}: identical rows={int((pm == 0).sum())}  changed={int((pm > 0).sum())}  "
              f"mean|delta|={d.mean():.6e}  median max={np.median(pm):.6e}  "
              f"p95={o[min(len(o) - 1, int(round(0.95 * (len(o) - 1))))]:.6e}  max={pm.max():.6e}")
        for i, c in enumerate(CLASS_ORDER):
            print(f"    mean signed movement {c}: {float((B[:, i] - A[:, i]).mean()):+.6e}")

    # ---------------- STEP 13 ----------------
    rule("STEP 13 -- COEFFICIENT SANITY")
    print("  NOTE: H1 and H2-C use DIFFERENT bases, so raw magnitudes are NOT directly")
    print("  comparable. Checked instead: finiteness, explosion, sign stability, convergence.")
    for n, cc in coefs.items():
        m2, m3, names = cc["h1_model"], cc["h2_model"], cc["h2_names"]
        c3 = pd.DataFrame(m3.coef_, index=list(m3.classes_), columns=names).reindex(CLASS_ORDER)
        print(f"\n  {n}: n_iter_ H1={m2.n_iter_}  H2C={m3.n_iter_}  "
              f"(max_iter={V1_LOGREG_BASE_KWARGS['max_iter']})")
        if int(np.max(m2.n_iter_)) >= V1_LOGREG_BASE_KWARGS["max_iter"] or \
           int(np.max(m3.n_iter_)) >= V1_LOGREG_BASE_KWARGS["max_iter"]:
            print("    *** FLAG: hit max_iter -- result UNCONVERGED ***")
        print(f"    finite coefficients: H1={bool(np.isfinite(m2.coef_).all())} "
              f"H2C={bool(np.isfinite(m3.coef_).all())}")
        print(f"    max|coef| H1={np.abs(m2.coef_).max():.6e}  H2C={np.abs(m3.coef_).max():.6e}")
        print("    H2-C triple coefficients (first 4 triples shown):")
        for stem in sorted(triples)[:4]:
            t = triples[stem]
            for cls in CLASS_ORDER:
                print(f"      {cls} {stem:<30} season={c3.loc[cls, t['season']]:+.6e} "
                      f"l10-season={c3.loc[cls, f'{stem}_last10_minus_season']:+.6e} "
                      f"l5-l10={c3.loc[cls, f'{stem}_last5_minus_last10']:+.6e}")

    # ---------------- STEP 14 ----------------
    rule("STEP 14 -- DETERMINISM (full repeat)")
    det = {"A1": True, "A3": True}
    for fold, train_ds, val_ds in iter_walk_forward_folds(dataset):
        n = fold.name
        Xtr, Xva = train_ds.X[list(H1_COLUMNS)], val_ds.X[list(H1_COLUMNS)]
        P1b, _, _ = fit_frozen_arm(Xtr, train_ds.y, Xva)
        P3b, _, _ = fit_arm(Xtr, train_ds.y, Xva,
                            ExperimentalPreprocessor(triples, apply_transform=True))
        det["A1"] &= P1b.tobytes() == results["A1"][n]["P"].tobytes()
        det["A3"] &= P3b.tobytes() == results["A3"][n]["P"].tobytes()
    for key, label in (("A1", "H1 control"), ("A3", "H2-C")):
        digest = hashlib.sha256(b"".join(results[key][n]["P"].tobytes()
                                         for n in sorted(results[key]))).hexdigest()
        print(f"  {label:<12} byte-identical repeat={det[key]}  sha256={digest}")
    if not all(det.values()):
        stop(f"determinism failed: {det}")
    print("  [PASS] deterministic")

    # ---------------- STEP 15 ----------------
    rule("STEP 15 -- VERDICT")
    d_ll = means["A3"]["log_loss"] - means["A1"]["log_loss"]
    d_br = means["A3"]["brier"] - means["A1"]["brier"]
    nfolds = sum(improved.values())
    print(f"  mean validation log-loss delta (H2-C - H1) : {d_ll:+.6e}")
    print(f"  mean Brier delta                            : {d_br:+.6e}")
    print(f"  folds improved                              : {nfolds} of {len(improved)}")
    print(f"  pooled net correct                          : {NC - NW:+d}")
    if d_ll > 0:
        verdict = "H2-C REJECTED"
        why = "mean validation log loss worsens"
    elif d_ll < 0 and nfolds == len(improved) and abs(d_ll) > 1e-4:
        verdict = "H2-C IMPROVES VALIDATION"
        why = "mean improves, all folds agree, magnitude above the numerical-noise scale"
    else:
        verdict = "H2-C MIXED / INCONCLUSIVE"
        why = ("mean improves but the evidence is not uniform across folds, "
               "or the gain is too small to distinguish from numerical effects "
               "(reference: H1's own gain was 2.9e-05 and the solver tol is 1e-4)")
    print(f"\n  VERDICT: {verdict}")
    print(f"  reason : {why}")

    rule("INTEGRITY AUDIT")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    if drift:
        stop(f"experiment altered pinned files: {drift}")
    print(f"  MODEL_VERSION: {MODEL_VERSION!r}")
    ser = [str(p.relative_to(REPO)) for pat in ("*.pkl", "*.joblib", "*.pickle")
           for p in REPO.rglob(pat) if "__pycache__" not in p.parts]
    print(f"  serialized estimators: {ser or 'NONE'}")
    for line in ("2025/26 accessed: NO", "C tuned: NO (fixed 0.0005)", "calibration: NONE",
                 "class weighting: NONE", "feature selection: NONE", "H3/H4 tested: NO",
                 "production source modified: NO", "features.db regenerated: NO",
                 "estimator persisted: NONE", "file created by this script: NONE"):
        print(f"  {line}")

    print("\n" + "=" * 78)
    print("  H2-C NOT FROZEN")
    print("  H2-C NOT PROMOTED")
    print("  2025/26 NOT TOUCHED")
    print("=" * 78)


if __name__ == "__main__":
    main()
