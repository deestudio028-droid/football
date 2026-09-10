"""STEP 7 -- closeness representation experiment (V1 vs V1 + closeness block).

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step7_closeness_representation_experiment.py

ONE QUESTION ONLY
Does adding an explicit non-monotonic home-vs-away CLOSENESS representation,
derived solely from existing production features, recover Draw-relevant
structure that the current linear representation cannot express?

WHY THIS TESTS THE STEP-6 HYPOTHESIS
A linear score f = a*x_home + b*x_away CAN represent the signed difference
(a=1, b=-1). It CANNOT represent |x_home - x_away|. Draw is structurally the
interior/balance outcome, and no linear combination of monotone features has
an interior maximum. The closeness block supplies exactly the component the
current form cannot construct for itself -- and NOTHING ELSE. No new
information source is introduced, so a null result is informative about
representation rather than about data availability.

PRE-REGISTERED, FIXED BEFORE ANY RESULT WAS SEEN
  construction   : closeness_j = abs(home_j - away_j)      (ONE construction)
  pairing        : strict suffix match home_<suf> <-> away_<suf>
  exclusions     : within-team differences (any '*_diff_*' / '*_goal_diff_*'),
                   league one-hot, competition_id, strength_diff,
                   and any feature without an exact counterpart
  estimator      : unchanged production LogisticRegression(C=1.0,
                   max_iter=2000, random_state=0)
  preprocessing  : unchanged production LogisticRegressionPreprocessor
  folds          : unchanged three walk-forward folds
  classification : A REPRESENTATION SUPPORTED / B NOT SUPPORTED / C AMBIGUOUS

NOT DONE HERE
No tuning of C, solver, thresholds, class weights or random_state. No feature
removal. No second construction. No comparison of abs versus squared. No new
raw column, no xG, possession, formation, referee, market, rest, or table
position. No 2025/26. No persistence of any model, matrix or prediction. No
production file is modified.

TERMINOLOGY: raw separation, coefficient direction, contribution, cancellation,
stability, probability geometry, validation performance. Never "importance".
No causal claims.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

POOLED_REF_LOG_LOSS = 0.9993791056968738
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_MD5 = "5e504427712b35778bb8a62a8496c7cd"
FORBIDDEN_SEASONS = [602681, 667780, 725788, 725793, 762170]

PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)

#: Pre-registered exclusion tokens for the pairing map.
EXCLUDE_TOKENS = ("_diff_", "diff_last", "diff_season", "_venue_")
CLOSENESS_PREFIX = "closeness_"


def rule(t): print("\n" + "=" * 122); print(t); print("=" * 122)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 122)
    print("HARD FAIL -- STEP 7 halted. No experiment result is produced.")
    print(msg)
    print("!" * 122)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        stop(f"stop condition: {label}")


def snapshot():
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    lg = REPO / "data/processed/leagues"
    if lg.exists():
        for p in sorted(lg.glob("*.db")):
            snap[f"leagues/{p.name}"] = md5(p)
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["artifact"] = md5(art)
    snap["src/models/train.py"] = md5(REPO / "src/models/train.py")
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def smd(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def cancel(terms):
    net = np.abs(terms.sum(axis=1))
    tot = np.abs(terms).sum(axis=1)
    per = np.divide(net, tot, out=np.full_like(net, np.nan), where=tot > 0)
    return float(np.nanmean(per)), (float(net.sum() / tot.sum()) if tot.sum() else float("nan"))


def draw_geometry(P, Y, order):
    iH, iD, iA = 0, 1, 2
    pred = np.array(order)[P.argmax(axis=1)]
    d = Y == "D"
    Pd = P[d]
    return {
        "draw_argmax_share": float((pred == "D").mean()),
        "draw_rank1_share": float(((Pd[:, iD] > Pd[:, iH]) & (Pd[:, iD] > Pd[:, iA])).mean()),
        "pd_mean": float(P[:, iD].mean()),
        "pd_sd": float(P[:, iD].std()),
        "pd_gt_third": float((P[:, iD] > 1 / 3).mean()),
        "d_gt_h": float((Pd[:, iD] > Pd[:, iH]).mean()),
        "d_gt_a": float((Pd[:, iD] > Pd[:, iA]).mean()),
        "d_gt_both": float(((Pd[:, iD] > Pd[:, iH]) & (Pd[:, iD] > Pd[:, iA])).mean()),
    }


def main():
    rule("RULE 1 -- PRODUCTION CODE TRACE (source text, never memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:70]}", tok in tsrc)

    rule("STEP 1 -- ENVIRONMENT + INTEGRITY")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")
    try:
        import sklearn
        print(f"  sklearn {sklearn.__version__}")
    except ImportError as exc:
        stop(f"scikit-learn is required: {exc}. Do not substitute another estimator.")

    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        FINAL_TEST_SEASONS, MODEL_VERSION, RECOMMENDED_CONTEXT_FEATURE,
        REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS,
    )
    from models.data import load_supervised_dataset
    from models.evaluate import evaluate
    from models.splits import iter_walk_forward_folds
    from models.train import train_logistic_regression

    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("contract is exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric production features == 79", len(numeric) == 79)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")

    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(rel, before[rel] == exp, before[rel])
    if "artifact" in before:
        check("v1_logreg.pkl unchanged", before["artifact"] == ARTIFACT_MD5, before["artifact"])
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    check("13/13 LOCKED_INPUTS at expected values",
          all(before[f"pin:{r}"] == v["expected"] for r, v in pins.items()))

    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    check("forbidden season set matches the documented list",
          sorted(test_ids) == sorted(FORBIDDEN_SEASONS), str(sorted(test_ids)))
    folds = list(iter_walk_forward_folds(ds))
    check("exactly three walk-forward folds", len(folds) == 3)
    for fold, tr, va in folds:
        seen = set(tr.metadata["season_id"]) | set(va.metadata["season_id"])
        check(f"{fold.name}: no final-test season_id", not (seen & test_ids))

    rule("STEP 3 -- EXPLICIT HOME/AWAY PAIRING AUDIT (built from the contract, printed in full)")
    pairs, excluded, unpaired = [], [], []
    for h in [c for c in numeric if c.startswith("home_")]:
        suf = h[len("home_"):]
        a = "away_" + suf
        if a not in numeric:
            unpaired.append((h, "no exact away_ counterpart"))
            continue
        reasons = [t for t in EXCLUDE_TOKENS if t in "_" + suf + "_" or t in suf]
        if reasons:
            excluded.append((h, a, ",".join(sorted(set(reasons)))))
        else:
            pairs.append((h, a, suf))
    nonsided = [c for c in numeric if not c.startswith(("home_", "away_"))]
    print(f"  home_* features: {sum(1 for c in numeric if c.startswith('home_'))}   "
          f"away_* features: {sum(1 for c in numeric if c.startswith('away_'))}   "
          f"non-sided: {len(nonsided)} {nonsided}")
    print(f"\n  INCLUDED PAIRS ({len(pairs)}):")
    for h, a, s in pairs:
        print(f"    {h:44s} <-> {a}")
    print(f"\n  EXCLUDED PAIRS ({len(excluded)}) -- within-team differences, not home-vs-away:")
    for h, a, r in excluded:
        print(f"    {h:44s} <-> {a:44s} [{r}]")
    print(f"\n  UNPAIRED home_* features ({len(unpaired)}):")
    for h, r in unpaired:
        print(f"    {h}  [{r}]")
    print(f"\n  NON-SIDED features excluded by rule: {nonsided}")
    print("    'strength_diff' is already a relational term and is NOT re-expressed as")
    print("    closeness -- the pre-registered rule forbids duplicating it.")
    check("at least one closeness pair was found", len(pairs) > 0)
    check("no closeness pair contains a within-team difference",
          not any(t in s for h, a, s in pairs for t in EXCLUDE_TOKENS))
    check("no league / competition_id column entered the pairing",
          RECOMMENDED_CONTEXT_FEATURE not in [h for h, _, _ in pairs] + [a for _, a, _ in pairs])

    rule("STEP 4 -- CLOSENESS CONSTRUCTION (ONE construction, pre-registered)")
    print("  closeness_<suffix> = abs(home_<suffix> - away_<suffix>)")
    print("  Computed ROW-WISE from existing production feature values only.")
    print("  No future information, no post-match field, no refit of the feature")
    print("  generator, and no validation row influences any other row.")
    close_names = [CLOSENESS_PREFIX + s for _, _, s in pairs]

    def add_closeness(X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for h, a, s in pairs:
            out[CLOSENESS_PREFIX + s] = (pd.to_numeric(out[h], errors="coerce")
                                         - pd.to_numeric(out[a], errors="coerce")).abs()
        return out

    print(f"  closeness columns created: {len(close_names)}")

    rule("STEP 2 -- V1 BASELINE REPRODUCTION  +  STEP 6 WALK-FORWARD V1 vs V1+CLOSENESS")
    iH, iD, iA = 0, 1, 2
    v1_folds, sc_folds = [], []
    v1_P, sc_P, Ys, Fs = [], [], [], []
    sc_fits = []
    for fold, tr, va in folds:
        Xtr_v1, Xva_v1 = tr.X[list(MODEL_B_COLUMNS)], va.X[list(MODEL_B_COLUMNS)]
        _, _, P1 = train_logistic_regression(Xtr_v1, tr.y, Xva_v1)
        r1 = evaluate(va.y, P1)
        v1_folds.append((fold.name, len(va), r1))

        Xtr_sc, Xva_sc = add_closeness(Xtr_v1), add_closeness(Xva_v1)
        check(f"{fold.name}: successor frame = 80 + {len(close_names)} columns",
              Xtr_sc.shape[1] == len(MODEL_B_COLUMNS) + len(close_names),
              str(Xtr_sc.shape))
        check(f"{fold.name}: original 80 columns unmodified",
              Xtr_sc[list(MODEL_B_COLUMNS)].equals(Xtr_v1))
        m2, pre2, P2 = train_logistic_regression(Xtr_sc, tr.y, Xva_sc)
        r2 = evaluate(va.y, P2)
        sc_folds.append((fold.name, len(va), r2))
        cls = list(m2.classes_)
        order = [cls.index(c) for c in CLASS_ORDER]
        sc_fits.append(dict(name=fold.name, model=m2, pre=pre2, order=order,
                            W=m2.coef_[order], b=m2.intercept_[order],
                            Tva=pre2.transform(Xva_sc), Xva=Xva_sc.reset_index(drop=True)))
        v1_P.append(P1); sc_P.append(P2); Ys.append(va.y.to_numpy())
        Fs.append(np.repeat(fold.name, len(va)))
        print(f"  {fold.name}: N={len(va)}  V1 log_loss={r1.log_loss:.16f} brier={r1.brier:.16f}"
              f"   V1+CLOSE log_loss={r2.log_loss:.16f} brier={r2.brier:.16f}")

    P1 = np.vstack(v1_P); P2 = np.vstack(sc_P)
    Y = np.concatenate(Ys); FOLD = np.concatenate(Fs)
    ll1 = float(np.mean([r.log_loss for _, _, r in v1_folds]))
    ll2 = float(np.mean([r.log_loss for _, _, r in sc_folds]))
    br1 = float(np.mean([r.brier for _, _, r in v1_folds]))
    br2 = float(np.mean([r.brier for _, _, r in sc_folds]))
    print(f"\n  V1 pooled mean log loss : {ll1:.16f}   (reference {POOLED_REF_LOG_LOSS:.16f})")
    check("V1 baseline reproduces the documented pooled log loss",
          abs(ll1 - POOLED_REF_LOG_LOSS) <= 1e-9, f"diff={ll1 - POOLED_REF_LOG_LOSS:.3e}")
    print(f"  V1+CLOSENESS pooled mean log loss : {ll2:.16f}")
    print(f"  V1 pooled mean Brier              : {br1:.16f}")
    print(f"  V1+CLOSENESS pooled mean Brier    : {br2:.16f}")
    print(f"\n  delta log loss (successor - V1)   : {ll2 - ll1:+.16f}   "
          f"({'improves' if ll2 < ll1 else 'worsens'})")
    print(f"  delta Brier    (successor - V1)   : {br2 - br1:+.16f}   "
          f"({'improves' if br2 < br1 else 'worsens'})")
    imp_ll = sum(1 for (_, _, a), (_, _, b) in zip(v1_folds, sc_folds) if b.log_loss < a.log_loss)
    imp_br = sum(1 for (_, _, a), (_, _, b) in zip(v1_folds, sc_folds) if b.brier < a.brier)
    print(f"  folds improving: log loss {imp_ll}/3   Brier {imp_br}/3")
    print("\n| Fold | N | V1 log loss | SC log loss | delta | V1 Brier | SC Brier | delta |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for (n, k, a), (_, _, b) in zip(v1_folds, sc_folds):
        print(f"| {n} | {k} | {a.log_loss:.8f} | {b.log_loss:.8f} | {b.log_loss-a.log_loss:+.8f} "
              f"| {a.brier:.8f} | {b.brier:.8f} | {b.brier-a.brier:+.8f} |")

    rule("STEP 5 -- SUCCESSOR CONTRACT / MATRIX AUDIT")
    f0 = sc_fits[0]
    n_close = len(close_names)
    n_num_sc = len(f0["pre"].numeric_columns)
    n_cat = len(f0["pre"].competition_categories)
    print(f"  successor numeric columns : {n_num_sc}  (= 79 production + {n_close} closeness)")
    print(f"  league one-hot columns    : {n_cat}")
    print(f"  transformed width         : {f0['Tva'].shape[1]}  (= {n_num_sc} + {n_cat})")
    check("successor numeric block == 79 + closeness", n_num_sc == 79 + n_close)
    check("transformed width == numeric + league", f0["Tva"].shape[1] == n_num_sc + n_cat)
    check("coefficient width matches transformed width", f0["W"].shape[1] == n_num_sc + n_cat)
    close_idx = [list(f0["pre"].numeric_columns).index(c) for c in close_names]
    prod_idx = [list(f0["pre"].numeric_columns).index(c) for c in numeric]
    check("every closeness column is present exactly once in the successor matrix",
          len(set(close_idx)) == n_close)
    check("all 79 production columns still present in the successor matrix",
          len(set(prod_idx)) == 79)
    print(f"  closeness column indices  : {min(close_idx)}..{max(close_idx)}")

    rule("STEP 7 -- RAW CLOSENESS SIGNAL (association only; not importance, not causal)")
    Xall = pd.concat([f["Xva"] for f in sc_fits], ignore_index=True)
    isD, isH, isA = Y == "D", Y == "H", Y == "A"
    r_dh = np.array([smd(pd.to_numeric(Xall[c], errors="coerce").to_numpy()[isD],
                         pd.to_numeric(Xall[c], errors="coerce").to_numpy()[isH])
                     for c in close_names])
    r_da = np.array([smd(pd.to_numeric(Xall[c], errors="coerce").to_numpy()[isD],
                         pd.to_numeric(Xall[c], errors="coerce").to_numpy()[isA])
                     for c in close_names])
    p_dh = np.array([smd(pd.to_numeric(Xall[c], errors="coerce").to_numpy()[isD],
                         pd.to_numeric(Xall[c], errors="coerce").to_numpy()[isH])
                     for c in numeric])
    p_da = np.array([smd(pd.to_numeric(Xall[c], errors="coerce").to_numpy()[isD],
                         pd.to_numeric(Xall[c], errors="coerce").to_numpy()[isA])
                     for c in numeric])
    print(f"  CLOSENESS block  mean |SMD| D-H = {np.nanmean(np.abs(r_dh)):.4f}   "
          f"D-A = {np.nanmean(np.abs(r_da)):.4f}   "
          f"max = {np.nanmax(np.abs(r_dh)):.4f}/{np.nanmax(np.abs(r_da)):.4f}")
    print(f"  PRODUCTION 79    mean |SMD| D-H = {np.nanmean(np.abs(p_dh)):.4f}   "
          f"D-A = {np.nanmean(np.abs(p_da)):.4f}   (D-38 recorded 0.2457 / 0.2281)")
    print(f"  closeness sign coherence D-H: "
          f"{len({np.sign(x) for x in r_dh[~np.isnan(r_dh)]}) == 1}")
    print("\n| closeness feature | SMD D-H | SMD D-A |")
    print("|---|---:|---:|")
    for c, a, b in sorted(zip(close_names, r_dh, r_da), key=lambda t: -abs(t[1])):
        print(f"| {c[:52]} | {a:+.4f} | {b:+.4f} |")

    rule("STEP 8 -- CLOSENESS COEFFICIENT / CONTRIBUTION GEOMETRY")
    Wc_dh = np.mean([f["W"][iD, close_idx] - f["W"][iH, close_idx] for f in sc_fits], axis=0)
    Wc_da = np.mean([f["W"][iD, close_idx] - f["W"][iA, close_idx] for f in sc_fits], axis=0)
    ok = ~np.isnan(r_dh)
    agree_dh = float((np.sign(r_dh[ok]) == np.sign(Wc_dh[ok])).mean())
    corr_dh = (float(np.corrcoef(r_dh[ok], Wc_dh[ok])[0, 1])
               if ok.sum() >= 3 and np.std(r_dh[ok]) > 0 and np.std(Wc_dh[ok]) > 0 else float("nan"))
    ok2 = ~np.isnan(r_da)
    agree_da = float((np.sign(r_da[ok2]) == np.sign(Wc_da[ok2])).mean())
    corr_da = (float(np.corrcoef(r_da[ok2], Wc_da[ok2])[0, 1])
               if ok2.sum() >= 3 and np.std(r_da[ok2]) > 0 and np.std(Wc_da[ok2]) > 0 else float("nan"))
    print(f"  raw-to-fitted alignment, CLOSENESS block:")
    print(f"    D-H sign agreement {100*agree_dh:.1f}%   Pearson r {corr_dh:+.4f}"
          f"   (D-38 production block: 57.0%, +0.0692)")
    print(f"    D-A sign agreement {100*agree_da:.1f}%   Pearson r {corr_da:+.4f}"
          f"   (D-38 production block: 55.7%, +0.0537)")

    close_terms_dh, close_terms_da, prod_terms_dh = [], [], []
    for f in sc_fits:
        m = FOLD == f["name"]
        dwH = f["W"][iD] - f["W"][iH]
        dwA = f["W"][iD] - f["W"][iA]
        T = f["Tva"]
        dsel = (Y[m] == "D")
        close_terms_dh.append((T[:, close_idx] * dwH[close_idx])[dsel])
        close_terms_da.append((T[:, close_idx] * dwA[close_idx])[dsel])
        prod_terms_dh.append((T[:, prod_idx] * dwH[prod_idx])[dsel])
    Cdh = np.vstack(close_terms_dh); Cda = np.vstack(close_terms_da)
    Pdh = np.vstack(prod_terms_dh)
    print(f"\n  on actual draws (positive favours Draw):")
    print(f"    closeness total contribution D-H : {Cdh.sum(axis=1).mean():+.6f}")
    print(f"    closeness total contribution D-A : {Cda.sum(axis=1).mean():+.6f}")
    print(f"    closeness mean |contribution| D-H: {np.abs(Cdh).sum(axis=1).mean():.6f}")
    print(f"    closeness mean |contribution| D-A: {np.abs(Cda).sum(axis=1).mean():.6f}")
    print(f"    production-79 total contribution D-H : {Pdh.sum(axis=1).mean():+.6f}")
    print(f"    production-79 mean |contribution| D-H: {np.abs(Pdh).sum(axis=1).mean():.6f}")

    rule("STEP 9 -- CANCELLATION COMPARISON")
    cp_dh, ca_dh = cancel(Cdh)
    cp_da, ca_da = cancel(Cda)
    pp_dh, pa_dh = cancel(Pdh)
    print("| block | pair | per-row cancel ratio | aggregate ratio |")
    print("|---|---|---:|---:|")
    print(f"| closeness | D-H | {cp_dh:.4f} | {ca_dh:.4f} |")
    print(f"| closeness | D-A | {cp_da:.4f} | {ca_da:.4f} |")
    print(f"| production 79 | D-H | {pp_dh:.4f} | {pa_dh:.4f} |")
    print("\n  D-38 recorded production cancellation 0.0607 (D-H) / 0.0463 (D-A);")
    print("  D-40 recorded 0.0643 / 0.0522 after grouping correlated features.")

    rule("STEP 10 -- DRAW PROBABILITY GEOMETRY")
    g1 = draw_geometry(P1, Y, CLASS_ORDER)
    g2 = draw_geometry(P2, Y, CLASS_ORDER)
    print("| quantity | V1 | V1+CLOSENESS | delta |")
    print("|---|---:|---:|---:|")
    for k, lab in (("draw_argmax_share", "Draw argmax share"),
                   ("draw_rank1_share", "actual-Draw rank-1 share"),
                   ("pd_mean", "mean P(D)"), ("pd_sd", "P(D) standard deviation"),
                   ("pd_gt_third", "P(D) > 1/3 share"),
                   ("d_gt_h", "D>H share"), ("d_gt_a", "D>A share"),
                   ("d_gt_both", "D>both share")):
        print(f"| {lab} | {g1[k]:.4f} | {g2[k]:.4f} | {g2[k]-g1[k]:+.4f} |")

    rule("STEP 11 -- THREE-FOLD STABILITY (ranges only; no standard deviation)")
    print("| quantity | fold_1 | fold_2 | fold_3 | min | max | range | sign stable |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    rows = {}
    for f in sc_fits:
        m = FOLD == f["name"]
        dsel = (Y[m] == "D")
        dwH = f["W"][iD] - f["W"][iH]
        dwA = f["W"][iD] - f["W"][iA]
        cdh = (f["Tva"][:, close_idx] * dwH[close_idx])[dsel]
        cda = (f["Tva"][:, close_idx] * dwA[close_idx])[dsel]
        rows[f["name"]] = [float(dwH[close_idx].sum()), float(dwA[close_idx].sum()),
                           float(cdh.sum(axis=1).mean()), float(cda.sum(axis=1).mean()),
                           cancel(cdh)[0]]
    labs = ["closeness coef contrast D-H (sum)", "closeness coef contrast D-A (sum)",
            "closeness contribution D-H", "closeness contribution D-A",
            "closeness cancel ratio D-H"]
    for i, lab in enumerate(labs):
        v = [rows[f["name"]][i] for f in sc_fits]
        print(f"| {lab} | {v[0]:+.5f} | {v[1]:+.5f} | {v[2]:+.5f} | {min(v):+.5f} | "
              f"{max(v):+.5f} | {max(v)-min(v):.5f} | {len(set(np.sign(v))) == 1} |")
    per_feat_flips = sum(
        1 for j in range(n_close)
        if len({np.sign((f["W"][iD] - f["W"][iH])[close_idx[j]]) for f in sc_fits}) > 1)
    print(f"\n  individual closeness coefficients with a D-H sign flip across folds: "
          f"{per_feat_flips}/{n_close}")

    rule("STEP 12 -- CLOSENESS ABLATION (mathematical attribution)")
    print("  Zeroing a standardised closeness column sets it to the training mean.")
    print("  This is mathematical attribution, not feature deletion, intervention,")
    print("  or retraining. No model is refitted.\n")
    S2 = np.empty((len(Y), 3)); S2b = np.empty((len(Y), 3))
    for f in sc_fits:
        m = FOLD == f["name"]
        T = f["Tva"]
        S2[m] = f["model"].decision_function(T)[:, f["order"]]
        Tz = T.copy(); Tz[:, close_idx] = 0.0
        S2b[m] = f["model"].decision_function(Tz)[:, f["order"]]
    e = np.exp(S2b - S2b.max(axis=1, keepdims=True))
    Pz = e / e.sum(axis=1, keepdims=True)
    gz = draw_geometry(Pz, Y, CLASS_ORDER)
    print(f"  delta mean P(D)          : {gz['pd_mean'] - g2['pd_mean']:+.6f}")
    print(f"  delta Draw argmax share  : {gz['draw_argmax_share'] - g2['draw_argmax_share']:+.6f}")
    print(f"  delta mean M(D-H)        : "
          f"{(S2b[:,iD]-S2b[:,iH]).mean() - (S2[:,iD]-S2[:,iH]).mean():+.6f}")
    print(f"  delta mean M(D-A)        : "
          f"{(S2b[:,iD]-S2b[:,iA]).mean() - (S2[:,iD]-S2[:,iA]).mean():+.6f}")

    rule("STEP 13 -- PRE-REGISTERED DIAGNOSIS CLASSIFICATION")
    coherent = (len({np.sign(rows[f["name"]][2]) for f in sc_fits}) == 1
                and len({np.sign(rows[f["name"]][0]) for f in sc_fits}) == 1)
    improved = (ll2 < ll1)
    print(f"  coherent directional contribution across all three folds : {coherent}")
    print(f"  validation log loss improves versus V1 at matched C=1.0  : {improved} "
          f"(delta {ll2 - ll1:+.10f})")
    if coherent and improved:
        verdict = "A) REPRESENTATION SUPPORTED"
    elif (not coherent) and (not improved):
        verdict = "B) REPRESENTATION NOT SUPPORTED"
    else:
        verdict = "C) AMBIGUOUS"
    print(f"\n  PRE-REGISTERED CLASSIFICATION: {verdict}")
    print("\n  This classification is descriptive. It asserts no causality, and a single")
    print("  experiment does not prove the Step-6 hypothesis false.")

    rule("STEP 14 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    check("src/models/train.py unchanged",
          after["src/models/train.py"] == before["src/models/train.py"])
    print("\nFILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")
    print("\nSTEP 7 EXPERIMENT COMPLETE -- NOTHING PERSISTED, NOTHING CHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
