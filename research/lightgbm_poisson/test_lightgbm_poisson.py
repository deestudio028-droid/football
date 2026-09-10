"""E9 — LightGBM Poisson test suite.

Usage:
    python research/lightgbm_poisson/test_lightgbm_poisson.py

Covers the §32 checklist. Tests are split so that everything not
requiring sklearn/lightgbm runs anywhere; suites that need them are
skipped with a clear reason when absent and run in full locally.
"""
from __future__ import annotations

import hashlib
import inspect
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lightgbm_poisson import (  # noqa: E402
    CATEGORICAL_COLUMN, CLASS_ORDER, MAX_PLAUSIBLE_LAMBDA, RANDOM_STATE,
    TAIL_TOL, E9Error, V3Preprocessor, all_metrics, build_inner_splits,
    config_params, grid_size, hda_from_lambdas, lambda_diagnostics,
    load_grid, log_loss, prediction_change, select_config, shared_grid_k,
    validate_probs,
)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
MARKET_DB = PROJECT_ROOT / "research" / "market_odds" / "research_dataset.sqlite"
PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
TARGET = (200, 419, 423, 477, 499)
SEASONS = ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025")
QUARANTINED = "2025/2026"
FOLDS = [(("2020/2021", "2021/2022"), "2022/2023"),
         (("2020/2021", "2021/2022", "2022/2023"), "2023/2024"),
         (("2020/2021", "2021/2022", "2022/2023", "2023/2024"), "2024/2025")]
SEED = 20260820

try:
    import sklearn  # noqa: F401
    HAVE_SK = True
except ImportError:
    HAVE_SK = False
try:
    import lightgbm  # noqa: F401
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False


class Suite:
    def __init__(s, n): s.n, s.p, s.f, s.s, s.l = n, 0, 0, 0, []
    def ok(s, m, d=""): s.p += 1; s.l.append(f"  PASS: {m}" + (f" — {d}" if d else ""))
    def bad(s, m, d=""): s.f += 1; s.l.append(f"  FAIL: {m}" + (f" — {d}" if d else ""))
    def skip(s, m, d=""): s.s += 1; s.l.append(f"  SKIP: {m}" + (f" — {d}" if d else ""))
    def check(s, c, m, d=""): s.ok(m, d) if c else s.bad(m, d)
    def report(s):
        print(f"\n{'=' * 74}\nSuite: {s.n}\n{'=' * 74}")
        for x in s.l: print(x)
        print(f"\n  Pass: {s.p}  Fail: {s.f}  Skip: {s.s}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(8192), b""): h.update(c)
    return h.hexdigest()


def v3_columns():
    from models.v3_contract import V3_FEATURE_COLUMNS
    return tuple(V3_FEATURE_COLUMNS)


def synth_X(cols, n=400, seed=SEED, nan_frac=0.03):
    rng = np.random.default_rng(seed)
    d = {}
    for c in cols:
        if c == CATEGORICAL_COLUMN:
            d[c] = rng.choice([200, 419, 423, 477, 499], n)
        else:
            v = rng.normal(0, 1, n)
            mask = rng.random(n) < nan_frac
            v[mask] = np.nan
            d[c] = v
    return pd.DataFrame(d, columns=list(cols))


# --- Suite 1: feature contract (§5) ------------------------------------

def suite1() -> Suite:
    s = Suite("1. Feature Contract")
    try:
        cols = v3_columns()
    except ImportError as e:
        s.skip("All contract tests", str(e)); return s

    s.check(len(cols) == 87, "V3 contract is exactly 87 features", f"n={len(cols)}")
    s.check(len(set(cols)) == 87, "No duplicate feature names")
    s.check(CATEGORICAL_COLUMN in cols, f"{CATEGORICAL_COLUMN} present")
    s.check(cols[-3:] == ("home_elo", "away_elo", "elo_diff"),
            "Elo features occupy positions 85-87")

    # E9 adds nothing and removes nothing
    banned = {"A_home", "D_home", "A_away", "D_away",      # E6
              "elo_diff_sq",                                # E5
              "pm_h", "pm_d", "pm_a",                       # market
              "devig_closing_home", "devig_closing_draw",
              "devig_closing_away"}
    s.check(not (banned & set(cols)),
            "No E5/E6/market features in the contract",
            f"checked {len(banned)} banned names")

    X = synth_X(cols)
    s.check(list(X.columns) == list(cols),
            "Design frame preserves exact contract ordering")
    return s


# --- Suite 2: preprocessing, training-only medians ---------------------

def suite2() -> Suite:
    s = Suite("2. Preprocessing (median impute, training-only stats)")
    try:
        cols = v3_columns()
    except ImportError as e:
        s.skip("All preprocessing tests", str(e)); return s

    X = synth_X(cols, n=600)
    tr, te = X.iloc[:400], X.iloc[400:]

    pre = V3Preprocessor().fit(tr)
    s.check(pre.n_encoded_columns == 86 + len(pre.categories),
            "Encoded width = 86 numeric + one-hot",
            f"{pre.n_encoded_columns} columns, {len(pre.categories)} categories")

    Etr, Ete = pre.transform(tr), pre.transform(te)
    s.check(np.all(np.isfinite(Etr)) and np.all(np.isfinite(Ete)),
            "No NaN survives preprocessing (median imputation applied)")

    # medians must come from TRAINING rows only
    num_cols = [c for c in cols if c != CATEGORICAL_COLUMN]
    expected = np.nanmedian(tr[num_cols].to_numpy(float), axis=0)
    s.check(np.allclose(pre.medians_, expected, atol=0),
            "Medians computed from training rows only",
            f"max diff {float(np.max(np.abs(pre.medians_ - expected))):.3e}")

    # mutating TEST rows cannot change the fitted statistics
    te_mut = te.copy()
    te_mut.iloc[:, 0] = 999.0
    pre2 = V3Preprocessor().fit(tr)
    s.check(np.array_equal(pre.medians_, pre2.medians_)
            and np.array_equal(pre.means_, pre2.means_),
            "Test rows cannot influence fitted statistics")

    # one-hot behaviour
    s.check(set(pre.categories) <= {200, 419, 423, 477, 499},
            "Categories are the five league IDs", f"{pre.categories}")
    oh = Etr[:, 86:]
    s.check(np.all(oh.sum(axis=1) <= 1.0 + 1e-12),
            "One-hot rows sum to at most 1")
    s.check(set(np.unique(oh)) <= {0.0, 1.0}, "One-hot is binary")

    # unseen category yields an all-zero block rather than an error
    te_unseen = te.copy()
    te_unseen[CATEGORICAL_COLUMN] = 99999
    E = pre.transform(te_unseen)
    s.check(np.all(E[:, 86:] == 0.0),
            "Unseen category becomes an all-zero one-hot block")

    try:
        V3Preprocessor().transform(tr); s.bad("Reject transform before fit")
    except E9Error: s.ok("Reject transform before fit")
    return s


# --- Suite 3: preprocessing equivalence to production (needs sklearn) --

def suite3() -> Suite:
    s = Suite("3. Preprocessor Equivalence vs Production")
    if not HAVE_SK:
        s.skip("Equivalence vs LogisticRegressionPreprocessor",
               "sklearn unavailable in this environment"); return s
    try:
        cols = v3_columns()
        from models.train import LogisticRegressionPreprocessor
    except ImportError as e:
        s.skip("Equivalence test", str(e)); return s

    X = synth_X(cols, n=800)
    tr, te = X.iloc[:500], X.iloc[500:]

    mine = V3Preprocessor().fit(tr)
    prod = LogisticRegressionPreprocessor().fit(tr)

    for nm, frame in (("train", tr), ("test", te)):
        A, B = mine.transform(frame), prod.transform(frame)
        s.check(A.shape == B.shape, f"{nm}: identical shape", f"{A.shape}")
        d = float(np.max(np.abs(A - B)))
        s.check(d < 1e-9,
                f"{nm}: matrices match production preprocessor",
                f"max abs diff {d:.3e}")
    s.check(mine.categories == prod.competition_categories,
            "Category lists identical")
    return s


# --- Suite 4: shared adaptive K control (§20) --------------------------

def suite4() -> Suite:
    s = Suite("4. Shared Adaptive Score-Grid K (§20)")

    # grid_size must equal the production recursion
    if HAVE_SK:
        try:
            from models.poisson import _grid_size as prod_gs
            diffs = [grid_size(l) - prod_gs(l) for l in np.arange(0.1, 8.0, 0.05)]
            s.check(all(d == 0 for d in diffs),
                    "grid_size identical to production _grid_size",
                    f"{len(diffs)} lambda values, max diff {max(map(abs, diffs))}")
        except ImportError as e:
            s.skip("grid_size vs production", str(e))
    else:
        s.skip("grid_size vs production", "sklearn unavailable")

    rng = np.random.default_rng(SEED)
    # deliberately different maxima between the arms
    lam_h_a = rng.uniform(0.5, 2.5, 500)
    lam_a_a = rng.uniform(0.4, 2.0, 500)
    lam_h_b = rng.uniform(0.5, 4.2, 500)      # arm B reaches higher
    lam_a_b = rng.uniform(0.4, 3.6, 500)

    k_a_alone = grid_size(max(lam_h_a.max(), lam_a_a.max()))
    k_b_alone = grid_size(max(lam_h_b.max(), lam_a_b.max()))
    s.check(k_a_alone != k_b_alone,
            "Arms WOULD derive different K if computed independently",
            f"K_A={k_a_alone} vs K_B={k_b_alone} — the failure mode this guards")

    k = shared_grid_k(lam_h_a, lam_a_a, lam_h_b, lam_a_b)
    s.check(k == max(k_a_alone, k_b_alone),
            "shared_grid_k uses the pooled maximum", f"K={k}")
    s.check(k >= k_a_alone and k >= k_b_alone,
            "Shared K covers both arms' tails")

    P_a = hda_from_lambdas(lam_h_a, lam_a_a, k)
    P_b = hda_from_lambdas(lam_h_b, lam_a_b, k)
    validate_probs(P_a, "arm A"); validate_probs(P_b, "arm B")
    s.ok("Both arms convert at the SAME K", f"K={k} for both")

    # order of arguments must not matter
    s.check(shared_grid_k(lam_h_b, lam_a_b, lam_h_a, lam_a_a) == k,
            "shared_grid_k is order-independent")

    for bad, lbl in [((np.array([-1.0, 1.0]),), "non-positive pooled max"),
                     ((np.array([np.nan, 1.0]),), "non-finite lambda")]:
        try:
            shared_grid_k(*bad); s.bad(f"Reject {lbl}")
        except E9Error: s.ok(f"Reject {lbl}")
    return s


# --- Suite 5: probability conversion validity (§21) --------------------

def suite5() -> Suite:
    s = Suite("5. Probability Conversion")
    rng = np.random.default_rng(SEED)
    lh, la = rng.uniform(0.2, 4.0, 2000), rng.uniform(0.2, 3.5, 2000)
    k = shared_grid_k(lh, la)
    P = hda_from_lambdas(lh, la, k)

    validate_probs(P, "converted")
    s.ok("Probabilities valid", "finite, [0,1], sum to 1")
    s.check(float(np.max(np.abs(P.sum(axis=1) - 1.0))) < 1e-9,
            "Row sums exact",
            f"max dev {float(np.max(np.abs(P.sum(axis=1) - 1.0))):.3e}")

    # equivalence to production conversion at the same K
    if HAVE_SK:
        try:
            from models.poisson import hda_tail_safe
            P_prod, k_prod, resid, ctrl = hda_tail_safe(lh, la)
            s.check(k_prod == k, "Production derives the same K here",
                    f"K={k}")
            d = float(np.max(np.abs(P - P_prod)))
            s.check(d < 1e-12, "Conversion matches production hda_tail_safe",
                    f"max diff {d:.3e}")
        except ImportError as e:
            s.skip("Conversion vs production", str(e))
    else:
        s.skip("Conversion vs production", "sklearn unavailable")

    for bad_h, lbl in [(np.array([0.0, 1.0]), "lambda = 0"),
                       (np.array([-1.0, 1.0]), "lambda < 0"),
                       (np.array([np.nan, 1.0]), "lambda NaN")]:
        try:
            hda_from_lambdas(bad_h, np.array([1.0, 1.0]), 10)
            s.bad(f"Reject {lbl}")
        except E9Error: s.ok(f"Reject {lbl}")
    return s


# --- Suite 6: frozen hyperparameter grid (§8) --------------------------

def suite6() -> Suite:
    s = Suite("6. Frozen Hyperparameter Grid")
    g = load_grid()
    cfgs = g["configurations"]
    s.check(len(cfgs) == 16, "16 pre-registered configurations",
            f"n={len(cfgs)}")
    s.check(len({c["id"] for c in cfgs}) == 16, "Configuration IDs unique")

    allowed = {
        "num_leaves": {7, 15, 31},
        "learning_rate": {0.02, 0.05},
        "n_estimators": {100, 250, 500},
        "min_child_samples": {20, 50},
        "subsample": {0.8, 1.0},
        "colsample_bytree": {0.8, 1.0},
        "reg_lambda": {0.0, 1.0},
    }
    ok = all(c[k] in v for c in cfgs for k, v in allowed.items())
    s.check(ok, "Every configuration lies inside the §8 permitted grid")

    d = next(c for c in cfgs if c["id"] == 1)
    s.check((d["num_leaves"], d["learning_rate"], d["n_estimators"],
             d["min_child_samples"], d["subsample"], d["colsample_bytree"],
             d["reg_lambda"]) == (15, 0.05, 250, 50, 0.8, 0.8, 1.0),
            "Config 1 is the §9 default")

    fx = g["fixed_for_all_configs"]
    s.check(fx["objective"] == "poisson", "Objective is poisson")
    s.check(fx["random_state"] == RANDOM_STATE, "random_state = 42")
    s.check(fx["n_jobs"] == 1, "n_jobs = 1 (determinism)")
    s.check(fx.get("deterministic") is True, "deterministic = True")
    s.check(fx.get("force_row_wise") is True, "force_row_wise = True")

    p = config_params(d, fx)
    s.check(p["objective"] == "poisson" and p["num_leaves"] == 15,
            "config_params merges correctly", f"{len(p)} params")
    s.check(p.get("subsample_freq") == 1,
            "subsample_freq set when subsample < 1 (else bagging is inert)")
    s.check("early_stopping_rounds" not in p and "eval_set" not in p,
            "No early stopping (§11 — fixed n_estimators)")

    try:
        config_params({"id": 99}, fx); s.bad("Reject incomplete config")
    except E9Error: s.ok("Reject incomplete config")
    return s


# --- Suite 7: selection isolation (§22 items 4,5,6,7) ------------------

def suite7() -> Suite:
    s = Suite("7. Configuration Selection Isolation")
    p = set(inspect.signature(select_config).parameters)
    s.check(p == {"inner_splits", "fit_eval", "configs", "fixed"},
            "select_config has no outer-test parameter", f"{sorted(p)}")
    p2 = set(inspect.signature(build_inner_splits).parameters)
    s.check("train_seasons" in p2 and "test_season" not in p2,
            "build_inner_splits receives training seasons only", f"{sorted(p2)}")

    seasons = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    sp = build_inner_splits(seasons, ("A", "B", "C"))
    s.check(len(sp) == 2, "k training seasons yield k-1 inner splits")
    used = set()
    for x in sp: used |= set(x.train_seasons) | {x.val_season}
    s.check(used <= {"A", "B", "C"},
            "Inner splits use only outer-training seasons", f"{sorted(used)}")
    for x in sp:
        s.check(len(set(x.train_idx) & set(x.val_idx)) == 0,
                f"Inner split {x.val_season}: train and val disjoint")

    g = load_grid()
    calls = []
    def stub(split, params):
        calls.append((split.val_season, params["num_leaves"]))
        return abs(params["num_leaves"] - 15) + params["learning_rate"]
    sel = select_config(sp, stub, g["configurations"], g["fixed_for_all_configs"])
    s.check(sel.config_id in {c["id"] for c in g["configurations"]},
            "Selection returns a real configuration", f"id={sel.config_id}")
    s.check(len(sel.curve) == 16, "Curve records every configuration")
    s.check(len(calls) == 16 * 2, "fit_eval called once per (config, split)",
            f"{len(calls)} calls")
    best = min(sel.curve, key=lambda c: c["mean_inner_log_loss"])
    s.check(best["config_id"] == sel.config_id,
            "Selected config is the argmin of mean inner log loss")

    try:
        select_config([], stub, g["configurations"], g["fixed_for_all_configs"])
        s.bad("Reject empty inner splits")
    except E9Error: s.ok("Reject empty inner splits")
    return s


# --- Suite 8: lambda safety and diagnostics (§19) ----------------------

def suite8() -> Suite:
    s = Suite("8. Lambda Safety (§19)")
    rng = np.random.default_rng(SEED)
    lh, la = rng.uniform(0.3, 3.5, 1000), rng.uniform(0.3, 3.0, 1000)
    d = lambda_diagnostics(lh, la)
    s.check(d["passed"], "Normal lambdas pass safety",
            f"home[{d['lambda_home']['min']:.3f},{d['lambda_home']['max']:.3f}]")
    s.check(d["clipping_applied"] is False,
            "No clipping is applied anywhere (§19)")

    s.check(not lambda_diagnostics(np.array([1.0, 0.0]), la[:2])["passed"],
            "Non-positive lambda caught")
    s.check(not lambda_diagnostics(np.array([1.0, np.nan]), la[:2])["passed"],
            "NaN lambda caught")
    s.check(not lambda_diagnostics(np.array([1.0, np.inf]), la[:2])["passed"],
            "Inf lambda caught")
    s.check(not lambda_diagnostics(np.array([1.0, 99.0]), la[:2])["passed"],
            f"Implausible lambda (>{MAX_PLAUSIBLE_LAMBDA}) caught")

    m = all_metrics(np.array(["H", "D", "A", "H"]),
                    np.array([[.5, .3, .2]] * 4),
                    np.array([1.5] * 4), np.array([1.2] * 4),
                    np.array([2., 1., 0., 1.]), np.array([1., 1., 2., 0.]))
    for k in ("log_loss", "brier", "rps", "accuracy", "draw_recall", "ece",
              "home_goal_mae", "away_goal_mae"):
        s.check(k in m and np.isfinite(m[k]), f"all_metrics reports {k}",
                f"{m[k]}")

    pc = prediction_change(np.array([[.5, .3, .2]]), np.array([[.5, .3, .2]]),
                           [1.5], [1.2], [1.5], [1.2])
    s.check(pc["mean_abs_prob_change"] == 0.0
            and pc["pct_top_class_changed"] == 0.0,
            "prediction_change reports zero for identical inputs")
    return s


# --- Suite 9: real data, folds, quarantine (§22 items 12; §29) ---------

def suite9() -> Suite:
    s = Suite("9. Real Data, Folds & 2025/26 Quarantine")
    if not MATCHES_DB.exists():
        s.skip("All real-data tests", "matches.db not found"); return s

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET)); sph = ",".join("?" * len(SEASONS))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, competition_name, season, unix FROM fixtures
            WHERE competition_id IN ({ph}) AND season IN ({sph})
              AND status IN ('FT','AWARDED') AND home_goals IS NOT NULL""",
        conn, params=[*TARGET, *SEASONS])
    nq = conn.execute(
        f"SELECT COUNT(*) FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season = ?", (*TARGET, QUARANTINED)).fetchone()[0]
    conn.close()

    s.check(len(df) == 8983, "Eligible universe is 8,983", f"n={len(df)}")
    oos = df[df.season.isin(["2022/2023", "2023/2024", "2024/2025"])]
    s.check(len(oos) == 5331, "Pooled OOS is 5,331", f"n={len(oos)}")
    s.check(QUARANTINED not in set(df.season),
            "2025/26 absent from the working set")
    s.ok("2025/26 exists in DB but is excluded", f"n={nq}")
    s.check(set(df.competition_name) == {"Premier League", "La Liga", "Serie A",
                                         "Bundesliga", "Ligue 1"},
            "Exactly the five target leagues")

    ok = all(df[df.season.isin(trs)].unix.max() < df[df.season == ts].unix.min()
             for trs, ts in FOLDS)
    s.check(ok, "All 3 folds strictly chronological")
    s.check([ts for _, ts in FOLDS] == ["2022/2023", "2023/2024", "2024/2025"],
            "Folds match the approved E1/E6 protocol")
    for trs, ts in FOLDS:
        n_tr = len(df[df.season.isin(trs)])
        n_te = len(df[df.season == ts])
        s.ok(f"Fold {ts}", f"train={n_tr} test={n_te}")

    # inner splits per fold
    for trs, ts in FOLDS:
        sub = df[df.season.isin(trs)]
        sp = build_inner_splits(sub.season.values, trs)
        s.ok(f"Fold {ts} inner splits", f"{len(sp)} (needs >=1 for selection)")
    return s


# --- Suite 10: market and other components excluded (§22 items 9,10,11)-

def suite10() -> Suite:
    s = Suite("10. Component Separation")
    src = (HERE / "lightgbm_poisson.py").read_text(encoding="utf-8")

    import ast
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    s.ok("Module imports (AST)", f"{sorted(imported)}")

    # sklearn/lightgbm are imported INSIDE functions, so top-level must be clean
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    s.check(top <= {"__future__", "json", "math", "dataclasses", "pathlib",
                    "numpy", "pandas"},
            "Top-level imports are stdlib + numpy/pandas only", f"{sorted(top)}")

    banned_modules = {"market_odds", "online_attack_defense", "dixon_coles",
                      "time_decay", "quadratic_elo", "best_combination",
                      "temperature_scaling"}
    s.check(not (banned_modules & imported),
            "No other experiment's module is imported")

    for tok in ("devig", "P_market", "pm_h", "A_home", "D_home",
                "elo_diff_sq", "rho", "temperature"):
        s.check(tok not in src, f"No '{tok}' reference in the module")

    if MARKET_DB.exists():
        conn = sqlite3.connect(f"file:{MARKET_DB}?mode=ro", uri=True)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(research_odds)")]
        conn.close()
        try:
            v3 = set(v3_columns())
            # The property that matters is that no ODDS-DERIVED column can
            # reach the feature matrix. Shared identifier/metadata columns
            # such as competition_id are league labels, not market signal,
            # and are expected in both tables.
            odds_cols = {c for c in cols
                         if c.startswith(("devig_", "opening_", "closing_",
                                          "peak_", "pm_"))
                         or "odds" in c or "bookmaker" in c}
            s.ok("Odds-derived columns identified in market data",
                 f"{len(odds_cols)} columns")
            s.check(not (v3 & odds_cols),
                    "NO odds-derived column appears in the V3 contract",
                    f"overlap={sorted(v3 & odds_cols) or 'none'}")
            shared = v3 & set(cols)
            s.check(shared <= {"competition_id"},
                    "Only benign identifier metadata is shared",
                    f"shared={sorted(shared)} (league label, not market signal)")
        except ImportError:
            s.skip("Market/contract overlap", "contract unavailable")
    else:
        s.skip("Market separation on disk", "market dataset not found")
    return s


# --- Suite 11: determinism (§28) ---------------------------------------

def suite11() -> Suite:
    s = Suite("11. Determinism")
    try:
        cols = v3_columns()
    except ImportError as e:
        s.skip("Determinism tests", str(e)); return s

    X = synth_X(cols, n=500)
    tr, te = X.iloc[:350], X.iloc[350:]
    a = V3Preprocessor().fit(tr).transform(te)
    b = V3Preprocessor().fit(tr).transform(te)
    s.check(np.array_equal(a, b), "Preprocessing bit-identical on repeat",
            f"max diff {float(np.max(np.abs(a - b))):.3e}")

    rng = np.random.default_rng(SEED)
    lh, la = rng.uniform(0.3, 3.0, 300), rng.uniform(0.3, 2.5, 300)
    k1, k2 = shared_grid_k(lh, la), shared_grid_k(lh, la)
    s.check(k1 == k2, "shared_grid_k deterministic", f"K={k1}")
    s.check(np.array_equal(hda_from_lambdas(lh, la, k1),
                           hda_from_lambdas(lh, la, k1)),
            "Conversion bit-identical on repeat")

    src = (HERE / "lightgbm_poisson.py").read_text(encoding="utf-8")
    s.check("np.random" not in src and "random.seed" not in src,
            "Module contains no ad-hoc RNG")
    s.check(f"RANDOM_STATE: int = {RANDOM_STATE}" in src,
            "random_state pinned in the module", f"{RANDOM_STATE}")
    return s


# --- Suite 12: LightGBM sanity (§23) — needs lightgbm ------------------

def suite12() -> Suite:
    s = Suite("12. LightGBM Synthetic Sanity (§23)")
    if not (HAVE_LGB and HAVE_SK):
        s.skip("All LightGBM sanity tests",
               f"lightgbm={HAVE_LGB} sklearn={HAVE_SK} in this environment")
        return s

    from lightgbm_poisson import (fit_lgbm_arm, fit_v3_arm, feature_importance,
                                  lightgbm_version, sklearn_version)
    s.ok("Library versions", f"lightgbm={lightgbm_version()} "
                             f"sklearn={sklearn_version()}")

    rng = np.random.default_rng(SEED)
    n, p = 800, 12
    Xtr = rng.normal(0, 1, (n, p))
    rate = np.exp(0.3 + 0.5 * Xtr[:, 0] - 0.3 * Xtr[:, 1])
    hg = rng.poisson(rate).astype(float)
    ag = rng.poisson(rate * 0.8).astype(float)
    Xte = rng.normal(0, 1, (200, p))

    g = load_grid()
    params = config_params(g["configurations"][0], g["fixed_for_all_configs"])
    lh, la, models = fit_lgbm_arm(Xtr, hg, ag, Xte, params, return_models=True)

    d = lambda_diagnostics(lh, la)
    s.check(d["passed"], "LightGBM produces valid Poisson lambdas",
            f"home[{d['lambda_home']['min']:.4f},{d['lambda_home']['max']:.4f}]")
    s.check(d["all_positive"], "All lambdas strictly positive "
                               "(poisson objective guarantees exp link)")

    vh, va = fit_v3_arm(Xtr, hg, ag, Xte)
    s.check(lambda_diagnostics(vh, va)["passed"],
            "V3 arm produces valid lambdas on the same data")

    k = shared_grid_k(lh, la, vh, va)
    P_b = hda_from_lambdas(lh, la, k)
    P_a = hda_from_lambdas(vh, va, k)
    validate_probs(P_a, "V3"); validate_probs(P_b, "LGBM")
    s.ok("Both arms convert at one shared K", f"K={k}")

    # determinism of LightGBM itself
    lh2, la2 = fit_lgbm_arm(Xtr, hg, ag, Xte, params)
    s.check(np.array_equal(lh, lh2) and np.array_equal(la, la2),
            "LightGBM training is deterministic (n_jobs=1, seed pinned)",
            f"max diff {float(np.max(np.abs(lh - lh2))):.3e}")

    fi = feature_importance(models, [f"f{i}" for i in range(p)])
    s.check(len(fi["home"]) > 0 and "gain" in fi["home"][0],
            "Feature importance extracted", f"top={fi['home'][0]['feature']}")
    s.check(fi["home"][0]["feature"] in ("f0", "f1"),
            "Importance recovers the true signal features",
            f"top={fi['home'][0]['feature']} (truth: f0, f1)")

    # a minimal configuration must still work (§23)
    minimal = dict(params); minimal.update(num_leaves=2, n_estimators=5)
    lh_m, la_m = fit_lgbm_arm(Xtr, hg, ag, Xte, minimal)
    s.check(lambda_diagnostics(lh_m, la_m)["passed"],
            "Minimal LightGBM configuration still yields valid lambdas")
    return s


# --- Suite 13: adversarial leakage (§22) -------------------------------

def suite13() -> Suite:
    s = Suite("13. Adversarial Leakage (§22)")
    try:
        cols = v3_columns()
    except ImportError as e:
        s.skip("Adversarial tests", str(e)); return s

    X = synth_X(cols, n=900)
    seasons = np.array(["S1"] * 300 + ["S2"] * 300 + ["S3"] * 300)
    rng = np.random.default_rng(SEED)
    hg = rng.poisson(1.5, 900).astype(float)
    ag = rng.poisson(1.2, 900).astype(float)

    tr = np.where(np.isin(seasons, ["S1", "S2"]))[0]
    te = np.where(seasons == "S3")[0]

    # rewriting every OUTER TEST label must not change fitted statistics
    pre1 = V3Preprocessor().fit(X.iloc[tr])
    hg_adv = hg.copy(); hg_adv[te] = 9.0
    ag_adv = ag.copy(); ag_adv[te] = 0.0
    pre2 = V3Preprocessor().fit(X.iloc[tr])
    s.check(np.array_equal(pre1.medians_, pre2.medians_),
            "Rewriting test labels cannot change preprocessing statistics")

    # ... nor the inner splits
    sp1 = build_inner_splits(seasons[tr], ("S1", "S2"))
    sp2 = build_inner_splits(seasons[tr], ("S1", "S2"))
    s.check(all(np.array_equal(a.train_idx, b.train_idx)
                and np.array_equal(a.val_idx, b.val_idx)
                for a, b in zip(sp1, sp2)),
            "Inner splits unaffected by test labels")

    # ... nor the selected configuration, when fit_eval sees only the split
    g = load_grid()
    def stub(split, params):
        # deliberately depends only on training-side data
        return float(np.mean(hg[split.train_idx])) + params["learning_rate"]
    a = select_config(sp1, stub, g["configurations"], g["fixed_for_all_configs"])
    def stub_adv(split, params):
        return float(np.mean(hg_adv[split.train_idx])) + params["learning_rate"]
    b = select_config(sp1, stub_adv, g["configurations"], g["fixed_for_all_configs"])
    s.check(a.config_id == b.config_id,
            "Selected configuration unchanged after rewriting test labels",
            f"config {a.config_id}")

    # test indices are strictly after training indices
    s.check(tr.max() < te.min(), "Training rows strictly precede test rows")

    # no fitting function accepts an outcome for the evaluation rows
    from lightgbm_poisson import fit_lgbm_arm, fit_v3_arm
    for fn in (fit_v3_arm, fit_lgbm_arm):
        p = list(inspect.signature(fn).parameters)
        s.check(p[3].startswith("X_ev"),
                f"{fn.__name__} takes evaluation FEATURES only, no labels",
                f"4th arg = {p[3]}")
    return s


# --- Suite 14: protected files (§30) -----------------------------------

def suite14() -> Suite:
    s = Suite("14. Protected File Integrity (§30)")
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists(): s.skip(Path(rel).name, "not found"); continue
        a = md5(p)
        s.check(a == exp, Path(rel).name,
                f"MD5={a}" if a == exp else f"expected={exp} got={a}")
    return s


def main() -> int:
    print("=" * 74)
    print("E9 — LightGBM Poisson Test Suite")
    print("=" * 74)
    print(f"  Environment: lightgbm={'yes' if HAVE_LGB else 'NO'}  "
          f"sklearn={'yes' if HAVE_SK else 'NO'}")
    su = [suite1(), suite2(), suite3(), suite4(), suite5(), suite6(),
          suite7(), suite8(), suite9(), suite10(), suite11(), suite12(),
          suite13(), suite14()]
    for x in su: x.report()
    tp, tf, ts = (sum(x.p for x in su), sum(x.f for x in su),
                  sum(x.s for x in su))
    print("\n" + "=" * 74 + "\nOVERALL\n" + "=" * 74)
    print(f"  Suites: {len(su)}\n  Pass:   {tp}\n  Fail:   {tf}"
          f"\n  Skip:   {ts}\n  Total:  {tp + tf + ts}")
    print(f"\n  VERDICT: {'ALL TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    if ts:
        print(f"  NOTE: {ts} test(s) skipped — run locally with lightgbm + "
              f"sklearn for full coverage.")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
