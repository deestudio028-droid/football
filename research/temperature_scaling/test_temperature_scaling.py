"""E8 — Temperature scaling test suite.

Usage:
    python research/temperature_scaling/test_temperature_scaling.py

Covers the 14 required leakage tests plus the formula, identity control,
sharpening/softening direction, normalisation, zero handling, ranking
preservation, determinism, fold integrity, fixture accounting, the
2025/26 quarantine and protected-file integrity.

Runs on numpy/pandas alone — E8's base is the stored market probability
dataset, so no model fitting (and no sklearn) is required.
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

from temperature_scaling import (  # noqa: E402
    CLASS_ORDER, EPS, IDENTITY_TOL, TEMPERATURE_GRID, TemperatureError,
    all_metrics, apply_temperature, confidence_profile, ece, identity_control,
    log_loss, max_calibration_error, ranking_preserved, reliability_bins,
    select_temperature, select_temperature_nested, temperature_effect,
    temperature_stability, validate_probs,
)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
MARKET_DB = PROJECT_ROOT / "research" / "market_odds" / "research_dataset.sqlite"
PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
TARGET = (200, 419, 423, 477, 499)
E8_TEST_SEASONS = ("2023/2024", "2024/2025")
QUARANTINED = "2025/2026"
SEED = 20260820


class Suite:
    def __init__(s, n): s.n, s.p, s.f, s.s, s.l = n, 0, 0, 0, []
    def ok(s, m, d=""): s.p += 1; s.l.append(f"  PASS: {m}" + (f" — {d}" if d else ""))
    def bad(s, m, d=""): s.f += 1; s.l.append(f"  FAIL: {m}" + (f" — {d}" if d else ""))
    def skip(s, m, d=""): s.s += 1; s.l.append(f"  SKIP: {m}" + (f" — {d}" if d else ""))
    def check(s, c, m, d=""): s.ok(m, d) if c else s.bad(m, d)
    def report(s):
        print(f"\n{'=' * 72}\nSuite: {s.n}\n{'=' * 72}")
        for x in s.l: print(x)
        print(f"\n  Pass: {s.p}  Fail: {s.f}  Skip: {s.s}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(8192), b""): h.update(c)
    return h.hexdigest()


def rand_probs(n, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.dirichlet([2.0, 1.2, 1.8], n)


def rand_y(n, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array(CLASS_ORDER), n, p=[0.44, 0.25, 0.31])


def real_market():
    conn = sqlite3.connect(f"file:{MARKET_DB}?mode=ro", uri=True)
    df = pd.read_sql_query(
        """SELECT fixture_id, season, competition_name,
                  devig_closing_home h, devig_closing_draw d,
                  devig_closing_away a FROM research_odds""", conn)
    conn.close()
    return df


# --- Suite 1: formula --------------------------------------------------

def suite1() -> Suite:
    s = Suite("1. Temperature Formula")
    P = rand_probs(500)

    # direct formula check against the power form
    for T in (0.8, 0.9, 1.1, 1.2):
        manual = P ** (1.0 / T)
        manual = manual / manual.sum(axis=1, keepdims=True)
        got = apply_temperature(P, T)
        d = float(np.max(np.abs(got - manual)))
        s.check(d < 1e-12, f"log-space matches P^(1/T) form at T={T}",
                f"max diff {d:.3e}")

    s.check(TEMPERATURE_GRID == (0.80, 0.85, 0.90, 0.95, 1.00,
                                 1.05, 1.10, 1.15, 1.20),
            "Frozen grid unchanged", f"n={len(TEMPERATURE_GRID)}")
    s.check(1.00 in TEMPERATURE_GRID,
            "Grid contains T=1.00 so calibration can decline to act")

    for bad, lbl in [(0.0, "T=0"), (-0.5, "T<0")]:
        try:
            apply_temperature(P, bad); s.bad(f"Reject {lbl}")
        except TemperatureError: s.ok(f"Reject {lbl}")
    for arr, lbl in [(np.array([[np.nan, .5, .5]]), "NaN input"),
                     (np.array([[np.inf, .5, .5]]), "Inf input"),
                     (np.array([[-0.1, .6, .5]]), "negative input"),
                     (np.array([[.3, .3, .2, .2]]), "wrong shape")]:
        try:
            apply_temperature(arr, 0.9); s.bad(f"Reject {lbl}")
        except TemperatureError: s.ok(f"Reject {lbl}")
    return s


# --- Suite 2: identity control (leakage 1) -----------------------------

def suite2() -> Suite:
    s = Suite("2. T=1 Identity Control")
    P, y = rand_probs(1000, 2), rand_y(1000, 3)

    P1 = apply_temperature(P, 1.0)
    d = float(np.max(np.abs(P1 - P)))
    s.check(d == 0.0, "T=1 reproduces probabilities EXACTLY (not just within tol)",
            f"max diff {d:.3e}, tol {IDENTITY_TOL:.0e}")

    ic = identity_control(y, P)
    s.check(ic["passed"], "Full identity control passes",
            f"probs={ic['probs_identical']} preds={ic['predictions_identical']} "
            f"metrics={ic['metrics_identical']}")
    for k in ("log_loss", "brier", "rps", "ece", "accuracy"):
        s.check(ic["base_metrics"][k] == ic["t1_metrics"][k],
                f"{k} identical at T=1", f"{ic['base_metrics'][k]}")

    # identity holds on the real market distribution too
    if MARKET_DB.exists():
        mk = real_market()
        Pm = mk[["h", "d", "a"]].to_numpy(float)
        dm = float(np.max(np.abs(apply_temperature(Pm, 1.0) - Pm)))
        s.check(dm == 0.0, "T=1 identity exact on real market probabilities",
                f"n={len(Pm)}, max diff {dm:.3e}")
    else:
        s.skip("Real-data identity", "market dataset not found")
    return s


# --- Suite 3: direction (sharpen / soften) -----------------------------

def suite3() -> Suite:
    s = Suite("3. Sharpening / Softening Direction")
    P = rand_probs(2000, 4)
    base_conf = P.max(axis=1).mean()

    for T in (0.80, 0.90, 0.95):
        c = apply_temperature(P, T).max(axis=1).mean()
        s.check(c > base_conf, f"T={T} sharpens (confidence up)",
                f"{base_conf:.6f} -> {c:.6f}")
    for T in (1.05, 1.10, 1.20):
        c = apply_temperature(P, T).max(axis=1).mean()
        s.check(c < base_conf, f"T={T} softens (confidence down)",
                f"{base_conf:.6f} -> {c:.6f}")

    confs = [apply_temperature(P, T).max(axis=1).mean()
             for T in TEMPERATURE_GRID]
    s.check(all(confs[i] > confs[i + 1] for i in range(len(confs) - 1)),
            "Mean confidence strictly decreasing in T across the grid",
            f"{[round(c, 4) for c in confs]}")

    for T in (0.85, 1.0, 1.15):
        e = temperature_effect(P, apply_temperature(P, T), T)
        s.check(e["direction_confirmed"],
                f"temperature_effect confirms direction at T={T}",
                e["direction"])
    return s


# --- Suite 4: normalisation, finiteness, zeros -------------------------

def suite4() -> Suite:
    s = Suite("4. Numerical Stability")
    P = rand_probs(2000, 5)

    for T in TEMPERATURE_GRID:
        Q = apply_temperature(P, T)
        validate_probs(Q, f"T={T}")
    s.ok("All grid temperatures produce valid probabilities",
         "finite, [0,1], sum to 1")

    worst = max(float(np.max(np.abs(apply_temperature(P, T).sum(axis=1) - 1.0)))
                for T in TEMPERATURE_GRID)
    s.check(worst < 1e-12, "Row sums exact across the grid",
            f"max deviation {worst:.3e}")

    # exact zeros handled without NaN
    Pz = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    for T in (0.80, 1.20):
        Q = apply_temperature(Pz, T)
        s.check(np.all(np.isfinite(Q)) and np.allclose(Q.sum(1), 1.0),
                f"Exact zeros handled safely at T={T}", f"{Q[1].round(6)}")

    # extreme confidence
    Pe = np.array([[1 - 2e-12, 1e-12, 1e-12], [0.34, 0.33, 0.33]])
    for T in (0.80, 1.20):
        Q = apply_temperature(Pe, T)
        s.check(np.all(np.isfinite(Q)) and np.allclose(Q.sum(1), 1.0),
                f"Extreme confidence handled at T={T}")

    s.check(EPS == 1e-300, "Documented epsilon unchanged", f"EPS={EPS}")
    # epsilon never binds on realistic data
    s.check(float(P.min()) > EPS * 1e100,
            "Epsilon is far below any realistic probability",
            f"min P={float(P.min()):.3e} vs EPS={EPS:.0e}")
    return s


# --- Suite 5: ranking preservation (§17) -------------------------------

def suite5() -> Suite:
    s = Suite("5. Ranking Preservation")
    P = rand_probs(3000, 6)
    for T in TEMPERATURE_GRID:
        Q = apply_temperature(P, T)
        s.check(ranking_preserved(P, Q), f"Class ordering preserved at T={T}")
    for T in (0.80, 1.20):
        Q = apply_temperature(P, T)
        s.check(np.array_equal(P.argmax(1), Q.argmax(1)),
                f"Top class unchanged at T={T}",
                "temperature cannot change argmax when all P>0")
    return s


# --- Suite 6: selection isolation (leakage 2,3,6,13,14) ----------------

def suite6() -> Suite:
    s = Suite("6. Temperature Selection Isolation")
    p = set(inspect.signature(select_temperature).parameters)
    s.check(p == {"y_train", "P_train", "grid", "method"},
            "select_temperature has no test parameter", f"{sorted(p)}")

    n_tr, n_te = 900, 600
    y, P = rand_y(n_tr + n_te, 7), rand_probs(n_tr + n_te, 8)
    tr, te = slice(0, n_tr), slice(n_tr, n_tr + n_te)

    sel = select_temperature(y[tr], P[tr])
    s.ok("select_temperature returns a choice",
         f"T={sel.T}, train LL={sel.train_log_loss:.6f}, n={sel.n_train}")

    # (leakage 2, 13) rewriting every test label cannot change T
    y_mut = y.copy(); y_mut[te] = "D"
    s.check(select_temperature(y_mut[tr], P[tr]).T == sel.T,
            "Rewriting test labels cannot change T", f"T stays {sel.T}")

    # (leakage 3, 14) changing future probabilities cannot change T
    P_mut = P.copy(); P_mut[te] = rand_probs(n_te, 99)
    s.check(select_temperature(y[tr], P_mut[tr]).T == sel.T,
            "Changing future probabilities cannot change T")

    # (leakage 6) outer-test rows physically absent from the call
    s.check(sel.n_train == n_tr,
            "Only training rows entered selection", f"n={sel.n_train}")

    s.check(len(sel.curve) == len(TEMPERATURE_GRID),
            "Curve records every grid candidate")
    best = min(sel.curve, key=lambda c: c["train_log_loss"])
    s.check(best["T"] == sel.T, "Selected T is the argmin of training log loss")

    try:
        select_temperature(np.array([]), P[:0]); s.bad("Reject empty train")
    except TemperatureError: s.ok("Reject empty training set")

    # nested variant: splits must come from training rows only
    pn = set(inspect.signature(select_temperature_nested).parameters)
    s.check("splits" in pn and "test_idx" not in pn,
            "select_temperature_nested takes splits, no test index",
            f"{sorted(pn)}")
    splits = [(np.arange(0, 400), np.arange(400, 700))]
    sn = select_temperature_nested(splits, y, P)
    s.check(sn.method == "nested_inner_validation",
            "Nested method labelled correctly", f"T={sn.T}")
    try:
        select_temperature_nested([], y, P); s.bad("Reject empty splits")
    except TemperatureError: s.ok("Reject empty splits")
    return s


# --- Suite 7: determinism (leakage 8,11,12) ----------------------------

def suite7() -> Suite:
    s = Suite("7. Determinism")
    P, y = rand_probs(800, 9), rand_y(800, 10)

    s.check(np.array_equal(apply_temperature(P, 0.9),
                           apply_temperature(P, 0.9)),
            "apply_temperature bit-identical on repeat")
    a, b = select_temperature(y, P), select_temperature(y, P)
    s.check(a.T == b.T and a.train_log_loss == b.train_log_loss,
            "select_temperature deterministic", f"T={a.T}")
    s.check(reliability_bins(y, P) == reliability_bins(y, P),
            "reliability_bins deterministic")
    s.check(ece(y, P) == ece(y, P), "ECE deterministic")

    src = (HERE / "temperature_scaling.py").read_text(encoding="utf-8")
    s.check("random" not in src.lower(),
            "temperature_scaling.py contains no RNG (leakage 11)")
    return s


# --- Suite 8: metrics sanity -------------------------------------------

def suite8() -> Suite:
    s = Suite("8. Metric & Diagnostic Sanity")
    y, P = rand_y(2000, 11), rand_probs(2000, 12)

    m = all_metrics(y, P)
    for k in ("n", "log_loss", "brier", "rps", "accuracy", "draw_recall",
              "ece", "max_calibration_error", "mean_confidence",
              "max_confidence", "p50"):
        s.check(k in m and np.isfinite(m[k]), f"Reports {k}", f"{m[k]}")

    # a perfectly-calibrated synthetic case should have small ECE
    rng = np.random.default_rng(42)
    Pp = rng.dirichlet([3, 2, 3], 20000)
    idx = np.array([rng.choice(3, p=row) for row in Pp])
    yp = np.array([CLASS_ORDER[i] for i in idx])
    s.check(ece(yp, Pp) < 0.02,
            "ECE near zero when labels are drawn from the probabilities",
            f"ECE={ece(yp, Pp):.6f}")

    b = reliability_bins(y, P)
    s.check(sum(x["n"] for x in b) == 3 * len(y),
            "Reliability bins cover every (fixture, class) pair",
            f"{sum(x['n'] for x in b)} = 3 x {len(y)}")
    s.check(max_calibration_error(y, P) >= ece(y, P),
            "Max calibration error >= ECE")

    c = confidence_profile(P)
    s.check(c["min_confidence"] >= 1 / 3 - 1e-9,
            "Top-class confidence is at least 1/3", f"{c['min_confidence']}")
    s.check(c["max_confidence"] <= 1.0, "Confidence at most 1")
    return s


# --- Suite 9: stability reporting --------------------------------------

def suite9() -> Suite:
    s = Suite("9. Temperature Stability Reporting")
    st = temperature_stability([1.0, 1.0])
    s.check(st["all_identity"] and "declined to act" in st["assessment"],
            "All-T=1 reported as calibration declining to act",
            st["assessment"])
    st = temperature_stability([0.95, 0.95])
    s.check(st["identical_across_folds"] and st["assessment"].startswith("STABLE"),
            "Identical selections reported STABLE")
    st = temperature_stability([0.95, 1.00])
    s.check(st["assessment"].startswith("REASONABLY"),
            "One-step difference reported REASONABLY STABLE", st["assessment"])
    st = temperature_stability([0.80, 1.20])
    s.check(st["assessment"].startswith("UNSTABLE"),
            "Wide spread reported UNSTABLE", st["assessment"])
    s.check(st["boundary_warning"] and st["n_at_grid_boundary"] == 2,
            "Grid-boundary selections flagged, not silently expanded",
            st["boundary_note"][:60])
    st = temperature_stability([0.90, 0.95])
    s.check(not st["boundary_warning"], "No false boundary warning")
    return s


# --- Suite 10: real data, folds, quarantine (leakage 4,5,7) -----------

def suite10() -> Suite:
    s = Suite("10. Real Data, Folds & 2025/26 Quarantine")
    if not (MATCHES_DB.exists() and MARKET_DB.exists()):
        s.skip("All real-data tests", "database missing"); return s

    mk = real_market()
    s.ok("Market dataset loaded", f"n={len(mk)}")

    Pm = mk[["h", "d", "a"]].to_numpy(float)
    validate_probs(Pm, "market closing")
    s.ok("Real market probabilities valid", "finite, [0,1], sum to 1")

    # (leakage 7) quarantine
    s.check(QUARANTINED not in set(mk.season),
            "2025/26 absent from the market dataset (leakage 7)",
            f"seasons={sorted(mk.season.unique())}")

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET))
    v3 = pd.read_sql_query(
        f"""SELECT fixture_id, season, unix, competition_name FROM fixtures
            WHERE competition_id IN ({ph})
              AND status IN ('FT','AWARDED') AND home_goals IS NOT NULL""",
        conn, params=list(TARGET))
    conn.close()

    inter = set(mk.fixture_id) & set(v3.fixture_id)
    eligible = mk[(mk.fixture_id.isin(inter))
                  & (mk.season.isin(E8_TEST_SEASONS))]
    s.check(len(inter) == 4001, "Intersection is 4,001", f"n={len(inter)}")
    s.check(len(eligible) == 3479, "E8 eligible OOS is 3,479",
            f"n={len(eligible)}")

    by_season = eligible.groupby("season").size().to_dict()
    s.check(by_season == {"2023/2024": 1727, "2024/2025": 1752},
            "Season split matches the audit", f"{by_season}")
    s.check(set(eligible.competition_name) ==
            {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"},
            "Exactly the five target leagues")

    # (leakage 4, 5) folds chronological, train strictly before test
    vm = v3[v3.fixture_id.isin(inter)]
    folds = [(("2022/2023",), "2023/2024"),
             (("2022/2023", "2023/2024"), "2024/2025")]
    ok = all(vm[vm.season.isin(trs)].unix.max() < vm[vm.season == ts].unix.min()
             for trs, ts in folds)
    s.check(ok, "Both folds strictly chronological (leakage 4, 5)")
    s.check([ts for _, ts in folds] == list(E8_TEST_SEASONS),
            "Fold structure matches the E2 protocol")

    # temperature applied to real market data stays valid
    for T in TEMPERATURE_GRID:
        validate_probs(apply_temperature(Pm, T), f"real T={T}")
    s.ok("Temperature valid on real market data at every grid point")

    s.ok("Fixture accounting", f"total={len(mk)}, eligible OOS={len(eligible)}, "
                               f"train pool (2022/23)={len(mk) - len(eligible)}")
    return s


# --- Suite 11: no unauthorised components ------------------------------

def suite11() -> Suite:
    s = Suite("11. Scope Discipline")
    src = (HERE / "temperature_scaling.py").read_text(encoding="utf-8")

    # Check what the module actually IMPORTS, not what its prose mentions.
    # A docstring saying "no sklearn" must not trip a naive string scan.
    import ast
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    s.ok("Module imports resolved by AST", f"{sorted(imported)}")

    banned_imports = {"sklearn", "scipy", "lightgbm", "torch", "tensorflow",
                      "models", "features"}
    for b in sorted(banned_imports):
        s.check(b not in imported, f"Does not import '{b}'")

    s.check(imported <= {"numpy", "dataclasses", "__future__"},
            "Imports are numpy + stdlib only", f"{sorted(imported)}")

    # Forbidden CALIBRATION METHODS must not be implemented. Check for
    # callable definitions, not incidental prose.
    defined = {n.name.lower() for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for meth in ("isotonic", "platt", "pava", "sigmoid"):
        s.check(not any(meth in d for d in defined),
                f"No '{meth}' calibrator defined")

    # per-class or per-league temperatures are forbidden
    s.check("class_temperature" not in src and "league_temperature" not in src,
            "No class-specific or league-specific temperature")

    # T must be a scalar
    P = rand_probs(100, 13)
    try:
        apply_temperature(P, np.array([0.9, 1.0, 1.1]))
        s.bad("Reject vector temperature")
    except (TemperatureError, ValueError, TypeError):
        s.ok("Reject vector temperature (single global T only)")
    return s


# --- Suite 12: protected files -----------------------------------------

def suite12() -> Suite:
    s = Suite("12. Protected File Integrity")
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists(): s.skip(Path(rel).name, "not found"); continue
        a = md5(p)
        s.check(a == exp, Path(rel).name,
                f"MD5={a}" if a == exp else f"expected={exp} got={a}")
    return s


def main() -> int:
    print("=" * 72)
    print("E8 — Temperature Scaling Test Suite")
    print("=" * 72)
    su = [suite1(), suite2(), suite3(), suite4(), suite5(), suite6(),
          suite7(), suite8(), suite9(), suite10(), suite11(), suite12()]
    for x in su: x.report()
    tp, tf, ts = (sum(x.p for x in su), sum(x.f for x in su),
                  sum(x.s for x in su))
    print("\n" + "=" * 72 + "\nOVERALL\n" + "=" * 72)
    print(f"  Suites: {len(su)}\n  Pass:   {tp}\n  Fail:   {tf}"
          f"\n  Skip:   {ts}\n  Total:  {tp + tf + ts}")
    print(f"\n  VERDICT: {'ALL TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
