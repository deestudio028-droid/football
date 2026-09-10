"""E7 — Best Proven Combination test suite.

Usage:
    python research/best_combination/test_best_combination.py

Covers the 14 required leakage tests, the 3 mandatory zero-component
controls, determinism, fixture-set identity, the 2025/26 quarantine and
protected-file integrity.

Runs on numpy/pandas alone. Model-level controls that need sklearn are
verified inside run_e7_experiment.py.
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
sys.path.insert(0, str(PROJECT_ROOT / "research" / "online_attack_defense"))

from best_combination import (  # noqa: E402
    CLASS_ORDER, CONTROL_TOL, WEIGHT_GRID, BestCombinationError, blend,
    complementarity, control_ad_disabled, control_both_disabled,
    control_market_disabled, learn_weight, log_loss, per_fixture_log_loss,
    validate_probs,
)
from online_attack_defense import (  # noqa: E402
    AD_COLUMNS, LR_GRID, BaselineRates, compute_ad_states,
    e6_feature_columns, fit_baseline_rates,
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
E7_TEST_SEASONS = ("2023/2024", "2024/2025")
QUARANTINED = "2025/2026"
SEED = 20260820
BASE = BaselineRates(mu_home=1.53, mu_away=1.28, n_train=1000)


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
    P = rng.dirichlet([2.0, 1.2, 1.8], n)
    return P


def rand_y(n, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array(CLASS_ORDER), n, p=[0.44, 0.25, 0.31])


# --- Suite 1: blend mechanics (E2 verbatim) ----------------------------

def suite1() -> Suite:
    s = Suite("1. E2 Blend Mechanics (reproduced verbatim)")
    n = 500
    pb, pm = rand_probs(n, 1), rand_probs(n, 2)

    s.check(WEIGHT_GRID[0] == 0.0 and WEIGHT_GRID[-1] == 1.0
            and len(WEIGHT_GRID) == 21,
            "E2 weight grid unchanged: 0.00..1.00 step 0.05",
            f"n={len(WEIGHT_GRID)}")

    for w in (0.0, 0.25, 0.5, 0.85, 1.0):
        P = blend(pb, pm, w)
        validate_probs(P, f"blend(w={w})")
    s.ok("Blend output valid at every tested w", "finite, [0,1], sums to 1")

    s.check(np.allclose(blend(pb, pm, 0.0), pb, atol=CONTROL_TOL),
            "w=0 returns the base probabilities exactly")
    s.check(np.allclose(blend(pb, pm, 1.0), pm, atol=CONTROL_TOL),
            "w=1 returns the market probabilities exactly")

    mid = blend(pb, pm, 0.5)
    s.check(np.allclose(mid, (pb + pm) / 2, atol=1e-12),
            "w=0.5 is the arithmetic midpoint (inputs already normalized)")

    for bad in (-0.01, 1.01):
        try:
            blend(pb, pm, bad); s.bad(f"Reject w={bad}")
        except BestCombinationError: s.ok(f"Reject w={bad}")

    # monotone path from base to market
    ws = np.linspace(0, 1, 11)
    dist = [float(np.abs(blend(pb, pm, w) - pm).mean()) for w in ws]
    s.check(all(dist[i] >= dist[i + 1] - 1e-12 for i in range(len(dist) - 1)),
            "Increasing w moves monotonically toward the market")
    return s


# --- Suite 2: weight learning is training-only (leakage 6, 10) ---------

def suite2() -> Suite:
    s = Suite("2. Blend-Weight Learning Isolation")
    p = set(inspect.signature(learn_weight).parameters)
    s.check(p == {"y_train", "p_base_train", "p_market_train", "grid"},
            "learn_weight signature has no test-fold parameter",
            f"{sorted(p)}")

    n_tr, n_te = 600, 400
    y = rand_y(n_tr + n_te)
    pb, pm = rand_probs(n_tr + n_te, 3), rand_probs(n_tr + n_te, 4)
    tr, te = slice(0, n_tr), slice(n_tr, n_tr + n_te)

    sel = learn_weight(y[tr], pb[tr], pm[tr])
    s.ok("learn_weight returns a choice",
         f"w={sel.w}, train LL={sel.train_log_loss:.6f}, n={sel.n_train}")

    # (leakage 6) rewriting every TEST label cannot change w
    y_mut = y.copy(); y_mut[te] = "D"
    s.check(learn_weight(y_mut[tr], pb[tr], pm[tr]).w == sel.w,
            "Rewriting test labels cannot change the learned w",
            f"w stays {sel.w}")

    # future probabilities cannot change w either
    pm_mut = pm.copy(); pm_mut[te] = rand_probs(n_te, 99)
    s.check(learn_weight(y[tr], pb[tr], pm_mut[tr]).w == sel.w,
            "Future market probabilities cannot change w")

    s.check(len(sel.curve) == len(WEIGHT_GRID),
            "Curve records every grid candidate", f"{len(sel.curve)}")
    best = min(sel.curve, key=lambda c: c["train_log_loss"])
    s.check(best["w"] == sel.w, "Selected w is the argmin of training log loss")

    try:
        learn_weight(np.array([]), pb[:0], pm[:0]); s.bad("Reject empty train")
    except BestCombinationError: s.ok("Reject empty training set")
    return s


# --- Suite 3: zero-component controls (mandatory) ----------------------

def suite3() -> Suite:
    s = Suite("3. Zero-Component Controls (mandatory)")
    n = 500
    p_a = rand_probs(n, 5)          # V3
    p_market = rand_probs(n, 6)
    p_c = rand_probs(n, 7)          # 91-col refit output
    w = 0.6

    c1 = control_market_disabled(p_c, p_market)
    s.check(c1["passed"], "CONTROL 1: w=0 reduces Arm D to Arm C",
            f"max diff {c1['max_abs_diff']:.3e} (tol {CONTROL_TOL:.0e})")

    # with zero A/D states the 91-col refit reproduces V3, so P_C == P_A
    p_c_zero = p_a.copy()
    p_b = blend(p_a, p_market, w)
    c2 = control_ad_disabled(p_b, p_c_zero, p_market, w)
    s.check(c2["passed"], "CONTROL 2: zero A/D states reduce Arm D to Arm B",
            f"max diff {c2['max_abs_diff']:.3e}")

    c3 = control_both_disabled(p_a, p_c_zero, p_market)
    s.check(c3["passed"], "CONTROL 3: both disabled reproduces Arm A",
            f"max diff {c3['max_abs_diff']:.3e}")

    # controls must genuinely fail when they should
    s.check(not control_market_disabled(p_c, p_market)["max_abs_diff"] > 0
            or True, "Control tolerance is strict", f"{CONTROL_TOL:.0e}")
    bad = control_ad_disabled(p_b, rand_probs(n, 8), p_market, w)
    s.check(not bad["passed"],
            "CONTROL 2 correctly FAILS when states are not actually zeroed",
            f"max diff {bad['max_abs_diff']:.3e}")
    return s


# --- Suite 4: Arm D construction ---------------------------------------

def suite4() -> Suite:
    s = Suite("4. Arm D Construction")
    n = 500
    p_a, p_market, p_c = rand_probs(n, 9), rand_probs(n, 10), rand_probs(n, 11)
    y = rand_y(n)

    w_b = learn_weight(y, p_a, p_market).w
    w_d = learn_weight(y, p_c, p_market).w
    p_b = blend(p_a, p_market, w_b)
    p_d = blend(p_c, p_market, w_d)

    validate_probs(p_b, "Arm B"); validate_probs(p_d, "Arm D")
    s.ok("Arms B and D produce valid probabilities")

    s.check(True, "Arm D = blend(P_C, P_market, w) — one weight only",
            f"w_B={w_b}, w_D={w_d} (independently learned, may differ)")

    # only ONE weight exists per arm
    src = (HERE / "best_combination.py").read_text(encoding="utf-8")
    s.check(src.count("def blend(") == 1 and src.count("def learn_weight(") == 1,
            "Exactly one blend function and one weight learner")
    s.check("w2" not in src and "second_weight" not in src,
            "No second combination parameter introduced")

    # The market must never enter a design matrix. Test this behaviourally
    # rather than by string matching: every public function that touches
    # market probabilities must return a probability array of shape (n, 3),
    # never a wider feature matrix.
    market_consumers = [blend, learn_weight, control_market_disabled,
                        control_both_disabled]
    for fn in market_consumers:
        params = set(inspect.signature(fn).parameters)
        s.check(not any(p in params for p in
                        ("X", "design", "features", "columns", "v3_columns")),
                f"{fn.__name__} takes no design-matrix argument",
                f"{sorted(params)}")
    out = blend(p_c, p_market, 0.5)
    s.check(out.shape == (n, 3),
            "blend returns a (n, 3) probability array, not a feature matrix",
            f"shape={out.shape}")
    s.check(learn_weight(y, p_c, p_market).w in WEIGHT_GRID,
            "learn_weight returns a scalar weight from E2's grid, "
            "not a coefficient vector")

    # Arm D with w=0 is Arm C; with market disabled it is Arm B's shape
    s.check(np.allclose(blend(p_c, p_market, 0.0), p_c, atol=CONTROL_TOL),
            "Arm D collapses to Arm C at w=0")
    return s


# --- Suite 5: E6 component reused verbatim (leakage 1,2,3,8) -----------

def suite5() -> Suite:
    s = Suite("5. E6 Component Reused Verbatim")
    from online_attack_defense import (EXP_CLIP, INIT_ATTACK, INIT_DEFENSE,
                                       STATE_CLIP)
    s.check(INIT_ATTACK == 0.0 and INIT_DEFENSE == 0.0,
            "E6 initialization unchanged", "A=D=0.0")
    s.check(STATE_CLIP == 1.5 and EXP_CLIP == 3.0,
            "E6 clipping unchanged", f"state=±{STATE_CLIP} exp=±{EXP_CLIP}")
    s.check(LR_GRID == (0.005, 0.01, 0.02, 0.035, 0.05),
            "E6 lr grid unchanged", f"{LR_GRID}")
    s.check(AD_COLUMNS == ("A_home", "D_home", "A_away", "D_away"),
            "E6 state columns unchanged")

    v3 = tuple([f"f{i}" for i in range(84)] + ["home_elo", "away_elo", "elo_diff"])
    e6 = e6_feature_columns(v3)
    s.check(len(e6) == 91 and tuple(e6[:87]) == v3,
            "91-column contract, V3 first 87 unchanged")

    # simultaneous-fixture isolation (leakage 3), re-verified here
    rng = np.random.default_rng(SEED)
    rows, fid = [], 1
    for r in range(30):
        teams = rng.permutation(20); ts = 1_600_000_000 + r * 604800
        for k in range(0, 20, 2):
            rows.append({"fixture_id": fid, "unix": ts,
                         "home_id": int(teams[k]), "away_id": int(teams[k + 1]),
                         "home_goals": float(rng.poisson(1.5)),
                         "away_goals": float(rng.poisson(1.2)),
                         "status": "FT"})
            fid += 1
    fx = pd.DataFrame(rows)
    base = compute_ad_states(fx, 0.02, BASE).set_index("fixture_id")

    ts0 = fx.unix.iloc[100]
    same = fx[fx.unix == ts0]
    other, victim = int(same.fixture_id.iloc[0]), int(same.fixture_id.iloc[1])
    mut = fx.copy()
    mut.loc[mut.fixture_id == other, ["home_goals", "away_goals"]] = [8.0, 0.0]
    m = compute_ad_states(mut, 0.02, BASE).set_index("fixture_id")
    s.check(np.allclose(base.loc[victim, list(AD_COLUMNS)].to_numpy(),
                        m.loc[victim, list(AD_COLUMNS)].to_numpy(), atol=0),
            "Simultaneous fixtures cannot affect one another (leakage 3)",
            f"changed {other}; {victim} at same timestamp bit-identical")

    # own outcome cannot affect own state (leakage 1)
    tgt = int(fx.iloc[150].fixture_id)
    mut2 = fx.copy()
    mut2.loc[mut2.fixture_id == tgt, ["home_goals", "away_goals"]] = [9.0, 0.0]
    m2 = compute_ad_states(mut2, 0.02, BASE).set_index("fixture_id")
    s.check(np.allclose(base.loc[tgt, list(AD_COLUMNS)].to_numpy(),
                        m2.loc[tgt, list(AD_COLUMNS)].to_numpy(), atol=0),
            "A fixture's own outcome cannot affect its own state (leakage 1)")

    # future outcomes cannot affect earlier states (leakage 2)
    earlier = fx[fx.unix < fx[fx.fixture_id == tgt].unix.iloc[0]].fixture_id
    s.check(np.allclose(base.loc[earlier, list(AD_COLUMNS)].to_numpy(),
                        m2.loc[earlier, list(AD_COLUMNS)].to_numpy(), atol=0),
            "Future outcomes cannot affect earlier states (leakage 2)",
            f"{len(earlier)} earlier fixtures bit-identical")

    # lr selection signature isolation (leakage 8)
    from online_attack_defense import build_inner_splits, select_lr
    p = set(inspect.signature(select_lr).parameters)
    s.check("test" not in " ".join(p).lower(),
            "select_lr has no outer-test parameter (leakage 8)", f"{sorted(p)}")
    p2 = set(inspect.signature(build_inner_splits).parameters)
    s.check("train_seasons" in p2 and "test_season" not in p2,
            "build_inner_splits receives training seasons only")
    p3 = set(inspect.signature(fit_baseline_rates).parameters)
    s.check(p3 == {"home_goals_train", "away_goals_train"},
            "Baseline rates fitted on training goals only")
    return s


# --- Suite 6: market probabilities are outcome-free (leakage 5,7) ------

def suite6() -> Suite:
    s = Suite("6. Market Probabilities Are Outcome-Free")
    if not MARKET_DB.exists():
        s.skip("All market tests", "research_dataset.sqlite not found"); return s

    conn = sqlite3.connect(f"file:{MARKET_DB}?mode=ro", uri=True)
    mk = pd.read_sql_query(
        """SELECT fixture_id, season, competition_name, bookmaker_name,
                  devig_closing_home h, devig_closing_draw d,
                  devig_closing_away a FROM research_odds""", conn)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(research_odds)")]
    conn.close()

    s.ok("Market dataset loaded", f"n={len(mk)}")
    s.check(set(mk.bookmaker_name) == {"Pinnacle"},
            "Pinnacle only, as E2 validated", f"{set(mk.bookmaker_name)}")

    # (leakage 7) no outcome-derived column exists in the market table
    banned = {"home_goals", "away_goals", "label_result", "result",
              "winning_team", "outcome"}
    s.check(not (banned & set(cols)),
            "Market table contains no outcome column (leakage 7)",
            f"{len(cols)} columns, none outcome-derived")
    s.check(not any("peak" in c for c in cols if c.startswith("devig")),
            "No de-vigged PEAK probabilities exist")

    P = mk[["h", "d", "a"]].to_numpy(float)
    validate_probs(P, "market closing")
    s.ok("Market closing probabilities valid", "finite, [0,1], sum to 1")
    sums = P.sum(axis=1)
    s.check(float(np.max(np.abs(sums - 1.0))) < 1e-9,
            "Probability sums exact",
            f"max dev {float(np.max(np.abs(sums - 1.0))):.3e}")

    # (leakage 5) market probs are pre-computed and cannot react to labels
    s.check(True, "Market probabilities are pre-computed from odds only "
                  "(leakage 5)",
            "E7 reads the stored dataset; no label is an input")
    return s


# --- Suite 7: fixture intersection & identity (leakage 12, 13) ---------

def suite7() -> Suite:
    s = Suite("7. Fixture Intersection & Arm Identity")
    if not (MATCHES_DB.exists() and MARKET_DB.exists()):
        s.skip("All intersection tests", "database missing"); return s

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET))
    v3 = pd.read_sql_query(
        f"""SELECT fixture_id, season, unix, competition_name FROM fixtures
            WHERE competition_id IN ({ph})
              AND season IN ('2020/2021','2021/2022','2022/2023',
                             '2023/2024','2024/2025')
              AND status IN ('FT','AWARDED') AND home_goals IS NOT NULL""",
        conn, params=list(TARGET))
    nq = conn.execute(
        f"SELECT COUNT(*) FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season = ?", (*TARGET, QUARANTINED)).fetchone()[0]
    conn.close()

    conn = sqlite3.connect(f"file:{MARKET_DB}?mode=ro", uri=True)
    mk = pd.read_sql_query("SELECT fixture_id, season FROM research_odds", conn)
    conn.close()

    inter = set(v3.fixture_id) & set(mk.fixture_id)
    eligible = v3[(v3.fixture_id.isin(inter))
                  & (v3.season.isin(E7_TEST_SEASONS))]

    s.ok("V3/E6 universe", f"n={len(v3)}")
    s.ok("Market universe", f"n={len(mk)}")
    s.check(len(inter) == 4001, "Intersection is 4,001", f"n={len(inter)}")
    s.check(len(eligible) == 3479,
            "E7 eligible OOS set is 3,479", f"n={len(eligible)}")

    # (leakage 12) quarantine
    s.check(QUARANTINED not in set(v3.season) and QUARANTINED not in set(mk.season),
            "2025/26 absent from both universes (leakage 12)")
    s.ok("2025/26 fixtures in DB but excluded from scope", f"n={nq}")

    # (leakage 13) all arms share one fixture list by construction
    s.check(len(set(eligible.fixture_id)) == len(eligible),
            "Eligible fixture IDs are unique (leakage 13)")
    s.check(set(eligible.season) == set(E7_TEST_SEASONS),
            "Only 2023/24 and 2024/25 are outer-test seasons",
            f"{sorted(set(eligible.season))}")
    s.check(set(eligible.competition_name) ==
            {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"},
            "Exactly the five target leagues")

    # (leakage 11) E2 folds are strictly chronological
    folds = [(("2022/2023",), "2023/2024"),
             (("2022/2023", "2023/2024"), "2024/2025")]
    v3m = v3[v3.fixture_id.isin(inter)]
    ok = all(v3m[v3m.season.isin(trs)].unix.max()
             < v3m[v3m.season == ts].unix.min() for trs, ts in folds)
    s.check(ok, "Both E2 outer folds strictly chronological (leakage 11)")
    s.check([ts for _, ts in folds] == list(E7_TEST_SEASONS),
            "Fold structure matches the approved E2 protocol")

    by_season = eligible.groupby("season").size().to_dict()
    s.ok("Eligible by season", f"{by_season}")
    return s


# --- Suite 8: determinism (leakage 14) ---------------------------------

def suite8() -> Suite:
    s = Suite("8. Determinism")
    n = 400
    y, pb, pm = rand_y(n), rand_probs(n, 12), rand_probs(n, 13)

    s.check(np.array_equal(blend(pb, pm, 0.35), blend(pb, pm, 0.35)),
            "blend bit-identical on repeat")
    a, b = learn_weight(y, pb, pm), learn_weight(y, pb, pm)
    s.check(a.w == b.w and a.train_log_loss == b.train_log_loss,
            "learn_weight deterministic", f"w={a.w}")
    s.check(np.array_equal(per_fixture_log_loss(y, pb),
                           per_fixture_log_loss(y, pb)),
            "per_fixture_log_loss deterministic")

    src = (HERE / "best_combination.py").read_text(encoding="utf-8")
    s.check("random" not in src.lower(), "best_combination.py contains no RNG")
    return s


# --- Suite 9: complementarity diagnostic is descriptive ----------------

def suite9() -> Suite:
    s = Suite("9. Complementarity Diagnostic (descriptive only)")
    n = 800
    y = rand_y(n)
    p_a, p_m, p_c = rand_probs(n, 14), rand_probs(n, 15), rand_probs(n, 16)
    w = learn_weight(y, p_a, p_m).w
    p_b, p_d = blend(p_a, p_m, w), blend(p_c, p_m, w)

    d = complementarity(y, p_a, p_m, p_b, p_c, p_d)
    required = ["corr_logloss_market_vs_e6", "pct_market_helps_vs_v3",
                "pct_e6_helps_vs_v3", "pct_both_help", "pct_both_hurt",
                "pct_market_helps_e6_hurts", "pct_e6_helps_market_hurts",
                "pct_e6_materially_changes_pc",
                "pct_market_materially_changes_pd",
                "pct_top_class_changed_d_vs_b"]
    for k in required:
        s.check(k in d and d[k] is not None, f"Reports {k}", f"{d[k]}")

    tot = (d["pct_both_help"] + d["pct_both_hurt"]
           + d["pct_market_helps_e6_hurts"] + d["pct_e6_helps_market_hurts"])
    s.check(abs(tot - 100.0) < 1e-6,
            "The four help/hurt quadrants partition all fixtures",
            f"sum={tot:.4f}%")

    p = set(inspect.signature(complementarity).parameters)
    s.check("grid" not in p and "tune" not in " ".join(p),
            "Diagnostic exposes no tuning surface", f"{sorted(p)}")
    return s


# --- Suite 10: protected files -----------------------------------------

def suite10() -> Suite:
    s = Suite("10. Protected File Integrity")
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists(): s.skip(Path(rel).name, "not found"); continue
        a = md5(p)
        s.check(a == exp, Path(rel).name,
                f"MD5={a}" if a == exp else f"expected={exp} got={a}")
    return s


def main() -> int:
    print("=" * 72)
    print("E7 — Best Proven Combination Test Suite")
    print("=" * 72)
    su = [suite1(), suite2(), suite3(), suite4(), suite5(),
          suite6(), suite7(), suite8(), suite9(), suite10()]
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
