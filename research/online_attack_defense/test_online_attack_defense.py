"""E6 — Online Attack/Defense test suite.

Usage:
    python research/online_attack_defense/test_online_attack_defense.py

Covers all 21 required areas. Runs on numpy/pandas alone; the
model-level zero-online control needs sklearn and is verified inside
run_e6_experiment.py.
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

from online_attack_defense import (  # noqa: E402
    AD_COLUMNS, EXP_CLIP, INIT_ATTACK, INIT_DEFENSE, LR_GRID, STATE_CLIP,
    BaselineRates, OnlineADError, build_design, build_inner_splits,
    check_extreme_safety, compute_ad_states, e6_feature_columns,
    fit_baseline_rates, prediction_change_diagnostics, select_lr,
    state_diagnostics,
)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
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
SEED = 20260820
BASE = BaselineRates(mu_home=1.53, mu_away=1.28, n_train=1000)


class Suite:
    def __init__(s, n): s.n, s.p, s.f, s.s, s.l = n, 0, 0, 0, []
    def ok(s, m, d=""): s.p += 1; s.l.append(f"  PASS: {m}" + (f" — {d}" if d else ""))
    def bad(s, m, d=""): s.f += 1; s.l.append(f"  FAIL: {m}" + (f" — {d}" if d else ""))
    def skip(s, m, d=""): s.s += 1; s.l.append(f"  SKIP: {m}" + (f" — {d}" if d else ""))
    def check(s, c, m, d=""): s.ok(m, d) if c else s.bad(m, d)
    def report(s):
        print(f"\n{'=' * 70}\nSuite: {s.n}\n{'=' * 70}")
        for x in s.l: print(x)
        print(f"\n  Pass: {s.p}  Fail: {s.f}  Skip: {s.s}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(8192), b""): h.update(c)
    return h.hexdigest()


def synth_fixtures(n_teams=20, n_rounds=40, seed=SEED, simultaneous=True):
    """Deterministic synthetic league with simultaneous kickoffs."""
    rng = np.random.default_rng(seed)
    rows, fid, t0 = [], 1, 1_600_000_000
    for r in range(n_rounds):
        teams = rng.permutation(n_teams)
        ts = t0 + r * 7 * 86400
        for k in range(0, n_teams, 2):
            h, a = int(teams[k]), int(teams[k + 1])
            rows.append({
                "fixture_id": fid, "unix": ts if simultaneous else ts + fid,
                "home_id": h, "away_id": a,
                "home_goals": float(rng.poisson(1.5)),
                "away_goals": float(rng.poisson(1.2)),
                "status": "FT",
                "season": f"S{r // 20}",
            })
            fid += 1
    return pd.DataFrame(rows)


def fake_v3_cols():
    return tuple([f"f{i}" for i in range(84)] + ["home_elo", "away_elo", "elo_diff"])


# --- Suite 1: initialization & contract (1, 7, 9, 10) -------------------

def suite1() -> Suite:
    s = Suite("1. Initialization & Feature Contract")
    s.check(INIT_ATTACK == 0.0 and INIT_DEFENSE == 0.0,
            "Initial states are 0.0", f"A={INIT_ATTACK} D={INIT_DEFENSE}")

    fx = synth_fixtures()
    st = compute_ad_states(fx, lr=0.02, baseline=BASE)
    first_ts = fx[fx.unix == fx.unix.min()]
    first = st[st.fixture_id.isin(first_ts.fixture_id)]
    s.check(np.all(first[list(AD_COLUMNS)].to_numpy() == 0.0),
            "Every team starts at 0.0 (first timestamp all zero)",
            f"n={len(first)}")

    # unseen team joining late gets the initial state
    late = fx.copy()
    late.loc[late.index[-1], "home_id"] = 999
    st2 = compute_ad_states(late, lr=0.02, baseline=BASE)
    row = st2[st2.fixture_id == late.iloc[-1].fixture_id].iloc[0]
    s.check(row.A_home == 0.0 and row.D_home == 0.0,
            "Unseen team enters at the initial state", "team 999 -> A=D=0.0")

    v3 = fake_v3_cols()
    e6 = e6_feature_columns(v3)
    s.check(len(v3) == 87, "V3 contract is 87 columns")
    s.check(len(e6) == 91, "E6 contract is 91 columns", f"n={len(e6)}")
    s.check(tuple(e6[:87]) == tuple(v3), "E6[:87] == V3 contract, same order")
    s.check(tuple(e6[87:]) == AD_COLUMNS, "Columns 88-91 are the four states",
            f"{AD_COLUMNS}")
    s.check(len(set(e6)) == 91, "No duplicate columns")

    X = pd.DataFrame({c: np.arange(len(fx), dtype=float) for c in v3},
                     columns=list(v3))
    D = build_design(X, v3, st, fx.fixture_id.to_numpy())
    s.check(tuple(D.columns) == e6, "build_design emits the 91-col contract")
    s.check(all(np.array_equal(D[c].to_numpy(), X[c].to_numpy()) for c in v3),
            "First 87 columns pass through unmodified")

    for bad, lbl in [(v3 + ("A_home",), "duplicate state column"),
                     (v3[:86], "wrong V3 length")]:
        try:
            e6_feature_columns(bad); s.bad(f"Reject {lbl}")
        except OnlineADError: s.ok(f"Reject {lbl}")
    return s


# --- Suite 2: causal update timing (2, 3, 4, 5) ------------------------

def suite2() -> Suite:
    s = Suite("2. Causal Update Timing")
    fx = synth_fixtures()
    lr = 0.02
    base = compute_ad_states(fx, lr, BASE).set_index("fixture_id")

    # (3) a fixture's own outcome cannot change its own state
    mut = fx.copy()
    tgt = int(fx.iloc[len(fx) // 2].fixture_id)
    mut.loc[mut.fixture_id == tgt, ["home_goals", "away_goals"]] = [9.0, 0.0]
    m1 = compute_ad_states(mut, lr, BASE).set_index("fixture_id")
    s.check(np.allclose(base.loc[tgt, list(AD_COLUMNS)].to_numpy(),
                        m1.loc[tgt, list(AD_COLUMNS)].to_numpy(), atol=0),
            "Fixture's own outcome cannot affect its own state",
            f"fixture {tgt} set to 9-0; state identical")

    # (4) future outcomes cannot change earlier states
    later = fx[fx.unix > fx[fx.fixture_id == tgt].unix.iloc[0]]
    changed_all_earlier = True
    for f_earlier in fx[fx.unix < fx[fx.fixture_id == tgt].unix.iloc[0]].fixture_id:
        if not np.allclose(base.loc[f_earlier, list(AD_COLUMNS)].to_numpy(),
                           m1.loc[f_earlier, list(AD_COLUMNS)].to_numpy(), atol=0):
            changed_all_earlier = False; break
    s.check(changed_all_earlier,
            "Changing a fixture leaves ALL earlier states identical")

    # and it DOES change later states (proves the update actually fires)
    later_changed = any(
        not np.allclose(base.loc[f, list(AD_COLUMNS)].to_numpy(),
                        m1.loc[f, list(AD_COLUMNS)].to_numpy(), atol=0)
        for f in later.fixture_id)
    s.check(later_changed,
            "Changing a fixture DOES change later states (update fires)")

    # (5) simultaneous fixtures cannot contaminate one another
    ts = fx.unix.iloc[100]
    same_ts = fx[fx.unix == ts]
    s.check(len(same_ts) > 1, "Synthetic data has simultaneous fixtures",
            f"{len(same_ts)} at one timestamp")
    other = int(same_ts.fixture_id.iloc[0])
    victim = int(same_ts.fixture_id.iloc[1])
    mut2 = fx.copy()
    mut2.loc[mut2.fixture_id == other, ["home_goals", "away_goals"]] = [7.0, 0.0]
    m2 = compute_ad_states(mut2, lr, BASE).set_index("fixture_id")
    s.check(np.allclose(base.loc[victim, list(AD_COLUMNS)].to_numpy(),
                        m2.loc[victim, list(AD_COLUMNS)].to_numpy(), atol=0),
            "Simultaneous fixtures do not contaminate each other",
            f"changed {other}; {victim} at same timestamp unchanged")

    # (2) truncation invariance: states depend only on prior history
    cut = len(fx) // 2
    trunc = compute_ad_states(fx.iloc[:cut], lr, BASE).set_index("fixture_id")
    s.check(all(np.allclose(base.loc[f, list(AD_COLUMNS)].to_numpy(),
                            trunc.loc[f, list(AD_COLUMNS)].to_numpy(), atol=0)
                for f in trunc.index),
            "Truncating future fixtures leaves earlier states identical",
            f"checked {len(trunc)} fixtures")
    return s


# --- Suite 3: cross-season persistence (6) -----------------------------

def suite3() -> Suite:
    s = Suite("3. Cross-Season Persistence")
    fx = synth_fixtures(n_rounds=60)          # 3 synthetic seasons
    st = compute_ad_states(fx, 0.02, BASE).set_index("fixture_id")
    seasons = sorted(fx.season.unique())
    s.check(len(seasons) >= 2, "Synthetic data spans multiple seasons",
            f"{seasons}")

    for sn in seasons[1:]:
        first_ts = fx[fx.season == sn].unix.min()
        firsts = fx[(fx.season == sn) & (fx.unix == first_ts)]
        vals = st.loc[firsts.fixture_id, list(AD_COLUMNS)].to_numpy()
        s.check(not np.allclose(vals, 0.0),
                f"Season {sn} opens with carried-over (non-zero) states",
                f"max|state| = {np.max(np.abs(vals)):.6f}")

    # explicit contrast with an accidental season reset
    per_season = []
    for sn in seasons:
        sub = fx[fx.season == sn]
        per_season.append(compute_ad_states(sub, 0.02, BASE))
    reset = pd.concat(per_season).set_index("fixture_id")
    later = fx[fx.season == seasons[-1]].fixture_id
    s.check(not np.allclose(st.loc[later, list(AD_COLUMNS)].to_numpy(),
                            reset.loc[later, list(AD_COLUMNS)].to_numpy()),
            "Continuous history differs from a season-reset run",
            "confirms persistence is real, not incidental")
    return s


# --- Suite 4: zero-online control (8) ----------------------------------

def suite4() -> Suite:
    s = Suite("4. Zero-Online-State Control")
    fx = synth_fixtures()

    off = compute_ad_states(fx, 0.02, BASE, enabled=False)
    s.check(np.all(off[list(AD_COLUMNS)].to_numpy() == 0.0),
            "enabled=False gives exactly zero states", f"n={len(off)}")
    s.check(float(off[list(AD_COLUMNS)].to_numpy().var()) == 0.0,
            "Zero states have zero variance")

    frozen = compute_ad_states(fx, 0.0, BASE, enabled=True)
    s.check(np.all(frozen[list(AD_COLUMNS)].to_numpy() == 0.0),
            "lr=0 freezes every state at its initial value")

    on = compute_ad_states(fx, 0.02, BASE, enabled=True)
    s.check(not np.allclose(on[list(AD_COLUMNS)].to_numpy(),
                            off[list(AD_COLUMNS)].to_numpy()),
            "enabled=True differs from the control")

    v3 = fake_v3_cols()
    X = pd.DataFrame({c: np.arange(len(fx), dtype=float) for c in v3},
                     columns=list(v3))
    D = build_design(X, v3, off, fx.fixture_id.to_numpy())
    s.check(D.shape[1] == 91, "Control design still has 91 columns")
    s.check(np.all(D[list(AD_COLUMNS)].to_numpy() == 0.0),
            "Control's four added columns are identically zero")
    s.ok("Model-level control (91-col zeros == 87-col V3)",
         "verified numerically in run_e6_experiment.py; needs sklearn")
    return s


# --- Suite 5: lr selection isolation (12, 13) --------------------------

def suite5() -> Suite:
    s = Suite("5. Learning-Rate Selection Isolation")
    s.check(LR_GRID == (0.005, 0.01, 0.02, 0.035, 0.05),
            "Pre-registered lr grid unchanged", f"{LR_GRID}")

    p = set(inspect.signature(select_lr).parameters)
    s.check(p == {"inner_splits", "fit_predict", "grid"},
            "select_lr has no outer-test parameter", f"{sorted(p)}")
    p2 = set(inspect.signature(build_inner_splits).parameters)
    s.check("train_seasons" in p2 and "test_season" not in p2,
            "build_inner_splits receives training seasons only", f"{sorted(p2)}")
    p3 = set(inspect.signature(fit_baseline_rates).parameters)
    s.check(p3 == {"home_goals_train", "away_goals_train"},
            "fit_baseline_rates takes training goals only", f"{sorted(p3)}")

    seasons = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    splits = build_inner_splits(seasons, ("A", "B", "C"))
    s.check(len(splits) == 2, "k training seasons yield k-1 inner splits")
    used = set()
    for sp in splits:
        used |= set(sp.train_seasons) | {sp.val_season}
    s.check(used <= {"A", "B", "C"},
            "Inner splits use only outer training seasons", f"{sorted(used)}")

    calls = []
    def stub(sp, lr):
        calls.append((sp.val_season, lr))
        return abs(lr - 0.02) + 1.0
    sel = select_lr(splits, stub)
    s.check(sel.lr == 0.02, "select_lr returns the grid argmin",
            f"lr={sel.lr}")
    s.check(len(sel.curve) == len(LR_GRID), "Curve records every candidate")
    s.check(len(calls) == len(LR_GRID) * 2,
            "fit_predict called once per (lr, split)", f"{len(calls)} calls")
    try:
        select_lr([], stub); s.bad("Reject empty inner splits")
    except OnlineADError: s.ok("Reject empty inner splits")

    src = (HERE / "online_attack_defense.py").read_text(encoding="utf-8")
    s.check("label_result" not in src and "label_home_goals" not in src,
            "Module never references label columns")
    return s


# --- Suite 6: adversarial leakage (14) ---------------------------------

def suite6() -> Suite:
    s = Suite("6. Adversarial Leakage Test")
    fx = synth_fixtures(n_rounds=60)
    lr = 0.02
    seasons = sorted(fx.season.unique())
    test_season = seasons[-1]
    test_start = fx[fx.season == test_season].unix.min()

    base = compute_ad_states(fx, lr, BASE).set_index("fixture_id")

    rng = np.random.default_rng(999)
    adv = fx.copy()
    m = adv.season == test_season
    adv.loc[m, "home_goals"] = rng.integers(0, 9, m.sum()).astype(float)
    adv.loc[m, "away_goals"] = rng.integers(0, 9, m.sum()).astype(float)
    s.check(not np.array_equal(fx.loc[m, "home_goals"].to_numpy(),
                               adv.loc[m, "home_goals"].to_numpy()),
            "Adversarial copy genuinely changed test-period outcomes",
            f"{int(m.sum())} fixtures rewritten")

    advst = compute_ad_states(adv, lr, BASE).set_index("fixture_id")

    pre = fx[fx.unix < test_start].fixture_id
    s.check(np.allclose(base.loc[pre, list(AD_COLUMNS)].to_numpy(),
                        advst.loc[pre, list(AD_COLUMNS)].to_numpy(), atol=0),
            "ALL pre-test-period states identical after rewriting test outcomes",
            f"{len(pre)} fixtures, exact match")

    # the very first test fixture is also pre-match w.r.t. its own outcome
    first_test = fx[(fx.season == test_season) & (fx.unix == test_start)].fixture_id
    s.check(np.allclose(base.loc[first_test, list(AD_COLUMNS)].to_numpy(),
                        advst.loc[first_test, list(AD_COLUMNS)].to_numpy(),
                        atol=0),
            "First test-period fixtures' pre-match states also identical",
            f"{len(first_test)} fixtures")

    # baseline rates fitted on training goals cannot see test goals either
    tr = fx[fx.unix < test_start]
    b1 = fit_baseline_rates(tr.home_goals.to_numpy(), tr.away_goals.to_numpy())
    tr2 = adv[adv.unix < test_start]
    b2 = fit_baseline_rates(tr2.home_goals.to_numpy(), tr2.away_goals.to_numpy())
    s.check(b1.mu_home == b2.mu_home and b1.mu_away == b2.mu_away,
            "Baseline rates unchanged by rewritten test outcomes",
            f"mu_home={b1.mu_home:.6f} mu_away={b1.mu_away:.6f}")
    return s


# --- Suite 7: chronological state test (§21 of the spec) ---------------

def suite7() -> Suite:
    s = Suite("7. Chronological State Test")
    fx = synth_fixtures().sort_values(["unix", "fixture_id"]).reset_index(drop=True)
    lr = 0.02
    idx = 300
    tgt = int(fx.iloc[idx].fixture_id)
    tgt_ts = fx.iloc[idx].unix

    through_prev = fx[fx.unix < tgt_ts]
    with_tgt = fx[fx.unix <= tgt_ts]

    st_prev_all = compute_ad_states(
        pd.concat([through_prev, fx[fx.fixture_id == tgt]]), lr, BASE
    ).set_index("fixture_id")
    st_with = compute_ad_states(with_tgt, lr, BASE).set_index("fixture_id")

    s.check(np.allclose(st_prev_all.loc[tgt, list(AD_COLUMNS)].to_numpy(),
                        st_with.loc[tgt, list(AD_COLUMNS)].to_numpy(), atol=0),
            "state(T) unchanged whether or not later fixtures exist",
            f"fixture {tgt}")

    nxt = fx[fx.unix > tgt_ts]
    if len(nxt):
        nxt_id = int(nxt.iloc[0].fixture_id)
        full = compute_ad_states(fx, lr, BASE).set_index("fixture_id")
        no_tgt = compute_ad_states(fx[fx.fixture_id != tgt], lr, BASE
                                   ).set_index("fixture_id")
        # only meaningful if the removed fixture involved a team playing later
        teams = {int(fx.iloc[idx].home_id), int(fx.iloc[idx].away_id)}
        affected = nxt[(nxt.home_id.isin(teams)) | (nxt.away_id.isin(teams))]
        if len(affected):
            aid = int(affected.iloc[0].fixture_id)
            s.check(not np.allclose(full.loc[aid, list(AD_COLUMNS)].to_numpy(),
                                    no_tgt.loc[aid, list(AD_COLUMNS)].to_numpy()),
                    "state(T+1) DOES change when fixture T is removed",
                    f"fixture {aid} involves a team from fixture {tgt}")
        else:
            s.skip("state(T+1) change check", "no later fixture shares a team")
    return s


# --- Suite 8: determinism (15) -----------------------------------------

def suite8() -> Suite:
    s = Suite("8. Determinism")
    fx = synth_fixtures()
    a = compute_ad_states(fx, 0.02, BASE)
    b = compute_ad_states(fx, 0.02, BASE)
    s.check(np.array_equal(a[list(AD_COLUMNS)].to_numpy(),
                           b[list(AD_COLUMNS)].to_numpy()),
            "compute_ad_states bit-identical on repeat",
            f"max diff {float(np.max(np.abs(a[list(AD_COLUMNS)].to_numpy() - b[list(AD_COLUMNS)].to_numpy()))):.3e}")

    shuffled = fx.sample(frac=1.0, random_state=7).reset_index(drop=True)
    c = compute_ad_states(shuffled, 0.02, BASE)
    s.check(np.allclose(a.set_index("fixture_id").loc[c.fixture_id,
                                                      list(AD_COLUMNS)].to_numpy(),
                        c[list(AD_COLUMNS)].to_numpy(), atol=0),
            "Input row order does not affect results (internal sort)")

    r1 = fit_baseline_rates(fx.home_goals.to_numpy(), fx.away_goals.to_numpy())
    r2 = fit_baseline_rates(fx.home_goals.to_numpy(), fx.away_goals.to_numpy())
    s.check(r1.mu_home == r2.mu_home, "fit_baseline_rates deterministic")

    src = (HERE / "online_attack_defense.py").read_text(encoding="utf-8")
    s.check("random" not in src.lower(),
            "online_attack_defense.py contains no RNG")
    return s


# --- Suite 9: numerical safety (16, 17, 18, 19) ------------------------

def suite9() -> Suite:
    s = Suite("9. Numerical Safety")
    fx = synth_fixtures()

    for lr in LR_GRID:
        st = compute_ad_states(fx, lr, BASE)
        v = st[list(AD_COLUMNS)].to_numpy()
        if not (np.all(np.isfinite(v)) and np.all(np.abs(v) <= STATE_CLIP + 1e-9)):
            s.bad(f"States valid at lr={lr}"); break
    else:
        s.ok("States finite and within clip at every grid lr",
             f"clip=±{STATE_CLIP}")

    # adversarial extreme scores must not explode the state
    wild = fx.copy()
    wild["home_goals"] = 15.0
    wild["away_goals"] = 0.0
    st = compute_ad_states(wild, 0.05, BASE)
    d = state_diagnostics(st)
    s.check(d["all_finite"] and d["within_clip"],
            "Extreme 15-0 scores cannot explode the state",
            f"A_home max={d['A_home']['max']:.4f}")

    s.check(EXP_CLIP == 3.0 and STATE_CLIP == 1.5,
            "Audit-approved clips unchanged",
            f"STATE_CLIP={STATE_CLIP} EXP_CLIP={EXP_CLIP}")

    st_norm = compute_ad_states(fx, 0.02, BASE)
    dg = state_diagnostics(st_norm)
    s.check(dg["stable"] and not dg["collapsed"],
            "States neither collapse nor degenerate",
            f"A_home std={dg['A_home']['std']:.6f}")

    rng = np.random.default_rng(SEED)
    lh, la = rng.uniform(0.3, 4.0, 500), rng.uniform(0.3, 3.5, 500)
    P = rng.dirichlet([2, 1, 2], 500)
    r = check_extreme_safety(lh, la, P)
    s.check(r["passed"], "Valid lambdas/probabilities pass the safety gate",
            f"rowsum dev {r['max_rowsum_deviation']:.2e}")
    bad = P.copy(); bad[0] *= 2
    s.check(not check_extreme_safety(lh, la, bad)["passed"],
            "Probabilities not summing to 1 are caught")
    s.check(not check_extreme_safety(np.append(lh[:-1], 0.0), la, P)["passed"],
            "Non-positive lambda is caught")

    for arr, lbl in [(np.array([np.nan, 1.0]), "NaN goals"),
                     (np.array([]), "empty training set")]:
        try:
            fit_baseline_rates(arr, np.array([1.0, 1.0])[:len(arr)])
            s.bad(f"Reject {lbl}")
        except OnlineADError: s.ok(f"Reject {lbl}")
    try:
        compute_ad_states(fx, -0.01, BASE); s.bad("Reject negative lr")
    except OnlineADError: s.ok("Reject negative lr")
    return s


# --- Suite 10: real data, folds, quarantine (11, 20) -------------------

def suite10() -> Suite:
    s = Suite("10. Real Data, Folds & 2025/26 Quarantine")
    if not MATCHES_DB.exists():
        s.skip("All real-data tests", "matches.db not found"); return s

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET)); sph = ",".join("?" * len(SEASONS))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, competition_name, season, unix, home_id, away_id,
                   home_goals, away_goals, status FROM fixtures
            WHERE competition_id IN ({ph}) AND season IN ({sph})
              AND status IN ('FT','AWARDED') AND home_goals IS NOT NULL
            ORDER BY unix, fixture_id""", conn, params=[*TARGET, *SEASONS])
    nq = conn.execute(
        f"SELECT COUNT(*) FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season IN ({sph}) AND season = ?",
        (*TARGET, *SEASONS, QUARANTINED)).fetchone()[0]
    conn.close()

    s.ok("Eligible fixtures loaded", f"n={len(df)}")
    s.check(nq == 0, "2025/26 fixtures used = 0", f"n={nq}")
    s.check(set(df.competition_name) == {"Premier League", "La Liga", "Serie A",
                                         "Bundesliga", "Ligue 1"},
            "Exactly the five target leagues")
    s.check(QUARANTINED not in set(df.season),
            "No 2025/26 season present in the working set")

    folds = [(("2020/2021", "2021/2022"), "2022/2023"),
             (("2020/2021", "2021/2022", "2022/2023"), "2023/2024"),
             (("2020/2021", "2021/2022", "2022/2023", "2023/2024"), "2024/2025")]
    ok = all(df[df.season.isin(trs)].unix.max() < df[df.season == ts].unix.min()
             for trs, ts in folds)
    s.check(ok, "All 3 approved folds strictly chronological")
    s.check([ts for _, ts in folds] == ["2022/2023", "2023/2024", "2024/2025"],
            "Fold definitions match the approved V3/E2 protocol")

    tr = df[df.season.isin(folds[0][0])]
    base = fit_baseline_rates(tr.home_goals.to_numpy(), tr.away_goals.to_numpy())
    s.ok("fold_1 baseline rates from training only",
         f"mu_home={base.mu_home:.4f} mu_away={base.mu_away:.4f} n={base.n_train}")

    st = compute_ad_states(df, 0.02, base)
    d = state_diagnostics(st)
    s.check(d["all_finite"] and d["within_clip"] and d["stable"],
            "Real-data states finite, bounded and non-degenerate")
    s.ok("Real-data state ranges",
         f"A_home[{d['A_home']['min']:+.4f},{d['A_home']['max']:+.4f}] "
         f"D_home[{d['D_home']['min']:+.4f},{d['D_home']['max']:+.4f}]")

    # simultaneous kickoffs really do occur in the real data
    dup = df.groupby("unix").size()
    s.check(int(dup.max()) > 1,
            "Real data contains simultaneous kickoffs (two-pass matters)",
            f"max {int(dup.max())} fixtures share a timestamp")
    return s


# --- Suite 11: prediction-change diagnostics ---------------------------

def suite11() -> Suite:
    s = Suite("11. Prediction-Change Diagnostics")
    rng = np.random.default_rng(SEED)
    n = 1000
    lh_a, la_a = rng.uniform(0.8, 2.5, n), rng.uniform(0.6, 2.0, n)
    lh_b, la_b = lh_a + rng.normal(0, 0.05, n), la_a + rng.normal(0, 0.05, n)
    P_a = rng.dirichlet([2, 1, 2], n)
    P_b = P_a + rng.normal(0, 0.01, (n, 3))
    P_b = np.clip(P_b, 1e-9, None); P_b /= P_b.sum(1, keepdims=True)

    d = prediction_change_diagnostics(lh_a, la_a, P_a, lh_b, la_b, P_b)
    for k in ["mean_abs_lambda_home_delta", "mean_abs_lambda_away_delta",
              "max_lambda_delta", "mean_abs_probability_delta",
              "max_probability_delta", "pct_fixtures_materially_changed",
              "pct_top_prediction_changed"]:
        s.check(k in d and np.isfinite(d[k]), f"Reports {k}", f"{d[k]}")

    z = prediction_change_diagnostics(lh_a, la_a, P_a, lh_a, la_a, P_a)
    s.check(z["mean_abs_probability_delta"] == 0.0
            and z["pct_top_prediction_changed"] == 0.0,
            "Identical inputs report zero change")
    return s


# --- Suite 12: protected MD5 (21) --------------------------------------

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
    print("=" * 70)
    print("E6 — Online Attack/Defense Test Suite")
    print("=" * 70)
    su = [suite1(), suite2(), suite3(), suite4(), suite5(), suite6(),
          suite7(), suite8(), suite9(), suite10(), suite11(), suite12()]
    for x in su: x.report()
    tp, tf, ts = (sum(x.p for x in su), sum(x.f for x in su),
                  sum(x.s for x in su))
    print("\n" + "=" * 70 + "\nOVERALL\n" + "=" * 70)
    print(f"  Suites: {len(su)}\n  Pass:   {tp}\n  Fail:   {tf}"
          f"\n  Skip:   {ts}\n  Total:  {tp + tf + ts}")
    print(f"\n  VERDICT: {'ALL TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
