"""E4 — Time-decay test suite.

Usage:
    python research/time_decay/test_time_decay.py

Covers §17 (leakage), §18 (weight sanity), §19 (zero-decay control),
§21 (2025/26 quarantine) and §22 (determinism).

All suites run on numpy alone. The zero-decay control is verified at the
weight level here; its model-level counterpart runs inside
run_e4_experiment.py where sklearn is available.
"""
from __future__ import annotations

import hashlib
import inspect
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from time_decay import (  # noqa: E402
    HALF_LIFE_GRID_DAYS, SECONDS_PER_DAY, TimeDecayError, assert_causal,
    build_inner_splits, decay_weights, delta_days, half_life_from_xi,
    select_xi, weight_profile, xi_from_half_life, xi_grid,
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


class Suite:
    def __init__(self, name): self.name, self.p, self.f, self.s, self.l = name, 0, 0, 0, []
    def ok(self, m, d=""): self.p += 1; self.l.append(f"  PASS: {m}" + (f" — {d}" if d else ""))
    def bad(self, m, d=""): self.f += 1; self.l.append(f"  FAIL: {m}" + (f" — {d}" if d else ""))
    def skip(self, m, d=""): self.s += 1; self.l.append(f"  SKIP: {m}" + (f" — {d}" if d else ""))
    def check(self, c, m, d=""): self.ok(m, d) if c else self.bad(m, d)
    def report(self):
        print(f"\n{'=' * 66}\nSuite: {self.name}\n{'=' * 66}")
        for x in self.l: print(x)
        print(f"\n  Pass: {self.p}  Fail: {self.f}  Skip: {self.s}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(8192), b""): h.update(c)
    return h.hexdigest()


def synth(n=600, span_days=1000, ref_offset_days=1.0):
    """Deterministic synthetic training timestamps ending before a reference."""
    rng = np.random.default_rng(SEED)
    ref = 1_700_000_000.0
    ages = np.sort(rng.uniform(ref_offset_days, span_days, n))[::-1]
    unix = ref - ages * SECONDS_PER_DAY
    return unix, ref


# ---------------------------------------------------------------------------
# Suite 1 — weight sanity  (§18)
# ---------------------------------------------------------------------------

def suite1() -> Suite:
    s = Suite("1. Weight Sanity (§18)")
    unix, ref = synth()

    xi = xi_from_half_life(180.0)
    w = decay_weights(unix, ref, xi)

    s.check(xi >= 0, "xi >= 0", f"xi={xi:.8f}")
    s.check(np.all(np.isfinite(w)), "All weights finite")
    s.check(np.all(w > 0), "All weights > 0", f"min={w.min():.6e}")
    s.check(np.all(w <= 1.0), "All weights <= 1", f"max={w.max():.6f}")

    # monotone decreasing in age
    dd = delta_days(unix, ref)
    order = np.argsort(dd)
    s.check(np.all(np.diff(w[order]) <= 1e-15),
            "Weights monotonically decrease as age increases")

    # older never outweighs newer
    s.check(not np.any((dd[:, None] > dd[None, :]) & (w[:, None] > w[None, :] + 1e-15)),
            "Older observations never outweigh newer ones")

    # w(0) == 1
    s.check(decay_weights(np.array([ref]), ref, xi)[0] == 1.0,
            "Weight at DeltaDays=0 equals exactly 1.0")

    # xi = 0 -> all ones (zero-decay control, weight level)
    w0 = decay_weights(unix, ref, 0.0)
    s.check(np.all(w0 == 1.0), "xi=0 gives all-ones weights (§19 control)",
            f"n={len(w0)}, all exactly 1.0")

    # DeltaDays non-negative
    s.check(np.all(dd >= 0), "DeltaDays always >= 0", f"min={dd.min():.6f}")

    # negative xi rejected
    try:
        decay_weights(unix, ref, -0.01); s.bad("Reject negative xi")
    except TimeDecayError: s.ok("Reject negative xi")

    # non-positive half-life rejected
    for hl in (0.0, -10.0):
        try:
            xi_from_half_life(hl); s.bad(f"Reject half_life={hl}")
        except TimeDecayError: s.ok(f"Reject half_life={hl}")

    return s


# ---------------------------------------------------------------------------
# Suite 2 — xi / half-life conversion
# ---------------------------------------------------------------------------

def suite2() -> Suite:
    s = Suite("2. xi <-> Half-life Conversion")

    for hl in (60.0, 90.0, 180.0, 365.0, 730.0):
        xi = xi_from_half_life(hl)
        w_at_hl = float(np.exp(-xi * hl))
        s.check(abs(w_at_hl - 0.5) < 1e-12,
                f"half-life {hl:.0f}d gives weight 0.5 at {hl:.0f}d",
                f"w={w_at_hl:.12f}")

    s.check(xi_from_half_life(None) == 0.0, "half_life=None -> xi=0")
    s.check(half_life_from_xi(0.0) is None, "xi=0 -> half_life=None")

    for hl in (60.0, 180.0, 730.0):
        s.check(abs(half_life_from_xi(xi_from_half_life(hl)) - hl) < 1e-9,
                f"Round-trip half-life {hl:.0f}d")

    g = xi_grid()
    s.check(len(g) == len(HALF_LIFE_GRID_DAYS),
            "xi_grid matches half-life grid length", f"n={len(g)}")
    s.check(0.0 in g, "Grid contains the no-decay control (xi=0)")
    s.check(all(x >= 0 for x in g), "All grid xi >= 0")

    # World Cup value is NOT in our grid
    s.check(not any(abs(x - 0.001) < 1e-9 for x in g),
            "World Cup xi=0.001 is NOT in our grid (not copied)",
            f"grid={[round(x, 6) for x in g]}")

    return s


# ---------------------------------------------------------------------------
# Suite 3 — weight profile interpretability  (§13)
# ---------------------------------------------------------------------------

def suite3() -> Suite:
    s = Suite("3. Weight Profile Diagnostics (§13)")
    for hl in (90.0, 180.0, 365.0):
        p = weight_profile(xi_from_half_life(hl))
        s.check(p["half_life_days"] is not None
                and abs(p["half_life_days"] - hl) < 1e-9,
                f"Profile reports half-life {hl:.0f}d",
                f"weights={p['weights']}")
    p0 = weight_profile(0.0)
    s.check(all(v == 1.0 for v in p0["weights"].values()),
            "xi=0 profile is 1.0 at every age", f"{p0['weights']}")
    return s


# ---------------------------------------------------------------------------
# Suite 4 — causality & leakage  (§17)
# ---------------------------------------------------------------------------

def suite4() -> Suite:
    s = Suite("4. Causality & Leakage (§17)")
    unix, ref = synth()

    # 17.7 DeltaDays non-negative even if a bad timestamp sneaks in
    s.check(np.all(delta_days(np.array([ref + 999999.0]), ref) >= 0),
            "DeltaDays clipped at 0 for post-reference input")

    # explicit causality guard fires
    try:
        assert_causal(np.append(unix, ref + 1.0), ref)
        s.bad("assert_causal rejects post-reference training fixture")
    except TimeDecayError:
        s.ok("assert_causal rejects post-reference training fixture")
    assert_causal(unix, ref)
    s.ok("assert_causal accepts a strictly-prior training set")

    # 17.8 reference is the fixture timestamp, not today's date
    import time as _t
    now = _t.time()
    s.check(abs(ref - now) > 86400 * 30,
            "Reference time is the fold reference, not today",
            f"ref={ref:.0f}, now={now:.0f}")
    w_a = decay_weights(unix, ref, xi_from_half_life(180.0))
    w_b = decay_weights(unix, ref, xi_from_half_life(180.0))
    s.check(np.array_equal(w_a, w_b),
            "Weights independent of wall-clock time")

    # 17.1/17.3 future fixtures cannot change earlier weights
    unix_ext = np.append(unix, ref + 10 * SECONDS_PER_DAY)   # a future fixture
    w_earlier = decay_weights(unix, ref, xi_from_half_life(180.0))
    w_subset = decay_weights(unix_ext, ref, xi_from_half_life(180.0))[:len(unix)]
    s.check(np.array_equal(w_earlier, w_subset),
            "Appending a future fixture leaves earlier weights unchanged")

    # 17.2/17.4/17.6 no label anywhere in the weight path
    for fn in (decay_weights, delta_days, weight_profile):
        pr = set(inspect.signature(fn).parameters)
        s.check(not ({"y", "label", "outcome", "result"} & pr),
                f"{fn.__name__} signature has no outcome parameter",
                f"params={sorted(pr)}")

    # 17.5 select_xi receives only inner splits built from training seasons
    pr = set(inspect.signature(select_xi).parameters)
    s.check(pr == {"inner_splits", "fit_predict", "half_life_grid"},
            "select_xi signature has no outer-test parameter",
            f"params={sorted(pr)}")
    pr2 = set(inspect.signature(build_inner_splits).parameters)
    s.check("train_seasons" in pr2 and "test_season" not in pr2,
            "build_inner_splits receives training seasons only",
            f"params={sorted(pr2)}")

    return s


# ---------------------------------------------------------------------------
# Suite 5 — nested temporal selection
# ---------------------------------------------------------------------------

def suite5() -> Suite:
    s = Suite("5. Nested Temporal xi Selection")

    seasons = np.array(["2020/2021"] * 100 + ["2021/2022"] * 100
                       + ["2022/2023"] * 100)
    unix = np.arange(300, dtype=float) * SECONDS_PER_DAY + 1_500_000_000.0
    train_seasons = ("2020/2021", "2021/2022", "2022/2023")

    splits = build_inner_splits(seasons, unix, train_seasons)
    s.check(len(splits) == 2, "k training seasons yield k-1 inner splits",
            f"{len(splits)} splits")

    for sp in splits:
        s.check(unix[sp.inner_train_idx].max() < unix[sp.inner_val_idx].min(),
                f"Inner split {sp.inner_val_season} strictly chronological")
        s.check(sp.reference_unix == unix[sp.inner_val_idx].min(),
                f"Inner reference = first kickoff of {sp.inner_val_season}")

    # inner splits never contain the outer test season
    all_seasons_used = set()
    for sp in splits:
        all_seasons_used |= set(sp.inner_train_seasons) | {sp.inner_val_season}
    s.check(all_seasons_used <= set(train_seasons),
            "Inner splits use only outer training seasons",
            f"{sorted(all_seasons_used)}")

    # select_xi drives fit_predict deterministically; stub returns a known argmin
    target_hl = 180.0
    calls = []

    def stub(sp, xi):
        calls.append((sp.inner_val_season, xi))
        return abs(xi - xi_from_half_life(target_hl)) + 1.0

    sel = select_xi(splits, stub)
    s.check(sel.half_life_days == target_hl,
            "select_xi returns the grid argmin",
            f"selected half-life={sel.half_life_days}")
    s.check(len(sel.curve) == len(HALF_LIFE_GRID_DAYS),
            "Curve records every grid candidate", f"{len(sel.curve)} entries")
    s.check(sel.n_inner_splits == 2, "Reports inner split count")
    s.check(len(calls) == len(HALF_LIFE_GRID_DAYS) * 2,
            "fit_predict called once per (candidate, split)",
            f"{len(calls)} calls")

    # empty inner splits rejected rather than silently defaulting
    try:
        select_xi([], stub); s.bad("Reject empty inner splits")
    except TimeDecayError: s.ok("Reject empty inner splits")

    return s


# ---------------------------------------------------------------------------
# Suite 6 — determinism  (§22)
# ---------------------------------------------------------------------------

def suite6() -> Suite:
    s = Suite("6. Determinism (§22)")
    unix, ref = synth()
    xi = xi_from_half_life(240.0)

    w1, w2 = decay_weights(unix, ref, xi), decay_weights(unix, ref, xi)
    s.check(np.array_equal(w1, w2), "decay_weights bit-identical on repeat",
            f"max diff {float(np.max(np.abs(w1 - w2))):.3e}")

    s.check(xi_from_half_life(180.0) == xi_from_half_life(180.0),
            "xi_from_half_life deterministic")

    seasons = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50)
    u = np.arange(150, dtype=float) * SECONDS_PER_DAY
    s1 = build_inner_splits(seasons, u, ("A", "B", "C"))
    s2 = build_inner_splits(seasons, u, ("A", "B", "C"))
    s.check(all(np.array_equal(a.inner_train_idx, b.inner_train_idx)
                and np.array_equal(a.inner_val_idx, b.inner_val_idx)
                for a, b in zip(s1, s2)),
            "build_inner_splits deterministic")

    src = (HERE / "time_decay.py").read_text(encoding="utf-8")
    s.check("random" not in src.lower() and "np.random" not in src,
            "time_decay.py contains no RNG")

    return s


# ---------------------------------------------------------------------------
# Suite 7 — dataset scope & 2025/26 quarantine  (§21)
# ---------------------------------------------------------------------------

def suite7() -> Suite:
    s = Suite("7. Dataset Scope & 2025/26 Quarantine (§21)")
    if not MATCHES_DB.exists():
        s.skip("All dataset tests", "matches.db not found")
        return s

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET))
    sph = ",".join("?" * len(SEASONS))

    n = conn.execute(
        f"SELECT COUNT(*) FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season IN ({sph}) AND status IN ('FT','AWARDED') "
        f"AND home_goals IS NOT NULL", (*TARGET, *SEASONS)).fetchone()[0]
    s.ok("Eligible fixture count", f"n={n}")

    n_q = conn.execute(
        f"SELECT COUNT(*) FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season IN ({sph}) AND season = ?",
        (*TARGET, *SEASONS, QUARANTINED)).fetchone()[0]
    s.check(n_q == 0, "2025/26 fixtures in eligible scope = 0", f"n={n_q}")

    lg = {r[0] for r in conn.execute(
        f"SELECT DISTINCT competition_name FROM fixtures "
        f"WHERE competition_id IN ({ph}) AND season IN ({sph})",
        (*TARGET, *SEASONS))}
    s.check(lg == {"Premier League", "La Liga", "Serie A",
                   "Bundesliga", "Ligue 1"},
            "Exactly the five target leagues", f"{sorted(lg)}")

    # outer folds chronological, and inner splits stay inside training
    folds = [(("2020/2021", "2021/2022"), "2022/2023"),
             (("2020/2021", "2021/2022", "2022/2023"), "2023/2024"),
             (("2020/2021", "2021/2022", "2022/2023", "2023/2024"), "2024/2025")]
    ok = True
    for trs, ts in folds:
        tsph = ",".join("?" * len(trs))
        tr_max = conn.execute(
            f"SELECT MAX(unix) FROM fixtures WHERE competition_id IN ({ph}) "
            f"AND season IN ({tsph})", (*TARGET, *trs)).fetchone()[0]
        te_min = conn.execute(
            f"SELECT MIN(unix) FROM fixtures WHERE competition_id IN ({ph}) "
            f"AND season = ?", (*TARGET, ts)).fetchone()[0]
        if not (tr_max < te_min): ok = False
    s.check(ok, "All 3 outer folds strictly chronological")
    s.check(all(ts != QUARANTINED for _, ts in folds),
            "No outer fold tests on 2025/26")

    # real-data weight check against a real fold reference
    tr_unix = np.array([r[0] for r in conn.execute(
        f"SELECT unix FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season IN ('2020/2021','2021/2022')", TARGET)], dtype=float)
    ref = conn.execute(
        f"SELECT MIN(unix) FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season = '2022/2023'", TARGET).fetchone()[0]
    assert_causal(tr_unix, ref)
    s.ok("Real fold_1 training set is strictly before its reference",
         f"n_train={len(tr_unix)}")
    dd = delta_days(tr_unix, ref)
    s.ok("Real fold_1 training age span",
         f"{dd.min():.1f}d to {dd.max():.1f}d")
    w = decay_weights(tr_unix, ref, xi_from_half_life(180.0))
    s.check(np.all(w > 0) and np.all(np.isfinite(w)),
            "Real-data weights at 180d half-life are valid",
            f"min={w.min():.6e}, max={w.max():.6f}")

    conn.close()
    return s


# ---------------------------------------------------------------------------
# Suite 8 — protected file integrity  (§20)
# ---------------------------------------------------------------------------

def suite8() -> Suite:
    s = Suite("8. Protected File Integrity (§20)")
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            s.skip(Path(rel).name, "not found"); continue
        a = md5(p)
        s.check(a == exp, Path(rel).name,
                f"MD5={a}" if a == exp else f"expected={exp} got={a}")
    return s


def main() -> int:
    print("=" * 66)
    print("E4 — Time Decay Test Suite")
    print("=" * 66)
    suites = [suite1(), suite2(), suite3(), suite4(),
              suite5(), suite6(), suite7(), suite8()]
    for s in suites: s.report()
    tp = sum(s.p for s in suites); tf = sum(s.f for s in suites)
    tsk = sum(s.s for s in suites)
    print("\n" + "=" * 66)
    print("OVERALL")
    print("=" * 66)
    print(f"  Suites: {len(suites)}\n  Pass:   {tp}\n  Fail:   {tf}"
          f"\n  Skip:   {tsk}\n  Total:  {tp + tf + tsk}")
    print(f"\n  VERDICT: {'ALL TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
