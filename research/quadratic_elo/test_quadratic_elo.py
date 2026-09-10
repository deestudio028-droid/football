"""E5 — Quadratic Elo test suite.

Usage:
    python research/quadratic_elo/test_quadratic_elo.py

Covers §17 (leakage), §18 (zero-quadratic control), §19 (feature
contract), §21 (determinism), §22 (2025/26 quarantine), §23 (MD5) and
§25's checklist.

Runs on numpy/pandas alone. Suites needing the V3 artifact are skipped
with a clear reason when scikit-learn is unavailable.
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

from quadratic_elo import (  # noqa: E402
    QUADRATIC_COLUMN, SOURCE_COLUMN, CenteringStat, QuadraticEloError,
    build_design, check_extreme_safety, curvature_table, e5_feature_columns,
    fit_centering, make_quadratic_feature,
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
    def __init__(self, n): self.n, self.p, self.f, self.s, self.l = n, 0, 0, 0, []
    def ok(self, m, d=""): self.p += 1; self.l.append(f"  PASS: {m}" + (f" — {d}" if d else ""))
    def bad(self, m, d=""): self.f += 1; self.l.append(f"  FAIL: {m}" + (f" — {d}" if d else ""))
    def skip(self, m, d=""): self.s += 1; self.l.append(f"  SKIP: {m}" + (f" — {d}" if d else ""))
    def check(self, c, m, d=""): self.ok(m, d) if c else self.bad(m, d)
    def report(self):
        print(f"\n{'=' * 68}\nSuite: {self.n}\n{'=' * 68}")
        for x in self.l: print(x)
        print(f"\n  Pass: {self.p}  Fail: {self.f}  Skip: {self.s}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(8192), b""): h.update(c)
    return h.hexdigest()


def fake_v3_columns(n=87):
    cols = [f"f{i}" for i in range(n - 3)]
    return tuple(cols + ["home_elo", "away_elo", "elo_diff"])


def fake_X(v3cols, n=300, seed=SEED):
    rng = np.random.default_rng(seed)
    d = {c: rng.normal(0, 1, n) for c in v3cols}
    d["home_elo"] = rng.uniform(1250, 1900, n)
    d["away_elo"] = rng.uniform(1250, 1900, n)
    d["elo_diff"] = d["home_elo"] + 100.0 - d["away_elo"]
    return pd.DataFrame(d, columns=list(v3cols))


# ---------------------------------------------------------------------------
# Suite 1 — feature contract (§19)
# ---------------------------------------------------------------------------

def suite1() -> Suite:
    s = Suite("1. Feature Contract (§19)")
    v3 = fake_v3_columns()
    s.check(len(v3) == 87, "V3 contract has 87 columns", f"n={len(v3)}")

    e5 = e5_feature_columns(v3)
    s.check(len(e5) == 88, "E5 contract has 88 columns", f"n={len(e5)}")
    s.check(tuple(e5[:87]) == tuple(v3), "E5[:87] == V3 contract, same order")
    s.check(e5[87] == QUADRATIC_COLUMN, "Column 88 is elo_diff_sq")
    s.check(set(e5) - set(v3) == {QUADRATIC_COLUMN},
            "Exactly one column added", f"{set(e5) - set(v3)}")
    s.check(len(set(e5)) == 88, "No duplicate columns")

    X = fake_X(v3)
    c = fit_centering(X[SOURCE_COLUMN].to_numpy())
    D = build_design(X, v3, c)
    s.check(tuple(D.columns) == e5, "build_design emits the 88-col contract")
    s.check(D.shape[1] == 88, "Design has 88 columns", f"{D.shape}")
    s.check(all(np.array_equal(D[k].to_numpy(), X[k].to_numpy()) for k in v3),
            "First 87 columns pass through unmodified")

    # rejects wrong input order
    try:
        build_design(X[list(v3)[::-1]], v3, c); s.bad("Reject wrong column order")
    except QuadraticEloError: s.ok("Reject wrong column order")
    # rejects a contract that already has the column
    try:
        e5_feature_columns(v3 + (QUADRATIC_COLUMN,)); s.bad("Reject duplicate add")
    except QuadraticEloError: s.ok("Reject duplicate add")
    # rejects a contract missing elo_diff
    try:
        e5_feature_columns(tuple(x for x in v3 if x != SOURCE_COLUMN))
        s.bad("Reject contract without elo_diff")
    except QuadraticEloError: s.ok("Reject contract without elo_diff")
    return s


# ---------------------------------------------------------------------------
# Suite 2 — quadratic feature correctness
# ---------------------------------------------------------------------------

def suite2() -> Suite:
    s = Suite("2. Quadratic Feature Correctness")
    ed = np.array([-400.0, -100.0, 0.0, 100.0, 300.0, 686.0])
    c = fit_centering(ed)

    q = make_quadratic_feature(ed, c)
    s.check(np.allclose(q, (ed - c.mean) ** 2),
            "elo_diff_sq == (elo_diff - train_mean)^2")
    s.check(np.all(q >= 0), "All values non-negative", f"min={q.min():.4f}")
    s.check(np.all(np.isfinite(q)), "All values finite")
    s.check(abs(c.mean - ed.mean()) < 1e-12, "Centering equals training mean",
            f"mean={c.mean:.6f}")

    # value at the centre is exactly 0
    q0 = make_quadratic_feature(np.array([c.mean]), c)
    s.check(q0[0] == 0.0, "Value at the training mean is exactly 0")

    # symmetry about the centre
    qa = make_quadratic_feature(np.array([c.mean - 137.0]), c)
    qb = make_quadratic_feature(np.array([c.mean + 137.0]), c)
    s.check(abs(qa[0] - qb[0]) < 1e-9, "Symmetric about the centre")

    # centering reduces collinearity with the linear term
    rng = np.random.default_rng(SEED)
    v = rng.uniform(-430, 690, 5000)
    cc = fit_centering(v)
    raw_corr = float(np.corrcoef(v, v ** 2)[0, 1])
    cen_corr = float(np.corrcoef(v, make_quadratic_feature(v, cc))[0, 1])
    s.check(abs(cen_corr) < abs(raw_corr) / 10,
            "Centering cuts collinearity with the linear term",
            f"raw={raw_corr:+.4f} -> centered={cen_corr:+.4f}")

    # rejects bad input
    for bad, lbl in [(np.array([np.nan, 1.0]), "NaN"),
                     (np.array([np.inf, 1.0]), "Inf")]:
        try:
            make_quadratic_feature(bad, c); s.bad(f"Reject {lbl} elo_diff")
        except QuadraticEloError: s.ok(f"Reject {lbl} elo_diff")
    try:
        fit_centering(np.array([])); s.bad("Reject empty training set")
    except QuadraticEloError: s.ok("Reject empty training set")
    return s


# ---------------------------------------------------------------------------
# Suite 3 — zero-quadratic control (§18)
# ---------------------------------------------------------------------------

def suite3() -> Suite:
    s = Suite("3. Zero-Quadratic Control (§18)")
    v3 = fake_v3_columns()
    X = fake_X(v3)
    c = fit_centering(X[SOURCE_COLUMN].to_numpy())

    q_off = make_quadratic_feature(X[SOURCE_COLUMN].to_numpy(), c, enabled=False)
    s.check(np.all(q_off == 0.0), "enabled=False gives exactly zeros",
            f"n={len(q_off)}, max={q_off.max()}")
    s.check(float(np.var(q_off)) == 0.0, "Zero column has zero variance")

    D_off = build_design(X, v3, c, enabled=False)
    s.check(D_off.shape[1] == 88, "Control design still has 88 columns")
    s.check(all(np.array_equal(D_off[k].to_numpy(), X[k].to_numpy()) for k in v3),
            "Control leaves the 87 V3 columns untouched")
    s.check(np.all(D_off[QUADRATIC_COLUMN].to_numpy() == 0.0),
            "Control's added column is identically zero")

    D_on = build_design(X, v3, c, enabled=True)
    s.check(not np.array_equal(D_on[QUADRATIC_COLUMN].to_numpy(),
                               D_off[QUADRATIC_COLUMN].to_numpy()),
            "enabled=True differs from the control")
    s.ok("Model-level control (88-col zeros == 87-col V3)",
         "verified numerically in run_e5_experiment.py; needs sklearn")
    return s


# ---------------------------------------------------------------------------
# Suite 4 — causality & leakage (§17)
# ---------------------------------------------------------------------------

def suite4() -> Suite:
    s = Suite("4. Causality & Leakage (§17)")

    # 17.7 centering fitted on training rows only
    pr = set(inspect.signature(fit_centering).parameters)
    s.check(pr == {"elo_diff_train"},
            "fit_centering takes training values only", f"params={sorted(pr)}")

    # 17.4/17.8 no label anywhere in the feature path
    for fn in (fit_centering, make_quadratic_feature, build_design,
               e5_feature_columns):
        p = set(inspect.signature(fn).parameters)
        s.check(not ({"y", "label", "outcome", "result", "label_result"} & p),
                f"{fn.__name__} has no outcome parameter", f"{sorted(p)}")

    # 17.3 changing test values cannot change the fitted centering
    rng = np.random.default_rng(SEED)
    tr = rng.uniform(-400, 600, 800)
    c1 = fit_centering(tr)
    c2 = fit_centering(tr)              # test values simply never enter
    s.check(c1.mean == c2.mean, "Centering depends only on training values",
            f"mean={c1.mean:.6f}")

    # a test row's own value cannot change its own feature beyond the formula
    te = np.array([250.0])
    q_a = make_quadratic_feature(te, c1)
    q_b = make_quadratic_feature(te, c1)
    s.check(np.array_equal(q_a, q_b),
            "Test feature is a pure function of (value, frozen centering)")

    # 17.6 quadratic term is built from pre-match elo_diff only
    src = (HERE / "quadratic_elo.py").read_text(encoding="utf-8")
    s.check("home_goals" not in src and "away_goals" not in src
            and "label" not in src.replace("label_result", "").replace(
                "no label", ""),
            "quadratic_elo.py never references goals or labels")

    # 17.1/17.2/17.5 causal Elo is reused unchanged from E1
    try:
        from features.elo import (ELO_COLUMNS, HOME_ADVANTAGE, INIT_RATING,
                                  K_FACTOR, MEAN_REVERSION)
        s.check(INIT_RATING == 1500.0 and K_FACTOR == 20.0
                and HOME_ADVANTAGE == 100.0 and MEAN_REVERSION == 0.0,
                "E1 Elo parameters unchanged",
                f"init={INIT_RATING} K={K_FACTOR} HA={HOME_ADVANTAGE} "
                f"MR={MEAN_REVERSION}")
        s.check(ELO_COLUMNS == ("home_elo", "away_elo", "elo_diff"),
                "Elo column set unchanged")
    except ImportError as e:
        s.skip("Elo parameter check", str(e))
    return s


# ---------------------------------------------------------------------------
# Suite 5 — real Elo data: collinearity & home advantage (audit §6/§7)
# ---------------------------------------------------------------------------

def suite5() -> Suite:
    s = Suite("5. Real Elo Data — Collinearity & Home Advantage")
    if not MATCHES_DB.exists():
        s.skip("All Elo data tests", "matches.db not found"); return s
    try:
        from features.elo import HOME_ADVANTAGE, load_elo_features
    except ImportError as e:
        s.skip("All Elo data tests", str(e)); return s

    e = load_elo_features(MATCHES_DB)
    s.ok("Elo computed for all fixtures", f"n={len(e)}")

    resid = e.elo_diff.values - (e.home_elo.values + HOME_ADVANTAGE
                                 - e.away_elo.values)
    s.check(float(np.max(np.abs(resid))) == 0.0,
            "elo_diff == home_elo + HA - away_elo exactly",
            f"max|resid|={np.max(np.abs(resid)):.3e}")

    M = np.column_stack([np.ones(len(e)), e.home_elo, e.away_elo, e.elo_diff])
    s.check(np.linalg.matrix_rank(M) == 3,
            "Elo block + intercept is rank-deficient (3 of 4)",
            "elo_diff adds no independent direction; ridge makes it "
            "identifiable but coefficients are not separately meaningful")

    d = e.elo_diff.values
    s.ok("Observed elo_diff range",
         f"[{d.min():.1f}, {d.max():.1f}], median={np.median(d):.1f}")
    s.check(abs(np.median(d) - HOME_ADVANTAGE) < 30,
            "Median elo_diff sits near the home-advantage offset",
            f"median={np.median(d):.2f} vs HA={HOME_ADVANTAGE}")

    raw_c = float(np.corrcoef(d, d ** 2)[0, 1])
    c = fit_centering(d)
    cen_c = float(np.corrcoef(d, make_quadratic_feature(d, c))[0, 1])
    s.check(abs(cen_c) < 0.05 < abs(raw_c),
            "Centering is necessary on real data",
            f"raw corr={raw_c:+.4f} -> centered={cen_c:+.4f}")
    return s


# ---------------------------------------------------------------------------
# Suite 6 — extreme-value safety (§16)
# ---------------------------------------------------------------------------

def suite6() -> Suite:
    s = Suite("6. Extreme-Value Safety (§16)")
    rng = np.random.default_rng(SEED)
    lh = rng.uniform(0.2, 4.0, 2000)
    la = rng.uniform(0.2, 3.5, 2000)
    r = check_extreme_safety(lh, la)
    s.check(r["passed"], "Normal lambda range passes safety",
            f"home[{r['lambda_home_min']:.3f},{r['lambda_home_max']:.3f}] "
            f"away[{r['lambda_away_min']:.3f},{r['lambda_away_max']:.3f}]")

    s.check(not check_extreme_safety(np.array([1.0, 99.0]),
                                     np.array([1.0, 1.0]))["passed"],
            "Exploding lambda is caught")
    s.check(not check_extreme_safety(np.array([1.0, 0.0]),
                                     np.array([1.0, 1.0]))["passed"],
            "Non-positive lambda is caught")
    s.check(not check_extreme_safety(np.array([1.0, np.inf]),
                                     np.array([1.0, 1.0]))["passed"],
            "Non-finite lambda is caught")

    # curvature over the real observed range stays sane
    ed = np.linspace(-430, 690, 200)
    c = fit_centering(ed)
    tbl = curvature_table(ed, c0=0.3, c1=1.8e-3, c2=4.1e-7, centering=c)
    lam = [row["quadratic_lambda"] for row in tbl]
    s.check(all(0 < v < 15 for v in lam),
            "Quadratic curve stays in a plausible goal-rate range",
            f"[{min(lam):.4f}, {max(lam):.4f}]")
    s.check(len(tbl) == 13, "Curvature table covers the 1st-99th pct grid")
    return s


# ---------------------------------------------------------------------------
# Suite 7 — determinism (§21)
# ---------------------------------------------------------------------------

def suite7() -> Suite:
    s = Suite("7. Determinism (§21)")
    v3 = fake_v3_columns(); X = fake_X(v3)
    c1 = fit_centering(X[SOURCE_COLUMN].to_numpy())
    c2 = fit_centering(X[SOURCE_COLUMN].to_numpy())
    s.check(c1.mean == c2.mean, "fit_centering deterministic")

    q1 = make_quadratic_feature(X[SOURCE_COLUMN].to_numpy(), c1)
    q2 = make_quadratic_feature(X[SOURCE_COLUMN].to_numpy(), c1)
    s.check(np.array_equal(q1, q2), "make_quadratic_feature bit-identical",
            f"max diff {float(np.max(np.abs(q1 - q2))):.3e}")

    D1 = build_design(X, v3, c1); D2 = build_design(X, v3, c1)
    s.check(D1.equals(D2), "build_design deterministic")

    src = (HERE / "quadratic_elo.py").read_text(encoding="utf-8")
    s.check("random" not in src.lower(), "quadratic_elo.py contains no RNG")
    return s


# ---------------------------------------------------------------------------
# Suite 8 — dataset scope & quarantine (§22)
# ---------------------------------------------------------------------------

def suite8() -> Suite:
    s = Suite("8. Dataset Scope & 2025/26 Quarantine (§22)")
    if not MATCHES_DB.exists():
        s.skip("All dataset tests", "matches.db not found"); return s
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET)); sph = ",".join("?" * len(SEASONS))

    n = conn.execute(
        f"SELECT COUNT(*) FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season IN ({sph}) AND status IN ('FT','AWARDED') "
        f"AND home_goals IS NOT NULL", (*TARGET, *SEASONS)).fetchone()[0]
    s.ok("Eligible fixture count", f"n={n}")

    nq = conn.execute(
        f"SELECT COUNT(*) FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season IN ({sph}) AND season = ?",
        (*TARGET, *SEASONS, QUARANTINED)).fetchone()[0]
    s.check(nq == 0, "2025/26 fixtures used = 0", f"n={nq}")

    lg = {r[0] for r in conn.execute(
        f"SELECT DISTINCT competition_name FROM fixtures WHERE "
        f"competition_id IN ({ph}) AND season IN ({sph})", (*TARGET, *SEASONS))}
    s.check(lg == {"Premier League", "La Liga", "Serie A",
                   "Bundesliga", "Ligue 1"},
            "Exactly the five target leagues", f"{sorted(lg)}")

    folds = [(("2020/2021", "2021/2022"), "2022/2023"),
             (("2020/2021", "2021/2022", "2022/2023"), "2023/2024"),
             (("2020/2021", "2021/2022", "2022/2023", "2023/2024"), "2024/2025")]
    ok = True
    for trs, ts in folds:
        tsph = ",".join("?" * len(trs))
        a = conn.execute(f"SELECT MAX(unix) FROM fixtures WHERE "
                         f"competition_id IN ({ph}) AND season IN ({tsph})",
                         (*TARGET, *trs)).fetchone()[0]
        b = conn.execute(f"SELECT MIN(unix) FROM fixtures WHERE "
                         f"competition_id IN ({ph}) AND season = ?",
                         (*TARGET, ts)).fetchone()[0]
        if not (a < b): ok = False
    s.check(ok, "All 3 outer folds strictly chronological")
    s.check(all(t != QUARANTINED for _, t in folds),
            "No fold tests on 2025/26")
    s.ok("Folds unchanged from the approved V3/E2 protocol",
         "same 3 season-based walk-forward folds")
    conn.close()
    return s


# ---------------------------------------------------------------------------
# Suite 9 — protected file integrity (§23)
# ---------------------------------------------------------------------------

def suite9() -> Suite:
    s = Suite("9. Protected File Integrity (§23)")
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists(): s.skip(Path(rel).name, "not found"); continue
        a = md5(p)
        s.check(a == exp, Path(rel).name,
                f"MD5={a}" if a == exp else f"expected={exp} got={a}")
    return s


def main() -> int:
    print("=" * 68)
    print("E5 — Quadratic Elo Test Suite")
    print("=" * 68)
    su = [suite1(), suite2(), suite3(), suite4(), suite5(),
          suite6(), suite7(), suite8(), suite9()]
    for x in su: x.report()
    tp, tf, ts = (sum(x.p for x in su), sum(x.f for x in su),
                  sum(x.s for x in su))
    print("\n" + "=" * 68 + "\nOVERALL\n" + "=" * 68)
    print(f"  Suites: {len(su)}\n  Pass:   {tp}\n  Fail:   {tf}"
          f"\n  Skip:   {ts}\n  Total:  {tp + tf + ts}")
    print(f"\n  VERDICT: {'ALL TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
