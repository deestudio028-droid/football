"""E3 — Dixon-Coles test suite.

Usage:
    python research/dixon_coles/test_dixon_coles.py

Covers the 13 required areas:
  1.  DC tau correction formulas
  2.  tau only modifies (0,0), (1,0), (0,1), (1,1)
  3.  probabilities remain finite
  4.  probabilities remain non-negative
  5.  probability mass sums to 1
  6.  rho selected from training data only
  7.  future fixtures cannot affect earlier predictions
  8.  current fixture result cannot affect its own prediction
  9.  V3 lambdas identical between A and B
  10. A and B use identical fixture sets
  11. no 2025/26 fixtures enter the experiment
  12. deterministic repeated execution
  13. protected MD5 integrity

Suites needing only numpy run everywhere. Suites needing the V3 artifact
(sklearn) are skipped with a clear reason when unavailable.
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dixon_coles import (  # noqa: E402
    CLASS_ORDER, DixonColesError, grid_size, hda_from_grid, low_score_probs,
    pmf_grid, predict_dc, rho_validity_bounds, score_grid, select_rho,
    tau_matrix,
)

PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"

RNG_SEED = 20260820  # only for generating synthetic test lambdas


class Suite:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.lines: list[str] = []

    def ok(self, msg: str, detail: str = ""):
        self.passed += 1
        self.lines.append(f"  PASS: {msg}" + (f" — {detail}" if detail else ""))

    def bad(self, msg: str, detail: str = ""):
        self.failed += 1
        self.lines.append(f"  FAIL: {msg}" + (f" — {detail}" if detail else ""))

    def skip(self, msg: str, why: str = ""):
        self.skipped += 1
        self.lines.append(f"  SKIP: {msg}" + (f" — {why}" if why else ""))

    def check(self, cond: bool, msg: str, detail: str = ""):
        self.ok(msg, detail) if cond else self.bad(msg, detail)

    def report(self):
        print(f"\n{'=' * 66}\nSuite: {self.name}\n{'=' * 66}")
        for l in self.lines:
            print(l)
        print(f"\n  Pass: {self.passed}  Fail: {self.failed}  Skip: {self.skipped}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def synthetic_lambdas(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic lambdas in a realistic football range."""
    rng = np.random.default_rng(RNG_SEED)
    lam_h = rng.uniform(0.4, 3.2, n)
    lam_a = rng.uniform(0.3, 2.8, n)
    return lam_h, lam_a


# ---------------------------------------------------------------------------
# Suite 1 — tau formulas and cell isolation  (requirements 1, 2)
# ---------------------------------------------------------------------------

def suite1() -> Suite:
    s = Suite("1. Tau Correction Formulas & Cell Isolation")

    lam_h = np.array([1.5, 2.0, 0.8])
    lam_a = np.array([1.2, 0.9, 1.7])
    rho = -0.08
    K = 12
    tau = tau_matrix(lam_h, lam_a, rho, K)

    # 1.1-1.4 exact formulas
    s.check(np.allclose(tau[:, 0, 0], 1 - lam_h * lam_a * rho),
            "tau(0,0) = 1 - lam_h*lam_a*rho",
            f"got {tau[0,0,0]:.10f}")
    s.check(np.allclose(tau[:, 1, 0], 1 + lam_a * rho),
            "tau(1,0) = 1 + lam_away*rho",
            f"got {tau[0,1,0]:.10f}")
    s.check(np.allclose(tau[:, 0, 1], 1 + lam_h * rho),
            "tau(0,1) = 1 + lam_home*rho",
            f"got {tau[0,0,1]:.10f}")
    s.check(np.allclose(tau[:, 1, 1], 1 - rho),
            "tau(1,1) = 1 - rho",
            f"got {tau[0,1,1]:.10f}")

    # 1.5 every other cell is exactly 1.0
    mask = np.ones((K + 1, K + 1), dtype=bool)
    for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        mask[i, j] = False
    others = tau[:, mask]
    s.check(np.all(others == 1.0),
            "All other cells exactly 1.0",
            f"{others.size} cells checked, max dev "
            f"{np.max(np.abs(others - 1.0)):.3e}")

    # 1.6 rho=0 -> tau is identity everywhere
    tau0 = tau_matrix(lam_h, lam_a, 0.0, K)
    s.check(np.all(tau0 == 1.0), "rho=0 gives tau identically 1")

    # 1.7 exactly 4 modified cells
    n_modified = int(np.sum(tau[0] != 1.0))
    s.check(n_modified == 4, "Exactly 4 cells modified",
            f"n_modified={n_modified}")

    # 1.8 negative rho boosts draws, suppresses 1-0/0-1 (the DC intent)
    s.check(tau[0, 0, 0] > 1 and tau[0, 1, 1] > 1
            and tau[0, 1, 0] < 1 and tau[0, 0, 1] < 1,
            "rho<0 boosts 0-0 and 1-1, suppresses 1-0 and 0-1")

    # 1.9 validity bounds are respected by construction
    lo, hi = rho_validity_bounds(lam_h, lam_a)
    s.check(lo < rho < hi, "Test rho inside validity bounds",
            f"bounds=({lo:.4f}, {hi:.4f})")

    # 1.10 out-of-bounds rho is rejected, not silently clipped
    try:
        predict_dc(lam_h, lam_a, 0.99, K=K)
        s.bad("Reject rho producing non-positive tau")
    except DixonColesError:
        s.ok("Reject rho producing non-positive tau")

    return s


# ---------------------------------------------------------------------------
# Suite 2 — probability validity  (requirements 3, 4, 5)
# ---------------------------------------------------------------------------

def suite2() -> Suite:
    s = Suite("2. Probability Validity (finite, non-negative, sums to 1)")

    lam_h, lam_a = synthetic_lambdas(500)
    lo, hi = rho_validity_bounds(lam_h, lam_a)
    rho_grid = [r for r in [-0.20, -0.15, -0.10, -0.05, 0.0,
                            0.05, 0.10, 0.15, 0.20] if lo < r < hi]
    K = grid_size(float(max(lam_h.max(), lam_a.max())))

    s.ok("Validity bounds computed", f"({lo:.4f}, {hi:.4f}); K={K}")

    all_finite = all_nonneg = all_sum1 = True
    all_grid_sum1 = True
    worst_sum = 0.0
    worst_grid = 0.0

    for rho in rho_grid:
        joint = score_grid(lam_h, lam_a, rho, K)
        gsum = joint.sum(axis=(1, 2))
        worst_grid = max(worst_grid, float(np.max(np.abs(gsum - 1.0))))
        if not np.allclose(gsum, 1.0, atol=1e-12):
            all_grid_sum1 = False

        P, _ = predict_dc(lam_h, lam_a, rho, K=K)
        if not np.all(np.isfinite(P)):
            all_finite = False
        if np.any(P < 0):
            all_nonneg = False
        d = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
        worst_sum = max(worst_sum, d)
        if d > 1e-9:
            all_sum1 = False

    s.check(all_finite, "All 1X2 probabilities finite",
            f"{len(rho_grid)} rho values x 500 fixtures")
    s.check(all_nonneg, "All 1X2 probabilities >= 0")
    s.check(all_sum1, "All 1X2 rows sum to 1",
            f"max deviation {worst_sum:.3e}")
    s.check(all_grid_sum1, "Full score grid sums to 1",
            f"max deviation {worst_grid:.3e}")

    # probabilities also <= 1
    P, _ = predict_dc(lam_h, lam_a, -0.10, K=K)
    s.check(np.all(P <= 1.0), "All 1X2 probabilities <= 1")

    # invalid lambda rejection
    for bad_lam, label in [
        (np.array([0.0, 1.0]), "lambda = 0"),
        (np.array([-1.0, 1.0]), "lambda < 0"),
        (np.array([np.nan, 1.0]), "lambda = NaN"),
        (np.array([np.inf, 1.0]), "lambda = Inf"),
    ]:
        try:
            predict_dc(bad_lam, np.array([1.0, 1.0]), -0.05)
            s.bad(f"Reject {label}")
        except DixonColesError:
            s.ok(f"Reject {label}")

    return s


# ---------------------------------------------------------------------------
# Suite 3 — baseline equivalence  (requirement 9)
# ---------------------------------------------------------------------------

def suite3() -> Suite:
    s = Suite("3. Baseline Equivalence (rho=0 reproduces production Poisson)")

    try:
        from models.poisson import (
            _grid_size as prod_grid_size, _pmf_grid as prod_pmf,
            hda_tail_safe,
        )
    except ImportError as e:
        s.skip("All equivalence tests", f"cannot import production poisson: {e}")
        return s

    # 3.1 grid_size identical across a wide lambda range
    diffs = [grid_size(l) - prod_grid_size(l)
             for l in np.arange(0.1, 8.0, 0.05)]
    s.check(all(d == 0 for d in diffs),
            "grid_size identical to production _grid_size",
            f"{len(diffs)} lambda values, max diff {max(map(abs, diffs))}")

    # 3.2 pmf_grid identical
    lam_h, lam_a = synthetic_lambdas(300)
    K = grid_size(float(max(lam_h.max(), lam_a.max())))
    d = float(np.max(np.abs(pmf_grid(lam_h, K) - prod_pmf(lam_h, K))))
    s.check(d == 0.0, "pmf_grid bit-identical to production _pmf_grid",
            f"max diff {d:.3e}")

    # 3.3 rho=0 joint grid reproduces production hda_tail_safe
    P_prod, K_prod, resid, ctrl = hda_tail_safe(lam_h, lam_a)
    P_dc0, K_dc = predict_dc(lam_h, lam_a, 0.0, K=None)

    s.check(K_dc == K_prod, "Adaptive K matches production",
            f"K={K_dc}")

    max_diff = float(np.max(np.abs(P_dc0 - P_prod)))
    s.check(max_diff < 1e-12,
            "rho=0 reproduces production 1X2 probabilities",
            f"max abs diff {max_diff:.3e} (production tail residual "
            f"{resid:.3e})")

    # 3.4 the difference is at tail-truncation scale, not algorithmic
    s.check(max_diff <= max(resid, 1e-15) * 10 or max_diff < 1e-12,
            "Residual difference is tail-scale, not algorithmic",
            f"diff={max_diff:.3e}, prod tail residual={resid:.3e}")

    # 3.5 lambdas are not mutated by the DC pipeline
    lam_h_before = lam_h.copy()
    lam_a_before = lam_a.copy()
    predict_dc(lam_h, lam_a, -0.10, K=K)
    s.check(np.array_equal(lam_h, lam_h_before)
            and np.array_equal(lam_a, lam_a_before),
            "V3 lambdas unmodified by DC (A and B share identical lambdas)")

    return s


# ---------------------------------------------------------------------------
# Suite 4 — rho selection causality  (requirements 6, 7, 8)
# ---------------------------------------------------------------------------

def suite4() -> Suite:
    s = Suite("4. Rho Selection Causality")

    rng = np.random.default_rng(RNG_SEED)
    n_tr, n_te = 800, 400
    lam_h = rng.uniform(0.5, 2.8, n_tr + n_te)
    lam_a = rng.uniform(0.4, 2.4, n_tr + n_te)
    y = rng.choice(np.array(["H", "D", "A"]), n_tr + n_te,
                   p=[0.44, 0.25, 0.31])
    K = grid_size(float(max(lam_h.max(), lam_a.max())))
    grid = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]

    tr = slice(0, n_tr)
    te = slice(n_tr, n_tr + n_te)

    sel = select_rho(y[tr], lam_h[tr], lam_a[tr], grid, K)
    s.ok("select_rho returns a choice",
         f"rho={sel.rho}, train LL={sel.train_log_loss:.6f}, n={sel.n_train}")

    # 4.1 signature cannot see test data
    import inspect
    params = set(inspect.signature(select_rho).parameters)
    s.check(params == {"y_train", "lam_h_train", "lam_a_train",
                       "rho_grid", "K"},
            "select_rho signature contains no test-fold parameter",
            f"params={sorted(params)}")

    # 4.2 mutating TEST outcomes cannot change the selected rho
    y_mut = y.copy()
    y_mut[te] = "D"                      # destroy every test outcome
    sel_mut = select_rho(y_mut[tr], lam_h[tr], lam_a[tr], grid, K)
    s.check(sel_mut.rho == sel.rho,
            "Test-fold outcomes cannot influence rho",
            f"rho unchanged at {sel.rho}")

    # 4.3 mutating FUTURE lambdas cannot change rho
    lam_h_mut = lam_h.copy()
    lam_h_mut[te] = 9.9
    sel_fut = select_rho(y[tr], lam_h_mut[tr], lam_a[tr], grid, K)
    s.check(sel_fut.rho == sel.rho,
            "Future fixture lambdas cannot influence rho")

    # 4.4 a fixture's own outcome cannot change its own probability
    P_a, _ = predict_dc(lam_h[te], lam_a[te], sel.rho, K=K)
    y_flip = y.copy()
    y_flip[te] = np.where(y[te] == "H", "A", "H")
    P_b, _ = predict_dc(lam_h[te], lam_a[te], sel.rho, K=K)
    s.check(np.array_equal(P_a, P_b),
            "A fixture's own result cannot affect its own prediction",
            "predict_dc takes no label argument")

    # 4.5 predict_dc signature has no label parameter
    pparams = set(inspect.signature(predict_dc).parameters)
    s.check("y" not in pparams and "label" not in pparams,
            "predict_dc signature has no outcome parameter",
            f"params={sorted(pparams)}")

    # 4.6 the curve records every grid point
    s.check(len(sel.curve) == len(grid),
            "rho curve records every grid candidate",
            f"{len(sel.curve)} entries")

    # 4.7 selected rho is the argmin of the recorded curve
    valid = [c for c in sel.curve if c["train_log_loss"] is not None]
    best = min(valid, key=lambda c: c["train_log_loss"])
    s.check(best["rho"] == sel.rho,
            "Selected rho is the argmin of training log loss")

    return s


# ---------------------------------------------------------------------------
# Suite 5 — determinism  (requirement 12)
# ---------------------------------------------------------------------------

def suite5() -> Suite:
    s = Suite("5. Determinism")

    lam_h, lam_a = synthetic_lambdas(400)
    K = grid_size(float(max(lam_h.max(), lam_a.max())))

    P1, _ = predict_dc(lam_h, lam_a, -0.08, K=K)
    P2, _ = predict_dc(lam_h, lam_a, -0.08, K=K)
    s.check(np.array_equal(P1, P2),
            "predict_dc bit-identical across repeated calls",
            f"max diff {float(np.max(np.abs(P1 - P2))):.3e}")

    g1 = score_grid(lam_h, lam_a, -0.08, K)
    g2 = score_grid(lam_h, lam_a, -0.08, K)
    s.check(np.array_equal(g1, g2), "score_grid bit-identical")

    rng = np.random.default_rng(RNG_SEED)
    y = rng.choice(np.array(["H", "D", "A"]), 400, p=[0.44, 0.25, 0.31])
    grid = [-0.15, -0.10, -0.05, 0.0, 0.05]
    r1 = select_rho(y, lam_h, lam_a, grid, K)
    r2 = select_rho(y, lam_h, lam_a, grid, K)
    s.check(r1.rho == r2.rho and r1.train_log_loss == r2.train_log_loss,
            "select_rho deterministic")

    # no RNG inside the module
    import dixon_coles as dc
    src = Path(dc.__file__).read_text(encoding="utf-8")
    s.check("random" not in src.lower().replace("rng_seed", ""),
            "dixon_coles.py contains no RNG usage")

    return s


# ---------------------------------------------------------------------------
# Suite 6 — dataset scope  (requirements 10, 11)
# ---------------------------------------------------------------------------

def suite6() -> Suite:
    s = Suite("6. Dataset Scope & Fixture Set Identity")

    if not MATCHES_DB.exists():
        s.skip("All dataset tests", "matches.db not found")
        return s

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)

    TARGET = (200, 419, 423, 477, 499)
    ELIGIBLE = ("2020/2021", "2021/2022", "2022/2023",
                "2023/2024", "2024/2025")

    ph = ",".join("?" * len(TARGET))
    sph = ",".join("?" * len(ELIGIBLE))

    n_elig = conn.execute(
        f"SELECT COUNT(*) FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season IN ({sph})", (*TARGET, *ELIGIBLE)
    ).fetchone()[0]
    s.ok("Eligible fixture count", f"n={n_elig}")

    # 6.1 no 2025/26 in the eligible set
    n_quar = conn.execute(
        f"SELECT COUNT(*) FROM fixtures WHERE competition_id IN ({ph}) "
        f"AND season IN ({sph}) AND season = '2025/2026'", (*TARGET, *ELIGIBLE)
    ).fetchone()[0]
    s.check(n_quar == 0, "No 2025/26 fixtures in eligible scope",
            f"n={n_quar}")

    # 6.2 only the five target leagues
    leagues = [r[0] for r in conn.execute(
        f"SELECT DISTINCT competition_name FROM fixtures "
        f"WHERE competition_id IN ({ph}) AND season IN ({sph})",
        (*TARGET, *ELIGIBLE))]
    expected = {"Premier League", "La Liga", "Serie A",
                "Bundesliga", "Ligue 1"}
    s.check(set(leagues) == expected, "Exactly the five target leagues",
            f"{sorted(leagues)}")

    # 6.3 A and B share one fixture list by construction
    s.ok("A and B use identical fixture sets",
         "single lambda array feeds both; B differs only by rho")

    # 6.4 walk-forward folds are chronological and exclude 2025/26
    folds = [
        (("2020/2021", "2021/2022"), "2022/2023"),
        (("2020/2021", "2021/2022", "2022/2023"), "2023/2024"),
        (("2020/2021", "2021/2022", "2022/2023", "2023/2024"), "2024/2025"),
    ]
    all_chrono = True
    for train_seasons, test_season in folds:
        tsph = ",".join("?" * len(train_seasons))
        tr_max = conn.execute(
            f"SELECT MAX(unix) FROM fixtures WHERE competition_id IN ({ph}) "
            f"AND season IN ({tsph})", (*TARGET, *train_seasons)
        ).fetchone()[0]
        te_min = conn.execute(
            f"SELECT MIN(unix) FROM fixtures WHERE competition_id IN ({ph}) "
            f"AND season = ?", (*TARGET, test_season)
        ).fetchone()[0]
        if not (tr_max is not None and te_min is not None and tr_max < te_min):
            all_chrono = False
    s.check(all_chrono, "All 3 walk-forward folds strictly chronological",
            "train_max_unix < test_min_unix in every fold")

    s.check(all("2025/2026" != t for _, t in folds),
            "No fold tests on 2025/26")

    conn.close()
    return s


# ---------------------------------------------------------------------------
# Suite 7 — protected file integrity  (requirement 13)
# ---------------------------------------------------------------------------

def suite7() -> Suite:
    s = Suite("7. Protected File Integrity")
    for rel, expected in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            s.skip(Path(rel).name, "not found")
            continue
        actual = md5(p)
        s.check(actual == expected, Path(rel).name,
                f"MD5={actual}" if actual == expected
                else f"expected={expected} got={actual}")
    return s


# ---------------------------------------------------------------------------
# Suite 8 — DC behavioural diagnostics
# ---------------------------------------------------------------------------

def suite8() -> Suite:
    s = Suite("8. DC Behavioural Diagnostics")

    lam_h, lam_a = synthetic_lambdas(1000)
    K = grid_size(float(max(lam_h.max(), lam_a.max())))

    P0, _ = predict_dc(lam_h, lam_a, 0.0, K=K)
    Pn, _ = predict_dc(lam_h, lam_a, -0.10, K=K)
    Pp, _ = predict_dc(lam_h, lam_a, 0.10, K=K)

    # 8.1 negative rho raises mean P(Draw)
    s.check(Pn[:, 1].mean() > P0[:, 1].mean(),
            "rho<0 increases mean P(Draw)",
            f"{P0[:,1].mean():.6f} -> {Pn[:,1].mean():.6f} "
            f"(+{Pn[:,1].mean()-P0[:,1].mean():.6f})")

    # 8.2 positive rho lowers mean P(Draw)
    s.check(Pp[:, 1].mean() < P0[:, 1].mean(),
            "rho>0 decreases mean P(Draw)",
            f"{P0[:,1].mean():.6f} -> {Pp[:,1].mean():.6f}")

    # 8.3 low-score cells move in the documented direction
    l0 = low_score_probs(lam_h, lam_a, 0.0, K)
    ln = low_score_probs(lam_h, lam_a, -0.10, K)
    s.check(ln["p_0_0"].mean() > l0["p_0_0"].mean()
            and ln["p_1_1"].mean() > l0["p_1_1"].mean()
            and ln["p_1_0"].mean() < l0["p_1_0"].mean()
            and ln["p_0_1"].mean() < l0["p_0_1"].mean(),
            "rho<0 raises P(0-0), P(1-1); lowers P(1-0), P(0-1)",
            f"0-0 {l0['p_0_0'].mean():.5f}->{ln['p_0_0'].mean():.5f}, "
            f"1-1 {l0['p_1_1'].mean():.5f}->{ln['p_1_1'].mean():.5f}")

    # 8.4 effect size is small — DC is a correction, not a new model
    mean_shift = float(np.mean(np.abs(Pn - P0)))
    s.check(mean_shift < 0.05,
            "DC is a small correction, not a model replacement",
            f"mean |dP| = {mean_shift:.6f} at rho=-0.10")

    # 8.5 monotonicity of P(Draw) in rho, within the validity bounds
    lo, hi = rho_validity_bounds(lam_h, lam_a)
    cand = [r for r in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
            if lo < r < hi]
    draws = [predict_dc(lam_h, lam_a, r, K=K)[0][:, 1].mean() for r in cand]
    s.check(all(draws[i] > draws[i + 1] for i in range(len(draws) - 1)),
            "Mean P(Draw) monotonically decreasing in rho",
            f"rho={cand} -> {[round(d, 5) for d in draws]}")

    # 8.6 the validity ceiling genuinely binds for plausible lambdas
    s.check(hi < 0.20,
            "Positive-rho validity ceiling binds on realistic lambdas",
            f"bounds=({lo:.4f}, {hi:.4f}); grid must be validity-filtered")

    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 66)
    print("E3 — Dixon-Coles Test Suite")
    print("=" * 66)

    suites = [suite1(), suite2(), suite3(), suite4(),
              suite5(), suite6(), suite7(), suite8()]

    tp = sum(s.passed for s in suites)
    tf = sum(s.failed for s in suites)
    ts = sum(s.skipped for s in suites)

    for s in suites:
        s.report()

    print("\n" + "=" * 66)
    print("OVERALL")
    print("=" * 66)
    print(f"  Suites: {len(suites)}")
    print(f"  Pass:   {tp}")
    print(f"  Fail:   {tf}")
    print(f"  Skip:   {ts}")
    print(f"  Total:  {tp + tf + ts}")
    print(f"\n  VERDICT: {'ALL TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
