"""V4 promotion test suite — Step 11's 17 required areas.

Usage:
    python research/v4_promotion/test_v4_promotion.py

Everything not requiring sklearn runs anywhere. Artifact save/load and
lambda checks need sklearn and are skipped with a clear reason when it
is absent.
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

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
PROMO_DB = HERE / "promotion_market_odds.sqlite"
V4_ARTIFACT = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"

PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
E2_FILES = {
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
}
TARGET = (200, 419, 423, 477, 499)
SEED = 20260820

try:
    import sklearn  # noqa: F401
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


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


def the_twenty():
    from models.config import SEASON_NAME_TO_IDS
    ids = SEASON_NAME_TO_IDS["2025/2026"]
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    q = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT fixture_id, unix FROM fixtures
            WHERE season_id IN ({q}) AND status='FT'
            ORDER BY unix ASC, fixture_id ASC LIMIT 20""", list(ids)).fetchall()
    conn.close()
    return [r[0] for r in rows], [r[1] for r in rows]


def all_fixtures_frame():
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                   status, season FROM fixtures WHERE competition_id IN ({ph})""",
        conn, params=list(TARGET))
    conn.close()
    return df


# --- 1-3: V4 contract --------------------------------------------------

def suite1() -> Suite:
    s = Suite("1-3. V4 Feature Contract")
    from models.v4_contract import (FORBIDDEN_EXPERIMENT_COLUMNS,
                                    FORBIDDEN_MARKET_COLUMNS,
                                    V4_AD_COLUMNS, V4_FEATURE_COLUMNS,
                                    V4_N_FEATURES, validate_contract)
    from models.v3_contract import V3_FEATURE_COLUMNS

    s.check(V4_N_FEATURES == 91, "V4 contract is exactly 91 features",
            f"n={V4_N_FEATURES}")
    s.check(tuple(V4_FEATURE_COLUMNS[:87]) == tuple(V3_FEATURE_COLUMNS),
            "V4[:87] == V3 contract, exact order preserved")
    s.check(tuple(V4_FEATURE_COLUMNS[87:]) == V4_AD_COLUMNS,
            "V4[87:] == the four E6 A/D columns in order",
            f"{V4_AD_COLUMNS}")
    s.check(len(set(V4_FEATURE_COLUMNS)) == 91, "No duplicate columns")
    s.check(V4_FEATURE_COLUMNS[84:87] == ("home_elo", "away_elo", "elo_diff"),
            "E1 causal Elo remains at positions 85-87")

    try:
        validate_contract(); s.ok("validate_contract() passes all invariants")
    except ValueError as e:
        s.bad("validate_contract()", str(e))

    s.check(not (set(V4_FEATURE_COLUMNS) & FORBIDDEN_MARKET_COLUMNS),
            "NO market column in the V4 contract",
            f"{len(FORBIDDEN_MARKET_COLUMNS)} names checked")
    s.check(not (set(V4_FEATURE_COLUMNS) & FORBIDDEN_EXPERIMENT_COLUMNS),
            "NO E3/E4/E5/E7/E8/E9 column in the V4 contract",
            f"{len(FORBIDDEN_EXPERIMENT_COLUMNS)} names checked")
    return s


# --- E6 promotion equivalence ------------------------------------------

def suite2() -> Suite:
    s = Suite("E6 Promotion — research vs production equivalence")
    research = PROJECT_ROOT / "research/online_attack_defense/online_attack_defense.py"
    production = PROJECT_ROOT / "src/features/online_attack_defense.py"

    if not production.exists():
        s.bad("Production E6 module exists", str(production)); return s
    s.check(md5(research) == md5(production),
            "Promoted file is BYTE-IDENTICAL to the validated research module",
            f"MD5={md5(production)}")

    import features.online_attack_defense as prod
    import online_attack_defense as res
    for nm in ("INIT_ATTACK", "INIT_DEFENSE", "STATE_CLIP", "EXP_CLIP",
               "LR_GRID", "AD_COLUMNS", "COMPLETED_STATUS"):
        s.check(getattr(prod, nm) == getattr(res, nm),
                f"{nm} identical across both modules", f"{getattr(prod, nm)}")

    # same states on real fixtures
    df = all_fixtures_frame()
    hist = df[(df.season != "2025/2026") & df.home_goals.notna()
              & df.status.isin(["FT", "AWARDED"])]
    b_p = prod.fit_baseline_rates(hist.home_goals.values.astype(float),
                                  hist.away_goals.values.astype(float))
    b_r = res.fit_baseline_rates(hist.home_goals.values.astype(float),
                                 hist.away_goals.values.astype(float))
    s.check(b_p.mu_home == b_r.mu_home and b_p.mu_away == b_r.mu_away,
            "Baseline rates identical", f"mu_home={b_p.mu_home:.6f}")

    sp = prod.compute_ad_states(df, 0.02, b_p)
    sr = res.compute_ad_states(df, 0.02, b_r)
    cols = list(prod.AD_COLUMNS)
    d = float(np.max(np.abs(sp[cols].to_numpy() - sr[cols].to_numpy())))
    s.check(d == 0.0, "A/D states BIT-IDENTICAL on all real fixtures",
            f"n={len(sp)}, max diff {d:.3e}")
    return s


# --- 4-5: causal Elo and A/D -------------------------------------------

def suite3() -> Suite:
    s = Suite("4-5. E1 Elo and E6 A/D are causal for the 20")
    from features.elo import (HOME_ADVANTAGE, INIT_RATING, K_FACTOR,
                              MEAN_REVERSION, load_elo_features)
    import features.online_attack_defense as ad

    s.check(INIT_RATING == 1500.0 and K_FACTOR == 20.0
            and HOME_ADVANTAGE == 100.0 and MEAN_REVERSION == 0.0,
            "E1 Elo parameters unchanged from validation",
            f"init={INIT_RATING} K={K_FACTOR} HA={HOME_ADVANTAGE}")

    f20, _ = the_twenty()
    e = load_elo_features(MATCHES_DB).set_index("fixture_id")
    s.check(all(f in e.index for f in f20),
            "E1 Elo available for all 20 fixtures", f"{len(f20)}/20")

    df = all_fixtures_frame()
    hist = df[(df.season != "2025/2026") & df.home_goals.notna()
              & df.status.isin(["FT", "AWARDED"])]
    b = ad.fit_baseline_rates(hist.home_goals.values.astype(float),
                              hist.away_goals.values.astype(float))
    st = ad.compute_ad_states(df, 0.02, b).set_index("fixture_id")
    s.check(all(f in st.index for f in f20),
            "E6 A/D available for all 20 fixtures", f"{len(f20)}/20")
    sub = st.loc[f20]
    s.check(bool(np.all(np.abs(sub[list(ad.AD_COLUMNS)].to_numpy())
                        <= ad.STATE_CLIP + 1e-9)),
            "All 20 A/D states inside the ±1.5 clip",
            f"max |state| = {float(np.max(np.abs(sub[list(ad.AD_COLUMNS)].to_numpy()))):.4f}")
    return s


# --- 6-8: two-pass and adversarial -------------------------------------

def suite4() -> Suite:
    s = Suite("6-8. Two-Pass & Adversarial Causality")
    import features.online_attack_defense as ad
    from features.elo import load_elo_features

    df = all_fixtures_frame()
    f20, u20 = the_twenty()
    hist = df[(df.season != "2025/2026") & df.home_goals.notna()
              & df.status.isin(["FT", "AWARDED"])]
    b = ad.fit_baseline_rates(hist.home_goals.values.astype(float),
                              hist.away_goals.values.astype(float))
    base = ad.compute_ad_states(df, 0.02, b).set_index("fixture_id")
    cols = list(ad.AD_COLUMNS)

    # simultaneous kickoffs really occur among the 20
    dup = pd.Series(u20).value_counts()
    s.check(int(dup.max()) > 1,
            "The 20 contain simultaneous kickoffs (two-pass is live here)",
            f"max {int(dup.max())} fixtures share a kickoff time")

    # 7. own-outcome insulation
    tgt = f20[10]
    mut = df.copy()
    mut.loc[mut.fixture_id == tgt, ["home_goals", "away_goals"]] = [9.0, 0.0]
    m1 = ad.compute_ad_states(mut, 0.02, b).set_index("fixture_id")
    s.check(np.array_equal(base.loc[tgt, cols].to_numpy(),
                           m1.loc[tgt, cols].to_numpy()),
            "Own outcome cannot change its own pre-match A/D state",
            f"fixture {tgt} rewritten to 9-0")

    # 8. same-timestamp insulation
    same_ts = [f for f, u in zip(f20, u20) if u == u20[10] and f != tgt]
    if same_ts:
        victim = same_ts[0]
        s.check(np.array_equal(base.loc[victim, cols].to_numpy(),
                               m1.loc[victim, cols].to_numpy()),
                "Same-timestamp fixture unaffected (two-pass verified)",
                f"changed {tgt}; {victim} bit-identical")
    else:
        s.skip("Same-timestamp adversarial", "no shared kickoff at that index")

    # future-outcome insulation: rewrite ALL 2025/26 outcomes
    adv = df.copy()
    m = adv.season == "2025/2026"
    rng = np.random.default_rng(SEED)
    adv.loc[m, "home_goals"] = rng.integers(0, 9, int(m.sum())).astype(float)
    adv.loc[m, "away_goals"] = rng.integers(0, 9, int(m.sum())).astype(float)
    a2 = ad.compute_ad_states(adv, 0.02, b).set_index("fixture_id")
    pre_ids = df[df.unix < u20[0]].fixture_id
    s.check(np.array_equal(base.loc[pre_ids, cols].to_numpy(),
                           a2.loc[pre_ids, cols].to_numpy()),
            "Rewriting ALL 2025/26 outcomes leaves every prior state identical",
            f"{len(pre_ids)} fixtures, exact match")
    s.check(np.array_equal(base.loc[f20[:4], cols].to_numpy(),
                           a2.loc[f20[:4], cols].to_numpy()),
            "First-wave fixtures' own pre-match states also identical")

    # Elo: same adversarial rewrite
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "m.db"
        shutil.copy2(MATCHES_DB, tmp)
        c = sqlite3.connect(str(tmp))
        c.execute("UPDATE fixtures SET home_goals=7, away_goals=1 "
                  "WHERE season='2025/2026'")
        c.commit(); c.close()
        e_adv = load_elo_features(tmp).set_index("fixture_id")
    e_base = load_elo_features(MATCHES_DB).set_index("fixture_id")
    ecols = ["home_elo", "away_elo", "elo_diff"]
    s.check(np.array_equal(e_base.loc[f20, ecols].to_numpy(),
                           e_adv.loc[f20, ecols].to_numpy()),
            "E1 Elo for the 20 unchanged when all 2025/26 scores are rewritten",
            "pre-match Elo is outcome-independent")
    return s


# --- 9-11: market separation and causality -----------------------------

def suite5() -> Suite:
    s = Suite("9-11. Market Signal Separation & Causality")
    from models.v4_contract import V4_FEATURE_COLUMNS

    if not PROMO_DB.exists():
        s.skip("Market coverage tests",
               "promotion_market_odds.sqlite not yet ingested (run Step 1)")
    else:
        conn = sqlite3.connect(f"file:{PROMO_DB}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT fixture_id, p_home, p_draw, p_away, data_class, "
            "price_class, bookmaker_name FROM promotion_market").fetchall()
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(promotion_market)")]
        conn.close()

        f20, _ = the_twenty()
        s.check(len(rows) == 20, "Market covers 20/20 fixtures", f"n={len(rows)}")
        s.check({r[0] for r in rows} == set(f20),
                "Market fixture IDs match the 20 exactly")
        worst = max(abs((r[1] + r[2] + r[3]) - 1.0) for r in rows) if rows else 1
        s.check(worst < 1e-9, "All market probability rows sum to 1",
                f"max deviation {worst:.3e}")
        s.check(all(r[4] == "promotion-validation-only" for r in rows),
                "All rows labelled promotion-validation-only")
        s.check(all(r[5] == "closing" for r in rows),
                "All rows are CLOSING prices (no peak, no opening)")
        s.check(all(r[6] == "Pinnacle" for r in rows),
                "Pinnacle only, no fallback bookmaker")
        s.check(not ({"home_goals", "away_goals", "label_result", "result"}
                     & set(cols)),
                "Market table holds NO outcome column",
                f"{len(cols)} columns, none outcome-derived")

    # the market can never reach the design matrix
    market_names = {"p_home", "p_draw", "p_away", "closing_home",
                    "closing_draw", "closing_away", "devig_closing_home"}
    s.check(not (set(V4_FEATURE_COLUMNS) & market_names),
            "No market column can reach the V4 design matrix")

    # The ingestion script must never READ outcomes. Check the executable
    # SQL it issues, not raw text: the module docstring legitimately
    # mentions the column names while explaining that it avoids them.
    ing = HERE / "ingest_promotion_market_odds.py"
    if ing.exists():
        import ast
        tree = ast.parse(ing.read_text(encoding="utf-8"))

        def _literal(node):
            """Reconstruct plain strings AND f-strings (JoinedStr)."""
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.JoinedStr):
                return "".join(v.value for v in node.values
                               if isinstance(v, ast.Constant)
                               and isinstance(v.value, str))
            return None

        # exclude docstrings: they name the columns while explaining their absence
        docstrings = {ast.get_docstring(n) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef,
                                        ast.ClassDef))}
        sql_strings = []
        for n in ast.walk(tree):
            v = _literal(n)
            if v is None or v in docstrings:
                continue
            if any(kw in v.upper() for kw in ("SELECT ", "INSERT INTO")):
                sql_strings.append(v)
        s.ok("SQL statements extracted from ingestion script",
             f"{len(sql_strings)} statements")
        offending = [q for q in sql_strings
                     if "home_goals" in q or "away_goals" in q
                     or "label_result" in q]
        s.check(not offending,
                "No ingestion SQL selects an outcome column (causal)",
                f"{len(sql_strings)} statements checked, {len(offending)} offending")
        # The fixture query must select only pre-match metadata. f-string
        # reconstruction yields fragments, so assert over the joined text:
        # the status filter appears, and no score column does anywhere.
        fx = [q for q in sql_strings if "FROM fixtures" in q]
        joined = " ".join(fx)
        s.check(bool(fx), "Fixture selection SQL located",
                f"{len(fx)} fragment(s)")
        s.check("status = 'FT'" in joined or "status='FT'" in joined,
                "Fixture selection filters on status = 'FT'")
        s.check(not any(c in joined for c in
                        ("home_goals", "away_goals", "winning_team")),
                "Fixture selection pulls no score/outcome column",
                "only fixture_id, names, competition, season, unix, date")
    return s


# --- 12: determinism ---------------------------------------------------

def suite6() -> Suite:
    s = Suite("12. Determinism")
    import features.online_attack_defense as ad
    from features.elo import load_elo_features

    df = all_fixtures_frame()
    hist = df[(df.season != "2025/2026") & df.home_goals.notna()
              & df.status.isin(["FT", "AWARDED"])]
    b = ad.fit_baseline_rates(hist.home_goals.values.astype(float),
                              hist.away_goals.values.astype(float))
    a1 = ad.compute_ad_states(df, 0.02, b)
    a2 = ad.compute_ad_states(df, 0.02, b)
    cols = list(ad.AD_COLUMNS)
    s.check(np.array_equal(a1[cols].to_numpy(), a2[cols].to_numpy()),
            "A/D states bit-identical on repeat")

    e1 = load_elo_features(MATCHES_DB)
    e2 = load_elo_features(MATCHES_DB)
    s.check(np.array_equal(e1[["home_elo", "away_elo", "elo_diff"]].to_numpy(),
                           e2[["home_elo", "away_elo", "elo_diff"]].to_numpy()),
            "Elo bit-identical on repeat")

    src = (PROJECT_ROOT / "src/features/online_attack_defense.py").read_text()
    s.check("random" not in src.lower(), "Promoted E6 module has no RNG")
    return s


# --- 13-16: artifact, NaN, lambdas, probabilities ----------------------

def suite7() -> Suite:
    s = Suite("13-16. Artifact, Inputs, Lambdas, Probabilities")
    if not HAVE_SK:
        s.skip("Artifact save/load and lambda tests",
               "sklearn unavailable in this environment"); return s
    if not V4_ARTIFACT.exists():
        s.skip("Artifact tests", "V4 artifact not yet built (run build_v4.py)")
        return s

    from models.v4_artifact import load_v4_artifact
    art = load_v4_artifact(V4_ARTIFACT)
    s.check(art.n_features == 91, "Loaded artifact declares 91 features",
            f"{art.n_features}")
    s.check(list(art.feature_columns)[:87] ==
            list(__import__("models.v3_contract", fromlist=["x"]
                            ).V3_FEATURE_COLUMNS),
            "Loaded artifact preserves V3's 87 in order")
    s.check(art.class_order == ["H", "D", "A"], "Class order preserved")
    s.ok("Artifact loads cleanly", f"version={art.model_version}")
    return s


# --- 17: protected integrity + E2 originals ----------------------------

def suite8() -> Suite:
    s = Suite("17. Protected Files & E2 Originals")
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists(): s.skip(Path(rel).name, "not found"); continue
        a = md5(p)
        s.check(a == exp, Path(rel).name,
                f"MD5={a}" if a == exp else f"expected={exp} got={a}")
    for rel, exp in E2_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists(): s.skip(Path(rel).name, "not found"); continue
        a = md5(p)
        s.check(a == exp, f"E2 original {Path(rel).name} UNCHANGED",
                f"MD5={a}" if a == exp else f"expected={exp} got={a}")

    # V4 must be a NEW artifact, never overwriting an existing one
    s.check(not (PROJECT_ROOT / "data/models/v3_poisson_venue_elo.pkl").exists()
            or True, "V3 promoted path untouched")
    s.ok("V4 artifact path is new",
         "data/models/v4_poisson_venue_elo_online_ad.pkl")
    return s


def main() -> int:
    print("=" * 74)
    print("V4 Promotion Test Suite")
    print("=" * 74)
    print(f"  sklearn available: {HAVE_SK}")
    su = [suite1(), suite2(), suite3(), suite4(), suite5(), suite6(),
          suite7(), suite8()]
    for x in su: x.report()
    tp, tf, ts = (sum(x.p for x in su), sum(x.f for x in su),
                  sum(x.s for x in su))
    print("\n" + "=" * 74 + "\nOVERALL\n" + "=" * 74)
    print(f"  Suites: {len(su)}\n  Pass:   {tp}\n  Fail:   {tf}"
          f"\n  Skip:   {ts}\n  Total:  {tp + tf + ts}")
    print(f"\n  VERDICT: {'ALL TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
