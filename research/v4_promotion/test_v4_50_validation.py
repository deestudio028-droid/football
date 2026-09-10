"""V4 50-match validation — pre-evaluation test suite.

Usage:
    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python research/v4_promotion/test_v4_50_validation.py

MUST pass with 0 failures before run_v4_50_validation.py is executed.

Per Phase 17 these are BEHAVIOURAL tests against real data. No fragile
string searches, no assertions about source-code prose. AST is used in
exactly one place, where structural source verification is genuinely the
property under test (that the market ingestion never queried an outcome
column).

This file tests the 50-match validation only. It never writes to, or
modifies, the frozen 20-match harness or its outputs.
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
PROMO_DB = HERE / "promotion_market_odds.sqlite"

V2_ARTIFACT = PROJECT_ROOT / "data/models/v2_poisson_venue.pkl"
V3_ARTIFACT = PROJECT_ROOT / "data/models/v3_poisson_venue_elo_candidate.pkl"
V4_ARTIFACT = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"

PROTECTED = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
}

#: The frozen 20, in order. The 50-fixture selection must begin with these.
FROZEN_20 = [
    343465962, 343465735, 342254747, 343465963, 342863654,
    342863842, 342863843, 342863844, 343465733, 342864234,
    343465729, 342864255, 343465732, 342863998, 342864289,
    343465726, 343465727, 343465734, 343465842, 343465728,
]

#: Confirmed deterministic composition (Option A). The earlier
#: 17/13/12/5/3 figure was an incorrect prior estimate.
EXPECTED_COMPOSITION = {
    "Premier League": 16, "La Liga": 13, "Ligue 1": 12,
    "Bundesliga": 7, "Serie A": 2,
}

E6_LEARNING_RATE = 0.02
AD_CLIP = 1.5

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
        print(f"\n{'=' * 78}\nSuite: {s.n}\n{'=' * 78}")
        for x in s.l: print(x)
        print(f"\n  Pass: {s.p}  Fail: {s.f}  Skip: {s.s}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()


def select_50():
    """The deterministic 50-fixture selection. Authoritative."""
    from models.config import SEASON_NAME_TO_IDS
    ids = SEASON_NAME_TO_IDS["2025/2026"]
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    q = ",".join("?" * len(ids))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, date, competition_name, home_name, away_name,
                   home_goals, away_goals, status, unix, season_id
            FROM fixtures WHERE season_id IN ({q}) AND status = 'FT'
            ORDER BY unix ASC, fixture_id ASC LIMIT 50""",
        conn, params=list(ids))
    conn.close()
    return df


def all_fixtures():
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    df = pd.read_sql_query(
        """SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                  status, season, season_id, competition_id
           FROM fixtures WHERE competition_id IN (200,419,423,477,499)""",
        conn)
    conn.close()
    return df


# --- 1. fixture selection ----------------------------------------------

def suite1() -> Suite:
    s = Suite("1. Deterministic 50-Fixture Selection")
    df = select_50()

    s.check(len(df) == 50, "Exactly 50 fixtures selected", f"n={len(df)}")
    s.check(df.fixture_id.nunique() == 50, "No duplicate fixture_id")
    s.check(bool((df.status == "FT").all()), "All fixtures status = FT")
    u = df.unix.to_numpy()
    s.check(bool(np.all(u[:-1] <= u[1:])), "Strict chronological ordering")
    s.check(df.fixture_id.tolist()[:20] == FROZEN_20,
            "First 20 are EXACTLY the frozen 20-match set")

    # reproducible: run it again
    df2 = select_50()
    s.check(df.fixture_id.tolist() == df2.fixture_id.tolist(),
            "Selection is reproducible across calls")

    comp = Counter(df.competition_name)
    s.check(dict(comp) == EXPECTED_COMPOSITION,
            "League composition matches the confirmed deterministic result",
            f"{dict(sorted(comp.items(), key=lambda kv: -kv[1]))}")
    s.check(sum(comp.values()) == 50, "Composition sums to 50")

    s.check(bool(df.home_goals.notna().all() and df.away_goals.notna().all()),
            "All 50 have valid goal labels")
    s.check(bool((df.home_goals >= 0).all() and (df.away_goals >= 0).all()),
            "All goal labels non-negative")
    return s


# --- 2. artifacts -------------------------------------------------------

def suite2() -> Suite:
    s = Suite("2. Artifact Verification (no retraining)")
    if not HAVE_SK:
        s.skip("All artifact tests", "sklearn unavailable"); return s

    from models.v2_artifact import load as load_v2
    from models.v3_artifact import load_v3_artifact
    from models.v4_artifact import load_v4_artifact
    from models.v3_contract import V3_FEATURE_COLUMNS

    v2 = load_v2(V2_ARTIFACT)
    s.ok("V2 artifact loads", f"version={v2.model_version}")
    s.check(list(v2.class_order) == ["H", "D", "A"], "V2 class order H/D/A")
    s.check(hasattr(v2.preprocessor, "transform")
            and hasattr(v2.model_home_goals, "predict"),
            "V2 has valid preprocessing and model objects")

    v3 = load_v3_artifact(V3_ARTIFACT)
    s.check(v3.n_features == 87, "V3 has exactly 87 features",
            f"{v3.n_features}")
    s.check(tuple(v3.feature_columns) == tuple(V3_FEATURE_COLUMNS),
            "V3 feature columns match the V3 contract exactly")

    v4 = load_v4_artifact(V4_ARTIFACT)
    s.check(v4.n_features == 91, "V4 has exactly 91 features",
            f"{v4.n_features}")
    s.check(tuple(v4.feature_columns[:87]) == tuple(V3_FEATURE_COLUMNS),
            "V4[:87] == V3 contract, exact order")
    s.check(tuple(v4.feature_columns[87:]) ==
            ("A_home", "D_home", "A_away", "D_away"),
            "V4[87:] == the four E6 A/D columns in order",
            f"{tuple(v4.feature_columns[87:])}")
    s.check("2025/2026" not in v4.training_seasons,
            "V4 training seasons exclude 2025/2026",
            f"{list(v4.training_seasons)}")
    s.check(v4.holdout_used_in_training is False,
            "V4 holdout_used_in_training is False")
    s.check(md5(V4_ARTIFACT) == PROTECTED[
        "data/models/v4_poisson_venue_elo_online_ad.pkl"],
        "V4 artifact MD5 unchanged", md5(V4_ARTIFACT))
    return s


# --- 3. causal features -------------------------------------------------

def suite3() -> Suite:
    s = Suite("3. Causal Feature Availability (E1 Elo, E6 A/D)")
    from features.elo import ELO_COLUMNS, load_elo_features
    from features.online_attack_defense import (AD_COLUMNS, STATE_CLIP,
                                                compute_ad_states,
                                                fit_baseline_rates)
    from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS

    df = select_50()
    f50 = df.fixture_id.tolist()

    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    s.check(all(f in elo.index for f in f50),
            "E1 causal Elo available for 50/50 fixtures")
    sub = elo.loc[f50, list(ELO_COLUMNS)]
    s.check(bool(np.all(np.isfinite(sub.to_numpy()))),
            "All Elo values finite",
            f"home_elo [{sub.home_elo.min():.1f}, {sub.home_elo.max():.1f}]")

    fx = all_fixtures()
    wanted = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted |= set(SEASON_NAME_TO_IDS[sn])
    hist = fx[fx.season_id.isin(wanted) & fx.home_goals.notna()
              & fx.status.isin(["FT", "AWARDED"])]
    s.check("2025/2026" not in set(hist.season),
            "A/D baseline rates fitted on pre-2025/26 fixtures only")
    base = fit_baseline_rates(hist.home_goals.values.astype(float),
                              hist.away_goals.values.astype(float))
    st = compute_ad_states(fx, E6_LEARNING_RATE, base).set_index("fixture_id")
    s.check(all(f in st.index for f in f50),
            "E6 online A/D available for 50/50 fixtures")
    vals = st.loc[f50, list(AD_COLUMNS)].to_numpy()
    s.check(bool(np.all(np.isfinite(vals))), "All A/D states finite")
    s.check(bool(np.all(np.abs(vals) <= STATE_CLIP + 1e-9)),
            f"All A/D states within ±{STATE_CLIP}",
            f"max |state| = {float(np.max(np.abs(vals))):.6f}")

    # feature rows exist for every fixture
    conn = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    n = conn.execute(
        "SELECT COUNT(*) FROM feature_rows WHERE fixture_id IN "
        f"({','.join('?' * 50)})", f50).fetchone()[0]
    conn.close()
    s.check(n == 50, "Feature rows exist for 50/50 fixtures", f"{n}/50")
    return s


# --- 4. adversarial causality ------------------------------------------

def suite4() -> Suite:
    s = Suite("4. Adversarial Causality (rewriting outcomes)")
    from features.online_attack_defense import (AD_COLUMNS, compute_ad_states,
                                                fit_baseline_rates)
    from features.elo import load_elo_features
    from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS

    df = select_50()
    f50 = df.fixture_id.tolist()
    fx = all_fixtures()

    wanted = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted |= set(SEASON_NAME_TO_IDS[sn])
    hist = fx[fx.season_id.isin(wanted) & fx.home_goals.notna()
              & fx.status.isin(["FT", "AWARDED"])]
    base = fit_baseline_rates(hist.home_goals.values.astype(float),
                              hist.away_goals.values.astype(float))
    cols = list(AD_COLUMNS)
    st_base = compute_ad_states(fx, E6_LEARNING_RATE, base).set_index("fixture_id")

    # The 50 fixtures span 15-23 Aug, so a blanket rewrite of ALL 2025/26
    # outcomes also rewrites fixtures #1-30. Later fixtures then correctly
    # update from those rewritten EARLIER results -- that is online
    # learning working, not leakage. The property that actually matters is
    # per-fixture: a fixture's state must be immune to its OWN outcome and
    # to every outcome at-or-after its kickoff.
    u50 = df.unix.tolist()
    rng = np.random.default_rng(20260820)
    ad_ok, elo_ok, probe = True, True, [0, 10, 20, 30, 40, 49]
    worst_ad = worst_elo = 0.0

    e_base = load_elo_features(MATCHES_DB).set_index("fixture_id")
    ec = ["home_elo", "away_elo", "elo_diff"]
    import shutil, tempfile

    for i in probe:
        cut = u50[i]
        adv = fx.copy()
        m = (adv.season == "2025/2026") & (adv.unix >= cut)
        adv.loc[m, "home_goals"] = rng.integers(0, 9, int(m.sum())).astype(float)
        adv.loc[m, "away_goals"] = rng.integers(0, 9, int(m.sum())).astype(float)
        st_adv = compute_ad_states(adv, E6_LEARNING_RATE,
                                   base).set_index("fixture_id")
        d = float(np.max(np.abs(st_base.loc[f50[i], cols].to_numpy()
                                - st_adv.loc[f50[i], cols].to_numpy())))
        worst_ad = max(worst_ad, d)
        ad_ok &= (d == 0.0)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "m.db"
            shutil.copy2(MATCHES_DB, tmp)
            c = sqlite3.connect(str(tmp))
            c.execute("UPDATE fixtures SET home_goals=7, away_goals=1 "
                      "WHERE season='2025/2026' AND unix >= ?", (int(cut),))
            c.commit(); c.close()
            e_adv = load_elo_features(tmp).set_index("fixture_id")
        de = float(np.max(np.abs(e_base.loc[f50[i], ec].to_numpy()
                                 - e_adv.loc[f50[i], ec].to_numpy())))
        worst_elo = max(worst_elo, de)
        elo_ok &= (de == 0.0)

    s.check(ad_ok,
            "A/D state immune to its own outcome and all at-or-later outcomes",
            f"{len(probe)} cut points probed, max diff {worst_ad:.3e}")
    s.check(elo_ok,
            "Elo immune to its own outcome and all at-or-later outcomes",
            f"{len(probe)} cut points probed, max diff {worst_elo:.3e}")

    # baseline rates must be immune to ALL 2025/26 outcomes
    adv_all = fx.copy()
    ma = adv_all.season == "2025/2026"
    adv_all.loc[ma, "home_goals"] = rng.integers(0, 9, int(ma.sum())).astype(float)
    adv_all.loc[ma, "away_goals"] = rng.integers(0, 9, int(ma.sum())).astype(float)
    s.check(not np.array_equal(fx.loc[ma, "home_goals"].to_numpy(),
                               adv_all.loc[ma, "home_goals"].to_numpy()),
            "Adversarial copy genuinely changed 2025/26 outcomes",
            f"{int(ma.sum())} fixtures rewritten")
    hist_adv = adv_all[adv_all.season_id.isin(wanted) & adv_all.home_goals.notna()
                       & adv_all.status.isin(["FT", "AWARDED"])]
    b2 = fit_baseline_rates(hist_adv.home_goals.values.astype(float),
                            hist_adv.away_goals.values.astype(float))
    s.check(b2.mu_home == base.mu_home and b2.mu_away == base.mu_away,
            "Baseline rates unaffected by ANY rewritten 2025/26 outcome",
            f"mu_home={base.mu_home:.6f}")

    # the frozen 20 predate every other 2025/26 result that could disturb them
    st_all = compute_ad_states(adv_all, E6_LEARNING_RATE,
                               base).set_index("fixture_id")
    s.check(np.array_equal(st_base.loc[f50[:20], cols].to_numpy(),
                           st_all.loc[f50[:20], cols].to_numpy()),
            "Frozen 20 A/D states unchanged even under a FULL 2025/26 rewrite",
            "they precede every rewritten result that could affect them")

    # simultaneous kickoffs occur in this sample
    dup = df.unix.value_counts()
    s.check(int(dup.max()) > 1,
            "The 50 contain simultaneous kickoffs (two-pass is load-bearing)",
            f"max {int(dup.max())} fixtures share a kickoff time")

    # changing one fixture must not disturb a simultaneous sibling
    busiest = dup.idxmax()
    grp = df[df.unix == busiest].fixture_id.tolist()
    tgt, victim = grp[0], grp[1]
    mut = fx.copy()
    mut.loc[mut.fixture_id == tgt, ["home_goals", "away_goals"]] = [9.0, 0.0]
    st_mut = compute_ad_states(mut, E6_LEARNING_RATE, base).set_index("fixture_id")
    s.check(np.array_equal(st_base.loc[victim, cols].to_numpy(),
                           st_mut.loc[victim, cols].to_numpy()),
            "Simultaneous fixture unaffected by its sibling's rewritten score",
            f"changed {tgt}; {victim} bit-identical")
    s.check(np.array_equal(st_base.loc[tgt, cols].to_numpy(),
                           st_mut.loc[tgt, cols].to_numpy()),
            "A fixture's own outcome cannot change its own pre-match state")
    return s


# --- 5. market reference ------------------------------------------------

def suite5() -> Suite:
    s = Suite("5. Market Reference (50/50, separate signal)")
    if not PROMO_DB.exists():
        s.bad("Market database exists", str(PROMO_DB)); return s

    df = select_50()
    f50 = df.fixture_id.tolist()
    conn = sqlite3.connect(f"file:{PROMO_DB}?mode=ro", uri=True)
    mk = pd.read_sql_query(
        """SELECT fixture_id, p_home, p_draw, p_away, bookmaker_name,
                  price_class, data_class FROM promotion_market""", conn)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(promotion_market)")]
    conn.close()

    s.check(len(mk) == 50, "Market table holds exactly 50 rows", f"n={len(mk)}")
    missing = [f for f in f50 if f not in set(mk.fixture_id)]
    s.check(not missing, "Market covers 50/50 fixtures",
            f"missing={missing or 'none'}")
    s.check(set(mk.fixture_id) == set(f50),
            "Market fixture IDs match the 50 exactly, no extras")

    P = mk[["p_home", "p_draw", "p_away"]].to_numpy(float)
    s.check(bool(np.all(np.isfinite(P))), "All market probabilities finite")
    s.check(bool(np.all(P >= 0) and np.all(P <= 1)),
            "All market probabilities within [0, 1]")
    worst = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
    s.check(worst < 1e-9, "Market probability rows sum to 1",
            f"max deviation {worst:.3e}")

    s.check(set(mk.bookmaker_name) == {"Pinnacle"}, "Pinnacle only")
    s.check(set(mk.price_class) == {"closing"}, "Closing prices only")
    s.check(set(mk.data_class) == {"promotion-validation-only"},
            "All rows labelled promotion-validation-only")
    s.check(not ({"home_goals", "away_goals", "label_result", "result",
                  "winning_team"} & set(cols)),
            "Market table holds NO outcome column",
            f"{len(cols)} columns, none outcome-derived")

    # market can never reach the V4 design matrix
    from models.v4_contract import V4_FEATURE_COLUMNS, FORBIDDEN_MARKET_COLUMNS
    s.check(not (set(V4_FEATURE_COLUMNS) & set(mk.columns)),
            "No market table column appears in the V4 contract")
    s.check(not (set(V4_FEATURE_COLUMNS) & FORBIDDEN_MARKET_COLUMNS),
            "V4 contract contains no forbidden market column")

    # AST: the ingestion never queried an outcome column. Structural source
    # verification is genuinely the property here, so AST is appropriate.
    ing = HERE / "ingest_50_market_odds.py"
    if ing.exists():
        import ast
        tree = ast.parse(ing.read_text(encoding="utf-8"))
        docs = {ast.get_docstring(n) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}

        def lit(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.JoinedStr):
                return "".join(v.value for v in node.values
                               if isinstance(v, ast.Constant)
                               and isinstance(v.value, str))
            return None

        sql = [v for n in ast.walk(tree)
               if (v := lit(n)) and v not in docs
               and any(k in v.upper() for k in ("SELECT ", "INSERT INTO"))]
        offending = [q for q in sql if "home_goals" in q or "away_goals" in q
                     or "label_result" in q]
        s.check(not offending,
                "Market ingestion SQL never selects an outcome column",
                f"{len(sql)} statements, {len(offending)} offending")
    else:
        s.skip("Ingestion SQL audit", "ingest_50_market_odds.py absent")
    return s


# --- 6. probability validity across all arms ---------------------------

def suite6() -> Suite:
    s = Suite("6. Model Probability Validity (V2/V3/V4)")
    if not HAVE_SK:
        s.skip("All prediction tests", "sklearn unavailable"); return s

    from features.elo import ELO_COLUMNS, load_elo_features
    from features.online_attack_defense import (AD_COLUMNS, compute_ad_states,
                                                fit_baseline_rates)
    from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
    from models.data import load_supervised_dataset
    from models.poisson import predict_poisson
    from models.v2_artifact import load as load_v2
    from models.v3_artifact import load_v3_artifact
    from models.v4_artifact import load_v4_artifact

    df = select_50()
    f50 = df.fixture_id.tolist()

    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])
    fx = all_fixtures()
    wanted = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted |= set(SEASON_NAME_TO_IDS[sn])
    hist = fx[fx.season_id.isin(wanted) & fx.home_goals.notna()
              & fx.status.isin(["FT", "AWARDED"])]
    base = fit_baseline_rates(hist.home_goals.values.astype(float),
                              hist.away_goals.values.astype(float))
    st = compute_ad_states(fx, E6_LEARNING_RATE, base).set_index("fixture_id")
    for c in AD_COLUMNS:
        X[c] = meta["fixture_id"].map(st[c])

    idx = [int(np.where(meta.fixture_id == f)[0][0]) for f in f50]
    X50 = X.iloc[idx].reset_index(drop=True)

    for name, art, loader in (("V2", V2_ARTIFACT, load_v2),
                              ("V3", V3_ARTIFACT, load_v3_artifact),
                              ("V4", V4_ARTIFACT, load_v4_artifact)):
        a = loader(art)
        E = a.preprocessor.transform(X50[list(a.feature_columns)])
        lh = a.model_home_goals.predict(E)
        la = a.model_away_goals.predict(E)
        s.check(bool(np.all(np.isfinite(lh)) and np.all(np.isfinite(la))),
                f"{name} lambdas finite")
        s.check(bool(np.all(lh > 0) and np.all(la > 0)),
                f"{name} lambdas strictly positive",
                f"min home={lh.min():.4f}")
        pr = predict_poisson(lh, la, list(a.class_order))
        P = np.array([[p.probabilities["H"], p.probabilities["D"],
                       p.probabilities["A"]] for p in pr])
        s.check(bool(np.all(np.isfinite(P))), f"{name} probabilities finite")
        s.check(bool(np.all(P >= 0) and np.all(P <= 1)),
                f"{name} probabilities within [0, 1]")
        d = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
        s.check(d < 1e-9, f"{name} probability rows sum to 1",
                f"max deviation {d:.3e}")
        s.check(P.shape == (50, 3), f"{name} produced 50 predictions",
                f"{P.shape}")
    return s


# --- 7. determinism -----------------------------------------------------

def suite7() -> Suite:
    s = Suite("7. Determinism")
    from features.elo import load_elo_features
    from features.online_attack_defense import (AD_COLUMNS, compute_ad_states,
                                                fit_baseline_rates)
    from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS

    fx = all_fixtures()
    wanted = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted |= set(SEASON_NAME_TO_IDS[sn])
    hist = fx[fx.season_id.isin(wanted) & fx.home_goals.notna()
              & fx.status.isin(["FT", "AWARDED"])]
    b = fit_baseline_rates(hist.home_goals.values.astype(float),
                           hist.away_goals.values.astype(float))
    c = list(AD_COLUMNS)
    a1 = compute_ad_states(fx, E6_LEARNING_RATE, b)
    a2 = compute_ad_states(fx, E6_LEARNING_RATE, b)
    s.check(np.array_equal(a1[c].to_numpy(), a2[c].to_numpy()),
            "A/D states bit-identical on repeat")
    e1 = load_elo_features(MATCHES_DB)
    e2 = load_elo_features(MATCHES_DB)
    s.check(np.array_equal(e1[["home_elo", "away_elo", "elo_diff"]].to_numpy(),
                           e2[["home_elo", "away_elo", "elo_diff"]].to_numpy()),
            "Elo bit-identical on repeat")
    s.check(select_50().fixture_id.tolist() == select_50().fixture_id.tolist(),
            "Fixture selection deterministic")
    return s


# --- 8. integrity -------------------------------------------------------

def suite8() -> Suite:
    s = Suite("8. Protected Files & Frozen 20-Match Outputs")
    for rel, exp in PROTECTED.items():
        p = PROJECT_ROOT / rel
        if not p.exists(): s.bad(Path(rel).name, "missing"); continue
        a = md5(p)
        s.check(a == exp, Path(rel).name,
                f"MD5={a}" if a == exp else f"expected={exp} got={a}")

    # the frozen 20-match outputs must exist and be readable, untouched
    for f in ("v4_20_validation_results.json", "v4_20_validation_report.md",
              "v4_20_validation_manifest.json", "run_v4_20_validation.py",
              "test_v4_20_validation.py"):
        p = HERE / f
        s.check(p.exists(), f"20-match file present: {f}",
                f"{p.stat().st_size} bytes" if p.exists() else "MISSING")
    return s


def main() -> int:
    print("=" * 78)
    print("V4 50-MATCH VALIDATION — PRE-EVALUATION TEST SUITE")
    print("=" * 78)
    print(f"  sklearn available: {HAVE_SK}")
    su = [suite1(), suite2(), suite3(), suite4(), suite5(), suite6(),
          suite7(), suite8()]
    for x in su: x.report()
    tp, tf, ts = (sum(x.p for x in su), sum(x.f for x in su),
                  sum(x.s for x in su))
    print("\n" + "=" * 78 + "\nOVERALL\n" + "=" * 78)
    print(f"  Suites: {len(su)}\n  Pass:   {tp}\n  Fail:   {tf}"
          f"\n  Skip:   {ts}\n  Total:  {tp + tf + ts}")
    print(f"\n  VERDICT: {'ALL TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    if tf:
        print("\n  STOP — do not run run_v4_50_validation.py.")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
