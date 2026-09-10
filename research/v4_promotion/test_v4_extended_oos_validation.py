"""V4 fresh-extended-300 OOS validation — pre-evaluation behavioural tests.

Usage:
    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python research/v4_promotion/test_v4_extended_oos_validation.py

MUST pass with 0 failures before run_v4_extended_oos_validation.py.

Every assertion is BEHAVIOURAL against real data. No string searching, no
assertions about comments or prose.

Causality is asserted using the CORRECTED per-fixture property: for each cut
point, rewrite every 2025/26 outcome AT OR AFTER that fixture's kickoff and
verify the target plus all strictly-earlier frozen fixtures are bit-identical.
The previously rejected incoherent blanket rewrite assertion (rewrite the
whole sample, expect every state unchanged) is NOT used: with a multi-matchday
sample, rewriting early fixtures legitimately propagates to later ones, which
is online learning working rather than leakage.

This file never writes to, or modifies, any frozen artifact or market DB.
"""
from __future__ import annotations

import hashlib
import json
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
EXT_MARKET_DB = HERE / "fresh_extended_market_odds.sqlite"
MARKET_DB_100 = HERE / "fresh_100_market_odds.sqlite"
MARKET_DB_50 = HERE / "promotion_market_odds.sqlite"
FROZEN_IDS = HERE / "fresh_extended_fixture_ids.json"
FROZEN_IDS_100 = HERE / "fresh_100_fixture_ids.json"

V2_ARTIFACT = PROJECT_ROOT / "data/models/v2_poisson_venue.pkl"
V3_ARTIFACT = PROJECT_ROOT / "data/models/v3_poisson_venue_elo_candidate.pkl"
V4_ARTIFACT = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"

PINNED = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
    "research/v4_promotion/promotion_market_odds.sqlite": "f8a41b79cd33afb412ccd9ae2892a196",
    "research/v4_promotion/fresh_100_market_odds.sqlite": "2cb80b79d772fbedd4f3707b39a32c13",
    "research/v4_promotion/fresh_100_fixture_ids.json": "761ad5cc571643e6985e671bd9c3d83a",
    "research/v4_promotion/fresh_extended_fixture_ids.json": "0526bfd6980dd51dae59c6f6aadab2f5",
    "research/v4_promotion/fresh_extended_market_odds.sqlite": "b4889d1791723ea653057af51ca00f8e",
}
REQUIRED_FROZEN_OUTPUTS = [
    "v4_20_validation_results.json", "v4_20_validation_manifest.json",
    "v4_50_validation_results.json", "v4_50_validation_manifest.json",
    "v4_fresh_100_validation_results.json",
    "v4_fresh_100_validation_manifest.json",
]
OPTIONAL_FROZEN_REPORTS = [
    "v4_20_validation_report.md", "v4_50_validation_report.md",
    "v4_fresh_100_validation_report.md",
]
TARGET_COMPS = (200, 419, 423, 477, 499)
E6_LR = 0.02
N = 300
CUTPOINTS = [1, 51, 101, 151, 201, 251, 300]
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
        print(f"\n{'=' * 78}\nSuite: {s.n}\n{'=' * 78}")
        for x in s.l: print(x)
        print(f"\n  Pass: {s.p}  Fail: {s.f}  Skip: {s.s}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()


def frozen_ids() -> list[int]:
    return [int(x) for x in json.loads(
        FROZEN_IDS.read_text(encoding="utf-8"))["fixture_ids"]]


def frozen_ids_100() -> list[int]:
    return [int(x) for x in json.loads(
        FROZEN_IDS_100.read_text(encoding="utf-8"))["fixture_ids"]]


def old_50_ids() -> list[int]:
    from models.config import SEASON_NAME_TO_IDS
    sid = SEASON_NAME_TO_IDS["2025/2026"]
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    q = ",".join("?" * len(sid))
    r = conn.execute(
        f"""SELECT fixture_id FROM fixtures WHERE season_id IN ({q})
            AND status='FT' ORDER BY unix ASC, fixture_id ASC LIMIT 50""",
        list(sid)).fetchall()
    conn.close()
    return [x[0] for x in r]


def fixture_frame(ids: list[int]) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(ids))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, date, competition_name, home_name, away_name,
                   home_goals, away_goals, status, unix
            FROM fixtures WHERE fixture_id IN ({ph})""", conn, params=ids)
    conn.close()
    return df.set_index("fixture_id").loc[ids].reset_index()


def all_fixtures() -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET_COMPS))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                   status, season, season_id, competition_id
            FROM fixtures WHERE competition_id IN ({ph})""",
        conn, params=list(TARGET_COMPS))
    conn.close()
    return df


def ad_baseline(fx: pd.DataFrame):
    from features.online_attack_defense import fit_baseline_rates
    from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
    w = set()
    for s in FINAL_TRAIN_SEASONS:
        w |= set(SEASON_NAME_TO_IDS[s])
    h = fx[fx.season_id.isin(w) & fx.home_goals.notna()
           & fx.status.isin(["FT", "AWARDED"])]
    return fit_baseline_rates(h.home_goals.values.astype(float),
                              h.away_goals.values.astype(float)), h


# --- 1. frozen fixture set ---------------------------------------------

def suite1() -> Suite:
    s = Suite("1. Frozen 300 Fixture Set")
    s.check(FROZEN_IDS.exists(), "Frozen fixture file exists")
    s.check(md5(FROZEN_IDS) == PINNED[
        "research/v4_promotion/fresh_extended_fixture_ids.json"],
        "Frozen fixture file MD5 unchanged", md5(FROZEN_IDS))

    ids = frozen_ids()
    s.check(len(ids) == N, f"Exactly {N} fixture IDs", f"n={len(ids)}")
    s.check(len(set(ids)) == N, "Zero duplicates")

    o50 = old_50_ids()
    i100 = frozen_ids_100()
    s.check(not (set(ids) & set(o50[:20])), "Zero overlap with the frozen 20",
            f"overlap={len(set(ids) & set(o50[:20]))}")
    s.check(not (set(ids) & set(o50)), "Zero overlap with the frozen 50",
            f"overlap={len(set(ids) & set(o50))}")
    s.check(not (set(ids) & set(i100)), "Zero overlap with the fresh 100",
            f"overlap={len(set(ids) & set(i100))}")

    df = fixture_frame(ids)
    s.check(len(df) == N, f"All {N} exist in matches.db", f"{len(df)}/{N}")
    s.check(bool((df.status == "FT").all()), "All status = FT")
    s.check(bool(df.home_goals.notna().all() and df.away_goals.notna().all()),
            "All have valid outcome labels")
    u = df.unix.to_numpy()
    s.check(bool(np.all(u[:-1] <= u[1:])), "Chronological order preserved")
    s.check(df.fixture_id.tolist() == ids,
            "Fixture order matches the frozen list exactly")

    comp = Counter(df.competition_name)
    s.check(sum(comp.values()) == N, f"Composition sums to {N}",
            " · ".join(f"{k} {v}" for k, v in
                       sorted(comp.items(), key=lambda kv: -kv[1])))
    s.check(len(comp) == 5, "All five target leagues represented",
            f"{len(comp)} leagues")
    s.check(min(comp.values()) >= 50,
            "Every league has n>=50 (best per-league coverage so far)",
            f"min n = {min(comp.values())}")

    y = np.where(df.home_goals > df.away_goals, "H",
                 np.where(df.home_goals == df.away_goals, "D", "A"))
    s.check(set(np.unique(y)) == {"H", "D", "A"},
            "All three outcome classes present",
            f"H={int((y=='H').sum())} D={int((y=='D').sum())} "
            f"A={int((y=='A').sum())}")
    return s


# --- 2. artifacts -------------------------------------------------------

def suite2() -> Suite:
    s = Suite("2. Model Artifacts (loaded, never retrained)")
    if not HAVE_SK:
        s.skip("All artifact tests", "sklearn unavailable"); return s

    from models.v2_artifact import load as load_v2
    from models.v3_artifact import load_v3_artifact
    from models.v4_artifact import load_v4_artifact
    from models.v3_contract import V3_FEATURE_COLUMNS

    v2 = load_v2(V2_ARTIFACT)
    s.ok("V2 loads", f"{v2.model_version}, {len(v2.feature_columns)} features")
    s.check(list(v2.class_order) == ["H", "D", "A"], "V2 class order H/D/A")
    s.check(hasattr(v2.preprocessor, "transform")
            and hasattr(v2.model_home_goals, "predict")
            and hasattr(v2.model_away_goals, "predict"),
            "V2 preprocessing and both goal models are callable")

    v3 = load_v3_artifact(V3_ARTIFACT)
    s.check(v3.n_features == 87, "V3 has exactly 87 features", f"{v3.n_features}")
    s.check(tuple(v3.feature_columns) == tuple(V3_FEATURE_COLUMNS),
            "V3 contract matches exactly")

    v4 = load_v4_artifact(V4_ARTIFACT)
    s.check(v4.n_features == 91, "V4 has exactly 91 features", f"{v4.n_features}")
    s.check(tuple(v4.feature_columns[:87]) == tuple(V3_FEATURE_COLUMNS),
            "V4[:87] == V3 contract, exact order")
    s.check(tuple(v4.feature_columns[87:]) ==
            ("A_home", "D_home", "A_away", "D_away"),
            "V4[87:] == the four E6 columns in order")
    s.check(tuple(v4.training_seasons) ==
            ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"),
            "V4 training seasons are exactly 2020/21..2024/25",
            f"{list(v4.training_seasons)}")
    s.check("2025/2026" not in v4.training_seasons,
            "2025/2026 absent from V4 training metadata")
    s.check(v4.holdout_used_in_training is False,
            "V4 holdout_used_in_training is False")
    s.check(md5(V4_ARTIFACT) == PINNED[
        "data/models/v4_poisson_venue_elo_online_ad.pkl"],
        "V4 artifact MD5 unchanged")

    # the 300 sample is strictly out of sample for every arm
    ids = set(frozen_ids())
    from models.config import SEASON_NAME_TO_IDS
    sid = set(SEASON_NAME_TO_IDS["2025/2026"])
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    q = ",".join("?" * len(sid))
    hold = set(r[0] for r in conn.execute(
        f"SELECT fixture_id FROM fixtures WHERE season_id IN ({q})",
        list(sid)))
    conn.close()
    s.check(ids <= hold, "All 300 fixtures belong to the quarantined 2025/26 "
                         "season (strictly out of sample)")
    return s


# --- 3. causal feature availability ------------------------------------

def suite3() -> Suite:
    s = Suite("3. Causal Feature Availability (E1 Elo, E6 A/D)")
    from features.elo import (ELO_COLUMNS, HOME_ADVANTAGE, INIT_RATING,
                              K_FACTOR, MEAN_REVERSION, load_elo_features)
    from features.online_attack_defense import (AD_COLUMNS, EXP_CLIP,
                                                INIT_ATTACK, INIT_DEFENSE,
                                                STATE_CLIP, compute_ad_states)

    s.check(INIT_RATING == 1500.0 and K_FACTOR == 20.0
            and HOME_ADVANTAGE == 100.0 and MEAN_REVERSION == 0.0,
            "E1 parameters unchanged",
            f"init={INIT_RATING} K={K_FACTOR} HA={HOME_ADVANTAGE}")
    s.check(INIT_ATTACK == 0.0 and INIT_DEFENSE == 0.0
            and STATE_CLIP == 1.5 and EXP_CLIP == 3.0,
            "E6 parameters unchanged",
            f"clip=±{STATE_CLIP} exp=±{EXP_CLIP}")

    ids = frozen_ids()
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    s.check(all(i in elo.index for i in ids), f"Elo available {N}/{N}")
    ev = elo.loc[ids, list(ELO_COLUMNS)].to_numpy()
    s.check(bool(np.all(np.isfinite(ev))), "All Elo values finite",
            f"home_elo range [{ev[:,0].min():.1f}, {ev[:,0].max():.1f}]")

    fx = all_fixtures()
    base, hist = ad_baseline(fx)
    s.check("2025/2026" not in set(hist.season),
            "A/D baseline fitted on pre-2025/26 fixtures only",
            f"mu_home={base.mu_home:.6f} n={base.n_train}")
    st = compute_ad_states(fx, E6_LR, base).set_index("fixture_id")
    s.check(all(i in st.index for i in ids), f"A/D available {N}/{N}")
    av = st.loc[ids, list(AD_COLUMNS)].to_numpy()
    s.check(bool(np.all(np.isfinite(av))), "All A/D states finite")
    s.check(bool(np.all(np.abs(av) <= STATE_CLIP + 1e-9)),
            f"All A/D states within ±{STATE_CLIP}",
            f"max |state| = {float(np.max(np.abs(av))):.6f}")

    conn = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    n = conn.execute("SELECT COUNT(*) FROM feature_rows WHERE fixture_id IN "
                     f"({','.join('?' * len(ids))})", ids).fetchone()[0]
    conn.close()
    s.check(n == N, f"Feature rows exist {N}/{N}", f"{n}/{N}")
    return s


# --- 4. multi-cutoff causality -----------------------------------------

def suite4() -> Suite:
    s = Suite("4. Multi-Cutoff Adversarial Causality (corrected property)")
    from features.elo import load_elo_features
    from features.online_attack_defense import AD_COLUMNS, compute_ad_states

    ids = frozen_ids()
    fx = all_fixtures()
    base, _ = ad_baseline(fx)
    cols = list(AD_COLUMNS)
    st = compute_ad_states(fx, E6_LR, base).set_index("fixture_id")
    unix = {int(r.fixture_id): int(r.unix) for r in fx.itertuples()}
    e_base = load_elo_features(MATCHES_DB).set_index("fixture_id")
    ec = ["home_elo", "away_elo", "elo_diff"]

    rng = np.random.default_rng(SEED)
    import shutil, tempfile
    ad_ok = elo_ok = True
    worst_ad = worst_elo = 0.0
    detail = []

    for k in CUTPOINTS:
        tgt = ids[k - 1]
        cut = unix[tgt]
        # every frozen fixture strictly earlier, plus the target itself
        earlier = [i for i in ids if unix[i] < cut] + [tgt]

        adv = fx.copy()
        m = (adv.season == "2025/2026") & (adv.unix >= cut)
        adv.loc[m, "home_goals"] = rng.integers(0, 9, int(m.sum())).astype(float)
        adv.loc[m, "away_goals"] = rng.integers(0, 9, int(m.sum())).astype(float)
        a = compute_ad_states(adv, E6_LR, base).set_index("fixture_id")
        d = float(np.max(np.abs(st.loc[earlier, cols].to_numpy()
                                - a.loc[earlier, cols].to_numpy())))
        worst_ad = max(worst_ad, d); ad_ok &= (d == 0.0)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "m.db"
            shutil.copy2(MATCHES_DB, tmp)
            c = sqlite3.connect(str(tmp))
            c.execute("UPDATE fixtures SET home_goals=7, away_goals=1 "
                      "WHERE season='2025/2026' AND unix >= ?", (cut,))
            c.commit(); c.close()
            e_adv = load_elo_features(tmp).set_index("fixture_id")
        de = float(np.max(np.abs(e_base.loc[earlier, ec].to_numpy()
                                 - e_adv.loc[earlier, ec].to_numpy())))
        worst_elo = max(worst_elo, de); elo_ok &= (de == 0.0)
        detail.append(f"#{k}: rewrote {int(m.sum())} at/after, "
                      f"{len(earlier)} protected, A/D {d:.1e} Elo {de:.1e}")

    for line in detail:
        s.ok("Cut-point causality detail", line)
    s.check(ad_ok, "A/D: target and all earlier fixtures immune to at-or-later "
                   "outcome rewrites",
            f"cut points {CUTPOINTS}, max diff {worst_ad:.3e}")
    s.check(elo_ok, "Elo: target and all earlier fixtures immune to at-or-later "
                    "outcome rewrites",
            f"cut points {CUTPOINTS}, max diff {worst_elo:.3e}")

    # own-outcome invariance
    tgt = ids[150]
    mut = fx.copy()
    mut.loc[mut.fixture_id == tgt, ["home_goals", "away_goals"]] = [9.0, 0.0]
    a = compute_ad_states(mut, E6_LR, base).set_index("fixture_id")
    s.check(np.array_equal(st.loc[tgt, cols].to_numpy(),
                           a.loc[tgt, cols].to_numpy()),
            "A fixture's own outcome cannot change its own pre-match state",
            f"fixture {tgt} rewritten to 9-0")

    # simultaneous kickoffs
    df = fixture_frame(ids)
    dup = df.unix.value_counts()
    s.check(int(dup.max()) > 1,
            "Sample contains simultaneous kickoffs (two-pass load-bearing)",
            f"max {int(dup.max())} fixtures share a kickoff time")
    busiest = dup.idxmax()
    grp = df[df.unix == busiest].fixture_id.tolist()
    victim = grp[1]
    mut2 = fx.copy()
    mut2.loc[mut2.fixture_id == grp[0], ["home_goals", "away_goals"]] = [8.0, 0.0]
    a2 = compute_ad_states(mut2, E6_LR, base).set_index("fixture_id")
    s.check(np.array_equal(st.loc[victim, cols].to_numpy(),
                           a2.loc[victim, cols].to_numpy()),
            "Simultaneous fixture unaffected by its sibling's rewritten score",
            f"changed {grp[0]}; {victim} bit-identical")
    return s


# --- 5. extended market -------------------------------------------------

def suite5() -> Suite:
    s = Suite("5. Extended Market Reference (300/300, isolated)")
    s.check(EXT_MARKET_DB.exists(), "Extended market DB exists")
    s.check(md5(EXT_MARKET_DB) == PINNED[
        "research/v4_promotion/fresh_extended_market_odds.sqlite"],
        "Extended market DB MD5 unchanged", md5(EXT_MARKET_DB))

    ids = frozen_ids()
    conn = sqlite3.connect(f"file:{EXT_MARKET_DB}?mode=ro", uri=True)
    mk = pd.read_sql_query(
        """SELECT fixture_id, p_home, p_draw, p_away, closing_home,
                  closing_draw, closing_away, bookmaker_name, price_class,
                  data_class, market_key, market_id
           FROM fresh_extended_market""", conn)
    cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(fresh_extended_market)")]
    conn.close()

    s.check(len(mk) == N, f"Exactly {N} rows", f"n={len(mk)}")
    s.check(set(mk.fixture_id) == set(ids), "Exact fixture ID match")
    s.check(not [i for i in ids if i not in set(mk.fixture_id)], "Zero missing")
    s.check(not (set(mk.fixture_id) - set(ids)), "Zero extra")
    s.check(set(mk.bookmaker_name) == {"Pinnacle"}, "Pinnacle only")
    s.check(set(mk.price_class) == {"closing"}, "Closing only")
    s.check(set(mk.market_key) == {"ft_result"} and set(mk.market_id) == {6},
            "1X2 (ft_result) only")
    s.check(set(mk.data_class) == {"promotion-validation-only"},
            "data_class = promotion-validation-only")

    P = mk[["p_home", "p_draw", "p_away"]].to_numpy(float)
    s.check(bool(np.all(np.isfinite(P))), "Probabilities finite")
    s.check(bool(np.all(P >= 0)), "Probabilities >= 0")
    s.check(bool(np.all(P <= 1)), "Probabilities <= 1")
    worst = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
    s.check(worst < 1e-9, "Rows sum to 1", f"max deviation {worst:.3e}")

    O = mk[["closing_home", "closing_draw", "closing_away"]].to_numpy(float)
    s.check(bool(np.all(O > 1.0)), "All closing odds > 1.0 (valid decimal)",
            f"min {O.min():.3f}")
    over = (1 / O).sum(axis=1)
    s.check(bool(np.all(over > 1.0)),
            "All raw overrounds > 1 (de-vig genuinely applied)",
            f"mean overround {over.mean():.4f}")

    s.check(not ({"home_goals", "away_goals", "label_result", "result",
                  "winning_team", "score"} & set(cols)),
            "No outcome column in the market table",
            f"{len(cols)} columns")

    from models.v4_contract import V4_FEATURE_COLUMNS, FORBIDDEN_MARKET_COLUMNS
    s.check(not (set(V4_FEATURE_COLUMNS) & set(mk.columns)),
            "No market table column appears in the V4 contract")
    s.check(not (set(V4_FEATURE_COLUMNS) & FORBIDDEN_MARKET_COLUMNS),
            "V4 contract holds no forbidden market column")

    # disjoint from BOTH prior market databases
    conn = sqlite3.connect(f"file:{MARKET_DB_50}?mode=ro", uri=True)
    old50 = set(r[0] for r in conn.execute(
        "SELECT fixture_id FROM promotion_market"))
    conn.close()
    conn = sqlite3.connect(f"file:{MARKET_DB_100}?mode=ro", uri=True)
    old100 = set(r[0] for r in conn.execute(
        "SELECT fixture_id FROM fresh_100_market"))
    conn.close()
    s.check(not (set(mk.fixture_id) & old50),
            "Extended market DB disjoint from the 50-match market DB",
            f"old={len(old50)} rows, overlap=0")
    s.check(not (set(mk.fixture_id) & old100),
            "Extended market DB disjoint from the fresh-100 market DB",
            f"old={len(old100)} rows, overlap=0")
    return s


# --- 6. model probability validity -------------------------------------

def suite6() -> Suite:
    s = Suite("6. Model Probability Validity (V2/V3/V4)")
    if not HAVE_SK:
        s.skip("All prediction tests", "sklearn unavailable"); return s

    from features.elo import ELO_COLUMNS, load_elo_features
    from features.online_attack_defense import AD_COLUMNS, compute_ad_states
    from models.data import load_supervised_dataset
    from models.poisson import predict_poisson
    from models.v2_artifact import load as load_v2
    from models.v3_artifact import load_v3_artifact
    from models.v4_artifact import load_v4_artifact

    ids = frozen_ids()
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])
    fx = all_fixtures()
    base, _ = ad_baseline(fx)
    st = compute_ad_states(fx, E6_LR, base).set_index("fixture_id")
    for c in AD_COLUMNS:
        X[c] = meta["fixture_id"].map(st[c])
    idx = [int(np.where(meta.fixture_id == f)[0][0]) for f in ids]
    XS = X.iloc[idx].reset_index(drop=True)

    for nm, path, loader in (("V2", V2_ARTIFACT, load_v2),
                             ("V3", V3_ARTIFACT, load_v3_artifact),
                             ("V4", V4_ARTIFACT, load_v4_artifact)):
        a = loader(path)
        E = a.preprocessor.transform(XS[list(a.feature_columns)])
        lh, la = a.model_home_goals.predict(E), a.model_away_goals.predict(E)
        s.check(bool(np.all(np.isfinite(lh)) and np.all(np.isfinite(la))),
                f"{nm} lambdas finite")
        s.check(bool(np.all(lh > 0) and np.all(la > 0)),
                f"{nm} lambdas strictly positive",
                f"min home {lh.min():.4f}, min away {la.min():.4f}")
        pr = predict_poisson(lh, la, list(a.class_order))
        P = np.array([[p.probabilities["H"], p.probabilities["D"],
                       p.probabilities["A"]] for p in pr])
        s.check(P.shape == (N, 3), f"{nm} produced {N} predictions", f"{P.shape}")
        s.check(bool(np.all(np.isfinite(P))), f"{nm} probabilities finite")
        s.check(bool(np.all(P >= 0) and np.all(P <= 1)),
                f"{nm} probabilities within [0,1]")
        d = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
        s.check(d < 1e-9, f"{nm} rows sum to 1", f"max deviation {d:.3e}")
    return s


# --- 7. determinism -----------------------------------------------------

def suite7() -> Suite:
    s = Suite("7. Determinism")
    from features.elo import load_elo_features
    from features.online_attack_defense import AD_COLUMNS, compute_ad_states

    fx = all_fixtures()
    base, _ = ad_baseline(fx)
    c = list(AD_COLUMNS)
    a1 = compute_ad_states(fx, E6_LR, base)
    a2 = compute_ad_states(fx, E6_LR, base)
    s.check(np.array_equal(a1[c].to_numpy(), a2[c].to_numpy()),
            "A/D states bit-identical on repeat")
    e1 = load_elo_features(MATCHES_DB)
    e2 = load_elo_features(MATCHES_DB)
    s.check(np.array_equal(e1[["home_elo", "away_elo", "elo_diff"]].to_numpy(),
                           e2[["home_elo", "away_elo", "elo_diff"]].to_numpy()),
            "Elo bit-identical on repeat")
    s.check(frozen_ids() == frozen_ids(), "Frozen fixture list stable on reload")

    conn = sqlite3.connect(f"file:{EXT_MARKET_DB}?mode=ro", uri=True)
    m1 = pd.read_sql_query("SELECT fixture_id,p_home,p_draw,p_away "
                           "FROM fresh_extended_market ORDER BY fixture_id", conn)
    m2 = pd.read_sql_query("SELECT fixture_id,p_home,p_draw,p_away "
                           "FROM fresh_extended_market ORDER BY fixture_id", conn)
    conn.close()
    s.check(m1.equals(m2), "Market probabilities stable on re-read")
    return s


# --- 8. integrity -------------------------------------------------------

def suite8() -> Suite:
    s = Suite("8. Protected Files & Frozen Artifacts")
    for rel, exp in PINNED.items():
        p = PROJECT_ROOT / rel
        if not p.exists(): s.bad(Path(rel).name, "missing"); continue
        a = md5(p)
        s.check(a == exp, Path(rel).name,
                f"MD5={a}" if a == exp else f"expected={exp} got={a}")

    # Load-bearing artifacts: the longitudinal phase reads the RESULTS JSONs,
    # so those and the manifests are required. The 50-match .md was declared
    # in that harness but never written -- a known cosmetic gap, with the data
    # itself intact in the JSON. Asserted separately so a missing report cannot
    # masquerade as missing results.
    for f in REQUIRED_FROZEN_OUTPUTS:
        p = HERE / f
        s.check(p.exists(), f"Required frozen artifact present: {f}",
                f"{p.stat().st_size} bytes" if p.exists() else "MISSING")

    for f in OPTIONAL_FROZEN_REPORTS:
        p = HERE / f
        (s.ok if p.exists() else s.skip)(
            f"Optional frozen report: {f}",
            f"{p.stat().st_size} bytes" if p.exists()
            else "not generated by that harness; data intact in results JSON")

    for f in ("v4_20_validation_results.json",
              "v4_50_validation_results.json",
              "v4_fresh_100_validation_results.json"):
        p = HERE / f
        if not p.exists():
            s.bad(f"{f} readable"); continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            has = any(k in d for k in ("scorecard", "metrics", "summary"))
            s.check(has, f"{f} carries a scorecard usable by the "
                         f"longitudinal phase", f"top-level keys: {len(d)}")
        except Exception as exc:                              # noqa: BLE001
            s.bad(f"{f} parses as JSON", str(exc))

    s.check(MARKET_DB_50.exists(), "50-match market DB present")
    s.check(MARKET_DB_100.exists(), "fresh-100 market DB present")
    return s


def main() -> int:
    print("=" * 78)
    print("V4 FRESH-EXTENDED-300 OOS VALIDATION — PRE-EVALUATION TEST SUITE")
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
        print("\n  STOP — do not run run_v4_extended_oos_validation.py.")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
