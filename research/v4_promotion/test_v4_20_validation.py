"""V4 20-Match Validation Harness — Behavioral Test Suite.

Validates the integrity, contracts, causality, determinism, and data separation
for the 20-match evaluation of V2, V3, V4 and Pinnacle Market.

Usage:
    python research/v4_promotion/test_v4_20_validation.py
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "market_odds"))

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
PROMO_DB = HERE / "promotion_market_odds.sqlite"
V2_ARTIFACT = PROJECT_ROOT / "data" / "models" / "v2_poisson_venue.pkl"
V3_ARTIFACT = PROJECT_ROOT / "data" / "models" / "v3_poisson_venue_elo_candidate.pkl"
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
EXPECTED_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"

EXPECTED_20_FIXTURE_IDS = [
    343465962, 343465735, 342254747, 343465963, 342863654,
    342863842, 342863843, 342863844, 343465733, 342864234,
    343465729, 342864255, 343465732, 342863998, 342864289,
    343465726, 343465727, 343465734, 343465842, 343465728,
]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(8192), b""):
            h.update(c)
    return h.hexdigest()


class Suite:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.logs = []

    def ok(self, msg: str, detail: str = ""):
        self.passed += 1
        self.logs.append(f"  PASS: {msg}" + (f" — {detail}" if detail else ""))

    def bad(self, msg: str, detail: str = ""):
        self.failed += 1
        self.logs.append(f"  FAIL: {msg}" + (f" — {detail}" if detail else ""))

    def skip(self, msg: str, detail: str = ""):
        self.skipped += 1
        self.logs.append(f"  SKIP: {msg}" + (f" — {detail}" if detail else ""))

    def check(self, cond: bool, msg: str, detail: str = ""):
        if cond:
            self.ok(msg, detail)
        else:
            self.bad(msg, detail)

    def report(self):
        print(f"\n{'=' * 74}\nSuite: {self.name}\n{'=' * 74}")
        for x in self.logs:
            print(x)
        print(f"\n  Pass: {self.passed}  Fail: {self.failed}  Skip: {self.skipped}")


# --- Suite 1: Protected Files & Artifact Hash Integrity ---
def test_protected_integrity() -> Suite:
    s = Suite("1. Protected Files & Artifact Hash Integrity")
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            s.bad(f"Missing file {rel}")
            continue
        act = md5(p)
        s.check(act == exp, f"Protected file {Path(rel).name} hash matches", f"MD5={act}")

    for rel, exp in E2_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            s.bad(f"Missing E2 file {rel}")
            continue
        act = md5(p)
        s.check(act == exp, f"E2 original {Path(rel).name} hash matches", f"MD5={act}")

    if not V4_ARTIFACT.exists():
        s.bad("V4 candidate artifact exists", str(V4_ARTIFACT))
    else:
        v4_act = md5(V4_ARTIFACT)
        s.check(v4_act == EXPECTED_V4_MD5, "V4 candidate artifact MD5 matches exact build", f"MD5={v4_act}")

    return s


# --- Suite 2: Exact 20 Fixture Verification ---
def test_fixture_selection() -> Suite:
    s = Suite("2. Exact 20 Fixture Selection & Attributes")
    from models.config import SEASON_NAME_TO_IDS
    ids = SEASON_NAME_TO_IDS["2025/2026"]
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    q = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT fixture_id, unix, home_name, away_name, competition_id, home_goals, away_goals, status
            FROM fixtures
            WHERE season_id IN ({q}) AND status='FT'
            ORDER BY unix ASC, fixture_id ASC
            LIMIT 20""",
        list(ids),
    ).fetchall()
    conn.close()

    fids = [r[0] for r in rows]
    s.check(len(fids) == 20, "Deterministic query returns exactly 20 fixtures", f"n={len(fids)}")
    s.check(fids == EXPECTED_20_FIXTURE_IDS, "Fixture IDs match registered list in exact order")
    s.check(len(set(fids)) == 20, "No duplicate fixture IDs among the 20")
    s.check(all(r[7] == "FT" for r in rows), "All 20 fixtures have status == 'FT'")
    s.check(all(r[5] is not None and r[6] is not None for r in rows), "All 20 fixtures have valid goal counts")
    s.check(all(r[5] >= 0 and r[6] >= 0 for r in rows), "All 20 fixtures have non-negative goal counts")

    # League validation (only 200, 419, 423)
    comp_ids = {r[4] for r in rows}
    s.check(comp_ids.issubset({200, 419, 423}), "Competitions restricted to Ligue 1 (200), La Liga (419), Premier League (423)", f"Found: {comp_ids}")

    return s


# --- Suite 3: Model Artifact Verification ---
def test_model_artifacts() -> Suite:
    s = Suite("3. Model Artifact Verification (V2, V3, V4)")
    from models.v2_artifact import load as load_v2
    from models.v3_artifact import load_v3_artifact
    from models.v4_artifact import load_v4_artifact
    from models.v3_contract import V3_FEATURE_COLUMNS
    from models.v4_contract import V4_FEATURE_COLUMNS, V4_AD_COLUMNS

    v2 = load_v2()
    s.check(v2.n_features == 84, "V2 artifact declares 84 features", f"n={v2.n_features}")
    s.check(v2.class_order == ["H", "D", "A"], "V2 class order is H/D/A")
    s.check(hasattr(v2.model_home_goals, "predict") and hasattr(v2.model_away_goals, "predict"), "V2 models have predict()")

    v3 = load_v3_artifact(V3_ARTIFACT)
    s.check(v3.n_features == 87, "V3 artifact declares 87 features", f"n={v3.n_features}")
    s.check(tuple(v3.feature_columns) == tuple(V3_FEATURE_COLUMNS), "V3 feature contract matches V3_FEATURE_COLUMNS")
    s.check(v3.class_order == ["H", "D", "A"], "V3 class order is H/D/A")

    v4 = load_v4_artifact(V4_ARTIFACT)
    s.check(v4.n_features == 91, "V4 artifact declares 91 features", f"n={v4.n_features}")
    s.check(tuple(v4.feature_columns[:87]) == tuple(V3_FEATURE_COLUMNS), "V4[:87] preserves exact V3 contract")
    s.check(tuple(v4.feature_columns[87:]) == V4_AD_COLUMNS, "V4[87:] matches E6 online A/D columns in order")
    s.check(v4.class_order == ["H", "D", "A"], "V4 class order is H/D/A")
    s.check(v4.holdout_used_in_training is False, "V4 holdout_used_in_training is False")
    s.check("2025/2026" not in v4.training_seasons, "2025/2026 absent from V4 training seasons", f"{v4.training_seasons}")

    return s


# --- Suite 4: Causal Feature Construction ---
def test_causal_features() -> Suite:
    s = Suite("4. Pre-Kickoff Causal Feature Construction (Elo & Online A/D)")
    from features.elo import load_elo_features, ELO_COLUMNS
    from features.online_attack_defense import fit_baseline_rates, compute_ad_states, AD_COLUMNS, STATE_CLIP
    from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
    from models.data import load_supervised_dataset

    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    f20_set = set(EXPECTED_20_FIXTURE_IDS)
    f20_meta = meta[meta["fixture_id"].isin(f20_set)]
    s.check(len(f20_meta) == 20, "All 20 fixtures have base feature rows in features.db")

    # Elo
    elo_df = load_elo_features(MATCHES_DB)
    elo_map = elo_df.set_index("fixture_id")
    s.check(all(fid in elo_map.index for fid in EXPECTED_20_FIXTURE_IDS), "Causal Elo available for all 20 fixtures")

    # A/D
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query(
        "SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals, status, season, season_id, competition_id "
        "FROM fixtures WHERE competition_id IN (200,419,423,477,499)",
        conn,
    )
    conn.close()
    wanted = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted |= set(SEASON_NAME_TO_IDS[sn])
    hist = fx[fx.season_id.isin(wanted) & fx.home_goals.notna() & fx.status.isin(["FT", "AWARDED"])]
    base = fit_baseline_rates(hist.home_goals.values.astype(float), hist.away_goals.values.astype(float))
    states = compute_ad_states(fx, 0.02, base).set_index("fixture_id")

    s.check(all(fid in states.index for fid in EXPECTED_20_FIXTURE_IDS), "Causal Online A/D available for all 20 fixtures")
    sub = states.loc[EXPECTED_20_FIXTURE_IDS, list(AD_COLUMNS)].to_numpy(float)
    s.check(bool(np.all(np.abs(sub) <= STATE_CLIP + 1e-9)), f"All 20 A/D states within ±{STATE_CLIP}", f"max={float(np.max(np.abs(sub))):.4f}")

    return s


# --- Suite 5: Adversarial Score Rewrite & Insulation ---
def test_adversarial_insulation() -> Suite:
    s = Suite("5. Adversarial Score Rewrite & Causality Insulation")
    from features.elo import load_elo_features
    from features.online_attack_defense import fit_baseline_rates, compute_ad_states, AD_COLUMNS
    from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query(
        "SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals, status, season, season_id, competition_id "
        "FROM fixtures WHERE competition_id IN (200,419,423,477,499)",
        conn,
    )
    conn.close()

    wanted = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted |= set(SEASON_NAME_TO_IDS[sn])
    hist = fx[fx.season_id.isin(wanted) & fx.home_goals.notna() & fx.status.isin(["FT", "AWARDED"])]
    base = fit_baseline_rates(hist.home_goals.values.astype(float), hist.away_goals.values.astype(float))

    st_clean = compute_ad_states(fx, 0.02, base).set_index("fixture_id")

    # Mutate 2025/26 fixture scores wildly
    fx_mut = fx.copy()
    mask_2025 = fx_mut["season"] == "2025/2026"
    fx_mut.loc[mask_2025, "home_goals"] = 8.0
    fx_mut.loc[mask_2025, "away_goals"] = 0.0
    st_mut = compute_ad_states(fx_mut, 0.02, base).set_index("fixture_id")

    # The 20 matches' states at kickoff must be unaffected by future outcomes
    cols = list(AD_COLUMNS)
    diff_first4 = np.max(np.abs(st_clean.loc[EXPECTED_20_FIXTURE_IDS[:4], cols].to_numpy() - st_mut.loc[EXPECTED_20_FIXTURE_IDS[:4], cols].to_numpy()))
    s.check(diff_first4 == 0.0, "First-wave 2025/26 fixtures have bit-identical pre-match states despite score manipulation")

    # Elo rewrite test in temp database
    with tempfile.TemporaryDirectory() as td:
        tmp_db = Path(td) / "matches_temp.db"
        import shutil
        shutil.copy2(MATCHES_DB, tmp_db)
        c = sqlite3.connect(str(tmp_db))
        c.execute("UPDATE fixtures SET home_goals = 10, away_goals = 0 WHERE season = '2025/2026'")
        c.commit()
        c.close()
        elo_clean = load_elo_features(MATCHES_DB).set_index("fixture_id")
        elo_mut = load_elo_features(tmp_db).set_index("fixture_id")

    elo_cols = ["home_elo", "away_elo", "elo_diff"]
    elo_diff = np.max(np.abs(elo_clean.loc[EXPECTED_20_FIXTURE_IDS, elo_cols].to_numpy() - elo_mut.loc[EXPECTED_20_FIXTURE_IDS, elo_cols].to_numpy()))
    s.check(elo_diff == 0.0, "Elo pre-match values for all 20 fixtures are bit-identical when 2025/26 scores are altered")

    return s


# --- Suite 6: Market Reference Database Verification ---
def test_market_reference() -> Suite:
    s = Suite("6. Market Reference Separation & Coverage")
    from models.v4_contract import V4_FEATURE_COLUMNS, FORBIDDEN_MARKET_COLUMNS

    s.check(PROMO_DB.exists(), "promotion_market_odds.sqlite exists", str(PROMO_DB))
    conn = sqlite3.connect(f"file:{PROMO_DB}?mode=ro", uri=True)
    rows = conn.execute("SELECT fixture_id, p_home, p_draw, p_away, data_class, price_class, bookmaker_name FROM promotion_market").fetchall()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(promotion_market)")]
    conn.close()

    s.check(len(rows) == 20, "Market database covers exactly 20/20 fixtures", f"n={len(rows)}")
    market_fids = [r[0] for r in rows]
    s.check(set(market_fids) == set(EXPECTED_20_FIXTURE_IDS), "Market fixture IDs match the exact 20 fixtures")

    max_dev = max(abs(r[1] + r[2] + r[3] - 1.0) for r in rows)
    s.check(max_dev < 1e-9, "Market probabilities sum to 1 within 1e-9", f"max dev={max_dev:.2e}")
    s.check(all(r[4] == "promotion-validation-only" for r in rows), "All market rows tagged 'promotion-validation-only'")
    s.check(all(r[5] == "closing" for r in rows), "All market rows are closing prices")
    s.check(all(r[6] == "Pinnacle" for r in rows), "All market rows from Pinnacle")

    forbidden_cols = {"home_goals", "away_goals", "label_result", "result", "winner"}
    s.check(not (forbidden_cols & set(cols)), "Market table contains NO outcome columns", f"cols: {cols}")

    overlap = set(V4_FEATURE_COLUMNS) & FORBIDDEN_MARKET_COLUMNS
    s.check(not overlap, "V4 contract has NO market columns", f"{len(FORBIDDEN_MARKET_COLUMNS)} forbidden names verified")

    return s


# --- Suite 7: Model Predictions & Mathematical Validity ---
def test_predictions_and_metrics() -> Suite:
    s = Suite("7. Model Predictions & Metric Robustness")
    from models.v2_artifact import load as load_v2
    from models.v3_artifact import load_v3_artifact
    from models.v4_artifact import load_v4_artifact
    from features.elo import load_elo_features, ELO_COLUMNS
    from features.online_attack_defense import fit_baseline_rates, compute_ad_states, AD_COLUMNS
    from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
    from models.data import load_supervised_dataset
    from models.poisson import predict_poisson
    from e2_metrics import all_metrics, log_loss, brier_score, rps, accuracy, draw_recall, expected_calibration_error

    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X_all = ds.X.reset_index(drop=True)

    elo_df = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for col in ELO_COLUMNS:
        X_all[col] = meta["fixture_id"].map(elo_df[col])

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query(
        "SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals, status, season, season_id, competition_id "
        "FROM fixtures WHERE competition_id IN (200,419,423,477,499)",
        conn,
    )
    conn.close()
    wanted = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted |= set(SEASON_NAME_TO_IDS[sn])
    hist = fx[fx.season_id.isin(wanted) & fx.home_goals.notna() & fx.status.isin(["FT", "AWARDED"])]
    base = fit_baseline_rates(hist.home_goals.values.astype(float), hist.away_goals.values.astype(float))
    states = compute_ad_states(fx, 0.02, base).set_index("fixture_id")
    for col in AD_COLUMNS:
        X_all[col] = meta["fixture_id"].map(states[col])

    f20_indices = [np.where(meta["fixture_id"] == fid)[0][0] for fid in EXPECTED_20_FIXTURE_IDS]
    X_20 = X_all.iloc[f20_indices].copy().reset_index(drop=True)

    # V2
    v2 = load_v2()
    E_v2 = v2.preprocessor.transform(X_20[list(v2.feature_columns)])
    lh2, la2 = v2.model_home_goals.predict(E_v2), v2.model_away_goals.predict(E_v2)
    p2_preds = predict_poisson(lh2, la2, v2.class_order)
    p2 = np.array([[p.probabilities["H"], p.probabilities["D"], p.probabilities["A"]] for p in p2_preds])

    # V3
    v3 = load_v3_artifact(V3_ARTIFACT)
    E_v3 = v3.preprocessor.transform(X_20[list(v3.feature_columns)])
    lh3, la3 = v3.model_home_goals.predict(E_v3), v3.model_away_goals.predict(E_v3)
    p3_preds = predict_poisson(lh3, la3, v3.class_order)
    p3 = np.array([[p.probabilities["H"], p.probabilities["D"], p.probabilities["A"]] for p in p3_preds])

    # V4
    v4 = load_v4_artifact(V4_ARTIFACT)
    E_v4 = v4.preprocessor.transform(X_20[list(v4.feature_columns)])
    lh4, la4 = v4.model_home_goals.predict(E_v4), v4.model_away_goals.predict(E_v4)
    p4_preds = predict_poisson(lh4, la4, v4.class_order)
    p4 = np.array([[p.probabilities["H"], p.probabilities["D"], p.probabilities["A"]] for p in p4_preds])

    for name, p_mat in [("V2", p2), ("V3", p3), ("V4", p4)]:
        s.check(p_mat.shape == (20, 3), f"{name} probability matrix has shape (20, 3)")
        s.check(bool(np.all(np.isfinite(p_mat))), f"{name} probabilities are all finite")
        s.check(bool(np.all(p_mat >= 0.0) and np.all(p_mat <= 1.0)), f"{name} probabilities within [0, 1]")
        row_sums = p_mat.sum(axis=1)
        s.check(bool(np.allclose(row_sums, 1.0, atol=1e-6)), f"{name} probability row sums equal 1.0 within 1e-6")

    # Metric functions sanity
    y_test = np.array(["H", "D", "A", "H"])
    p_test = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6], [0.8, 0.1, 0.1]])
    m = all_metrics(y_test, p_test)
    s.check(m["accuracy"] == 1.0, "Accuracy computation 100% on perfect predictions")
    s.check(m["log_loss"] < 0.6, "Log loss finite and low on high-confidence correct predictions")
    s.check(m["rps"] >= 0.0, "RPS non-negative")

    return s


# --- Suite 8: Determinism Verification ---
def test_determinism() -> Suite:
    s = Suite("8. Determinism (Repeated Pipeline Execution)")
    from models.v2_artifact import load as load_v2
    from models.v3_artifact import load_v3_artifact
    from models.v4_artifact import load_v4_artifact
    from features.elo import load_elo_features, ELO_COLUMNS
    from features.online_attack_defense import fit_baseline_rates, compute_ad_states, AD_COLUMNS
    from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS
    from models.data import load_supervised_dataset
    from models.poisson import predict_poisson

    def run_pass():
        ds = load_supervised_dataset(FEATURES_DB)
        meta = ds.metadata.reset_index(drop=True)
        X_all = ds.X.reset_index(drop=True)
        elo_df = load_elo_features(MATCHES_DB).set_index("fixture_id")
        for col in ELO_COLUMNS:
            X_all[col] = meta["fixture_id"].map(elo_df[col])
        conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
        fx = pd.read_sql_query(
            "SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals, status, season, season_id, competition_id "
            "FROM fixtures WHERE competition_id IN (200,419,423,477,499)",
            conn,
        )
        conn.close()
        wanted = set()
        for sn in FINAL_TRAIN_SEASONS:
            wanted |= set(SEASON_NAME_TO_IDS[sn])
        hist = fx[fx.season_id.isin(wanted) & fx.home_goals.notna() & fx.status.isin(["FT", "AWARDED"])]
        base = fit_baseline_rates(hist.home_goals.values.astype(float), hist.away_goals.values.astype(float))
        states = compute_ad_states(fx, 0.02, base).set_index("fixture_id")
        for col in AD_COLUMNS:
            X_all[col] = meta["fixture_id"].map(states[col])

        f20_indices = [np.where(meta["fixture_id"] == fid)[0][0] for fid in EXPECTED_20_FIXTURE_IDS]
        X_20 = X_all.iloc[f20_indices].copy().reset_index(drop=True)

        v2 = load_v2()
        E_v2 = v2.preprocessor.transform(X_20[list(v2.feature_columns)])
        p2 = predict_poisson(v2.model_home_goals.predict(E_v2), v2.model_away_goals.predict(E_v2), v2.class_order)

        v3 = load_v3_artifact(V3_ARTIFACT)
        E_v3 = v3.preprocessor.transform(X_20[list(v3.feature_columns)])
        p3 = predict_poisson(v3.model_home_goals.predict(E_v3), v3.model_away_goals.predict(E_v3), v3.class_order)

        v4 = load_v4_artifact(V4_ARTIFACT)
        E_v4 = v4.preprocessor.transform(X_20[list(v4.feature_columns)])
        p4 = predict_poisson(v4.model_home_goals.predict(E_v4), v4.model_away_goals.predict(E_v4), v4.class_order)

        conn = sqlite3.connect(f"file:{PROMO_DB}?mode=ro", uri=True)
        mkt = pd.read_sql_query("SELECT fixture_id, p_home, p_draw, p_away FROM promotion_market", conn).set_index("fixture_id")
        conn.close()

        p2_arr = np.array([[p.probabilities["H"], p.probabilities["D"], p.probabilities["A"]] for p in p2])
        p3_arr = np.array([[p.probabilities["H"], p.probabilities["D"], p.probabilities["A"]] for p in p3])
        p4_arr = np.array([[p.probabilities["H"], p.probabilities["D"], p.probabilities["A"]] for p in p4])
        pmkt_arr = np.array([[mkt.loc[fid, "p_home"], mkt.loc[fid, "p_draw"], mkt.loc[fid, "p_away"]] for fid in EXPECTED_20_FIXTURE_IDS])

        return p2_arr, p3_arr, p4_arr, pmkt_arr

    p2_a, p3_a, p4_a, pm_a = run_pass()
    p2_b, p3_b, p4_b, pm_b = run_pass()

    s.check(float(np.max(np.abs(p2_a - p2_b))) == 0.0, "V2 predictions bit-identical on repeat (diff = 0.000e+00)")
    s.check(float(np.max(np.abs(p3_a - p3_b))) == 0.0, "V3 predictions bit-identical on repeat (diff = 0.000e+00)")
    s.check(float(np.max(np.abs(p4_a - p4_b))) == 0.0, "V4 predictions bit-identical on repeat (diff = 0.000e+00)")
    s.check(float(np.max(np.abs(pm_a - pm_b))) == 0.0, "Market reference values bit-identical on repeat (diff = 0.000e+00)")

    return s


def main() -> int:
    print("=" * 74)
    print("V4 20-Match Validation Harness — Test Suite")
    print("=" * 74)

    suites = [
        test_protected_integrity(),
        test_fixture_selection(),
        test_model_artifacts(),
        test_causal_features(),
        test_adversarial_insulation(),
        test_market_reference(),
        test_predictions_and_metrics(),
        test_determinism(),
    ]

    for su in suites:
        su.report()

    total_p = sum(s.passed for s in suites)
    total_f = sum(s.failed for s in suites)
    total_s = sum(s.skipped for s in suites)

    print("\n" + "=" * 74 + "\nOVERALL SUITE SUMMARY\n" + "=" * 74)
    print(f"  Suites: {len(suites)}")
    print(f"  Pass:   {total_p}")
    print(f"  Fail:   {total_f}")
    print(f"  Skip:   {total_s}")
    print(f"  Total:  {total_p + total_f + total_s}")
    print(f"\n  VERDICT: {'ALL TESTS PASSED' if total_f == 0 else 'TESTS FAILED'}")

    return 1 if total_f > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
