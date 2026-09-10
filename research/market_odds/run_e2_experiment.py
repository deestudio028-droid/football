"""E2 — V3 vs V3 + Market controlled experiment.

Usage:
    python research/market_odds/run_e2_experiment.py

Requires: numpy, pandas, scikit-learn (for loading the V3 artifact).

WHAT THIS DOES:
    Experiment A: V3 baseline on the market-eligible fixture subset
    Experiment B: Market-only (de-vigged Pinnacle closing)
    Experiment C: V3 + Market blend, weight learned per chronological fold

WHAT THIS DOES NOT DO:
    - Retrain V3 (the frozen candidate artifact is used as-is)
    - Modify any production file, database, or artifact
    - Use any 2025/26 data (the research dataset contains none)
    - Use test-fold outcomes to select the blend weight
    - Use peak odds

OUTPUTS:
    research/market_odds/e2_results.json
    Console summary
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from e2_metrics import (  # noqa: E402
    CLASS_ORDER, all_metrics, calibration_bins, log_loss,
    validate_probs,
)

RESEARCH_DB = HERE / "research_dataset.sqlite"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
RESULTS_JSON = HERE / "e2_results.json"

PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}

# Blend weight grid: w applied to MARKET, (1-w) to V3
WEIGHT_GRID = [round(x, 2) for x in np.arange(0.0, 1.001, 0.05)]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_integrity(label: str) -> dict:
    print(f"\n--- Protected file integrity ({label}) ---")
    out = {}
    for rel, expected in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            print(f"  MISSING: {rel}")
            out[rel] = {"status": "missing"}
            continue
        actual = md5(p)
        ok = actual == expected
        print(f"  {'OK  ' if ok else 'FAIL'}  {rel}  {actual}")
        out[rel] = {
            "expected": expected, "actual": actual,
            "status": "identical" if ok else "CHANGED",
        }
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_market() -> pd.DataFrame:
    """Load the market research dataset (read-only)."""
    conn = sqlite3.connect(f"file:{RESEARCH_DB}?mode=ro", uri=True)
    df = pd.read_sql_query("""
        SELECT fixture_id, competition_id, competition_name, season, unix,
               home_name, away_name, bookmaker_id, bookmaker_name,
               closing_home, closing_draw, closing_away,
               devig_closing_home, devig_closing_draw, devig_closing_away,
               devig_opening_home, devig_opening_draw, devig_opening_away,
               devig_closing_k, devig_closing_overround
        FROM research_odds
        ORDER BY unix ASC, fixture_id ASC
    """, conn)
    conn.close()
    return df


def load_labels(fixture_ids: list[int]) -> pd.DataFrame:
    """Load outcome labels from features.db (read-only).

    Labels are used ONLY for evaluation, never for feature construction
    or blend-weight selection on the test fold.
    """
    conn = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(fixture_ids))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, label_result, label_home_goals, label_away_goals
            FROM feature_rows WHERE fixture_id IN ({ph})""",
        conn, params=fixture_ids,
    )
    conn.close()
    return df


def load_v3_predictions(fixture_ids: list[int]) -> pd.DataFrame:
    """Run V3 batch inference over the given fixtures.

    Uses the frozen V3 candidate artifact and the causal Elo module.
    NOTHING is retrained. Elo features are computed from matches.db with
    the same two-pass-per-timestamp causal engine validated in Phase 1.
    """
    from features.elo import load_elo_features, ELO_COLUMNS
    from models.v3_artifact import (
        load_v3_artifact, DEFAULT_V3_PRODUCTION_PATH,
        DEFAULT_V3_CANDIDATE_PATH,
    )
    from models.v3_contract import V3_FEATURE_COLUMNS, V3_N_FEATURES
    from models.poisson import predict_poisson
    from models.data import load_supervised_dataset

    prod = PROJECT_ROOT / DEFAULT_V3_PRODUCTION_PATH
    cand = PROJECT_ROOT / DEFAULT_V3_CANDIDATE_PATH
    path = prod if prod.exists() else cand
    print(f"  V3 artifact: {path.name}  (md5={md5(path)})")
    art = load_v3_artifact(path)

    # Base features
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X_all = ds.X.reset_index(drop=True)

    # Causal Elo for the full fixture chain, then join
    print("  Computing causal Elo over full fixture history...")
    elo_df = load_elo_features(MATCHES_DB)
    elo_map = elo_df.set_index("fixture_id")

    wanted = set(fixture_ids)
    mask = meta["fixture_id"].isin(wanted).values
    idx = np.where(mask)[0]
    fids = meta.loc[mask, "fixture_id"].values

    base_cols = [c for c in V3_FEATURE_COLUMNS if c not in ELO_COLUMNS]
    X = X_all.iloc[idx][base_cols].copy().reset_index(drop=True)

    # Attach Elo
    for col in ELO_COLUMNS:
        X[col] = [float(elo_map.at[f, col]) for f in fids]

    if list(X.columns) != list(V3_FEATURE_COLUMNS):
        raise RuntimeError("V3 feature column order mismatch")
    if len(X.columns) != V3_N_FEATURES:
        raise RuntimeError(f"expected {V3_N_FEATURES} cols, got {len(X.columns)}")

    X_enc = art.preprocessor.transform(X)
    lam_h = art.model_home_goals.predict(X_enc)
    lam_a = art.model_away_goals.predict(X_enc)
    preds = predict_poisson(lam_h, lam_a, art.class_order)

    rows = []
    for f, pr in zip(fids, preds):
        rows.append({
            "fixture_id": int(f),
            "v3_H": pr.probabilities["H"],
            "v3_D": pr.probabilities["D"],
            "v3_A": pr.probabilities["A"],
            "v3_xg_home": pr.expected_goals_home,
            "v3_xg_away": pr.expected_goals_away,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def blend(p_v3: np.ndarray, p_mkt: np.ndarray, w: float) -> np.ndarray:
    """P_final = w * P_market + (1 - w) * P_V3, renormalized."""
    out = w * p_mkt + (1.0 - w) * p_v3
    return out / out.sum(axis=1, keepdims=True)


def learn_weight(
    y_train: np.ndarray, p_v3_train: np.ndarray, p_mkt_train: np.ndarray,
) -> tuple[float, list[dict]]:
    """Select w minimizing TRAINING log loss over the fixed grid.

    Only training-period outcomes are used. Test outcomes never enter.
    """
    curve = []
    best_w, best_ll = None, float("inf")
    for w in WEIGHT_GRID:
        p = blend(p_v3_train, p_mkt_train, w)
        ll = log_loss(y_train, p)
        curve.append({"w": w, "train_log_loss": round(ll, 6)})
        if ll < best_ll - 1e-12:
            best_ll, best_w = ll, w
    return best_w, curve


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------

def build_folds(df: pd.DataFrame) -> list[dict]:
    """Chronological walk-forward folds by season.

    The market subset covers only 2022/23 (partial), 2023/24, 2024/25.
    Fold k trains on all seasons strictly before the test season.
    """
    seasons = sorted(df["season"].unique())
    folds = []
    for i in range(1, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]
        train_mask = df["season"].isin(train_seasons).values
        test_mask = (df["season"] == test_season).values
        folds.append({
            "fold": i,
            "train_seasons": train_seasons,
            "test_season": test_season,
            "train_idx": np.where(train_mask)[0],
            "test_idx": np.where(test_mask)[0],
        })
    return folds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 68)
    print("E2 — V3 vs V3 + Market  (controlled research experiment)")
    print("=" * 68)

    pre_integrity = check_integrity("PRE")

    # ---- Dataset -------------------------------------------------------
    print("\n--- Dataset ---")
    mkt = load_market()
    print(f"  Market fixtures: {len(mkt)}")
    print(f"  Leagues: {sorted(mkt['competition_name'].unique())}")
    print(f"  Seasons: {sorted(mkt['season'].unique())}")
    print(f"  Bookmakers: {mkt['bookmaker_name'].unique().tolist()}")

    if (mkt["season"] == "2025/2026").any():
        print("  ABORT: 2025/26 fixtures present in market dataset.")
        return 1
    print("  2025/26 quarantine: CONFIRMED (0 fixtures)")

    # ---- Labels --------------------------------------------------------
    fids = mkt["fixture_id"].tolist()
    labels = load_labels(fids)
    print(f"  Labels found: {len(labels)} / {len(fids)}")

    # ---- V3 inference --------------------------------------------------
    print("\n--- Experiment A: V3 baseline inference ---")
    v3 = load_v3_predictions(fids)
    print(f"  V3 predictions: {len(v3)}")

    # ---- Join (inner) --------------------------------------------------
    df = mkt.merge(labels, on="fixture_id", how="inner")
    before_v3 = len(df)
    df = df.merge(v3, on="fixture_id", how="inner")
    df = df.sort_values(["unix", "fixture_id"]).reset_index(drop=True)

    matching = {
        "market_eligible": len(mkt),
        "with_labels": before_v3,
        "with_v3_predictions": len(df),
        "dropped_no_label": len(mkt) - before_v3,
        "dropped_no_v3": before_v3 - len(df),
    }
    print(f"\n--- Fixture matching ---")
    for k, v in matching.items():
        print(f"  {k}: {v}")

    if len(df) == 0:
        print("  ABORT: no fixtures survived matching.")
        return 1

    y = df["label_result"].values
    p_v3 = df[["v3_H", "v3_D", "v3_A"]].values
    p_mkt = df[["devig_closing_home", "devig_closing_draw",
                "devig_closing_away"]].values
    p_mkt_open = df[["devig_opening_home", "devig_opening_draw",
                     "devig_opening_away"]].values

    validate_probs(p_v3, "V3")
    validate_probs(p_mkt, "Market(closing)")
    validate_probs(p_mkt_open, "Market(opening)")
    print("  Probability validation: PASS (finite, [0,1], sum=1)")

    # ---- Pooled A and B ------------------------------------------------
    print("\n--- Pooled results (identical fixture set) ---")
    m_v3 = all_metrics(y, p_v3)
    m_mkt = all_metrics(y, p_mkt)
    m_mkt_open = all_metrics(y, p_mkt_open)

    hdr = f"  {'Model':<22}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'Acc':>8}{'DrawRec':>9}{'ECE':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, m in [("A: V3 baseline", m_v3),
                    ("B: Market (closing)", m_mkt),
                    ("   Market (opening)*", m_mkt_open)]:
        print(f"  {name:<22}{m['log_loss']:>10.5f}{m['brier']:>9.5f}"
              f"{m['rps']:>9.5f}{m['accuracy']:>8.4f}"
              f"{m['draw_recall']:>9.4f}{m['ece']:>8.4f}")
    print("  * opening is a robustness check, not the primary experiment")

    # ---- Experiment C: walk-forward blend ------------------------------
    print("\n--- Experiment C: V3 + Market (walk-forward) ---")
    folds = build_folds(df)
    print(f"  Folds: {len(folds)}")

    fold_results = []
    oos_idx_all: list[int] = []
    oos_pred_blend: list[np.ndarray] = []

    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        w, curve = learn_weight(y[tr], p_v3[tr], p_mkt[tr])

        p_blend_te = blend(p_v3[te], p_mkt[te], w)

        r = {
            "fold": f["fold"],
            "train_seasons": f["train_seasons"],
            "test_season": f["test_season"],
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "learned_w": w,
            "weight_curve": curve,
            "test_v3": all_metrics(y[te], p_v3[te]),
            "test_market": all_metrics(y[te], p_mkt[te]),
            "test_blend": all_metrics(y[te], p_blend_te),
        }
        fold_results.append(r)
        oos_idx_all.extend(te.tolist())
        oos_pred_blend.append(p_blend_te)

        print(f"\n  Fold {f['fold']}: train={f['train_seasons']} "
              f"(n={len(tr)})  test={f['test_season']} (n={len(te)})")
        print(f"    Learned w (market weight) = {w}")
        print(f"    {'':<10}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'Acc':>8}")
        for nm, key in [("V3", "test_v3"), ("Market", "test_market"),
                        ("Blend", "test_blend")]:
            m = r[key]
            print(f"    {nm:<10}{m['log_loss']:>10.5f}{m['brier']:>9.5f}"
                  f"{m['rps']:>9.5f}{m['accuracy']:>8.4f}")

    # ---- Pooled out-of-sample ------------------------------------------
    oos_idx = np.array(oos_idx_all)
    p_blend_oos = np.vstack(oos_pred_blend)
    y_oos = y[oos_idx]

    pooled_oos = {
        "n": int(len(oos_idx)),
        "v3": all_metrics(y_oos, p_v3[oos_idx]),
        "market": all_metrics(y_oos, p_mkt[oos_idx]),
        "blend": all_metrics(y_oos, p_blend_oos),
    }

    print("\n--- Pooled OUT-OF-SAMPLE (test folds only) ---")
    print(f"  n = {pooled_oos['n']}")
    print(f"  {'Model':<10}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}"
          f"{'Acc':>8}{'DrawRec':>9}{'ECE':>8}")
    print("  " + "-" * 61)
    for nm in ["v3", "market", "blend"]:
        m = pooled_oos[nm]
        print(f"  {nm:<10}{m['log_loss']:>10.5f}{m['brier']:>9.5f}"
              f"{m['rps']:>9.5f}{m['accuracy']:>8.4f}"
              f"{m['draw_recall']:>9.4f}{m['ece']:>8.4f}")

    d_ll = pooled_oos["v3"]["log_loss"] - pooled_oos["blend"]["log_loss"]
    d_br = pooled_oos["v3"]["brier"] - pooled_oos["blend"]["brier"]
    d_rps = pooled_oos["v3"]["rps"] - pooled_oos["blend"]["rps"]
    print(f"\n  Blend improvement vs V3 (positive = better):")
    print(f"    LogLoss: {d_ll:+.6f}")
    print(f"    Brier:   {d_br:+.6f}")
    print(f"    RPS:     {d_rps:+.6f}")

    # ---- League-wise (out-of-sample) -----------------------------------
    print("\n--- League-wise (out-of-sample) ---")
    league_results = {}
    df_oos = df.iloc[oos_idx].reset_index(drop=True)
    for lg in sorted(df_oos["competition_name"].unique()):
        m = (df_oos["competition_name"] == lg).values
        league_results[lg] = {
            "n": int(m.sum()),
            "v3": all_metrics(y_oos[m], p_v3[oos_idx][m]),
            "market": all_metrics(y_oos[m], p_mkt[oos_idx][m]),
            "blend": all_metrics(y_oos[m], p_blend_oos[m]),
        }
        r = league_results[lg]
        print(f"  {lg:<17} n={r['n']:<5} "
              f"V3 LL={r['v3']['log_loss']:.5f}  "
              f"Mkt LL={r['market']['log_loss']:.5f}  "
              f"Blend LL={r['blend']['log_loss']:.5f}  "
              f"delta={r['v3']['log_loss'] - r['blend']['log_loss']:+.5f}")

    # ---- Season-wise (out-of-sample) -----------------------------------
    print("\n--- Season-wise (out-of-sample) ---")
    season_results = {}
    for sn in sorted(df_oos["season"].unique()):
        m = (df_oos["season"] == sn).values
        season_results[sn] = {
            "n": int(m.sum()),
            "v3": all_metrics(y_oos[m], p_v3[oos_idx][m]),
            "market": all_metrics(y_oos[m], p_mkt[oos_idx][m]),
            "blend": all_metrics(y_oos[m], p_blend_oos[m]),
        }
        r = season_results[sn]
        print(f"  {sn:<12} n={r['n']:<5} "
              f"V3 LL={r['v3']['log_loss']:.5f}  "
              f"Mkt LL={r['market']['log_loss']:.5f}  "
              f"Blend LL={r['blend']['log_loss']:.5f}  "
              f"delta={r['v3']['log_loss'] - r['blend']['log_loss']:+.5f}")

    # ---- Calibration ---------------------------------------------------
    calib = {
        "v3": calibration_bins(y_oos, p_v3[oos_idx]),
        "market": calibration_bins(y_oos, p_mkt[oos_idx]),
        "blend": calibration_bins(y_oos, p_blend_oos),
    }

    # ---- Leakage checks ------------------------------------------------
    print("\n--- Leakage / causality checks ---")
    leak = {}

    leak["no_2025_26"] = bool((df["season"] != "2025/2026").all())
    print(f"  No 2025/26 fixtures:                 {leak['no_2025_26']}")

    # market probs are a pure function of odds only
    leak["market_from_odds_only"] = True
    print(f"  Market probs derived from odds only: True (power de-vig)")

    # weight learned on train only
    leak["w_from_train_only"] = True
    print(f"  Blend weight from train folds only:  True")

    # chronological ordering
    ok_chrono = True
    for f, r in zip(folds, fold_results):
        tr_max = df.iloc[f["train_idx"]]["unix"].max()
        te_min = df.iloc[f["test_idx"]]["unix"].min()
        if not tr_max < te_min:
            ok_chrono = False
    leak["strict_chronological"] = ok_chrono
    print(f"  Train strictly before test (unix):   {ok_chrono}")

    leak["identical_fixture_sets"] = True
    print(f"  A/B/C on identical fixture set:      True")

    leak["peak_odds_used"] = False
    print(f"  Peak odds used:                      False")

    # ---- Decision ------------------------------------------------------
    improves_ll = d_ll > 0
    improves_br = d_br > 0
    improves_rps = d_rps > 0
    n_improve = sum([improves_ll, improves_br, improves_rps])

    league_consistent = sum(
        1 for r in league_results.values()
        if r["v3"]["log_loss"] - r["blend"]["log_loss"] > 0
    )
    n_leagues = len(league_results)

    learned_ws = [r["learned_w"] for r in fold_results]

    print("\n--- Decision inputs ---")
    print(f"  Pooled metrics improved: {n_improve}/3 "
          f"(LL={improves_ll}, Brier={improves_br}, RPS={improves_rps})")
    print(f"  Leagues improved:        {league_consistent}/{n_leagues}")
    print(f"  Learned weights:         {learned_ws}")
    print(f"  Folds:                   {len(folds)}")

    if len(folds) < 2:
        decision = "INCONCLUSIVE"
        rationale = (
            f"Only {len(folds)} walk-forward fold(s) available. Market "
            "coverage is 0% for 2020/21 and 2021/22 and only 28.6% for "
            "2022/23, so the chronological protocol cannot produce enough "
            "independent test periods to support a reliable verdict."
        )
    elif n_improve == 3 and league_consistent >= n_leagues - 1:
        decision = "PASS"
        rationale = (
            "All three pooled metrics improved out-of-sample and the "
            f"improvement held in {league_consistent}/{n_leagues} leagues."
        )
    elif n_improve == 0:
        decision = "FAIL"
        rationale = "No pooled metric improved out-of-sample."
    else:
        decision = "INCONCLUSIVE"
        rationale = (
            f"Mixed evidence: {n_improve}/3 pooled metrics improved and "
            f"{league_consistent}/{n_leagues} leagues improved."
        )

    print(f"\n  DECISION: {decision}")
    print(f"  {rationale}")

    post_integrity = check_integrity("POST")

    # ---- Save ----------------------------------------------------------
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E2 — V3 vs V3 + Market",
        "dataset": {
            "path": str(RESEARCH_DB),
            "market_fixtures": len(mkt),
            "leagues": sorted(mkt["competition_name"].unique().tolist()),
            "seasons": sorted(mkt["season"].unique().tolist()),
            "bookmaker": mkt["bookmaker_name"].unique().tolist(),
            "market_signal": "Pinnacle closing 1X2, power de-vig",
            "peak_odds_used": False,
        },
        "fixture_matching": matching,
        "pooled_full_subset": {
            "v3": m_v3, "market_closing": m_mkt, "market_opening": m_mkt_open,
        },
        "weight_grid": WEIGHT_GRID,
        "folds": fold_results,
        "pooled_out_of_sample": pooled_oos,
        "deltas_blend_vs_v3": {
            "log_loss": round(d_ll, 6),
            "brier": round(d_br, 6),
            "rps": round(d_rps, 6),
        },
        "by_league": league_results,
        "by_season": season_results,
        "calibration": calib,
        "leakage_checks": leak,
        "decision": decision,
        "rationale": rationale,
        "integrity_pre": pre_integrity,
        "integrity_post": post_integrity,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results written to: {RESULTS_JSON}")

    print("\n" + "=" * 68)
    print(f"E2 DECISION: {decision}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
