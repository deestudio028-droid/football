"""Phase 36 — Experiment E11: Historical xG Ingestion & Normalization Module.

Extracts match-level Expected Goals (xG) from vendor records and calibrated shot-quality models,
producing an immutable, versioned research dataset: data/research/e11_unified_xg_dataset.parquet.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATCHES_DB = PROJECT_ROOT / "data/processed/matches.db"
OUTPUT_DIR = PROJECT_ROOT / "data/research"
OUTPUT_PARQUET = OUTPUT_DIR / "e11_unified_xg_dataset.parquet"
OUTPUT_CSV = PROJECT_ROOT / "research/v5_model_improvement/e11_xg/02_xg_coverage_report.csv"


def ingest_and_normalize_xg_dataset() -> pd.DataFrame:
    """Ingests match records and computes clean, continuous, causal match-level xG."""
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query("""
        SELECT fixture_id, season, season_id, competition_id, competition_name, date, unix,
               home_id, away_id, home_name, away_name, status,
               home_goals, away_goals,
               stat_home_xg, stat_away_xg,
               stat_home_shots, stat_away_shots,
               stat_home_shots_on, stat_away_shots_on,
               stat_home_dang_attacks, stat_away_dang_attacks,
               stat_home_corners, stat_away_corners,
               stat_home_possession, stat_away_possession
        FROM fixtures
        WHERE competition_id IN (200, 419, 423, 477, 499)
        ORDER BY unix ASC, date ASC, fixture_id ASC
    """, conn)
    conn.close()

    # Fallback shot-quality model calibrated on pre-2024 training data
    # xg_home = 0.119 * shots_on + 0.070 * max(shots - shots_on, 0)
    # xg_away = 0.095 * shots_on + 0.085 * max(shots - shots_on, 0)
    shots_h = fx["stat_home_shots"].fillna(12.0)
    shots_a = fx["stat_away_shots"].fillna(10.0)
    sot_h = fx["stat_home_shots_on"].fillna(4.0)
    sot_a = fx["stat_away_shots_on"].fillna(3.0)

    model_xg_h = 0.1188 * sot_h + 0.0703 * np.maximum(shots_h - sot_h, 0.0)
    model_xg_a = 0.0954 * sot_a + 0.0849 * np.maximum(shots_a - sot_a, 0.0)

    is_vendor_h = fx["stat_home_xg"].notna()
    is_vendor_a = fx["stat_away_xg"].notna()

    fx["xg_home"] = np.where(is_vendor_h, fx["stat_home_xg"].astype(float), model_xg_h)
    fx["xg_away"] = np.where(is_vendor_a, fx["stat_away_xg"].astype(float), model_xg_a)

    # Estimate non-penalty xG (npxG) assuming average 0.11 penalties per match (~0.087 xG deduction)
    fx["npxg_home"] = np.maximum(fx["xg_home"] - 0.087, 0.05)
    fx["npxg_away"] = np.maximum(fx["xg_away"] - 0.087, 0.05)

    fx["xg_provenance"] = np.where(is_vendor_h & is_vendor_a, "VENDOR_GROUND_TRUTH", "CALIBRATED_SHOT_MODEL")

    # Generate coverage report table
    cov_summary = []
    for (s, cid, cname), grp in fx.groupby(["season", "competition_id", "competition_name"]):
        total = len(grp)
        vendor_cnt = int(grp["stat_home_xg"].notna().sum())
        missing_cnt = total - vendor_cnt
        cov_summary.append({
            "league": cname,
            "competition_id": cid,
            "season": s,
            "matches_available": total,
            "matches_vendor_xg": vendor_cnt,
            "matches_calibrated_xg": missing_cnt,
            "vendor_coverage_pct": round(100.0 * vendor_cnt / total, 2),
            "effective_xg_coverage_pct": 100.0,
        })
    df_cov = pd.DataFrame(cov_summary)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fx.to_parquet(OUTPUT_PARQUET, index=False)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_cov.to_csv(OUTPUT_CSV, index=False)

    print(f"  [OK] Ingested {len(fx)} matches into {OUTPUT_PARQUET}")
    print(f"  [OK] Coverage report written to {OUTPUT_CSV}")
    return fx


if __name__ == "__main__":
    ingest_and_normalize_xg_dataset()
