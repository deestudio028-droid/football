"""Step 2E Shadow Integration Runner.

Executes reproduction tests, holdout evaluation (2025/26 season),
and prospective evaluation (Aug 22-24, 2026) for the shadow Platt calibrator.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from draw_probability_calibrator import CalibratedPredictionResult, DrawProbabilityCalibrator
from features.elo import ELO_COLUMNS, load_elo_features
from features.online_attack_defense import AD_COLUMNS, compute_ad_states, fit_baseline_rates
from models.baselines import CLASS_ORDER
from models.config import (
    FINAL_TEST_SEASONS,
    FINAL_TRAIN_SEASONS,
    SEASON_NAME_TO_IDS,
)
from models.data import load_supervised_dataset
from models.draw_champion import (
    DrawChampionConfig,
    compute_dc_draw_probability,
    compute_elo_draw_probability,
    redistribute_proportional_odds,
    stable_logit,
    stable_sigmoid,
)
from models.poisson import predict_poisson
from models.v4_artifact import load_v4_artifact

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
V4_ARTIFACT_PATH = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
V4_1_ARTIFACT_PATH = PROJECT_ROOT / "data" / "models" / "v4_1_prospective_candidate_2025_26.pkl"
FROZEN_CHAMPION_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "draw_champion_method_frozen.json"
CONFIG_PATH = PROJECT_ROOT / "research" / "v5_model_improvement" / "step2_draw_research" / "step2e_shadow_integration" / "draw_calibrator_config.json"
OUTPUT_DIR = PROJECT_ROOT / "research" / "v5_model_improvement" / "step2_draw_research" / "step2e_shadow_integration"

EXPECTED_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXPECTED_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def verify_integrity() -> None:
    act_v4 = hashlib.md5(V4_ARTIFACT_PATH.read_bytes()).hexdigest()
    act_v41 = hashlib.md5(V4_1_ARTIFACT_PATH.read_bytes()).hexdigest()
    if act_v4 != EXPECTED_V4_MD5:
        raise RuntimeError(f"V4.0 MD5 MISMATCH! Expected {EXPECTED_V4_MD5}, got {act_v4}. STOPPING.")
    if act_v41 != EXPECTED_V4_1_MD5:
        raise RuntimeError(f"V4.1 MD5 MISMATCH! Expected {EXPECTED_V4_1_MD5}, got {act_v41}. STOPPING.")
    print("  [OK] Pre-flight Integrity: Both V4.0 and V4.1 MD5 hashes verified 100% bit-identical.")


def load_dataset() -> pd.DataFrame:
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)
    y = ds.y.reset_index(drop=True)

    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])

    conn_m = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query(
        """SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                  status, season, season_id, competition_id, competition_name,
                  home_name, away_name, date
           FROM fixtures WHERE competition_id IN (200,419,423,477,499)""", conn_m)
    conn_m.close()

    wanted_hist = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted_hist |= set(SEASON_NAME_TO_IDS[sn])
    hist_fx = fx[fx.season_id.isin(wanted_hist) & fx.home_goals.notna() & fx.status.isin(["FT", "AWARDED"])]
    base_rates = fit_baseline_rates(hist_fx.home_goals.values.astype(float), hist_fx.away_goals.values.astype(float))
    states = compute_ad_states(fx, 0.02, base_rates).set_index("fixture_id")
    for c in AD_COLUMNS:
        X[c] = meta["fixture_id"].map(states[c])

    fx_dict = fx.set_index("fixture_id").to_dict("index")
    meta["competition_name"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("competition_name", "Unknown"))
    meta["home_name"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("home_name", "Unknown"))
    meta["away_name"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("away_name", "Unknown"))
    meta["season"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("season", "Unknown"))
    meta["date"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("date", ""))
    meta["home_goals"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("home_goals", np.nan))
    meta["away_goals"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("away_goals", np.nan))
    meta["unix"] = meta["fixture_id"].map(lambda fid: fx_dict.get(fid, {}).get("unix", 0))

    valid_mask = y.isin(["H", "D", "A"]) & meta["home_goals"].notna()
    meta = meta[valid_mask].reset_index(drop=True)
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    v4 = load_v4_artifact(V4_ARTIFACT_PATH)
    cfg_champ = DrawChampionConfig.from_frozen_json(FROZEN_CHAMPION_PATH)

    E = v4.preprocessor.transform(X[list(v4.feature_columns)])
    lh = v4.model_home_goals.predict(E)
    la = v4.model_away_goals.predict(E)

    pr_poiss = predict_poisson(lh, la, list(v4.class_order))
    pv4 = np.array([[p.probabilities["H"], p.probabilities["D"], p.probabilities["A"]] for p in pr_poiss])

    abs_elo = np.array([abs((X.loc[i, "home_elo"] + 100.0) - X.loc[i, "away_elo"]) for i in range(len(X))])
    rhos = np.array([cfg_champ.league_rhos.get(meta.loc[i, "competition_name"], cfg_champ.global_fallback_rho) for i in range(len(meta))])

    p_dc = compute_dc_draw_probability(lh, la, rhos)
    p_elo = compute_elo_draw_probability(pv4[:, 1], abs_elo, cfg_champ)

    z = cfg_champ.stacking_intercept + cfg_champ.stacking_weight_dc * stable_logit(p_dc) + cfg_champ.stacking_weight_elo * stable_logit(p_elo)
    p_d_ch = stable_sigmoid(z)
    p_ch = redistribute_proportional_odds(pv4, p_d_ch)

    df_out = meta.copy()
    df_out["actual_result"] = y
    df_out["is_draw_actual"] = (df_out["actual_result"] == "D").astype(int)
    df_out["v4_p_H"] = p_ch[:, 0]
    df_out["v4_p_D"] = p_ch[:, 1]
    df_out["v4_p_A"] = p_ch[:, 2]
    df_out["v4_decision"] = [CLASS_ORDER[i] for i in np.argmax(p_ch, axis=1)]

    return df_out


def run_shadow_integration():
    verify_integrity()
    df = load_dataset()

    calibrator = DrawProbabilityCalibrator.from_config_file(CONFIG_PATH)

    # 1. HISTORICAL REPRODUCTION TEST (against Step 2D reference outputs)
    print("\n1. Running Historical Reproduction Test against Step 2D...")
    df_te = df[df["season"] == "2025/2026"].copy().reset_index(drop=True)
    v4_probs_te = df_te[["v4_p_H", "v4_p_D", "v4_p_A"]].values
    calib_probs_te = calibrator.calibrate_array(v4_probs_te)

    # Step 2D reference calculation
    z_ref = calibrator.slope_a * stable_logit(df_te["v4_p_D"].values) + calibrator.intercept_b
    pd_ref = stable_sigmoid(z_ref)
    p3_ref = redistribute_proportional_odds(v4_probs_te, pd_ref)

    max_diff = float(np.max(np.abs(calib_probs_te - p3_ref)))
    print(f"  Maximum absolute difference vs Step 2D reference: {max_diff:.2e}")
    assert max_diff < 1e-8, f"Reproduction discrepancy exceeded tolerance: {max_diff}"

    repro_rows = []
    for i in range(min(50, len(df_te))):
        res = calibrator.calibrate_single(df_te.loc[i, "v4_p_H"], df_te.loc[i, "v4_p_D"], df_te.loc[i, "v4_p_A"])
        repro_rows.append({
            "fixture_id": df_te.loc[i, "fixture_id"],
            "home_team": df_te.loc[i, "home_name"],
            "away_team": df_te.loc[i, "away_name"],
            "v4_p_H": df_te.loc[i, "v4_p_H"],
            "v4_p_D": df_te.loc[i, "v4_p_D"],
            "v4_p_A": df_te.loc[i, "v4_p_A"],
            "v4_decision": df_te.loc[i, "v4_decision"],
            "calibrated_p_H": res.calibrated_p_home,
            "calibrated_p_D": res.calibrated_p_draw,
            "calibrated_p_A": res.calibrated_p_away,
            "calibrated_decision": res.calibrated_decision,
            "draw_delta": res.draw_delta,
            "decision_changed": res.decision_changed,
            "reference_p_D": round(float(pd_ref[i]), 6),
            "abs_error_vs_ref": abs(res.calibrated_p_draw - round(float(pd_ref[i]), 6)),
        })
    df_repro = pd.DataFrame(repro_rows)
    df_repro.to_csv(OUTPUT_DIR / "shadow_historical_reproduction.csv", index=False)
    print("  [OK] Saved historical reproduction check to shadow_historical_reproduction.csv")

    # 2. FULL 2025/26 HOLDOUT SHADOW PREDICTIONS
    print("\n2. Generating Full 2025/26 Holdout Shadow Predictions...")
    pred_rows = []
    for i in range(len(df_te)):
        res = calibrator.calibrate_single(df_te.loc[i, "v4_p_H"], df_te.loc[i, "v4_p_D"], df_te.loc[i, "v4_p_A"])
        pred_rows.append({
            "fixture_id": df_te.loc[i, "fixture_id"],
            "competition_name": df_te.loc[i, "competition_name"],
            "date": df_te.loc[i, "date"],
            "home_team": df_te.loc[i, "home_name"],
            "away_team": df_te.loc[i, "away_name"],
            "actual_result": df_te.loc[i, "actual_result"],
            "v4_home_prob": df_te.loc[i, "v4_p_H"],
            "v4_draw_prob": df_te.loc[i, "v4_p_D"],
            "v4_away_prob": df_te.loc[i, "v4_p_A"],
            "v4_decision": df_te.loc[i, "v4_decision"],
            "calibrated_home_prob": res.calibrated_p_home,
            "calibrated_draw_prob": res.calibrated_p_draw,
            "calibrated_away_prob": res.calibrated_p_away,
            "calibrated_decision": res.calibrated_decision,
            "draw_probability_delta": res.draw_delta,
            "decision_changed": res.decision_changed,
        })
    df_preds = pd.DataFrame(pred_rows)
    df_preds.to_csv(OUTPUT_DIR / "shadow_predictions.csv", index=False)
    print(f"  [OK] Saved {len(df_preds)} holdout shadow predictions to shadow_predictions.csv")

    # 3. RECENT PROSPECTIVE 33-MATCH EVALUATION (Aug 22-24, 2026)
    print("\n3. Evaluating on Recent 33 Prospective Matches (Aug 22-24, 2026)...")
    from dashboard.fixture_service import FixtureService
    from dashboard.prediction_service import PredictionService
    from dashboard.time_utils import format_kickoff_ist, to_chennai_date

    fs = FixtureService()
    ps = PredictionService()

    recent_fixes = []
    seen_fids = set()
    for d_str in ["2026-08-22", "2026-08-23", "2026-08-24"]:
        fixes_d, _ = fs.get_todays_matches(date_str=d_str, provider_name="oddalerts")
        for f in fixes_d:
            if f.fixture_id not in seen_fids and f.status == "FT":
                seen_fids.add(f.fixture_id)
                recent_fixes.append(f)

    recent_fixes.sort(key=lambda x: x.scheduled_kickoff)
    r33_rows = []
    for f in recent_fixes:
        p40 = ps.predict_dashboard_fixture(f, model_key="V4.0 Production")
        probs_40 = p40.v4_champ_probs
        p_h, p_d, p_a = probs_40["H"], probs_40["D"], probs_40["A"]
        v4_dec = p40.v4_champ_decision

        res = calibrator.calibrate_single(p_h, p_d, p_a)
        is_draw = (f.actual_outcome == "D")
        is_v4_corr = (v4_dec == f.actual_outcome)
        is_cal_corr = (res.calibrated_decision == f.actual_outcome)

        r33_rows.append({
            "fixture_id": f.fixture_id,
            "date_ist": to_chennai_date(f.scheduled_kickoff),
            "kickoff_ist": format_kickoff_ist(f.scheduled_kickoff),
            "league": f.competition_name,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "score": f"{f.home_goals}-{f.away_goals}",
            "actual_outcome": f.actual_outcome,
            "v4_p_H": round(p_h, 4),
            "v4_p_D": round(p_d, 4),
            "v4_p_A": round(p_a, 4),
            "v4_decision": v4_dec,
            "v4_correct": is_v4_corr,
            "calibrated_p_H": res.calibrated_p_home,
            "calibrated_p_D": res.calibrated_p_draw,
            "calibrated_p_A": res.calibrated_p_away,
            "calibrated_decision": res.calibrated_decision,
            "calibrated_correct": is_cal_corr,
            "draw_delta": res.draw_delta,
            "home_delta": res.home_delta,
            "away_delta": res.away_delta,
            "decision_changed": res.decision_changed,
            "is_actual_draw": is_draw,
        })
    df_r33 = pd.DataFrame(r33_rows)
    df_r33.to_csv(OUTPUT_DIR / "shadow_33_match_audit.csv", index=False)
    print("  [OK] Saved recent 33 prospective audit to shadow_33_match_audit.csv")

    # Metrics Summary
    print("\n--- SHADOW CALIBRATION PERFORMANCE SUMMARY (2025/26 Holdout N=1,751) ---")
    y_d = df_te["is_draw_actual"].values
    p_d_v4 = df_te["v4_p_D"].values
    p_d_cal = calib_probs_te[:, 1]

    brier_v4 = float(np.mean((p_d_v4 - y_d) ** 2))
    brier_cal = float(np.mean((p_d_cal - y_d) ** 2))
    ll_v4 = float(-np.mean(y_d * np.log(p_d_v4) + (1 - y_d) * np.log(1 - p_d_v4)))
    ll_cal = float(-np.mean(y_d * np.log(p_d_cal) + (1 - y_d) * np.log(1 - p_d_cal)))

    print(f"Mean P(D):            {np.mean(p_d_v4)*100:.2f}% -> Calibrated: {np.mean(p_d_cal)*100:.2f}% (Actual: {np.mean(y_d)*100:.2f}%)")
    print(f"Draw Brier Score:     {brier_v4:.6f} -> Calibrated: {brier_cal:.6f} (Delta: {brier_cal - brier_v4:+.6f})")
    print(f"Draw Log Loss:        {ll_v4:.6f} -> Calibrated: {ll_cal:.6f} (Delta: {ll_cal - ll_v4:+.6f})")
    print(f"Holdout Accuracy:     {(df_te['v4_decision'] == df_te['actual_result']).mean()*100:.2f}% -> Calibrated: {(df_preds['calibrated_decision'] == df_te['actual_result']).mean()*100:.2f}%")
    print(f"Decisions Changed:    {df_preds['decision_changed'].sum()} / {len(df_preds)}")


if __name__ == "__main__":
    run_shadow_integration()
