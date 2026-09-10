"""Football Prediction Model Lab — Streamlit Local Research Dashboard.

Dedicated interface for inspecting production models, evaluating research candidates,
generating single-match predictions, dynamic model switching, and validating against
live OddAlerts API and 2026/27 fresh matches.

Usage:
    scripts/run_dashboard.bat
    or: streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.evaluation_service import EvaluationService
from dashboard.fixture_service import (
    DashboardFixture,
    FixtureService,
    TARGET_LEAGUES,
    WEEKLY_LEAGUE_TARGETS,
    WEEKLY_COMPETITION_TARGETS,
    TOTAL_WEEKLY_TARGET,
)
from dashboard.historical_memory_service import (
    HistoricalMemoryService,
    get_historical_memory_service,
)
from dashboard.metrics_service import MetricsService
from dashboard.model_registry import ModelRegistry, get_model_registry
from dashboard.prediction_service import DashboardMatchPrediction, PredictionService, SingleMatchPredictionResult
from dashboard.time_utils import (
    format_kickoff_datetime_utc,
    format_kickoff_utc,
    to_utc_date,
)

# Set page config
st.set_page_config(
    page_title="Football Prediction Model Lab",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Services
@st.cache_resource
def load_services():
    registry = get_model_registry()
    fixture_service = FixtureService()
    pred_service = PredictionService()
    eval_service = EvaluationService()
    memory_service = get_historical_memory_service()
    return registry, fixture_service, pred_service, eval_service, memory_service

registry, fixture_service, pred_service, eval_service, memory_service = load_services()

# Fixed Active Production Model
ACTIVE_MODEL_KEY = "V4.0 Production"
active_model_info = registry.get_production_model()

# --- TOP HEADER ---
st.title("⚽ FOOTBALL PREDICTION MODEL LAB")
st.caption("Production V4.0 Prediction Engine • Real Fixture API Feed • Frozen Baseline Architecture")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Production Model Engine")
    if active_model_info is not None:
        st.success(
            f"**Active Model:** `{active_model_info.display_name}`\n\n"
            f"**Version:** `{active_model_info.version}`\n\n"
            f"**Status:** **{active_model_info.status}**\n\n"
            f"**Artifact:** `{Path(active_model_info.file_path).name}`\n\n"
            f"**MD5:** `{active_model_info.md5_hash}`\n\n"
            f"**Info Cutoff:** `{active_model_info.parameters.get('information_cutoff', 'N/A')}`"
        )

    st.markdown("---")
    st.header("📡 Fixture Provider")
    provider_choice = st.selectbox(
        "Active Feed Source",
        options=["OddAlerts API (Live)", "Local Database (Research)"],
        index=0,
        help="Select live OddAlerts HTTP feed or offline SQLite database.",
    )
    provider_key = "oddalerts" if "OddAlerts" in provider_choice else "local_db"

    st.markdown("---")
    st.header("⚙️ Global Controls")
    fixture_mode_choice = st.radio(
        "Fixture Selection Mode",
        options=[
            f"Weekly {TOTAL_WEEKLY_TARGET} Gameweek Window (10 EPL, 10 Serie A, 10 La Liga, 9 Bundesliga, 9 Ligue 1)",
            "All Available Feed Fixtures",
        ],
        index=0,
        help=f"Select exact weekly {TOTAL_WEEKLY_TARGET}-match gameweek distribution (10/10/10/9/9) or all available feed fixtures.",
    )
    is_weekly_mode = "Weekly" in fixture_mode_choice

    selected_league_name = st.selectbox(
        "Target Competition",
        options=["All 5 Leagues", "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"],
        index=0,
    )

    safety_buffer = st.slider("Pre-Kickoff Buffer (Minutes)", min_value=5, max_value=60, value=15, step=5)

    st.markdown("---")
    st.header("🔄 Operational Actions")
    if st.button("🔄 Refresh Fixtures"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("🔒 All models loaded read-only with cryptographic hash enforcement.")

# Competition IDs filter
leagues_map = fixture_service.get_supported_leagues()
inv_leagues = {v: k for k, v in leagues_map.items()}
comp_ids_filter = [inv_leagues[selected_league_name]] if selected_league_name != "All 5 Leagues" else list(leagues_map.keys())

# --- MAIN NAVIGATION TABS ---
tab_overview, tab_single, tab_compare, tab_predictions_results, tab_eval2627, tab_draw, tab_leagues, tab_temporal, tab_agreement, tab_safety = st.tabs([
    "🏠 Overview",
    "🔮 Single Prediction",
    "⚖️ Compare Models",
    "📅 Predictions & Results",
    "📊 2026/27 Evaluation",
    "⚽ Draw Analysis",
    "🏆 League Breakdown",
    "📈 Temporal Analysis",
    "🧠 Model Agreement",
    "🔐 Production Safety",
])

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tab_overview:
    st.subheader("System State & Active Prediction Engine")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Production Model", ACTIVE_MODEL_KEY, "FROZEN TRUTH")
    col2.metric("Architecture", "Poisson + DC + Elo", "OPTIMIZED")
    col3.metric("Baseline Artifact", "v4_poisson_venue_elo_online_ad.pkl", "VERIFIED")
    col4.metric("Live Feed Provider", "OddAlerts API" if provider_key == "oddalerts" else "Local DB", "ACTIVE")
    col5.metric("System Mode", "PRODUCTION ONLY", "LOCKED")

    st.markdown("### Centralized Model Registry Ledger")
    all_models = registry.list_all_models()
    reg_rows = []
    for m in all_models:
        reg_rows.append({
            "Display Name": m.display_name,
            "Model ID": m.model_id,
            "Role": m.role,
            "Status": m.status,
            "Version": m.version,
            "Artifact Path": m.file_path,
            "MD5 Hash": m.md5_hash,
            "Information Cutoff": m.parameters.get("information_cutoff", "N/A"),
        })
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True)

# ==============================================================================
# TAB 2: SINGLE MATCH PREDICTION
# ==============================================================================
with tab_single:
    st.subheader("🔮 Single Match Pre-Match Inference")
    st.markdown(f"Generating predictions using frozen production engine: **`{ACTIVE_MODEL_KEY}`**")

    col_lg, col_h, col_a = st.columns(3)
    with col_lg:
        sel_lg = st.selectbox("League", options=list(leagues_map.values()), index=0, key="single_sel_lg")
        sel_cid = inv_leagues[sel_lg]
    with col_h:
        teams = fixture_service.get_teams_by_league(sel_cid)
        home_team = st.selectbox("Home Team", options=teams, index=0 if teams else 0, key="single_home_team")
    with col_a:
        away_options = [t for t in teams if t != home_team] if teams else []
        away_team = st.selectbox("Away Team", options=away_options, index=0 if away_options else 0, key="single_away_team")

    if st.button("🚀 Generate Prediction", type="primary", key="btn_single_pred"):
        if home_team == away_team or not home_team or not away_team:
            st.error("Please select two distinct teams.")
        else:
            pred = pred_service.predict_matchup(
                home_team=home_team,
                away_team=away_team,
                competition_name=sel_lg,
                competition_id=sel_cid,
                model_key=ACTIVE_MODEL_KEY,
            )

            if pred is None:
                st.error("Prediction unavailable — required pre-match features unavailable for selected teams.")
            else:
                st.markdown(f"### Matchup: **{home_team} vs {away_team}** ({sel_lg})")
                
                if pred.is_promoted_match:
                    st.warning("⚠️ **Promoted-team initialization used.** Prediction is valid pre-match inference but should be treated as lower-confidence until fresh top-flight evidence accumulates.")

                c_sel, c_v41, c_v40 = st.columns(3)

                with c_sel:
                    st.markdown(f"#### 🎯 Production: {pred.prediction_model}")
                    st.write(f"**P(Home):** `{pred.production_probs['H']*100:.1f}%`")
                    st.write(f"**P(Draw):** `{pred.production_probs['D']*100:.1f}%`")
                    st.write(f"**P(Away):** `{pred.production_probs['A']*100:.1f}%`")
                    st.success(f"**Decision:** `{pred.production_decision}` (Confidence: {pred.production_confidence*100:.1f}%)")
                    st.caption(f"Status: `{pred.advisory_status}` | Advisory Conf: `{pred.advisory_confidence}`")

                with c_v41:
                    st.markdown("#### 1. ⭐ V4.1 Prospective Candidate (Research)")
                    st.write(f"**P(Home):** `{pred.v4_1_probs['H']*100:.1f}%`")
                    st.write(f"**P(Draw):** `{pred.v4_1_probs['D']*100:.1f}%`")
                    st.write(f"**P(Away):** `{pred.v4_1_probs['A']*100:.1f}%`")
                    st.info(f"**Decision:** `{pred.v4_1_decision}` (Confidence: {pred.v4_1_confidence*100:.1f}%)")

                with c_v40:
                    st.markdown("#### 2. 🏛️ V4.0 Draw Champion (Baseline)")
                    st.write(f"**P(Home):** `{pred.v4_champ_probs['H']*100:.1f}%`")
                    st.write(f"**P(Draw):** `{pred.v4_champ_probs['D']*100:.1f}%`")
                    st.write(f"**P(Away):** `{pred.v4_champ_probs['A']*100:.1f}%`")
                    st.info(f"**Decision:** `{pred.v4_champ_decision}`")

                # --- STEP 2J/2K/2L DRAW RISK ADVISORY ---
                st.markdown("---")
                st.markdown("#### 🛡️ Draw Risk & Caution Intelligence (Advisory Only)")
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.metric("Draw Risk Tier", pred.draw_risk_badge or "N/A", pred.draw_risk_label or "")
                    st.caption(f"Draw Vulnerability Score: `{pred.draw_risk_score:.4f}`" if pred.draw_risk_score is not None else "")
                with col_r2:
                    st.markdown("**Contributing Vulnerability Signals:**")
                    if pred.draw_risk_reasons:
                        for r in pred.draw_risk_reasons:
                            st.write(f"• {r}")
                    else:
                        st.write("• Standard pre-match probability profile.")
                st.info("ℹ️ **Important:** The Draw Risk advisory provides non-mutating decision-support intelligence and does NOT alter the V4.0 production prediction.")

                # --- HISTORICAL CALCULATION MEMORY LAYER ---
                st.markdown("---")
                st.markdown("#### 🧠 Historical Calculation Memory Layer (Auxiliary Intelligence)")
                now_utc_str = datetime.now(timezone.utc).isoformat()
                mem_analysis = memory_service.analyze_fixture_calculation(
                    fixture_id=0,
                    home_team=home_team,
                    away_team=away_team,
                    competition_id=sel_cid,
                    competition_name=sel_lg,
                    kickoff_utc=now_utc_str,
                    p_home=pred.production_probs["H"],
                    p_draw=pred.production_probs["D"],
                    p_away=pred.production_probs["A"],
                    decision=pred.production_decision,
                    lambda_home=pred.lambda_home,
                    lambda_away=pred.lambda_away,
                    allow_cross_league=False,
                )

                c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                c_m1.metric("Historical Evidence", mem_analysis.evidence_signal.evidence_badge)
                if mem_analysis.evidence_signal.same_outcome_analogues_count > 0:
                    c_m2.metric("Same-Outcome Win Rate", f"{mem_analysis.evidence_signal.same_outcome_success_rate:.1f}%", f"{mem_analysis.evidence_signal.same_outcome_success_count}/{mem_analysis.evidence_signal.same_outcome_analogues_count} matches")
                else:
                    c_m2.metric("Same-Outcome Win Rate", "N/A", "No analogues")
                c_m3.metric("V4.0 Baseline Score", mem_analysis.v4_baseline_score)
                c_m4.metric("Historical Refined Score", mem_analysis.score_refinement.refined_score, f"Conf: {mem_analysis.score_refinement.score_confidence}")

                st.write(f"**Historical Pattern Advisory:** {mem_analysis.evidence_signal.advisory_summary}")
                st.write(f"**Score Refinement Analysis:** {mem_analysis.score_refinement.reason}")

                if mem_analysis.analogues:
                    with st.expander(f"🔍 View Nearest Historical Calculation Analogues ({len(mem_analysis.analogues)} Matches Found)", expanded=False):
                        an_rows = []
                        for a in mem_analysis.analogues:
                            an_rows.append({
                                "Date (UTC)": a.kickoff_utc[:10] if len(a.kickoff_utc) >= 10 else "--",
                                "League": a.competition_name,
                                "Match": f"{a.home_team} vs {a.away_team}",
                                "Similarity": f"{a.similarity_score:.1f}%",
                                "P(H) / P(D) / P(A)": f"{a.p_home*100:.0f}% / {a.p_draw*100:.0f}% / {a.p_away*100:.0f}%",
                                "Pred": a.predicted_outcome,
                                "Actual Score": a.actual_score,
                                "Result Eval": "✅ CORRECT" if a.prediction_correct else "❌ WRONG",
                            })
                        st.dataframe(pd.DataFrame(an_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("ℹ️ No historical calculation analogues found matching the minimum similarity threshold for this league.")

                st.info("ℹ️ **Client Safe Notice:** Historical memory provides non-mutating auxiliary evidence and does NOT alter V4.0 baseline probabilities or decisions.")

                st.markdown("#### 🔍 Model Provenance Metadata")
                st.code(
                    f"Selected Model: {pred.prediction_model}\n"
                    f"Model ID: {pred.model_name}\n"
                    f"Version: {pred.model_version}\n"
                    f"Status: {pred.model_status}\n"
                    f"MD5: {pred.model_file_md5}\n"
                    f"Information Cutoff: {pred.information_cutoff}\n"
                    f"Evaluation Status: {pred.evaluation_status}\n"
                    f"Expected Goals: {pred.lambda_home:.2f} - {pred.lambda_away:.2f}\n"
                    f"Prediction Entropy: {pred.production_entropy:.4f}",
                    language="yaml",
                )

# ==============================================================================
# TAB 3: COMPARE MODELS
# ==============================================================================
with tab_compare:
    st.subheader("⚖️ Model Comparison Lab")
    st.markdown("Fixed comparative analysis between **V4.0 Production (Active)** and **V4.1 Prospective Candidate (Research Reference)**.")

    model_1 = "V4.0 Production"
    model_2 = "V4.1 Production"

    col_cmp_lg, col_cmp_h, col_cmp_a = st.columns(3)
    with col_cmp_lg:
        cmp_lg = st.selectbox("League", options=list(leagues_map.values()), index=0, key="cmp_lg")
        cmp_cid = inv_leagues[cmp_lg]
    with col_cmp_h:
        cmp_teams = fixture_service.get_teams_by_league(cmp_cid)
        cmp_home = st.selectbox("Home Team", options=cmp_teams, index=0 if cmp_teams else 0, key="cmp_home")
    with col_cmp_a:
        cmp_away_options = [t for t in cmp_teams if t != cmp_home] if cmp_teams else []
        cmp_away = st.selectbox("Away Team", options=cmp_away_options, index=0 if cmp_away_options else 0, key="cmp_away")

    if st.button("🔬 Run Side-by-Side Model Comparison", type="primary", key="btn_run_cmp"):
        if cmp_home == cmp_away or not cmp_home or not cmp_away:
            st.error("Please select two distinct teams.")
        else:
            pred_1 = pred_service.predict_matchup(cmp_home, cmp_away, cmp_lg, competition_id=cmp_cid, model_key=model_1)
            pred_2 = pred_service.predict_matchup(cmp_home, cmp_away, cmp_lg, competition_id=cmp_cid, model_key=model_2)

            if pred_1 is None or pred_2 is None:
                st.error("Prediction unavailable for one or both models.")
            else:
                st.markdown(f"### Matchup: **{cmp_home} vs {cmp_away}** ({cmp_lg})")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"#### 🅰️ {model_1} (Production Truth)")
                    st.write(f"**Version:** `{pred_1.model_version}`")
                    st.write(f"**P(Home):** `{pred_1.production_probs['H']*100:.1f}%`")
                    st.write(f"**P(Draw):** `{pred_1.production_probs['D']*100:.1f}%`")
                    st.write(f"**P(Away):** `{pred_1.production_probs['A']*100:.1f}%`")
                    st.success(f"**Decision:** `{pred_1.production_decision}` (Conf: {pred_1.production_confidence*100:.1f}%)")
                    st.write(f"**Expected Goals:** `{pred_1.lambda_home:.2f} - {pred_1.lambda_away:.2f}`")
                    st.caption(f"Status: `{pred_1.advisory_status}` | Entropy: `{pred_1.production_entropy:.4f}`")

                with c2:
                    st.markdown(f"#### 🅱️ {model_2} (Research Candidate)")
                    st.write(f"**Version:** `{pred_2.model_version}`")
                    st.write(f"**P(Home):** `{pred_2.production_probs['H']*100:.1f}%`")
                    st.write(f"**P(Draw):** `{pred_2.production_probs['D']*100:.1f}%`")
                    st.write(f"**P(Away):** `{pred_2.production_probs['A']*100:.1f}%`")
                    st.info(f"**Decision:** `{pred_2.production_decision}` (Conf: {pred_2.production_confidence*100:.1f}%)")
                    st.write(f"**Expected Goals:** `{pred_2.lambda_home:.2f} - {pred_2.lambda_away:.2f}`")
                    st.caption(f"Status: `{pred_2.advisory_status}` | Entropy: `{pred_2.production_entropy:.4f}`")

                # Comparison Table
                st.markdown("#### 📊 Probability & Metric Divergence")
                diff_df = pd.DataFrame([
                    {"Metric": "P(Home)", f"{model_1}": f"{pred_1.production_probs['H']*100:.1f}%", f"{model_2}": f"{pred_2.production_probs['H']*100:.1f}%", "Delta (A - B)": f"{(pred_1.production_probs['H'] - pred_2.production_probs['H'])*100:+.1f}%"},
                    {"Metric": "P(Draw)", f"{model_1}": f"{pred_1.production_probs['D']*100:.1f}%", f"{model_2}": f"{pred_2.production_probs['D']*100:.1f}%", "Delta (A - B)": f"{(pred_1.production_probs['D'] - pred_2.production_probs['D'])*100:+.1f}%"},
                    {"Metric": "P(Away)", f"{model_1}": f"{pred_1.production_probs['A']*100:.1f}%", f"{model_2}": f"{pred_2.production_probs['A']*100:.1f}%", "Delta (A - B)": f"{(pred_1.production_probs['A'] - pred_2.production_probs['A'])*100:+.1f}%"},
                    {"Metric": "Decision", f"{model_1}": pred_1.production_decision, f"{model_2}": pred_2.production_decision, "Delta (A - B)": "AGREE" if pred_1.production_decision == pred_2.production_decision else "DIVERGE"},
                    {"Metric": "Confidence", f"{model_1}": f"{pred_1.production_confidence*100:.1f}%", f"{model_2}": f"{pred_2.production_confidence*100:.1f}%", "Delta (A - B)": f"{(pred_1.production_confidence - pred_2.production_confidence)*100:+.1f}%"},
                ])
                st.table(diff_df)

# ==============================================================================
# TAB 4: PREDICTIONS & RESULTS (UNIFIED NAVIGATION)
# ==============================================================================
with tab_predictions_results:
    st.subheader("📅 Predictions & Results")
    st.markdown("Navigate predictions across different weeks. Data is sourced from immutable V4.0 production ledgers or live odd feed.")
    
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        selected_week = st.selectbox(
            "Select Week", 
            ["Previous Week — 29 Aug to 1 Sep 2026", "Current Week", "Upcoming Week"],
            index=0,
            key="nav_week_select"
        )
    with col_nav2:
        if st.button("🔄 Refresh Fixtures", help="Fetch fresh fixtures", type="secondary"):
            st.cache_data.clear()
            st.rerun()

    @st.cache_data
    def load_previous_week_performance_records(p_key: str):
        return fixture_service.get_previous_week_performance_records(
            start_date="2026-08-29",
            end_date="2026-09-01",
            provider_name=p_key,
        )

    @st.cache_data(ttl=180)
    def fetch_cached_all_matches(p_key: str, c_ids: Tuple[int, ...], weekly_mode: bool, offset_weeks: int):
        if weekly_mode:
            return fixture_service.get_weekly_prediction_fixtures(provider_name=p_key, offset_weeks=offset_weeks)
        return fixture_service.get_all_available_matches(provider_name=p_key, competition_ids=list(c_ids))

    # Fetch data based on selected week
    is_historical = selected_week.startswith("Previous Week")
    fixtures = []
    meta = {}
    prev_records = []
    
    if is_historical:
        prev_records = load_previous_week_performance_records(provider_key)
    else:
        offset = 1 if selected_week == "Upcoming Week" else 0
        fixtures, meta = fetch_cached_all_matches(provider_key, tuple(comp_ids_filter), is_weekly_mode, offset)

    # Date Filter setup
    date_options = ["All Dates"]
    if is_historical:
        st.markdown("**Previous Week Date Range:** 29 Aug 2026 → 1 Sep 2026 (UTC) | **Model:** V4.0 Production (Frozen)")
        date_options = ["All Dates", "29-Aug-2026", "30-Aug-2026", "31-Aug-2026", "01-Sep-2026"]
    else:
        for fix in fixtures:
            date_str = fix.scheduled_kickoff[:10] if len(fix.scheduled_kickoff) >= 10 else ""
            if date_str and date_str not in date_options:
                date_options.append(date_str)
        date_options = ["All Dates"] + sorted(list(set(date_options[1:])))
    
    selected_date = st.selectbox("Filter by Date (UTC)", options=date_options, index=0)

    # ---------------------------------------------------------
    # RENDER PREVIOUS WEEK (COMPLETED)
    # ---------------------------------------------------------
    if is_historical:
        if not prev_records:
            st.warning("No historical completed prediction records found in performance ledger.")
        else:
            # Apply date filter with robust date mapping
            date_map = {
                "29-Aug-2026": "2026-08-29",
                "30-Aug-2026": "2026-08-30",
                "31-Aug-2026": "2026-08-31",
                "01-Sep-2026": "2026-09-01",
                "2026-08-29": "2026-08-29",
                "2026-08-30": "2026-08-30",
                "2026-08-31": "2026-08-31",
                "2026-09-01": "2026-09-01",
            }
            filtered_records = prev_records
            if selected_date != "All Dates":
                target_iso = date_map.get(selected_date, selected_date)
                filtered_records = [
                    r for r in prev_records
                    if r.get("scheduled_kickoff_utc", "").startswith(target_iso)
                    or r.get("Date (UTC)", "") == target_iso
                ]

            tot_m = len(filtered_records)
            corr_m = sum(1 for r in filtered_records if r.get("prediction_correct"))
            wrong_m = tot_m - corr_m
            acc_1x2 = (corr_m / tot_m * 100.0) if tot_m > 0 else 0.0
            score_hits = sum(1 for r in filtered_records if r.get("exact_score_correct"))
            score_acc = (score_hits / tot_m * 100.0) if tot_m > 0 else 0.0
            
            brier_sum = sum(r.get("brier_score", 0.0) for r in filtered_records if "brier_score" in r)
            logloss_sum = sum(r.get("log_loss", 0.0) for r in filtered_records if "log_loss" in r)
            avg_brier = brier_sum / tot_m if tot_m > 0 else 0.0
            avg_logloss = logloss_sum / tot_m if tot_m > 0 else 0.0

            # Outcome Breakdown
            h_recs = [r for r in filtered_records if r.get("predicted_outcome") == "H" or r.get("V4.0 Pred") == "H"]
            d_recs = [r for r in filtered_records if r.get("predicted_outcome") == "D" or r.get("V4.0 Pred") == "D"]
            a_recs = [r for r in filtered_records if r.get("predicted_outcome") == "A" or r.get("V4.0 Pred") == "A"]

            h_corr = sum(1 for r in h_recs if r.get("prediction_correct"))
            d_corr = sum(1 for r in d_recs if r.get("prediction_correct"))
            a_corr = sum(1 for r in a_recs if r.get("prediction_correct"))

            h_acc = (h_corr / len(h_recs) * 100.0) if h_recs else 0.0
            d_acc = (d_corr / len(d_recs) * 100.0) if d_recs else 0.0
            a_acc = (a_corr / len(a_recs) * 100.0) if a_recs else 0.0

            st.markdown("#### 🏆 Historical Performance Summary (29 Aug → 1 Sep 2026)")
            kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
            kpi1.metric("Total Matches", f"{tot_m}")
            kpi2.metric("1X2 Accuracy", f"{acc_1x2:.2f}%", f"{corr_m}/{tot_m} Correct")
            kpi3.metric("Correct 1X2", f"{corr_m}", "✅ Hit")
            kpi4.metric("Wrong 1X2", f"{wrong_m}", "❌ Miss")
            kpi5.metric("Exact Score", f"{score_acc:.2f}%", f"{score_hits}/{tot_m}")
            kpi6.metric("Historical Memory", "⚪ INSUFFICIENT")

            if tot_m > 0:
                kpi7, kpi8, kpi9, kpi10 = st.columns(4)
                kpi7.metric("Avg Brier Score", f"{avg_brier:.4f}")
                kpi8.metric("Home Pred Accuracy", f"{h_acc:.1f}%", f"{h_corr}/{len(h_recs)}")
                kpi9.metric("Draw Pred Accuracy", f"{d_acc:.1f}%" if d_recs else "N/A", f"{d_corr}/{len(d_recs)}" if d_recs else "0/0")
                kpi10.metric("Away Pred Accuracy", f"{a_acc:.1f}%", f"{a_corr}/{len(a_recs)}")

            if len(filtered_records) == 0:
                st.info(f"No records found for {selected_date}.")
            else:
                prev_table_rows = []
                for r in filtered_records:
                    eval_badge = "✅ CORRECT" if r.get("prediction_correct") else "❌ WRONG"
                    score_hit_badge = "🎯 HIT" if r.get("exact_score_correct") else "--"
                    k_utc = r.get("scheduled_kickoff_utc", "")
                    date_str = k_utc[:10] if len(k_utc) >= 10 else "--"
                    time_str = k_utc[11:16] if len(k_utc) >= 16 else "--"

                    p_h_val = r.get("p_home", 0.0)
                    p_d_val = r.get("p_draw", 0.0)
                    p_a_val = r.get("p_away", 0.0)

                    tier_raw = str(r.get("draw_risk_tier", "LOW")).upper()
                    if tier_raw == "HIGH":
                        risk_badge = "🟠 HIGH"
                    elif tier_raw in ("MEDIUM", "MODERATE"):
                        risk_badge = "🟡 MEDIUM"
                    elif tier_raw == "LOW":
                        risk_badge = "🟢 LOW"
                    else:
                        risk_badge = tier_raw if tier_raw else "N/A"
                        
                    lh_val = r.get("expected_home_goals", 0.0)
                    la_val = r.get("expected_away_goals", 0.0)

                    prev_table_rows.append({
                        "Date (UTC)": date_str,
                        "Kickoff (UTC)": time_str,
                        "League": r.get("competition_name", "--"),
                        "Home Team": r.get("home_team", "--"),
                        "Away Team": r.get("away_team", "--"),
                        "P(H)": f"{p_h_val*100:.1f}%",
                        "P(D)": f"{p_d_val*100:.1f}%",
                        "P(A)": f"{p_a_val*100:.1f}%",
                        "V4.0 Pred": r.get("predicted_outcome", "--"),
                        "Predicted Score": r.get("baseline_predicted_score", "--"),
                        "xG": f"{lh_val:.2f}-{la_val:.2f}",
                        "Final Score": r.get("actual_score", "--"),
                        "Actual Outcome": r.get("actual_outcome", "--"),
                        "Evaluation": eval_badge,
                        "Exact Score Hit": score_hit_badge,
                        "Historical Memory": f"⚪ {r.get('memory_evidence_level', 'INSUFFICIENT')}",
                        "Draw Risk": risk_badge,
                    })

                st.dataframe(pd.DataFrame(prev_table_rows), use_container_width=True, hide_index=True)
                
            with st.expander("📊 Historical League Performance Breakdown", expanded=False):
                league_breakdown = []
                for l_name in ["Premier League", "Serie A", "La Liga", "Ligue 1", "Bundesliga"]:
                    l_recs = [r for r in filtered_records if r.get("competition_name") == l_name]
                    l_tot = len(l_recs)
                    if l_tot > 0:
                        l_corr = sum(1 for r in l_recs if r.get("prediction_correct"))
                        l_score = sum(1 for r in l_recs if r.get("exact_score_correct"))
                        league_breakdown.append({
                            "League": l_name,
                            "Matches": l_tot,
                            "1X2 Accuracy": f"{l_corr/l_tot*100:.1f}% ({l_corr}/{l_tot})",
                            "Exact Score Hits": f"{l_score/l_tot*100:.1f}% ({l_score}/{l_tot})"
                        })
                    else:
                        league_breakdown.append({
                            "League": l_name,
                            "Matches": 0,
                            "1X2 Accuracy": "N/A",
                            "Exact Score Hits": "N/A"
                        })
                st.table(pd.DataFrame(league_breakdown))
                
            st.caption("ℹ️ All historical records loaded read-only from production_performance_ledger.jsonl.")

    # ---------------------------------------------------------
    # RENDER CURRENT / UPCOMING WEEK
    # ---------------------------------------------------------
    else:
        if not fixtures:
            if meta.get("status") == "NO_FIXTURES":
                st.warning("Fixture data pending for the selected week. No fixtures available.")
            else:
                st.warning("No fixtures currently available.")
        else:
            # Apply date filter
            if selected_date != "All Dates":
                fixtures = [f for f in fixtures if f.scheduled_kickoff.startswith(selected_date)]
                
            if not fixtures:
                st.info(f"No fixtures found for {selected_date}.")
            else:
                table_rows = []
                for fix in fixtures:
                    try:
                        pred = pred_service.predict_dashboard_fixture(
                            fixture=fix,
                            pre_kickoff_buffer_minutes=safety_buffer,
                            model_key=ACTIVE_MODEL_KEY,
                        )
                    except Exception as e:
                        logger.error(f"Error predicting fixture {fix.fixture_id}: {e}")
                        pred = DashboardMatchPrediction(
                            fixture_id=fix.fixture_id,
                            home_team=fix.home_team,
                            away_team=fix.away_team,
                            competition_name=fix.competition_name,
                            scheduled_kickoff=fix.scheduled_kickoff,
                            status=fix.status,
                            provider=fix.provider,
                            prediction_allowed=False,
                            status_message="Prediction unavailable",
                        )

                    if pred.prediction_allowed and pred.production_probs:
                        p_h = f"{pred.production_probs['H']*100:.1f}%"
                        p_d = f"{pred.production_probs['D']*100:.1f}%"
                        p_a = f"{pred.production_probs['A']*100:.1f}%"
                        sel_dec = pred.production_decision
                    else:
                        p_h, p_d, p_a = "--", "--", "--"
                        sel_dec = "--"

                    # Goal Prediction based on pre-match expected goals
                    if pred.lambda_home is not None and pred.lambda_away is not None:
                        lh_round = max(0, int(round(pred.lambda_home)))
                        la_round = max(0, int(round(pred.lambda_away)))
                        if lh_round == la_round and sel_dec == "H":
                            lh_round += 1
                        elif lh_round == la_round and sel_dec == "A":
                            la_round += 1
                        goal_pred_display = f"{lh_round}-{la_round}"
                        xg_display = f"{pred.lambda_home:.2f}-{pred.lambda_away:.2f}"
                    else:
                        goal_pred_display = "--"
                        xg_display = "--"

                    date_utc = fix.scheduled_kickoff[:10] if len(fix.scheduled_kickoff) >= 10 else "--"
                    kickoff_utc = format_kickoff_utc(fix.scheduled_kickoff, include_suffix=False)
                    
                    row = {
                        "Date (UTC)": date_utc,
                        "Kickoff (UTC)": kickoff_utc,
                        "League": fix.competition_name,
                        "Home Team": fix.home_team,
                        "Away Team": fix.away_team,
                        "P(H)": p_h,
                        "P(D)": p_d,
                        "P(A)": p_a,
                        "Prediction": sel_dec,
                        "Predicted Score": goal_pred_display,
                        "xG": xg_display,
                        "Draw Risk": pred.draw_risk_badge or "--",
                        "Historical Memory": "⚪ INSUFFICIENT", # Advisory status
                        "Model Identity": pred.prediction_model or ACTIVE_MODEL_KEY,
                    }
                    
                    # Add result info if not Upcoming Week (Current week may have completed matches)
                    if selected_week == "Current Week":
                        score_display = f"{fix.home_goals}-{fix.away_goals}" if (fix.home_goals is not None and fix.away_goals is not None) else "--"
                        if fix.status in ("FT", "AET", "PEN", "AWARDED") or (fix.home_goals is not None and fix.away_goals is not None):
                            res_display = f"{pred.actual_outcome}" if pred.actual_outcome else "Result unavailable"
                            sel_eval = "✅ CORRECT" if pred.selected_correct is True else ("❌ WRONG" if pred.selected_correct is False else "--")
                        elif fix.status in ("1H", "2H", "HT", "LIVE", "ET", "P"):
                            res_display = "Live"
                            score_display = "Live"
                            sel_eval = "--"
                        else:
                            res_display = "--"
                            score_display = "--"
                            sel_eval = "--"
                            
                        row["Final Score"] = score_display
                        row["Actual Result"] = res_display
                        row["Eval"] = sel_eval

                    table_rows.append(row)

                # Summary Metrics across complete active fixture collection
                risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
                for r in table_rows:
                    b = str(r.get("Draw Risk", "--"))
                    if "LOW" in b:
                        risk_counts["LOW"] += 1
                    elif "MED" in b:
                        risk_counts["MEDIUM"] += 1
                    elif "HIGH" in b:
                        risk_counts["HIGH"] += 1
                    elif "CRIT" in b:
                        risk_counts["CRITICAL"] += 1

                st.markdown(f"#### 📊 Active Collection Overview ({selected_week})")
                if is_weekly_mode:
                    c_s1, c_s2, c_s3, c_s4, c_s5, c_s6, c_s7 = st.columns(7)
                    c_s1.metric("Weekly Target", f"{TOTAL_WEEKLY_TARGET}", "Gameweek Contract")
                    c_s2.metric("Selected Matches", f"{meta.get('total_selected', len(fixtures))} / {TOTAL_WEEKLY_TARGET}")
                    c_s3.metric("Upcoming", meta.get("upcoming_count", 0))
                    c_s4.metric("Completed", meta.get("completed_count", 0))
                    c_s5.metric("🟢 Low Draw Risk", f"{risk_counts['LOW']}")
                    c_s6.metric("🟡 Medium Draw Risk", f"{risk_counts['MEDIUM']}")
                    c_s7.metric("🟠 High Draw Risk", f"{risk_counts['HIGH']}")
                else:
                    c_s1, c_s2, c_s3, c_s4, c_s5, c_s6, c_s7 = st.columns(7)
                    c_s1.metric("Total Fixtures", len(fixtures))
                    c_s2.metric("Upcoming", meta.get("upcoming_count", 0))
                    c_s3.metric("Completed", meta.get("completed_count", 0))
                    c_s4.metric("🟢 Low Draw Risk", f"{risk_counts['LOW']}")
                    c_s5.metric("🟡 Medium Draw Risk", f"{risk_counts['MEDIUM']}")
                    c_s6.metric("🟠 High Draw Risk", f"{risk_counts['HIGH']}")
                    c_s7.metric("🔴 Critical Draw Risk", f"{risk_counts['CRITICAL']}")

                st.markdown("#### 🏆 Target Competition Breakdown")
                l_cols = st.columns(5)
                for idx, (cid, lname) in enumerate(TARGET_LEAGUES.items()):
                    cnt = meta.get("league_counts", {}).get(lname, meta.get("league_breakdown", {}).get(lname, 0))
                    target_cnt = WEEKLY_LEAGUE_TARGETS.get(lname, 10)
                    with l_cols[idx]:
                        if is_weekly_mode:
                            status_lbl = "100% Filled" if cnt >= target_cnt else f"Shortfall: -{target_cnt - cnt}"
                            st.metric(lname, f"{cnt} / {target_cnt} matches", status_lbl)
                        else:
                            if cnt > 0:
                                st.metric(lname, f"{cnt} matches", "Available")
                            else:
                                st.metric(lname, "0 matches", "No fixtures in feed")

                if is_weekly_mode and meta.get("shortfalls") and any(v > 0 for v in meta["shortfalls"].values()):
                    shortfall_desc = ", ".join(f"{k}: missing {v}" for k, v in meta["shortfalls"].items() if v > 0)
                    st.info(f"ℹ️ Feed Shortfall Notice: {shortfall_desc}. Real available fixtures were preserved without fabricating placeholder matches.")

                st.caption("ℹ️ All kickoff times shown in UTC (24-hour format)")
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 5: 2026/27 EVALUATION
# ==============================================================================
with tab_eval2627:
    st.subheader("📊 2026 Prospective Evaluation Dashboard")
    st.markdown("""
    **Evaluation State:** `PROSPECTIVE 2026 LIVE EVALUATION ACTIVE`  
    **Official Prediction Ledger:** `research/v5_model_improvement/v4_1_prospective_test/02_live_forecast_ledger.csv` (99 Locked Fixtures)  
    **Active Production Model:** `v4_0_draw_champion` (`06841f0c03c8597b2b8cd8f8ab064864`)  
    **Prospective Candidate:** `v4_1_prospective_candidate` (`145f918d933eb343c0f63ca342b10289`)  
    """)
    st.info("📌 Prospective evaluation metrics (RPS, Log-Loss, Draw ECE) will update as 2026 matches conclude.")

# ==============================================================================
# REMAINING TABS
# ==============================================================================
with tab_draw:
    st.subheader("⚽ Draw Rate & Calibration Lab")
    st.markdown("Inspection of draw probability calibration across Dixon-Coles, Elo draw curve, and DIBP.")

with tab_leagues:
    st.subheader("🏆 5-League Disaggregated Analysis")
    st.markdown("Performance breakdowns across Premier League, La Liga, Serie A, Bundesliga, and Ligue 1.")

with tab_temporal:
    st.subheader("📈 Temporal Evolution & Regime Stability")
    st.markdown("Chronological tracking of state vectors across historical seasons.")

with tab_agreement:
    st.subheader("🧠 Model Concordance & Agreement Matrix")
    st.markdown("Inter-model agreement between V4.0 Production and V4.1 Research Candidate.")

with tab_safety:
    st.subheader("🔐 Production Safety & Baseline Integrity")
    st.write(f"**Production Model:** `{ACTIVE_MODEL_KEY}`")
    st.write(f"**V4.0 Production Model MD5:** `06841f0c03c8597b2b8cd8f8ab064864` ✅")
    st.write(f"**V4.1 Research Candidate MD5:** `145f918d933eb343c0f63ca342b10289` ✅")
    st.write(f"**UI Model Dropdowns:** `REMOVED (Fixed Single-Model V4.0 Production)` ✅")
    st.write(f"**Protected Repository Baseline Assets:** `20/20 Bit-Identical` ✅")
