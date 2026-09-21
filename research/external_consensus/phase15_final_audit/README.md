# Phase 15 Final Results Audit

**Cohort:** 2026-09-18 → 2026-09-21  
**Big-5 Leagues:** Premier League, La Liga, Serie A, Bundesliga, Ligue 1  
**V4 SHA256:** `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`  
**Status:** ANALYSIS ONLY

## Files

| File | Description |
|------|-------------|
| `01_phase15_final_results_ledger.jsonl` | Joined predictions + actuals for all 48 fixtures |
| `02_phase15_metrics.json` | Core metrics (1X2, Brier, LogLoss, calibration, etc.) |
| `03_phase15_risk_analysis.json` | Draw risk tier analysis |
| `04_phase15_signal_analysis.json` | 2-0 profile and strong home signal analysis |
| `05_phase15_league_analysis.json` | Per-league accuracy breakdown |
| `06_phase15_forensics.json` | Biggest misses, miss analysis |
| `07_phase15_final_report.md` | Full markdown audit report |
| `08_phase15_final_dashboard.html` | Interactive HTML dashboard |

## Critical Constraints

- V4 model was NOT modified, retrained, or re-parameterized.
- All predictions are pre-kickoff, pre-locked.
- Postponed fixtures are NOT counted as failures.
- Actual results sourced from OddAlerts /fixtures/multiple.
- Sportmonks data NOT included (excluded by design).
