# Phase 16 — V4 + Sportmonks Research Arm

## ⚠️ RESEARCH ONLY — NOT PRODUCTION

**Cohort:** 2026-09-18 → 2026-09-21  
**V4 SHA256:** `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`  
**Status:** TEMPORAL_UNKNOWN (Sportmonks data fetched post-kickoff)

## Files

| File | Description |
|------|-------------|
| `01_phase16_fixture_universe.jsonl` | Full 48-fixture universe with V4 + SM + actuals joined |
| `02_sportmonks_raw_predictions.jsonl` | Per-fixture Sportmonks extracted predictions |
| `03_phase16_source_validation.json` | Data quality and validation report |
| `04_phase16_v4_baseline.json` | V4-only metrics (arm A) |
| `05_phase16_sportmonks_only.json` | Sportmonks-only metrics |
| `06_phase16_blend_results.json` | All 5 blend method results |
| `07_phase16_draw_analysis.json` | Deep draw prediction analysis |
| `08_phase16_signal_analysis.json` | 2-0 profile and strong home signal analysis |
| `09_phase16_comparison.json` | Full method comparison with governance decision |
| `10_phase16_report.md` | Complete research report |
| `11_phase16_dashboard.html` | Offline research dashboard |

## Critical Constraints

- **V4 model NOT modified.** SHA256: `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`
- **Sportmonks is RESEARCH ONLY.** Not promoted to production.
- **All data is TEMPORAL_UNKNOWN.** Cannot be used as clean prospective evidence.
- **Production predictions unchanged.** No modification to Phase 15 ledger.
- **No leakage.** Post-kickoff data explicitly labeled.

## Governance

> Sportmonks remains RESEARCH ONLY.

Continue prospective V4 vs V4+Sportmonks validation by capturing
Sportmonks predictions BEFORE kickoff for the next upcoming cohort.
