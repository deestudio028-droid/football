# Handover Manifest — Included File Inventory

This manifest catalogues every file included in the handover package, detailing its relative path, category, purpose, and rationale for the external ML audit.

| Relative Path | Category | Purpose | Why Needed by Freelancer |
|---|---|---|---|
| `README.md` | Documentation | Handover guide and architecture documentation | Overall project setup and execution instructions |
| `HANDOVER_MANIFEST.md` | Documentation | File-by-file inventory manifest | Verification of package contents |
| `SECURITY_AUDIT.md` | Documentation | Security & secret sanitization report | Proof of zero-secret packaging |
| `requirements.txt` | Configuration | Python dependencies list | Dependency installation |
| `config/competitions.json` | Configuration | League metadata and season mappings | Resolves league IDs to canonical competition names |
| `config/.env.example` | Configuration | Sanitized environment template | Example config without private secrets |
| `data/models/v4_poisson_venue_elo_online_ad.pkl` | Model Artifact | Frozen V4 model bundle | Deserialized by V4 loader for baseline inference |
| `data/processed/matches.db` | Data (SQLite) | Canonical match results database | Provides match fixtures, scores, team IDs for 5 leagues |
| `data/processed/features.db` | Data (SQLite) | Pre-match causal feature store | Provides pre-computed rolling features for V4 |
| `src/models/draw_champion.py` | Source Code | Production Draw Champion layer (v4.0) | Authoritative production implementation |
| `src/models/draw_champion_v41.py` | Source Code | Candidate Draw Champion layer (v4.1) | Isolated candidate implementation for calibration testing |
| `src/models/v4_artifact.py` | Source Code | V4 model artifact loader & contract | Loads and evaluates V4 model pipeline |
| `src/models/poisson.py` | Source Code | Poisson distribution math & grids | Probability simplex redistribution and modal scorelines |
| `src/models/baselines.py` | Source Code | Baseline classes & constants | Shared `CLASS_ORDER` ('H', 'D', 'A') |
| `src/models/config.py` | Source Code | Model configurations and season lists | Season IDs and training split constants |
| `src/models/data.py` | Source Code | Supervised dataset loader | Loads features and metadata from `features.db` |
| `src/models/splits.py` | Source Code | Train/validation split logic | Historical train split definitions |
| `src/features/elo.py` | Source Code | Pre-match Elo rating calculation | Chronological causal Elo feature extraction |
| `src/features/online_attack_defense.py` | Source Code | Online Attack/Defense Poisson state | Causal online team rating updates |
| `src/features/feature_builder.py` | Source Code | Feature extraction orchestrator | Feature engineering pipeline |
| `src/features/storage.py` | Source Code | Feature database read/write helpers | SQLite interface for feature storage |
| `src/monitoring/prospective_pipeline.py` | Source Code | Prospective store and validation monitor | Audit trail and metric calculations |
| `src/monitoring/prospective_pilot.py` | Source Code | Prospective pilot harness | Operational validation milestones |
| `research/v4_promotion/draw_champion_method_frozen.json` | Specification | Frozen Champion parameters | MD5-pinned reference specification |
| `research/v4_promotion/prospective_validation_protocol.json` | Specification | Frozen validation protocol | Protocol rules and thresholds |
| `research/v4_promotion/fresh_100_fixture_ids.json` | Cohort Data | Fresh 100 pilot fixture IDs | Reused diagnostic cohort definition |
| `research/v4_promotion/fresh_extended_fixture_ids.json` | Cohort Data | Fresh Extended 300 fixture IDs | Reused research OOS cohort definition |
| `research/v4_promotion/v4_50_validation_results.json` | Cohort Data | Frozen 50 fixture IDs | Reused historical validation cohort definition |
| `research/v4_promotion/draw_champion_1301_evaluation.py` | Research Code | 1,301-match diagnostic script | Replays and evaluates all 1,301 matches |
| `research/v4_promotion/draw_champion_1301_results.json` | Evaluation Data | Full 1,301 scorecard and metrics | Benchmark results for V4 and Champion |
| `research/v4_promotion/draw_champion_1301_report.md` | Evaluation Report | 1,301 diagnostic report | Executive summary of 1,301 evaluation |
| `research/v4_promotion/draw_champion_v41_candidate_analysis.py` | Research Code | Candidate sensitivity script | Tests candidate intercepts, curves, bootstrap |
| `research/v4_promotion/draw_champion_v41_candidate_results.json` | Evaluation Data | Candidate sensitivity results JSON | Machine-readable candidate metrics |
| `research/v4_promotion/draw_champion_v41_candidate_report.md` | Evaluation Report | Candidate analysis report | Detailed candidate evaluation findings |
| `research/v4_promotion/season_2025_26_dataset_inventory.py` | Research Code | 2025/26 dataset audit script | Full inventory of 1,752 season fixtures |
| `research/v4_promotion/final_production_e2e_smoke_test.py` | Verification Code | End-to-end production smoke test | Validates full inference path on synthetic fixture |
| `tests/test_draw_champion_production.py` | Test Code | Production Champion unit test suite | 58 invariant and golden tests |
| `tests/test_draw_champion_v41.py` | Test Code | Candidate v4.1 unit test suite | 13 candidate invariant tests |
| `tests/test_draw_champion_1301_evaluation.py` | Test Code | 1,301 evaluation test suite | 20 evaluation invariant tests |
| `tests/test_2025_26_dataset_inventory.py` | Test Code | Dataset inventory test suite | 15 inventory and cohort overlap tests |
| `tests/test_prospective_validation_pipeline.py` | Test Code | Prospective pipeline test suite | 30 contract and security tests |
| `tests/test_prospective_operational_collection.py` | Test Code | Operational collection test suite | 30 operational safety tests |
| `tests/test_fresh_100_prospective_pilot.py` | Test Code | Pilot harness test suite | 20 pilot invariant tests |
