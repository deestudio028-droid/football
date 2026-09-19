# Phase 15 — Upcoming Big-5 Prediction Dashboard Directory

## Overview
This directory contains the complete Phase 15 prospective prediction batch and validation artifacts for the upcoming Big-5 European matchweek (18 Sep 2026 → 21 Sep 2026).

## Directory Manifest
- `01_upcoming_fixture_ledger.jsonl`: Full machine-readable ledger of all 48 upcoming fixtures with canonical V4 predictions.
- `02_2_0_signal_ledger.jsonl`: Filtered ledger of prioritized 2-0 and Strong Home signals.
- `03_phase15_validation.json`: Machine-readable validation metrics, counts, and cryptographic hashes.
- `04_phase15_report.md`: Comprehensive executive report and match breakdown.
- `05_phase15_dashboard.html`: Standalone, self-contained HTML prediction dashboard with Phase 15.1 live score polling.
- `phase15_engine.py`: Self-contained script to verify invariants and regenerate all Phase 15 deliverables.

## Cryptographic Verification
- Frozen V4 Model: `data/models/v4_poisson_venue_elo_online_ad.pkl`
- Expected SHA256: `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`
