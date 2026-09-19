# Phase 15.1 — Live Score Integration Directory

## Overview
This directory contains the audit, verification, and schema validation artifacts for Phase 15.1, integrating real-time live match score and status updates into the 18–21 Sep 2026 Big-5 prediction dashboard via the existing deployed Railway backend.

## Directory Manifest
- `01_live_integration_audit.md`: Architecture audit of the Railway FastAPI backend (`/api/live-scores`), CORS headers, failure recovery, and status mapping.
- `02_live_schema_validation.json`: Captured live response schema validation, proving connection to `https://web-production-d8a09.up.railway.app/api/live-scores`.
- `03_static_prediction_integrity.json`: Deterministic cryptographic snapshot proving 100% immutability of the 48 static prediction rows.
- `04_live_integration_test_report.json`: Execution results of the Phase 15.1 live integration test suite.
- `README.md`: This directory manifest.

## Invariants & Guardrails
1. **Zero New Backends**: Reuses the deployed Railway live-score backend (`https://web-production-d8a09.up.railway.app`).
2. **Prediction Immutability**: All prediction columns ($P(H), P(D), P(A)$, PRED, Predicted Score, Draw Risk, Signals) are frozen at generation and cannot be modified by live scores.
3. **Primary Key Matching**: Fixtures match strictly on `fixture_id` with fallback to `date + home_team + away_team`.
4. **Resilient Fail-Open**: In the event of API timeout or network failure, static predictions remain fully functional and visible.
5. **No Frontend Secrets**: Zero API tokens or credentials in client HTML/JS.
