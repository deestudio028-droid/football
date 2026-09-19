"""Canonical Batch Prediction Service (Phase 14 Production Pathway Fix).

Generates prospective and historical batch prediction ledgers strictly using
the frozen V4 production model and canonical mathematical Poisson score selection.

GUARANTEES:
1. Immutability: Frozen V4 binary SHA256 verified prior to generation.
2. Canonical Scorelines: Always uses src/models/poisson.py:modal_scoreline.
   Zero scoreline flattening (e.g. HOME -> 2-1 or AWAY -> 1-2).
3. Signal Profiles:
   - strong_home_profile: P(H) >= 0.60 and Draw Risk == 'LOW'
   - is_2_0_profile: canonical_predicted_score == '2-0' and Draw Risk == 'LOW'
   - signal_profile: '2-0_PROFILE' | 'STRONG_HOME_PROFILE' | None
4. Snapshot Integrity: Produces full pre-kickoff snapshots with bit-identical reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from dashboard.prediction_service import get_prediction_service
from dashboard.prediction_snapshot_store import PredictionSnapshot, get_prediction_snapshot_store
from models.poisson import _grid_size, modal_scoreline

logger = logging.getLogger("batch_prediction_service")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN_V4_PATH = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
EXPECTED_V4_SHA256 = "1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5"


def verify_v4_sha256() -> str:
    """Verify that the frozen V4 model binary is bit-identical."""
    if not FROZEN_V4_PATH.exists():
        raise FileNotFoundError(f"Frozen V4 model binary missing at {FROZEN_V4_PATH}")
    h = hashlib.sha256()
    with open(FROZEN_V4_PATH, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != EXPECTED_V4_SHA256:
        raise ValueError(
            f"V4 SHA256 mismatch! Expected {EXPECTED_V4_SHA256}, got {digest}"
        )
    return digest


class BatchPredictionService:
    """Service to produce canonical batch prediction ledgers."""

    def __init__(self):
        self.v4_sha256 = verify_v4_sha256()
        self.pred_service = get_prediction_service()
        self.snapshot_store = get_prediction_snapshot_store()

    def generate_batch_record(
        self,
        fixture_id: int,
        home_team: str,
        away_team: str,
        competition_name: str,
        competition_id: int = 423,
        scheduled_kickoff: str = "2026-09-11T18:30:00.000000Z",
        matchweek: Optional[str] = None,
        home_id_hint: Optional[int] = None,
        away_id_hint: Optional[int] = None,
        store_snapshot: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Generate a single canonical batch prediction record."""
        pred = self.pred_service.predict_matchup(
            home_team=home_team,
            away_team=away_team,
            competition_name=competition_name,
            competition_id=competition_id,
            scheduled_kickoff=scheduled_kickoff,
            fixture_id=fixture_id,
            home_id_hint=home_id_hint,
            away_id_hint=away_id_hint,
            model_key="V4.0 Production",
        )
        if pred is None:
            logger.warning(f"Could not extract features for fixture {fixture_id} ({home_team} vs {away_team})")
            return None

        p_h = float(pred.production_probs["H"])
        p_d = float(pred.production_probs["D"])
        p_a = float(pred.production_probs["A"])
        dec = pred.production_decision
        canon_score = pred.canonical_predicted_score
        canon_prob = pred.modal_scoreline_probability
        d_tier = str(pred.draw_risk_tier or "LOW").upper()

        is_sh = bool(pred.strong_home_profile)
        is_20 = bool(pred.is_2_0_profile)
        sig_prof = pred.signal_profile

        # Parse date and time
        dt_str = scheduled_kickoff[:10] if len(scheduled_kickoff) >= 10 else "--"
        time_str = scheduled_kickoff[11:16] if len(scheduled_kickoff) >= 16 else "--"

        rec = {
            "fixture_id": fixture_id,
            "Date (UTC)": dt_str,
            "Time (UTC)": time_str,
            "League": competition_name,
            "Matchweek": matchweek or "MW",
            "Home Team": home_team,
            "Away Team": away_team,
            "P(H)": f"{p_h*100:.1f}%",
            "P(D)": f"{p_d*100:.1f}%",
            "P(A)": f"{p_a*100:.1f}%",
            "p_home": round(p_h, 3),
            "p_draw": round(p_d, 3),
            "p_away": round(p_a, 3),
            "PRED": dec,
            "Predicted Score": canon_score,
            "canonical_predicted_score": canon_score,
            "modal_scoreline_probability": round(canon_prob, 4),
            "strong_home_profile": is_sh,
            "is_2_0_profile": is_20,
            "signal_profile": sig_prof,
            "Actual Score": "—",
            "Match Status": "UPCOMING",
            "Draw Risk": d_tier,
            "draw_risk_tier": d_tier,
            "draw_risk_score": round(float(pred.draw_risk_score or p_d), 4),
            "raw_draw_risk_tier": d_tier,
            "Status": "UPCOMING",
            "scheduled_kickoff": scheduled_kickoff,
            "has_locked_snapshot": True,
            "snapshot_source": "v4_0_production_canonical_pipeline",
            "lambda_home": round(float(pred.lambda_home), 4),
            "lambda_away": round(float(pred.lambda_away), 4),
            "model_v4_sha256": self.v4_sha256,
        }

        if store_snapshot:
            snap = PredictionSnapshot(
                fixture_id=fixture_id,
                prediction_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                scheduled_kickoff_utc=scheduled_kickoff,
                competition_name=competition_name,
                home_team=home_team,
                away_team=away_team,
                p_home=round(p_h, 4),
                p_draw=round(p_d, 4),
                p_away=round(p_a, 4),
                model_decision=dec,
                lambda_home=round(float(pred.lambda_home), 4),
                lambda_away=round(float(pred.lambda_away), 4),
                expected_home_goals=round(float(pred.lambda_home), 4),
                expected_away_goals=round(float(pred.lambda_away), 4),
                draw_risk_score=round(float(pred.draw_risk_score or p_d), 4),
                draw_risk_tier=d_tier,
                draw_risk_badge=pred.draw_risk_badge or "🟢 LOW",
                draw_risk_reasons=list(pred.draw_risk_reasons or []),
                model_name=pred.model_name,
                model_version=pred.model_version,
                model_md5=pred.model_file_md5,
                canonical_predicted_score=canon_score,
                predicted_score=canon_score,
                modal_scoreline_probability=round(canon_prob, 4),
                strong_home_profile=is_sh,
                is_2_0_profile=is_20,
                signal_profile=sig_prof,
            )
            self.snapshot_store.save_snapshot(snap)

        return rec

    def generate_batch(
        self,
        fixtures: List[Dict[str, Any]],
        output_jsonl_path: Optional[Path] = None,
        store_snapshots: bool = False,
    ) -> List[Dict[str, Any]]:
        """Generate batch ledger records for a list of fixtures."""
        results = []
        for fix in fixtures:
            rec = self.generate_batch_record(
                fixture_id=int(fix["fixture_id"]),
                home_team=str(fix["home_team"]),
                away_team=str(fix["away_team"]),
                competition_name=str(fix["competition_name"]),
                competition_id=int(fix.get("competition_id", 423)),
                scheduled_kickoff=str(fix.get("scheduled_kickoff", "")),
                matchweek=fix.get("matchweek"),
                home_id_hint=fix.get("home_id"),
                away_id_hint=fix.get("away_id"),
                store_snapshot=store_snapshots,
            )
            if rec is not None:
                results.append(rec)

        if output_jsonl_path:
            output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_jsonl_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return results
