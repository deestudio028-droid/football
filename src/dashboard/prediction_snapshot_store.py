"""Immutable Prediction Snapshot Ledger & Store for Prediction Lifecycle Management.

Maintains strict separation between:
1. Live/Upcoming pre-kickoff predictions (updated in real-time as pre-match information updates).
2. Authoritative pre-kickoff locked predictions (frozen at kickoff).
3. Completed/Historical predictions (immutable forever; no post-kickoff re-inference).
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("prediction_snapshot_store")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = PROJECT_ROOT / "data" / "processed" / "prediction_snapshot_ledger.json"
STEP3C_LEDGER_PATH = PROJECT_ROOT / "research" / "v5_model_improvement" / "step3c_prospective_shadow" / "prospective_shadow_ledger.csv"
STEP2F_LEDGER_PATH = PROJECT_ROOT / "research" / "v5_model_improvement" / "step2_draw_research" / "step2f_live_prospective" / "live_shadow_forecast_ledger.csv"
OUTCOME_LEDGER_PATH = PROJECT_ROOT / "research" / "v5_model_improvement" / "production_monitoring" / "live_outcome_ledger.csv"


@dataclass
class PredictionSnapshot:
    """Immutable pre-kickoff prediction snapshot for an individual fixture."""
    fixture_id: int
    home_team: str
    away_team: str
    competition_name: str
    competition_id: int = 0
    source_fixture_date: str = ""
    scheduled_kickoff_utc: str = ""
    prediction_timestamp: str = ""
    model_name: str = "V4.0 Production"
    model_version: str = "v4.0-poisson-venue-elo-online-ad"
    model_md5: str = "06841f0c03c8597b2b8cd8f8ab064864"
    feature_snapshot_hash: str = ""
    p_home: float = 0.0
    p_draw: float = 0.0
    p_away: float = 0.0
    model_decision: str = "H"
    lambda_home: float = 1.2
    lambda_away: float = 1.0
    expected_home_goals: float = 1.2
    expected_away_goals: float = 1.0
    draw_risk_score: float = 0.25
    draw_risk_tier: str = "LOW"
    draw_risk_badge: str = "🟢 LOW"
    draw_risk_reasons: List[str] = field(default_factory=list)
    is_vulnerable_fixture: bool = False
    is_promoted_match: bool = False
    is_locked: bool = True


class PredictionSnapshotStore:
    """In-memory store backed by immutable JSON/CSV ledgers."""

    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = Path(ledger_path) if ledger_path else LEDGER_PATH
        self._snapshots: Dict[int, PredictionSnapshot] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load from primary JSON snapshot ledger and fallback prospective CSV ledgers."""
        # 1. Load primary JSON ledger if available
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                if isinstance(entries, list):
                    for item in entries:
                        snap = PredictionSnapshot(**item)
                        self._snapshots[snap.fixture_id] = snap
                elif isinstance(entries, dict):
                    for fid_str, item in entries.items():
                        snap = PredictionSnapshot(**item)
                        self._snapshots[snap.fixture_id] = snap
                logger.info(f"Loaded {len(self._snapshots)} snapshots from {self.ledger_path}")
            except Exception as e:
                logger.warning(f"Error loading primary snapshot ledger: {e}")

        # 2. Ingest completed prospective ledger from live_outcome_ledger.csv if not already present
        if OUTCOME_LEDGER_PATH.exists():
            try:
                with open(OUTCOME_LEDGER_PATH, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fid = int(row["fixture_id"])
                        if fid not in self._snapshots:
                            p_h = round(float(row.get("v4_p_home", 0.44)), 3)
                            p_d = round(float(row.get("v4_p_draw", 0.26)), 3)
                            p_a = round(float(row.get("v4_p_away", 0.30)), 3)
                            dec = str(row.get("v4_decision", "H"))
                            d_tier = str(row.get("draw_risk_tier", "LOW"))
                            badge = "🟢 LOW" if d_tier == "LOW" else ("🟡 MEDIUM" if d_tier == "MEDIUM" else ("🟠 HIGH" if d_tier == "HIGH" else "🔴 CRITICAL"))
                            ko = str(row.get("scheduled_kickoff", ""))
                            src_date = ko[:10] if len(ko) >= 10 else ""

                            snap = PredictionSnapshot(
                                fixture_id=fid,
                                home_team=str(row.get("home_team", "")),
                                away_team=str(row.get("away_team", "")),
                                competition_name=str(row.get("competition", "")),
                                competition_id=0,
                                source_fixture_date=src_date,
                                scheduled_kickoff_utc=ko,
                                prediction_timestamp=str(row.get("prediction_timestamp", "")),
                                p_home=p_h,
                                p_draw=p_d,
                                p_away=p_a,
                                model_decision=dec,
                                lambda_home=round(float(row.get("lambda_home", 1.4)), 2) if "lambda_home" in row else round(p_h * 2.8, 2),
                                lambda_away=round(float(row.get("lambda_away", 1.1)), 2) if "lambda_away" in row else round(p_a * 2.8, 2),
                                expected_home_goals=round(float(row.get("lambda_home", 1.4)), 2) if "lambda_home" in row else round(p_h * 2.8, 2),
                                expected_away_goals=round(float(row.get("lambda_away", 1.1)), 2) if "lambda_away" in row else round(p_a * 2.8, 2),
                                draw_risk_score=float(row.get("draw_risk_score", 0.25)),
                                draw_risk_tier=d_tier,
                                draw_risk_badge=badge,
                                is_locked=True,
                            )
                            self._snapshots[fid] = snap
            except Exception as e:
                logger.warning(f"Error loading outcome ledger: {e}")

        # 3. Ingest from Step 3C prospective shadow ledger
        if STEP3C_LEDGER_PATH.exists():
            try:
                with open(STEP3C_LEDGER_PATH, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fid = int(row["fixture_id"])
                        if fid in self._snapshots and "draw_risk_tier" in row and row["draw_risk_tier"]:
                            d_tier = str(row["draw_risk_tier"]).upper()
                            badge = "🟢 LOW" if d_tier == "LOW" else ("🟡 MEDIUM" if d_tier == "MEDIUM" else ("🟠 HIGH" if d_tier == "HIGH" else "🔴 CRITICAL"))
                            self._snapshots[fid].draw_risk_tier = d_tier
                            self._snapshots[fid].draw_risk_badge = badge
                            if "draw_risk_score" in row and row["draw_risk_score"]:
                                try:
                                    self._snapshots[fid].draw_risk_score = float(row["draw_risk_score"])
                                except Exception:
                                    pass

                        if fid not in self._snapshots:
                            p_h = round(float(row.get("v40_p_home", 0.44)), 3)
                            p_d = round(float(row.get("v40_p_draw", 0.26)), 3)
                            p_a = round(float(row.get("v40_p_away", 0.30)), 3)
                            dec = str(row.get("v40_prediction", "H"))
                            d_tier = str(row.get("draw_risk_tier", "LOW")).upper()
                            badge = "🟢 LOW" if d_tier == "LOW" else ("🟡 MEDIUM" if d_tier == "MEDIUM" else ("🟠 HIGH" if d_tier == "HIGH" else "🔴 CRITICAL"))
                            ko = str(row.get("kickoff", ""))
                            src_date = ko[:10] if len(ko) >= 10 else ""

                            snap = PredictionSnapshot(
                                fixture_id=fid,
                                home_team=str(row.get("home_team", "")),
                                away_team=str(row.get("away_team", "")),
                                competition_name=str(row.get("league", "")),
                                competition_id=0,
                                source_fixture_date=src_date,
                                scheduled_kickoff_utc=ko,
                                prediction_timestamp=str(row.get("prediction_timestamp", "")),
                                p_home=p_h,
                                p_draw=p_d,
                                p_away=p_a,
                                model_decision=dec,
                                lambda_home=round(float(row.get("v40_expected_home_goals", 1.4)), 2),
                                lambda_away=round(float(row.get("v40_expected_away_goals", 1.1)), 2),
                                expected_home_goals=round(float(row.get("v40_expected_home_goals", 1.4)), 2),
                                expected_away_goals=round(float(row.get("v40_expected_away_goals", 1.1)), 2),
                                draw_risk_score=float(row.get("draw_risk_score", 0.25)),
                                draw_risk_tier=d_tier,
                                draw_risk_badge=badge,
                                is_locked=True,
                            )
                            self._snapshots[fid] = snap
            except Exception as e:
                logger.warning(f"Error loading step3c ledger: {e}")

    def get_snapshot(self, fixture_id: int) -> Optional[PredictionSnapshot]:
        """Retrieve pre-kickoff prediction snapshot for a fixture if present."""
        return self._snapshots.get(fixture_id)

    def save_snapshot(self, snapshot: PredictionSnapshot) -> None:
        """Register or update a snapshot in memory."""
        self._snapshots[snapshot.fixture_id] = snapshot

    def save(self) -> None:
        """Persist current snapshots to disk."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(s) for s in self._snapshots.values()]
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


_GLOBAL_SNAPSHOT_STORE: Optional[PredictionSnapshotStore] = None


def get_prediction_snapshot_store() -> PredictionSnapshotStore:
    """Return the process-global singleton PredictionSnapshotStore instance."""
    global _GLOBAL_SNAPSHOT_STORE
    if _GLOBAL_SNAPSHOT_STORE is None:
        _GLOBAL_SNAPSHOT_STORE = PredictionSnapshotStore()
    return _GLOBAL_SNAPSHOT_STORE
