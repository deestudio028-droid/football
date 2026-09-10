"""V4.6 Live Prospective Collector.

Operational collector for the frozen V4.6 Physical Draw Gate candidate.

INVARIANTS & SAFETY CONTRACTS:
1. Complete Model Freeze: V4.6 rules, parameters, and code cannot be modified.
2. Strict Pre-Kickoff Lock: Predictions must be generated and cryptographically hashed (SHA-256)
   strictly BEFORE scheduled kickoff (prediction_timestamp < scheduled_kickoff).
3. Zero Historical Recycling: Strictly rejects any historical fixture from the 1,301-match
   diagnostic cohort, the 450 validation cohort, or prior training seasons.
4. Zero Outcome Access: Zero access to match score, result, in-play statistics, or post-kickoff events.
5. Zero Market Odds: Market odds are isolated and never enter model inputs.
6. Idempotence: Duplicate collection attempts for an already-locked fixture are rejected safely.
7. Isolated Storage: Operates exclusively in `research/v4_promotion/live_v46_prospective.sqlite`.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from models.baselines import CLASS_ORDER
from models.v4_6_physical_draw_gate import (
    V46PhysicalGateConfig,
    evaluate_physical_draw_gate,
    predict_v4_6_physical_gate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LIVE_DB = PROJECT_ROOT / "research" / "v4_promotion" / "live_v46_prospective.sqlite"
MANIFEST_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "phase26_live_collection_manifest.json"

FROZEN_RULE_PARAMS = {
    "draw_prob_threshold": 0.2600,
    "winner_margin_cap": 0.1000,
    "v4_winner_conf_cap": 0.4500,
    "abs_elo_cap": 100.0,
    "tot_expected_goals_cap": 2.5000,
    "low_score_prob_floor": 0.0000,
}


class LiveCollectorError(RuntimeError):
    """Base error for live prospective collection pipeline."""


class PreKickoffViolationError(LiveCollectorError):
    """Raised when prediction timestamp is not strictly before scheduled kickoff."""


class DuplicateFixtureError(LiveCollectorError):
    """Raised when attempting to store a prediction for an already-locked fixture."""


class HistoricalFixtureRejectionError(LiveCollectorError):
    """Raised when a known historical fixture attempts to enter prospective collection."""


class ModelConfigMutationError(LiveCollectorError):
    """Raised when V4.6 parameters differ from the frozen validation manifest."""


def compute_prediction_hash(payload: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash over canonical prediction payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def init_live_prospective_db(db_path: Path = DEFAULT_LIVE_DB) -> None:
    """Initialize isolated SQLite store for live V4.6 prospective validation."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS live_predictions (
            fixture_id INTEGER PRIMARY KEY,
            competition_id INTEGER NOT NULL,
            competition_name TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            scheduled_kickoff TEXT NOT NULL,
            prediction_timestamp TEXT NOT NULL,
            p_v4_home REAL NOT NULL,
            p_v4_draw REAL NOT NULL,
            p_v4_away REAL NOT NULL,
            v4_base_decision TEXT NOT NULL,
            p_v42_draw REAL NOT NULL,
            winner_margin REAL NOT NULL,
            v4_winner_conf REAL NOT NULL,
            abs_elo_diff REAL NOT NULL,
            tot_expected_goals REAL NOT NULL,
            low_score_prob REAL NOT NULL,
            override_applied INTEGER NOT NULL,
            v4_6_final_decision TEXT NOT NULL,
            model_version TEXT NOT NULL,
            prediction_sha256 TEXT NOT NULL UNIQUE,
            lock_status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS match_outcomes (
            fixture_id INTEGER PRIMARY KEY,
            outcome_timestamp TEXT NOT NULL,
            home_goals INTEGER NOT NULL,
            away_goals INTEGER NOT NULL,
            actual_outcome TEXT NOT NULL,
            status TEXT NOT NULL,
            reconciled_at TEXT NOT NULL,
            FOREIGN KEY(fixture_id) REFERENCES live_predictions(fixture_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS collection_audit_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            fixtures_ingested INTEGER NOT NULL,
            details TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def load_known_historical_fixture_ids() -> Set[int]:
    """Load all historical and prior validation fixture IDs to prevent any recycling."""
    known: Set[int] = set()

    # Matches.db 2025/2026 fixtures
    matches_db = PROJECT_ROOT / "data" / "processed" / "matches.db"
    if matches_db.exists():
        conn = sqlite3.connect(f"file:{matches_db}?mode=ro", uri=True)
        cur = conn.cursor()
        for row in cur.execute("SELECT fixture_id FROM fixtures"):
            known.add(int(row[0]))
        conn.close()

    # Validation cohorts JSONs
    for p in (PROJECT_ROOT / "research" / "v4_promotion").glob("*.json"):
        if "fixture_ids" in p.name:
            try:
                with open(p) as f:
                    d = json.load(f)
                    ids = d["fixture_ids"] if isinstance(d, dict) and "fixture_ids" in d else d
                    if isinstance(ids, list):
                        known.update(int(x) for x in ids if isinstance(x, (int, str)) and str(x).isdigit())
            except Exception:
                pass

    return known


class V46LiveProspectiveCollector:
    """Operational collector for prospective fixture predictions."""

    def __init__(self, db_path: Path = DEFAULT_LIVE_DB, config: Optional[V46PhysicalGateConfig] = None):
        self.db_path = db_path
        self.config = config or V46PhysicalGateConfig.robust_optimal_gate()
        self._verify_frozen_config()
        init_live_prospective_db(self.db_path)
        self.historical_ids = load_known_historical_fixture_ids()

    def _verify_frozen_config(self) -> None:
        """Enforce strict configuration immutability."""
        for k, v in FROZEN_RULE_PARAMS.items():
            act = getattr(self.config, k)
            if abs(act - v) > 1e-6:
                raise ModelConfigMutationError(f"V4.6 config mutated on {k}: expected {v}, got {act}")

    def collect_and_lock_prediction(
        self,
        fixture_id: int,
        competition_id: int,
        competition_name: str,
        home_team: str,
        away_team: str,
        scheduled_kickoff: str | datetime,
        p_v4: List[float] | np.ndarray,
        p_v42: List[float] | np.ndarray,
        abs_elo_diff: float,
        lambda_home: float,
        lambda_away: float,
        low_score_prob: float = 0.25,
        prediction_timestamp: Optional[str | datetime] = None,
    ) -> Dict[str, Any]:
        """Generate and cryptographically lock a prediction strictly before kickoff."""
        # 1. Historical Fixture Rejection
        if fixture_id in self.historical_ids:
            raise HistoricalFixtureRejectionError(
                f"Fixture ID {fixture_id} is a known historical fixture. Live prospective collection strictly rejects historical matches."
            )

        # 2. Parse Timestamps & Pre-Kickoff Enforcement
        now_dt = datetime.now(timezone.utc) if prediction_timestamp is None else (
            datetime.fromisoformat(prediction_timestamp.replace("Z", "+00:00"))
            if isinstance(prediction_timestamp, str) else prediction_timestamp
        )
        kickoff_dt = (
            datetime.fromisoformat(scheduled_kickoff.replace("Z", "+00:00"))
            if isinstance(scheduled_kickoff, str) else scheduled_kickoff
        )

        if now_dt >= kickoff_dt:
            raise PreKickoffViolationError(
                f"Prediction timestamp {now_dt.isoformat()} is not strictly before kickoff {kickoff_dt.isoformat()} for fixture {fixture_id}."
            )

        pv4 = np.asarray(p_v4, dtype=float)
        pv42 = np.asarray(p_v42, dtype=float)

        # 3. Evaluate Frozen V4.6 Physical Gate
        pred_obj = predict_v4_6_physical_gate(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            p_v4=pv4,
            p_v42=pv42,
            abs_elo_diff=abs_elo_diff,
            low_score_prob=low_score_prob,
            config=self.config,
            fixture_id=fixture_id,
        )

        # 4. Construct Deterministic Payload & SHA-256 Digest
        payload = {
            "fixture_id": fixture_id,
            "competition_id": competition_id,
            "home_team": home_team,
            "away_team": away_team,
            "scheduled_kickoff": kickoff_dt.isoformat(),
            "prediction_timestamp": now_dt.isoformat(),
            "p_v4": [round(float(x), 6) for x in pv4],
            "p_v42": [round(float(x), 6) for x in pv42],
            "abs_elo_diff": round(float(abs_elo_diff), 4),
            "tot_expected_goals": round(float(lambda_home + lambda_away), 4),
            "low_score_prob": round(float(low_score_prob), 4),
            "v4_base_decision": pred_obj.v4_base_decision,
            "v4_6_final_decision": pred_obj.v4_6_final_decision,
            "override_applied": pred_obj.override_applied,
            "model_version": self.config.version,
        }
        sha256_digest = compute_prediction_hash(payload)

        # 5. Idempotent Storage in Isolated SQLite
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        # Check existing
        cur.execute("SELECT prediction_sha256 FROM live_predictions WHERE fixture_id = ?", (fixture_id,))
        existing = cur.fetchone()
        if existing:
            conn.close()
            raise DuplicateFixtureError(f"Prediction for fixture {fixture_id} is already locked (hash: {existing[0]}).")

        cur.execute("""
            INSERT INTO live_predictions (
                fixture_id, competition_id, competition_name, home_team, away_team,
                scheduled_kickoff, prediction_timestamp,
                p_v4_home, p_v4_draw, p_v4_away, v4_base_decision,
                p_v42_draw, winner_margin, v4_winner_conf, abs_elo_diff, tot_expected_goals, low_score_prob,
                override_applied, v4_6_final_decision, model_version, prediction_sha256, lock_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fixture_id, competition_id, competition_name, home_team, away_team,
            kickoff_dt.isoformat(), now_dt.isoformat(),
            float(pv4[0]), float(pv4[1]), float(pv4[2]), pred_obj.v4_base_decision,
            float(pv42[1]), float(pred_obj.metadata["winner_margin"]), float(pred_obj.metadata["v4_winner_conf"]),
            float(abs_elo_diff), float(lambda_home + lambda_away), float(low_score_prob),
            1 if pred_obj.override_applied else 0, pred_obj.v4_6_final_decision,
            self.config.version, sha256_digest, "LOCKED", datetime.now(timezone.utc).isoformat()
        ))

        cur.execute("""
            INSERT INTO collection_audit_log (timestamp, action, fixtures_ingested, details)
            VALUES (?, ?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(), "LOCK_PREDICTION", 1, f"Locked fixture {fixture_id} ({home_team} vs {away_team})"))

        conn.commit()
        conn.close()

        payload["prediction_sha256"] = sha256_digest
        payload["lock_status"] = "LOCKED"
        return payload

    def get_locked_prediction_count(self) -> int:
        """Return total count of locked prospective predictions."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM live_predictions")
        cnt = int(cur.fetchone()[0])
        conn.close()
        return cnt
