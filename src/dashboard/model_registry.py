"""Centralized Model Registry and Model Loader for Football Prediction Lab.

Manages model metadata, cryptographic hash enforcement (fail-closed),
and cached model artifact loading.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pickle

logger = logging.getLogger("dashboard_model_registry")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Pinned Expected Cryptographic Hashes (fail-closed if mismatch)
PINNED_MODEL_HASHES: Dict[str, Dict[str, Any]] = {
    "V4.0 Production": {
        "artifact_rel_path": "data/models/v4_poisson_venue_elo_online_ad.pkl",
        "expected_md5": "06841f0c03c8597b2b8cd8f8ab064864",
        "expected_sha256": "4cf1f964fc5db6ba67eec4379435b6a782803b0d23cb602fa97d519b7a4beae6",
        "model_id": "v4_0_draw_champion",
        "display_name": "V4.0 Production",
        "version": "v4.0-champion-dc-elo-stacking",
        "role": "PRODUCTION",
        "status": "FROZEN PRODUCTION TRUTH",
        "description": "Historical production baseline model combining Dixon-Coles, Elo draw curve, and proportional probability stacking.",
        "parameters": {
            "class_order": ["H", "D", "A"],
            "features": 91,
            "stacking_intercept": 0.2250,
            "global_fallback_rho": -0.0560,
            "training_seasons": 5,
            "training_rows": 8983,
            "information_cutoff": "2025-05-25T19:00:00.000000Z",
            "evaluation_status": "FROZEN PRODUCTION TRUTH",
        },
    },
    "V4.1 Production": {
        "artifact_rel_path": "data/models/v4_1_prospective_candidate_2025_26.pkl",
        "expected_md5": "145f918d933eb343c0f63ca342b10289",
        "expected_sha256": "cbb00b32c3ce4dbf600564a95d11b67a0c236f6b34cd184a9ca8abcc8ab94ac8",
        "model_id": "v4_1_prospective_candidate",
        "display_name": "V4.1 Prospective Candidate",
        "version": "v4.1-champion-dc-elo-stacking-2025-26-trained",
        "role": "RESEARCH",
        "status": "PROSPECTIVE 2026 RESEARCH CANDIDATE",
        "description": "Production candidate model trained through complete 2025/26 historical dataset with Dixon-Coles rho=-0.08 and online Attack/Defense states.",
        "parameters": {
            "class_order": ["H", "D", "A"],
            "features": 91,
            "dixon_coles_rho": -0.08,
            "training_seasons": 6,
            "training_rows": 10734,
            "information_cutoff": "2026-05-24T19:45:00.000000Z",
            "evaluation_status": "PROSPECTIVE 2026 LIVE EVALUATION ACTIVE",
        },
    },
    "V4.0 Draw-Enhanced": {
        "artifact_rel_path": "data/models/v4_poisson_venue_elo_online_ad.pkl",
        "expected_md5": "06841f0c03c8597b2b8cd8f8ab064864",
        "expected_sha256": "4cf1f964fc5db6ba67eec4379435b6a782803b0d23cb602fa97d519b7a4beae6",
        "model_id": "v4_0_draw_enhanced_candidate",
        "display_name": "V4.0 Draw-Enhanced (Candidate)",
        "version": "v4.0-draw-enhanced-platt-calibrated",
        "role": "RESEARCH",
        "status": "RESEARCH INTEGRATION CANDIDATE — PLATT CALIBRATED",
        "description": "V4.0 Production baseline with validated Platt continuous draw probability refinement and proportional H/A odds redistribution.",
        "parameters": {
            "class_order": ["H", "D", "A"],
            "features": 91,
            "calibrator_type": "platt_logistic",
            "slope_a": 0.942103,
            "intercept_b": 0.128363,
            "redistribution": "proportional_non_draw",
            "decision_rule": "argmax",
            "training_seasons": 5,
            "training_rows": 8983,
            "information_cutoff": "2025-05-25T19:00:00.000000Z",
            "evaluation_status": "SHADOW EVALUATION ACTIVE",
        },
    },
    "V3 Poisson Elo": {
        "artifact_rel_path": "data/models/v3_poisson_venue_elo_candidate.pkl",
        "expected_md5": "a2850a7687822a5916663301f5ccc96c",
        "expected_sha256": "ba96181be329cbfd43c2c10b4ba79a32cba9bb4b316ae306a445d4e1ce630eec",
        "model_id": "v3_poisson_venue_elo",
        "display_name": "V3 Poisson Elo Candidate",
        "version": "v3.0-poisson-venue-elo",
        "role": "RESEARCH",
        "status": "FROZEN HISTORICAL CANDIDATE",
        "description": "Poisson goal model incorporating team Elo ratings without online AD states.",
        "parameters": {
            "class_order": ["H", "D", "A"],
            "features": 87,
            "information_cutoff": "2024-05-25T19:00:00.000000Z",
            "evaluation_status": "HISTORICAL CANDIDATE",
        },
    },
    "V2 Poisson Venue": {
        "artifact_rel_path": "data/models/v2_poisson_venue.pkl",
        "expected_md5": "25935b4e93fc4074f67f16e3181ed4df",
        "expected_sha256": "b4f0b2f569b76c8c4959db6b567a21aa9098cbbf63cfcbf7f1e737ec30424560",
        "model_id": "v2_poisson_venue",
        "display_name": "V2 Poisson Venue",
        "version": "v2.0-poisson-venue",
        "role": "RESEARCH",
        "status": "FROZEN HISTORICAL BASELINE",
        "description": "Baseline Poisson goal model with venue features.",
        "parameters": {
            "class_order": ["H", "D", "A"],
            "features": 84,
            "information_cutoff": "2023-05-25T19:00:00.000000Z",
            "evaluation_status": "HISTORICAL BASELINE",
        },
    },
    "V1 Logistic Regression": {
        "artifact_rel_path": "data/models/v1_logreg.pkl",
        "expected_md5": "5e504427712b35778bb8a62a8496c7cd",
        "expected_sha256": "841961eeab96f947e937d57f6144e5be2646cf05d54c4aaae5b376d8a39d48b1",
        "model_id": "v1_logreg",
        "display_name": "V1 Logistic Regression",
        "version": "v1.0-logreg",
        "role": "RESEARCH",
        "status": "FROZEN HISTORICAL BASELINE",
        "description": "Original multinomial logistic regression baseline.",
        "parameters": {
            "class_order": ["H", "D", "A"],
            "features": 80,
            "information_cutoff": "2022-05-25T19:00:00.000000Z",
            "evaluation_status": "HISTORICAL BASELINE",
        },
    },
}

# In-memory artifact cache to avoid repeatedly unpickling
_MODEL_CACHE: Dict[str, Any] = {}


@dataclass(frozen=True)
class ModelInfo:
    """Metadata and status for a registered model."""
    model_id: str
    display_name: str
    version: str
    role: str  # "PRODUCTION", "BENCHMARK", "SHADOW", "RESEARCH"
    status: str  # "FROZEN PRODUCTION", "FROZEN BASELINE", "ACTIVE", "EXPERIMENTAL"
    file_path: str
    md5_hash: str
    sha256_hash: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_file_hashes(file_path: Path) -> Tuple[str, str]:
    """Compute MD5 and SHA-256 hashes for a file."""
    if not file_path.exists():
        return ("MISSING", "MISSING")
    data = file_path.read_bytes()
    md5_h = hashlib.md5(data).hexdigest()
    sha256_h = hashlib.sha256(data).hexdigest()
    return (md5_h, sha256_h)


def load_model_artifact(model_key: str, project_root: Path = PROJECT_ROOT) -> Any:
    """Load and cache a model artifact after verifying its cryptographic hash.

    Fails closed if the artifact file is missing or if the MD5 hash does not match
    the pinned expected hash.
    """
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    if model_key not in PINNED_MODEL_HASHES:
        # Check by model_id alias
        found_key = None
        for k, v in PINNED_MODEL_HASHES.items():
            if v["model_id"] == model_key:
                found_key = k
                break
        if found_key:
            model_key = found_key
        else:
            raise ValueError(f"Unknown model key '{model_key}'. Available: {list(PINNED_MODEL_HASHES.keys())}")

    cfg = PINNED_MODEL_HASHES[model_key]
    art_path = project_root / cfg["artifact_rel_path"]

    if not art_path.exists():
        raise RuntimeError(f"Model artifact file missing at {art_path}! FAILING CLOSED.")

    data = art_path.read_bytes()
    act_md5 = hashlib.md5(data).hexdigest()
    exp_md5 = cfg["expected_md5"]

    if act_md5 != exp_md5:
        raise RuntimeError(
            f"Cryptographic hash integrity violation for '{model_key}'!\n"
            f"Expected MD5: {exp_md5}\n"
            f"Actual MD5:   {act_md5}\n"
            f"FAILING CLOSED."
        )

    with open(art_path, "rb") as f:
        model_obj = pickle.load(f)

    _MODEL_CACHE[model_key] = model_obj
    logger.info(f"Loaded model '{model_key}' from {art_path.name} (MD5: {act_md5})")
    return model_obj


class ModelRegistry:
    """Read-only registry of production models and research candidates."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = Path(project_root)
        self.models: Dict[str, ModelInfo] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        for k, entry in PINNED_MODEL_HASHES.items():
            art_path = self.project_root / entry["artifact_rel_path"]
            md5_h, sha256_h = compute_file_hashes(art_path)
            if md5_h != entry["expected_md5"]:
                raise RuntimeError(
                    f"Integrity check failed for model '{k}'!\n"
                    f"Expected MD5: {entry['expected_md5']}, got {md5_h}. FAILING CLOSED."
                )

            info = ModelInfo(
                model_id=entry["model_id"],
                display_name=entry["display_name"],
                version=entry["version"],
                role=entry["role"],
                status=entry["status"],
                file_path=entry["artifact_rel_path"],
                md5_hash=md5_h,
                sha256_hash=sha256_h,
                description=entry["description"],
                parameters=entry["parameters"],
            )
            self.models[k] = info
            # Also index by model_id for backward compatibility
            self.models[entry["model_id"]] = info

        # 3. Research Candidates (V4.2, V4.6, Historical Candidate H)
        v46_md5, v46_sha256 = compute_file_hashes(self.project_root / "src/models/v4_6_physical_draw_gate.py")
        self.models["v4_6_physical_draw_gate"] = ModelInfo(
            model_id="v4_6_physical_draw_gate",
            display_name="V4.6 Physical Draw Gate (Shadow Candidate)",
            version="v4.6-physical-draw-gate",
            role="SHADOW",
            status="FROZEN",
            file_path="src/models/v4_6_physical_draw_gate.py",
            md5_hash=v46_md5,
            sha256_hash=v46_sha256,
            description="Selective 5-condition physical gate overriding fragile V4 wins to Draw without degrading baseline accuracy.",
            parameters={
                "draw_prob_threshold": 0.2600,
                "winner_margin_cap": 0.1000,
                "v4_winner_conf_cap": 0.4500,
                "abs_elo_cap": 100.0,
                "tot_expected_goals_cap": 2.5000,
            },
        )

        hist_md5, hist_sha256 = compute_file_hashes(self.project_root / "src/models/historical_draw_research_candidate.py")
        self.models["historical_candidate_h"] = ModelInfo(
            model_id="historical_candidate_h",
            display_name="Historical Candidate H (Pre-2025/26 Retrained)",
            version="v4.6-historical-only-retrained",
            role="RESEARCH",
            status="FROZEN",
            file_path="src/models/historical_draw_research_candidate.py",
            md5_hash=hist_md5,
            sha256_hash=hist_sha256,
            description="Model trained strictly on pre-2025/26 historical data (N=8,983) demonstrating generalization on 2025/26 blind replay.",
            parameters={
                "training_cutoff": "2024/2025",
                "training_matches_N": 8983,
                "draw_prob_threshold": 0.2600,
                "winner_margin_cap": 0.1000,
                "v4_winner_conf_cap": 0.4500,
                "abs_elo_cap": 100.0,
                "tot_expected_goals_cap": 2.5000,
            },
        )

    def get_available_models(self) -> List[str]:
        """Return list of selectable model keys for the dashboard."""
        return list(PINNED_MODEL_HASHES.keys())

    def get_default_model(self) -> str:
        """Return the default production model key."""
        return "V4.0 Production"

    def get_production_model(self) -> ModelInfo:
        """Get the active production model (V4.0 Production)."""
        return self.models["V4.0 Production"]

    def get_baseline_model(self) -> ModelInfo:
        """Get the frozen benchmark baseline model (V4.0 Production)."""
        return self.models["V4.0 Production"]

    def get_shadow_model(self) -> ModelInfo:
        return self.models["v4_6_physical_draw_gate"]

    def get_model(self, model_key_or_id: str) -> Optional[ModelInfo]:
        return self.models.get(model_key_or_id)

    def list_all_models(self) -> List[ModelInfo]:
        # Return unique ModelInfo instances
        seen = set()
        out = []
        for m in self.models.values():
            if m.model_id not in seen:
                seen.add(m.model_id)
                out.append(m)
        return out


def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
