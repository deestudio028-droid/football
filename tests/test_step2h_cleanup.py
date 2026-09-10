"""Unit tests for Step 2H-CLEANUP — Restoring V4.0 Production as the only active model."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys_paths = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from dashboard.model_registry import get_model_registry
from dashboard.prediction_service import PredictionService

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
APP_PATH = PROJECT_ROOT / "src/dashboard/app.py"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


@pytest.fixture(scope="module")
def registry():
    return get_model_registry()


@pytest.fixture(scope="module")
def prediction_service():
    return PredictionService()


def test_1_production_model_is_v4_0(registry):
    """TEST 1: Verify production model is V4.0 Production in registry."""
    default_model = registry.get_default_model()
    assert default_model == "V4.0 Production", f"Expected 'V4.0 Production', got '{default_model}'"

    prod_info = registry.get_production_model()
    assert prod_info.model_id == "v4_0_draw_champion"
    assert prod_info.role == "PRODUCTION"
    assert prod_info.md5_hash == EXP_V4_MD5


def test_2_dashboard_has_no_model_dropdown():
    """TEST 2: Verify production dashboard source code contains no model selection widget."""
    assert APP_PATH.exists()
    content = APP_PATH.read_text(encoding="utf-8")
    assert "key=\"model_selector_dropdown\"" not in content
    assert "options=available_models" not in content
    assert "ACTIVE_MODEL_KEY = \"V4.0 Production\"" in content


def test_3_production_predictions_use_v4_0_by_default(prediction_service):
    """TEST 3: Verify default prediction uses V4.0 Production."""
    res = prediction_service.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    assert res is not None
    assert res.prediction_model == "V4.0 Production"
    assert res.model_name == "v4_0_draw_champion"
    assert res.production_probs == res.benchmark_v4_0_probs
    assert res.production_decision == res.benchmark_v4_0_decision


def test_4_v4_0_artifact_md5_unchanged():
    """TEST 4: Verify V4.0 artifact MD5 hash is 100% unchanged."""
    assert V4_PATH.exists()
    act = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_MD5, f"V4.0 MD5 mutated: expected {EXP_V4_MD5}, got {act}"


def test_5_v4_1_artifact_md5_unchanged():
    """TEST 5: Verify V4.1 artifact MD5 hash is 100% unchanged."""
    assert V4_1_PATH.exists()
    act = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_1_MD5, f"V4.1 MD5 mutated: expected {EXP_V4_1_MD5}, got {act}"


def test_6_draw_enhanced_not_used_in_production(prediction_service):
    """TEST 6: Verify Draw-Enhanced is not accidentally used for normal production inference."""
    res = prediction_service.predict_manual_matchup("Fulham", "Chelsea", "Premier League")
    assert res is not None
    # If Draw-Enhanced was accidentally used as primary, production_probs would equal v4_draw_enhanced_probs
    assert res.production_probs != res.v4_draw_enhanced_probs
    assert res.production_probs == res.benchmark_v4_0_probs


def test_7_v4_0_output_remains_exact_baseline(prediction_service):
    """TEST 7: Verify known fixture matches exact V4.0 baseline output."""
    res = prediction_service.predict_manual_matchup("Nice", "Lorient", "Ligue 1")
    assert res is not None
    assert res.production_probs == {"H": 0.393, "D": 0.266, "A": 0.341}
    assert res.production_decision == "H"


def test_8_research_draw_enhanced_remains_available_isolated(prediction_service):
    """TEST 8: Verify research callers can still explicitly request V4.0 Draw-Enhanced without affecting default."""
    res_enh = prediction_service.predict_manual_matchup(
        "Nice", "Lorient", "Ligue 1", model_key="V4.0 Draw-Enhanced"
    )
    assert res_enh is not None
    assert res_enh.production_probs == {"H": 0.373, "D": 0.304, "A": 0.324}
    assert res_enh.production_decision == "H"
