"""Generate Step 2L UX and Prospective validation artifacts."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STEP2L_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2l_risk_ux"
STEP2L_DIR.mkdir(parents=True, exist_ok=True)

STEP2K_VAL_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2k_advisory_integration/step2k_advisory_validation.csv"
STEP2K_33_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2k_advisory_integration/step2k_33_match_audit.csv"

# Load Step 2K validated data
df_val = pd.read_csv(STEP2K_VAL_PATH)
df_33 = pd.read_csv(STEP2K_33_PATH)

# Add UX Labels & Badges
label_map = {
    "LOW": "Low draw vulnerability",
    "MEDIUM": "Moderate draw vulnerability",
    "HIGH": "High draw vulnerability",
    "CRITICAL": "Critical draw vulnerability",
}
badge_map = {
    "LOW": "🟢 LOW",
    "MEDIUM": "🟡 MEDIUM",
    "HIGH": "🟠 HIGH",
    "CRITICAL": "🔴 CRITICAL",
}

df_val["ux_label"] = df_val["tier"].map(label_map)
df_val["ux_badge"] = df_val["tier"].map(badge_map)
df_val.to_csv(STEP2L_DIR / "step2l_ux_validation.csv", index=False)

df_33["ux_label"] = df_33["draw_risk_tier"].map(label_map)
df_33["ux_badge"] = df_33["draw_risk_tier"].map(badge_map)
df_33.to_csv(STEP2L_DIR / "step2l_33_match_audit.csv", index=False)

print("Step 2L CSV artifacts successfully created.")
