"""Phase 38 — Experiment E13: Hierarchical Empirical Bayes Shrinkage Module.

Implements Empirical Bayes hierarchical shrinkage for sparse-history & newly promoted teams:
    theta_shrunk = (n_i / (n_i + m_0)) * theta_raw + (m_0 / (n_i + m_0)) * mu_prior
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def apply_hierarchical_shrinkage(
    raw_values: np.ndarray,
    sample_sizes: np.ndarray,
    prior_mean: float = 0.0,
    m0: float = 5.0,
) -> np.ndarray:
    """Applies Empirical Bayes shrinkage weight based on observed sample size."""
    weights = sample_sizes / (sample_sizes + m0)
    return weights * raw_values + (1.0 - weights) * prior_mean
