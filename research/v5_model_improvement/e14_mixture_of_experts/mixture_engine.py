"""Phase 39 — Experiment E14: Mixture-of-Experts Probability Blending Engine.

Combines expert forecast probability distributions under gating weights:
    P_blend(i) = sum_{j=1}^K w_j(i) * P_j(i)

Guarantees:
- Strict probability simplex invariance (non-negativity, sum to 1.0)
- Entropy and sharpness diagnostic calculations
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np


def blend_expert_probabilities(
    expert_probs_list: List[np.ndarray],
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blends multiple expert probability distributions using gating weights.

    Args:
        expert_probs_list: List of K (N, 3) probability arrays.
        weights: (N, K) array of gating weights.

    Returns:
        blended_probs: (N, 3) blended probability array.
        entropies: (N,) array of prediction Shannon entropies.
        sharpness: (N,) array of maximum class confidence.
    """
    n_matches = len(expert_probs_list[0])
    n_experts = len(expert_probs_list)

    # Normalize weights to sum to 1 across experts
    w_norm = np.clip(weights, 0.0, 1.0)
    w_sum = w_norm.sum(axis=1, keepdims=True)
    w_norm = np.where(w_sum > 0, w_norm / w_sum, 1.0 / n_experts)

    blended_probs = np.zeros((n_matches, 3), dtype=np.float64)
    for j in range(n_experts):
        blended_probs += w_norm[:, j : j + 1] * expert_probs_list[j]

    # Ensure simplex normalization
    p_sum = blended_probs.sum(axis=1, keepdims=True)
    blended_probs = np.where(p_sum > 0, blended_probs / p_sum, 1.0 / 3.0)

    entropies = -np.sum(blended_probs * np.log(np.clip(blended_probs, 1e-12, 1.0)), axis=1)
    sharpness = np.max(blended_probs, axis=1)

    return blended_probs, entropies, sharpness
