"""
Weighted score fusion for Hybrid + VQC re-ranking.

This module combines normalized Hybrid retrieval scores and normalized
VQC relevance scores using a fixed fusion weight.
"""

import numpy as np


def normalize_scores(values):
    """
    Normalize score values into the range [0, 1].
    """
    values = np.array(values, dtype=float)

    min_value = values.min()
    max_value = values.max()

    if max_value - min_value == 0:
        return np.ones_like(values) * 0.5

    return (values - min_value) / (max_value - min_value)


def weighted_score_fusion(hybrid_scores, vqc_scores, alpha=0.8):
    """
    Compute weighted fusion score.

    final_score = alpha * hybrid_score + (1 - alpha) * vqc_score
    """
    hybrid_norm = normalize_scores(hybrid_scores)
    vqc_norm = normalize_scores(vqc_scores)

    return alpha * hybrid_norm + (1.0 - alpha) * vqc_norm