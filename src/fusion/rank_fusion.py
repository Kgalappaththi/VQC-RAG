"""
Rank-based fusion for Hybrid + VQC re-ranking.

This module combines Hybrid rank and VQC rank rather than raw scores.
Lower rank is better, so final ranking is based on lower fused rank score.
"""

import numpy as np
import pandas as pd


def score_to_rank(scores):
    """
    Convert scores into ranks.

    Highest score receives rank 1.
    """
    return pd.Series(scores).rank(ascending=False, method="first").to_numpy()


def rank_fusion(hybrid_scores, vqc_scores, alpha=0.8):
    """
    Fuse Hybrid and VQC rankings.

    fused_rank = alpha * hybrid_rank + (1 - alpha) * vqc_rank

    Lower fused score indicates better ranking.
    """
    hybrid_rank = score_to_rank(hybrid_scores)
    vqc_rank = score_to_rank(vqc_scores)

    fused_rank = alpha * hybrid_rank + (1.0 - alpha) * vqc_rank

    return -fused_rank