"""
Reciprocal Rank Fusion (RRF) for Hybrid + VQC re-ranking.

RRF is widely used in information retrieval to combine ranked lists.
Higher RRF score indicates better ranking.
"""

import numpy as np
import pandas as pd


def score_to_rank(scores):
    """
    Convert scores into ranks.

    Highest score receives rank 1.
    """
    return pd.Series(scores).rank(ascending=False, method="first").to_numpy()


def reciprocal_rank_fusion(hybrid_scores, vqc_scores, k=60):
    """
    Apply Reciprocal Rank Fusion.

    score = 1 / (k + hybrid_rank) + 1 / (k + vqc_rank)
    """
    hybrid_rank = score_to_rank(hybrid_scores)
    vqc_rank = score_to_rank(vqc_scores)

    return 1.0 / (k + hybrid_rank) + 1.0 / (k + vqc_rank)