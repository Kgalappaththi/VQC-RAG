import numpy as np

from src.fusion.weighted_fusion import weighted_score_fusion
from src.fusion.rank_fusion import rank_fusion
from src.fusion.rrf_fusion import reciprocal_rank_fusion


def test_fusion_methods():
    hybrid_scores = np.array([0.9, 0.8, 0.7])
    vqc_scores = np.array([0.2, 0.9, 0.5])

    weighted = weighted_score_fusion(hybrid_scores, vqc_scores, alpha=0.8)
    ranked = rank_fusion(hybrid_scores, vqc_scores, alpha=0.8)
    rrf = reciprocal_rank_fusion(hybrid_scores, vqc_scores)

    print("Weighted:", weighted)
    print("Rank fusion:", ranked)
    print("RRF:", rrf)

    assert len(weighted) == 3
    assert len(ranked) == 3
    assert len(rrf) == 3

    assert np.all(np.isfinite(weighted))
    assert np.all(np.isfinite(ranked))
    assert np.all(np.isfinite(rrf))

    print("Fusion tests passed.")


if __name__ == "__main__":
    test_fusion_methods()