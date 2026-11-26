
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.weighted_fusion import normalize_scores
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    average_precision,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION_DIR = PROJECT_ROOT / "results/scientific_reports_revision"

SCORES_PATH = REVISION_DIR / "heldout_scored_candidates.csv"
QRELS_PATH = PROJECT_ROOT / "data/qrels/qrels.csv"
SPLITS_PATH = REVISION_DIR / "query_splits.csv"

ALPHA = 0.5

METRICS = [
    "Recall@3", "Recall@5", "Recall@10", "Recall@20",
    "MRR", "nDCG@5", "nDCG@10", "MAP",
]


def evaluate_ranking(ranked_ids, relevant_ids):
    return {
        "Recall@3": recall_at_k(ranked_ids, relevant_ids, 3),
        "Recall@5": recall_at_k(ranked_ids, relevant_ids, 5),
        "Recall@10": recall_at_k(ranked_ids, relevant_ids, 10),
        "Recall@20": recall_at_k(ranked_ids, relevant_ids, 20),
        "MRR": reciprocal_rank(ranked_ids, relevant_ids),
        "nDCG@5": ndcg_at_k(ranked_ids, relevant_ids, 5),
        "nDCG@10": ndcg_at_k(ranked_ids, relevant_ids, 10),
        "MAP": average_precision(ranked_ids, relevant_ids),
    }


def main():
    candidates = pd.read_csv(SCORES_PATH)
    qrels = pd.read_csv(QRELS_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    test_ids = set(
        splits.loc[splits["split"] == "test", "query_id"]
    )

    assert len(test_ids) == 60
    assert set(candidates["query_id"]) == test_ids
    assert len(candidates) == 1200
    assert not candidates.duplicated(
        ["query_id", "document_id"]
    ).any()

    assert candidates[
        ["hybrid_score", "mlp_score", "vqc_score"]
    ].notna().all().all()

    query_results = []
    scored_results = []

    for query_id in sorted(test_ids):
        group = candidates.loc[
            candidates["query_id"] == query_id
        ].copy()

        assert len(group) == 20

        relevant_ids = set(
            qrels.loc[
                (qrels["query_id"] == query_id)
                & (qrels["relevance"] > 0),
                "document_id",
            ].astype(str)
        )

        assert len(relevant_ids) > 0

        hybrid_norm = normalize_scores(
            group["hybrid_score"].to_numpy()
        )
        mlp_norm = normalize_scores(
            group["mlp_score"].to_numpy()
        )
        vqc_norm = normalize_scores(
            group["vqc_score"].to_numpy()
        )

        group["hybrid_mlp_fusion_score"] = (
            ALPHA * hybrid_norm
            + (1.0 - ALPHA) * mlp_norm
        )

        group["hybrid_vqc_fusion_score"] = (
            ALPHA * hybrid_norm
            + (1.0 - ALPHA) * vqc_norm
        )

        scored_results.append(group)

        methods = {
            "Hybrid": "hybrid_score",
            "Hybrid+MLP": "hybrid_mlp_fusion_score",
            "Hybrid+VQC": "hybrid_vqc_fusion_score",
        }

        for method, score_column in methods.items():
            ranked = group.sort_values(
                score_column,
                ascending=False,
                kind="stable",
            )

            ranked_ids = (
                ranked["document_id"].astype(str).tolist()
            )

            query_results.append({
                "query_id": query_id,
                "method": method,
                **evaluate_ranking(ranked_ids, relevant_ids),
            })

    query_results = pd.DataFrame(query_results)
    scored_results = pd.concat(
        scored_results, ignore_index=True
    )

    assert len(query_results) == 180

    recall20 = query_results.pivot(
        index="query_id",
        columns="method",
        values="Recall@20",
    )

    assert recall20.nunique(axis=1).eq(1).all()

    summary = (
        query_results.groupby("method")[METRICS]
        .mean()
        .reindex(["Hybrid", "Hybrid+MLP", "Hybrid+VQC"])
    )

    scored_results.to_csv(
        REVISION_DIR / "heldout_fusion_scored_candidates.csv",
        index=False,
    )

    query_results.to_csv(
        REVISION_DIR / "heldout_fusion_query_metrics.csv",
        index=False,
    )

    summary.to_csv(
        REVISION_DIR / "heldout_fusion_metrics_summary.csv"
    )

    print("\nHeld-out fusion comparison completed.")
    print("Test queries:", len(test_ids))
    print("Candidates:", len(scored_results))
    print("Fusion alpha:", ALPHA)
    print("\nMean retrieval metrics:")
    print(summary.round(4).to_string())
    print("\nRecall@20 identical across methods: True")


if __name__ == "__main__":
    main()
