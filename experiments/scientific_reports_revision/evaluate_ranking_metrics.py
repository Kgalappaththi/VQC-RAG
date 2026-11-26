
from pathlib import Path

import pandas as pd

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


def main():
    candidates = pd.read_csv(SCORES_PATH)
    qrels = pd.read_csv(QRELS_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    test_ids = set(
        splits.loc[splits["split"] == "test", "query_id"]
    )

    assert len(test_ids) == 60
    assert set(candidates["query_id"]) == test_ids

    methods = {
        "Hybrid": "hybrid_score",
        "MLP": "mlp_score",
        "VQC": "vqc_score",
    }

    results = []

    for query_id in sorted(test_ids):
        query_candidates = candidates[
            candidates["query_id"] == query_id
        ]

        relevant_ids = set(
            qrels.loc[
                (qrels["query_id"] == query_id)
                & (qrels["relevance"] > 0),
                "document_id",
            ].astype(str)
        )

        assert len(query_candidates) == 20
        assert len(relevant_ids) > 0

        for method, score_column in methods.items():
            ranked = query_candidates.sort_values(
                score_column,
                ascending=False,
                kind="stable",
            )

            retrieved_ids = ranked["document_id"].astype(str).tolist()

            results.append({
                "query_id": query_id,
                "method": method,
                "Recall@3": recall_at_k(
                    retrieved_ids, relevant_ids, 3
                ),
                "Recall@5": recall_at_k(
                    retrieved_ids, relevant_ids, 5
                ),
                "Recall@10": recall_at_k(
                    retrieved_ids, relevant_ids, 10
                ),
                "Recall@20": recall_at_k(
                    retrieved_ids, relevant_ids, 20
                ),
                "MRR": reciprocal_rank(
                    retrieved_ids, relevant_ids
                ),
                "nDCG@5": ndcg_at_k(
                    retrieved_ids, relevant_ids, 5
                ),
                "nDCG@10": ndcg_at_k(
                    retrieved_ids, relevant_ids, 10
                ),
                "MAP": average_precision(
                    retrieved_ids, relevant_ids
                ),
            })

    query_results = pd.DataFrame(results)

    assert len(query_results) == 180

    metric_columns = [
        "Recall@3", "Recall@5", "Recall@10",
        "Recall@20", "MRR",
        "nDCG@5", "nDCG@10", "MAP",
    ]

    summary = (
        query_results.groupby("method")[metric_columns]
        .mean()
        .reindex(["Hybrid", "MLP", "VQC"])
    )

    # All methods rank the same Top-20 candidate set.
    recall20 = query_results.pivot(
        index="query_id",
        columns="method",
        values="Recall@20",
    )

    assert recall20.nunique(axis=1).eq(1).all()

    query_results.to_csv(
        REVISION_DIR / "heldout_query_metrics.csv",
        index=False,
    )

    summary.to_csv(
        REVISION_DIR / "heldout_metrics_summary.csv"
    )

    print("\nHeld-out ranking evaluation completed.")
    print("Queries evaluated:", len(test_ids))
    print("Query-method records:", len(query_results))
    print("\nMean retrieval metrics:")
    print(summary.round(4).to_string())
    print("\nRecall@20 identical across methods: True")


if __name__ == "__main__":
    main()

