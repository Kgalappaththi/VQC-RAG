
"""Evaluate held-out VQC ranking performance across three seeds."""

from pathlib import Path

import pandas as pd

from src.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    average_precision,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "scientific_reports_revision"

SPLITS_PATH = RESULTS_DIR / "query_splits.csv"
QRELS_PATH = PROJECT_ROOT / "data" / "qrels" / "qrels.csv"

METRICS = [
    "Recall@3", "Recall@5", "Recall@10", "Recall@20",
    "MRR", "nDCG@5", "nDCG@10", "MAP",
]


def calculate_metrics(retrieved_ids, relevant_ids):
    return {
        "Recall@3": recall_at_k(retrieved_ids, relevant_ids, 3),
        "Recall@5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "Recall@10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "Recall@20": recall_at_k(retrieved_ids, relevant_ids, 20),
        "MRR": reciprocal_rank(retrieved_ids, relevant_ids),
        "nDCG@5": ndcg_at_k(retrieved_ids, relevant_ids, 5),
        "nDCG@10": ndcg_at_k(retrieved_ids, relevant_ids, 10),
        "MAP": average_precision(retrieved_ids, relevant_ids),
    }


def main():
    splits = pd.read_csv(SPLITS_PATH)
    qrels = pd.read_csv(QRELS_PATH)

    test_ids = set(
        splits.loc[splits["split"] == "test", "query_id"]
    )
    assert len(test_ids) == 60

    original = pd.read_csv(
        RESULTS_DIR / "heldout_scored_candidates.csv"
    )

    assert set(original["query_id"]) == test_ids
    assert len(original) == 1200

    records = []

    for seed in [42, 7, 21]:
        print(f"\nEvaluating ranking metrics for seed {seed}")

        if seed == 42:
            candidates = original.copy()
        else:
            candidates = pd.read_csv(
                RESULTS_DIR / f"heldout_vqc_seed{seed}_scores.csv"
            )

            # Ensure candidate identity and row ordering are unchanged.
            assert len(candidates) == len(original)
            assert (
                candidates["query_id"].astype(str).tolist()
                == original["query_id"].astype(str).tolist()
            )
            assert (
                candidates["document_id"].astype(str).tolist()
                == original["document_id"].astype(str).tolist()
            )

        assert candidates["vqc_score"].notna().all()
        assert set(candidates["query_id"]) == test_ids

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

            ranked = query_candidates.sort_values(
                "vqc_score",
                ascending=False,
                kind="stable",
            )

            retrieved_ids = ranked["document_id"].astype(str).tolist()

            records.append({
                "seed": seed,
                "query_id": query_id,
                **calculate_metrics(retrieved_ids, relevant_ids),
            })

    query_results = pd.DataFrame(records)

    assert len(query_results) == 180
    assert not query_results.duplicated(
        ["seed", "query_id"]
    ).any()

    seed_summary = (
        query_results.groupby("seed")[METRICS]
        .mean()
        .reindex([42, 7, 21])
    )

    multiseed_summary = pd.DataFrame({
        "mean": seed_summary.mean(axis=0),
        "std": seed_summary.std(axis=0, ddof=1),
        "min": seed_summary.min(axis=0),
        "max": seed_summary.max(axis=0),
    })

    # All seeds rank the same fixed Top-20 candidate sets.
    recall20 = query_results.pivot(
        index="query_id",
        columns="seed",
        values="Recall@20",
    )
    assert recall20.nunique(axis=1).eq(1).all()

    query_results.to_csv(
        RESULTS_DIR / "heldout_multiseed_query_metrics.csv",
        index=False,
    )

    seed_summary.to_csv(
        RESULTS_DIR / "heldout_multiseed_seed_summary.csv"
    )

    multiseed_summary.to_csv(
        RESULTS_DIR / "heldout_multiseed_mean_std.csv"
    )

    print("\nPer-seed retrieval metrics:")
    print(seed_summary.round(4).to_string())

    print("\nAcross-seed mean and sample standard deviation:")
    print(multiseed_summary.round(4).to_string())

    print("\nRecall@20 identical across seeds: True")
    print("Multi-seed ranking evaluation completed.")


if __name__ == "__main__":
    main()
