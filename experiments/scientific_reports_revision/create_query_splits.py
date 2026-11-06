
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QRELS_PATH = PROJECT_ROOT / "data/qrels/qrels.csv"
CANDIDATES_PATH = PROJECT_ROOT / "results/hybrid_top20_candidates.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/scientific_reports_revision"

RANDOM_SEED = 42


def main():
    # Load the existing relevance judgements and retrieval candidates.
    qrels = pd.read_csv(QRELS_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)

    # Obtain the complete query population.
    query_ids = np.sort(qrels["query_id"].unique())

    # Randomise query IDs reproducibly.
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_ids = rng.permutation(query_ids)

    # Create 60% training, 20% validation and 20% test partitions.
    n_queries = len(shuffled_ids)
    n_train = int(n_queries * 0.60)
    n_val = int(n_queries * 0.20)

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]

    # Verify that no query occurs in more than one partition.
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == set(query_ids)

    # Identify queries eligible for positive-negative training pairs.
    relevant_pairs = set(
        zip(
            qrels.loc[qrels["relevance"] > 0, "query_id"],
            qrels.loc[qrels["relevance"] > 0, "document_id"],
        )
    )

    candidate_groups = candidates.groupby("query_id")

    eligible_ids = set()

    for query_id, group in candidate_groups:
        relevance_flags = [
            (query_id, doc_id) in relevant_pairs
            for doc_id in group["document_id"]
        ]

        if any(relevance_flags) and not all(relevance_flags):
            eligible_ids.add(query_id)

    # Save the partition assignments.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    partitions = {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
    }

    records = []

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Total queries: {n_queries}")

    for split_name, ids in partitions.items():
        eligible_count = len(set(ids) & eligible_ids)

        print(
            f"{split_name}: {len(ids)} queries, "
            f"{eligible_count} eligible for training pairs"
        )

        for query_id in ids:
            records.append({
                "query_id": int(query_id),
                "split": split_name,
                "pair_eligible": int(query_id in eligible_ids),
            })

    output_path = OUTPUT_DIR / "query_splits.csv"

    pd.DataFrame(records).sort_values(
        "query_id"
    ).to_csv(output_path, index=False)

    print("Query overlap: 0")
    print(f"Saved partitions to: {output_path}")


if __name__ == "__main__":
    main()
