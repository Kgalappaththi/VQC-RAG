
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.quantum.features import (
    create_pair_feature,
    fit_feature_reducer,
    transform_features,
)
from src.quantum.triplets import build_training_triplets


# Locate the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Original experimental inputs.
CANDIDATES_PATH = PROJECT_ROOT / "results/hybrid_top20_candidates.csv"
QRELS_PATH = PROJECT_ROOT / "data/qrels/qrels.csv"

DOC_EMBEDDINGS_PATH = PROJECT_ROOT / "results/document_embeddings.npy"
DOC_METADATA_PATH = PROJECT_ROOT / "results/document_metadata.csv"

QUERY_EMBEDDINGS_PATH = PROJECT_ROOT / "results/query_embeddings.npy"
QUERY_METADATA_PATH = PROJECT_ROOT / "results/query_metadata.csv"

# Revised query partitions and output directory.
OUTPUT_DIR = PROJECT_ROOT / "results/scientific_reports_revision"
SPLITS_PATH = OUTPUT_DIR / "query_splits.csv"


def load_experimental_data():
    """Load experimental data and verify query partitions."""

    candidates = pd.read_csv(CANDIDATES_PATH)
    qrels = pd.read_csv(QRELS_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    doc_embeddings = np.load(DOC_EMBEDDINGS_PATH)
    query_embeddings = np.load(QUERY_EMBEDDINGS_PATH)

    doc_meta = pd.read_csv(DOC_METADATA_PATH)
    query_meta = pd.read_csv(QUERY_METADATA_PATH)

    # Confirm that metadata rows correspond to embedding rows.
    assert len(doc_meta) == len(doc_embeddings)
    assert len(query_meta) == len(query_embeddings)

    # Confirm that query identifiers are unique across partitions.
    assert not splits["query_id"].duplicated().any()

    expected_splits = {"train", "validation", "test"}
    assert set(splits["split"]) == expected_splits

    # Confirm that every evaluated query belongs to one partition.
    assert set(candidates["query_id"]) == set(splits["query_id"])
    assert set(qrels["query_id"]) == set(splits["query_id"])

    print("Experimental data loaded successfully.")
    print("Document embeddings:", doc_embeddings.shape)
    print("Query embeddings:", query_embeddings.shape)
    print("Query partitions verified:", len(splits))

    return (
        candidates,
        qrels,
        splits,
        doc_embeddings,
        query_embeddings,
        doc_meta,
        query_meta,
    )


def create_embedding_index_maps(doc_meta, query_meta):
    """Map document and query IDs to embedding row positions."""

    # Check that metadata identifiers are unique.
    assert not doc_meta["document_id"].duplicated().any()
    assert not query_meta["query_id"].duplicated().any()

    doc_id_to_index = {
        str(doc_id): idx
        for idx, doc_id in enumerate(doc_meta["document_id"])
    }

    query_id_to_index = {
        str(query_id): idx
        for idx, query_id in enumerate(query_meta["query_id"])
    }

    print("Document index mappings:", len(doc_id_to_index))
    print("Query index mappings:", len(query_id_to_index))

    return doc_id_to_index, query_id_to_index


def create_partition_triplets(candidates, qrels, splits):
    """Build positive-negative triplets separately for each query split."""

    partition_triplets = {}

    for split_name in ["train", "validation", "test"]:

        # Select only the queries assigned to this partition.
        split_ids = set(
            splits.loc[
                splits["split"] == split_name,
                "query_id"
            ]
        )

        split_candidates = candidates[
            candidates["query_id"].isin(split_ids)
        ].copy()

        split_qrels = qrels[
            qrels["query_id"].isin(split_ids)
        ].copy()

        # Preserve the original triplet-generation procedure.
        triplets = build_training_triplets(
            split_candidates,
            split_qrels
        )

        # Ensure no triplet belongs to another partition.
        assert set(triplets["query_id"]).issubset(split_ids)

        partition_triplets[split_name] = triplets

        print(
            f"{split_name}: {len(triplets)} triplets "
            f"from {len(split_ids)} queries"
        )

    return partition_triplets


def extract_triplet_features(
    triplets,
    doc_embeddings,
    query_embeddings,
    doc_id_to_index,
    query_id_to_index,
):
    """Extract positive and negative features for a query partition."""

    positive_features = []
    negative_features = []
    kept_rows = []

    for _, row in triplets.iterrows():
        query_id = str(row["query_id"])
        positive_id = str(row["positive_doc_id"])
        negative_id = str(row["negative_doc_id"])

        # Check that all required embeddings are available.
        if (
            query_id not in query_id_to_index
            or positive_id not in doc_id_to_index
            or negative_id not in doc_id_to_index
        ):
            raise ValueError(
                f"Missing embedding for query {query_id}"
            )

        query_embedding = query_embeddings[
            query_id_to_index[query_id]
        ]

        positive_embedding = doc_embeddings[
            doc_id_to_index[positive_id]
        ]

        negative_embedding = doc_embeddings[
            doc_id_to_index[negative_id]
        ]

        positive_features.append(
            create_pair_feature(
                query_embedding,
                positive_embedding
            )
        )

        negative_features.append(
            create_pair_feature(
                query_embedding,
                negative_embedding
            )
        )

        kept_rows.append({
            "query_id": query_id,
            "positive_doc_id": positive_id,
            "negative_doc_id": negative_id,
        })

    if not positive_features:
        raise ValueError("No valid triplets found in this partition.")

    return (
        np.asarray(positive_features),
        np.asarray(negative_features),
        kept_rows,
    )


def preprocess_partition_features(partition_features, n_components=4):
    """Fit preprocessing on training features only."""

    # Obtain the training features.
    train_pos, train_neg, _ = partition_features["train"]

    # Fit scaler and PCA using training features exclusively.
    train_all = np.vstack([train_pos, train_neg])

    _, scaler, pca = fit_feature_reducer(
        train_all,
        n_components=n_components,
    )

    processed_partitions = {}

    for split_name in ["train", "validation", "test"]:
        pos_features, neg_features, kept_rows = (
            partition_features[split_name]
        )

        # Apply the training-fitted transformations.
        pos_reduced = transform_features(
            pos_features, scaler, pca
        )

        neg_reduced = transform_features(
            neg_features, scaler, pca
        )

        processed_partitions[split_name] = (
            pos_reduced,
            neg_reduced,
            kept_rows,
        )

        print(
            f"{split_name}: "
            f"positive {pos_reduced.shape}, "
            f"negative {neg_reduced.shape}"
        )

    return processed_partitions, scaler, pca


def save_processed_datasets(
    processed_partitions,
    scaler,
    pca,
    n_components,
):
    """Save partitioned datasets and training-fitted preprocessing."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, data in processed_partitions.items():
        pos_features, neg_features, kept_rows = data

        output_path = (
            OUTPUT_DIR / f"{split_name}_vqc_dataset_{n_components}q.npz"
        )

        np.savez(
            output_path,
            pos_features=pos_features,
            neg_features=neg_features,
            query_ids=np.array([
                row["query_id"] for row in kept_rows
            ]),
            positive_doc_ids=np.array([
                row["positive_doc_id"] for row in kept_rows
            ]),
            negative_doc_ids=np.array([
                row["negative_doc_id"] for row in kept_rows
            ]),
        )

        print(f"Saved {split_name} dataset: {output_path.name}")

    # Save the scaler and PCA fitted on training data only.
    preprocessor_path = (
        OUTPUT_DIR / f"training_preprocessor_{n_components}q.pkl"
    )

    with open(preprocessor_path, "wb") as file:
        pickle.dump(
            {
                "scaler": scaler,
                "pca": pca,
                "n_components": n_components,
                "fitted_on": "training_queries_only",
            },
            file,
        )

    print(f"Saved training preprocessor: {preprocessor_path.name}")


def main():
    """Prepare query-disjoint datasets without preprocessing leakage."""

    print("Starting leakage-free VQC dataset preparation.")

    (
        candidates,
        qrels,
        splits,
        doc_embeddings,
        query_embeddings,
        doc_meta,
        query_meta,
    ) = load_experimental_data()

    # Build ID-to-embedding mappings.
    doc_id_to_index, query_id_to_index = (
        create_embedding_index_maps(doc_meta, query_meta)
    )

    # Generate triplets independently for each query partition.
    partition_triplets = create_partition_triplets(
        candidates, qrels, splits
    )

    # Construct the original 1536-dimensional feature vectors.
    partition_features = {}

    for split_name, triplets in partition_triplets.items():
        partition_features[split_name] = extract_triplet_features(
            triplets,
            doc_embeddings,
            query_embeddings,
            doc_id_to_index,
            query_id_to_index,
        )

    # Prepare datasets for the original 4-, 6- and 8-qubit configurations.
    for n_components in [4, 6, 8]:
        print(f"\nPreparing {n_components}-component features.")

        processed_partitions, scaler, pca = (
            preprocess_partition_features(
                partition_features,
                n_components=n_components,
            )
        )

        save_processed_datasets(
            processed_partitions,
            scaler,
            pca,
            n_components,
        )

    print("\nLeakage-free dataset preparation completed.")


if __name__ == "__main__":
    main()

