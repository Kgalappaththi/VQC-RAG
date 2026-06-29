import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.quantum.features import (
    create_pair_feature,
    fit_feature_reducer,
    transform_features
)
from src.quantum.triplets import build_training_triplets


def build_vqc_training_dataset(
    candidates_csv: str,
    qrels_csv: str,
    document_embeddings_npy: str,
    document_metadata_csv: str,
    query_embeddings_npy: str,
    query_metadata_csv: str,
    output_npz: str,
    output_preprocessor: str,
    n_components: int = 4,
):
    candidates = pd.read_csv(candidates_csv)
    qrels = pd.read_csv(qrels_csv)

    doc_embeddings = np.load(document_embeddings_npy)
    query_embeddings = np.load(query_embeddings_npy)

    doc_meta = pd.read_csv(document_metadata_csv)
    query_meta = pd.read_csv(query_metadata_csv)

    doc_id_to_index = {
        str(row["document_id"]): idx
        for idx, row in doc_meta.iterrows()
    }

    query_id_to_index = {
        str(row["query_id"]): idx
        for idx, row in query_meta.iterrows()
    }

    triplets = build_training_triplets(candidates, qrels)

    pos_features = []
    neg_features = []
    kept_rows = []

    for _, row in triplets.iterrows():
        qid = str(row["query_id"])
        pos_id = str(row["positive_doc_id"])
        neg_id = str(row["negative_doc_id"])

        if qid not in query_id_to_index:
            continue
        if pos_id not in doc_id_to_index:
            continue
        if neg_id not in doc_id_to_index:
            continue

        q_emb = query_embeddings[query_id_to_index[qid]]
        pos_emb = doc_embeddings[doc_id_to_index[pos_id]]
        neg_emb = doc_embeddings[doc_id_to_index[neg_id]]

        pos_features.append(create_pair_feature(q_emb, pos_emb))
        neg_features.append(create_pair_feature(q_emb, neg_emb))

        kept_rows.append({
            "query_id": qid,
            "positive_doc_id": pos_id,
            "negative_doc_id": neg_id
        })

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)

    all_features = np.vstack([pos_features, neg_features])

    reduced_all, scaler, pca = fit_feature_reducer(
        all_features,
        n_components=n_components
    )

    n = len(pos_features)
    pos_reduced = reduced_all[:n]
    neg_reduced = reduced_all[n:]

    Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        pos_features=pos_reduced,
        neg_features=neg_reduced,
        query_ids=np.array([r["query_id"] for r in kept_rows]),
        positive_doc_ids=np.array([r["positive_doc_id"] for r in kept_rows]),
        negative_doc_ids=np.array([r["negative_doc_id"] for r in kept_rows])
    )

    Path(output_preprocessor).parent.mkdir(parents=True, exist_ok=True)
    with open(output_preprocessor, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "pca": pca,
            "n_components": n_components
        }, f)

    print("Triplets kept:", len(kept_rows))
    print("Positive reduced shape:", pos_reduced.shape)
    print("Negative reduced shape:", neg_reduced.shape)
    print("Saved dataset:", output_npz)
    print("Saved preprocessor:", output_preprocessor)