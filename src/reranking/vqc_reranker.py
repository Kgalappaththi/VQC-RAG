import pickle
import numpy as np
import pandas as pd

from src.quantum.features import (
    create_pair_feature,
    transform_features
)

from src.quantum.trainer import (
    build_trainable_vqc,
    quantum_relevance_score
)


class VQCReranker:

    def __init__(self, model_path):

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        self.weights = model["weights"]
        self.scaler = model["scaler"]
        self.pca = model["pca"]

        self.n_qubits = model["config"]["n_qubits"]
        self.depth = model["config"]["depth"]

        self.circuit = build_trainable_vqc(
            n_qubits=self.n_qubits,
            depth=self.depth
        )

    def compute_vqc_score(
        self,
        query_embedding,
        document_embedding
    ):

        feature = create_pair_feature(
            query_embedding,
            document_embedding
        )

        reduced = transform_features(
            feature.reshape(1, -1),
            self.scaler,
            self.pca
        )

        score = quantum_relevance_score(
            self.circuit,
            reduced[0],
            self.weights
        )

        return float(score)

    def rerank(
        self,
        candidates_df,
        query_embedding,
        document_embeddings,
        alpha=0.8
    ):

        quantum_scores = []

        for _, row in candidates_df.iterrows():

            doc_embedding = document_embeddings[
                int(row["embedding_index"])
            ]

            quantum_scores.append(

                self.compute_vqc_score(
                    query_embedding,
                    doc_embedding
                )

            )

        reranked = candidates_df.copy()

        reranked["vqc_score"] = quantum_scores

        reranked["final_score"] = (

            alpha * reranked["hybrid_score"]

            +

            (1 - alpha) * reranked["vqc_score"]

        )

        reranked = reranked.sort_values(

            "final_score",

            ascending=False

        )

        return reranked.reset_index(drop=True)