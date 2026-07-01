"""
=============================================================
Experiment : E006
Module     : Hybrid + VQC Re-ranking
Project    : Quantum-Enhanced Hybrid Retrieval-Augmented Generation
Author     : Kalpana Galappaththi
Repository : VQC-RAG
=============================================================

Purpose
-------
This experiment evaluates whether a trained Variational Quantum
Circuit (VQC) can improve the ranking quality of a classical
Hybrid Retrieval-Augmented Generation (RAG) pipeline.

Research Question
-----------------
Can a trained 6-qubit VQC re-ranker improve retrieval effectiveness
when applied to the Top-20 candidate documents retrieved by a
classical Hybrid Retriever?

Workflow
--------
User query
    ↓
Hybrid retrieval
    ↓
Top-20 candidate documents
    ↓
Query-document interaction feature construction
    ↓
StandardScaler + PCA transformation
    ↓
Trained 6-qubit VQC relevance scoring
    ↓
Hybrid–Quantum score fusion
    ↓
Re-ranked candidate list
    ↓
Retrieval evaluation

Evaluation Metrics
------------------
Recall@3, Recall@5, Recall@10, MRR, nDCG@5, nDCG@10, MAP, latency.

Important Note
--------------
The Hybrid Retriever is not modified. The VQC is used only as a
post-retrieval re-ranking module. This ensures fair comparison with
the existing Hybrid baseline.
"""

from datetime import datetime
from src.fusion.weighted_fusion import weighted_score_fusion
from src.fusion.rank_fusion import rank_fusion
from src.fusion.rrf_fusion import reciprocal_rank_fusion
import time
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.logging_utils import save_experiment_log
from src.retrieval.hybrid import hybrid_search
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    average_precision,
)
from src.quantum.features import create_pair_feature, transform_features
from src.quantum.trainer import build_trainable_vqc, quantum_relevance_score


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

DOCS_CSV = "data/processed/documents_clean.csv"
DOC_EMBEDDINGS_NPY = "results/document_embeddings.npy"
QUERIES_CSV = "data/queries/queries.csv"
QRELS_CSV = "data/qrels/qrels.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Best-performing VQC model
VQC_MODEL_PATH = "results/vqc/trained_vqc_6q_depth4_e100_lr001.pkl"

# ---------------------------------------------------------
# Fusion method
# Options:
#   "weighted"
#   "rank"
#   "rrf"
# ---------------------------------------------------------
FUSION_METHOD = "rrf"

# Weight used by weighted and rank fusion
ALPHA = 0.8

TOP_K_RETRIEVAL = 20

OUTPUT_RESULTS = f"results/vqc_reranking_{FUSION_METHOD}_results.csv"


def normalize_scores(values):
    """
    Normalize scores into the range [0, 1].

    Rationale
    ---------
    Hybrid retrieval scores and VQC scores may have different numeric
    scales. Min-max normalization ensures that both scores are comparable
    before weighted fusion.

    Parameters
    ----------
    values : array-like
        Raw retrieval or VQC scores.

    Returns
    -------
    numpy.ndarray
        Normalized scores in the range [0, 1].
    """
    values = np.array(values, dtype=float)

    min_value = values.min()
    max_value = values.max()

    if max_value - min_value == 0:
        return np.ones_like(values) * 0.5

    return (values - min_value) / (max_value - min_value)


def load_vqc_model(model_path):
    """
    Load the trained VQC model and its preprocessing components.

    The saved model contains:
    - trained VQC weights,
    - fitted StandardScaler,
    - fitted PCA transformer,
    - circuit configuration.

    Parameters
    ----------
    model_path : str
        Path to the saved VQC model file.

    Returns
    -------
    dict
        Loaded VQC model dictionary.
    """
    with open(model_path, "rb") as file:
        return pickle.load(file)


def build_document_index(docs_df):
    """
    Build a mapping from document ID to embedding row index.

    This allows fast retrieval of the correct dense embedding for each
    candidate document returned by the Hybrid Retriever.

    Parameters
    ----------
    docs_df : pandas.DataFrame
        Document dataframe containing document_id.

    Returns
    -------
    dict
        Mapping from document_id to row index in the embedding matrix.
    """
    return {
        str(row["document_id"]): index
        for index, row in docs_df.iterrows()
    }


def compute_vqc_score(
    query_embedding,
    document_embedding,
    scaler,
    pca,
    circuit,
    weights,
):
    """
    Compute the trained VQC relevance score for a query-document pair.

    Steps
    -----
    1. Construct query-document interaction features.
    2. Apply the fitted StandardScaler and PCA transformation.
    3. Feed the reduced feature vector into the trained VQC.
    4. Return the quantum relevance score.

    Parameters
    ----------
    query_embedding : numpy.ndarray
        Dense embedding of the query.

    document_embedding : numpy.ndarray
        Dense embedding of the candidate document.

    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used during VQC training.

    pca : sklearn.decomposition.PCA
        Fitted PCA transformer used during VQC training.

    circuit : pennylane.QNode
        Trainable VQC circuit structure.

    weights : numpy.ndarray
        Trained VQC parameters.

    Returns
    -------
    float
        VQC relevance score in the range [0, 1].
    """
    pair_feature = create_pair_feature(query_embedding, document_embedding)

    reduced_feature = transform_features(
        pair_feature.reshape(1, -1),
        scaler,
        pca,
    )[0]

    return float(
        quantum_relevance_score(
            circuit,
            reduced_feature,
            weights,
        )
    )


def rerank_with_vqc(
    retrieved_df,
    query_embedding,
    document_embeddings,
    doc_id_to_index,
    scaler,
    pca,
    circuit,
    weights,
    alpha,
):
    """
    Re-rank Hybrid Retriever candidates using trained VQC scores.

    The final score is:

        final_score = alpha * normalized_hybrid_score
                    + (1 - alpha) * normalized_vqc_score

    Parameters
    ----------
    retrieved_df : pandas.DataFrame
        Top-k candidates returned by the Hybrid Retriever.

    query_embedding : numpy.ndarray
        Dense query embedding.

    document_embeddings : numpy.ndarray
        Matrix of document embeddings.

    doc_id_to_index : dict
        Mapping from document ID to embedding row index.

    scaler, pca, circuit, weights
        Components of the trained VQC re-ranker.

    alpha : float
        Fusion weight between Hybrid and VQC scores.

    Returns
    -------
    pandas.DataFrame
        Re-ranked candidate dataframe.
    """
    vqc_scores = []

    for _, row in retrieved_df.iterrows():
        doc_id = str(row["document_id"])

        if doc_id not in doc_id_to_index:
            # If a document ID cannot be mapped to an embedding, assign
            # a neutral score rather than failing the full experiment.
            vqc_scores.append(0.5)
            continue

        doc_embedding = document_embeddings[doc_id_to_index[doc_id]]

        vqc_score = compute_vqc_score(
            query_embedding=query_embedding,
            document_embedding=doc_embedding,
            scaler=scaler,
            pca=pca,
            circuit=circuit,
            weights=weights,
        )

        vqc_scores.append(vqc_score)

    reranked = retrieved_df.copy()

    reranked["vqc_score"] = vqc_scores
    reranked["hybrid_score_norm"] = normalize_scores(reranked["hybrid_score"])
    reranked["vqc_score_norm"] = normalize_scores(reranked["vqc_score"])

    if FUSION_METHOD == "weighted":
        reranked["final_score"] = weighted_score_fusion(
            reranked["hybrid_score"],
            reranked["vqc_score"],
            alpha=ALPHA
        )

    elif FUSION_METHOD == "rank":
        reranked["final_score"] = rank_fusion(
            reranked["hybrid_score"],
            reranked["vqc_score"],
            alpha=ALPHA
        )

    elif FUSION_METHOD == "rrf":
        reranked["final_score"] = reciprocal_rank_fusion(
            reranked["hybrid_score"],
            reranked["vqc_score"]
        )

    else:
        raise ValueError(f"Unknown fusion method: {FUSION_METHOD}")

    return reranked.sort_values("final_score", ascending=False).reset_index(drop=True)


def evaluate_ranking(retrieved_ids, relevant_ids):
    """
    Compute retrieval evaluation metrics for a ranked document list.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ranked document IDs returned by the system.

    relevant_ids : list[str]
        Ground-truth relevant document IDs.

    Returns
    -------
    dict
        Retrieval metric values for one query.
    """
    return {
        "recall_at_3": recall_at_k(retrieved_ids, relevant_ids, 3),
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
        "ndcg_at_5": ndcg_at_k(retrieved_ids, relevant_ids, 5),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, relevant_ids, 10),
        "map": average_precision(retrieved_ids, relevant_ids),
    }


def summarize_results(results_df):
    """
    Compute mean retrieval metrics across all evaluated queries.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Per-query evaluation results.

    Returns
    -------
    dict
        Mean values of all evaluation metrics.
    """
    return {
        "Recall@3": float(results_df["recall_at_3"].mean()),
        "Recall@5": float(results_df["recall_at_5"].mean()),
        "Recall@10": float(results_df["recall_at_10"].mean()),
        "MRR": float(results_df["mrr"].mean()),
        "nDCG@5": float(results_df["ndcg_at_5"].mean()),
        "nDCG@10": float(results_df["ndcg_at_10"].mean()),
        "MAP": float(results_df["map"].mean()),
        "Latency_ms": float(results_df["latency_ms"].mean()),
    }


def main():
    """
    Execute the Hybrid + VQC re-ranking experiment.
    """
    queries = pd.read_csv(QUERIES_CSV)
    qrels = pd.read_csv(QRELS_CSV)
    docs = pd.read_csv(DOCS_CSV)

    document_embeddings = np.load(DOC_EMBEDDINGS_NPY)

    doc_id_to_index = build_document_index(docs)

    embedding_model = SentenceTransformer(MODEL_NAME)

    vqc_model = load_vqc_model(VQC_MODEL_PATH)

    weights = vqc_model["weights"]
    scaler = vqc_model["scaler"]
    pca = vqc_model["pca"]

    config = vqc_model["config"]
    n_qubits = config["n_qubits"]
    depth = config["depth"]

    circuit = build_trainable_vqc(
        n_qubits=n_qubits,
        depth=depth,
    )

    results = []

    for _, query_row in queries.iterrows():
        query_id = query_row["query_id"]
        query_text = query_row["query"]

        start_time = time.perf_counter()

        # Step 1: Retrieve Top-20 candidate documents using the existing
        # validated Hybrid Retriever.
        retrieved = hybrid_search(
            query_text,
            DOCS_CSV,
            DOC_EMBEDDINGS_NPY,
            top_k=TOP_K_RETRIEVAL,
        )

        # Step 2: Encode the query once and use it for all candidate pairs.
        query_embedding = embedding_model.encode([query_text])[0]

        # Step 3: Apply VQC-based post-retrieval re-ranking.
        reranked = rerank_with_vqc(
            retrieved_df=retrieved,
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            doc_id_to_index=doc_id_to_index,
            scaler=scaler,
            pca=pca,
            circuit=circuit,
            weights=weights,
            alpha=ALPHA,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        retrieved_ids = reranked["document_id"].astype(str).tolist()

        relevant_ids = (
            qrels[qrels["query_id"] == query_id]["document_id"]
            .astype(str)
            .tolist()
        )

        query_metrics = evaluate_ranking(retrieved_ids, relevant_ids)
        query_metrics["query_id"] = query_id
        query_metrics["latency_ms"] = latency_ms

        results.append(query_metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_RESULTS, index=False)

    summary = summarize_results(results_df)

    print(results_df)
    print("Mean Recall@3:", summary["Recall@3"])
    print("Mean Recall@5:", summary["Recall@5"])
    print("Mean Recall@10:", summary["Recall@10"])
    print("MRR:", summary["MRR"])
    print("nDCG@5:", summary["nDCG@5"])
    print("nDCG@10:", summary["nDCG@10"])
    print("MAP:", summary["MAP"])
    print("Mean Latency (ms):", summary["Latency_ms"])

    save_experiment_log({
        "experiment_id": "E006",
        "timestamp": datetime.now().isoformat(),
        "method": f"hybrid_plus_vqc_{FUSION_METHOD}_fusion",
        "dataset": "BEIR SciFact",
        "embedding_model": MODEL_NAME,
        "vqc_model": VQC_MODEL_PATH,
        "top_k": TOP_K_RETRIEVAL,
        "fusion_alpha": ALPHA,
        "metrics": summary,
        "fusion_method": FUSION_METHOD,
    })


if __name__ == "__main__":
    main()