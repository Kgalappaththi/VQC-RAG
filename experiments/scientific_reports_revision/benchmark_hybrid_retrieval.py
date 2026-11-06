
"""Validate and benchmark optimised Hybrid Top-20 retrieval."""

from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from src.retrieval.hybrid import tokenize

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
REVISION = RESULTS / "scientific_reports_revision"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 20
ALPHA = 0.5
REPEATS = 3


def retrieve(query, model, bm25, docs, doc_embeddings):
    """Retrieve Top-20 using preloaded resources."""

    start = time.perf_counter()
    query_embedding = model.encode([query])
    encoding_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()

    bm25_scores = bm25.get_scores(tokenize(query))
    dense_scores = cosine_similarity(
        query_embedding, doc_embeddings
    )[0]

    score_matrix = np.column_stack([
        bm25_scores, dense_scores
    ])

    normalised = MinMaxScaler().fit_transform(
        score_matrix
    )

    hybrid_scores = (
        ALPHA * normalised[:, 0]
        + (1 - ALPHA) * normalised[:, 1]
    )

    ranked_indices = np.argsort(
        -hybrid_scores, kind="stable"
    )[:TOP_K]

    scoring_ms = (time.perf_counter() - start) * 1000

    retrieved_ids = (
        docs.iloc[ranked_indices]["document_id"]
        .astype(str)
        .tolist()
    )

    return retrieved_ids, encoding_ms, scoring_ms


def main():
    queries = pd.read_csv(
        ROOT / "data/queries/queries.csv"
    )

    splits = pd.read_csv(
        REVISION / "query_splits.csv"
    )

    original = pd.read_csv(
        RESULTS / "hybrid_top20_candidates.csv"
    )

    test_ids = set(
        splits.loc[
            splits["split"] == "test", "query_id"
        ].astype(str)
    )

    test_queries = queries[
        queries["query_id"].astype(str).isin(test_ids)
    ].copy()

    assert len(test_queries) == 60

    print("Loading retrieval resources...", flush=True)

    docs = pd.read_csv(
        ROOT / "data/processed/documents_clean.csv"
    )

    doc_embeddings = np.load(
        RESULTS / "document_embeddings.npy"
    )

    assert len(docs) == len(doc_embeddings) == 5183

    tokenised = [
        tokenize(text) for text in docs["text"]
    ]

    bm25 = BM25Okapi(tokenised)
    model = SentenceTransformer(MODEL_NAME)

    print("Resources loaded successfully.", flush=True)

    validation = []
    timings = []

    for number, row in enumerate(
        test_queries.itertuples(index=False), start=1
    ):
        query_id = str(row.query_id)
        query_text = row.query

        expected = (
            original[
                original["query_id"].astype(str) == query_id
            ]["document_id"]
            .astype(str)
            .tolist()
        )

        assert len(expected) == TOP_K

        # Warm-up run and retrieval consistency check.
        actual, _, _ = retrieve(
            query_text, model, bm25, docs, doc_embeddings
        )

        exact_match = actual == expected
        set_match = set(actual) == set(expected)

        validation.append({
            "query_id": query_id,
            "exact_order_match": exact_match,
            "candidate_set_match": set_match,
        })

        if not exact_match:
            print(
                f"Ranking mismatch for query {query_id}",
                flush=True
            )

        for repeat in range(1, REPEATS + 1):
            actual, encoding_ms, scoring_ms = retrieve(
                query_text, model, bm25, docs, doc_embeddings
            )

            timings.append({
                "query_id": query_id,
                "repeat": repeat,
                "candidate_count": TOP_K,
                "query_encoding_ms": encoding_ms,
                "retrieval_scoring_ms": scoring_ms,
                "total_retrieval_ms": (
                    encoding_ms + scoring_ms
                ),
            })

        if number % 10 == 0:
            print(
                f"Processed {number}/60 queries",
                flush=True
            )

    validation_df = pd.DataFrame(validation)
    timings_df = pd.DataFrame(timings)

    validation_df.to_csv(
        REVISION / "hybrid_retrieval_validation.csv",
        index=False
    )

    timings_df.to_csv(
        REVISION / "hybrid_retrieval_latency_per_query.csv",
        index=False
    )

    columns = [
        "query_encoding_ms",
        "retrieval_scoring_ms",
        "total_retrieval_ms",
    ]

    summary = timings_df[columns].agg(
        ["mean", "median", "std", "min", "max"]
    ).T

    summary["p95"] = timings_df[columns].quantile(
        0.95
    )

    summary.to_csv(
        REVISION / "hybrid_retrieval_latency_summary.csv"
    )

    environment = pd.DataFrame([{
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "model": MODEL_NAME,
        "documents": len(docs),
        "test_queries": len(test_queries),
        "top_k": TOP_K,
        "alpha": ALPHA,
        "repeats": REPEATS,
        "resources_preloaded": True,
    }])

    environment.to_csv(
        REVISION / "hybrid_retrieval_environment.csv",
        index=False
    )

    print("\nRetrieval consistency:")
    print(
        "Exact ranking matches:",
        int(validation_df["exact_order_match"].sum()),
        "/60"
    )
    print(
        "Candidate set matches:",
        int(validation_df["candidate_set_match"].sum()),
        "/60"
    )

    print("\nHybrid retrieval latency (ms):")
    print(summary.round(4).to_string())

    if not validation_df["exact_order_match"].all():
        print(
            "\nWARNING: Some rankings differ from "
            "the original candidate file."
        )
        print(
            "Do not claim exact retrieval equivalence "
            "until differences are investigated."
        )

    print("\nHybrid retrieval benchmark completed.")


if __name__ == "__main__":
    main()
