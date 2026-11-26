
"""Benchmark held-out MLP and simulated VQC re-ranking latency."""

from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from src.quantum.features import transform_features
from src.quantum.trainer import quantum_relevance_score

from experiments.scientific_reports_revision.evaluate_heldout_ranking import (
    load_evaluation_data,
    load_evaluation_models,
    build_candidate_features,
)

RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "scientific_reports_revision"
)

REPEATS = 3
WARMUP_RUNS = 1


def main():
    (
        candidates,
        _,
        doc_embeddings,
        query_embeddings,
        doc_id_to_index,
        query_id_to_index,
    ) = load_evaluation_data()

    mlp_checkpoint, vqc_checkpoint, circuit = (
        load_evaluation_models()
    )

    # Prepare features once, outside the timing experiment.
    features = build_candidate_features(
        candidates,
        doc_embeddings,
        query_embeddings,
        doc_id_to_index,
        query_id_to_index,
    )

    candidates = candidates.reset_index(drop=True)
    assert features.shape == (1200, 1536)

    mlp_model = mlp_checkpoint["model"]
    scaler = mlp_checkpoint["scaler"]
    pca = mlp_checkpoint["pca"]
    weights = vqc_checkpoint["weights"]

    assert np.allclose(
        scaler.mean_,
        vqc_checkpoint["scaler"].mean_
    )
    assert np.allclose(
        pca.components_,
        vqc_checkpoint["pca"].components_
    )

    query_ids = sorted(candidates["query_id"].unique())
    assert len(query_ids) == 60

    def preprocess(x):
        return transform_features(x, scaler, pca)

    def mlp_inference(x):
        return mlp_model.predict_proba(x)

    def vqc_inference(x):
        return np.asarray([
            float(quantum_relevance_score(circuit, row, weights))
            for row in x
        ])

    records = []

    for query_number, query_id in enumerate(query_ids, start=1):
        indices = np.flatnonzero(
            candidates["query_id"].to_numpy() == query_id
        )

        assert len(indices) == 20
        x = features[indices]

        # Warm up both models without recording timings.
        for _ in range(WARMUP_RUNS):
            reduced = preprocess(x)
            mlp_inference(reduced)
            vqc_inference(reduced)

        for repeat in range(1, REPEATS + 1):
            # Common preprocessing measurement.
            start = time.perf_counter()
            reduced = preprocess(x)
            preprocessing_ms = (
                time.perf_counter() - start
            ) * 1000

            # MLP inference on the same 20 candidates.
            start = time.perf_counter()
            mlp_scores = mlp_inference(reduced)
            mlp_ms = (time.perf_counter() - start) * 1000

            # Simulated VQC inference on the same 20 candidates.
            start = time.perf_counter()
            vqc_scores = vqc_inference(reduced)
            vqc_ms = (time.perf_counter() - start) * 1000

            assert mlp_scores.shape == (20, 2)
            assert vqc_scores.shape == (20,)
            assert np.isfinite(mlp_scores).all()
            assert np.isfinite(vqc_scores).all()

            records.append({
                "query_id": query_id,
                "repeat": repeat,
                "candidate_count": 20,
                "preprocessing_ms": preprocessing_ms,
                "mlp_inference_ms": mlp_ms,
                "vqc_inference_ms": vqc_ms,
                "mlp_reranking_ms": preprocessing_ms + mlp_ms,
                "vqc_reranking_ms": preprocessing_ms + vqc_ms,
            })

        if query_number % 10 == 0:
            print(
                f"Benchmarked {query_number}/60 queries",
                flush=True,
            )

    timings = pd.DataFrame(records)

    assert len(timings) == 60 * REPEATS

    columns = [
        "preprocessing_ms",
        "mlp_inference_ms",
        "vqc_inference_ms",
        "mlp_reranking_ms",
        "vqc_reranking_ms",
    ]

    summary = timings[columns].agg(
        ["mean", "median", "std", "min", "max"]
    ).T

    summary["p95"] = timings[columns].quantile(0.95)

    timings.to_csv(
        RESULTS_DIR / "heldout_latency_per_query.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS_DIR / "heldout_latency_summary.csv"
    )

    environment = pd.DataFrame([{
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "queries": 60,
        "candidates_per_query": 20,
        "repeats": REPEATS,
        "warmup_runs_per_query": WARMUP_RUNS,
        "quantum_backend": "PennyLane default.qubit simulator",
        "vqc_qubits": 4,
        "vqc_depth": 4,
        "timing_scope": (
            "Precomputed features; preprocessing and "
            "inference measured separately"
        ),
    }])

    environment.to_csv(
        RESULTS_DIR / "heldout_latency_environment.csv",
        index=False,
    )

    print("\nLatency summary (milliseconds):")
    print(summary.round(4).to_string())
    print("\nLatency benchmarking completed.")


if __name__ == "__main__":
    main()
