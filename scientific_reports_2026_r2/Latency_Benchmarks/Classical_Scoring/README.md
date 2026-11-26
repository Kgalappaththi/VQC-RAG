# Classical scoring benchmark

Run from the repository root: `python Latency_Benchmarks/Classical_Scoring/benchmark_trained_classical.py --repo VQC_SciFact_Reproducibility_v1 --repeats 5 --out new_classical_latency`. The seed-42 MLP is refitted on frozen training pairs; RankSVM uses archived coefficients. Timings cover cached features, not end-to-end retrieval. Vectorised classical and per-candidate VQC implementations differ; no intrinsic speedup claim is made.
