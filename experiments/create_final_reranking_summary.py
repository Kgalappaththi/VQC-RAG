import pandas as pd
from pathlib import Path


FILES = {
    "Hybrid": "results/hybrid_results.csv",
    "Hybrid + VQC Weighted": "results/vqc_reranking_results.csv",
    "Hybrid + VQC Rank": "results/vqc_reranking_rank_results.csv",
    "Hybrid + VQC RRF": "results/vqc_reranking_rrf_results.csv",
}


def summarize(path):
    df = pd.read_csv(path)
    return {
        "Recall@3": df["recall_at_3"].mean(),
        "Recall@5": df["recall_at_5"].mean(),
        "Recall@10": df["recall_at_10"].mean(),
        "MRR": df["mrr"].mean(),
        "nDCG@5": df["ndcg_at_5"].mean(),
        "nDCG@10": df["ndcg_at_10"].mean(),
        "MAP": df["map"].mean(),
        "Latency_ms": df["latency_ms"].mean(),
    }


def main():
    rows = []
    for method, path in FILES.items():
        metrics = summarize(path)
        metrics["Method"] = method
        rows.append(metrics)

    summary = pd.DataFrame(rows)
    summary = summary[
        ["Method", "Recall@3", "Recall@5", "Recall@10",
         "MRR", "nDCG@5", "nDCG@10", "MAP", "Latency_ms"]
    ]

    Path("results/statistics").mkdir(parents=True, exist_ok=True)
    output = "results/statistics/final_reranking_comparison.csv"
    summary.to_csv(output, index=False)

    print(summary.to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()