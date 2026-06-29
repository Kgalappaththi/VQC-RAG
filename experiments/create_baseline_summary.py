import pandas as pd

files = {
    "BM25": "results/bm25_results.csv",
    "Dense": "results/dense_results.csv",
    "Hybrid": "results/hybrid_results.csv"
}

summary = []

for method, file_path in files.items():
    df = pd.read_csv(file_path)

    summary.append({
        "Method": method,
        "Recall@3": round(df["recall_at_3"].mean(), 4),
        "Recall@5": round(df["recall_at_5"].mean(), 4),
        "Recall@10": round(df["recall_at_10"].mean(), 4),
        "MRR": round(df["mrr"].mean(), 4),
        "nDCG@5": round(df["ndcg_at_5"].mean(), 4),
        "nDCG@10": round(df["ndcg_at_10"].mean(), 4),
        "MAP": round(df["map"].mean(), 4),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("results/baseline_summary.csv", index=False)

print(summary_df)