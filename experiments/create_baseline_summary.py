import pandas as pd

summary = []

files = {
    "BM25": "results/bm25_results.csv",
    "Dense": "results/dense_results.csv",
    "Hybrid": "results/hybrid_results.csv"
}

for method, file_path in files.items():
    df = pd.read_csv(file_path)

    summary.append({
        "Method": method,
        "Recall@3": round(df["recall_at_3"].mean(), 4),
        "MRR": round(df["rr"].mean(), 4)
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("results/baseline_summary.csv", index=False)

print(summary_df)