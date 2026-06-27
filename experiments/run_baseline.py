import pandas as pd
from src.retrieval.hybrid import hybrid_search
from src.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank

docs_csv = "data/processed/documents_clean.csv"
embeddings_npy = "results/document_embeddings.npy"

queries = pd.read_csv("data/queries/queries.csv")
qrels = pd.read_csv("data/qrels/qrels.csv")

results = []

for _, query_row in queries.iterrows():
    query_id = query_row["query_id"]
    query_text = query_row["query"]

    retrieved = hybrid_search(
        query_text,
        docs_csv,
        embeddings_npy,
        top_k=3
    )

    retrieved_ids = retrieved["document_id"].tolist()

    relevant_ids = qrels[qrels["query_id"] == query_id]["document_id"].tolist()

    results.append({
        "query_id": query_id,
        "recall_at_3": recall_at_k(retrieved_ids, relevant_ids, 3),
        "rr": reciprocal_rank(retrieved_ids, relevant_ids)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results/baseline_results.csv", index=False)

print(results_df)
print("Mean Recall@3:", results_df["recall_at_3"].mean())
print("MRR:", results_df["rr"].mean())

save_experiment_log({
    "experiment_id": "E001",
    "method": "hybrid_baseline",
    "dataset": "sample",
    "top_k": 3,
    "embedding_model": "all-MiniLM-L6-v2",
    "mean_recall_at_3": float(mean_recall),
    "mean_mrr": float(mean_mrr)
})