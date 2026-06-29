import time
import pandas as pd
from src.retrieval.dense import dense_search
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    average_precision,
)

docs_csv = "data/processed/documents_clean.csv"
embeddings_npy = "results/document_embeddings.npy"

queries = pd.read_csv("data/queries/queries.csv")
qrels = pd.read_csv("data/qrels/qrels.csv")

results = []

for _, query_row in queries.iterrows():
    query_id = query_row["query_id"]
    query_text = query_row["query"]

    start_time = time.perf_counter()
    retrieved = dense_search(query_text, docs_csv, embeddings_npy, top_k=10)
    latency_ms = (time.perf_counter() - start_time) * 1000
    retrieved_ids = retrieved["document_id"].astype(str).tolist()

    relevant_ids = qrels[qrels["query_id"] == query_id]["document_id"].astype(str).tolist()

    results.append({
        "query_id": query_id,
        "recall_at_3": recall_at_k(retrieved_ids, relevant_ids, 3),
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
        "ndcg_at_5": ndcg_at_k(retrieved_ids, relevant_ids, 5),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, relevant_ids, 10),
        "map": average_precision(retrieved_ids, relevant_ids),
        "latency_ms": latency_ms,
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results/dense_results.csv", index=False)

print(results_df)
print("Mean Recall@3:", results_df["recall_at_3"].mean())
print("Mean Recall@5:", results_df["recall_at_5"].mean())
print("Mean Recall@10:", results_df["recall_at_10"].mean())
print("MRR:", results_df["mrr"].mean())
print("nDCG@5:", results_df["ndcg_at_5"].mean())
print("nDCG@10:", results_df["ndcg_at_10"].mean())
print("MAP:", results_df["map"].mean())
print("Mean Latency (ms):", results_df["latency_ms"].mean())