from datetime import datetime
import time
import pandas as pd
from src.utils.logging_utils import save_experiment_log
from src.retrieval.hybrid import hybrid_search
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
    retrieved = hybrid_search(query_text, docs_csv, embeddings_npy, top_k=10)
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
results_df.to_csv("results/hybrid_results.csv", index=False)

mean_recall_3 = results_df["recall_at_3"].mean()
mean_recall_5 = results_df["recall_at_5"].mean()
mean_recall_10 = results_df["recall_at_10"].mean()
mean_mrr = results_df["mrr"].mean()
mean_ndcg_5 = results_df["ndcg_at_5"].mean()
mean_ndcg_10 = results_df["ndcg_at_10"].mean()
mean_map = results_df["map"].mean()
mean_latency = results_df["latency_ms"].mean()

print(results_df)
print("Mean Recall@3:", mean_recall_3)
print("Mean Recall@5:", mean_recall_5)
print("Mean Recall@10:", mean_recall_10)
print("MRR:", mean_mrr)
print("nDCG@5:", mean_ndcg_5)
print("nDCG@10:", mean_ndcg_10)
print("MAP:", mean_map)
print("Mean Latency (ms):", mean_latency)

save_experiment_log({
    "experiment_id": "E003",

    "timestamp": datetime.now().isoformat(),

    "method": "hybrid_baseline",

    "dataset": "BEIR SciFact",

    "embedding_model": "all-MiniLM-L6-v2",

    "fusion_method": "Weighted Sum",

    "top_k": 10,

    "random_seed": 42,

    "metrics": {
        "Recall@3": float(mean_recall_3),
        "Recall@5": float(mean_recall_5),
        "Recall@10": float(mean_recall_10),
        "MRR": float(mean_mrr),
        "nDCG@5": float(mean_ndcg_5),
        "nDCG@10": float(mean_ndcg_10),
        "MAP": float(mean_map),
        "Latency_ms": float(mean_latency)
    }
})