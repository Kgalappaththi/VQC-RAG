import pandas as pd
from src.retrieval.bm25 import bm25_search
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    average_precision,
)

documents_df = pd.read_csv("data/processed/documents_clean.csv")
queries = pd.read_csv("data/queries/queries.csv")
qrels = pd.read_csv("data/qrels/qrels.csv")

results = []

for _, query_row in queries.iterrows():
    query_id = query_row["query_id"]
    query_text = query_row["query"]

    retrieved = bm25_search(query_text, documents_df, top_k=10)
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
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results/bm25_results.csv", index=False)

print(results_df)
print("Mean Recall@3:", results_df["recall_at_3"].mean())
print("Mean Recall@5:", results_df["recall_at_5"].mean())
print("Mean Recall@10:", results_df["recall_at_10"].mean())
print("MRR:", results_df["mrr"].mean())
print("nDCG@5:", results_df["ndcg_at_5"].mean())
print("nDCG@10:", results_df["ndcg_at_10"].mean())
print("MAP:", results_df["map"].mean())