import pandas as pd
from src.retrieval.dense import dense_search
from src.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank

docs_csv = "data/processed/documents_clean.csv"
embeddings_npy = "results/document_embeddings.npy"

queries = pd.read_csv("data/queries/queries.csv")
qrels = pd.read_csv("data/qrels/qrels.csv")

results = []

for _, query_row in queries.iterrows():
    query_id = query_row["query_id"]
    query_text = query_row["query"]

    retrieved = dense_search(query_text, docs_csv, embeddings_npy, top_k=3)
    retrieved_ids = retrieved["document_id"].tolist()

    relevant_ids = qrels[qrels["query_id"] == query_id]["document_id"].tolist()

    results.append({
        "query_id": query_id,
        "recall_at_3": recall_at_k(retrieved_ids, relevant_ids, 3),
        "rr": reciprocal_rank(retrieved_ids, relevant_ids)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results/dense_results.csv", index=False)

mean_recall = results_df["recall_at_3"].mean()
mean_mrr = results_df["rr"].mean()

print(results_df)
print("Mean Recall@3:", mean_recall)
print("MRR:", mean_mrr)