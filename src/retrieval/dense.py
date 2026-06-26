import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def dense_search(query, docs_csv, embeddings_npy, top_k=5):
    docs = pd.read_csv(docs_csv)
    doc_embeddings = np.load(embeddings_npy)

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query])

    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    results = docs.copy()
    results["dense_score"] = scores

    return results.sort_values("dense_score", ascending=False).head(top_k)

if __name__ == "__main__":
    results = dense_search(
        "What is quantum computing?",
        "data/processed/documents_clean.csv",
        "results/document_embeddings.npy",
        top_k=3
    )
    print(results[["document_id", "title", "dense_score"]])