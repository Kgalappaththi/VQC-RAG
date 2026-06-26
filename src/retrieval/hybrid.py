import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def tokenize(text):
    return str(text).lower().split()

def hybrid_search(query, docs_csv, embeddings_npy, top_k=5, alpha=0.5):
    docs = pd.read_csv(docs_csv)
    doc_embeddings = np.load(embeddings_npy)

    tokenized_corpus = [tokenize(doc) for doc in docs["text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenize(query))

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query])
    dense_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    scaler = MinMaxScaler()
    score_matrix = np.column_stack([bm25_scores, dense_scores])
    normalized = scaler.fit_transform(score_matrix)

    hybrid_scores = alpha * normalized[:, 0] + (1 - alpha) * normalized[:, 1]

    results = docs.copy()
    results["bm25_score"] = bm25_scores
    results["dense_score"] = dense_scores
    results["hybrid_score"] = hybrid_scores

    return results.sort_values("hybrid_score", ascending=False).head(top_k)

if __name__ == "__main__":
    results = hybrid_search(
        "What is quantum computing?",
        "data/processed/documents_clean.csv",
        "results/document_embeddings.npy",
        top_k=3
    )
    print(results[["document_id", "title", "hybrid_score"]])