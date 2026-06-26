import pandas as pd
from rank_bm25 import BM25Okapi

def tokenize(text):
    return str(text).lower().split()

def bm25_search(query, documents_df, top_k=5):
    corpus = documents_df["text"].tolist()
    tokenized_corpus = [tokenize(doc) for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize(query))

    results = documents_df.copy()
    results["bm25_score"] = scores

    return results.sort_values("bm25_score", ascending=False).head(top_k)

if __name__ == "__main__":
    docs = pd.read_csv("data/processed/documents_clean.csv")
    results = bm25_search("What is quantum computing?", docs, top_k=3)
    print(results[["document_id", "title", "bm25_score"]])