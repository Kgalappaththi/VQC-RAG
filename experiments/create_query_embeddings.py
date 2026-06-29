import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    queries_csv = "data/queries/queries.csv"
    output_embeddings = "results/query_embeddings.npy"
    output_metadata = "results/query_metadata.csv"

    queries = pd.read_csv(queries_csv)

    model = SentenceTransformer(MODEL_NAME)
    query_texts = queries["query"].astype(str).tolist()

    embeddings = model.encode(query_texts, show_progress_bar=True)

    Path("results").mkdir(exist_ok=True)

    np.save(output_embeddings, embeddings)

    queries[["query_id", "query"]].to_csv(output_metadata, index=False)

    print("Query embeddings saved to:", output_embeddings)
    print("Query metadata saved to:", output_metadata)
    print("Embedding shape:", embeddings.shape)


if __name__ == "__main__":
    main()