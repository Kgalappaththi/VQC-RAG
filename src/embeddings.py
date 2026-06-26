import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def generate_embeddings(input_csv: str, output_npy: str, metadata_csv: str):
    df = pd.read_csv(input_csv)
    model = SentenceTransformer(MODEL_NAME)

    texts = df["text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    Path(output_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, embeddings)

    df[["document_id", "title"]].to_csv(metadata_csv, index=False)

if __name__ == "__main__":
    generate_embeddings(
        "data/processed/documents_clean.csv",
        "results/document_embeddings.npy",
        "results/document_metadata.csv"
    )