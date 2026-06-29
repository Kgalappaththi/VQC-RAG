import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.retrieval.hybrid import hybrid_search


def main():
    docs_csv = "data/processed/documents_clean.csv"
    embeddings_npy = "results/document_embeddings.npy"
    queries_csv = "data/queries/queries.csv"

    output_path = "results/hybrid_top20_candidates.csv"

    queries = pd.read_csv(queries_csv)

    all_candidates = []

    for _, row in tqdm(queries.iterrows(), total=len(queries)):
        query_id = row["query_id"]
        query_text = row["query"]

        candidates = hybrid_search(
            query=query_text,
            docs_csv=docs_csv,
            embeddings_npy=embeddings_npy,
            top_k=20,
            alpha=0.5
        )

        candidates = candidates.copy()
        candidates["query_id"] = query_id
        candidates["query"] = query_text

        all_candidates.append(candidates)

    result = pd.concat(all_candidates, ignore_index=True)

    Path("results").mkdir(exist_ok=True)
    result.to_csv(output_path, index=False)

    print("Hybrid Top-20 candidates saved to:", output_path)
    print("Total rows:", len(result))


if __name__ == "__main__":
    main()