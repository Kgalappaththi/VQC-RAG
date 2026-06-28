from pathlib import Path
import pandas as pd
import numpy as np


def check_file(path):
    path = Path(path)
    if path.exists():
        print(f"OK: {path}")
        return True
    else:
        print(f"MISSING: {path}")
        return False


def main():
    print("\nVQC-RAG Pipeline Verification\n")

    required_files = [
        "data/raw/documents.csv",
        "data/processed/documents_clean.csv",
        "data/queries/queries.csv",
        "data/qrels/qrels.csv",
        "results/document_embeddings.npy",
        "results/document_metadata.csv",
        "results/baseline_results.csv",
    ]

    all_ok = True

    print("1. Checking required files")
    for file in required_files:
        if not check_file(file):
            all_ok = False

    if not all_ok:
        print("\nSome files are missing. Run preprocessing, embeddings, and baseline scripts first.")
        return

    print("\n2. Checking documents and metadata")
    documents = pd.read_csv("data/processed/documents_clean.csv")
    metadata = pd.read_csv("results/document_metadata.csv")
    embeddings = np.load("results/document_embeddings.npy")

    print(f"Processed documents: {len(documents)}")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Embedding shape: {embeddings.shape}")

    if len(documents) == len(metadata) == embeddings.shape[0]:
        print("OK: Documents, metadata, and embeddings match.")
    else:
        print("ERROR: Documents, metadata, and embeddings do not match.")

    print("\n3. Checking queries and qrels")
    queries = pd.read_csv("data/queries/queries.csv")
    qrels = pd.read_csv("data/qrels/qrels.csv")

    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}")

    missing_qrels = set(queries["query_id"]) - set(qrels["query_id"])
    if missing_qrels:
        print(f"WARNING: These queries have no relevance labels: {missing_qrels}")
    else:
        print("OK: All queries have relevance labels.")

    print("\n4. Checking baseline results")
    baseline = pd.read_csv("results/baseline_results.csv")

    print(baseline)

    if "recall_at_3" in baseline.columns and "rr" in baseline.columns:
        mean_recall = baseline["recall_at_3"].mean()
        mean_mrr = baseline["rr"].mean()

        print(f"\nMean Recall@3: {mean_recall:.4f}")
        print(f"MRR: {mean_mrr:.4f}")
    else:
        print("ERROR: baseline_results.csv does not contain expected columns.")

    print("\nVerification completed.")


if __name__ == "__main__":
    main()