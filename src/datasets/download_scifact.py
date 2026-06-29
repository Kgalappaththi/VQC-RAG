from beir import util
from beir.datasets.data_loader import GenericDataLoader
from pathlib import Path
import pandas as pd


def main():
    dataset = "scifact"
    out_dir = Path("data/beir")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

    print("Downloading SciFact dataset...")
    data_path = util.download_and_unzip(url, str(out_dir))

    print("Loading dataset...")
    corpus, queries, qrels = GenericDataLoader(data_folder="data/beir/scifact").load(split="test")

    docs = []
    for doc_id, doc in corpus.items():
        docs.append({
            "document_id": doc_id,
            "title": doc.get("title", ""),
            "text": doc.get("text", "")
        })

    query_rows = []
    for query_id, query_text in queries.items():
        query_rows.append({
            "query_id": query_id,
            "query": query_text
        })

    qrels_rows = []
    for query_id, rel_docs in qrels.items():
        for doc_id, score in rel_docs.items():
            qrels_rows.append({
                "query_id": query_id,
                "document_id": doc_id,
                "relevance": score
            })

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/queries").mkdir(parents=True, exist_ok=True)
    Path("data/qrels").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(docs).to_csv("data/raw/documents.csv", index=False)
    pd.DataFrame(query_rows).to_csv("data/queries/queries.csv", index=False)
    pd.DataFrame(qrels_rows).to_csv("data/qrels/qrels.csv", index=False)

    print("SciFact converted successfully.")
    print("Documents:", len(docs))
    print("Queries:", len(query_rows))
    print("Qrels:", len(qrels_rows))


if __name__ == "__main__":
    main()