import pandas as pd

from src.quantum.triplets import build_training_triplets


def test_build_training_triplets():
    candidates = pd.DataFrame({
        "query_id": [1, 1, 1, 2, 2],
        "document_id": ["D1", "D2", "D3", "D4", "D5"],
        "hybrid_score": [0.9, 0.8, 0.7, 0.9, 0.6]
    })

    qrels = pd.DataFrame({
        "query_id": [1, 2],
        "document_id": ["D2", "D5"],
        "relevance": [1, 1]
    })

    triplets = build_training_triplets(candidates, qrels)

    print(triplets)

    assert len(triplets) == 2
    assert triplets.iloc[0]["positive_doc_id"] == "D2"
    assert triplets.iloc[0]["negative_doc_id"] == "D1"

    print("Triplet tests passed.")


if __name__ == "__main__":
    test_build_training_triplets()