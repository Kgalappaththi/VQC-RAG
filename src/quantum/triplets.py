import pandas as pd


def build_training_triplets(candidates_df, qrels_df):
    """
    Build triplets: query_id, positive_doc_id, negative_doc_id.

    Positive document:
        A relevant document from qrels.

    Negative document:
        A retrieved candidate document that is not relevant.
    """

    triplets = []

    for query_id in candidates_df["query_id"].unique():
        query_candidates = candidates_df[candidates_df["query_id"] == query_id]

        relevant_docs = set(
            qrels_df[qrels_df["query_id"] == query_id]["document_id"].astype(str)
        )

        if not relevant_docs:
            continue

        positives = query_candidates[
            query_candidates["document_id"].astype(str).isin(relevant_docs)
        ]

        negatives = query_candidates[
            ~query_candidates["document_id"].astype(str).isin(relevant_docs)
        ]

        if positives.empty or negatives.empty:
            continue

        positive_doc_id = str(positives.iloc[0]["document_id"])
        negative_doc_id = str(negatives.iloc[0]["document_id"])

        triplets.append({
            "query_id": query_id,
            "positive_doc_id": positive_doc_id,
            "negative_doc_id": negative_doc_id
        })

    return pd.DataFrame(triplets)