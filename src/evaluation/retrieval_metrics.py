import math


def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    if len(relevant_set) == 0:
        return 0.0

    hits = len(set(retrieved_k) & relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    dcg = 0.0

    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    return dcg


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)

    if len(relevant_set) == 0:
        return 0.0

    ideal_hits = min(len(relevant_set), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(retrieved_ids, relevant_ids, k) / ideal_dcg


def average_precision(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)

    if len(relevant_set) == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant_set)