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

    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    return dcg


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    ideal_hits = min(len(set(relevant_ids)), k)

    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
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

def mrr(all_retrieved, all_relevant):
    scores = []
    for retrieved_ids, relevant_ids in zip(all_retrieved, all_relevant):
        scores.append(reciprocal_rank(retrieved_ids, relevant_ids))
    return sum(scores) / len(scores)