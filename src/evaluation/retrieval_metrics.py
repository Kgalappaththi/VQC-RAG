def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_at_k = retrieved_ids[:k]
    relevant_found = set(retrieved_at_k).intersection(set(relevant_ids))
    return len(relevant_found) / len(relevant_ids) if relevant_ids else 0

def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank
    return 0

def mrr(all_retrieved, all_relevant):
    scores = []
    for retrieved_ids, relevant_ids in zip(all_retrieved, all_relevant):
        scores.append(reciprocal_rank(retrieved_ids, relevant_ids))
    return sum(scores) / len(scores)