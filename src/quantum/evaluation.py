import numpy as np

from src.quantum.trainer import build_trainable_vqc, quantum_relevance_score


def compute_pairwise_scores(pos_features, neg_features, weights, n_qubits=4, depth=2):
    circuit = build_trainable_vqc(n_qubits=n_qubits, depth=depth)

    pos_scores = []
    neg_scores = []

    for pos_x, neg_x in zip(pos_features, neg_features):
        pos_scores.append(float(quantum_relevance_score(circuit, pos_x, weights)))
        neg_scores.append(float(quantum_relevance_score(circuit, neg_x, weights)))

    return np.array(pos_scores), np.array(neg_scores)


def pairwise_accuracy(pos_scores, neg_scores):
    return float(np.mean(pos_scores > neg_scores))


def summarize_pairwise_validation(pos_scores, neg_scores):
    return {
        "mean_positive_score": float(np.mean(pos_scores)),
        "mean_negative_score": float(np.mean(neg_scores)),
        "std_positive_score": float(np.std(pos_scores)),
        "std_negative_score": float(np.std(neg_scores)),
        "pairwise_accuracy": pairwise_accuracy(pos_scores, neg_scores),
        "num_pairs": int(len(pos_scores)),
    }