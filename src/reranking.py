import numpy as np
from src.quantum.vqc import quantum_score

def hybrid_quantum_score(classical_score, reduced_feature, weights, lambda_value=0.5):
    q_score = quantum_score(reduced_feature, weights)
    h_score = lambda_value * classical_score + (1 - lambda_value) * q_score
    return float(h_score)

def rerank_candidates(candidates_df, reduced_features, weights, lambda_value=0.5):
    scores = []

    for i, row in candidates_df.iterrows():
        classical_score = row["hybrid_score"]
        q_feature = reduced_features[i]
        score = hybrid_quantum_score(
            classical_score,
            q_feature,
            weights,
            lambda_value
        )
        scores.append(score)

    candidates_df = candidates_df.copy()
    candidates_df["vqc_hybrid_score"] = scores

    return candidates_df.sort_values("vqc_hybrid_score", ascending=False)