import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.quantum.evaluation import (
    compute_pairwise_scores,
    summarize_pairwise_validation
)
from src.config.vqc_experiments import VQC_EXPERIMENTS


def main():
    if len(sys.argv) < 2:
        raise ValueError("Please provide experiment ID, e.g. E004_6q_d4")

    experiment_id = sys.argv[1]
    config = VQC_EXPERIMENTS[experiment_id]

    output_dir = Path("results/vqc_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(config["dataset_path"], allow_pickle=True)

    pos_features = data["pos_features"]
    neg_features = data["neg_features"]

    with open(config["model_output"], "rb") as f:
        model = pickle.load(f)

    weights = model["weights"]

    pos_scores, neg_scores = compute_pairwise_scores(
        pos_features=pos_features,
        neg_features=neg_features,
        weights=weights,
        n_qubits=config["n_qubits"],
        depth=config["depth"]
    )

    summary = summarize_pairwise_validation(pos_scores, neg_scores)
    summary["experiment_id"] = experiment_id
    summary["n_qubits"] = config["n_qubits"]
    summary["depth"] = config["depth"]
    summary["epochs"] = config["epochs"]
    summary["learning_rate"] = config["learning_rate"]

    scores_df = pd.DataFrame({
        "query_id": data["query_ids"],
        "positive_doc_id": data["positive_doc_ids"],
        "negative_doc_id": data["negative_doc_ids"],
        "positive_score": pos_scores,
        "negative_score": neg_scores,
        "correct_pairwise_ranking": pos_scores > neg_scores
    })

    summary_df = pd.DataFrame([summary])

    scores_path = output_dir / f"{experiment_id}_pairwise_scores.csv"
    summary_path = output_dir / f"{experiment_id}_pairwise_summary.csv"

    scores_df.to_csv(scores_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("VQC Pairwise Validation Summary")
    print(summary_df)

    print("Saved scores to:", scores_path)
    print("Saved summary to:", summary_path)


if __name__ == "__main__":
    main()