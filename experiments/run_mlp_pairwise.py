import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.mlp_pairwise import (
    build_pairwise_classification_dataset,
    train_mlp_classifier,
    evaluate_mlp_classifier,
    pairwise_accuracy_from_classifier
)


def main():
    dataset_path = "results/vqc/vqc_training_dataset_6q.npz"
    output_dir = Path("results/classical_controls")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=True)

    pos_features = data["pos_features"]
    neg_features = data["neg_features"]

    X, y = build_pairwise_classification_dataset(pos_features, neg_features)

    model = train_mlp_classifier(X, y, seed=42)

    metrics = evaluate_mlp_classifier(model, X, y)
    metrics["pairwise_accuracy"] = pairwise_accuracy_from_classifier(
        model,
        pos_features,
        neg_features
    )
    metrics["model"] = "MLP"
    metrics["dataset"] = dataset_path
    metrics["n_components"] = 6

    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "mlp_pairwise_6q_summary.csv"
    model_path = output_dir / "mlp_pairwise_6q.pkl"

    metrics_df.to_csv(metrics_path, index=False)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("MLP Pairwise Summary")
    print(metrics_df.to_string(index=False))
    print("Saved metrics to:", metrics_path)
    print("Saved model to:", model_path)


if __name__ == "__main__":
    main()