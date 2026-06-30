import sys
import pickle

import numpy as np

from src.quantum.trainer import (
    train_vqc_pairwise,
    save_vqc_model
)
from src.config.vqc_experiments import VQC_EXPERIMENTS


def main():
    if len(sys.argv) < 2:
        raise ValueError("Please provide experiment ID, e.g. E004_6q_d4")

    experiment_id = sys.argv[1]
    config = VQC_EXPERIMENTS[experiment_id]

    data = np.load(config["dataset_path"], allow_pickle=True)

    pos_features = data["pos_features"]
    neg_features = data["neg_features"]

    with open(config["preprocessor_path"], "rb") as f:
        preprocessor = pickle.load(f)

    print("Experiment:", experiment_id)
    print("Positive features:", pos_features.shape)
    print("Negative features:", neg_features.shape)

    weights, loss_history, training_metrics = train_vqc_pairwise(
        pos_features=pos_features,
        neg_features=neg_features,
        depth=config["depth"],
        n_qubits=config["n_qubits"],
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        margin=config["margin"],
        seed=config["seed"]
    )

    training_metrics["loss_history"] = loss_history
    training_metrics["experiment_id"] = experiment_id

    save_vqc_model(
        weights=weights,
        scaler=preprocessor["scaler"],
        pca=preprocessor["pca"],
        output_path=config["model_output"],
        config=config,
        training_metrics=training_metrics
    )

    print("Trained VQC model saved to:", config["model_output"])
    print("Initial loss:", training_metrics["initial_loss"])
    print("Final loss:", training_metrics["final_loss"])


if __name__ == "__main__":
    main()