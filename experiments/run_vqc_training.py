import pickle
import numpy as np

from src.quantum.trainer import (
    train_vqc_pairwise,
    save_vqc_model
)


def main():
    dataset_path = "results/vqc/vqc_training_dataset_4q.npz"
    preprocessor_path = "results/vqc/vqc_preprocessor_4q.pkl"
    output_model = "results/vqc/trained_vqc_4q_depth2.pkl"

    data = np.load(dataset_path, allow_pickle=True)

    pos_features = data["pos_features"]
    neg_features = data["neg_features"]

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    print("Positive features:", pos_features.shape)
    print("Negative features:", neg_features.shape)

    weights, loss_history, training_metrics = train_vqc_pairwise(
        pos_features=pos_features,
        neg_features=neg_features,
        depth=2,
        n_qubits=4,
        epochs=20,
        learning_rate=0.05,
        margin=0.1,
        seed=42
    )

    training_metrics["loss_history"] = loss_history

    save_vqc_model(
        weights=weights,
        scaler=preprocessor["scaler"],
        pca=preprocessor["pca"],
        output_path=output_model,
        config={
            "n_qubits": 4,
            "depth": 2,
            "encoding": "angle",
            "loss": "pairwise_margin",
            "dataset": dataset_path,
            "seed": 42
        },
        training_metrics=training_metrics
    )

    print("Trained VQC model saved to:", output_model)
    print("Initial loss:", training_metrics["initial_loss"])
    print("Final loss:", training_metrics["final_loss"])


if __name__ == "__main__":
    main()