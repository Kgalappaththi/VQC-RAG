
import pickle
import time
from pathlib import Path

import numpy as np
import pennylane as qml

from src.quantum.trainer import (
    initialize_weights,
    build_trainable_vqc,
    quantum_relevance_score,
    pairwise_margin_loss,
    save_vqc_model,
)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "results" / "scientific_reports_revision"
MODEL_DIR = DATA_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets(n_qubits):
    """Load query-disjoint training and validation datasets."""

    train_path = DATA_DIR / f"train_vqc_dataset_{n_qubits}q.npz"
    validation_path = DATA_DIR / f"validation_vqc_dataset_{n_qubits}q.npz"

    train_data = np.load(train_path)
    validation_data = np.load(validation_path)

    train_pos = train_data["pos_features"]
    train_neg = train_data["neg_features"]

    val_pos = validation_data["pos_features"]
    val_neg = validation_data["neg_features"]

    assert train_pos.shape == train_neg.shape
    assert val_pos.shape == val_neg.shape

    assert train_pos.shape[1] == n_qubits
    assert val_pos.shape[1] == n_qubits

    # Ensure training and validation queries do not overlap.
    train_ids = set(train_data["query_ids"].astype(str))
    val_ids = set(validation_data["query_ids"].astype(str))

    assert train_ids.isdisjoint(val_ids), (
        "Data leakage detected: overlapping query IDs."
    )

    print(f"Training pairs: {len(train_pos)}")
    print(f"Validation pairs: {len(val_pos)}")
    print("Training-validation query overlap: 0")

    return train_pos, train_neg, val_pos, val_neg


def calculate_pairwise_loss(
    circuit,
    pos_features,
    neg_features,
    weights,
    margin=0.1,
):
    """Calculate the average pairwise margin loss."""

    total_loss = 0.0

    for pos_x, neg_x in zip(pos_features, neg_features):
        loss = pairwise_margin_loss(
            circuit,
            pos_x,
            neg_x,
            weights,
            margin=margin,
        )
        total_loss += float(loss)

    return total_loss / len(pos_features)



def train_vqc_with_validation(
    n_qubits=4,
    depth=4,
    epochs=100,
    learning_rate=0.01,
    margin=0.1,
    seed=42,
):
    """Train a VQC using training data and validation-based selection."""

    start_time = time.time()

    # Load only training and validation datasets.
    train_pos, train_neg, val_pos, val_neg = load_datasets(n_qubits)

    # Initialise the original VQC architecture.
    weights = initialize_weights(depth, n_qubits, seed)
    circuit = build_trainable_vqc(n_qubits=n_qubits, depth=depth)

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = None

    def objective(current_weights):
        """Differentiable training loss used by the optimiser."""

        total_loss = 0.0

        for pos_x, neg_x in zip(train_pos, train_neg):
            total_loss += pairwise_margin_loss(
                circuit,
                pos_x,
                neg_x,
                current_weights,
                margin=margin,
            )

        return total_loss / len(train_pos)


    # Train using training data only.
    for epoch in range(1, epochs + 1):

        weights = optimizer.step(objective, weights)

        # Evaluate the updated weights after each epoch.
        train_loss = calculate_pairwise_loss(
            circuit, train_pos, train_neg, weights, margin
        )

        val_loss = calculate_pairwise_loss(
            circuit, val_pos, val_neg, weights, margin
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Select the checkpoint using validation loss only.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = np.array(
                weights, copy=True
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.6f} | "
            f"Validation loss: {val_loss:.6f}"
        )

    training_time = time.time() - start_time

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Training time: {training_time:.2f} seconds")

    return {
        "best_weights": best_weights,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "train_loss_history": train_loss_history,
        "validation_loss_history": val_loss_history,
        "training_time_seconds": training_time,
    }


def save_best_model(result, n_qubits, depth, seed):
    """Save the best validation-selected VQC checkpoint."""

    preprocessor_path = (
        DATA_DIR / f"training_preprocessor_{n_qubits}q.pkl"
    )

    with open(preprocessor_path, "rb") as file:
        preprocessor = pickle.load(file)

    model_path = (
        MODEL_DIR / f"vqc_{n_qubits}q_d{depth}_seed{seed}.pkl"
    )

    config = {
        "n_qubits": n_qubits,
        "depth": depth,
        "seed": seed,
        "selection_method": "minimum_validation_loss",
        "training_partition": "train",
        "validation_partition": "validation",
    }

    training_metrics = {
        "best_epoch": result["best_epoch"],
        "best_validation_loss": result["best_validation_loss"],
        "train_loss_history": result["train_loss_history"],
        "validation_loss_history": result["validation_loss_history"],
        "training_time_seconds": result["training_time_seconds"],
    }

    save_vqc_model(
        weights=result["best_weights"],
        scaler=preprocessor["scaler"],
        pca=preprocessor["pca"],
        output_path=model_path,
        config=config,
        training_metrics=training_metrics,
    )

    print(f"Best model saved to: {model_path}")

    return model_path


def main():
    """Run a validation-aware VQC training experiment."""

    # Full experiment: 4-qubit VQC, depth 4, seed 42.
    n_qubits = 4
    depth = 4
    epochs = 100
    learning_rate = 0.01
    margin = 0.1
    seed = 42

    print("Starting revised VQC training experiment.")
    print(f"Qubits: {n_qubits}")
    print(f"Circuit depth: {depth}")
    print(f"Epochs: {epochs}")
    print(f"Random seed: {seed}")

    result = train_vqc_with_validation(
        n_qubits=n_qubits,
        depth=depth,
        epochs=epochs,
        learning_rate=learning_rate,
        margin=margin,
        seed=seed,
    )

    save_best_model(
        result=result,
        n_qubits=n_qubits,
        depth=depth,
        seed=seed,
    )

    print("Revised VQC experiment completed successfully.")


if __name__ == "__main__":
    main()

