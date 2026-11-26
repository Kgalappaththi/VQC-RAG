
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.baselines.mlp_pairwise import (
    build_pairwise_classification_dataset,
    evaluate_mlp_classifier,
    pairwise_accuracy_from_classifier,
)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "results" / "scientific_reports_revision"
MODEL_DIR = DATA_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_mlp_datasets(n_components=4):
    """Load query-disjoint training and validation data."""

    train_path = DATA_DIR / f"train_vqc_dataset_{n_components}q.npz"
    val_path = DATA_DIR / f"validation_vqc_dataset_{n_components}q.npz"

    with np.load(train_path) as train_data:
        train_pos = train_data["pos_features"]
        train_neg = train_data["neg_features"]
        train_ids = set(train_data["query_ids"].astype(str))

    with np.load(val_path) as val_data:
        val_pos = val_data["pos_features"]
        val_neg = val_data["neg_features"]
        val_ids = set(val_data["query_ids"].astype(str))

    # Verify feature dimensions and pair counts.
    assert train_pos.shape == train_neg.shape
    assert val_pos.shape == val_neg.shape

    assert train_pos.shape[1] == n_components
    assert val_pos.shape[1] == n_components

    # Ensure there is no query overlap.
    assert train_ids.isdisjoint(val_ids), (
        "Data leakage detected: training and validation queries overlap."
    )

    # Construct classification datasets.
    X_train, y_train = build_pairwise_classification_dataset(
        train_pos, train_neg
    )

    X_val, y_val = build_pairwise_classification_dataset(
        val_pos, val_neg
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Training-validation query overlap: 0")

    return X_train, y_train, X_val, y_val, val_pos, val_neg


def train_mlp_with_validation(n_components=4, seed=42):
    """Train the MLP and evaluate it on validation queries."""

    X_train, y_train, X_val, y_val, val_pos, val_neg = (
        load_mlp_datasets(n_components)
    )

    # Retain the original MLP architecture.
    model = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=seed,
    )

    start_time = time.time()

    # Fit the model using training data only.
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Evaluate on separate validation queries.
    val_metrics = evaluate_mlp_classifier(
        model, X_val, y_val
    )

    val_pairwise_accuracy = pairwise_accuracy_from_classifier(
        model, val_pos, val_neg
    )

    print("\nMLP training completed.")
    print(f"Training time: {training_time:.2f} seconds")
    print(
        f"Validation accuracy: "
        f"{val_metrics['classification_accuracy']:.6f}"
    )
    print(
        f"Validation ROC-AUC: "
        f"{val_metrics['roc_auc']:.6f}"
    )
    print(
        f"Validation pairwise accuracy: "
        f"{val_pairwise_accuracy:.6f}"
    )

    return {
        "model": model,
        "seed": seed,
        "n_components": n_components,
        "validation_metrics": val_metrics,
        "validation_pairwise_accuracy": val_pairwise_accuracy,
        "training_time_seconds": training_time,
    }


def save_mlp_model(result):
    """Save the MLP model and its training-fitted preprocessing."""

    n_components = result["n_components"]
    seed = result["seed"]

    preprocessor_path = (
        DATA_DIR / f"training_preprocessor_{n_components}q.pkl"
    )

    with open(preprocessor_path, "rb") as file:
        preprocessor = pickle.load(file)

    model_path = (
        MODEL_DIR / f"mlp_{n_components}components_seed{seed}.pkl"
    )

    model_package = {
        "model": result["model"],
        "scaler": preprocessor["scaler"],
        "pca": preprocessor["pca"],
        "n_components": n_components,
        "seed": seed,
        "validation_metrics": result["validation_metrics"],
        "validation_pairwise_accuracy": (
            result["validation_pairwise_accuracy"]
        ),
        "training_time_seconds": (
            result["training_time_seconds"]
        ),
        "training_partition": "train",
        "validation_partition": "validation",
    }

    with open(model_path, "wb") as file:
        pickle.dump(model_package, file)

    print(f"MLP model saved to: {model_path}")

    return model_path

def main():
    """Run the revised MLP baseline experiment."""

    n_components = 4
    seed = 42

    print("Starting revised MLP baseline experiment.")
    print(f"PCA components: {n_components}")
    print(f"Random seed: {seed}")

    result = train_mlp_with_validation(
        n_components=n_components,
        seed=seed,
    )

    save_mlp_model(result)

    print("Revised MLP experiment completed successfully.")


if __name__ == "__main__":
    main()
