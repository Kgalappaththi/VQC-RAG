
import pickle
from pathlib import Path

import numpy as np

from src.quantum.trainer import (
    build_trainable_vqc,
    quantum_relevance_score,
)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "results" / "scientific_reports_revision"
MODEL_DIR = DATA_DIR / "models"

# Evaluation configuration
N_QUBITS = 4
DEPTH = 4
SEED = 42

# Use only the validation dataset.
VALIDATION_PATH = (
    DATA_DIR / f"validation_vqc_dataset_{N_QUBITS}q.npz"
)

VQC_MODEL_PATH = (
    MODEL_DIR / f"vqc_{N_QUBITS}q_d{DEPTH}_seed{SEED}.pkl"
)

MLP_MODEL_PATH = (
    MODEL_DIR / f"mlp_{N_QUBITS}components_seed{SEED}.pkl"
)


def load_validation_models():
    """Load validation pairs and the two trained models."""

    # Load the validation features.
    with np.load(VALIDATION_PATH) as data:
        pos_features = data["pos_features"]
        neg_features = data["neg_features"]
        query_ids = data["query_ids"]

    # Verify the expected validation dataset.
    assert pos_features.shape == neg_features.shape
    assert pos_features.shape[1] == N_QUBITS
    assert len(pos_features) == 50
    assert len(np.unique(query_ids)) == 50

    # Load the validation-selected VQC checkpoint.
    with open(VQC_MODEL_PATH, "rb") as file:
        vqc_model = pickle.load(file)

    # Load the trained MLP model.
    with open(MLP_MODEL_PATH, "rb") as file:
        mlp_model = pickle.load(file)

    # Verify model compatibility.
    assert vqc_model["weights"].shape == (
        DEPTH, N_QUBITS, 2
    )
    assert mlp_model["model"].n_features_in_ == N_QUBITS

    print("Validation pairs:", len(pos_features))
    print("Unique validation queries:", len(np.unique(query_ids)))
    print("VQC and MLP checkpoints loaded successfully.")

    return pos_features, neg_features, vqc_model, mlp_model


def evaluate_vqc_pairwise(pos_features, neg_features, vqc_model):
    """Evaluate VQC pairwise accuracy on validation queries."""

    weights = vqc_model["weights"]

    # Reconstruct the trained quantum circuit.
    circuit = build_trainable_vqc(
        n_qubits=N_QUBITS,
        depth=DEPTH,
    )

    correct_pairs = 0

    for pos_x, neg_x in zip(pos_features, neg_features):

        # Score both documents using the saved VQC weights.
        pos_score = float(
            quantum_relevance_score(circuit, pos_x, weights)
        )

        neg_score = float(
            quantum_relevance_score(circuit, neg_x, weights)
        )

        if pos_score > neg_score:
            correct_pairs += 1

    pairwise_accuracy = correct_pairs / len(pos_features)

    print("\nVQC validation results")
    print(f"Correctly ranked pairs: {correct_pairs}")
    print(f"Total validation pairs: {len(pos_features)}")
    print(f"VQC pairwise accuracy: {pairwise_accuracy:.6f}")

    return pairwise_accuracy


def compare_validation_models(
    pos_features,
    neg_features,
    vqc_model,
    mlp_model,
):
    """Compare VQC and MLP on identical validation pairs."""

    # Evaluate the saved VQC checkpoint.
    vqc_accuracy = evaluate_vqc_pairwise(
        pos_features,
        neg_features,
        vqc_model,
    )

    # Evaluate the saved MLP classifier.
    classifier = mlp_model["model"]

    pos_scores = classifier.predict_proba(pos_features)[:, 1]
    neg_scores = classifier.predict_proba(neg_features)[:, 1]

    mlp_correct = int(np.sum(pos_scores > neg_scores))
    mlp_accuracy = mlp_correct / len(pos_features)

    print("\nValidation pairwise comparison")
    print("-" * 45)
    print(f"VQC accuracy: {vqc_accuracy:.6f}")
    print(f"MLP accuracy: {mlp_accuracy:.6f}")
    print(f"Validation pairs: {len(pos_features)}")
    print("-" * 45)

    return {
        "vqc_pairwise_accuracy": vqc_accuracy,
        "mlp_pairwise_accuracy": mlp_accuracy,
        "n_validation_pairs": len(pos_features),
    }


def main():
    """Evaluate VQC and MLP on the validation partition."""

    print("Starting VQC–MLP validation comparison.")

    # Load the same validation pairs for both models.
    pos_features, neg_features, vqc_model, mlp_model = (
        load_validation_models()
    )

    # Calculate pairwise accuracy for both models.
    results = compare_validation_models(
        pos_features,
        neg_features,
        vqc_model,
        mlp_model,
    )

    print("\nValidation comparison completed successfully.")


if __name__ == "__main__":
    main()
