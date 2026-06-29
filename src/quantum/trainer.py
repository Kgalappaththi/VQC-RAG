import pickle
import numpy as np
from pathlib import Path


def initialize_weights(depth, n_qubits, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, size=(depth, n_qubits, 2))


def save_vqc_model(weights, scaler, pca, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump({
            "weights": weights,
            "scaler": scaler,
            "pca": pca
        }, f)


def load_vqc_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)