import numpy as np
import pandas as pd
from pathlib import Path

from src.quantum.trainer import initialize_weights, save_vqc_model


def main():
    depth = 2
    n_qubits = 4
    seed = 42

    weights = initialize_weights(depth=depth, n_qubits=n_qubits, seed=seed)

    print("Initial VQC weights shape:", weights.shape)

    output_path = "results/models/vqc_initial_model.pkl"

    save_vqc_model(
        weights=weights,
        scaler=None,
        pca=None,
        output_path=output_path
    )

    print("Initial VQC model saved to:", output_path)


if __name__ == "__main__":
    main()