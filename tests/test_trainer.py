import os
import numpy as np

from src.quantum.trainer import (
    initialize_weights,
    save_vqc_model,
    load_vqc_model
)


def test_trainer_utilities():
    weights = initialize_weights(depth=2, n_qubits=4, seed=42)

    print("Weights shape:", weights.shape)

    assert weights.shape == (2, 4, 2)

    model_path = "results/models/test_vqc_model.pkl"

    save_vqc_model(
        weights=weights,
        scaler=None,
        pca=None,
        output_path=model_path
    )

    assert os.path.exists(model_path)

    loaded = load_vqc_model(model_path)

    assert "weights" in loaded
    assert np.allclose(weights, loaded["weights"])

    print("Trainer tests passed.")


if __name__ == "__main__":
    test_trainer_utilities()