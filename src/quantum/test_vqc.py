from pennylane import numpy as np
from src.quantum.vqc import quantum_score


def test_vqc_score():
    np.random.seed(42)

    x = np.array([0.1, 0.2, 0.3, 0.4])
    weights = np.random.normal(0, 0.1, size=(2, 4, 2))

    score = quantum_score(x, weights)

    print("Quantum score:", score)

    assert 0.0 <= score <= 1.0
    print("VQC test passed.")


if __name__ == "__main__":
    test_vqc_score()