from pennylane import numpy as np
from src.quantum.vqc import quantum_score

inputs = np.array([0.1, 0.2, 0.3, 0.4])
weights = np.random.normal(0, 0.1, size=(2, 4, 2))

score = quantum_score(inputs, weights)

print("Quantum score:", score)

assert 0 <= score <= 1