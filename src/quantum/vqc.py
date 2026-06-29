import pennylane as qml
from pennylane import numpy as np


def build_vqc(n_qubits: int = 4, depth: int = 2):
    """
    Build a Variational Quantum Circuit for query-document relevance scoring.

    Encoding:
        Angle encoding using RY gates.

    Ansatz:
        Repeated RY-RZ trainable rotations followed by linear CNOT entanglement.

    Measurement:
        Average Pauli-Z expectation over all qubits.
    """

    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")

    if depth <= 0:
        raise ValueError("depth must be positive.")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, weights):
        if len(x) != n_qubits:
            raise ValueError("Input feature length must equal number of qubits.")

        expected_shape = (depth, n_qubits, 2)
        if weights.shape != expected_shape:
            raise ValueError(f"weights must have shape {expected_shape}.")

        # Angle encoding
        for qubit in range(n_qubits):
            qml.RY(x[qubit], wires=qubit)

        # Variational layers
        for layer in range(depth):
            for qubit in range(n_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)

            # Linear entanglement
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]

    return circuit


def quantum_score(x, weights) -> float:
    """
    Convert VQC Pauli-Z expectation values into a normalized relevance score.

    Output range:
        0 <= score <= 1
    """

    x = np.array(x, dtype=float)
    weights = np.array(weights, dtype=float)

    n_qubits = len(x)
    depth = weights.shape[0]

    circuit = build_vqc(n_qubits=n_qubits, depth=depth)
    measurements = circuit(x, weights)

    avg_measurement = np.mean(np.array(measurements))

    # Pauli-Z expectation is in [-1, 1], so normalize to [0, 1]
    score = (avg_measurement + 1.0) / 2.0

    return float(score)