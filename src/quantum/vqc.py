import pennylane as qml
from pennylane import numpy as np

def create_vqc(n_qubits=4, depth=2):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Angle encoding
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers
        for layer in range(depth):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit

def quantum_score(inputs, weights):
    n_qubits = len(inputs)
    circuit = create_vqc(n_qubits=n_qubits, depth=weights.shape[0])
    measurements = circuit(inputs, weights)
    avg_measurement = np.mean(np.array(measurements))
    score = (avg_measurement + 1) / 2
    return score