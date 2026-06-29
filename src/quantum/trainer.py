import pickle
import time
from pathlib import Path

import pennylane as qml
from pennylane import numpy as np


def initialize_weights(depth, n_qubits, seed=42):
    np.random.seed(seed)
    return np.random.normal(0, 0.1, size=(depth, n_qubits, 2), requires_grad=True)


def save_vqc_model(weights, scaler, pca, output_path, config=None, training_metrics=None):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump({
            "weights": np.array(weights),
            "scaler": scaler,
            "pca": pca,
            "config": config or {},
            "training_metrics": training_metrics or {}
        }, f)


def load_vqc_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def build_trainable_vqc(n_qubits=4, depth=2):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        for qubit in range(n_qubits):
            qml.RY(x[qubit], wires=qubit)

        for layer in range(depth):
            for qubit in range(n_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)

            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]

    return circuit


def quantum_relevance_score(circuit, x, weights):
    measurements = circuit(x, weights)
    avg_measurement = np.mean(np.stack(measurements))
    return (avg_measurement + 1.0) / 2.0


def pairwise_margin_loss(circuit, pos_x, neg_x, weights, margin=0.1):
    pos_score = quantum_relevance_score(circuit, pos_x, weights)
    neg_score = quantum_relevance_score(circuit, neg_x, weights)
    return np.maximum(0.0, margin - pos_score + neg_score)


def train_vqc_pairwise(
    pos_features,
    neg_features,
    depth=2,
    n_qubits=4,
    epochs=20,
    learning_rate=0.05,
    margin=0.1,
    seed=42,
):
    start_time = time.time()

    weights = initialize_weights(depth, n_qubits, seed)
    circuit = build_trainable_vqc(n_qubits=n_qubits, depth=depth)

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)

    loss_history = []

    def objective(current_weights):
        total_loss = 0.0

        for pos_x, neg_x in zip(pos_features, neg_features):
            total_loss = total_loss + pairwise_margin_loss(
                circuit,
                pos_x,
                neg_x,
                current_weights,
                margin=margin
            )

        return total_loss / len(pos_features)

    for epoch in range(epochs):
        weights, loss_value = optimizer.step_and_cost(objective, weights)
        loss_history.append(float(loss_value))

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {float(loss_value):.6f}")

    training_time = time.time() - start_time

    metrics = {
        "final_loss": float(loss_history[-1]),
        "initial_loss": float(loss_history[0]),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "margin": margin,
        "training_time_seconds": training_time
    }

    return weights, loss_history, metrics