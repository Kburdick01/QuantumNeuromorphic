#!/usr/bin/env python3
"""
Create publication-quality quantum circuit diagram using PennyLane's drawer.
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

def create_quantum_circuit_diagram(n_qubits=4, n_layers=3, save_path='paper_figures/quantum_circuit_real.png'):
    """
    Create a proper quantum circuit diagram using PennyLane's drawer.
    """

    # Create a quantum device
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def quantum_reservoir(inputs, weights):
        """
        The actual quantum reservoir circuit used in Q-TCRNet.
        """
        for layer in range(n_layers):
            # Data encoding (angle embedding with re-uploading)
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational rotations
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

            # Ring entanglement
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Create sample inputs and weights
    inputs = np.random.uniform(-np.pi, np.pi, n_qubits)
    weights = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 2))

    # Draw the circuit
    fig, ax = qml.draw_mpl(quantum_reservoir, style='black_white', expansion_strategy='device')(inputs, weights)

    # Customize appearance
    fig.set_size_inches(16, 6)
    ax.set_title(f'Q-TCRNet Quantum Temporal Reservoir\n({n_qubits} qubits, {n_layers} layers, ring entanglement)',
                 fontsize=14, fontweight='bold', pad=20)

    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

    return save_path


def create_simple_circuit_for_readme(save_path='paper_figures/quantum_circuit_simple.png'):
    """
    Create a simplified 1-layer circuit for README clarity.
    """
    n_qubits = 4
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def simple_circuit(inputs, weights):
        # One layer only for clarity
        # Data encoding
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational
        for i in range(n_qubits):
            qml.RY(weights[i, 0], wires=i)
            qml.RZ(weights[i, 1], wires=i)

        # Ring CNOT
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    inputs = np.random.uniform(-np.pi, np.pi, n_qubits)
    weights = np.random.uniform(-np.pi, np.pi, (n_qubits, 2))

    fig, ax = qml.draw_mpl(simple_circuit, style='black_white')(inputs, weights)
    fig.set_size_inches(12, 5)
    ax.set_title('Quantum Reservoir - Single Layer Structure\n(Data Encoding → Variational → Ring Entanglement)',
                 fontsize=12, fontweight='bold', pad=15)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

    return save_path


if __name__ == '__main__':
    from pathlib import Path
    Path('paper_figures').mkdir(exist_ok=True)

    print("Creating quantum circuit diagrams...")
    create_quantum_circuit_diagram()
    create_simple_circuit_for_readme()
    print("Done!")
