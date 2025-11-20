#!/usr/bin/env python3
"""
Generate publication-quality quantum circuit diagram using PennyLane's drawer.
This creates an actual circuit pathway visualization.
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


def create_qtcr_circuit_diagram():
    """Create the actual QTCR-Net quantum circuit and draw it."""

    n_qubits = 4
    n_layers = 3

    # Create quantum device
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(features, params):
        """
        QTCR-Net Quantum Reservoir Circuit
        - Angle embedding (RY with input features)
        - Variational layers (RY, RZ rotations)
        - Ring entanglement (CNOTs in ring topology)
        """
        for layer in range(n_layers):
            # Data encoding (angle embedding) - first layer and data reuploading
            if layer == 0:
                for i in range(n_qubits):
                    qml.RY(features[i], wires=i)

            # Variational rotations
            for i in range(n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)

            # Ring entanglement
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Create sample inputs
    features = np.random.randn(n_qubits)
    params = np.random.randn(n_layers, n_qubits, 2)

    # Draw the circuit
    fig, ax = qml.draw_mpl(circuit, style='black_white', expansion_strategy='device')(features, params)

    # Customize appearance
    fig.set_size_inches(16, 6)
    ax.set_title('QTCR-Net Quantum Reservoir Circuit\n(4 Qubits, 3 Layers, Ring Entanglement)',
                 fontsize=14, fontweight='bold', pad=20)

    # Save
    plt.tight_layout()
    plt.savefig('paper_figures/quantum_circuit_actual.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('paper_figures/quantum_circuit_actual.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: paper_figures/quantum_circuit_actual.png")
    print("Saved: paper_figures/quantum_circuit_actual.pdf")
    plt.close()


def create_circuit_with_labels():
    """Create annotated circuit showing the flow."""

    n_qubits = 4
    n_layers = 3

    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def single_layer_circuit(features, params):
        """Single layer for clearer visualization."""
        # Angle embedding
        for i in range(n_qubits):
            qml.RY(features[i], wires=i)

        # Variational
        for i in range(n_qubits):
            qml.RY(params[i, 0], wires=i)
            qml.RZ(params[i, 1], wires=i)

        # Ring entanglement
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    features = np.random.randn(n_qubits)
    params = np.random.randn(n_qubits, 2)

    # Draw single layer
    fig, ax = qml.draw_mpl(single_layer_circuit, style='black_white')(features, params)
    fig.set_size_inches(12, 5)
    ax.set_title('Single Layer Structure (×3 layers total)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('paper_figures/quantum_circuit_single_layer.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: paper_figures/quantum_circuit_single_layer.png")
    plt.close()


def create_full_pipeline_circuit():
    """Show the complete data flow through quantum processing."""

    n_qubits = 4
    n_layers = 3

    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def full_circuit(features, params):
        """Full circuit with barriers for visual separation."""

        for layer in range(n_layers):
            # Angle embedding
            for i in range(n_qubits):
                qml.RY(features[i], wires=i)

            qml.Barrier(wires=range(n_qubits))

            # Variational rotations
            for i in range(n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)

            qml.Barrier(wires=range(n_qubits))

            # Ring entanglement
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

            if layer < n_layers - 1:
                qml.Barrier(wires=range(n_qubits))

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    features = np.random.randn(n_qubits)
    params = np.random.randn(n_layers, n_qubits, 2)

    # Draw with barriers
    fig, ax = qml.draw_mpl(full_circuit, style='black_white', expansion_strategy='device')(features, params)
    fig.set_size_inches(20, 6)
    ax.set_title('QTCR-Net Quantum Circuit with Layer Separation\n(Barriers show: Encoding | Variational | Entanglement)',
                 fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('paper_figures/quantum_circuit_with_barriers.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('paper_figures/quantum_circuit_with_barriers.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: paper_figures/quantum_circuit_with_barriers.png")
    print("Saved: paper_figures/quantum_circuit_with_barriers.pdf")
    plt.close()


def create_text_circuit():
    """Create ASCII text representation for LaTeX."""

    n_qubits = 4
    n_layers = 3

    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(features, params):
        for layer in range(n_layers):
            if layer == 0:
                for i in range(n_qubits):
                    qml.RY(features[i], wires=i)

            for i in range(n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)

            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    features = np.random.randn(n_qubits)
    params = np.random.randn(n_layers, n_qubits, 2)

    # Text drawing
    print("\n" + "="*60)
    print("ASCII Circuit Representation (for LaTeX/documentation):")
    print("="*60)
    print(qml.draw(circuit)(features, params))
    print("="*60 + "\n")


if __name__ == '__main__':
    from pathlib import Path
    Path('paper_figures').mkdir(exist_ok=True)

    print("Generating quantum circuit diagrams...")
    print("-" * 50)

    # Generate all versions
    create_qtcr_circuit_diagram()
    create_circuit_with_labels()
    create_full_pipeline_circuit()
    create_text_circuit()

    print("\nDone! Check paper_figures/ for the circuit diagrams.")
