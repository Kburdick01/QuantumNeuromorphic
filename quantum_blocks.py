#!/usr/bin/env python3
"""
QTCR-Net Quantum Reservoir Blocks
Implements quantum temporal reservoirs using PennyLane for hybrid quantum-classical learning.

Key features:
- Multiple small QNodes (4-8 qubits each) instead of one large circuit
- Angle embedding with data re-uploading
- Randomized variational layers with entanglement
- Multiple expectation value measurements
- Frozen reservoir parameters (reservoir computing) with optional fine-tuning

Author: QTCR-Net Research Team
Date: 2025
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings

# Check if PennyLane-Lightning is available for faster simulation
try:
    import pennylane_lightning
    DEFAULT_DEVICE = "lightning.qubit"
except ImportError:
    DEFAULT_DEVICE = "default.qubit"
    warnings.warn("pennylane-lightning not found, using default.qubit (slower)")


class QuantumTemporalReservoir(nn.Module):
    """
    Single quantum temporal reservoir implemented as a PennyLane QNode.

    Architecture:
    1. Angle embedding of input features
    2. Multiple variational layers with randomized parameters
    3. Entanglement (ring, chain, or all-to-all)
    4. Measurement of multiple observables (PauliZ, PauliX, PauliY, etc.)

    The reservoir acts as a high-dimensional nonlinear feature extractor.
    Parameters are typically frozen (random) for reservoir computing,
    but can be unfrozen for fine-tuning.
    """

    def __init__(self,
                 num_qubits: int = 6,
                 feature_dim: int = 8,
                 num_layers: int = 3,
                 entanglement: str = 'ring',
                 data_reuploading: bool = True,
                 trainable: bool = False,
                 trainable_ratio: float = 0.3,
                 diff_method: str = 'backprop',
                 device_name: str = DEFAULT_DEVICE,
                 reservoir_id: int = 0):
        """
        Initialize quantum temporal reservoir.

        Args:
            num_qubits: Number of qubits in the circuit
            feature_dim: Dimension of input features (will be projected to num_qubits)
            num_layers: Number of variational layers
            entanglement: Entanglement pattern ('ring', 'chain', 'all')
            data_reuploading: Whether to re-upload data at each layer
            trainable: Whether reservoir parameters are trainable
            trainable_ratio: If trainable, what fraction of parameters to train
            diff_method: PennyLane differentiation method
            device_name: PennyLane device name
            reservoir_id: Unique ID for this reservoir (for device naming)
        """
        super().__init__()

        self.num_qubits = num_qubits
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.entanglement = entanglement
        self.data_reuploading = data_reuploading
        self.trainable = trainable
        self.trainable_ratio = trainable_ratio
        self.diff_method = diff_method
        self.reservoir_id = reservoir_id

        # Create PennyLane device
        self.dev = qml.device(device_name, wires=num_qubits)

        # Feature projection: map feature_dim to num_qubits
        self.feature_projection = nn.Linear(feature_dim, num_qubits, bias=False)

        # Initialize quantum circuit parameters
        # Shape: [num_layers, num_qubits, 2] for RY and RZ rotations
        self.num_params = num_layers * num_qubits * 2

        # Initialize random parameters (reservoir)
        params_init = np.random.uniform(-np.pi, np.pi, size=(num_layers, num_qubits, 2))
        self.quantum_params = nn.Parameter(
            torch.tensor(params_init, dtype=torch.float32),
            requires_grad=trainable
        )

        # If partially trainable, freeze subset of parameters
        if trainable and trainable_ratio < 1.0:
            self._freeze_subset_params()

        # Create QNode
        self.qnode = qml.QNode(
            self._quantum_circuit,
            self.dev,
            interface='torch',
            diff_method=diff_method
        )

        # Number of measurements (one Z per qubit)
        self.num_measurements = num_qubits

        print(f"[QuantumReservoir {reservoir_id}] Initialized")
        print(f"  Qubits: {num_qubits}, Layers: {num_layers}, Entanglement: {entanglement}")
        print(f"  Trainable: {trainable}, Device: {device_name}")

    def _freeze_subset_params(self):
        """Freeze a subset of quantum parameters to avoid barren plateaus."""
        # Randomly select which parameters to keep trainable
        num_trainable = int(self.trainable_ratio * self.num_params)

        # Create mask for trainable parameters
        mask = torch.zeros(self.quantum_params.shape, dtype=torch.bool)
        flat_mask = mask.view(-1)

        # Randomly select trainable indices
        trainable_indices = torch.randperm(self.num_params)[:num_trainable]
        flat_mask[trainable_indices] = True
        mask = flat_mask.view(self.quantum_params.shape)

        # Store mask for use in forward pass
        self.register_buffer('trainable_mask', mask.float())

        print(f"  Partial training: {num_trainable}/{self.num_params} parameters trainable")

    def _apply_entanglement(self, pattern: str):
        """Apply entanglement pattern to qubits."""
        if pattern == 'ring':
            # Ring: each qubit connected to next, last to first
            for i in range(self.num_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

        elif pattern == 'chain':
            # Chain: each qubit connected to next (no wraparound)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        elif pattern == 'all':
            # All-to-all: each qubit connected to every other
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    qml.CNOT(wires=[i, j])

        else:
            raise ValueError(f"Unknown entanglement pattern: {pattern}")

    def _quantum_circuit(self, features, params):
        """
        Quantum circuit for temporal reservoir.

        Args:
            features: Input features [num_qubits]
            params: Quantum parameters [num_layers, num_qubits, 2]

        Returns:
            List of expectation values (one per qubit)
        """
        # DO NOT detach - PennyLane with interface='torch' needs the grad graph
        for layer in range(self.num_layers):
            # Data encoding (angle embedding)
            if layer == 0 or self.data_reuploading:
                for i in range(self.num_qubits):
                    qml.RY(features[i], wires=i)

            # Variational layer with randomized/learned rotations
            for i in range(self.num_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)

            # Entanglement
            self._apply_entanglement(self.entanglement)

        # Measurements: PauliZ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum reservoir.

        Args:
            x: Input tensor [batch_size, feature_dim]

        Returns:
            Output tensor [batch_size, num_measurements]
        """
        batch_size = x.shape[0]

        # Project features to qubit dimension
        x_proj = self.feature_projection(x)  # [batch_size, num_qubits]

        # Normalize features to [-pi, pi] for angle encoding
        x_proj = torch.tanh(x_proj) * np.pi

        # Apply quantum circuit to each sample in batch
        outputs = []
        for i in range(batch_size):
            # Get quantum parameters (apply mask if partially trainable)
            if self.trainable and hasattr(self, 'trainable_mask'):
                params = self.quantum_params * self.trainable_mask
            else:
                params = self.quantum_params

            # Run quantum circuit
            result = self.qnode(x_proj[i], params)

            # Handle result - preserve gradients for torch interface
            if isinstance(result, (list, tuple)):
                result = torch.stack([r if isinstance(r, torch.Tensor) else torch.as_tensor(r) for r in result])
            elif not isinstance(result, torch.Tensor):
                result = torch.as_tensor(result)

            # type_as preserves gradient chain (unlike .to() which breaks it)
            outputs.append(result.type_as(x))

        # Stack results
        outputs = torch.stack(outputs)  # [batch_size, num_measurements]

        return outputs


class MultiReservoirQuantumLayer(nn.Module):
    """
    Multiple quantum reservoirs operating in parallel on different feature groups.

    This is the core quantum component of QTCR-Net, splitting temporal features
    into K groups and processing each with a separate small quantum circuit.
    """

    def __init__(self,
                 input_channels: int,
                 num_groups: int = 6,
                 qubits_per_group: int = 6,
                 feature_dim_per_group: int = 8,
                 num_layers: int = 3,
                 entanglement: str = 'ring',
                 data_reuploading: bool = True,
                 trainable: bool = False,
                 trainable_ratio: float = 0.3,
                 diff_method: str = 'backprop',
                 device_name: str = DEFAULT_DEVICE):
        """
        Initialize multi-reservoir quantum layer.

        Args:
            input_channels: Number of input channels from convolutional features
            num_groups: Number of quantum reservoirs (K)
            qubits_per_group: Number of qubits per reservoir
            feature_dim_per_group: Feature dimension per group
            num_layers: Number of layers per quantum circuit
            entanglement: Entanglement pattern
            data_reuploading: Whether to re-upload data
            trainable: Whether reservoirs are trainable
            trainable_ratio: Fraction of trainable parameters
            diff_method: Differentiation method
            device_name: PennyLane device
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_groups = num_groups
        self.qubits_per_group = qubits_per_group
        self.feature_dim_per_group = feature_dim_per_group

        # Channels per group
        self.channels_per_group = input_channels // num_groups
        if input_channels % num_groups != 0:
            print(f"[WARNING] input_channels ({input_channels}) not divisible by num_groups ({num_groups})")
            print(f"  Using {self.channels_per_group * num_groups} channels")

        # Feature extraction from convolutional features to quantum inputs
        # For each group: [batch, channels_per_group, T, H, W] -> [batch, feature_dim_per_group]
        self.group_encoders = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d((4, 4, 4)),  # Less aggressive pooling
                nn.Flatten(),
                nn.Linear(self.channels_per_group * 64, feature_dim_per_group),  # 4*4*4=64
                nn.Tanh()
            )
            for _ in range(num_groups)
        ])

        # Create quantum reservoirs
        self.reservoirs = nn.ModuleList([
            QuantumTemporalReservoir(
                num_qubits=qubits_per_group,
                feature_dim=feature_dim_per_group,
                num_layers=num_layers,
                entanglement=entanglement,
                data_reuploading=data_reuploading,
                trainable=trainable,
                trainable_ratio=trainable_ratio,
                diff_method=diff_method,
                device_name=device_name,
                reservoir_id=i
            )
            for i in range(num_groups)
        ])

        # Total output dimension
        self.output_dim = num_groups * qubits_per_group

        print(f"[MultiReservoirQuantumLayer] Initialized")
        print(f"  Input channels: {input_channels}")
        print(f"  Num groups: {num_groups}, Qubits/group: {qubits_per_group}")
        print(f"  Total output dimension: {self.output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-reservoir quantum layer.

        Args:
            x: Input tensor [batch_size, channels, T, H, W]

        Returns:
            Quantum features [batch_size, num_groups * qubits_per_group]
        """
        batch_size = x.shape[0]

        # Split input into groups
        group_outputs = []
        for i in range(self.num_groups):
            start_ch = i * self.channels_per_group
            end_ch = start_ch + self.channels_per_group

            # Extract group features
            x_group = x[:, start_ch:end_ch, :, :, :]  # [batch, channels_per_group, T, H, W]

            # Encode to quantum input
            x_encoded = self.group_encoders[i](x_group)  # [batch, feature_dim_per_group]

            # Pass through quantum reservoir
            q_output = self.reservoirs[i](x_encoded)  # [batch, qubits_per_group]

            group_outputs.append(q_output)

        # Concatenate all quantum outputs
        quantum_features = torch.cat(group_outputs, dim=1)  # [batch, num_groups * qubits_per_group]

        return quantum_features


class QuantumReservoirClassifier(nn.Module):
    """
    Complete quantum reservoir classifier for testing.
    Simplified version without full QTCR-Net architecture.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int, int],
                 num_waveform_classes: int = 4,
                 num_voltage_classes: int = 4,
                 num_groups: int = 6,
                 qubits_per_group: int = 6):
        """
        Initialize quantum classifier.

        Args:
            input_shape: Input shape (C, T, H, W)
            num_waveform_classes: Number of waveform classes
            num_voltage_classes: Number of voltage classes
            num_groups: Number of quantum reservoirs
            qubits_per_group: Qubits per reservoir
        """
        super().__init__()

        C, T, H, W = input_shape

        # Simple feature extractor (will be replaced by full TCN in QTCR-Net)
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(C, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        # Quantum layer
        self.quantum_layer = MultiReservoirQuantumLayer(
            input_channels=64,
            num_groups=num_groups,
            qubits_per_group=qubits_per_group
        )

        # Classification heads
        quantum_output_dim = num_groups * qubits_per_group

        self.waveform_head = nn.Sequential(
            nn.Linear(quantum_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_waveform_classes)
        )

        self.voltage_head = nn.Sequential(
            nn.Linear(quantum_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_voltage_classes)
        )

    def forward(self, x):
        """Forward pass."""
        # Extract features
        features = self.feature_extractor(x)

        # Quantum processing
        quantum_features = self.quantum_layer(features)

        # Classification
        waveform_logits = self.waveform_head(quantum_features)
        voltage_logits = self.voltage_head(quantum_features)

        return waveform_logits, voltage_logits


def test_quantum_reservoir():
    """Test quantum reservoir implementation."""
    print("[Testing Quantum Reservoir]\n")

    # Create single reservoir
    reservoir = QuantumTemporalReservoir(
        num_qubits=6,
        feature_dim=8,
        num_layers=3,
        entanglement='ring',
        trainable=False
    )

    # Test forward pass
    batch_size = 4
    feature_dim = 8
    x = torch.randn(batch_size, feature_dim)

    print(f"Input shape: {x.shape}")

    output = reservoir(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test multi-reservoir layer
    print("\n[Testing Multi-Reservoir Layer]\n")

    multi_reservoir = MultiReservoirQuantumLayer(
        input_channels=64,
        num_groups=4,
        qubits_per_group=6,
        trainable=False
    )

    # Test with 3D conv features
    x_3d = torch.randn(batch_size, 64, 16, 8, 8)  # [B, C, T, H, W]
    print(f"Input shape: {x_3d.shape}")

    output_3d = multi_reservoir(x_3d)
    print(f"Output shape: {output_3d.shape}")
    print(f"Output range: [{output_3d.min():.3f}, {output_3d.max():.3f}]")

    print("\n[Quantum Reservoir Test] Complete!")


if __name__ == '__main__':
    test_quantum_reservoir()
