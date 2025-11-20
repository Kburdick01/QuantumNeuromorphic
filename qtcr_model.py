#!/usr/bin/env python3
"""
QTCR-Net: Quantum Temporal Convolutional Reservoir Network
A novel fully convolutional quantum-hybrid architecture for DVS event camera classification.

Key innovations:
1. Fully convolutional spatio-temporal feature extraction (NO MLP encoders)
2. Temporal Convolutional Network (TCN) with multi-scale dilations
3. Multiple quantum temporal reservoirs operating on learned feature groups
4. Hybrid quantum-classical causal temporal modeling
5. Dual-head classification (waveform + voltage)

Architecture:
    Input [2, T, H, W] voxel grid
    → 3D Convolutional block (spatio-temporal feature extraction)
    → TCN stack with dilations [1, 2, 4, 8] (multi-scale temporal patterns)
    → Spatial refinement (2D convolutions)
    → Split into K groups → K quantum reservoirs (4-8 qubits each)
    → Concatenate quantum outputs
    → Convolutional classification head (1×1 conv + GAP)
    → Dual linear outputs (waveform, voltage)

Author: QTCR-Net Research Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import yaml

from quantum_blocks import MultiReservoirQuantumLayer


class TemporalConvBlock(nn.Module):
    """
    Temporal convolutional block with dilation for TCN.
    Processes temporal dimension with causal convolutions.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 dropout: float = 0.1,
                 causal: bool = True):
        """
        Initialize temporal conv block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Temporal kernel size
            dilation: Temporal dilation rate
            dropout: Dropout probability
            causal: Whether to use causal (no future) convolutions
        """
        super().__init__()

        self.causal = causal

        # Compute padding for causal or centered convolutions
        if causal:
            # Causal: pad only on the left (past)
            self.padding = (kernel_size - 1) * dilation
        else:
            # Centered: pad equally on both sides
            self.padding = ((kernel_size - 1) * dilation) // 2

        # 1D temporal convolution
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding if not causal else 0
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, channels, time]

        Returns:
            Output tensor [batch, channels, time]
        """
        if self.causal:
            # Manual causal padding
            x = F.pad(x, (self.padding, 0))

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out


class TemporalConvolutionalNetwork(nn.Module):
    """
    Temporal Convolutional Network (TCN) with residual connections.
    Multi-scale temporal feature extraction using dilated convolutions.
    """

    def __init__(self,
                 in_channels: int,
                 channels: List[int] = [32, 64, 128, 128],
                 kernel_size: int = 3,
                 dilations: List[int] = [1, 2, 4, 8],
                 dropout: float = 0.1,
                 causal: bool = True,
                 residual: bool = True):
        """
        Initialize TCN.

        Args:
            in_channels: Number of input channels
            channels: List of output channels for each block
            kernel_size: Temporal kernel size
            dilations: List of dilation rates for each block
            dropout: Dropout probability
            causal: Whether to use causal convolutions
            residual: Whether to use residual connections
        """
        super().__init__()

        self.residual = residual
        num_blocks = len(channels)

        if len(dilations) != num_blocks:
            raise ValueError(f"Length of dilations ({len(dilations)}) must match channels ({num_blocks})")

        self.blocks = nn.ModuleList()

        current_channels = in_channels
        for i, (out_ch, dilation) in enumerate(zip(channels, dilations)):
            block = TemporalConvBlock(
                in_channels=current_channels,
                out_channels=out_ch,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                causal=causal
            )
            self.blocks.append(block)

            # Residual connection (1x1 conv for channel matching)
            if residual and current_channels != out_ch:
                self.blocks.append(
                    nn.Conv1d(current_channels, out_ch, kernel_size=1)
                )
            elif residual:
                self.blocks.append(nn.Identity())

            current_channels = out_ch

        self.output_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, channels, time]

        Returns:
            Output tensor [batch, channels, time]
        """
        out = x

        # Process blocks with residual connections
        for i in range(0, len(self.blocks), 2):
            block = self.blocks[i]
            residual_proj = self.blocks[i + 1] if i + 1 < len(self.blocks) else None

            residual = out
            out = block(out)

            # Add residual connection
            if self.residual and residual_proj is not None:
                residual = residual_proj(residual)
                # Match temporal dimension if needed
                if out.shape[2] != residual.shape[2]:
                    residual = residual[:, :, :out.shape[2]]
                out = out + residual

        return out


class SpatialRefinementBlock(nn.Module):
    """
    2D spatial convolution block for refining spatial features.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        """
        Initialize spatial refinement block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Spatial kernel size
            padding: Spatial padding
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, channels, height, width]

        Returns:
            Output [batch, channels, height, width]
        """
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class QTCRNet(nn.Module):
    """
    QTCR-Net: Quantum Temporal Convolutional Reservoir Network

    Novel architecture combining:
    - Fully convolutional spatio-temporal feature extraction
    - Temporal Convolutional Networks (TCN)
    - Multiple quantum temporal reservoirs
    - Dual-head classification
    """

    def __init__(self, config: dict):
        """
        Initialize QTCR-Net.

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        self.config = config
        model_config = config['model']
        data_config = config['data']

        # Input shape: [batch, 2, T, H, W]
        self.input_polarities = data_config['sensor']['polarities']
        self.input_temporal_bins = data_config['window']['temporal_bins']

        # === Initial Spatio-Temporal Feature Extraction ===

        if model_config['feature_extractor']['initial_3d']['enabled']:
            # 3D convolution for joint spatio-temporal features
            self.initial_conv = nn.Sequential(
                nn.Conv3d(
                    self.input_polarities,
                    model_config['feature_extractor']['initial_3d']['out_channels'],
                    kernel_size=tuple(model_config['feature_extractor']['initial_3d']['kernel_size']),
                    padding=tuple(model_config['feature_extractor']['initial_3d']['padding'])
                ),
                nn.BatchNorm3d(model_config['feature_extractor']['initial_3d']['out_channels']),
                nn.ReLU(),
                nn.Dropout3d(0.1)
            )
            initial_channels = model_config['feature_extractor']['initial_3d']['out_channels']

        else:
            # Separable convolution: temporal first, then spatial
            sep_config = model_config['feature_extractor']['separable']
            self.initial_conv = nn.Sequential(
                # Temporal conv (treat as [B*H*W, C, T])
                nn.Conv1d(
                    self.input_polarities,
                    sep_config['temporal_channels'],
                    kernel_size=sep_config['temporal_kernel'],
                    padding=sep_config['temporal_kernel'] // 2
                ),
                nn.BatchNorm1d(sep_config['temporal_channels']),
                nn.ReLU(),

                # Will need reshape for spatial conv
            )
            initial_channels = sep_config['spatial_channels']

        # === Temporal Convolutional Network (TCN) ===

        tcn_config = model_config['feature_extractor']['tcn']
        self.tcn_enabled = tcn_config['num_blocks'] > 0

        if self.tcn_enabled:
            self.tcn = TemporalConvolutionalNetwork(
                in_channels=initial_channels,
                channels=tcn_config['channels'],
                kernel_size=tcn_config['kernel_size'],
                dilations=tcn_config['dilations'],
                dropout=tcn_config['dropout'],
                causal=tcn_config['causal'],
                residual=tcn_config['residual']
            )
            tcn_output_channels = tcn_config['channels'][-1]
        else:
            self.tcn = None
            tcn_output_channels = initial_channels

        # === Spatial Refinement ===

        spatial_config = model_config['feature_extractor']['spatial_refine']
        if spatial_config['enabled']:
            spatial_blocks = []
            spatial_ch = [tcn_output_channels] + spatial_config['channels']

            for i in range(len(spatial_config['channels'])):
                spatial_blocks.append(
                    SpatialRefinementBlock(
                        in_channels=spatial_ch[i],
                        out_channels=spatial_ch[i + 1],
                        kernel_size=spatial_config['kernel_size'],
                        padding=spatial_config['padding']
                    )
                )

            self.spatial_refine = nn.Sequential(*spatial_blocks)
            spatial_output_channels = spatial_config['channels'][-1]
        else:
            self.spatial_refine = None
            spatial_output_channels = tcn_output_channels

        # === Quantum Temporal Reservoir ===

        qr_config = model_config['quantum_reservoir']

        # Check if quantum should be bypassed
        self.bypass_quantum = qr_config.get('bypass', False)

        if not self.bypass_quantum:
            # QMLP-style: compress → quantum → expand (not a bottleneck)
            n_qubits = qr_config.get('n_qubits', 8)
            n_layers = qr_config['circuit']['num_layers']

            self.quantum_compress = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(spatial_output_channels, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_qubits),  # Compress to quantum size
                nn.Tanh()  # Scale to [-1, 1] for angle encoding
            )

            # Single quantum reservoir for the compressed features
            from quantum_blocks import QuantumTemporalReservoir
            self.quantum_circuit = QuantumTemporalReservoir(
                num_qubits=n_qubits,
                feature_dim=n_qubits,
                num_layers=n_layers,
                entanglement=qr_config['circuit']['entanglement'],
                data_reuploading=qr_config['circuit']['data_reuploading'],
                trainable=qr_config['trainable_quantum'],
                trainable_ratio=qr_config['trainable_subset'],
                diff_method=qr_config['diff_method'],
                device_name=qr_config['backend'],
                reservoir_id=0
            )

            # Expand back from quantum
            self.quantum_expand = nn.Sequential(
                nn.Linear(n_qubits, 64),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            quantum_output_dim = 64
            print(f"[QTCR-Net] QMLP-style quantum: compress to {n_qubits} qubits, expand to 64")
        else:
            # Classical bypass: global pool + linear
            print("[QTCR-Net] Quantum layer BYPASSED - using classical path")
            self.quantum_compress = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(spatial_output_channels, 128),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.quantum_circuit = None
            self.quantum_expand = None
            quantum_output_dim = 128

        # === Convolutional Classification Head ===

        classifier_config = model_config['classifier']

        # Reshape quantum features back to spatial map
        # [B, quantum_dim] -> [B, quantum_dim, 1, 1]
        # Then apply 1x1 conv

        self.pre_classifier = nn.Sequential(
            nn.Linear(quantum_output_dim, classifier_config['conv1x1_channels']),
            nn.ReLU(),
            nn.Dropout(classifier_config['dropout'])
        )

        # Final dual heads
        self.waveform_head = nn.Sequential(
            nn.Linear(classifier_config['conv1x1_channels'], classifier_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(classifier_config['dropout']),
            nn.Linear(classifier_config['hidden_dim'], classifier_config['waveform_classes'])
        )

        self.voltage_head = nn.Sequential(
            nn.Linear(classifier_config['conv1x1_channels'], classifier_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(classifier_config['dropout']),
            nn.Linear(classifier_config['hidden_dim'], classifier_config['voltage_classes'])
        )

        # Count parameters
        self._print_model_info()

    def _print_model_info(self):
        """Print model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n[QTCR-Net] Model initialized")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Quantum layer info
        quantum_params = sum(p.numel() for p in self.quantum_compress.parameters())
        if self.quantum_circuit is not None:
            quantum_params += sum(p.numel() for p in self.quantum_circuit.parameters())
            quantum_params += sum(p.numel() for p in self.quantum_expand.parameters())

        quantum_trainable = sum(p.numel() for p in self.quantum_compress.parameters() if p.requires_grad)
        if self.quantum_circuit is not None:
            quantum_trainable += sum(p.numel() for p in self.quantum_circuit.parameters() if p.requires_grad)
            quantum_trainable += sum(p.numel() for p in self.quantum_expand.parameters() if p.requires_grad)

        print(f"  Quantum layer parameters: {quantum_params:,}")
        print(f"  Quantum trainable: {quantum_trainable:,}")
        print()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through QTCR-Net.

        Args:
            x: Input voxel tensor [batch, 2, T, H, W]

        Returns:
            Tuple of (waveform_logits, voltage_logits)
        """
        batch_size = x.shape[0]

        # === Initial spatio-temporal feature extraction ===
        # [B, 2, T, H, W] -> [B, C, T, H, W]
        features = self.initial_conv(x)

        # === Temporal processing with TCN ===
        if self.tcn_enabled:
            # TCN operates on temporal dimension
            # Reshape: [B, C, T, H, W] -> [B*H*W, C, T]
            B, C, T, H, W = features.shape
            features_temporal = features.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, C, T]
            features_temporal = features_temporal.view(B * H * W, C, T)

            # Apply TCN
            features_temporal = self.tcn(features_temporal)  # [B*H*W, C', T]

            # Reshape back: [B*H*W, C', T] -> [B, C', T, H, W]
            C_new = features_temporal.shape[1]
            features = features_temporal.view(B, H, W, C_new, T)
            features = features.permute(0, 3, 4, 1, 2).contiguous()  # [B, C', T, H, W]

        # === Spatial refinement ===
        if self.spatial_refine is not None:
            # Apply spatial conv to each time step
            # [B, C, T, H, W] -> [B*T, C, H, W] -> [B*T, C', H, W] -> [B, C', T, H, W]
            B, C, T, H, W = features.shape
            features_spatial = features.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
            features_spatial = features_spatial.view(B * T, C, H, W)

            features_spatial = self.spatial_refine(features_spatial)  # [B*T, C', H, W]

            C_new = features_spatial.shape[1]
            features = features_spatial.view(B, T, C_new, H, W)
            features = features.permute(0, 2, 1, 3, 4).contiguous()  # [B, C', T, H, W]

        # === Quantum temporal reservoir ===
        compressed = self.quantum_compress(features)  # [B, n_qubits] or [B, 128]

        if self.quantum_circuit is not None:
            # Pass through quantum circuit
            quantum_out = self.quantum_circuit(compressed)  # [B, n_qubits]
            quantum_features = self.quantum_expand(quantum_out)  # [B, 64]
        else:
            # Bypass mode - compressed already has 128 dims
            quantum_features = compressed

        # === Classification ===
        pre_class_features = self.pre_classifier(quantum_features)  # [B, hidden_dim]

        waveform_logits = self.waveform_head(pre_class_features)  # [B, num_waveform_classes]
        voltage_logits = self.voltage_head(pre_class_features)  # [B, num_voltage_classes]

        return waveform_logits, voltage_logits

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for visualization.

        Args:
            x: Input tensor

        Returns:
            Dictionary of feature maps at different stages
        """
        features_dict = {}

        # Initial features
        features = self.initial_conv(x)
        features_dict['initial'] = features

        # TCN features
        if self.tcn_enabled:
            B, C, T, H, W = features.shape
            features_temporal = features.permute(0, 3, 4, 1, 2).contiguous()
            features_temporal = features_temporal.view(B * H * W, C, T)
            features_temporal = self.tcn(features_temporal)
            C_new = features_temporal.shape[1]
            features = features_temporal.view(B, H, W, C_new, T)
            features = features.permute(0, 3, 4, 1, 2).contiguous()
            features_dict['tcn'] = features

        # Spatial features
        if self.spatial_refine is not None:
            B, C, T, H, W = features.shape
            features_spatial = features.permute(0, 2, 1, 3, 4).contiguous()
            features_spatial = features_spatial.view(B * T, C, H, W)
            features_spatial = self.spatial_refine(features_spatial)
            C_new = features_spatial.shape[1]
            features = features_spatial.view(B, T, C_new, H, W)
            features = features.permute(0, 2, 1, 3, 4).contiguous()
            features_dict['spatial'] = features

        # Quantum features
        quantum_features = self.quantum_layer(features)
        features_dict['quantum'] = quantum_features

        return features_dict


def test_qtcr_model():
    """Test QTCR-Net model."""
    print("[Testing QTCR-Net]\n")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create model
    model = QTCRNet(config)

    # Test input
    batch_size = 2
    polarities = 2
    T = config['data']['window']['temporal_bins']
    H = config['data']['sensor']['height'] // config['data']['spatial']['patch_size']
    W = config['data']['sensor']['width'] // config['data']['spatial']['patch_size']

    x = torch.randn(batch_size, polarities, T, H, W)
    print(f"Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        waveform_logits, voltage_logits = model(x)

    print(f"\nWaveform logits shape: {waveform_logits.shape}")
    print(f"Voltage logits shape: {voltage_logits.shape}")

    print("\n[QTCR-Net Test] Complete!")


if __name__ == '__main__':
    test_qtcr_model()
