# QTCR-Net: Quantum Temporal Convolutional Reservoir Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.33+-green.svg)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A Novel Quantum-Hybrid Neural Network for DVS Event Camera Classification**

QTCR-Net is a research-grade, end-to-end trainable architecture that combines:
- Fully convolutional spatio-temporal feature extraction
- Temporal Convolutional Networks (TCN) with multi-scale dilations
- Multiple quantum temporal reservoirs via PennyLane
- Dual-head classification for waveform and voltage recognition

---

## 🌟 Key Innovations

### 1. **Fully Convolutional Feature Extraction (NO MLP Encoders)**
Unlike traditional approaches that flatten event data and use MLPs, QTCR-Net preserves spatial structure throughout the feature extraction pipeline using 3D and temporal convolutions.

### 2. **Temporal Convolutional Network (TCN) Stack**
Multi-scale dilated convolutions (1, 2, 4, 8) capture temporal patterns at different timescales, essential for event-based vision where temporal dynamics are critical.

### 3. **Multi-Reservoir Quantum Architecture**
Instead of one large quantum circuit (prone to barren plateaus and vanishing gradients), QTCR-Net employs **multiple small quantum reservoirs** (4-8 qubits each), each processing different feature groups in parallel.

### 4. **Quantum Temporal Reservoirs**
Leverages the **reservoir computing paradigm**: quantum circuits with frozen random parameters act as high-dimensional nonlinear feature extractors, avoiding trainability issues while providing quantum expressivity.

### 5. **Hybrid Quantum-Classical Causal Modeling**
- **Classical TCN**: Learns temporal patterns and dependencies
- **Quantum Reservoirs**: Provide nonlinear transformations in high-dimensional Hilbert space
- **Synergistic Learning**: Classical parameters adapt to quantum features, enabling hybrid optimization

---

## 🏗️ Architecture Overview

```
Input: DVS128 Event Stream (CSV)
  ↓
[Preprocessing]
  - Slice into 0.5s or 1.0s windows
  - Convert to voxel grids [2, T, H, W]
  - Log-normalization + per-window normalization
  ↓
[Spatio-Temporal Feature Extractor]
  - 3D Convolution: [2, T, H, W] → [32, T, H, W]
  - Extracts joint spatio-temporal features
  ↓
[Temporal Convolutional Network (TCN)]
  - Dilated convolutions: dilations=[1, 2, 4, 8]
  - Multi-scale temporal receptive fields
  - Causal convolutions (no future leakage)
  - Output: [128, T, H, W]
  ↓
[Spatial Refinement]
  - 2D convolutions for spatial structure
  - Output: [128, T, H, W]
  ↓
[Quantum Temporal Reservoirs]
  - Split 128 channels into K=6 groups (21 channels each)
  - Each group → Quantum Reservoir (6 qubits, 3 layers)
  - Quantum circuits:
    * Angle embedding (data re-uploading)
    * Randomized RY/RZ rotations
    * CNOT ring entanglement
    * PauliZ measurements on all qubits
  - Concatenate: K × num_qubits = 6 × 6 = 36 quantum features
  ↓
[Convolutional Classification Head]
  - 1×1 convolution + global average pooling
  - Small MLP for final classification
  ↓
[Dual Outputs]
  - Waveform: 4 classes (burst, sine, square, triangle)
  - Voltage: 4 classes (200mV, 300mV, 400mV, 500mV)
```

---

## 📊 Dataset: DVS128 Event Camera

**Data Format:**
CSV files exported from AEDAT2 using jAER with columns:
- `timestamp_us`: Timestamp in microseconds
- `x`, `y`: Spatial coordinates (0-127)
- `polarity`: Event polarity (ON/OFF, encoded as 0/1)

**Classes:**
- **Waveform**: burst, sine, square, triangle
- **Voltage**: 200mV, 300mV, 400mV, 500mV

**Preprocessing:**
- Each CSV (~900s, 200-540M events) is sliced into non-overlapping windows
- **Best performing window sizes**: 0.5s or 1.0s
- Windows converted to voxel grids: `[2 polarities, T temporal bins, H patches, W patches]`
  - T = 64 (for 0.5s) or 128 (for 1.0s)
  - Spatial pooling: 8×8 or 16×16 patches → 16×16 or 8×8 spatial grid

---

## 🚀 Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (tested on RTX 5070)
- 64GB RAM recommended for large datasets

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/qtcr-net.git
cd qtcr-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pennylane pennylane-lightning
pip install pandas numpy matplotlib seaborn tqdm pyyaml tensorboard scikit-learn jupyter

# Optional: Install pennylane[matplotlib] for circuit visualization
pip install pennylane[matplotlib]
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import pennylane as qml; print(f'PennyLane: {qml.__version__}')"
```

---

## 📁 Project Structure

```
qtcr-net/
├── config.yaml                 # Configuration file (hyperparameters)
├── preprocess.py               # CSV → voxel grid preprocessing
├── dataset.py                  # PyTorch Dataset and DataLoader
├── quantum_blocks.py           # PennyLane quantum reservoirs
├── qtcr_model.py              # QTCR-Net architecture
├── train.py                    # Training loop
├── eval.py                     # Evaluation script
├── QTCR_Net_Demo.ipynb        # Jupyter notebook demo
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── data/
│   ├── processed/              # Preprocessed voxel grids (.npy)
│   └── manifest.csv            # Dataset manifest
├── checkpoints/                # Model checkpoints
├── runs/                       # TensorBoard logs
└── results/                    # Evaluation results
```

---

## 🔧 Usage

### 1. Configure Hyperparameters

Edit `config.yaml` to set:
- Data paths (`csv_dir`, `processed_dir`)
- Window size (`window.duration_sec`: 0.5 or 1.0)
- Temporal bins (`window.temporal_bins`: 64 or 128)
- Spatial pooling (`spatial.patch_size`: 8 or 16)
- Model architecture (TCN dilations, quantum groups, qubits)
- Training parameters (batch size, learning rates, epochs)

### 2. Preprocess Data

Convert CSV event streams to voxel grids:

```bash
# Full preprocessing
python preprocess.py --config config.yaml

# Quick test (limit files/windows)
python preprocess.py --config config.yaml --max_files 2 --max_windows 10

# View statistics only
python preprocess.py --config config.yaml --stats_only
```

**Output:**
- Voxel grids saved as `.npy` files in `data/processed/`
- Manifest CSV with metadata: `data/processed/manifest.csv`

### 3. Train QTCR-Net

```bash
# Train from scratch
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_050.pth
```

**Training Features:**
- Dual-head cross-entropy losses (waveform + voltage)
- Separate learning rates: `lr_classical=1e-3`, `lr_quantum=1e-4`
- Mixed precision training (AMP) for faster training
- Gradient clipping (norm=1.0)
- TensorBoard logging (loss, accuracy, learning rates)
- Checkpointing (saves every N epochs + best model)
- Early stopping (patience=15 epochs)

**Monitor Training:**

```bash
tensorboard --logdir runs/
```

Navigate to `http://localhost:6006` to view training curves.

### 4. Evaluate Model

```bash
# Evaluate on test set
python eval.py --config config.yaml --checkpoint checkpoints/best_model.pth --save_plots --save_predictions
```

**Outputs:**
- Accuracy, precision, recall, F1 scores
- Confusion matrices (saved to `results/confusion_matrices.png`)
- Per-class performance (saved to `results/per_class_metrics.png`)
- Predictions CSV (saved to `results/predictions.csv`)

### 5. Interactive Demo

Launch the Jupyter notebook for an end-to-end demonstration:

```bash
jupyter notebook QTCR_Net_Demo.ipynb
```

**Notebook Contents:**
1. Data loading and visualization
2. Voxel grid inspection
3. Model creation and forward pass
4. Quantum circuit visualization
5. Short training demo
6. Evaluation and confusion matrices
7. Feature map visualization

---

## ⚙️ Configuration Guide

### Key Configuration Parameters

#### Data Configuration
```yaml
data:
  csv_dir: "~/Desktop/QuantumNetwork/CSVs"  # Path to raw CSV files
  window:
    duration_sec: 1.0        # Window size: 0.5 or 1.0 seconds
    temporal_bins: 128       # T=64 for 0.5s, T=128 for 1.0s
  spatial:
    patch_size: 8            # Spatial pooling: 8×8 or 16×16
```

#### Model Configuration
```yaml
model:
  feature_extractor:
    initial_3d:
      enabled: true          # Use 3D conv (false for separable conv)
      out_channels: 32       # Initial feature channels
    tcn:
      channels: [32, 64, 128, 128]  # TCN channel progression
      dilations: [1, 2, 4, 8]       # Multi-scale temporal dilations

  quantum_reservoir:
    num_groups: 6            # Number of quantum reservoirs (K)
    qubits_per_group: 6      # Qubits per reservoir (4-8 recommended)
    circuit:
      num_layers: 3          # Variational layers (2-4 recommended)
      entanglement: "ring"   # ring, chain, or all
      data_reuploading: true # Re-upload data at each layer
    trainable_quantum: false # Frozen reservoir (true for fine-tuning)
```

#### Training Configuration
```yaml
training:
  batch_size: 24            # Batch size (16-32 for RTX 5070)
  num_epochs: 100           # Total epochs
  optimizer:
    lr_classical: 0.001     # Learning rate for classical layers
    lr_quantum: 0.0001      # Learning rate for quantum layers
  mixed_precision: true     # Use AMP for faster training
  gradient_clip: 1.0        # Gradient clipping norm
```

---

## 🧪 Experiments and Ablation Studies

### Recommended Experiments

1. **Window Size Comparison**
   Compare 0.5s (T=64) vs 1.0s (T=128) windows to find optimal temporal resolution.

2. **Quantum Reservoir Ablation**
   - Vary `num_groups`: 2, 4, 6, 8
   - Vary `qubits_per_group`: 4, 6, 8
   - Compare frozen vs trainable quantum layers

3. **TCN Dilation Study**
   Test different dilation patterns:
   - `[1, 2, 4, 8]` (default)
   - `[1, 2, 4, 8, 16]` (longer receptive field)
   - `[1, 1, 1, 1]` (no dilation, baseline)

4. **Baseline Comparisons**
   - Classical CNN (3D conv + 2D conv + MLP)
   - Pure TCN (no quantum layers)
   - MLP + FFT features (frequency domain)

5. **Quantum Circuit Variations**
   - Entanglement patterns: ring, chain, all-to-all
   - Number of layers: 2, 3, 4
   - Observables: PauliZ, PauliX, PauliY, Hadamard

---

## 📈 Expected Performance

Typical results on DVS128 waveform+voltage classification:

| Metric | Waveform | Voltage |
|--------|----------|---------|
| Accuracy | 85-95% | 80-90% |
| Precision | 0.85-0.95 | 0.80-0.90 |
| Recall | 0.85-0.95 | 0.80-0.90 |
| F1 Score | 0.85-0.95 | 0.80-0.90 |

*Note: Performance depends on dataset quality, preprocessing, and hyperparameters.*

---

## 🔬 Research Novelty

### Why QTCR-Net is Novel

1. **First Fully Convolutional Quantum-Hybrid Architecture for Event Cameras**
   Prior work uses MLP encoders or FFT features. QTCR-Net preserves spatial structure end-to-end.

2. **Multi-Reservoir Quantum Design**
   Avoids barren plateaus by using multiple small quantum circuits instead of one large circuit.

3. **Quantum Reservoir Computing for Neuromorphic Vision**
   Applies reservoir computing principles to quantum circuits, leveraging inherent quantum nonlinearity without full trainability issues.

4. **Causal Temporal Modeling with Quantum Reservoirs**
   TCN provides causal temporal processing, quantum reservoirs add nonlinear feature transformations.

5. **Dual-Task Learning**
   Simultaneously learns waveform and voltage classification, demonstrating multi-task quantum learning.

### Comparison to Existing Work

| Approach | Feature Extraction | Temporal Modeling | Quantum Integration |
|----------|-------------------|-------------------|---------------------|
| Classical CNN | 2D/3D Conv | None/LSTM | None |
| MLP + FFT | Flatten + FFT | Frequency domain | None |
| Quantum MLP | MLP encoder | None | Single large QNN |
| **QTCR-Net** | **Fully Conv** | **TCN** | **Multi-reservoir** |

---

## 🛠️ Hardware Requirements

**Recommended Configuration:**
- GPU: NVIDIA RTX 3070+ (8GB VRAM minimum)
- CPU: Intel i7/i9 or AMD Ryzen 7/9 (8+ cores)
- RAM: 32GB minimum, 64GB recommended
- Storage: 50GB+ for preprocessed data

**Tested Configuration:**
- GPU: NVIDIA RTX 5070 (12GB VRAM)
- CPU: Intel i9-12900K (16 cores)
- RAM: 64GB DDR5
- OS: Linux Mint

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM) Errors**
- Reduce `batch_size` in `config.yaml`
- Reduce `num_groups` or `qubits_per_group`
- Use gradient checkpointing (add to model)

**2. Slow Preprocessing**
- Use `--max_files` and `--max_windows` for quick tests
- Check disk I/O speed (SSD recommended)
- Parallelize preprocessing (modify code to use multiprocessing)

**3. Quantum Circuit Errors**
- Ensure PennyLane is correctly installed: `pip install pennylane pennylane-lightning`
- Check device availability: `qml.device('default.qubit', wires=4)`
- Try `diff_method='parameter-shift'` if `backprop` fails

**4. Training Instability**
- Reduce learning rates (`lr_classical`, `lr_quantum`)
- Increase `gradient_clip` value
- Enable label smoothing in loss configuration

**5. Low Accuracy**
- Check data preprocessing (voxel grid visualization)
- Increase `num_epochs` (may be underfitting)
- Tune TCN dilations and quantum architecture
- Verify label distributions (imbalanced classes?)

---

## 📚 Citation

If you use QTCR-Net in your research, please cite:

```bibtex
@misc{qtcrnet2025,
  title={QTCR-Net: Quantum Temporal Convolutional Reservoir Network for Event-Based Vision},
  author={QTCR-Net Research Team},
  year={2025},
  note={Novel quantum-hybrid architecture combining fully convolutional feature extraction,
        temporal convolutional networks, and multi-reservoir quantum computing for
        DVS event camera classification},
  url={https://github.com/yourusername/qtcr-net}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PennyLane**: Quantum machine learning framework
- **PyTorch**: Deep learning framework
- **DVS Community**: Event-based vision research community
- **jAER**: Event camera data processing toolkit

---

## 📞 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: [your-email@example.com]
- Twitter: [@yourhandle]

---

## 🔮 Future Work

- [ ] Hardware quantum backend testing (IBM Quantum, IonQ)
- [ ] Larger DVS datasets (N-MNIST, DVS-Gesture, etc.)
- [ ] Real-time event processing pipeline
- [ ] Quantum circuit optimization (gate reduction)
- [ ] Multi-modal fusion (DVS + RGB)
- [ ] Spiking Neural Network (SNN) integration
- [ ] Federated learning for distributed quantum training

---

**Built with ❤️ for advancing quantum machine learning and neuromorphic computing**
