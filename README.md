# Q-TCRNet: Quantum Temporal Convolutional Reservoir Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.33+-green.svg)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Spatiotemporal Feature Expansion via Hybrid Quantum Temporal Convolutional Reservoirs

A novel quantum-hybrid architecture for neuromorphic vision that preserves the native spatiotemporal integrity of DVS event streams through causal temporal convolutions and quantum feature mapping in high-dimensional Hilbert space.

---

## Abstract

Neuromorphic vision sensors (DVS) offer microsecond temporal resolution, producing asynchronous event streams that encode motion dynamics in a sparse, continuous time-domain representation. We propose the **Quantum Temporal Convolutional Reservoir Network (Q-TCRNet)**, a hybrid quantum-classical architecture that:

- Processes voxel-based embeddings via TCN with causal, dilated kernels
- Maps classical features into high-dimensional Hilbert space via quantum temporal reservoirs
- Utilizes entanglement for non-linear kernel separation on complex motion waveforms

**Key Result**: 94.2% waveform accuracy, 88.7% voltage accuracy on DVS chopper-wheel dataset.

---

## Mathematical Framework

### 1. DVS Event Model

Each DVS pixel fires an event $e_k = (\mathbf{x}_k, t_k, p_k)$ when the brightness change reaches the contrast threshold $C$ [5]:

$$\Delta L(\mathbf{x}_k, t_k) = L(\mathbf{x}_k, t_k) - L(\mathbf{x}_k, t_k - \Delta t_k) = p_k C$$

where $L \doteq \log(I)$ is the log photocurrent (brightness), $C > 0$ is the contrast sensitivity threshold, and $p_k \in \{+1, -1\}$ is the polarity (ON/OFF).

### 2. Experimental Setup: Optical Chopper

A New Focus 3501 10-blade optical chopper wheel modulates light intensity as blades pass through the DVS field of view:

- **Hardware**: DVS128 camera (128×128, 1μs resolution), Agilent 33120A waveform generator
- **Control**: Waveform generator sync output → chopper sync input (TTL trigger on rising edge)
- **Voltage levels**: 200mV, 300mV, 400mV, 500mV (controls rotation speed)
- **Waveforms**: Sine, Square, Triangle, Burst

The different waveform shapes create distinct spatiotemporal event patterns that the network learns to classify.

---

## Spatiotemporal Voxelization

### Voxel Grid Representation

We preserve sub-frame temporal dynamics using dense voxel grids:

$$V \in \mathbb{R}^{C \times T \times H \times W}$$

where $C=2$ (polarities), $T$ (temporal bins), $H \times W$ (spatial pooling).

### Event Binning

Given event stream $\mathcal{E} = \{e_k\}_{k=1}^N$, we accumulate events into discrete voxel bins:

$$V(c,t,y,x) = \sum_{k} \mathbb{I}(p_k=c) \cdot \mathbb{I}(t_k \in \text{bin}_t) \cdot \mathbb{I}(x_k \in \text{bin}_x) \cdot \mathbb{I}(y_k \in \text{bin}_y)$$

Events are counted in spatial patches (8×8 pixels → 16×16 grid) and temporal bins (128 bins per window).

![Voxel Grid Visualization](paper_figures/voxel_grid_visualization.png)
*Figure 1: DVS voxel grid representation showing temporal evolution, spatial distribution, and polarity channels.*

### Normalization Pipeline

1. **Log-scale**: $V \leftarrow \log(1 + V + \epsilon)$ with $\epsilon = 10^{-6}$
2. **Standardization**: $V \leftarrow \frac{V - \mu_V}{\sigma_V + 10^{-8}}$
3. **Clipping**: $V \leftarrow \text{clip}(V, -10, 10)$

---

## Temporal Convolutional Network (TCN)

### Dilated Causal Convolution

For input sequence $x(t)$ and filter kernel $f: \{0, \ldots, K-1\} \rightarrow \mathbb{R}$:

$$(\mathbf{x} *_d f)(t) = \sum_{i=0}^{K-1} f(i) \cdot x(t - d \cdot i)$$

where $d$ is the dilation factor.

### Receptive Field Calculation

For $L$ layers with dilations $\{d_1, d_2, \ldots, d_L\}$ and kernel size $K$:

$$R = 1 + (K-1) \sum_{\ell=1}^{L} d_\ell$$

**Q-TCRNet Configuration** ($K=3$, $d \in \{1, 2, 4, 8\}$):

$$R = 1 + 2(1 + 2 + 4 + 8) = 31 \text{ timesteps}$$

### Residual TCN Block

$$h_\ell = h_{\ell-1} + \text{Dropout}\left(\text{ReLU}\left(\text{BN}\left(W_\ell *_{d_\ell} h_{\ell-1}\right)\right)\right)$$

Channel progression: $32 \rightarrow 64 \rightarrow 128 \rightarrow 128$

![Architecture Comparison](paper_figures/architecture_comparison.png)
*Figure 2: Q-TCRNet architecture showing 3D CNN encoder, TCN with dilated convolutions, and quantum reservoir.*

---

## Quantum Temporal Reservoir

### Feature Projection

Compress spatial-temporal features and project to quantum parameter space:

$$h_{\text{global}} = \text{AdaptiveAvgPool}(h) \in \mathbb{R}^{128}$$

$$\theta = \pi \cdot \tanh(W_{\text{proj}} h_{\text{global}} + b) \in [-\pi, \pi]^{N_q}$$

### Quantum Circuit Evolution

Initialize in ground state: $|\psi_0\rangle = |0\rangle^{\otimes N_q}$

Apply unitary evolution:

$$|\psi(\theta)\rangle = U_{\text{ent}}^{(L)} U_{\text{var}}^{(L)}(\phi^{(L)}) \cdots U_{\text{ent}}^{(1)} U_{\text{enc}}(\theta) |0\rangle^{\otimes N_q}$$

#### Angle Embedding Layer

$$U_{\text{enc}}(\theta) = \bigotimes_{j=1}^{N_q} R_y(\theta_j) = \bigotimes_{j=1}^{N_q} \exp\left(-i \frac{\theta_j}{2} \sigma_y\right)$$

#### Variational Layer

$$U_{\text{var}}^{(\ell)} = \prod_{j=1}^{N_q} R_y(\phi_j^{(\ell)}) R_z(\phi_j^{(\ell)})$$

#### Entanglement Layer (Ring Topology)

$$U_{\text{ent}} = \prod_{j=1}^{N_q} \text{CNOT}(j, (j+1) \mod N_q)$$

### Measurement and Observables

Measure Pauli-Z expectation value for each qubit:

$$m_j = \langle\psi(\theta)| \sigma_z^{(j)} |\psi(\theta)\rangle = \text{Tr}(\rho(\theta) \sigma_z^{(j)})$$

Output measurement vector:

$$\mathbf{m} = [m_1, m_2, \ldots, m_{N_q}]^T \in [-1, 1]^{N_q}$$

![Quantum Circuit](paper_figures/quantum_circuit_comparison.png)
*Figure 3: Quantum temporal reservoir circuit with RY/RZ rotations and ring entanglement topology.*

---

## Quantum Kernel Theory

### Implicit Kernel Function

The quantum circuit computes a kernel in $2^{N_q}$-dimensional Hilbert space:

$$K_Q(h, h') = |\langle\psi(\theta)|\psi(\theta')\rangle|^2$$

where $\theta = f(h)$ and $\theta' = f(h')$.

### Expressivity Bound

By the universal approximation theorem for quantum circuits, a parameterized circuit with $L$ layers can approximate any function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ with error $\|f - f_Q\| < \epsilon$ for sufficiently large $L$ and $N_q$.

Parameters scale as $P = O(N_q \cdot L)$ compared to $O(d^2)$ for classical neural networks.

---

## Quantum Feature Compression

To keep quantum circuits tractable, we compress classical features before quantum processing:

1. **Global pooling**: $h_{\text{global}} = \text{AdaptiveAvgPool3d}(h) \in \mathbb{R}^{128}$
2. **Compression**: $\mathbb{R}^{128} \rightarrow \mathbb{R}^{64} \rightarrow \mathbb{R}^{N_q}$ via linear layers
3. **Quantum circuit**: $N_q = 4$ qubits, $L = 3$ layers
4. **Expansion**: $\mathbb{R}^{N_q} \rightarrow \mathbb{R}^{64}$ for classification

### Computational Complexity

- **Classical**: $O(B \cdot L \cdot T \cdot C_{\text{in}} \cdot C_{\text{out}})$
- **Quantum**: $O(B \cdot 2^{N_q})$

For $B=24$, $N_q=4$:

$$O_{\text{quantum}} = 24 \cdot 2^4 = 384 \text{ operations/batch}$$

---

## Architecture Overview

```
Input: DVS voxel grid [B, 2, 128, 16, 16]
    |
[3D Convolution Block]
    Conv3d(2->32, kernel=3x3x3)
    BatchNorm3d + ReLU + Dropout(0.1)
    | [B, 32, 128, 16, 16]

[Temporal Convolutional Network]
    TCN Layer 1: Conv1d(32->64, dilation=1)
    TCN Layer 2: Conv1d(64->128, dilation=2)
    TCN Layer 3: Conv1d(128->128, dilation=4)
    TCN Layer 4: Conv1d(128->128, dilation=8)
    | [B, 128, 128, 16, 16]

[Quantum Compression]
    AdaptiveAvgPool3d -> Flatten
    Linear(128->64) + ReLU + Dropout(0.3)
    Linear(64->4) + Tanh
    | [B, 4]

[Quantum Temporal Reservoir]
    Nqubits = 4, Nlayers = 3
    Circuit: RY(theta) -> (RY, RZ) x 3 -> CNOT_ring x 3 -> <Z>
    | [B, 4]

[Classification Heads]
    Waveform: Linear(64->128)->Linear(128->4)
    Voltage: Linear(64->128)->Linear(128->4)
    |
Output: (waveform_logits, voltage_logits)
```

---

## Training Configuration

### Loss Function

$$\mathcal{L}_{\text{total}} = \lambda_w \mathcal{L}_{\text{CE}}(\hat{y}_w, y_w) + \lambda_v \mathcal{L}_{\text{CE}}(\hat{y}_v, y_v)$$

with label smoothing ($\alpha = 0.1$).

### Optimizer

Separate learning rates for classical and quantum parameters:
- Classical layers: $\eta_c = 10^{-3}$ (AdamW)
- Quantum layers: $\eta_q = 10^{-3}$
- Weight decay: $\lambda_{\text{decay}} = 10^{-2}$

### Gradient Clipping

$$\nabla \leftarrow \nabla \cdot \min\left(1, \frac{C}{\|\nabla\|_2}\right) \quad \text{with } C = 1.0$$

---

## Theoretical Analysis

### Why TCN Over LSTM?

LSTMs suffer from vanishing gradients:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

TCNs maintain constant gradient pathways through residual connections:

$$\frac{\partial \mathcal{L}}{\partial h_0} \propto I + \frac{\partial}{\partial h_0}\left[\sum_{\ell} f_\ell(h_\ell)\right]$$

### Quantum Dimensionality Advantage

**Classical reservoir**: $d_{\text{reservoir}} \approx 1000$ neurons

**Quantum reservoir**: Hilbert space dimension $2^{N_q}$
- $N_q = 10 \Rightarrow 2^{10} = 1{,}024$ dimensions
- $N_q = 20 \Rightarrow 2^{20} \approx 10^6$ dimensions

### Cover's Theorem

Probability of linear separability in $d$ dimensions:

$$P(d, n) = \frac{1}{2^n} \sum_{k=0}^{d-1} \binom{n-1}{k}$$

For $n=1000$ samples:
- $d=1000$ (classical): $P \approx 0.5$
- $d=2^{10}$ (quantum): $P \approx 1.0$

### Temporal Derivative Encoding

The helix curvature $\kappa$ encodes waveform information:

$$\kappa(t) = \frac{\|\mathbf{r}'(t) \times \mathbf{r}''(t)\|}{\|\mathbf{r}'(t)\|^3}$$

For Sine vs. Triangle: $\kappa_{\text{sine}}(t) \propto \omega^2 A$ vs. $\kappa_{\text{triangle}}(t) = 0$

---

## Quantum Advantage Analysis

### Entanglement Entropy

Von Neumann entropy of the reduced density matrix:

$$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$$

where $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$ traces out subsystem $B$.

For $N_q=4$ qubits with ring entanglement: $S_{\text{avg}} \approx 1.87$ bits

### Quantum Fisher Information

Quantifies parameter sensitivity:

$$F_Q[\rho(\theta)] = 4 \sum_i \frac{(\partial_\theta p_i)^2}{p_i}$$

where $p_i = \langle i|\rho|i\rangle$ are eigenvalues.

### Classical Simulation Cost

- **State vector**: $O(2^N)$ complex amplitudes
- **Gate application**: $O(2^N)$ per gate
- **Measurement**: $O(2^N)$ expectation value calculation

This motivates using small quantum circuits: a 4-qubit circuit ($2^4 = 16$ dimensions) remains tractable for classical simulation while providing nonlinear feature transformation.

---

## Experimental Results

### Dataset Statistics

- Total samples: 2,856 windows
- Train/Val/Test: 70% / 15% / 15%
- Window size: 1.0s (128 temporal bins)
- Spatial resolution: 16x16

### Performance Metrics

| Metric | Waveform | Voltage |
|--------|----------|---------|
| Accuracy | **94.2%** | **88.7%** |
| Precision | 0.943 | 0.891 |
| Recall | 0.942 | 0.887 |
| F1-Score | 0.942 | 0.888 |

### Ablation Studies

| Configuration | Waveform Acc | Voltage Acc |
|--------------|--------------|-------------|
| Q-TCRNet (Full) | **94.2%** | **88.7%** |
| Without Quantum | 89.3% | 84.1% |
| Without TCN | 82.7% | 79.5% |
| Frame-based CNN | 76.4% | 72.8% |
| MLP + FFT | 81.2% | 75.3% |

**Key Finding**: Quantum layer contributes **+4.9%** to waveform accuracy. TCN backbone provides **+11.5%** improvement.

![Frequency Spectrum Analysis](enhanced_analysis/comprehensive_300mV.png)
*Figure 4: Comprehensive waveform analysis showing spatiotemporal helix patterns, frequency spectra, and event rates for each waveform type.*

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (tested on RTX 5070)
- 64GB RAM recommended

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/qtcr-net.git
cd qtcr-net

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pennylane pennylane-lightning
pip install pandas numpy matplotlib seaborn tqdm pyyaml tensorboard scikit-learn
```

---

## Usage

### 1. Preprocess Data

```bash
python preprocess.py --config config.yaml
```

### 2. Train Model

```bash
python train.py --config config.yaml
```

### 3. Evaluate

```bash
python eval.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

### 4. Monitor Training

```bash
tensorboard --logdir runs/
```

---

## Hyperparameter Sensitivity

| Parameter | Values Tested | Best |
|-----------|--------------|------|
| Window Size | 0.5s, 1.0s, 2.0s | **1.0s** |
| Qubits ($N_q$) | 4, 6, 8 | **4-6** |
| TCN Dilations | [1,2,4,8], [1,2,4,8,16] | **[1,2,4,8]** |
| Quantum Layers | 2, 3, 4 | **3** |

---

## Hardware Configuration

**Tested Setup**:
- GPU: NVIDIA RTX 5070 (12GB VRAM)
- CPU: Intel i9-12900K (16 cores)
- RAM: 64GB DDR5
- OS: Linux Mint

**Training Time**: ~6 hours (200 epochs, batch size 24)

---

## Citation

```bibtex
@misc{qtcrnet2025,
  title={Q-TCRNet: Spatiotemporal Feature Expansion via Hybrid Quantum
         Temporal Convolutional Reservoirs},
  author={Q-TCRNet Research Team},
  year={2025},
  note={Hybrid quantum-classical architecture for DVS event camera classification},
  url={https://github.com/yourusername/qtcr-net}
}
```

---

## References

1. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. I. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226. [https://doi.org/10.22331/q-2020-02-06-226](https://doi.org/10.22331/q-2020-02-06-226)

2. Schuld, M., & Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces. *Physical Review Letters*, 122(4), 040504. [https://doi.org/10.1103/PhysRevLett.122.040504](https://doi.org/10.1103/PhysRevLett.122.040504)

3. Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567(7747), 209-212. [https://doi.org/10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)

4. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*. [https://arxiv.org/abs/1803.01271](https://arxiv.org/abs/1803.01271)

5. Gallego, G., et al. (2020). Event-based vision: A survey. *IEEE TPAMI*, 44(1), 154-180. [https://doi.org/10.1109/TPAMI.2020.3008413](https://doi.org/10.1109/TPAMI.2020.3008413)

6. McClean, J. R., et al. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9(1), 4812. [https://doi.org/10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)

7. Cover, T. M. (1965). Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. *IEEE Transactions on Electronic Computers*, (3), 326-334. [https://doi.org/10.1109/PGEC.1965.264137](https://doi.org/10.1109/PGEC.1965.264137)

8. Biamonte, J., et al. (2017). Quantum machine learning. *Nature*, 549(7671), 195-202. [https://doi.org/10.1038/nature23474](https://doi.org/10.1038/nature23474)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Future Work

- [ ] Hardware quantum deployment (IBM Quantum, IonQ)
- [ ] Larger DVS datasets (N-MNIST, DVS-Gesture)
- [ ] Quantum circuit optimization and error mitigation
- [ ] Theoretical analysis of quantum advantage bounds
- [ ] Real-time event processing pipeline
