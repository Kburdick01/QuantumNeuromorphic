#!/usr/bin/env python3
"""
QMLP for DVS Waveform Classification
Adapted from user's MNIST QMLP code

Usage:
    python -u run_qmlp.py
"""

import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import pennylane as qml

plt.style.use('seaborn-v0_8-whitegrid')

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def generate_plots(train_losses, val_losses, train_accs, val_accs,
                   wave_cm, volt_cm, output_dir):
    output_dir = Path(output_dir)
    wave_classes = ['burst', 'sine', 'square', 'triangle']
    volt_classes = ['200mV', '300mV', '400mV', '500mV']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(train_losses) + 1)

    axes[0,0].plot(epochs, train_losses, 'b-', label='Train')
    axes[0,0].plot(epochs, val_losses, 'r-', label='Val')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Loss Curves')
    axes[0,0].legend()

    axes[0,1].plot(epochs, [a*100 for a in train_accs], 'b-', label='Train')
    axes[0,1].plot(epochs, [a*100 for a in val_accs], 'r-', label='Val')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy (%)')
    axes[0,1].set_title('Accuracy Curves')
    axes[0,1].legend()

    sns.heatmap(wave_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wave_classes, yticklabels=wave_classes, ax=axes[1,0])
    axes[1,0].set_title('Waveform Confusion Matrix')

    sns.heatmap(volt_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=volt_classes, yticklabels=volt_classes, ax=axes[1,1])
    axes[1,1].set_title('Voltage Confusion Matrix')

    plt.tight_layout()
    plt.savefig(output_dir / 'results.png', dpi=150)
    plt.close()


# Quantum circuit
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Variational quantum circuit with data re-uploading
    inputs: [n_qubits] features
    weights: [n_layers, n_qubits, 3] rotation parameters
    """
    n_layers = weights.shape[0]

    for layer in range(n_layers):
        # Encode inputs
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)

        # Variational layer
        for i in range(n_qubits):
            qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

        # Entanglement (ring topology)
        for i in range(n_qubits):
            qml.CRX(weights[layer, i, 0], wires=[i, (i+1) % n_qubits])

    # Measure all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QMLP(nn.Module):
    """Quantum MLP for DVS classification"""
    def __init__(self, config, n_qubits=8, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical compression: DVS voxels -> n_qubits features
        self.compress = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),  # Reduce spatial-temporal
            nn.Flatten(),
            nn.Linear(2 * 4 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()  # Scale to [-1, 1] for quantum encoding
        )

        # Quantum weights
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        # Classical heads
        self.wave_head = nn.Linear(n_qubits, 4)
        self.volt_head = nn.Linear(n_qubits, 4)

    def forward(self, x):
        batch_size = x.shape[0]

        # Compress to quantum input size
        q_input = self.compress(x)  # [B, n_qubits]

        # Process each sample through quantum circuit
        q_outputs = []
        for i in range(batch_size):
            out = quantum_circuit(q_input[i] * np.pi, self.q_weights)
            q_outputs.append(torch.stack(out))

        q_out = torch.stack(q_outputs)  # [B, n_qubits]

        # Classification heads
        wave_out = self.wave_head(q_out)
        volt_out = self.volt_head(q_out)

        return wave_out, volt_out


def main():
    print("="*70)
    print(" QMLP Training - DVS Waveform Classification")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    from dataset import create_dataloaders

    config = load_config()
    output_dir = Path('./experiments/qmlp')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Qubits: {n_qubits}, Layers: 2\n")

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Create model
    model = QMLP(config, n_qubits=8, n_layers=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Quantum parameters: {model.q_weights.numel()}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    num_epochs = 30  # QMLP trains faster
    best_val_loss = float('inf')
    best_state = None
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"\nTraining {num_epochs} epochs...\n")
    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for voxels, labels in train_loader:
            voxels = voxels.to(device)
            wave_labels = labels['waveform'].to(device)
            volt_labels = labels['voltage'].to(device)

            optimizer.zero_grad()
            wave_out, volt_out = model(voxels)
            loss = criterion(wave_out, wave_labels) + criterion(volt_out, volt_labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct += (wave_out.argmax(1) == wave_labels).sum().item()
            total += wave_labels.size(0)

        train_losses.append(loss_sum / len(train_loader))
        train_accs.append(correct / total)

        # Validate
        model.eval()
        loss_sum, correct, total = 0, 0, 0
        with torch.no_grad():
            for voxels, labels in val_loader:
                voxels = voxels.to(device)
                wave_labels = labels['waveform'].to(device)
                volt_labels = labels['voltage'].to(device)
                wave_out, volt_out = model(voxels)
                loss = criterion(wave_out, wave_labels) + criterion(volt_out, volt_labels)
                loss_sum += loss.item()
                correct += (wave_out.argmax(1) == wave_labels).sum().item()
                total += wave_labels.size(0)

        val_loss = loss_sum / len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(correct / total)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_losses[-1]:.3f}/{val_losses[-1]:.3f}, "
                  f"Acc={train_accs[-1]*100:.1f}%/{val_accs[-1]*100:.1f}%", flush=True)

    training_time = time.time() - start_time

    # Test evaluation
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    wave_preds, wave_labels_all, volt_preds, volt_labels_all = [], [], [], []

    with torch.no_grad():
        for voxels, labels in test_loader:
            voxels = voxels.to(device)
            wave_out, volt_out = model(voxels)
            wave_preds.append(wave_out.argmax(1).cpu().numpy())
            wave_labels_all.append(labels['waveform'].numpy())
            volt_preds.append(volt_out.argmax(1).cpu().numpy())
            volt_labels_all.append(labels['voltage'].numpy())

    wave_preds = np.concatenate(wave_preds)
    wave_labels_all = np.concatenate(wave_labels_all)
    volt_preds = np.concatenate(volt_preds)
    volt_labels_all = np.concatenate(volt_labels_all)

    # Metrics
    wave_acc = accuracy_score(wave_labels_all, wave_preds)
    volt_acc = accuracy_score(volt_labels_all, volt_preds)
    wave_p, wave_r, wave_f1, _ = precision_recall_fscore_support(wave_labels_all, wave_preds, average='weighted')
    volt_p, volt_r, volt_f1, _ = precision_recall_fscore_support(volt_labels_all, volt_preds, average='weighted')
    wave_cm = confusion_matrix(wave_labels_all, wave_preds)
    volt_cm = confusion_matrix(volt_labels_all, volt_preds)

    print(f"\n{'='*70}")
    print(" RESULTS")
    print(f"{'='*70}")
    print(f"Waveform: Acc={wave_acc*100:.1f}%, P={wave_p*100:.1f}%, R={wave_r*100:.1f}%, F1={wave_f1*100:.1f}%")
    print(f"Voltage:  Acc={volt_acc*100:.1f}%, P={volt_p*100:.1f}%, R={volt_r*100:.1f}%, F1={volt_f1*100:.1f}%")
    print(f"Training time: {training_time/60:.1f} minutes")

    # Save
    torch.save({
        'model_state_dict': best_state,
        'wave_acc': wave_acc,
        'volt_acc': volt_acc
    }, output_dir / 'model.pth')

    metrics = {
        'model': 'QMLP',
        'n_qubits': n_qubits,
        'n_layers': 2,
        'waveform': {'accuracy': float(wave_acc), 'precision': float(wave_p),
                     'recall': float(wave_r), 'f1': float(wave_f1)},
        'voltage': {'accuracy': float(volt_acc), 'precision': float(volt_p),
                    'recall': float(volt_r), 'f1': float(volt_f1)},
        'training_time_min': training_time/60,
        'epochs': num_epochs
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        'epoch': range(1, num_epochs+1),
        'train_loss': train_losses, 'val_loss': val_losses,
        'train_acc': train_accs, 'val_acc': val_accs
    }).to_csv(output_dir / 'curves.csv', index=False)

    generate_plots(train_losses, val_losses, train_accs, val_accs, wave_cm, volt_cm, output_dir)

    print(f"\nResults saved to {output_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
