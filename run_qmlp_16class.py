#!/usr/bin/env python3
"""
QMLP for 16-class DVS Classification
Based on user's MNIST QMLP with data re-uploading

Usage:
    python -u run_qmlp_16class.py
"""

import time
import json
import copy
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

NUM_CLASSES = 16
n_qubits = 16  # Match your MNIST code

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_class_names():
    waveforms = ['burst', 'sine', 'square', 'triangle']
    voltages = ['200mV', '300mV', '400mV', '500mV']
    names = []
    for w in waveforms:
        for v in voltages:
            names.append(f"{w}_{v}")
    return names

def generate_plots(train_losses, val_losses, train_accs, val_accs, cm, output_dir):
    output_dir = Path(output_dir)
    class_names = get_class_names()

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(epochs, [a*100 for a in train_accs], 'b-', label='Train')
    ax2.plot(epochs, [a*100 for a in val_accs], 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()

    ax3 = fig.add_subplot(2, 1, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax3,
                annot_kws={"size": 6})
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('16-Class Confusion Matrix')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'results.png', dpi=150)
    plt.close()


# Quantum device
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights_rot_1, weights_crx_1, weights_rot_2, weights_crx_2):
    """
    2-layer variational circuit with data re-uploading
    Similar to your MNIST QMLP structure
    """
    # Layer 1: Encode + Rot + CRX
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)

    for i in range(n_qubits):
        qml.Rot(weights_rot_1[i, 0], weights_rot_1[i, 1], weights_rot_1[i, 2], wires=i)

    for i in range(n_qubits):
        qml.CRX(weights_crx_1[i], wires=[i, (i+1) % n_qubits])

    # Layer 2: Re-encode + Rot + CRX
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)

    for i in range(n_qubits):
        qml.Rot(weights_rot_2[i, 0], weights_rot_2[i, 1], weights_rot_2[i, 2], wires=i)

    for i in range(n_qubits):
        qml.CRX(weights_crx_2[i], wires=[i, (i+1) % n_qubits])

    # Measure all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QMLP(nn.Module):
    """Quantum MLP for 16-class DVS classification"""
    def __init__(self, config):
        super().__init__()

        # Classical compression: DVS voxels -> 16 features for qubits
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 1))  # Similar to AvgPool in MNIST
        self.compress = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 4 * 4, n_qubits),  # 32 -> 16
        )

        # Quantum weights (like your MNIST code)
        # Layer 1
        self.weights_rot_1 = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)
        self.weights_crx_1 = nn.Parameter(torch.randn(n_qubits) * 0.1)
        # Layer 2
        self.weights_rot_2 = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)
        self.weights_crx_2 = nn.Parameter(torch.randn(n_qubits) * 0.1)

        # Classical head: 16 qubits -> 16 classes
        self.fc = nn.Linear(n_qubits, NUM_CLASSES)

    def forward(self, x):
        batch_size = x.shape[0]

        # Compress DVS voxels to 16 features
        x = self.pool(x)
        x = self.compress(x)  # [B, 16]

        # Process through quantum circuit
        q_outputs = []
        for i in range(batch_size):
            out = qnode(x[i], self.weights_rot_1, self.weights_crx_1,
                       self.weights_rot_2, self.weights_crx_2)
            q_outputs.append(torch.stack(out))

        q_out = torch.stack(q_outputs)  # [B, 16]

        # Classification
        out = self.fc(q_out)
        return F.log_softmax(out, dim=1)


def main():
    print("="*70)
    print(" QMLP Training - 16 Class DVS Classification")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Qubits: {n_qubits}, Classes: {NUM_CLASSES}")
    print("Circuit: 2-layer with data re-uploading\n")

    from dataset import create_dataloaders

    config = load_config()
    output_dir = Path('./experiments/qmlp_16class')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Create model
    model = QMLP(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    q_params = (model.weights_rot_1.numel() + model.weights_crx_1.numel() +
                model.weights_rot_2.numel() + model.weights_crx_2.numel())
    print(f"Total parameters: {total_params:,}")
    print(f"Quantum parameters: {q_params}")

    # Training setup (like your MNIST code)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    num_epochs = 30
    best_val_loss = float('inf')
    best_state = None
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"\nTraining {num_epochs} epochs...\n")
    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        model.train()
        loss_sum, correct, total = 0, 0, 0

        for batch_idx, (voxels, labels) in enumerate(train_loader):
            voxels = voxels.to(device)
            wave_labels = labels['waveform'].to(device)
            volt_labels = labels['voltage'].to(device)
            combined_labels = wave_labels * 4 + volt_labels

            optimizer.zero_grad()
            out = model(voxels)
            loss = F.nll_loss(out, combined_labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, predicted = torch.max(out.data, dim=1)
            total += combined_labels.size(0)
            correct += (predicted == combined_labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f'  epoch {epoch+1}, batch {batch_idx+1}, '
                      f'acc: {100*correct/total:.1f}%, loss: {loss_sum/(batch_idx+1):.3f}', flush=True)

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
                combined_labels = wave_labels * 4 + volt_labels

                out = model(voxels)
                loss = F.nll_loss(out, combined_labels)
                loss_sum += loss.item()
                _, predicted = torch.max(out.data, dim=1)
                total += combined_labels.size(0)
                correct += (predicted == combined_labels).sum().item()

        val_loss = loss_sum / len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(correct / total)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch+1}/{num_epochs}: Train Acc={train_accs[-1]*100:.1f}%, '
              f'Val Acc={val_accs[-1]*100:.1f}%', flush=True)

    training_time = time.time() - start_time

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for voxels, labels in test_loader:
            voxels = voxels.to(device)
            wave_labels = labels['waveform']
            volt_labels = labels['voltage']
            combined_labels = wave_labels * 4 + volt_labels

            out = model(voxels)
            _, predicted = torch.max(out.data, dim=1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(combined_labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*70}")
    print(" RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {acc*100:.1f}%")
    print(f"Precision: {prec*100:.1f}%")
    print(f"Recall: {rec*100:.1f}%")
    print(f"F1 Score: {f1*100:.1f}%")
    print(f"Training time: {training_time/60:.1f} minutes")

    # Save
    torch.save({
        'model_state_dict': best_state,
        'accuracy': acc
    }, output_dir / 'model.pth')

    metrics = {
        'model': 'QMLP',
        'n_qubits': n_qubits,
        'n_layers': 2,
        'num_classes': NUM_CLASSES,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'training_time_min': training_time/60,
        'epochs': num_epochs,
        'quantum_params': q_params
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        'epoch': range(1, num_epochs+1),
        'train_loss': train_losses, 'val_loss': val_losses,
        'train_acc': train_accs, 'val_acc': val_accs
    }).to_csv(output_dir / 'curves.csv', index=False)

    generate_plots(train_losses, val_losses, train_accs, val_accs, cm, output_dir)

    print(f"\nResults saved to {output_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
