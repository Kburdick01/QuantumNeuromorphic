#!/usr/bin/env python3
"""
Fast Ablation Studies - 16 Class Combined Task
Combines waveform (4) x voltage (4) = 16 classes

Usage:
    python -u run_ablations_16class.py
"""

import os
import sys
import copy
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

NUM_CLASSES = 16  # 4 waveforms x 4 voltages

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_combined_label(waveform_idx, voltage_idx):
    """Combine waveform and voltage into single 16-class label"""
    return waveform_idx * 4 + voltage_idx

def get_class_names():
    """Get 16 combined class names"""
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

    # Loss curves
    ax1 = fig.add_subplot(2, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()

    # Accuracy curves
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(epochs, [a*100 for a in train_accs], 'b-', label='Train')
    ax2.plot(epochs, [a*100 for a in val_accs], 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()

    # Confusion matrix (16x16)
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


# ============================================================================
# ABLATION MODELS - 16 CLASS OUTPUT
# ============================================================================

class ClassicalTCNOnly(nn.Module):
    """Without Quantum - Classical TCN only"""
    def __init__(self, config):
        super().__init__()
        self.cnn3d = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Dropout3d(0.3)
        )
        self.tcn = nn.Sequential(
            nn.Conv1d(1024, 128, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.head = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        b = x.size(0)
        x = self.cnn3d(x)
        x = x.view(b, -1, x.size(2))
        x = self.tcn(x)
        x = x.mean(dim=2)
        return self.head(x)


class CNN3DNoTCN(nn.Module):
    """Without TCN - 3D CNN only"""
    def __init__(self, config):
        super().__init__()
        self.cnn3d = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Dropout3d(0.3)
        )
        self.head = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn3d(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


class FrameBasedCNN(nn.Module):
    """Frame-based 2D CNN (no temporal)"""
    def __init__(self, config):
        super().__init__()
        self.cnn2d = None
        self.head = None

    def _build(self, in_channels):
        self.cnn2d = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout2d(0.3)
        )
        self.head = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        b, c, t, h, w = x.shape
        in_channels = c * t
        if self.cnn2d is None:
            self._build(in_channels)
            self.cnn2d = self.cnn2d.to(x.device)
            self.head = self.head.to(x.device)
        x = x.view(b, in_channels, h, w)
        x = self.cnn2d(x)
        x = x.view(b, -1)
        return self.head(x)


class MLPFFT(nn.Module):
    """MLP + FFT baseline"""
    def __init__(self, config):
        super().__init__()
        self.mlp = None
        self.head = None

    def _build(self, input_dim, device):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.3),
        ).to(device)
        self.head = nn.Linear(128, NUM_CLASSES).to(device)

    def forward(self, x):
        b = x.size(0)
        spatial = x.mean(dim=(2,3,4))
        temporal = x.mean(dim=(1,3,4))
        fft = torch.fft.rfft(temporal, dim=1)
        fft_feat = torch.abs(fft)
        feat = torch.cat([spatial, temporal, fft_feat], dim=1)

        if self.mlp is None:
            self._build(feat.shape[1], x.device)

        x = self.mlp(feat)
        return self.head(x)


# ============================================================================
# TRAINING
# ============================================================================

def train_ablation(model, model_name, config, output_dir, num_epochs=50):
    from dataset import create_dataloaders

    start = time.time()
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Dummy forward pass for lazy models
    with torch.no_grad():
        sample_batch = next(iter(train_loader))[0].to(device)
        model(sample_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float('inf')
    best_state = None
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    train_noise_std = 0.1
    test_noise_std = 0.05

    print(f"\n[{model_name}] Training {num_epochs} epochs (16-class)", flush=True)

    for epoch in range(num_epochs):
        # Train
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for voxels, labels in train_loader:
            voxels = voxels.to(device)
            voxels = voxels + torch.randn_like(voxels) * train_noise_std

            # Combine labels into 16 classes
            wave_labels = labels['waveform'].to(device)
            volt_labels = labels['voltage'].to(device)
            combined_labels = wave_labels * 4 + volt_labels

            optimizer.zero_grad()
            out = model(voxels)
            loss = criterion(out, combined_labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct += (out.argmax(1) == combined_labels).sum().item()
            total += combined_labels.size(0)

        train_losses.append(loss_sum / len(train_loader))
        train_accs.append(correct / total)

        # Validate
        model.eval()
        loss_sum, correct, total = 0, 0, 0
        with torch.no_grad():
            for voxels, labels in val_loader:
                voxels = voxels.to(device)
                voxels = voxels + torch.randn_like(voxels) * test_noise_std
                wave_labels = labels['waveform'].to(device)
                volt_labels = labels['voltage'].to(device)
                combined_labels = wave_labels * 4 + volt_labels

                out = model(voxels)
                loss = criterion(out, combined_labels)
                loss_sum += loss.item()
                correct += (out.argmax(1) == combined_labels).sum().item()
                total += combined_labels.size(0)

        val_loss = loss_sum / len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(correct / total)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={train_losses[-1]:.3f}/{val_losses[-1]:.3f}, "
                  f"Acc={train_accs[-1]*100:.1f}%/{val_accs[-1]*100:.1f}%", flush=True)

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for voxels, labels in test_loader:
            voxels = voxels.to(device)
            voxels = voxels + torch.randn_like(voxels) * test_noise_std
            wave_labels = labels['waveform']
            volt_labels = labels['voltage']
            combined_labels = wave_labels * 4 + volt_labels

            out = model(voxels)
            all_preds.append(out.argmax(1).cpu().numpy())
            all_labels.append(combined_labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    duration = time.time() - start

    print(f"  Results: Acc={acc*100:.1f}% (P={prec*100:.1f}%, R={rec*100:.1f}%, F1={f1*100:.1f}%), "
          f"Time={duration/60:.1f}min", flush=True)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'model': model_name,
        'num_classes': NUM_CLASSES,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'training_time_min': duration/60,
        'epochs': num_epochs
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        'epoch': range(1, num_epochs+1),
        'train_loss': train_losses, 'val_loss': val_losses,
        'train_acc': train_accs, 'val_acc': val_accs
    }).to_csv(output_dir / 'curves.csv', index=False)

    generate_plots(train_losses, val_losses, train_accs, val_accs, cm, output_dir)

    return {
        'model': model_name,
        'acc': acc, 'f1': f1,
        'time_min': duration/60
    }


def main():
    print("="*70)
    print(" FAST ABLATION STUDIES - 16 CLASS COMBINED TASK")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Classes: waveform (4) x voltage (4) = 16 combined\n")

    config = load_config()
    base_dir = Path('./experiments/ablation_16class')

    ablations = [
        ('Classical TCN (No Quantum)', ClassicalTCNOnly, 'classical_tcn'),
        ('3D CNN (No TCN)', CNN3DNoTCN, 'cnn3d_only'),
        ('Frame-based 2D CNN', FrameBasedCNN, 'frame_cnn'),
        ('MLP + FFT Baseline', MLPFFT, 'mlp_fft'),
    ]

    results = []
    total_start = time.time()

    for name, model_class, folder in ablations:
        print(f"\n{'='*70}")
        print(f" {name}")
        print(f"{'='*70}")

        model = model_class(config)
        result = train_ablation(model, name, config, base_dir / folder, num_epochs=50)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(" 16-CLASS ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'Accuracy':>10} {'F1':>10} {'Time':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['model']:<35} {r['acc']*100:>9.1f}% {r['f1']*100:>9.1f}% {r['time_min']:>7.1f}m")

    total_time = (time.time() - total_start) / 60
    print(f"\nTotal time: {total_time:.1f} minutes")

    # Save summary
    pd.DataFrame(results).to_csv(base_dir / 'summary.csv', index=False)
    with open(base_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {base_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
