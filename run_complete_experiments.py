#!/usr/bin/env python3
"""
Q-TCRNet Complete Retraining & Hyperparameter Sensitivity Study

This script:
1. Reprocesses all CSV data with proper train/val/test splits
2. Trains Q-TCRNet from scratch
3. Runs full hyperparameter sensitivity studies
4. Generates comprehensive results with timing

Expected runtime: 4-8 hours depending on GPU

Usage:
    python run_complete_experiments.py

Author: Q-TCRNet Research Team
"""

import os
import sys
import copy
import time
import json
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Global timing tracker
TIMING_LOG = []


def log_time(task_name, duration):
    """Log timing for a task."""
    TIMING_LOG.append({
        'task': task_name,
        'duration_sec': duration,
        'duration_min': duration / 60,
        'timestamp': datetime.now().isoformat()
    })
    print(f"  [TIME] {task_name}: {duration/60:.1f} min ({duration:.1f}s)")


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)


def print_section(text):
    """Print section header."""
    print(f"\n--- {text} ---")


def create_experiment_dirs():
    """Create organized directory structure for all experiments."""
    base_dir = Path('./experiments')

    dirs = {
        'base': base_dir,
        'preprocessing': base_dir / '01_preprocessing',
        'initial_training': base_dir / '02_initial_training',
        'hyperparameter_studies': base_dir / '03_hyperparameter_studies',
        'hp_window_size': base_dir / '03_hyperparameter_studies' / 'window_size',
        'hp_qubits': base_dir / '03_hyperparameter_studies' / 'n_qubits',
        'hp_dilations': base_dir / '03_hyperparameter_studies' / 'tcn_dilations',
        'hp_qlayers': base_dir / '03_hyperparameter_studies' / 'quantum_layers',
        'final_training': base_dir / '04_final_training',
        'ablation_studies': base_dir / '05_ablation_studies',
        'ablation_no_quantum': base_dir / '05_ablation_studies' / 'without_quantum',
        'ablation_no_tcn': base_dir / '05_ablation_studies' / 'without_tcn',
        'ablation_frame_cnn': base_dir / '05_ablation_studies' / 'frame_based_cnn',
        'ablation_mlp_fft': base_dir / '05_ablation_studies' / 'mlp_fft',
        'summary': base_dir / '06_summary',
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


# ==============================================================================
# STEP 1: REPROCESS DATA
# ==============================================================================

def reprocess_data(config, output_dir=None):
    """Reprocess all CSV data with fresh splits."""
    start_time = time.time()

    # Remove old processed data
    processed_dir = Path(config['data']['processed_dir'])
    if processed_dir.exists():
        print(f"Removing old processed data: {processed_dir}")
        shutil.rmtree(processed_dir)

    # Import and run preprocessor
    from preprocess import DVSPreprocessor

    preprocessor = DVSPreprocessor(config)
    preprocessor.process_all()

    duration = time.time() - start_time
    log_time("Data Preprocessing", duration)

    # Save preprocessing info
    if output_dir:
        info = {
            'duration_sec': duration,
            'processed_dir': str(processed_dir),
            'window_duration': config['data']['window']['duration_sec'],
            'temporal_bins': config['data']['window']['temporal_bins'],
            'timestamp': datetime.now().isoformat()
        }
        with open(output_dir / 'preprocessing_info.json', 'w') as f:
            json.dump(info, f, indent=2)

    return True


# ==============================================================================
# STEP 2: TRAINING UTILITIES
# ==============================================================================

def train_model_full(config, model_name="Q-TCRNet", num_epochs=200, output_dir=None):
    """Train a model with full logging and return metrics."""
    from qtcr_model import QTCRNet
    from dataset import create_dataloaders

    training_start = time.time()

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config, train_augment=False)

    # Create model
    model = QTCRNet(config).to(device)

    # Optimizers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr_classical'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    best_val_loss = float('inf')
    best_state = None
    train_losses = []
    val_losses = []
    epoch_times = []

    print(f"\n[Training {model_name}] {num_epochs} epochs")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0
        for voxels, labels in train_loader:
            voxels = voxels.to(device)
            wave_labels = labels['waveform'].to(device)
            volt_labels = labels['voltage'].to(device)

            optimizer.zero_grad()
            wave_logits, volt_logits = model(voxels)
            loss = criterion(wave_logits, wave_labels) + criterion(volt_logits, volt_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for voxels, labels in val_loader:
                voxels = voxels.to(device)
                wave_labels = labels['waveform'].to(device)
                volt_labels = labels['voltage'].to(device)

                wave_logits, volt_logits = model(voxels)
                loss = criterion(wave_logits, wave_labels) + criterion(volt_logits, volt_labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0:
            avg_time = np.mean(epoch_times[-20:])
            print(f"  Epoch {epoch+1}/{num_epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, Time={avg_time:.2f}s")

    # Load best model and evaluate
    model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    all_wave_preds, all_wave_labels = [], []
    all_volt_preds, all_volt_labels = [], []

    with torch.no_grad():
        for voxels, labels in test_loader:
            voxels = voxels.to(device)
            wave_logits, volt_logits = model(voxels)

            all_wave_preds.append(wave_logits.argmax(dim=1).cpu().numpy())
            all_wave_labels.append(labels['waveform'].numpy())
            all_volt_preds.append(volt_logits.argmax(dim=1).cpu().numpy())
            all_volt_labels.append(labels['voltage'].numpy())

    wave_preds = np.concatenate(all_wave_preds)
    wave_labels = np.concatenate(all_wave_labels)
    volt_preds = np.concatenate(all_volt_preds)
    volt_labels = np.concatenate(all_volt_labels)

    wave_acc = accuracy_score(wave_labels, wave_preds)
    volt_acc = accuracy_score(volt_labels, volt_preds)

    avg_epoch_time = np.mean(epoch_times)
    total_training_time = time.time() - training_start

    # Check for overfitting
    train_val_gap = train_losses[-1] - val_losses[-1]
    overfit_indicator = "OVERFIT" if train_val_gap < -0.5 else "OK"

    results = {
        'wave_acc': wave_acc,
        'volt_acc': volt_acc,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'avg_epoch_time': avg_epoch_time,
        'total_training_time': total_training_time,
        'overfit_status': overfit_indicator,
        'model_state': best_state,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    log_time(f"Training {model_name}", total_training_time)

    print(f"  Final: Wave={wave_acc*100:.1f}%, Volt={volt_acc*100:.1f}%, "
          f"Train/Val Gap={train_val_gap:.3f} [{overfit_indicator}]")

    # Save to output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save({
            'model_state_dict': best_state,
            'epoch': num_epochs,
            'wave_acc': wave_acc,
            'volt_acc': volt_acc
        }, output_dir / 'model.pth')

        # Save metrics
        metrics = {
            'model_name': model_name,
            'wave_acc': float(wave_acc),
            'volt_acc': float(volt_acc),
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1]),
            'avg_epoch_time': float(avg_epoch_time),
            'total_training_time_sec': float(total_training_time),
            'total_training_time_min': float(total_training_time / 60),
            'num_epochs': num_epochs,
            'overfit_status': overfit_indicator,
            'timestamp': datetime.now().isoformat()
        }
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save loss curves
        loss_df = pd.DataFrame({
            'epoch': range(1, num_epochs + 1),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        loss_df.to_csv(output_dir / 'loss_curves.csv', index=False)

    return results


# ==============================================================================
# STEP 3: HYPERPARAMETER SENSITIVITY STUDIES
# ==============================================================================

def run_hyperparameter_studies(base_config, dirs):
    """Run all hyperparameter sensitivity experiments."""
    print_header("STEP 3: HYPERPARAMETER SENSITIVITY STUDIES")

    results = []
    hp_start = time.time()

    # Study 1: Window Size
    print_section("Study 1: Window Size")
    window_sizes = [0.5, 1.0, 2.0]
    temporal_bins = [64, 128, 256]  # Corresponding bins

    for ws, tb in zip(window_sizes, temporal_bins):
        config = copy.deepcopy(base_config)
        config['data']['window']['duration_sec'] = ws
        config['data']['window']['temporal_bins'] = tb

        print(f"\n  Testing Window Size = {ws}s (T={tb})")

        # Create output dir for this experiment
        exp_dir = dirs['hp_window_size'] / f'window_{ws}s'

        # Reprocess with new window size
        reprocess_data(config, exp_dir)

        # Train
        res = train_model_full(config, f"Window-{ws}s", num_epochs=100, output_dir=exp_dir)
        results.append({
            'study': 'window_size',
            'value': f'{ws}s',
            'wave_acc': res['wave_acc'],
            'volt_acc': res['volt_acc'],
            'time_per_epoch': res['avg_epoch_time'],
            'total_time': res['total_training_time']
        })

    # Restore default window
    reprocess_data(base_config, dirs['preprocessing'])

    # Study 2: Number of Qubits
    print_section("Study 2: Number of Qubits")
    qubit_values = [4, 6, 8]

    for nq in qubit_values:
        config = copy.deepcopy(base_config)
        config['model']['quantum_reservoir']['n_qubits'] = nq

        exp_dir = dirs['hp_qubits'] / f'qubits_{nq}'

        print(f"\n  Testing N_qubits = {nq}")
        res = train_model_full(config, f"Qubits-{nq}", num_epochs=100, output_dir=exp_dir)
        results.append({
            'study': 'n_qubits',
            'value': str(nq),
            'wave_acc': res['wave_acc'],
            'volt_acc': res['volt_acc'],
            'time_per_epoch': res['avg_epoch_time'],
            'total_time': res['total_training_time']
        })

    # Study 3: TCN Dilations
    print_section("Study 3: TCN Dilations")
    dilation_configs = [
        ([1, 2, 4, 8], "1_2_4_8"),
        ([1, 2, 4, 8, 16], "1_2_4_8_16")
    ]

    for dilations, name in dilation_configs:
        config = copy.deepcopy(base_config)
        config['model']['feature_extractor']['tcn']['dilations'] = dilations
        config['model']['feature_extractor']['tcn']['num_blocks'] = len(dilations)

        # Adjust channels to match
        if len(dilations) == 4:
            config['model']['feature_extractor']['tcn']['channels'] = [32, 64, 128, 128]
        else:
            config['model']['feature_extractor']['tcn']['channels'] = [32, 64, 128, 128, 128]

        exp_dir = dirs['hp_dilations'] / f'dilations_{name}'

        print(f"\n  Testing Dilations = [{name.replace('_', ',')}]")
        res = train_model_full(config, f"Dilations-{name}", num_epochs=100, output_dir=exp_dir)
        results.append({
            'study': 'tcn_dilations',
            'value': f'[{name.replace("_", ",")}]',
            'wave_acc': res['wave_acc'],
            'volt_acc': res['volt_acc'],
            'time_per_epoch': res['avg_epoch_time'],
            'total_time': res['total_training_time']
        })

    # Study 4: Quantum Layers
    print_section("Study 4: Quantum Layers")
    layer_values = [2, 3, 4]

    for nl in layer_values:
        config = copy.deepcopy(base_config)
        config['model']['quantum_reservoir']['circuit']['num_layers'] = nl

        exp_dir = dirs['hp_qlayers'] / f'layers_{nl}'

        print(f"\n  Testing Quantum Layers = {nl}")
        res = train_model_full(config, f"QLayers-{nl}", num_epochs=100, output_dir=exp_dir)
        results.append({
            'study': 'quantum_layers',
            'value': str(nl),
            'wave_acc': res['wave_acc'],
            'volt_acc': res['volt_acc'],
            'time_per_epoch': res['avg_epoch_time'],
            'total_time': res['total_training_time']
        })

    hp_duration = time.time() - hp_start
    log_time("All Hyperparameter Studies", hp_duration)

    # Save combined results
    hp_df = pd.DataFrame(results)
    hp_df.to_csv(dirs['hyperparameter_studies'] / 'all_results.csv', index=False)

    return results


# ==============================================================================
# STEP 4: FINAL FULL TRAINING
# ==============================================================================

def run_final_training(config, dirs):
    """Run final full training with best hyperparameters."""
    print_header("STEP 4: FINAL FULL TRAINING (200 epochs)")

    results = train_model_full(config, "Q-TCRNet-Final", num_epochs=200, output_dir=dirs['final_training'])

    # Also save to checkpoints
    checkpoint_dir = Path('./checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    torch.save({
        'model_state_dict': results['model_state'],
        'epoch': 200,
        'wave_acc': results['wave_acc'],
        'volt_acc': results['volt_acc']
    }, checkpoint_dir / 'best_model_retrained.pth')

    print(f"\nFinal model saved to checkpoints/best_model_retrained.pth")

    return results


# ==============================================================================
# STEP 5: ABLATION STUDIES
# ==============================================================================

def run_ablation_studies(config, dirs):
    """Run all ablation study configurations."""
    print_header("STEP 5: ABLATION STUDIES")

    from dataset import create_dataloaders
    import torch.nn.functional as F

    ablation_start = time.time()

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = create_dataloaders(config, train_augment=False)

    results = []

    # Helper function for ablation model training
    def train_ablation_model(model, name, output_dir, num_epochs=100):
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_loss = float('inf')
        best_state = None
        train_losses, val_losses = [], []
        epoch_times = []

        train_start = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            model.train()
            train_loss = 0
            for voxels, labels in train_loader:
                voxels = voxels.to(device)
                wave_labels = labels['waveform'].to(device)
                volt_labels = labels['voltage'].to(device)

                optimizer.zero_grad()
                wave_logits, volt_logits = model(voxels)
                loss = criterion(wave_logits, wave_labels) + criterion(volt_logits, volt_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for voxels, labels in val_loader:
                    voxels = voxels.to(device)
                    wave_labels = labels['waveform'].to(device)
                    volt_labels = labels['voltage'].to(device)
                    wave_logits, volt_logits = model(voxels)
                    loss = criterion(wave_logits, wave_labels) + criterion(volt_logits, volt_labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 20 == 0:
                print(f"    [{name}] Epoch {epoch+1}/{num_epochs}, Val: {val_loss:.4f}, Time: {np.mean(epoch_times[-20:]):.2f}s")

        model.load_state_dict(best_state)
        total_time = time.time() - train_start

        # Evaluate
        model.eval()
        all_wave_preds, all_wave_labels = [], []
        all_volt_preds, all_volt_labels = [], []

        with torch.no_grad():
            for voxels, labels in test_loader:
                voxels = voxels.to(device)
                wave_logits, volt_logits = model(voxels)
                all_wave_preds.append(wave_logits.argmax(dim=1).cpu().numpy())
                all_wave_labels.append(labels['waveform'].numpy())
                all_volt_preds.append(volt_logits.argmax(dim=1).cpu().numpy())
                all_volt_labels.append(labels['voltage'].numpy())

        wave_acc = accuracy_score(np.concatenate(all_wave_labels), np.concatenate(all_wave_preds))
        volt_acc = accuracy_score(np.concatenate(all_volt_labels), np.concatenate(all_volt_preds))

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, output_dir / 'model.pth')

        metrics = {
            'name': name,
            'wave_acc': float(wave_acc),
            'volt_acc': float(volt_acc),
            'avg_epoch_time': float(np.mean(epoch_times)),
            'total_time_sec': float(total_time),
            'total_time_min': float(total_time / 60)
        }
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        pd.DataFrame({'train': train_losses, 'val': val_losses}).to_csv(output_dir / 'losses.csv', index=False)

        log_time(f"Ablation: {name}", total_time)
        print(f"    {name}: Wave={wave_acc*100:.1f}%, Volt={volt_acc*100:.1f}%")

        return wave_acc, volt_acc, np.mean(epoch_times), total_time

    # Define ablation models (imported from run_ablation_studies.py concepts)

    # 1. Without Quantum
    print_section("Ablation 1: Without Quantum")

    class ClassicalTCNOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv3d = nn.Sequential(
                nn.Conv3d(2, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32), nn.ReLU(), nn.Dropout(0.1))
            self.tcn = nn.ModuleList()
            channels = [32, 64, 128, 128]
            for i in range(3):
                self.tcn.append(nn.Sequential(
                    nn.Conv1d(channels[i], channels[i+1], 3, dilation=2**i, padding=2**i),
                    nn.BatchNorm1d(channels[i+1]), nn.ReLU(), nn.Dropout(0.3)))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5))
            self.waveform_head = nn.Linear(64, 4)
            self.voltage_head = nn.Linear(64, 4)

        def forward(self, x):
            x = self.conv3d(x)
            B, C, T, H, W = x.shape
            x = x.permute(0, 1, 3, 4, 2).reshape(B, C * H * W, T).mean(dim=1, keepdim=True).expand(-1, 32, -1)
            for tcn in self.tcn:
                x = tcn(x)
            x = self.pool(x).squeeze(-1)
            x = self.classifier(x)
            return self.waveform_head(x), self.voltage_head(x)

    w, v, t, tt = train_ablation_model(ClassicalTCNOnly(), "Without Quantum", dirs['ablation_no_quantum'])
    results.append(('Without Quantum', w, v, t, tt))

    # 2. Without TCN
    print_section("Ablation 2: Without TCN")

    class CNN3DNoTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(2, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
                nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
                nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.AdaptiveAvgPool3d(1))
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5))
            self.waveform_head = nn.Linear(64, 4)
            self.voltage_head = nn.Linear(64, 4)

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return self.waveform_head(x), self.voltage_head(x)

    w, v, t, tt = train_ablation_model(CNN3DNoTCN(), "Without TCN", dirs['ablation_no_tcn'])
    results.append(('Without TCN', w, v, t, tt))

    # 3. Frame-based CNN
    print_section("Ablation 3: Frame-based CNN")

    class FrameBasedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5))
            self.waveform_head = nn.Linear(64, 4)
            self.voltage_head = nn.Linear(64, 4)

        def forward(self, x):
            x = x.mean(dim=2)
            x = self.features(x)
            x = self.classifier(x)
            return self.waveform_head(x), self.voltage_head(x)

    w, v, t, tt = train_ablation_model(FrameBasedCNN(), "Frame-based CNN", dirs['ablation_frame_cnn'])
    results.append(('Frame-based CNN', w, v, t, tt))

    # 4. MLP + FFT
    print_section("Ablation 4: MLP + FFT")

    class MLPFFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.fft_dim = 128
            self.mlp = nn.Sequential(
                nn.Linear(2 * 65 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.5))
            self.waveform_head = nn.Linear(64, 4)
            self.voltage_head = nn.Linear(64, 4)

        def forward(self, x):
            B = x.shape[0]
            x_fft = torch.fft.rfft(x, dim=2)
            x_mag = torch.abs(x_fft)
            x_flat = x_mag.reshape(B, -1)
            expected = 2 * 65 * 16 * 16
            if x_flat.shape[1] < expected:
                x_flat = F.pad(x_flat, (0, expected - x_flat.shape[1]))
            else:
                x_flat = x_flat[:, :expected]
            x = self.mlp(x_flat)
            return self.waveform_head(x), self.voltage_head(x)

    w, v, t, tt = train_ablation_model(MLPFFT(), "MLP + FFT", dirs['ablation_mlp_fft'])
    results.append(('MLP + FFT', w, v, t, tt))

    ablation_duration = time.time() - ablation_start
    log_time("All Ablation Studies", ablation_duration)

    # Save combined results
    ablation_df = pd.DataFrame(results, columns=['Config', 'Wave_Acc', 'Volt_Acc', 'Time_per_Epoch', 'Total_Time'])
    ablation_df.to_csv(dirs['ablation_studies'] / 'all_results.csv', index=False)

    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    start_time = time.time()

    print_header("Q-TCRNet COMPLETE RETRAINING & EXPERIMENTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create directory structure
    dirs = create_experiment_dirs()
    print(f"Experiment directory: {dirs['base']}")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Step 1: Reprocess data
    print_header("STEP 1: REPROCESSING DATA")
    reprocess_data(config, dirs['preprocessing'])

    # Step 2: Initial full training
    print_header("STEP 2: INITIAL FULL TRAINING (200 epochs)")
    initial_results = train_model_full(config, "Q-TCRNet-Initial", num_epochs=200, output_dir=dirs['initial_training'])

    # Step 3: Hyperparameter studies
    hp_results = run_hyperparameter_studies(config, dirs)

    # Step 4: Final training with default config
    final_results = run_final_training(config, dirs)

    # Step 5: Ablation studies
    ablation_results = run_ablation_studies(config, dirs)

    # ==============================================================================
    # RESULTS SUMMARY
    # ==============================================================================

    print_header("EXPERIMENT RESULTS SUMMARY")

    # Performance metrics
    print("\nFinal Model Performance:")
    print(f"{'Metric':<15} | {'Waveform':<10} | {'Voltage':<10}")
    print("-" * 40)
    print(f"{'Accuracy':<15} | {final_results['wave_acc']*100:>8.1f}% | {final_results['volt_acc']*100:>8.1f}%")

    # Ablation Studies
    print("\n\nAblation Studies:")
    print(f"{'Configuration':<20} | {'Waveform':<10} | {'Voltage':<10} | {'Time/Epoch':<12} | {'Total':<10}")
    print("-" * 70)
    for name, wave, volt, t_epoch, t_total in ablation_results:
        print(f"{name:<20} | {wave*100:>8.1f}% | {volt*100:>8.1f}% | {t_epoch:>10.2f}s | {t_total/60:>8.1f}m")

    # Hyperparameter sensitivity
    print("\n\nHyperparameter Sensitivity:")

    # Group by study
    studies = {}
    for r in hp_results:
        study = r['study']
        if study not in studies:
            studies[study] = []
        studies[study].append(r)

    for study_name, study_results in studies.items():
        print(f"\n{study_name.replace('_', ' ').title()}:")
        print(f"  {'Value':<20} | {'Wave Acc':<10} | {'Volt Acc':<10} | {'Time/Epoch':<12} | {'Total':<10}")
        print("  " + "-" * 70)

        best_wave = max(study_results, key=lambda x: x['wave_acc'])

        for r in study_results:
            marker = " *" if r == best_wave else ""
            print(f"  {r['value']:<20} | {r['wave_acc']*100:>8.1f}% | {r['volt_acc']*100:>8.1f}% | {r['time_per_epoch']:>10.2f}s | {r['total_time']/60:>8.1f}m{marker}")

    # Timing Summary
    print("\n\nTiming Summary:")
    print(f"{'Task':<40} | {'Duration':<15}")
    print("-" * 60)
    for entry in TIMING_LOG:
        print(f"{entry['task']:<40} | {entry['duration_min']:>10.1f} min")

    # Save all results to summary directory
    summary_dir = dirs['summary']

    # Save timing log
    timing_df = pd.DataFrame(TIMING_LOG)
    timing_df.to_csv(summary_dir / 'timing_log.csv', index=False)

    # Save hyperparameter results
    hp_df = pd.DataFrame(hp_results)
    hp_df.to_csv(summary_dir / 'hyperparameter_sensitivity.csv', index=False)

    # Save ablation results
    ablation_df = pd.DataFrame(ablation_results, columns=['Config', 'Wave_Acc', 'Volt_Acc', 'Time_per_Epoch', 'Total_Time'])
    ablation_df.to_csv(summary_dir / 'ablation_studies.csv', index=False)

    # Save comprehensive summary
    total_time = time.time() - start_time
    summary = {
        'final_wave_acc': float(final_results['wave_acc']),
        'final_volt_acc': float(final_results['volt_acc']),
        'final_train_loss': float(final_results['final_train_loss']),
        'final_val_loss': float(final_results['final_val_loss']),
        'overfit_status': final_results['overfit_status'],
        'total_time_sec': total_time,
        'total_time_hours': total_time / 3600,
        'num_experiments': len(hp_results) + len(ablation_results) + 2,  # +2 for initial and final
        'timestamp': datetime.now().isoformat()
    }

    with open(summary_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final timing
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)

    print(f"\n\nTotal experiment time: {hours}h {minutes}m")
    print(f"\nAll results saved to: {dirs['base']}/")
    print(f"  - 01_preprocessing/")
    print(f"  - 02_initial_training/")
    print(f"  - 03_hyperparameter_studies/")
    print(f"  - 04_final_training/")
    print(f"  - 05_ablation_studies/")
    print(f"  - 06_summary/")
    print_header("EXPERIMENTS COMPLETE")


if __name__ == '__main__':
    main()
