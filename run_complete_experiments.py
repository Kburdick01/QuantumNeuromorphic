#!/usr/bin/env python3
"""
Q-TCRNet Complete Retraining & Hyperparameter Sensitivity Study

This script:
1. Reprocesses all CSV data with proper train/val/test splits
2. Trains Q-TCRNet from scratch
3. Runs full hyperparameter sensitivity studies
4. Generates comprehensive results

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


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)


def print_section(text):
    """Print section header."""
    print(f"\n--- {text} ---")


# ==============================================================================
# STEP 1: REPROCESS DATA
# ==============================================================================

def reprocess_data(config):
    """Reprocess all CSV data with fresh splits."""
    print_header("STEP 1: REPROCESSING DATA")

    # Remove old processed data
    processed_dir = Path(config['data']['processed_dir'])
    if processed_dir.exists():
        print(f"Removing old processed data: {processed_dir}")
        shutil.rmtree(processed_dir)

    # Import and run preprocessor
    from preprocess import DVSPreprocessor

    preprocessor = DVSPreprocessor(config)
    preprocessor.process_all()

    print("Data reprocessing complete!")
    return True


# ==============================================================================
# STEP 2: TRAINING UTILITIES
# ==============================================================================

def train_model_full(config, model_name="Q-TCRNet", num_epochs=200):
    """Train a model with full logging and return metrics."""
    from qtcr_model import QTCRNet
    from dataset import create_dataloaders

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
        'overfit_status': overfit_indicator,
        'model_state': best_state
    }

    print(f"  Final: Wave={wave_acc*100:.1f}%, Volt={volt_acc*100:.1f}%, "
          f"Train/Val Gap={train_val_gap:.3f} [{overfit_indicator}]")

    return results


# ==============================================================================
# STEP 3: HYPERPARAMETER SENSITIVITY STUDIES
# ==============================================================================

def run_hyperparameter_studies(base_config):
    """Run all hyperparameter sensitivity experiments."""
    print_header("STEP 3: HYPERPARAMETER SENSITIVITY STUDIES")

    results = []

    # Study 1: Window Size
    print_section("Study 1: Window Size")
    window_sizes = [0.5, 1.0, 2.0]
    temporal_bins = [64, 128, 256]  # Corresponding bins

    for ws, tb in zip(window_sizes, temporal_bins):
        config = copy.deepcopy(base_config)
        config['data']['window']['duration_sec'] = ws
        config['data']['window']['temporal_bins'] = tb

        print(f"\n  Testing Window Size = {ws}s (T={tb})")

        # Reprocess with new window size
        reprocess_data(config)

        # Train
        res = train_model_full(config, f"Window-{ws}s", num_epochs=100)
        results.append({
            'study': 'window_size',
            'value': f'{ws}s',
            'wave_acc': res['wave_acc'],
            'volt_acc': res['volt_acc'],
            'time': res['avg_epoch_time']
        })

    # Restore default window
    reprocess_data(base_config)

    # Study 2: Number of Qubits
    print_section("Study 2: Number of Qubits")
    qubit_values = [4, 6, 8]

    for nq in qubit_values:
        config = copy.deepcopy(base_config)
        config['model']['quantum_reservoir']['n_qubits'] = nq

        print(f"\n  Testing N_qubits = {nq}")
        res = train_model_full(config, f"Qubits-{nq}", num_epochs=100)
        results.append({
            'study': 'n_qubits',
            'value': str(nq),
            'wave_acc': res['wave_acc'],
            'volt_acc': res['volt_acc'],
            'time': res['avg_epoch_time']
        })

    # Study 3: TCN Dilations
    print_section("Study 3: TCN Dilations")
    dilation_configs = [
        ([1, 2, 4, 8], "1,2,4,8"),
        ([1, 2, 4, 8, 16], "1,2,4,8,16")
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

        print(f"\n  Testing Dilations = [{name}]")
        res = train_model_full(config, f"Dilations-{name}", num_epochs=100)
        results.append({
            'study': 'tcn_dilations',
            'value': f'[{name}]',
            'wave_acc': res['wave_acc'],
            'volt_acc': res['volt_acc'],
            'time': res['avg_epoch_time']
        })

    # Study 4: Quantum Layers
    print_section("Study 4: Quantum Layers")
    layer_values = [2, 3, 4]

    for nl in layer_values:
        config = copy.deepcopy(base_config)
        config['model']['quantum_reservoir']['circuit']['num_layers'] = nl

        print(f"\n  Testing Quantum Layers = {nl}")
        res = train_model_full(config, f"QLayers-{nl}", num_epochs=100)
        results.append({
            'study': 'quantum_layers',
            'value': str(nl),
            'wave_acc': res['wave_acc'],
            'volt_acc': res['volt_acc'],
            'time': res['avg_epoch_time']
        })

    return results


# ==============================================================================
# STEP 4: FINAL FULL TRAINING
# ==============================================================================

def run_final_training(config):
    """Run final full training with best hyperparameters."""
    print_header("STEP 4: FINAL FULL TRAINING (200 epochs)")

    results = train_model_full(config, "Q-TCRNet-Final", num_epochs=200)

    # Save best model
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
# MAIN
# ==============================================================================

def main():
    start_time = time.time()

    print_header("Q-TCRNet COMPLETE RETRAINING & EXPERIMENTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Step 1: Reprocess data
    reprocess_data(config)

    # Step 2: Initial full training
    print_header("STEP 2: INITIAL FULL TRAINING")
    initial_results = train_model_full(config, "Q-TCRNet-Initial", num_epochs=200)

    # Step 3: Hyperparameter studies
    hp_results = run_hyperparameter_studies(config)

    # Step 4: Final training with default config
    final_results = run_final_training(config)

    # ==============================================================================
    # RESULTS SUMMARY
    # ==============================================================================

    print_header("EXPERIMENT RESULTS SUMMARY")

    # Performance metrics
    print("\nPerformance Metrics:")
    print(f"{'Metric':<15} | {'Waveform':<10} | {'Voltage':<10}")
    print("-" * 40)
    print(f"{'Accuracy':<15} | {final_results['wave_acc']*100:>8.1f}% | {final_results['volt_acc']*100:>8.1f}%")

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
        print(f"  {'Value':<20} | {'Wave Acc':<10} | {'Volt Acc':<10} | {'Time/Epoch':<10}")
        print("  " + "-" * 55)

        best_wave = max(study_results, key=lambda x: x['wave_acc'])

        for r in study_results:
            marker = " *" if r == best_wave else ""
            print(f"  {r['value']:<20} | {r['wave_acc']*100:>8.1f}% | {r['volt_acc']*100:>8.1f}% | {r['time']:>8.2f}s{marker}")

    # Save all results
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)

    # Save hyperparameter results
    hp_df = pd.DataFrame(hp_results)
    hp_df.to_csv(results_dir / 'hyperparameter_sensitivity.csv', index=False)

    # Save summary
    summary = {
        'final_wave_acc': final_results['wave_acc'],
        'final_volt_acc': final_results['volt_acc'],
        'final_train_loss': final_results['final_train_loss'],
        'final_val_loss': final_results['final_val_loss'],
        'overfit_status': final_results['overfit_status'],
        'total_time_hours': (time.time() - start_time) / 3600,
        'timestamp': datetime.now().isoformat()
    }

    with open(results_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print timing
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)

    print(f"\n\nTotal experiment time: {hours}h {minutes}m")
    print(f"Results saved to {results_dir}/")
    print_header("EXPERIMENTS COMPLETE")


if __name__ == '__main__':
    main()
