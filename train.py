#!/usr/bin/env python3
"""
QTCR-Net Training Script
Trains the Quantum Temporal Convolutional Reservoir Network with:
- Dual-head cross entropy losses (waveform + voltage)
- Mixed precision training (AMP)
- Gradient clipping
- TensorBoard logging
- Checkpointing
- Early stopping
- Separate learning rates for classical and quantum parameters

Author: QTCR-Net Research Team
Date: 2025
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import yaml
from datetime import datetime
import json
import matplotlib.pyplot as plt

from qtcr_model import QTCRNet
from dataset import create_dataloaders_csv_split
from typing import Dict, Tuple


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = 'min'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class QTCRNetTrainer:
    """Trainer for QTCR-Net model."""

    def __init__(self, config: dict, resume_from: str = None):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

        print(f"\n[Trainer] Initializing on device: {self.device}")

        # Set random seeds for reproducibility
        self._set_seeds(config['training']['random_seed'])

        # Create model
        self.model = QTCRNet(config).to(self.device)

        # Create data loaders with CSV-level splitting
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders_csv_split(
            config, train_augment=True
        )

        # Setup optimizer with separate learning rates
        self._setup_optimizer()

        # Setup learning rate scheduler
        self._setup_scheduler()

        # Setup loss functions
        self._setup_losses()

        # Setup AMP scaler
        self.use_amp = config['training']['mixed_precision']
        self.scaler = GradScaler(enabled=self.use_amp)

        # Setup logging
        self._setup_logging()

        # Setup checkpointing
        self.checkpoint_dir = Path(config['training']['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        if config['training']['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=config['training']['early_stopping']['patience'],
                min_delta=config['training']['early_stopping']['min_delta'],
                mode='min'
            )
        else:
            self.early_stopping = None

        # Tracking
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0

        # Training history for plotting
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_waveform_acc': [], 'val_waveform_acc': [],
            'train_voltage_acc': [], 'val_voltage_acc': [],
            'learning_rate': []
        }

        # Resume from checkpoint if provided
        if resume_from:
            self._load_checkpoint(resume_from)

        print(f"[Trainer] Setup complete")

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if self.config['experiment']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _setup_optimizer(self):
        """Setup optimizer with separate learning rates for classical and quantum layers."""
        lr_classical = self.config['training']['optimizer']['lr_classical']
        lr_quantum = self.config['training']['optimizer']['lr_quantum']
        weight_decay = self.config['training']['optimizer']['weight_decay']

        # Separate parameters
        quantum_params = []
        classical_params = []

        for name, param in self.model.named_parameters():
            if 'quantum' in name.lower():
                quantum_params.append(param)
            else:
                classical_params.append(param)

        # Create parameter groups
        param_groups = [
            {'params': classical_params, 'lr': lr_classical, 'name': 'classical'},
            {'params': quantum_params, 'lr': lr_quantum, 'name': 'quantum'}
        ]

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=self.config['training']['optimizer']['betas']
        )

        print(f"[Optimizer] AdamW with lr_classical={lr_classical}, lr_quantum={lr_quantum}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if not self.config['training']['scheduler']['enabled']:
            self.scheduler = None
            return

        scheduler_type = self.config['training']['scheduler']['type']

        if scheduler_type == 'CosineAnnealingWarmRestarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['training']['scheduler']['T_0'],
                T_mult=self.config['training']['scheduler']['T_mult'],
                eta_min=self.config['training']['scheduler']['eta_min']
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['training']['scheduler']['factor'],
                patience=self.config['training']['scheduler']['patience'],
                min_lr=self.config['training']['scheduler']['min_lr']
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        print(f"[Scheduler] {scheduler_type}")

    def _setup_losses(self):
        """Setup loss functions."""
        label_smoothing = self.config['training']['loss']['label_smoothing']

        self.criterion_waveform = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion_voltage = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.waveform_weight = self.config['training']['loss']['waveform_weight']
        self.voltage_weight = self.config['training']['loss']['voltage_weight']

        print(f"[Loss] Dual CrossEntropy with label_smoothing={label_smoothing}")
        print(f"  Waveform weight: {self.waveform_weight}, Voltage weight: {self.voltage_weight}")

    def _setup_logging(self):
        """Setup TensorBoard logging."""
        log_dir = Path(self.config['training']['logging']['tensorboard_dir'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config['experiment']['name']
        self.log_dir = log_dir / f"{exp_name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Save config
        config_path = self.log_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

        print(f"[Logging] TensorBoard logs: {self.log_dir}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"[Checkpoint] Loading from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)

        print(f"  Resuming from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Saved best model to {best_path}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        waveform_correct = 0
        voltage_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d} [Train]")

        for batch_idx, (voxels, labels) in enumerate(pbar):
            voxels = voxels.to(self.device)
            waveform_labels = labels['waveform'].to(self.device)
            voltage_labels = labels['voltage'].to(self.device)

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                waveform_logits, voltage_logits = self.model(voxels)

                # Compute losses
                loss_waveform = self.criterion_waveform(waveform_logits, waveform_labels)
                loss_voltage = self.criterion_voltage(voltage_logits, voltage_labels)

                loss = (self.waveform_weight * loss_waveform +
                       self.voltage_weight * loss_voltage)

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            total_loss += loss.item()

            waveform_pred = waveform_logits.argmax(dim=1)
            voltage_pred = voltage_logits.argmax(dim=1)

            waveform_correct += (waveform_pred == waveform_labels).sum().item()
            voltage_correct += (voltage_pred == voltage_labels).sum().item()
            total_samples += voxels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'wave_acc': f"{100 * waveform_correct / total_samples:.2f}%",
                'volt_acc': f"{100 * voltage_correct / total_samples:.2f}%"
            })

            # TensorBoard logging
            if batch_idx % self.config['training']['logging']['log_freq'] == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Loss_Waveform', loss_waveform.item(), self.global_step)
                self.writer.add_scalar('Train/Loss_Voltage', loss_voltage.item(), self.global_step)

            self.global_step += 1

        # Epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        waveform_acc = 100 * waveform_correct / total_samples
        voltage_acc = 100 * voltage_correct / total_samples

        metrics = {
            'loss': avg_loss,
            'waveform_acc': waveform_acc,
            'voltage_acc': voltage_acc
        }

        return metrics

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        waveform_correct = 0
        voltage_correct = 0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:3d} [Val]  ")

        for voxels, labels in pbar:
            voxels = voxels.to(self.device)
            waveform_labels = labels['waveform'].to(self.device)
            voltage_labels = labels['voltage'].to(self.device)

            # Forward pass
            with autocast(enabled=self.use_amp):
                waveform_logits, voltage_logits = self.model(voxels)

                loss_waveform = self.criterion_waveform(waveform_logits, waveform_labels)
                loss_voltage = self.criterion_voltage(voltage_logits, voltage_labels)

                loss = (self.waveform_weight * loss_waveform +
                       self.voltage_weight * loss_voltage)

            total_loss += loss.item()

            waveform_pred = waveform_logits.argmax(dim=1)
            voltage_pred = voltage_logits.argmax(dim=1)

            waveform_correct += (waveform_pred == waveform_labels).sum().item()
            voltage_correct += (voltage_pred == voltage_labels).sum().item()
            total_samples += voxels.size(0)

            pbar.set_postfix({
                'loss': loss.item(),
                'wave_acc': f"{100 * waveform_correct / total_samples:.2f}%",
                'volt_acc': f"{100 * voltage_correct / total_samples:.2f}%"
            })

        avg_loss = total_loss / len(self.val_loader)
        waveform_acc = 100 * waveform_correct / total_samples
        voltage_acc = 100 * voltage_correct / total_samples

        metrics = {
            'loss': avg_loss,
            'waveform_acc': waveform_acc,
            'voltage_acc': voltage_acc
        }

        return metrics

    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']

        print(f"\n[Training] Starting for {num_epochs} epochs\n")

        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate_epoch(epoch)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Logging
            print(f"\nEpoch {epoch:3d} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Wave Acc: {train_metrics['waveform_acc']:.2f}% | "
                  f"Volt Acc: {train_metrics['voltage_acc']:.2f}%")
            print(f"  Val   Loss: {val_metrics['loss']:.4f} | "
                  f"Wave Acc: {val_metrics['waveform_acc']:.2f}% | "
                  f"Volt Acc: {val_metrics['voltage_acc']:.2f}%\n")

            # TensorBoard logging
            self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Train_Waveform_Acc', train_metrics['waveform_acc'], epoch)
            self.writer.add_scalar('Epoch/Val_Waveform_Acc', val_metrics['waveform_acc'], epoch)
            self.writer.add_scalar('Epoch/Train_Voltage_Acc', train_metrics['voltage_acc'], epoch)
            self.writer.add_scalar('Epoch/Val_Voltage_Acc', val_metrics['voltage_acc'], epoch)

            # Learning rate logging
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f'LR/{param_group["name"]}', param_group['lr'], epoch)

            # Record history for plotting
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_waveform_acc'].append(train_metrics['waveform_acc'])
            self.history['val_waveform_acc'].append(val_metrics['waveform_acc'])
            self.history['train_voltage_acc'].append(train_metrics['voltage_acc'])
            self.history['val_voltage_acc'].append(val_metrics['voltage_acc'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Checkpointing
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            if (epoch % self.config['training']['checkpoint']['save_freq'] == 0 or
                is_best):
                self._save_checkpoint(epoch, is_best=is_best)

            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_metrics['loss']):
                    print(f"\n[Early Stopping] No improvement for {self.early_stopping.patience} epochs")
                    break

        print("\n[Training] Complete!")
        self.writer.close()

        # Save training history for plotting
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        print(f"[History] Saved to {history_path}")

        # Generate training plots
        self._plot_training_curves()

        # Final evaluation on test set
        print("\n[Evaluation] Running on test set...")
        test_metrics = self.test()
        print(f"Test Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Waveform Acc: {test_metrics['waveform_acc']:.2f}%")
        print(f"  Voltage Acc: {test_metrics['voltage_acc']:.2f}%")

    def _plot_training_curves(self):
        """Generate and save training/validation plots."""
        if not self.history['train_loss']:
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Waveform accuracy
        axes[0, 1].plot(epochs, self.history['train_waveform_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_waveform_acc'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Waveform Classification Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Voltage accuracy
        axes[1, 0].plot(epochs, self.history['train_voltage_acc'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_voltage_acc'], 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Voltage Classification Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 1].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        if max(self.history['learning_rate']) / min(self.history['learning_rate']) > 10:
            axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Plots] Saved training curves to {plot_path}")

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()

        total_loss = 0.0
        waveform_correct = 0
        voltage_correct = 0
        total_samples = 0

        for voxels, labels in tqdm(self.test_loader, desc="Testing"):
            voxels = voxels.to(self.device)
            waveform_labels = labels['waveform'].to(self.device)
            voltage_labels = labels['voltage'].to(self.device)

            waveform_logits, voltage_logits = self.model(voxels)

            loss_waveform = self.criterion_waveform(waveform_logits, waveform_labels)
            loss_voltage = self.criterion_voltage(voltage_logits, voltage_labels)
            loss = self.waveform_weight * loss_waveform + self.voltage_weight * loss_voltage

            total_loss += loss.item()

            waveform_pred = waveform_logits.argmax(dim=1)
            voltage_pred = voltage_logits.argmax(dim=1)

            waveform_correct += (waveform_pred == waveform_labels).sum().item()
            voltage_correct += (voltage_pred == voltage_labels).sum().item()
            total_samples += voxels.size(0)

        metrics = {
            'loss': total_loss / len(self.test_loader),
            'waveform_acc': 100 * waveform_correct / total_samples,
            'voltage_acc': 100 * voltage_correct / total_samples
        }

        return metrics


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train QTCR-Net")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer
    trainer = QTCRNetTrainer(config, resume_from=args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
