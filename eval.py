#!/usr/bin/env python3
"""
QTCR-Net Evaluation Script
Evaluates trained model on test set with:
- Accuracy, precision, recall, F1 metrics
- Confusion matrices for both tasks
- Per-class performance analysis
- Predictions export

Author: QTCR-Net Research Team
Date: 2025
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support
)

from qtcr_model import QTCRNet
from dataset import create_dataloaders


class QTCRNetEvaluator:
    """Evaluator for QTCR-Net model."""

    def __init__(self, config: dict, checkpoint_path: str):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary
            checkpoint_path: Path to trained model checkpoint
        """
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

        print(f"[Evaluator] Initializing on device: {self.device}")

        # Create model
        self.model = QTCRNet(config).to(self.device)

        # Load checkpoint
        self._load_checkpoint(checkpoint_path)

        # Create data loaders
        _, _, self.test_loader = create_dataloaders(config, train_augment=False)

        # Class names
        self.waveform_classes = config['data']['waveform_classes']
        self.voltage_classes = config['data']['voltage_classes']

        # Results directory
        self.results_dir = Path('./results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Evaluator] Setup complete")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load trained model checkpoint."""
        print(f"[Checkpoint] Loading from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', 'unknown')
        print(f"  Loaded model from epoch {epoch}")

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        # Storage for predictions and labels
        all_waveform_preds = []
        all_waveform_labels = []
        all_voltage_preds = []
        all_voltage_labels = []

        all_waveform_logits = []
        all_voltage_logits = []

        print("\n[Evaluation] Running on test set...")

        for voxels, labels in tqdm(self.test_loader, desc="Evaluating"):
            voxels = voxels.to(self.device)
            waveform_labels = labels['waveform'].cpu().numpy()
            voltage_labels = labels['voltage'].cpu().numpy()

            # Forward pass
            waveform_logits, voltage_logits = self.model(voxels)

            # Get predictions
            waveform_preds = waveform_logits.argmax(dim=1).cpu().numpy()
            voltage_preds = voltage_logits.argmax(dim=1).cpu().numpy()

            # Store
            all_waveform_preds.append(waveform_preds)
            all_waveform_labels.append(waveform_labels)
            all_voltage_preds.append(voltage_preds)
            all_voltage_labels.append(voltage_labels)

            all_waveform_logits.append(waveform_logits.cpu().numpy())
            all_voltage_logits.append(voltage_logits.cpu().numpy())

        # Concatenate all batches
        all_waveform_preds = np.concatenate(all_waveform_preds)
        all_waveform_labels = np.concatenate(all_waveform_labels)
        all_voltage_preds = np.concatenate(all_voltage_preds)
        all_voltage_labels = np.concatenate(all_voltage_labels)

        all_waveform_logits = np.concatenate(all_waveform_logits)
        all_voltage_logits = np.concatenate(all_voltage_logits)

        # Compute metrics
        results = self._compute_metrics(
            all_waveform_preds, all_waveform_labels,
            all_voltage_preds, all_voltage_labels
        )

        # Store predictions for later analysis
        self.predictions = {
            'waveform_preds': all_waveform_preds,
            'waveform_labels': all_waveform_labels,
            'waveform_logits': all_waveform_logits,
            'voltage_preds': all_voltage_preds,
            'voltage_labels': all_voltage_labels,
            'voltage_logits': all_voltage_logits
        }

        return results

    def _compute_metrics(self,
                        waveform_preds: np.ndarray,
                        waveform_labels: np.ndarray,
                        voltage_preds: np.ndarray,
                        voltage_labels: np.ndarray) -> dict:
        """
        Compute evaluation metrics.

        Args:
            waveform_preds: Waveform predictions
            waveform_labels: Waveform ground truth
            voltage_preds: Voltage predictions
            voltage_labels: Voltage ground truth

        Returns:
            Dictionary of metrics
        """
        results = {}

        # === Waveform Metrics ===
        wave_acc = accuracy_score(waveform_labels, waveform_preds)
        wave_prec, wave_rec, wave_f1, _ = precision_recall_fscore_support(
            waveform_labels, waveform_preds, average='macro', zero_division=0
        )

        results['waveform'] = {
            'accuracy': wave_acc,
            'precision': wave_prec,
            'recall': wave_rec,
            'f1': wave_f1
        }

        # Per-class metrics
        wave_prec_per_class, wave_rec_per_class, wave_f1_per_class, wave_support = \
            precision_recall_fscore_support(
                waveform_labels, waveform_preds, average=None, zero_division=0
            )

        results['waveform']['per_class'] = {
            self.waveform_classes[i]: {
                'precision': wave_prec_per_class[i],
                'recall': wave_rec_per_class[i],
                'f1': wave_f1_per_class[i],
                'support': int(wave_support[i])
            }
            for i in range(len(self.waveform_classes))
        }

        # === Voltage Metrics ===
        volt_acc = accuracy_score(voltage_labels, voltage_preds)
        volt_prec, volt_rec, volt_f1, _ = precision_recall_fscore_support(
            voltage_labels, voltage_preds, average='macro', zero_division=0
        )

        results['voltage'] = {
            'accuracy': volt_acc,
            'precision': volt_prec,
            'recall': volt_rec,
            'f1': volt_f1
        }

        # Per-class metrics
        volt_prec_per_class, volt_rec_per_class, volt_f1_per_class, volt_support = \
            precision_recall_fscore_support(
                voltage_labels, voltage_preds, average=None, zero_division=0
            )

        results['voltage']['per_class'] = {
            self.voltage_classes[i]: {
                'precision': volt_prec_per_class[i],
                'recall': volt_rec_per_class[i],
                'f1': volt_f1_per_class[i],
                'support': int(volt_support[i])
            }
            for i in range(len(self.voltage_classes))
        }

        return results

    def plot_confusion_matrices(self, save: bool = True):
        """
        Plot confusion matrices for both tasks.

        Args:
            save: Whether to save plots to disk
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Waveform confusion matrix
        wave_cm = confusion_matrix(
            self.predictions['waveform_labels'],
            self.predictions['waveform_preds']
        )

        sns.heatmap(
            wave_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.waveform_classes,
            yticklabels=self.waveform_classes,
            ax=axes[0],
            cbar_kws={'label': 'Count'}
        )
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('True', fontsize=12)
        axes[0].set_title('Waveform Classification Confusion Matrix', fontsize=14, fontweight='bold')

        # Voltage confusion matrix
        volt_cm = confusion_matrix(
            self.predictions['voltage_labels'],
            self.predictions['voltage_preds']
        )

        sns.heatmap(
            volt_cm,
            annot=True,
            fmt='d',
            cmap='Greens',
            xticklabels=self.voltage_classes,
            yticklabels=self.voltage_classes,
            ax=axes[1],
            cbar_kws={'label': 'Count'}
        )
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('True', fontsize=12)
        axes[1].set_title('Voltage Classification Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = self.results_dir / 'confusion_matrices.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[Plot] Saved confusion matrices to {save_path}")

        plt.show()

    def plot_per_class_metrics(self, save: bool = True):
        """
        Plot per-class performance metrics.

        Args:
            save: Whether to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract metrics
        wave_classes = self.waveform_classes
        wave_f1 = [self.results['waveform']['per_class'][c]['f1'] for c in wave_classes]
        wave_prec = [self.results['waveform']['per_class'][c]['precision'] for c in wave_classes]
        wave_rec = [self.results['waveform']['per_class'][c]['recall'] for c in wave_classes]

        volt_classes = self.voltage_classes
        volt_f1 = [self.results['voltage']['per_class'][c]['f1'] for c in volt_classes]
        volt_prec = [self.results['voltage']['per_class'][c]['precision'] for c in volt_classes]
        volt_rec = [self.results['voltage']['per_class'][c]['recall'] for c in volt_classes]

        # Waveform F1 scores
        axes[0, 0].bar(wave_classes, wave_f1, color='steelblue')
        axes[0, 0].set_ylabel('F1 Score', fontsize=11)
        axes[0, 0].set_title('Waveform Classification: F1 Scores', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylim([0, 1.05])
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Voltage F1 scores
        axes[0, 1].bar(volt_classes, volt_f1, color='seagreen')
        axes[0, 1].set_ylabel('F1 Score', fontsize=11)
        axes[0, 1].set_title('Voltage Classification: F1 Scores', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim([0, 1.05])
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Waveform Precision/Recall
        x = np.arange(len(wave_classes))
        width = 0.35
        axes[1, 0].bar(x - width/2, wave_prec, width, label='Precision', color='cornflowerblue')
        axes[1, 0].bar(x + width/2, wave_rec, width, label='Recall', color='lightcoral')
        axes[1, 0].set_ylabel('Score', fontsize=11)
        axes[1, 0].set_title('Waveform: Precision & Recall', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(wave_classes)
        axes[1, 0].set_ylim([0, 1.05])
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Voltage Precision/Recall
        x = np.arange(len(volt_classes))
        axes[1, 1].bar(x - width/2, volt_prec, width, label='Precision', color='mediumseagreen')
        axes[1, 1].bar(x + width/2, volt_rec, width, label='Recall', color='lightcoral')
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title('Voltage: Precision & Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(volt_classes)
        axes[1, 1].set_ylim([0, 1.05])
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.results_dir / 'per_class_metrics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Plot] Saved per-class metrics to {save_path}")

        plt.show()

    def save_predictions(self):
        """Save predictions to CSV file."""
        predictions_df = pd.DataFrame({
            'waveform_true': self.predictions['waveform_labels'],
            'waveform_pred': self.predictions['waveform_preds'],
            'waveform_true_label': [self.waveform_classes[i] for i in self.predictions['waveform_labels']],
            'waveform_pred_label': [self.waveform_classes[i] for i in self.predictions['waveform_preds']],
            'voltage_true': self.predictions['voltage_labels'],
            'voltage_pred': self.predictions['voltage_preds'],
            'voltage_true_label': [self.voltage_classes[i] for i in self.predictions['voltage_labels']],
            'voltage_pred_label': [self.voltage_classes[i] for i in self.predictions['voltage_preds']],
        })

        save_path = self.results_dir / 'predictions.csv'
        predictions_df.to_csv(save_path, index=False)
        print(f"\n[Predictions] Saved to {save_path}")

    def print_results(self, results: dict):
        """
        Print evaluation results in a formatted way.

        Args:
            results: Results dictionary
        """
        print("\n" + "="*70)
        print("QTCR-Net Evaluation Results")
        print("="*70)

        # Waveform results
        print("\n[WAVEFORM CLASSIFICATION]")
        print(f"  Accuracy:  {results['waveform']['accuracy']:.4f} ({results['waveform']['accuracy']*100:.2f}%)")
        print(f"  Precision: {results['waveform']['precision']:.4f}")
        print(f"  Recall:    {results['waveform']['recall']:.4f}")
        print(f"  F1 Score:  {results['waveform']['f1']:.4f}")

        print("\n  Per-class metrics:")
        for class_name, metrics in results['waveform']['per_class'].items():
            print(f"    {class_name:10s}: F1={metrics['f1']:.3f}, "
                  f"Prec={metrics['precision']:.3f}, "
                  f"Rec={metrics['recall']:.3f}, "
                  f"Support={metrics['support']}")

        # Voltage results
        print("\n[VOLTAGE CLASSIFICATION]")
        print(f"  Accuracy:  {results['voltage']['accuracy']:.4f} ({results['voltage']['accuracy']*100:.2f}%)")
        print(f"  Precision: {results['voltage']['precision']:.4f}")
        print(f"  Recall:    {results['voltage']['recall']:.4f}")
        print(f"  F1 Score:  {results['voltage']['f1']:.4f}")

        print("\n  Per-class metrics:")
        for class_name, metrics in results['voltage']['per_class'].items():
            print(f"    {class_name:10s}: F1={metrics['f1']:.3f}, "
                  f"Prec={metrics['precision']:.3f}, "
                  f"Rec={metrics['recall']:.3f}, "
                  f"Support={metrics['support']}")

        print("\n" + "="*70 + "\n")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate QTCR-Net")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to disk')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to CSV')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create evaluator
    evaluator = QTCRNetEvaluator(config, args.checkpoint)

    # Evaluate
    results = evaluator.evaluate()
    evaluator.results = results

    # Print results
    evaluator.print_results(results)

    # Plot confusion matrices
    evaluator.plot_confusion_matrices(save=args.save_plots)

    # Plot per-class metrics
    evaluator.plot_per_class_metrics(save=args.save_plots)

    # Save predictions
    if args.save_predictions:
        evaluator.save_predictions()

    print("[Evaluation] Complete!")


if __name__ == '__main__':
    main()
