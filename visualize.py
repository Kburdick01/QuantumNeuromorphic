#!/usr/bin/env python3
"""
Visualization script for QTCR-Net results.
Generates publication-ready figures for the paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from qtcr_model import QTCRNet
from dataset import create_dataloaders_csv_split


def load_model_and_data(config_path='config.yaml', checkpoint_path='checkpoints/best_model.pth'):
    """Load trained model and test data."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QTCRNet(config).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data using the same split as training
    _, _, test_loader = create_dataloaders_csv_split(config, train_augment=False)

    return model, test_loader, device, config


def evaluate_model(model, test_loader, device):
    """Run evaluation and collect predictions."""
    all_waveform_preds = []
    all_waveform_labels = []
    all_voltage_preds = []
    all_voltage_labels = []

    with torch.no_grad():
        for voxels, labels in tqdm(test_loader, desc='Evaluating'):
            voxels = voxels.to(device)
            waveform_labels = labels['waveform'].to(device)
            voltage_labels = labels['voltage'].to(device)

            waveform_logits, voltage_logits = model(voxels)

            waveform_preds = waveform_logits.argmax(dim=1)
            voltage_preds = voltage_logits.argmax(dim=1)

            all_waveform_preds.extend(waveform_preds.cpu().numpy())
            all_waveform_labels.extend(waveform_labels.cpu().numpy())
            all_voltage_preds.extend(voltage_preds.cpu().numpy())
            all_voltage_labels.extend(voltage_labels.cpu().numpy())

    return {
        'waveform_preds': np.array(all_waveform_preds),
        'waveform_labels': np.array(all_waveform_labels),
        'voltage_preds': np.array(all_voltage_preds),
        'voltage_labels': np.array(all_voltage_labels)
    }


def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'{title} - Counts')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(f'{title} - Normalized')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_classification_metrics(y_true, y_pred, classes, title, save_path):
    """Plot precision, recall, F1 scores as bar chart."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#9b59b6')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(f'{title} - Classification Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_architecture_diagram(config, save_path):
    """Create a visual diagram of the QTCR-Net architecture."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    colors = {
        'input': '#e74c3c',
        'cnn': '#3498db',
        'tcn': '#2ecc71',
        'quantum': '#9b59b6',
        'output': '#f39c12',
        'arrow': '#34495e'
    }

    # Helper function to draw boxes
    def draw_box(x, y, w, h, text, color, fontsize=9):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', wrap=True)

    # Helper function to draw arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # Input
    draw_box(0.5, 4, 2, 2, 'Input\n[2, 128, 16, 16]\nDVS Voxel', colors['input'])

    # CNN Block
    draw_box(3.5, 4, 2.5, 2, 'CNN Encoder\n3D Conv\n32→64→128', colors['cnn'])
    draw_arrow(2.5, 5, 3.5, 5)

    # TCN Block
    draw_box(7, 4, 2.5, 2, 'TCN\nDilated Conv\nd=[1,2,4,8]', colors['tcn'])
    draw_arrow(6, 5, 7, 5)

    # Quantum compression
    draw_box(10.5, 6, 2, 1.5, 'Compress\n128→64→4\nTanh', colors['quantum'], fontsize=8)
    draw_arrow(9.5, 5.5, 10.5, 6.5)

    # Quantum circuit
    draw_box(10.5, 4, 2, 1.5, 'Quantum\nCircuit\n4 Qubits', colors['quantum'], fontsize=8)
    draw_arrow(11.5, 6, 11.5, 5.5)

    # Quantum expansion
    draw_box(10.5, 1.5, 2, 1.5, 'Expand\n4→64', colors['quantum'], fontsize=8)
    draw_arrow(11.5, 4, 11.5, 3)

    # Output heads
    draw_box(13.5, 5.5, 2, 1.5, 'Waveform\nHead\n4 classes', colors['output'], fontsize=8)
    draw_box(13.5, 3, 2, 1.5, 'Voltage\nHead\n4 classes', colors['output'], fontsize=8)
    draw_arrow(12.5, 2.25, 13.5, 3.75)
    draw_arrow(12.5, 2.25, 13.5, 6.25)

    # Title
    ax.text(8, 9, 'QTCR-Net: Quantum Temporal Convolutional Reservoir Network',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Add parameter counts
    n_qubits = config['model']['quantum_reservoir']['n_qubits']
    n_layers = config['model']['quantum_reservoir']['circuit']['num_layers']
    ax.text(8, 0.5, f'Quantum: {n_qubits} qubits, {n_layers} layers | Total params: ~423K',
            ha='center', va='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_quantum_circuit_diagram(config, save_path):
    """Create diagram of the quantum circuit structure."""
    n_qubits = config['model']['quantum_reservoir']['n_qubits']
    n_layers = config['model']['quantum_reservoir']['circuit']['num_layers']

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, n_qubits + 1)
    ax.axis('off')

    # Draw qubit lines
    for i in range(n_qubits):
        y = n_qubits - i - 0.5
        ax.plot([1, 13], [y, y], 'k-', linewidth=1)
        ax.text(0.5, y, f'q{i}', ha='center', va='center', fontsize=10)

    # Draw layers
    x_pos = 2
    for layer in range(n_layers):
        # RY gates (angle embedding)
        for i in range(n_qubits):
            y = n_qubits - i - 0.5
            rect = plt.Rectangle((x_pos - 0.3, y - 0.3), 0.6, 0.6,
                                 facecolor='#3498db', edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x_pos, y, 'RY', ha='center', va='center', fontsize=7, color='white')

        x_pos += 1

        # RZ gates
        for i in range(n_qubits):
            y = n_qubits - i - 0.5
            rect = plt.Rectangle((x_pos - 0.3, y - 0.3), 0.6, 0.6,
                                 facecolor='#2ecc71', edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x_pos, y, 'RZ', ha='center', va='center', fontsize=7, color='white')

        x_pos += 1

        # CNOT ring entanglement
        for i in range(n_qubits):
            y1 = n_qubits - i - 0.5
            y2 = n_qubits - ((i + 1) % n_qubits) - 0.5

            # Control dot
            ax.plot(x_pos, y1, 'ko', markersize=6)
            # Target
            circle = plt.Circle((x_pos, y2), 0.2, facecolor='white', edgecolor='black', linewidth=1)
            ax.add_patch(circle)
            ax.plot([x_pos - 0.15, x_pos + 0.15], [y2, y2], 'k-', linewidth=1)
            ax.plot([x_pos, x_pos], [y2 - 0.15, y2 + 0.15], 'k-', linewidth=1)
            # Line
            ax.plot([x_pos, x_pos], [min(y1, y2) + 0.2, max(y1, y2) - 0.2], 'k-', linewidth=1)

            x_pos += 0.5

        x_pos += 0.5

    # Measurements
    for i in range(n_qubits):
        y = n_qubits - i - 0.5
        rect = plt.Rectangle((12, y - 0.3), 0.8, 0.6,
                             facecolor='#9b59b6', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(12.4, y, '⟨Z⟩', ha='center', va='center', fontsize=8, color='white')

    # Title
    ax.text(7, n_qubits + 0.5, f'Quantum Circuit: {n_qubits} Qubits, {n_layers} Layers (Ring Entanglement)',
            ha='center', va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_metrics_table(results, waveform_classes, voltage_classes):
    """Print detailed metrics table."""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT - WAVEFORM")
    print("="*60)
    print(classification_report(results['waveform_labels'], results['waveform_preds'],
                               target_names=waveform_classes))

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT - VOLTAGE")
    print("="*60)
    print(classification_report(results['voltage_labels'], results['voltage_preds'],
                               target_names=voltage_classes))


def plot_training_curves(history_path, save_dir):
    """Plot training and validation curves from saved history."""
    import json

    if not Path(history_path).exists():
        print(f"Warning: {history_path} not found. Skipping training curves.")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Waveform accuracy
    axes[0, 1].plot(epochs, history['train_waveform_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_waveform_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Waveform Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Voltage accuracy
    axes[1, 0].plot(epochs, history['train_voltage_acc'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_voltage_acc'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Voltage Classification Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    # Setup output directory
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)

    print("Loading model and data...")
    model, test_loader, device, config = load_model_and_data()

    # Get class names
    waveform_classes = ['burst', 'sine', 'square', 'triangle']
    voltage_classes = ['200mV', '300mV', '400mV', '500mV']

    print("Running evaluation...")
    results = evaluate_model(model, test_loader, device)

    # Generate all visualizations
    print("\nGenerating visualizations...")

    # 1. Confusion matrices
    plot_confusion_matrix(
        results['waveform_labels'], results['waveform_preds'],
        waveform_classes, 'Waveform Classification',
        output_dir / 'confusion_matrix_waveform.png'
    )

    plot_confusion_matrix(
        results['voltage_labels'], results['voltage_preds'],
        voltage_classes, 'Voltage Classification',
        output_dir / 'confusion_matrix_voltage.png'
    )

    # 2. Classification metrics bar charts
    plot_classification_metrics(
        results['waveform_labels'], results['waveform_preds'],
        waveform_classes, 'Waveform',
        output_dir / 'metrics_waveform.png'
    )

    plot_classification_metrics(
        results['voltage_labels'], results['voltage_preds'],
        voltage_classes, 'Voltage',
        output_dir / 'metrics_voltage.png'
    )

    # 3. Architecture diagrams
    plot_architecture_diagram(config, output_dir / 'architecture_qtcr_net.png')
    plot_quantum_circuit_diagram(config, output_dir / 'quantum_circuit.png')

    # 4. Training curves
    plot_training_curves('checkpoints/training_history.json', output_dir)

    # 5. Print metrics table
    generate_metrics_table(results, waveform_classes, voltage_classes)

    print(f"\nAll figures saved to: {output_dir}/")
    print("Generated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
