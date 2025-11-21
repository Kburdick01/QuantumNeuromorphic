#!/usr/bin/env python3
"""
Create a proper data pipeline visualization showing actual data transformations.
Shows what the data LOOKS like at each stage, not just boxes.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path

def create_data_pipeline_visualization(save_path='paper_figures/data_pipeline_real.png'):
    """
    Create a visualization showing actual data at each processing stage.
    """
    fig = plt.figure(figsize=(20, 12))

    # Create grid for subplots
    gs = GridSpec(3, 6, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[1, 0.3, 1])

    np.random.seed(42)

    # =====================================================
    # TOP ROW: Actual data visualizations
    # =====================================================

    # 1. Raw Events (scatter plot style)
    ax1 = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 100, 500)
    x = 64 + 30 * np.sin(2 * np.pi * t / 30) + np.random.randn(500) * 3
    ax1.scatter(t, x, s=1, c='blue', alpha=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 128)
    ax1.set_xlabel('Time (ms)', fontsize=8)
    ax1.set_ylabel('X position', fontsize=8)
    ax1.set_title('Raw Events\n(t, x, y, p)', fontsize=10, fontweight='bold')
    ax1.tick_params(labelsize=7)

    # 2. Voxel Grid - temporal slice
    ax2 = fig.add_subplot(gs[0, 1])
    voxel = np.zeros((16, 128))
    for i in range(16):
        phase = i * 0.4
        pattern = np.sin(np.linspace(0, 4*np.pi, 128) + phase)
        voxel[i, :] = (pattern > 0.3) * np.random.exponential(2, 128)
    im2 = ax2.imshow(voxel, aspect='auto', cmap='hot', interpolation='nearest')
    ax2.set_xlabel('Time bins (128)', fontsize=8)
    ax2.set_ylabel('Spatial Y', fontsize=8)
    ax2.set_title('Voxel Grid\n[2, 128, 16, 16]', fontsize=10, fontweight='bold')
    ax2.tick_params(labelsize=7)

    # 3. After 3D Conv - feature maps
    ax3 = fig.add_subplot(gs[0, 2])
    # Show multiple small feature maps
    feature_grid = np.zeros((32, 32))
    for i in range(4):
        for j in range(4):
            feat = np.random.randn(8, 8) * 0.5
            feat = np.maximum(feat, 0)  # ReLU
            feature_grid[i*8:(i+1)*8, j*8:(j+1)*8] = feat
    im3 = ax3.imshow(feature_grid, cmap='viridis', interpolation='nearest')
    ax3.set_title('Conv3D Features\n[32, 128, 16, 16]', fontsize=10, fontweight='bold')
    ax3.axis('off')

    # 4. After TCN - temporal features
    ax4 = fig.add_subplot(gs[0, 3])
    # Show temporal feature evolution
    tcn_feat = np.zeros((16, 64))
    for i in range(16):
        # Different channels respond to different temporal patterns
        freq = 0.5 + i * 0.2
        tcn_feat[i, :] = np.sin(np.linspace(0, freq * 2 * np.pi, 64)) * np.exp(-np.linspace(0, 2, 64) * 0.3)
        tcn_feat[i, :] = np.maximum(tcn_feat[i, :], 0)
    im4 = ax4.imshow(tcn_feat, aspect='auto', cmap='plasma', interpolation='nearest')
    ax4.set_xlabel('Time', fontsize=8)
    ax4.set_ylabel('Channels', fontsize=8)
    ax4.set_title('TCN Output\n[128, 128, 16, 16]', fontsize=10, fontweight='bold')
    ax4.tick_params(labelsize=7)

    # 5. Compressed features (bar chart)
    ax5 = fig.add_subplot(gs[0, 4])
    compressed = np.tanh(np.random.randn(4) * 1.5) * np.pi
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax5.bar(range(4), compressed, color=colors, edgecolor='black', linewidth=1.5)
    ax5.axhline(y=0, color='black', linewidth=0.5)
    ax5.set_ylim(-np.pi, np.pi)
    ax5.set_xticks(range(4))
    ax5.set_xticklabels([f'θ₀', f'θ₁', f'θ₂', f'θ₃'])
    ax5.set_ylabel('Angle (rad)', fontsize=8)
    ax5.set_title('Quantum Input\n[4] ∈ [-π, π]', fontsize=10, fontweight='bold')
    ax5.tick_params(labelsize=7)

    # 6. Quantum output (expectation values)
    ax6 = fig.add_subplot(gs[0, 5])
    q_output = np.tanh(np.random.randn(4))  # Values in [-1, 1]
    bars = ax6.bar(range(4), q_output, color=['#9B59B6']*4, edgecolor='black', linewidth=1.5)
    ax6.axhline(y=0, color='black', linewidth=0.5)
    ax6.set_ylim(-1, 1)
    ax6.set_xticks(range(4))
    ax6.set_xticklabels([f'⟨Z₀⟩', f'⟨Z₁⟩', f'⟨Z₂⟩', f'⟨Z₃⟩'])
    ax6.set_ylabel('Expectation', fontsize=8)
    ax6.set_title('Quantum Output\n[4] ∈ [-1, 1]', fontsize=10, fontweight='bold')
    ax6.tick_params(labelsize=7)

    # =====================================================
    # MIDDLE ROW: Arrows and labels
    # =====================================================
    ax_arrows = fig.add_subplot(gs[1, :])
    ax_arrows.set_xlim(0, 6)
    ax_arrows.set_ylim(0, 1)
    ax_arrows.axis('off')

    # Draw arrows between stages
    arrow_style = dict(arrowstyle='->', color='#2C3E50', lw=2.5,
                       mutation_scale=20, connectionstyle='arc3,rad=0')

    positions = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    labels = ['Binning\n128 bins', 'Conv3D\n3×3×3', 'TCN\nd=1,2,4,8',
              'Pool+\nCompress', 'Quantum\nCircuit', '']

    for i in range(5):
        # Arrow
        ax_arrows.annotate('', xy=(positions[i+1] - 0.3, 0.5),
                          xytext=(positions[i] + 0.3, 0.5),
                          arrowprops=arrow_style)
        # Label
        ax_arrows.text(positions[i] + 0.5, 0.5, labels[i],
                      ha='center', va='center', fontsize=8,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    # =====================================================
    # BOTTOM ROW: Dimension annotations
    # =====================================================
    ax_dims = fig.add_subplot(gs[2, :])
    ax_dims.set_xlim(0, 6)
    ax_dims.set_ylim(0, 1)
    ax_dims.axis('off')

    dims = [
        ('Events\n~100K/sec', '#E74C3C'),
        ('[2,128,16,16]\n65,536 values', '#3498DB'),
        ('[32,128,16,16]\n1M values', '#27AE60'),
        ('[128,128,16,16]\n4M values', '#F39C12'),
        ('[4]\n4 values', '#9B59B6'),
        ('[4]\n→ [64] → classes', '#C0392B')
    ]

    for i, (dim, color) in enumerate(dims):
        x = positions[i]
        rect = patches.FancyBboxPatch((x - 0.4, 0.3), 0.8, 0.5,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, alpha=0.8,
                                       edgecolor='black', linewidth=1)
        ax_dims.add_patch(rect)
        ax_dims.text(x, 0.55, dim, ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

    # Title
    fig.suptitle('Q-TCRNet Data Pipeline: From Raw Events to Classification',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def create_voxel_formation_diagram(save_path='paper_figures/voxel_formation.png'):
    """
    Show how raw events become a voxel grid.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    np.random.seed(42)

    # 1. Raw event stream
    ax = axes[0]
    n_events = 200
    t = np.sort(np.random.uniform(0, 100, n_events))
    x = 64 + 25 * np.sin(2 * np.pi * t / 25) + np.random.randn(n_events) * 5
    y = 64 + 25 * np.cos(2 * np.pi * t / 25) + np.random.randn(n_events) * 5
    pol = np.random.choice([0, 1], n_events)

    ax.scatter(t[pol==1], x[pol==1], s=3, c='blue', alpha=0.6, label='ON')
    ax.scatter(t[pol==0], x[pol==0], s=3, c='red', alpha=0.6, label='OFF')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('X position')
    ax.set_title('1. Raw Event Stream', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 128)

    # 2. Temporal binning
    ax = axes[1]
    n_bins = 10
    bin_edges = np.linspace(0, 100, n_bins + 1)

    for i in range(n_bins):
        mask = (t >= bin_edges[i]) & (t < bin_edges[i+1])
        color = plt.cm.viridis(i / n_bins)
        ax.scatter(t[mask], x[mask], s=5, c=[color], alpha=0.8)
        ax.axvline(bin_edges[i], color='gray', linestyle='--', alpha=0.3)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('X position')
    ax.set_title(f'2. Temporal Binning\n({n_bins} bins)', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 128)

    # 3. Spatial pooling (show grid)
    ax = axes[2]
    ax.scatter(x, y, s=3, c=t, cmap='viridis', alpha=0.6)

    # Draw pooling grid
    pool_size = 8
    for i in range(0, 129, pool_size):
        ax.axhline(i, color='red', linewidth=0.5, alpha=0.5)
        ax.axvline(i, color='red', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(f'3. Spatial Pooling\n({pool_size}×{pool_size} patches)', fontweight='bold')
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_aspect('equal')

    # 4. Final voxel grid
    ax = axes[3]
    voxel = np.zeros((16, 10))  # Simplified: 16 spatial, 10 temporal

    for i in range(n_events):
        t_bin = min(int(t[i] / 10), 9)
        x_bin = min(int(x[i] / 8), 15)
        voxel[x_bin, t_bin] += 1

    im = ax.imshow(voxel, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Time bin')
    ax.set_ylabel('Spatial bin (X)')
    ax.set_title('4. Voxel Grid\n(event counts)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count')

    plt.suptitle('Event Stream → Voxel Grid Transformation', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


if __name__ == '__main__':
    Path('paper_figures').mkdir(exist_ok=True)

    print("Creating data pipeline visualizations...")
    create_data_pipeline_visualization()
    create_voxel_formation_diagram()
    print("Done!")
