#!/usr/bin/env python3
"""
Create publication-quality network architecture diagram for Q-TCRNet.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path


def create_qtcrnet_architecture(save_path='paper_figures/qtcrnet_architecture.png'):
    """
    Create a clean, publication-quality architecture diagram.
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color scheme (professional, muted)
    colors = {
        'input': '#4A90A4',      # Blue-gray
        'conv3d': '#5B8C5A',     # Forest green
        'tcn': '#8B6914',        # Dark gold
        'quantum': '#7B4B94',    # Purple
        'output': '#C75450',     # Muted red
        'text': '#2C3E50',       # Dark gray
        'arrow': '#34495E'       # Gray
    }

    y_center = 5
    box_height = 1.8
    box_width = 2.2

    # Helper function for boxes
    def draw_block(x, y, w, h, color, label, sublabel='', fontsize=9):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor=color, edgecolor='black',
                              linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y + 0.15, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white')
        if sublabel:
            ax.text(x, y - 0.35, sublabel, ha='center', va='center',
                    fontsize=7, color='white', alpha=0.9)

    def draw_arrow(x1, x2, y):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'],
                                   lw=2, mutation_scale=15))

    # Input
    x_pos = 1.5
    draw_block(x_pos, y_center, box_width, box_height, colors['input'],
               'Input', '[B, 2, 128, 16, 16]')
    ax.text(x_pos, y_center + 1.5, 'Voxel Grid', ha='center', fontsize=8,
            style='italic', color=colors['text'])

    # Arrow
    draw_arrow(x_pos + box_width/2 + 0.1, x_pos + box_width/2 + 0.9, y_center)

    # 3D Conv
    x_pos = 4.5
    draw_block(x_pos, y_center, box_width, box_height, colors['conv3d'],
               '3D Conv', 'Conv3d + BN + ReLU')
    ax.text(x_pos, y_center - 1.5, '2→32 channels\nkernel: 3×3×3',
            ha='center', fontsize=7, color=colors['text'])

    draw_arrow(x_pos + box_width/2 + 0.1, x_pos + box_width/2 + 0.9, y_center)

    # TCN Block (multiple dilations shown)
    x_pos = 7.5
    # Main TCN box
    draw_block(x_pos, y_center, box_width + 0.5, box_height + 0.8, colors['tcn'],
               'TCN Backbone', '')

    # Dilation indicators
    dilations = ['d=1', 'd=2', 'd=4', 'd=8']
    for i, d in enumerate(dilations):
        y_off = 0.5 - i * 0.35
        ax.text(x_pos, y_center + y_off, d, ha='center', va='center',
                fontsize=7, color='white', alpha=0.8)

    ax.text(x_pos, y_center - 1.8, '32→64→128→128\nCausal Dilated Conv',
            ha='center', fontsize=7, color=colors['text'])

    draw_arrow(x_pos + (box_width+0.5)/2 + 0.1, x_pos + (box_width+0.5)/2 + 0.9, y_center)

    # Compression
    x_pos = 10.8
    draw_block(x_pos, y_center, box_width, box_height * 0.8, colors['quantum'],
               'Compress', '128→64→4')

    draw_arrow(x_pos + box_width/2 + 0.1, x_pos + box_width/2 + 0.7, y_center)

    # Quantum Reservoir (highlighted)
    x_pos = 13.5
    # Outer glow effect
    glow = FancyBboxPatch((x_pos - box_width/2 - 0.15, y_center - box_height/2 - 0.15),
                          box_width + 0.3, box_height + 0.3,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=colors['quantum'], edgecolor='none',
                          alpha=0.3)
    ax.add_patch(glow)

    draw_block(x_pos, y_center, box_width, box_height, colors['quantum'],
               'Quantum', 'Reservoir')
    ax.text(x_pos, y_center - 1.5, '4 qubits × 3 layers\nRY, RZ, CNOT ring',
            ha='center', fontsize=7, color=colors['text'])

    draw_arrow(x_pos + box_width/2 + 0.1, x_pos + box_width/2 + 0.7, y_center)

    # Dual output heads
    x_pos = 16.5

    # Waveform head
    draw_block(x_pos, y_center + 1, box_width * 0.9, box_height * 0.7, colors['output'],
               'Waveform', '→ 4 classes')

    # Voltage head
    draw_block(x_pos, y_center - 1, box_width * 0.9, box_height * 0.7, colors['output'],
               'Voltage', '→ 4 classes')

    # Split arrows to dual heads
    ax.plot([15.2, 15.8, 15.8], [y_center, y_center, y_center + 1],
            color=colors['arrow'], lw=2)
    ax.plot([15.8, 15.8], [y_center, y_center - 1],
            color=colors['arrow'], lw=2)

    # Arrow heads
    ax.annotate('', xy=(x_pos - box_width*0.45, y_center + 1),
                xytext=(15.8, y_center + 1),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.annotate('', xy=(x_pos - box_width*0.45, y_center - 1),
                xytext=(15.8, y_center - 1),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # Title
    ax.text(9, 9.2, 'Q-TCRNet Architecture',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=colors['text'])

    # Legend
    legend_y = 1.5
    legend_items = [
        ('Input', colors['input']),
        ('Classical Conv', colors['conv3d']),
        ('TCN', colors['tcn']),
        ('Quantum', colors['quantum']),
        ('Output', colors['output'])
    ]
    for i, (name, color) in enumerate(legend_items):
        x = 2 + i * 3
        rect = Rectangle((x - 0.3, legend_y - 0.15), 0.6, 0.3,
                         facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.5, legend_y, name, ha='left', va='center',
                fontsize=8, color=colors['text'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def create_data_flow_diagram(save_path='paper_figures/data_flow_clean.png'):
    """
    Create a clean data flow diagram showing tensor shapes.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)
    ax.axis('off')

    colors = {
        'box': '#E8E8E8',
        'text': '#2C3E50',
        'dim': '#7F8C8D',
        'arrow': '#34495E'
    }

    stages = [
        ('Input', '[B,2,128,16,16]'),
        ('Conv3D', '[B,32,128,16,16]'),
        ('TCN', '[B,128,128,16,16]'),
        ('Pool', '[B,128]'),
        ('Compress', '[B,4]'),
        ('Quantum', '[B,4]'),
        ('Expand', '[B,64]'),
        ('Output', '[B,4]×2'),
    ]

    x_positions = np.linspace(1, 15, len(stages))

    for i, (name, shape) in enumerate(stages):
        x = x_positions[i]

        # Box
        rect = FancyBboxPatch((x - 0.7, 1.5), 1.4, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor=colors['box'],
                              edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        # Name
        ax.text(x, 2.5, name, ha='center', va='center',
                fontsize=9, fontweight='bold', color=colors['text'])

        # Shape
        ax.text(x, 1.8, shape, ha='center', va='center',
                fontsize=7, color=colors['dim'], family='monospace')

        # Arrow
        if i < len(stages) - 1:
            ax.annotate('', xy=(x_positions[i+1] - 0.8, 2.25),
                       xytext=(x + 0.8, 2.25),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    ax.text(8, 3.5, 'Q-TCRNet Data Flow',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=colors['text'])

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


if __name__ == '__main__':
    Path('paper_figures').mkdir(exist_ok=True)

    print("Creating architecture diagrams...")
    create_qtcrnet_architecture()
    create_data_flow_diagram()
    print("Done!")
