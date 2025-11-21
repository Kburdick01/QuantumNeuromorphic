#!/usr/bin/env python3
"""
Dataset-Specific Figure Generation for Q-TCRNet

Generates publication-quality figures from actual dataset:
1. Real voxel grid visualizations from preprocessed data
2. Event stream 3D helix plots
3. Class distribution and dataset statistics
4. Training curves (if available)

Author: Q-TCRNet Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import yaml
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')


def load_config(config_path='config.yaml'):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def plot_real_voxel_grid(npy_path, save_dir, sample_name='sample'):
    """
    Visualize an actual voxel grid from preprocessed data.

    Args:
        npy_path: Path to .npy voxel grid file
        save_dir: Directory to save figure
        sample_name: Name for the output file
    """
    # Load voxel grid
    voxel = np.load(npy_path)  # Shape: [2, T, H, W]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Get dimensions
    n_pol, n_time, n_h, n_w = voxel.shape

    # 1. ON polarity temporal slice (sum over space)
    ax1 = fig.add_subplot(gs[0, 0])
    temporal_on = voxel[1].sum(axis=(1, 2))  # Sum over H, W
    ax1.plot(temporal_on, 'b-', linewidth=1.5)
    ax1.set_xlabel('Time Bin')
    ax1.set_ylabel('Event Count')
    ax1.set_title('ON Events Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. OFF polarity temporal slice
    ax2 = fig.add_subplot(gs[0, 1])
    temporal_off = voxel[0].sum(axis=(1, 2))
    ax2.plot(temporal_off, 'r-', linewidth=1.5)
    ax2.set_xlabel('Time Bin')
    ax2.set_ylabel('Event Count')
    ax2.set_title('OFF Events Over Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Spatial distribution (sum over time)
    ax3 = fig.add_subplot(gs[0, 2])
    spatial_total = voxel.sum(axis=(0, 1))  # Sum over polarity and time
    im = ax3.imshow(spatial_total, cmap='hot', aspect='equal')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Spatial Event Distribution', fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Total Events')

    # 4. Time-space heatmap (ON polarity, sum over Y)
    ax4 = fig.add_subplot(gs[1, 0])
    time_x = voxel[1].sum(axis=1)  # [T, W]
    im4 = ax4.imshow(time_x.T, aspect='auto', cmap='viridis', origin='lower')
    ax4.set_xlabel('Time Bin')
    ax4.set_ylabel('X Position')
    ax4.set_title('ON Events: Time vs X', fontweight='bold')
    plt.colorbar(im4, ax=ax4)

    # 5. Time-space heatmap (OFF polarity)
    ax5 = fig.add_subplot(gs[1, 1])
    time_x_off = voxel[0].sum(axis=1)
    im5 = ax5.imshow(time_x_off.T, aspect='auto', cmap='magma', origin='lower')
    ax5.set_xlabel('Time Bin')
    ax5.set_ylabel('X Position')
    ax5.set_title('OFF Events: Time vs X', fontweight='bold')
    plt.colorbar(im5, ax=ax5)

    # 6. Polarity balance over time
    ax6 = fig.add_subplot(gs[1, 2])
    on_rate = temporal_on / (temporal_on + temporal_off + 1e-8)
    ax6.plot(on_rate, 'g-', linewidth=1.5)
    ax6.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Time Bin')
    ax6.set_ylabel('ON Ratio')
    ax6.set_title('Polarity Balance', fontweight='bold')
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'Voxel Grid Analysis: {Path(npy_path).stem}',
                 fontsize=14, fontweight='bold')

    save_path = save_dir / f'voxel_analysis_{sample_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_event_helix(csv_path, save_dir, start_ms=0, duration_ms=500, max_events=50000):
    """
    Create 3D helix visualization of raw events.

    Args:
        csv_path: Path to raw CSV event file
        save_dir: Directory to save figure
        start_ms: Start time in milliseconds
        duration_ms: Duration to plot
        max_events: Maximum events to plot
    """
    # Load events
    df = pd.read_csv(csv_path, comment='#')
    df.columns = df.columns.str.strip().str.lower()

    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'timestamp_us'}, inplace=True)

    # Convert to milliseconds
    df['t_ms'] = (df['timestamp_us'] - df['timestamp_us'].min()) / 1000.0

    # Filter time window
    end_ms = start_ms + duration_ms
    df = df[(df['t_ms'] >= start_ms) & (df['t_ms'] < end_ms)].copy()

    # Subsample if needed
    if len(df) > max_events:
        df = df.sample(n=max_events, random_state=42).sort_values('t_ms')

    # Map polarity
    if df['polarity'].min() < 0:
        df['polarity'] = (df['polarity'] + 1) // 2

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Separate by polarity
    on_events = df[df['polarity'] == 1]
    off_events = df[df['polarity'] == 0]

    # Plot
    if len(on_events) > 0:
        ax.scatter(on_events['t_ms'], on_events['x'], on_events['y'],
                   c='blue', s=0.5, alpha=0.3, label='ON')
    if len(off_events) > 0:
        ax.scatter(off_events['t_ms'], off_events['x'], off_events['y'],
                   c='red', s=0.5, alpha=0.3, label='OFF')

    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('X (pixels)', fontsize=10)
    ax.set_zlabel('Y (pixels)', fontsize=10)
    ax.set_title(f'Event Stream 3D Visualization\n{Path(csv_path).stem}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    save_path = save_dir / f'helix_3d_{Path(csv_path).stem}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_dataset_statistics(manifest_path, save_dir):
    """
    Plot dataset statistics from manifest file.

    Args:
        manifest_path: Path to manifest CSV
        save_dir: Directory to save figure
    """
    df = pd.read_csv(manifest_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Waveform distribution
    ax1 = axes[0, 0]
    waveform_counts = df['waveform_label'].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(waveform_counts)))
    bars = ax1.bar(waveform_counts.index, waveform_counts.values, color=colors)
    ax1.set_xlabel('Waveform')
    ax1.set_ylabel('Number of Windows')
    ax1.set_title('Waveform Class Distribution', fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # 2. Voltage distribution
    ax2 = axes[0, 1]
    voltage_counts = df['voltage_label'].value_counts().sort_index()
    bars = ax2.bar(voltage_counts.index, voltage_counts.values, color='steelblue')
    ax2.set_xlabel('Voltage')
    ax2.set_ylabel('Number of Windows')
    ax2.set_title('Voltage Level Distribution', fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # 3. Events per window distribution
    ax3 = axes[1, 0]
    ax3.hist(df['num_events'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Events per Window')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Event Count Distribution', fontweight='bold')
    ax3.axvline(df['num_events'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["num_events"].mean():.0f}')
    ax3.legend()

    # 4. Waveform x Voltage heatmap
    ax4 = axes[1, 1]
    pivot = pd.crosstab(df['waveform_label'], df['voltage_label'])
    im = ax4.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(pivot.columns)))
    ax4.set_xticklabels(pivot.columns)
    ax4.set_yticks(range(len(pivot.index)))
    ax4.set_yticklabels(pivot.index)
    ax4.set_xlabel('Voltage')
    ax4.set_ylabel('Waveform')
    ax4.set_title('Samples per Class', fontweight='bold')

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax4.text(j, i, pivot.values[i, j], ha='center', va='center', fontsize=10)

    plt.colorbar(im, ax=ax4)

    plt.suptitle(f'Dataset Statistics (Total: {len(df)} windows)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / 'dataset_statistics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_waveform_comparison(csv_dir, save_dir, voltage='300mV'):
    """
    Compare all waveform types side by side.

    Args:
        csv_dir: Directory containing CSV files
        save_dir: Directory to save figure
        voltage: Voltage level to compare
    """
    waveforms = ['sine', 'square', 'triangle', 'burst']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, waveform in enumerate(waveforms):
        # Find CSV file
        csv_path = None
        for pattern in [f'{waveform}-{voltage}.csv', f'{waveform}_{voltage}.csv']:
            matches = list(Path(csv_dir).glob(pattern))
            if matches:
                csv_path = matches[0]
                break

        if csv_path is None:
            print(f"Warning: No CSV found for {waveform} at {voltage}")
            continue

        # Load events
        df = pd.read_csv(csv_path, comment='#')
        df.columns = df.columns.str.strip().str.lower()
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'timestamp_us'}, inplace=True)

        df['t_ms'] = (df['timestamp_us'] - df['timestamp_us'].min()) / 1000.0

        # Filter to first 500ms
        df = df[df['t_ms'] < 500]

        if len(df) > 50000:
            df = df.sample(n=50000, random_state=42).sort_values('t_ms')

        # Top row: Time vs X
        ax_top = axes[0, i]
        ax_top.scatter(df['t_ms'], df['x'], s=0.3, alpha=0.3, c='blue')
        ax_top.set_xlabel('Time (ms)')
        ax_top.set_ylabel('X (pixels)')
        ax_top.set_title(f'{waveform.upper()}', fontweight='bold', fontsize=12)
        ax_top.set_ylim(0, 128)

        # Bottom row: Event rate histogram
        ax_bot = axes[1, i]
        bins = np.arange(0, 500, 5)
        ax_bot.hist(df['t_ms'], bins=bins, color='coral', edgecolor='none', alpha=0.7)
        ax_bot.set_xlabel('Time (ms)')
        ax_bot.set_ylabel('Events / 5ms')
        ax_bot.set_title(f'Event Rate', fontsize=10)

    plt.suptitle(f'Waveform Comparison at {voltage}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / f'waveform_comparison_{voltage}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def main():
    """Generate all dataset-specific figures."""
    # Setup
    config = load_config()

    output_dir = Path('dataset_figures')
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Dataset-Specific Figures")
    print("=" * 60)

    # Paths from config
    processed_dir = Path(config['data']['processed_dir'])
    manifest_path = Path(config['data']['manifest_path'])
    csv_dir = Path(config['data']['csv_dir']).expanduser()

    # 1. Dataset statistics
    if manifest_path.exists():
        print("\n1. Generating dataset statistics...")
        plot_dataset_statistics(manifest_path, output_dir)
    else:
        print(f"Warning: Manifest not found at {manifest_path}")

    # 2. Sample voxel grid visualization
    if processed_dir.exists():
        print("\n2. Generating voxel grid visualizations...")
        npy_files = list(processed_dir.glob('*.npy'))
        if npy_files:
            # Pick samples from different classes
            manifest_df = pd.read_csv(manifest_path) if manifest_path.exists() else None

            if manifest_df is not None:
                for waveform in ['sine', 'square', 'triangle', 'burst']:
                    samples = manifest_df[manifest_df['waveform_label'] == waveform]
                    if len(samples) > 0:
                        sample_path = samples.iloc[0]['npy_path']
                        if Path(sample_path).exists():
                            plot_real_voxel_grid(sample_path, output_dir, waveform)

    # 3. Waveform comparison
    if csv_dir.exists():
        print("\n3. Generating waveform comparison...")
        plot_waveform_comparison(csv_dir, output_dir, voltage='300mV')

        # 4. 3D helix for one waveform
        print("\n4. Generating 3D helix visualization...")
        for waveform in ['sine', 'burst']:
            for pattern in [f'{waveform}-300mV.csv', f'{waveform}_300mV.csv']:
                matches = list(csv_dir.glob(pattern))
                if matches:
                    plot_event_helix(matches[0], output_dir)
                    break

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)

    # List generated files
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
