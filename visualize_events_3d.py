#!/usr/bin/env python3
"""
3D Event Visualization for DVS128 Event Camera Data
Creates 3D plots showing temporal and spatial structure of events.

Similar to MATLAB PlotPoint3D but for event camera data:
- Time vs X vs Y (with polarity color-coding)
- Spatial distribution (X vs Y)
- Temporal slices
- Density heatmaps

Author: QTCR-Net Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
from typing import Optional, Tuple


def load_events_from_csv(csv_path: str,
                         max_events: Optional[int] = 10000,
                         time_window: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    """
    Load events from CSV file.

    Args:
        csv_path: Path to CSV file with columns [timestamp_us, x, y, polarity]
        max_events: Maximum number of events to load (for performance)
        time_window: Optional (start_sec, end_sec) to load only specific time range

    Returns:
        DataFrame with columns [timestamp_s, x, y, polarity]
    """
    print(f"Loading events from {csv_path}...")

    # Read CSV, skipping comment lines starting with #
    df = pd.read_csv(csv_path, comment='#')

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename columns
    column_mapping = {
        'timestamp': 'timestamp_us',
        'time': 'timestamp_us',
        't': 'timestamp_us',
        'pol': 'polarity',
        'p': 'polarity'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Convert timestamp to seconds
    df['timestamp_s'] = df['timestamp_us'] / 1e6

    # Normalize timestamps to start at 0
    df['timestamp_s'] = df['timestamp_s'] - df['timestamp_s'].min()

    # Map polarity to 0/1 if needed
    unique_polarities = df['polarity'].unique()
    if set(unique_polarities) == {-1, 1}:
        df['polarity'] = (df['polarity'] + 1) // 2  # -1→0, 1→1

    # Filter by time window if specified
    if time_window:
        start_sec, end_sec = time_window
        df = df[(df['timestamp_s'] >= start_sec) & (df['timestamp_s'] < end_sec)]
        print(f"Filtered to time window [{start_sec}, {end_sec}] seconds")

    # Limit number of events for visualization
    if max_events and len(df) > max_events:
        # Sample uniformly across time
        indices = np.linspace(0, len(df)-1, max_events, dtype=int)
        df = df.iloc[indices]
        print(f"Sampled {max_events} events for visualization")

    print(f"Loaded {len(df)} events")
    print(f"Time range: {df['timestamp_s'].min():.3f} to {df['timestamp_s'].max():.3f} seconds")
    print(f"Spatial range: X=[{df['x'].min()}, {df['x'].max()}], Y=[{df['y'].min()}, {df['y'].max()}]")

    return df[['timestamp_s', 'x', 'y', 'polarity']]


def plot_3d_events(events: pd.DataFrame,
                   title: str = "DVS128 Events 3D Visualization",
                   save_path: Optional[str] = None):
    """
    Create 3D visualizations of events similar to MATLAB PlotPoint3D.

    Args:
        events: DataFrame with columns [timestamp_s, x, y, polarity]
        title: Main title for the figure
        save_path: Optional path to save the figure
    """
    # Separate ON and OFF events
    on_events = events[events['polarity'] == 1]
    off_events = events[events['polarity'] == 0]

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # --- Plot 1: Time vs X vs Y (main 3D plot) ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Plot ON events (blue)
    if len(on_events) > 0:
        ax1.scatter(on_events['timestamp_s'],
                   on_events['x'],
                   on_events['y'],
                   c='blue',
                   marker='o',
                   s=1,
                   alpha=0.6,
                   label='ON events')

    # Plot OFF events (red)
    if len(off_events) > 0:
        ax1.scatter(off_events['timestamp_s'],
                   off_events['x'],
                   off_events['y'],
                   c='red',
                   marker='o',
                   s=1,
                   alpha=0.6,
                   label='OFF events')

    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('X (pixels)', fontsize=10)
    ax1.set_zlabel('Y (pixels)', fontsize=10)
    ax1.set_title('Time vs X vs Y', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: X vs Y (spatial distribution) ---
    ax2 = fig.add_subplot(2, 2, 2)

    if len(on_events) > 0:
        ax2.scatter(on_events['x'],
                   on_events['y'],
                   c='blue',
                   marker='o',
                   s=1,
                   alpha=0.3,
                   label='ON events')

    if len(off_events) > 0:
        ax2.scatter(off_events['x'],
                   off_events['y'],
                   c='red',
                   marker='o',
                   s=1,
                   alpha=0.3,
                   label='OFF events')

    ax2.set_xlabel('X (pixels)', fontsize=10)
    ax2.set_ylabel('Y (pixels)', fontsize=10)
    ax2.set_title('Spatial Distribution (X vs Y)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # --- Plot 3: Time vs X projection ---
    ax3 = fig.add_subplot(2, 2, 3)

    if len(on_events) > 0:
        ax3.scatter(on_events['timestamp_s'],
                   on_events['x'],
                   c='blue',
                   marker='o',
                   s=1,
                   alpha=0.3,
                   label='ON events')

    if len(off_events) > 0:
        ax3.scatter(off_events['timestamp_s'],
                   off_events['x'],
                   c='red',
                   marker='o',
                   s=1,
                   alpha=0.3,
                   label='OFF events')

    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('X (pixels)', fontsize=10)
    ax3.set_title('Time vs X Projection', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Time vs Y projection ---
    ax4 = fig.add_subplot(2, 2, 4)

    if len(on_events) > 0:
        ax4.scatter(on_events['timestamp_s'],
                   on_events['y'],
                   c='blue',
                   marker='o',
                   s=1,
                   alpha=0.3,
                   label='ON events')

    if len(off_events) > 0:
        ax4.scatter(off_events['timestamp_s'],
                   off_events['y'],
                   c='red',
                   marker='o',
                   s=1,
                   alpha=0.3,
                   label='OFF events')

    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Y (pixels)', fontsize=10)
    ax4.set_title('Time vs Y Projection', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_3d_density(events: pd.DataFrame,
                   time_bins: int = 50,
                   spatial_bins: int = 32,
                   save_path: Optional[str] = None):
    """
    Create 3D density visualization (voxel-like).

    Args:
        events: DataFrame with events
        time_bins: Number of temporal bins
        spatial_bins: Number of spatial bins for X and Y
        save_path: Optional save path
    """
    # Create 3D histogram
    t = events['timestamp_s'].values
    x = events['x'].values
    y = events['y'].values

    # Compute 3D histogram
    H, edges = np.histogramdd(
        np.column_stack([t, x, y]),
        bins=[time_bins, spatial_bins, spatial_bins]
    )

    # Create figure
    fig = plt.figure(figsize=(16, 6))

    # Plot 1: Time-X density
    ax1 = fig.add_subplot(1, 3, 1)
    density_tx = H.sum(axis=2)  # Sum over Y
    im1 = ax1.imshow(density_tx.T,
                     aspect='auto',
                     origin='lower',
                     cmap='hot',
                     interpolation='nearest')
    ax1.set_xlabel('Time bins', fontsize=10)
    ax1.set_ylabel('X bins', fontsize=10)
    ax1.set_title('Event Density: Time vs X', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Event count')

    # Plot 2: Time-Y density
    ax2 = fig.add_subplot(1, 3, 2)
    density_ty = H.sum(axis=1)  # Sum over X
    im2 = ax2.imshow(density_ty.T,
                     aspect='auto',
                     origin='lower',
                     cmap='hot',
                     interpolation='nearest')
    ax2.set_xlabel('Time bins', fontsize=10)
    ax2.set_ylabel('Y bins', fontsize=10)
    ax2.set_title('Event Density: Time vs Y', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Event count')

    # Plot 3: X-Y density
    ax3 = fig.add_subplot(1, 3, 3)
    density_xy = H.sum(axis=0)  # Sum over time
    im3 = ax3.imshow(density_xy.T,
                     aspect='equal',
                     origin='lower',
                     cmap='hot',
                     interpolation='nearest')
    ax3.set_xlabel('X bins', fontsize=10)
    ax3.set_ylabel('Y bins', fontsize=10)
    ax3.set_title('Event Density: X vs Y (all time)', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Event count')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved density figure to {save_path}")

    plt.show()


def plot_temporal_slices(events: pd.DataFrame,
                        num_slices: int = 6,
                        save_path: Optional[str] = None):
    """
    Plot spatial distribution at different time slices.

    Args:
        events: DataFrame with events
        num_slices: Number of time slices to show
        save_path: Optional save path
    """
    # Divide time into slices
    t_min = events['timestamp_s'].min()
    t_max = events['timestamp_s'].max()
    time_edges = np.linspace(t_min, t_max, num_slices + 1)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i in range(num_slices):
        t_start = time_edges[i]
        t_end = time_edges[i + 1]

        # Filter events in this time slice
        slice_events = events[
            (events['timestamp_s'] >= t_start) &
            (events['timestamp_s'] < t_end)
        ]

        on_events = slice_events[slice_events['polarity'] == 1]
        off_events = slice_events[slice_events['polarity'] == 0]

        ax = axes[i]

        if len(on_events) > 0:
            ax.scatter(on_events['x'], on_events['y'],
                      c='blue', s=1, alpha=0.5, label='ON')

        if len(off_events) > 0:
            ax.scatter(off_events['x'], off_events['y'],
                      c='red', s=1, alpha=0.5, label='OFF')

        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_title(f't = [{t_start:.2f}, {t_end:.2f}]s\n{len(slice_events)} events',
                    fontsize=10)
        ax.set_xlim(0, 128)
        ax.set_ylim(0, 128)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Spatial Distribution Across Time Slices',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved temporal slices to {save_path}")

    plt.show()


def main():
    """Main visualization script."""
    parser = argparse.ArgumentParser(description="3D Event Visualization for DVS128")
    parser.add_argument('csv_path', type=str, help='Path to CSV event file')
    parser.add_argument('--max_events', type=int, default=10000,
                       help='Maximum events to visualize (default: 10000)')
    parser.add_argument('--time_start', type=float, default=None,
                       help='Start time in seconds (default: from beginning)')
    parser.add_argument('--time_end', type=float, default=None,
                       help='End time in seconds (default: to end)')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='Directory to save plots')
    parser.add_argument('--all', action='store_true',
                       help='Generate all visualization types')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine time window
    time_window = None
    if args.time_start is not None or args.time_end is not None:
        time_window = (
            args.time_start if args.time_start is not None else 0,
            args.time_end if args.time_end is not None else float('inf')
        )

    # Load events
    events = load_events_from_csv(
        args.csv_path,
        max_events=args.max_events,
        time_window=time_window
    )

    # Get base filename for saving
    csv_name = Path(args.csv_path).stem

    # 3D scatter plots
    print("\nGenerating 3D scatter plots...")
    plot_3d_events(
        events,
        title=f"DVS128 Events: {csv_name}",
        save_path=str(save_dir / f"{csv_name}_3d_scatter.png")
    )

    if args.all:
        # Density plots
        print("\nGenerating density plots...")
        plot_3d_density(
            events,
            save_path=str(save_dir / f"{csv_name}_density.png")
        )

        # Temporal slices
        print("\nGenerating temporal slices...")
        plot_temporal_slices(
            events,
            save_path=str(save_dir / f"{csv_name}_temporal_slices.png")
        )

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
