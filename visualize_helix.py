#!/usr/bin/env python3
"""
DVS128 Helix Visualization - Mimics jAER SpaceTimeRollingEventDisplay

Creates rolling time window visualizations similar to jAER's display method.
Optimized for viewing helical/spiral patterns in DVS data.

Based on jAER parameters:
- Time window: ~310ms rolling window
- Time slice: 1.02ms (1,020 microseconds)
- Event-based rendering showing temporal structure

Author: QTCR-Net Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
from matplotlib.animation import FuncAnimation
from typing import Optional


def load_events_rolling(csv_path: str,
                       start_time_ms: float = 0.0,
                       window_duration_ms: float = 310.0,
                       max_events: Optional[int] = 500000) -> pd.DataFrame:
    """
    Load events for rolling window visualization.

    Args:
        csv_path: Path to CSV file
        start_time_ms: Start time in milliseconds
        window_duration_ms: Rolling window duration (default 310ms like jAER)
        max_events: Maximum events to load

    Returns:
        DataFrame with events in the time window
    """
    print(f"Loading events from {csv_path}...")

    # Read CSV (skip comment lines)
    df = pd.read_csv(csv_path, comment='#')

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename timestamp column
    if 'timestamp' in df.columns and 'timestamp_us' not in df.columns:
        df.rename(columns={'timestamp': 'timestamp_us'}, inplace=True)

    # Convert to milliseconds
    df['timestamp_ms'] = df['timestamp_us'] / 1000.0

    # Normalize to start at 0
    df['timestamp_ms'] = df['timestamp_ms'] - df['timestamp_ms'].min()

    # Filter by time window
    end_time_ms = start_time_ms + window_duration_ms
    df_window = df[(df['timestamp_ms'] >= start_time_ms) &
                   (df['timestamp_ms'] < end_time_ms)].copy()

    # Limit events if needed
    if max_events and len(df_window) > max_events:
        # Sample uniformly
        indices = np.linspace(0, len(df_window)-1, max_events, dtype=int)
        df_window = df_window.iloc[indices]

    # Map polarity to 0/1 if needed
    if df_window['polarity'].min() < 0:
        df_window['polarity'] = (df_window['polarity'] + 1) // 2

    print(f"  Loaded {len(df_window):,} events")
    print(f"  Time range: {df_window['timestamp_ms'].min():.2f} to {df_window['timestamp_ms'].max():.2f} ms")
    print(f"  Event rate: {len(df_window)/window_duration_ms:.1f} K events/ms")

    return df_window[['timestamp_ms', 'x', 'y', 'polarity']]


def plot_helix_view(events: pd.DataFrame,
                   time_slice_ms: float = 1.02,
                   title: str = "DVS128 Helix Pattern",
                   view_angle: tuple = (30, 45),
                   save_path: Optional[str] = None):
    """
    Create helix visualization similar to jAER SpaceTimeRollingEventDisplay.

    Args:
        events: DataFrame with events
        time_slice_ms: Time slice for coloring (default 1.02ms like jAER)
        title: Plot title
        view_angle: (elevation, azimuth) for 3D view
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(16, 12))

    # Main 3D helix view
    ax_main = fig.add_subplot(2, 2, 1, projection='3d')

    # Separate ON and OFF events
    on_events = events[events['polarity'] == 1]
    off_events = events[events['polarity'] == 0]

    # Color by time slice (creates layers in the helix)
    if len(on_events) > 0:
        # Normalize time for color mapping
        t_norm = (on_events['timestamp_ms'] - on_events['timestamp_ms'].min()) / time_slice_ms

        scatter_on = ax_main.scatter(on_events['timestamp_ms'],
                                     on_events['x'],
                                     on_events['y'],
                                     c=t_norm % 10,  # Color cycles every 10 time slices
                                     cmap='Blues',
                                     marker='o',
                                     s=3,
                                     alpha=0.6,
                                     label='ON events')

    if len(off_events) > 0:
        t_norm = (off_events['timestamp_ms'] - off_events['timestamp_ms'].min()) / time_slice_ms

        scatter_off = ax_main.scatter(off_events['timestamp_ms'],
                                      off_events['x'],
                                      off_events['y'],
                                      c=t_norm % 10,
                                      cmap='Reds',
                                      marker='o',
                                      s=3,
                                      alpha=0.6,
                                      label='OFF events')

    ax_main.set_xlabel('Time (ms)', fontsize=11)
    ax_main.set_ylabel('X (pixels)', fontsize=11)
    ax_main.set_zlabel('Y (pixels)', fontsize=11)
    ax_main.set_title('Helix View (Time × X × Y)', fontsize=13, fontweight='bold')
    ax_main.legend(loc='upper right')
    ax_main.view_init(elev=view_angle[0], azim=view_angle[1])
    ax_main.grid(True, alpha=0.3)

    # Side view 1: Time vs X
    ax2 = fig.add_subplot(2, 2, 2)
    if len(on_events) > 0:
        ax2.scatter(on_events['timestamp_ms'], on_events['x'],
                   c='blue', s=1, alpha=0.4, label='ON')
    if len(off_events) > 0:
        ax2.scatter(off_events['timestamp_ms'], off_events['x'],
                   c='red', s=1, alpha=0.4, label='OFF')
    ax2.set_xlabel('Time (ms)', fontsize=10)
    ax2.set_ylabel('X (pixels)', fontsize=10)
    ax2.set_title('Side View: Time vs X', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Side view 2: Time vs Y
    ax3 = fig.add_subplot(2, 2, 3)
    if len(on_events) > 0:
        ax3.scatter(on_events['timestamp_ms'], on_events['y'],
                   c='blue', s=1, alpha=0.4, label='ON')
    if len(off_events) > 0:
        ax3.scatter(off_events['timestamp_ms'], off_events['y'],
                   c='red', s=1, alpha=0.4, label='OFF')
    ax3.set_xlabel('Time (ms)', fontsize=10)
    ax3.set_ylabel('Y (pixels)', fontsize=10)
    ax3.set_title('Side View: Time vs Y', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Top view: X vs Y (spatial)
    ax4 = fig.add_subplot(2, 2, 4)
    if len(on_events) > 0:
        ax4.scatter(on_events['x'], on_events['y'],
                   c='blue', s=2, alpha=0.3, label='ON')
    if len(off_events) > 0:
        ax4.scatter(off_events['x'], off_events['y'],
                   c='red', s=2, alpha=0.3, label='OFF')
    ax4.set_xlabel('X (pixels)', fontsize=10)
    ax4.set_ylabel('Y (pixels)', fontsize=10)
    ax4.set_title('Top View: X vs Y', fontsize=11, fontweight='bold')
    ax4.set_xlim(0, 128)
    ax4.set_ylim(0, 128)
    ax4.set_aspect('equal')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_temporal_slices_helix(events: pd.DataFrame,
                               slice_duration_ms: float = 1.02,
                               num_slices: int = 6,
                               save_path: Optional[str] = None):
    """
    Show helix pattern broken into time slices.

    Args:
        events: DataFrame with events
        slice_duration_ms: Duration of each time slice (default 1.02ms like jAER)
        num_slices: Number of slices to show
        save_path: Optional save path
    """
    t_min = events['timestamp_ms'].min()
    t_max = events['timestamp_ms'].max()

    # Create slices
    slice_edges = np.linspace(t_min, t_max, num_slices + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i in range(num_slices):
        slice_start = slice_edges[i]
        slice_end = slice_edges[i + 1]

        # Filter events in this slice
        slice_events = events[(events['timestamp_ms'] >= slice_start) &
                             (events['timestamp_ms'] < slice_end)]

        on = slice_events[slice_events['polarity'] == 1]
        off = slice_events[slice_events['polarity'] == 0]

        ax = axes[i]

        if len(on) > 0:
            ax.scatter(on['x'], on['y'], c='blue', s=3, alpha=0.5, label='ON')
        if len(off) > 0:
            ax.scatter(off['x'], off['y'], c='red', s=3, alpha=0.5, label='OFF')

        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_title(f'Slice {i+1}: t=[{slice_start:.1f}, {slice_end:.1f}] ms\n{len(slice_events):,} events',
                    fontsize=10)
        ax.set_xlim(0, 128)
        ax.set_ylim(0, 128)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f'Helix Pattern: Temporal Slices ({slice_duration_ms:.2f}ms each)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Main helix visualization script."""
    parser = argparse.ArgumentParser(description="DVS128 Helix Visualization (jAER-style)")
    parser.add_argument('csv_path', type=str, help='Path to CSV file')
    parser.add_argument('--start_ms', type=float, default=0.0,
                       help='Start time in milliseconds (default: 0)')
    parser.add_argument('--window_ms', type=float, default=310.0,
                       help='Rolling window duration in ms (default: 310, like jAER)')
    parser.add_argument('--slice_ms', type=float, default=1.02,
                       help='Time slice in ms (default: 1.02, like jAER)')
    parser.add_argument('--max_events', type=int, default=500000,
                       help='Max events to visualize (default: 500000)')
    parser.add_argument('--elevation', type=int, default=30,
                       help='3D view elevation angle (default: 30)')
    parser.add_argument('--azimuth', type=int, default=45,
                       help='3D view azimuth angle (default: 45)')
    parser.add_argument('--slices', action='store_true',
                       help='Show temporal slices view')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='Directory to save plots')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_name = Path(args.csv_path).stem

    # Load events
    events = load_events_rolling(
        args.csv_path,
        start_time_ms=args.start_ms,
        window_duration_ms=args.window_ms,
        max_events=args.max_events
    )

    if len(events) == 0:
        print("No events in time window!")
        return

    # Statistics
    print(f"\nEvent Statistics:")
    print(f"  Total events: {len(events):,}")
    print(f"  ON events: {(events['polarity']==1).sum():,}")
    print(f"  OFF events: {(events['polarity']==0).sum():,}")
    print(f"  Duration: {events['timestamp_ms'].max() - events['timestamp_ms'].min():.2f} ms")
    print(f"  Event rate: {len(events) / args.window_ms:.1f} K events/ms")

    # Main helix visualization
    print(f"\nCreating helix visualization...")
    plot_helix_view(
        events,
        time_slice_ms=args.slice_ms,
        title=f"DVS128 Helix Pattern: {csv_name} [{args.start_ms:.1f}-{args.start_ms+args.window_ms:.1f} ms]",
        view_angle=(args.elevation, args.azimuth),
        save_path=str(save_dir / f"{csv_name}_helix.png")
    )

    # Temporal slices if requested
    if args.slices:
        print(f"\nCreating temporal slices...")
        plot_temporal_slices_helix(
            events,
            slice_duration_ms=args.slice_ms,
            save_path=str(save_dir / f"{csv_name}_slices.png")
        )

    print("\nDone! Rotate the 3D plot to see the helix from different angles.")


if __name__ == '__main__':
    main()
