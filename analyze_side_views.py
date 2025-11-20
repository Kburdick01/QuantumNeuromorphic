#!/usr/bin/env python3
"""
DVS128 Side View Analysis
Shows Time vs X and Time vs Y projections across multiple time windows.

Focuses on the useful 2D projections that clearly show helix patterns.
Generates multiple time windows to see pattern evolution.

Author: QTCR-Net Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple
import matplotlib
matplotlib.use('Agg')  # Save to files


def load_events_window(csv_path: str,
                       start_ms: float,
                       duration_ms: float,
                       max_events: int = None) -> pd.DataFrame:
    """Load events from a specific time window."""
    print(f"  Loading {start_ms:.0f}-{start_ms+duration_ms:.0f}ms...", end='')

    # Read CSV
    df = pd.read_csv(csv_path, comment='#')
    df.columns = df.columns.str.strip().str.lower()

    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'timestamp_us'}, inplace=True)

    # Convert to milliseconds
    df['timestamp_ms'] = df['timestamp_us'] / 1000.0
    df['timestamp_ms'] = df['timestamp_ms'] - df['timestamp_ms'].min()

    # Filter time window
    end_ms = start_ms + duration_ms
    df = df[(df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] < end_ms)].copy()

    # Limit events
    if max_events and len(df) > max_events:
        indices = np.linspace(0, len(df)-1, max_events, dtype=int)
        df = df.iloc[indices]

    # Map polarity
    if df['polarity'].min() < 0:
        df['polarity'] = (df['polarity'] + 1) // 2

    print(f" {len(df):,} events")
    return df[['timestamp_ms', 'x', 'y', 'polarity']]


def plot_side_views_grid(csv_path: str,
                         time_windows: List[Tuple[float, float]],
                         max_events_per_window: int = 100000,
                         save_path: str = None):
    """
    Create grid of side views across multiple time windows.

    Args:
        csv_path: Path to CSV file
        time_windows: List of (start_ms, duration_ms) tuples
        max_events_per_window: Max events per window
        save_path: Where to save the plot
    """
    csv_name = Path(csv_path).stem
    num_windows = len(time_windows)

    # Create figure with 2 columns (Time vs X, Time vs Y) and N rows (one per window)
    fig, axes = plt.subplots(num_windows, 2, figsize=(16, 4*num_windows))

    # Handle single window case
    if num_windows == 1:
        axes = axes.reshape(1, -1)

    print(f"\nProcessing {csv_name}:")

    for i, (start_ms, duration_ms) in enumerate(time_windows):
        # Load events for this window
        events = load_events_window(csv_path, start_ms, duration_ms, max_events_per_window)

        if len(events) == 0:
            print(f"  WARNING: No events in window {i+1}")
            continue

        # Separate polarities
        on_events = events[events['polarity'] == 1]
        off_events = events[events['polarity'] == 0]

        # Plot 1: Time vs X
        ax_x = axes[i, 0]
        if len(on_events) > 0:
            ax_x.scatter(on_events['timestamp_ms'], on_events['x'],
                        c='blue', s=0.5, alpha=0.4, label='ON', rasterized=True)
        if len(off_events) > 0:
            ax_x.scatter(off_events['timestamp_ms'], off_events['x'],
                        c='red', s=0.5, alpha=0.4, label='OFF', rasterized=True)

        ax_x.set_xlabel('Time (ms)', fontsize=11)
        ax_x.set_ylabel('X (pixels)', fontsize=11)
        ax_x.set_title(f'Time vs X | Window {i+1}: [{start_ms:.0f}, {start_ms+duration_ms:.0f}]ms | {len(events):,} events',
                      fontsize=12, fontweight='bold')
        ax_x.set_ylim(0, 128)
        ax_x.grid(True, alpha=0.3)
        if i == 0:
            ax_x.legend(loc='upper right', fontsize=9)

        # Plot 2: Time vs Y
        ax_y = axes[i, 1]
        if len(on_events) > 0:
            ax_y.scatter(on_events['timestamp_ms'], on_events['y'],
                        c='blue', s=0.5, alpha=0.4, label='ON', rasterized=True)
        if len(off_events) > 0:
            ax_y.scatter(off_events['timestamp_ms'], off_events['y'],
                        c='red', s=0.5, alpha=0.4, label='OFF', rasterized=True)

        ax_y.set_xlabel('Time (ms)', fontsize=11)
        ax_y.set_ylabel('Y (pixels)', fontsize=11)
        ax_y.set_title(f'Time vs Y | Window {i+1}: [{start_ms:.0f}, {start_ms+duration_ms:.0f}]ms | {len(events):,} events',
                      fontsize=12, fontweight='bold')
        ax_y.set_ylim(0, 128)
        ax_y.grid(True, alpha=0.3)
        if i == 0:
            ax_y.legend(loc='upper right', fontsize=9)

    plt.suptitle(f'Side View Analysis: {csv_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to: {save_path}")
    else:
        plt.show()


def generate_time_windows(strategy: str,
                          file_duration_ms: float = 120000) -> List[Tuple[float, float]]:
    """
    Generate time windows for analysis.

    Strategies:
        'sequential': Sequential non-overlapping 310ms windows
        'long': Longer windows (500ms, 1000ms, 2000ms)
        'varied': Different durations starting at same point
        'samples': Sample windows throughout recording
    """
    if strategy == 'sequential':
        # Sequential 310ms windows (like jAER rolling window)
        window_duration = 310.0
        num_windows = min(6, int(file_duration_ms / window_duration))
        return [(i * window_duration, window_duration) for i in range(num_windows)]

    elif strategy == 'long':
        # Different long durations starting from same point
        start = 1000.0
        return [
            (start, 500.0),   # 500ms
            (start, 1000.0),  # 1 second
            (start, 2000.0),  # 2 seconds
            (start, 5000.0),  # 5 seconds
        ]

    elif strategy == 'varied':
        # Same start, different durations
        start = 1000.0
        return [
            (start, 100.0),
            (start, 200.0),
            (start, 310.0),
            (start, 500.0),
            (start, 1000.0),
        ]

    elif strategy == 'samples':
        # Sample windows throughout the recording
        duration = 310.0
        # Take samples from beginning, quarter, half, three-quarter, end
        positions = [0.05, 0.25, 0.5, 0.75, 0.9]  # Fractions of file duration
        return [(file_duration_ms * pos, duration) for pos in positions]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    """Main side view analysis script."""
    parser = argparse.ArgumentParser(description="DVS128 Side View Analysis")
    parser.add_argument('csv_path', type=str, help='Path to CSV file')
    parser.add_argument('--strategy', type=str, default='sequential',
                       choices=['sequential', 'long', 'varied', 'samples'],
                       help='Time window strategy (default: sequential)')
    parser.add_argument('--max_events', type=int, default=100000,
                       help='Max events per window (default: 100000)')
    parser.add_argument('--duration', type=float, default=120000,
                       help='File duration in ms (default: 120000)')
    parser.add_argument('--save_dir', type=str, default='./side_views',
                       help='Directory to save plots')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    csv_name = Path(args.csv_path).stem

    # Generate time windows
    time_windows = generate_time_windows(args.strategy, args.duration)

    print("="*70)
    print(f"Side View Analysis: {csv_name}")
    print("="*70)
    print(f"Strategy: {args.strategy}")
    print(f"Windows: {len(time_windows)}")
    for i, (start, duration) in enumerate(time_windows):
        print(f"  {i+1}. [{start:.0f}, {start+duration:.0f}]ms ({duration:.0f}ms)")

    # Generate plot
    save_path = save_dir / f"{csv_name}_{args.strategy}.png"
    plot_side_views_grid(
        args.csv_path,
        time_windows,
        max_events_per_window=args.max_events,
        save_path=str(save_path)
    )

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"View: xdg-open {save_path}")


if __name__ == '__main__':
    main()
