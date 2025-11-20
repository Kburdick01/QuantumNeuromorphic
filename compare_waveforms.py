#!/usr/bin/env python3
"""
DVS128 Waveform Comparison - Time vs X Only
Shows clear Time vs X side views and comparison plots to distinguish waveforms.

Author: QTCR-Net Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')


def load_events_window(csv_path: str,
                       start_ms: float,
                       duration_ms: float,
                       max_events: int = None) -> pd.DataFrame:
    """Load events from a specific time window."""
    df = pd.read_csv(csv_path, comment='#')
    df.columns = df.columns.str.strip().str.lower()

    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'timestamp_us'}, inplace=True)

    df['timestamp_ms'] = df['timestamp_us'] / 1000.0
    df['timestamp_ms'] = df['timestamp_ms'] - df['timestamp_ms'].min()

    end_ms = start_ms + duration_ms
    df = df[(df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] < end_ms)].copy()

    if max_events and len(df) > max_events:
        indices = np.linspace(0, len(df)-1, max_events, dtype=int)
        df = df.iloc[indices]

    if df['polarity'].min() < 0:
        df['polarity'] = (df['polarity'] + 1) // 2

    return df[['timestamp_ms', 'x', 'y', 'polarity']]


def plot_single_file_time_vs_x(csv_path: str,
                                time_windows: list,
                                max_events: int = 100000,
                                save_path: str = None):
    """
    Plot Time vs X for multiple windows (single column layout).

    Args:
        csv_path: Path to CSV file
        time_windows: List of (start_ms, duration_ms)
        max_events: Max events per window
        save_path: Save path
    """
    csv_name = Path(csv_path).stem
    num_windows = len(time_windows)

    # Single column layout
    fig, axes = plt.subplots(num_windows, 1, figsize=(14, 4*num_windows))

    if num_windows == 1:
        axes = [axes]

    print(f"\nProcessing {csv_name}:")

    for i, (start_ms, duration_ms) in enumerate(time_windows):
        print(f"  Window {i+1}: [{start_ms:.0f}-{start_ms+duration_ms:.0f}]ms...", end='')

        events = load_events_window(csv_path, start_ms, duration_ms, max_events)

        if len(events) == 0:
            print(" no events!")
            continue

        on_events = events[events['polarity'] == 1]
        off_events = events[events['polarity'] == 0]

        ax = axes[i]

        # Plot events
        if len(on_events) > 0:
            ax.scatter(on_events['timestamp_ms'], on_events['x'],
                      c='blue', s=0.5, alpha=0.5, label='ON', rasterized=True)
        if len(off_events) > 0:
            ax.scatter(off_events['timestamp_ms'], off_events['x'],
                      c='red', s=0.5, alpha=0.5, label='OFF', rasterized=True)

        ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('X (pixels)', fontsize=12, fontweight='bold')
        ax.set_title(f'{csv_name} | Window {i+1}: [{start_ms:.0f}, {start_ms+duration_ms:.0f}]ms | {len(events):,} events',
                    fontsize=13, fontweight='bold')
        ax.set_ylim(0, 128)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)

        print(f" {len(events):,} events")

    plt.suptitle(f'Time vs X Analysis: {csv_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_waveform_comparison(csv_files: dict,
                             start_ms: float = 1000.0,
                             duration_ms: float = 500.0,
                             max_events: int = 100000,
                             save_path: str = None):
    """
    Compare all 4 waveforms in one plot (4 rows, one per waveform).

    Args:
        csv_files: Dict mapping waveform names to CSV paths
        start_ms: Start time
        duration_ms: Window duration
        max_events: Max events per waveform
        save_path: Save path
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 16))

    waveform_order = ['sine', 'square', 'triangle', 'burst']

    print(f"\nComparing waveforms [{start_ms:.0f}-{start_ms+duration_ms:.0f}]ms:")

    for i, waveform in enumerate(waveform_order):
        if waveform not in csv_files:
            print(f"  {waveform}: NOT FOUND")
            continue

        csv_path = csv_files[waveform]
        print(f"  {waveform}: {Path(csv_path).stem}...", end='')

        events = load_events_window(csv_path, start_ms, duration_ms, max_events)

        if len(events) == 0:
            print(" no events!")
            continue

        on_events = events[events['polarity'] == 1]
        off_events = events[events['polarity'] == 0]

        ax = axes[i]

        # Plot with distinct colors per waveform
        colors = {
            'sine': ('royalblue', 'lightcoral'),
            'square': ('darkgreen', 'orange'),
            'triangle': ('purple', 'gold'),
            'burst': ('darkred', 'pink')
        }

        on_color, off_color = colors[waveform]

        if len(on_events) > 0:
            ax.scatter(on_events['timestamp_ms'], on_events['x'],
                      c=on_color, s=1.0, alpha=0.6, label='ON', rasterized=True)
        if len(off_events) > 0:
            ax.scatter(off_events['timestamp_ms'], off_events['x'],
                      c=off_color, s=1.0, alpha=0.6, label='OFF', rasterized=True)

        ax.set_ylabel('X (pixels)', fontsize=12, fontweight='bold')
        ax.set_title(f'{waveform.upper()} | {len(events):,} events | Pattern: {_describe_pattern(waveform)}',
                    fontsize=13, fontweight='bold')
        ax.set_ylim(0, 128)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        # Only bottom plot gets x-label
        if i == 3:
            ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        else:
            ax.set_xticklabels([])

        print(f" {len(events):,} events")

    plt.suptitle(f'Waveform Comparison: Time vs X | [{start_ms:.0f}, {start_ms+duration_ms:.0f}]ms',
                fontsize=18, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")


def _describe_pattern(waveform: str) -> str:
    """Describe expected pattern for each waveform."""
    patterns = {
        'sine': 'Smooth curved stripes, gradual transitions',
        'square': 'Sharp vertical edges, abrupt transitions',
        'triangle': 'Linear diagonal stripes, constant slope',
        'burst': 'Random scattered events, no clear pattern'
    }
    return patterns.get(waveform, 'Unknown pattern')


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(description="DVS128 Waveform Comparison (Time vs X only)")
    parser.add_argument('--data_dir', type=str,
                       default='~/Desktop/QuantumNetwork/data/raw_truncated',
                       help='Directory with CSV files')
    parser.add_argument('--voltage', type=str, default='300mV',
                       help='Voltage to analyze (default: 300mV)')
    parser.add_argument('--start_ms', type=float, default=1000.0,
                       help='Start time in ms (default: 1000)')
    parser.add_argument('--duration_ms', type=float, default=500.0,
                       help='Window duration in ms (default: 500)')
    parser.add_argument('--max_events', type=int, default=100000,
                       help='Max events per window (default: 100000)')
    parser.add_argument('--save_dir', type=str, default='./waveform_comparison',
                       help='Save directory')
    parser.add_argument('--mode', type=str, default='comparison',
                       choices=['comparison', 'individual'],
                       help='comparison: all 4 waveforms, individual: separate files')

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Find CSV files for this voltage
    csv_files = {}
    for waveform in ['sine', 'square', 'triangle', 'burst']:
        pattern = f"{waveform}-{args.voltage}.csv"
        matches = list(data_dir.glob(pattern))
        if matches:
            csv_files[waveform] = str(matches[0])

    print("="*70)
    print(f"Waveform Comparison Analysis")
    print("="*70)
    print(f"Voltage: {args.voltage}")
    print(f"Window: [{args.start_ms:.0f}, {args.start_ms + args.duration_ms:.0f}]ms")
    print(f"Files found: {len(csv_files)}/4")
    for waveform, path in csv_files.items():
        print(f"  {waveform}: {Path(path).name}")

    if len(csv_files) == 0:
        print("\nERROR: No CSV files found!")
        return

    if args.mode == 'comparison':
        # Single comparison plot with all 4 waveforms
        save_path = save_dir / f'comparison_{args.voltage}_{args.start_ms:.0f}ms_{args.duration_ms:.0f}ms.png'
        plot_waveform_comparison(
            csv_files,
            start_ms=args.start_ms,
            duration_ms=args.duration_ms,
            max_events=args.max_events,
            save_path=str(save_path)
        )

    else:
        # Individual files
        time_windows = [(args.start_ms, args.duration_ms)]
        for waveform, csv_path in csv_files.items():
            save_path = save_dir / f'{waveform}_{args.voltage}_{args.start_ms:.0f}ms.png'
            plot_single_file_time_vs_x(
                csv_path,
                time_windows,
                max_events=args.max_events,
                save_path=str(save_path)
            )

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"Saved to: {save_dir}/")
    print(f"View: xdg-open {save_dir}/")


if __name__ == '__main__':
    main()
