#!/usr/bin/env python3
"""
Quick visualization script for your DVS128 data.
Pre-configured with your data paths.

Usage:
    python visualize_my_data.py                    # Visualize sine-300mV (default)
    python visualize_my_data.py burst-200mV        # Visualize specific file
    python visualize_my_data.py sine-400mV --all   # All visualization types
"""

import sys
from pathlib import Path
from visualize_events_3d import load_events_from_csv, plot_3d_events, plot_3d_density, plot_temporal_slices

# Your data directory
DATA_DIR = Path("~/Desktop/QuantumNetwork/data/raw_truncated").expanduser()

# Configuration
TIME_START = 0.0      # Start time in seconds
TIME_END = 2.0        # End time (visualize 2 seconds)
MAX_EVENTS = 10000    # Number of events to plot

# Parse command line arguments
if len(sys.argv) > 1:
    # User specified a file
    file_prefix = sys.argv[1].replace('.csv', '')  # Remove .csv if provided
    csv_file = DATA_DIR / f"{file_prefix}.csv"
    show_all = '--all' in sys.argv
else:
    # Default: use sine-300mV
    csv_file = DATA_DIR / "sine-300mV.csv"
    show_all = False

# Check if file exists
if not csv_file.exists():
    print(f"ERROR: File not found: {csv_file}")
    print(f"\nAvailable files in {DATA_DIR}:")
    for f in sorted(DATA_DIR.glob("*.csv")):
        print(f"  - {f.stem}")
    print(f"\nUsage: python visualize_my_data.py <filename>")
    print(f"Example: python visualize_my_data.py burst-200mV")
    sys.exit(1)

print(f"Visualizing: {csv_file.name}")
print(f"Time window: {TIME_START} to {TIME_END} seconds")
print(f"Max events: {MAX_EVENTS:,}")
print()

# Load events
events = load_events_from_csv(
    str(csv_file),
    max_events=MAX_EVENTS,
    time_window=(TIME_START, TIME_END)
)

# Main 3D visualization
print("\nCreating 3D scatter plots...")
plot_3d_events(
    events,
    title=f"DVS128 Events: {csv_file.stem}"
)

# Additional visualizations if requested
if show_all:
    print("\nCreating density plots...")
    plot_3d_density(events)

    print("\nCreating temporal slices...")
    plot_temporal_slices(events, num_slices=6)

print("\nDone! Close the plot windows when finished.")
print("\nTip: Run with --all flag for more visualizations:")
print(f"  python visualize_my_data.py {csv_file.stem} --all")
