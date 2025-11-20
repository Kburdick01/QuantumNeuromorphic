#!/usr/bin/env python3
"""
Simple example of 3D event visualization.
Quick script to visualize your DVS128 events.
"""

from visualize_events_3d import load_events_from_csv, plot_3d_events, plot_3d_density, plot_temporal_slices
from pathlib import Path

# Configuration
CSV_PATH = "~/Desktop/QuantumNetwork/data/raw_truncated/sine-300mV.csv"  # ← Your CSV files
TIME_START = 0.0      # Start time in seconds
TIME_END = 1.0        # End time in seconds (visualize 1 second)
MAX_EVENTS = 10000    # Limit for performance

# Expand path
csv_path = Path(CSV_PATH).expanduser()

if not csv_path.exists():
    print(f"ERROR: CSV file not found at {csv_path}")
    print("Please update CSV_PATH in this script to point to your CSV file.")
    exit(1)

# Load events
print("Loading events...")
events = load_events_from_csv(
    str(csv_path),
    max_events=MAX_EVENTS,
    time_window=(TIME_START, TIME_END)
)

# Plot 3D visualization (Time vs X vs Y)
print("\nCreating 3D plots...")
plot_3d_events(
    events,
    title=f"DVS128 Events: {csv_path.name} [{TIME_START}-{TIME_END}s]"
)

# Optional: Density visualization
print("\nCreating density plots...")
plot_3d_density(events)

# Optional: Temporal slices
print("\nCreating temporal slices...")
plot_temporal_slices(events, num_slices=6)

print("\nDone! Close the plot windows when finished.")
