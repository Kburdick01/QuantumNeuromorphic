#!/usr/bin/env python3
"""
Quick helix visualization for your DVS128 data.
Pre-configured with jAER-like parameters.

Usage:
    python show_helix.py                    # Visualize sine-300mV (default)
    python show_helix.py sine-400mV         # Visualize specific file
    python show_helix.py burst-200mV --slices  # With temporal slices
"""

import sys
from pathlib import Path

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend, save to file

from visualize_helix import load_events_rolling, plot_helix_view, plot_temporal_slices_helix

# Your data directory
DATA_DIR = Path("~/Desktop/QuantumNetwork/data/raw_truncated").expanduser()

# jAER-like parameters
START_TIME_MS = 1000.0    # Start at 1 second (skip initial noise)
WINDOW_MS = 310.0         # 310ms rolling window (like jAER)
TIME_SLICE_MS = 1.02      # 1.02ms time slice (like jAER)
MAX_EVENTS = 500000       # 500K events max

# 3D view angles (adjust to see helix better)
ELEVATION = 20            # Viewing angle from above
AZIMUTH = 45              # Rotation angle

# Parse arguments
if len(sys.argv) > 1:
    file_prefix = sys.argv[1].replace('.csv', '')
    csv_file = DATA_DIR / f"{file_prefix}.csv"
    show_slices = '--slices' in sys.argv
else:
    csv_file = DATA_DIR / "sine-300mV.csv"
    show_slices = False

# Check file exists
if not csv_file.exists():
    print(f"ERROR: File not found: {csv_file}")
    print(f"\nAvailable files in {DATA_DIR}:")
    for f in sorted(DATA_DIR.glob("*.csv")):
        print(f"  - {f.stem}")
    print(f"\nUsage: python show_helix.py <filename>")
    sys.exit(1)

print("="*70)
print("DVS128 Helix Visualization (jAER-style)")
print("="*70)
print(f"\nFile: {csv_file.name}")
print(f"Time window: {WINDOW_MS}ms (starting at {START_TIME_MS}ms)")
print(f"Time slice: {TIME_SLICE_MS}ms")
print(f"Max events: {MAX_EVENTS:,}")
print()

# Load events
events = load_events_rolling(
    str(csv_file),
    start_time_ms=START_TIME_MS,
    window_duration_ms=WINDOW_MS,
    max_events=MAX_EVENTS
)

if len(events) == 0:
    print("ERROR: No events in time window!")
    print(f"Try adjusting START_TIME_MS in the script.")
    sys.exit(1)

# Statistics
print(f"\nEvent Statistics:")
print(f"  Total events: {len(events):,}")
print(f"  ON events: {(events['polarity']==1).sum():,}")
print(f"  OFF events: {(events['polarity']==0).sum():,}")
print(f"  Event rate: {len(events) / WINDOW_MS:.1f} K events/ms")

# Create output directory
output_dir = Path("./helix_visualizations")
output_dir.mkdir(exist_ok=True)

# Main helix view
print(f"\nCreating helix visualization...")
helix_file = output_dir / f"{csv_file.stem}_helix.png"

plot_helix_view(
    events,
    time_slice_ms=TIME_SLICE_MS,
    title=f"Helix Pattern: {csv_file.stem}",
    view_angle=(ELEVATION, AZIMUTH),
    save_path=str(helix_file)
)

print(f"✓ Saved helix plot to: {helix_file}")

# Temporal slices if requested
if show_slices:
    print(f"\nCreating temporal slices...")
    slices_file = output_dir / f"{csv_file.stem}_slices.png"

    plot_temporal_slices_helix(
        events,
        slice_duration_ms=TIME_SLICE_MS,
        num_slices=6,
        save_path=str(slices_file)
    )

    print(f"✓ Saved temporal slices to: {slices_file}")

print("\n" + "="*70)
print("DONE! Your visualizations are saved in: ./helix_visualizations/")
print("="*70)
print(f"\nOpen the images with:")
print(f"  xdg-open {helix_file}")
if show_slices:
    print(f"  xdg-open {output_dir / f'{csv_file.stem}_slices.png'}")

print("\nTips:")
print("  - View images: ls ./helix_visualizations/")
print("  - Run with --slices to see temporal evolution: python show_helix.py sine-300mV --slices")
print("  - Try different files: python show_helix.py burst-400mV")
print("  - Compare waveforms by running on all 4 types!")
