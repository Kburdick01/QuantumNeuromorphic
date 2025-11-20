#!/usr/bin/env python3
"""
Generate side view analyses for all CSV files.
Creates comprehensive Time vs X and Time vs Y visualizations.

This is what you want! Shows the helix patterns clearly.

Usage:
    python show_all_side_views.py                  # All files, sequential windows
    python show_all_side_views.py --strategy long  # Long duration windows
    python show_all_side_views.py --files sine     # Only sine waveforms
"""

import sys
from pathlib import Path
from analyze_side_views import generate_time_windows, plot_side_views_grid

# Your data directory
DATA_DIR = Path("~/Desktop/QuantumNetwork/data/raw_truncated").expanduser()

# Settings
STRATEGY = 'sequential'  # Change to: 'sequential', 'long', 'varied', 'samples'
MAX_EVENTS = 100000      # Events per window
FILE_DURATION_MS = 120000  # Your files are 120 seconds = 120,000ms

# Parse command line
filter_pattern = None
if '--files' in sys.argv:
    idx = sys.argv.index('--files')
    filter_pattern = sys.argv[idx + 1]

if '--strategy' in sys.argv:
    idx = sys.argv.index('--strategy')
    STRATEGY = sys.argv[idx + 1]

# Find CSV files
csv_files = sorted(DATA_DIR.glob("*.csv"))

if filter_pattern:
    csv_files = [f for f in csv_files if filter_pattern in f.stem]

if len(csv_files) == 0:
    print(f"ERROR: No CSV files found!")
    sys.exit(1)

# Create output directory
output_dir = Path("./side_views")
output_dir.mkdir(exist_ok=True)

print("="*70)
print("Side View Analysis for All Files")
print("="*70)
print(f"Strategy: {STRATEGY}")
print(f"Files to process: {len(csv_files)}")
print(f"Output: ./side_views/")
print()

# Strategy descriptions
strategy_info = {
    'sequential': "6 sequential 310ms windows (like jAER)",
    'long': "Multiple long durations (500ms, 1s, 2s, 5s)",
    'varied': "Same start point, different durations",
    'samples': "Sample windows throughout recording"
}
print(f"Strategy '{STRATEGY}': {strategy_info.get(STRATEGY, 'Custom')}")

# Generate time windows
time_windows = generate_time_windows(STRATEGY, FILE_DURATION_MS)
print(f"\nTime windows ({len(time_windows)} total):")
for i, (start, duration) in enumerate(time_windows):
    print(f"  {i+1}. [{start:.0f}, {start+duration:.0f}]ms ({duration:.0f}ms)")

print("\n" + "="*70)

# Process each file
for i, csv_file in enumerate(csv_files):
    print(f"\n[{i+1}/{len(csv_files)}] Processing {csv_file.stem}...")

    save_path = output_dir / f"{csv_file.stem}_{STRATEGY}.png"

    plot_side_views_grid(
        str(csv_file),
        time_windows,
        max_events_per_window=MAX_EVENTS,
        save_path=str(save_path)
    )

# Summary
print("\n" + "="*70)
print("ALL DONE!")
print("="*70)
print(f"\nGenerated {len(csv_files)} visualizations in: ./side_views/")
print(f"\nView all images:")
print(f"  nautilus ./side_views/")
print(f"\nView specific file:")
print(f"  xdg-open ./side_views/sine-300mV_{STRATEGY}.png")

print(f"\n" + "="*70)
print("Compare Waveforms:")
print("="*70)
print("Look at these files to see helix differences:")
for waveform in ['sine', 'square', 'triangle', 'burst']:
    matching = [f for f in csv_files if waveform in f.stem]
    if matching:
        print(f"  {waveform.upper():8s}: {matching[0].stem}_{STRATEGY}.png")

print(f"\n" + "="*70)
print("Next Steps:")
print("="*70)
print("1. Try different strategies:")
print("   python show_all_side_views.py --strategy long")
print("   python show_all_side_views.py --strategy varied")
print("   python show_all_side_views.py --strategy samples")
print("\n2. Filter to specific waveforms:")
print("   python show_all_side_views.py --files sine")
print("   python show_all_side_views.py --files burst")
print("\n3. Once you find good settings, adjust STRATEGY in the script")
