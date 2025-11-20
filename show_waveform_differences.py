#!/usr/bin/env python3
"""
Quick waveform comparison - see all 4 waveforms on one plot!

This is what you want - easy to see differences between:
- SINE: Smooth curved stripes
- SQUARE: Sharp vertical edges
- TRIANGLE: Linear diagonal stripes
- BURST: Random scattered events

Usage:
    python show_waveform_differences.py              # Default: 300mV
    python show_waveform_differences.py --voltage 400mV
    python show_waveform_differences.py --duration 1000  # Longer window
"""

import sys
from pathlib import Path
from compare_waveforms import plot_waveform_comparison, load_events_window
import matplotlib
matplotlib.use('Agg')

# Settings
DATA_DIR = Path("~/Desktop/QuantumNetwork/data/raw_truncated").expanduser()
VOLTAGE = '300mV'      # Change to: 200mV, 300mV, 400mV, 500mV
START_MS = 1000.0      # Start at 1 second
DURATION_MS = 500.0    # 500ms window
MAX_EVENTS = 150000    # Max events per waveform

# Parse arguments
if '--voltage' in sys.argv:
    idx = sys.argv.index('--voltage')
    VOLTAGE = sys.argv[idx + 1]

if '--duration' in sys.argv:
    idx = sys.argv.index('--duration')
    DURATION_MS = float(sys.argv[idx + 1])

if '--start' in sys.argv:
    idx = sys.argv.index('--start')
    START_MS = float(sys.argv[idx + 1])

# Find CSV files
csv_files = {}
for waveform in ['sine', 'square', 'triangle', 'burst']:
    pattern = f"{waveform}-{VOLTAGE}.csv"
    matches = list(DATA_DIR.glob(pattern))
    if matches:
        csv_files[waveform] = str(matches[0])

print("="*70)
print("Waveform Comparison - Time vs X Analysis")
print("="*70)
print(f"Voltage: {VOLTAGE}")
print(f"Time window: [{START_MS:.0f}, {START_MS + DURATION_MS:.0f}]ms ({DURATION_MS:.0f}ms)")
print(f"\nFiles found: {len(csv_files)}/4")

if len(csv_files) < 4:
    print("\nWARNING: Not all waveforms found!")
    missing = set(['sine', 'square', 'triangle', 'burst']) - set(csv_files.keys())
    print(f"Missing: {', '.join(missing)}")

for waveform in ['sine', 'square', 'triangle', 'burst']:
    if waveform in csv_files:
        print(f"  ✓ {waveform.upper():8s}: {Path(csv_files[waveform]).name}")
    else:
        print(f"  ✗ {waveform.upper():8s}: NOT FOUND")

if len(csv_files) == 0:
    print("\nERROR: No CSV files found!")
    sys.exit(1)

# Create output
output_dir = Path("./waveform_comparison")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / f'comparison_{VOLTAGE}_{START_MS:.0f}ms_{DURATION_MS:.0f}ms.png'

print(f"\nGenerating comparison plot...")
print(f"Expected patterns:")
print(f"  SINE:     Smooth curved stripes (gradual transitions)")
print(f"  SQUARE:   Sharp vertical edges (abrupt on/off)")
print(f"  TRIANGLE: Linear diagonal stripes (constant slope)")
print(f"  BURST:    Random scattered events (no clear pattern)")

plot_waveform_comparison(
    csv_files,
    start_ms=START_MS,
    duration_ms=DURATION_MS,
    max_events=MAX_EVENTS,
    save_path=str(output_file)
)

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"\nSaved to: {output_file}")
print(f"\nOpen with:")
print(f"  xdg-open {output_file}")

print(f"\n" + "="*70)
print("What to Look For:")
print("="*70)
print("SINE waveform:")
print("  - Smooth, curved diagonal stripes")
print("  - Events form gentle waves across X axis")
print("  - Gradual transitions between ON (blue) and OFF (red)")

print("\nSQUARE waveform:")
print("  - Sharp, vertical edges")
print("  - Abrupt transitions (sudden jumps)")
print("  - Events cluster in vertical bands")

print("\nTRIANGLE waveform:")
print("  - Linear diagonal stripes")
print("  - Constant slope (straight lines)")
print("  - Regular, predictable pattern")

print("\nBURST waveform:")
print("  - Random, scattered events")
print("  - No clear temporal structure")
print("  - Chaotic distribution")

print(f"\n" + "="*70)
print("Try Different Settings:")
print("="*70)
print("Different voltages:")
print("  python show_waveform_differences.py --voltage 200mV")
print("  python show_waveform_differences.py --voltage 500mV")

print("\nLonger time window:")
print("  python show_waveform_differences.py --duration 1000")
print("  python show_waveform_differences.py --duration 2000")

print("\nDifferent start time:")
print("  python show_waveform_differences.py --start 5000 --duration 500")
