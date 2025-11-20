#!/usr/bin/env python3
"""
Quick setup and verification script for your DVS128 data.
Checks that all files are accessible and ready for preprocessing.

Author: QTCR-Net Research Team
"""

import os
from pathlib import Path
import pandas as pd

# Your data directory
DATA_DIR = Path("~/Desktop/QuantumNetwork/data/raw_truncated").expanduser()

print("="*70)
print("QTCR-Net Data Setup Verification")
print("="*70)

# Check if directory exists
print(f"\n1. Checking data directory...")
if not DATA_DIR.exists():
    print(f"   ✗ ERROR: Directory not found: {DATA_DIR}")
    print(f"   Please check the path in setup_my_data.py")
    exit(1)
else:
    print(f"   ✓ Directory found: {DATA_DIR}")

# List all CSV files
print(f"\n2. Finding CSV files...")
csv_files = sorted(DATA_DIR.glob("*.csv"))

if len(csv_files) == 0:
    print(f"   ✗ ERROR: No CSV files found in {DATA_DIR}")
    exit(1)
else:
    print(f"   ✓ Found {len(csv_files)} CSV files:")
    for csv in csv_files:
        print(f"      - {csv.name}")

# Expected files
expected_waveforms = ['burst', 'sine', 'square', 'triangle']
expected_voltages = ['200mV', '300mV', '400mV', '500mV']

print(f"\n3. Verifying file coverage...")
found_combinations = set()
for csv in csv_files:
    # Parse filename
    name = csv.stem
    if '-' in name:
        parts = name.split('-')
        if len(parts) >= 2:
            waveform = parts[0]
            voltage = parts[1]
            found_combinations.add((waveform, voltage))

print(f"   Expected: {len(expected_waveforms)} × {len(expected_voltages)} = {len(expected_waveforms) * len(expected_voltages)} files")
print(f"   Found: {len(csv_files)} files")

# Check for missing combinations
missing = []
for waveform in expected_waveforms:
    for voltage in expected_voltages:
        if (waveform, voltage) not in found_combinations:
            missing.append(f"{waveform}-{voltage}")

if missing:
    print(f"   ⚠ Warning: Missing files: {', '.join(missing)}")
else:
    print(f"   ✓ All expected combinations present!")

# Sample one file to check format
print(f"\n4. Checking CSV format (sampling first file)...")
test_csv = csv_files[0]
print(f"   Reading: {test_csv.name}")

try:
    # Read first few rows (skip comment lines starting with #)
    df = pd.read_csv(test_csv, comment='#', nrows=10)

    print(f"   ✓ File readable")
    print(f"   Columns: {list(df.columns)}")

    # Check for required columns
    df.columns = df.columns.str.strip().str.lower()

    required = set()
    if 'timestamp_us' in df.columns or 'timestamp' in df.columns or 't' in df.columns:
        required.add('timestamp')
    if 'x' in df.columns:
        required.add('x')
    if 'y' in df.columns:
        required.add('y')
    if 'polarity' in df.columns or 'pol' in df.columns or 'p' in df.columns:
        required.add('polarity')

    if len(required) == 4:
        print(f"   ✓ All required columns present (timestamp, x, y, polarity)")
    else:
        print(f"   ⚠ Warning: Missing some columns. Found: {required}")

    # Show sample data
    print(f"\n   Sample data (first 3 rows):")
    print(df.head(3).to_string(index=False))

except Exception as e:
    print(f"   ✗ ERROR reading file: {e}")
    exit(1)

# Get file sizes
print(f"\n5. Checking file sizes...")
total_size = 0
for csv in csv_files:
    size_mb = csv.stat().st_size / (1024 * 1024)
    total_size += size_mb

avg_size_mb = total_size / len(csv_files)
print(f"   Total size: {total_size:.1f} MB")
print(f"   Average size per file: {avg_size_mb:.1f} MB")

# Estimate events per file (rough)
if avg_size_mb > 0:
    # Rough estimate: ~50 bytes per event (CSV text format)
    est_events = (avg_size_mb * 1024 * 1024) / 50
    print(f"   Estimated events per file: ~{est_events:,.0f}")

# Check config.yaml
print(f"\n6. Checking config.yaml...")
config_path = Path("config.yaml")
if config_path.exists():
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    csv_dir_config = Path(config['data']['csv_dir']).expanduser()

    if csv_dir_config == DATA_DIR:
        print(f"   ✓ config.yaml points to correct directory")
    else:
        print(f"   ⚠ Warning: config.yaml points to different directory:")
        print(f"      Config: {csv_dir_config}")
        print(f"      Actual: {DATA_DIR}")
else:
    print(f"   ✗ config.yaml not found in current directory")

print(f"\n" + "="*70)
print("Summary:")
print("="*70)
print(f"✓ Data directory: {DATA_DIR}")
print(f"✓ CSV files: {len(csv_files)}")
print(f"✓ Total size: {total_size:.1f} MB")
print(f"\nYour data is ready for preprocessing!")

print(f"\n" + "="*70)
print("Next Steps:")
print("="*70)
print(f"\n1. Quick test (2 files, 10 windows each):")
print(f"   python preprocess.py --config config.yaml --max_files 2 --max_windows 10")

print(f"\n2. Process one file completely:")
print(f"   python preprocess.py --config config.yaml --max_files 1")

print(f"\n3. Process ALL files (will take time!):")
print(f"   python preprocess.py --config config.yaml")

print(f"\n4. Visualize events from one file:")
print(f"   python visualize_events_3d.py {csv_files[0]} --max_events 10000 --time_start 0 --time_end 2 --all")

print(f"\n" + "="*70)
