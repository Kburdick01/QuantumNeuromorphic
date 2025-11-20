#!/usr/bin/env python3
"""
CSV Format Inspector
Checks the actual format of your CSV files to understand the structure.
"""

import sys
from pathlib import Path

# Default file to inspect
if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1]).expanduser()
else:
    csv_path = Path("~/Desktop/QuantumNetwork/data/raw_truncated/sine-300mV.csv").expanduser()

print("="*70)
print(f"CSV Format Inspector")
print("="*70)
print(f"\nFile: {csv_path}")

if not csv_path.exists():
    print(f"ERROR: File not found!")
    sys.exit(1)

print(f"Size: {csv_path.stat().st_size / (1024*1024):.2f} MB")

# Read first 50 lines raw
print(f"\n{'='*70}")
print(f"First 30 lines (raw text):")
print(f"{'='*70}\n")

with open(csv_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 30:
            break
        # Show line number and content
        print(f"Line {i+1:3d}: {line.rstrip()}")

# Check for different delimiters
print(f"\n{'='*70}")
print(f"Delimiter Analysis:")
print(f"{'='*70}\n")

with open(csv_path, 'r') as f:
    first_lines = [next(f) for _ in range(20)]

for delim_name, delim_char in [('Comma', ','), ('Tab', '\t'), ('Space', ' '), ('Semicolon', ';')]:
    counts = [line.count(delim_char) for line in first_lines]
    print(f"{delim_name:12s}: {counts[:10]}")

# Try reading with pandas with different options
print(f"\n{'='*70}")
print(f"Pandas Read Attempts:")
print(f"{'='*70}\n")

import pandas as pd

# Try 1: Default
try:
    df = pd.read_csv(csv_path, nrows=5)
    print(f"✓ Default read successful!")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    print(f"\n  First few rows:")
    print(df.to_string(index=False))
except Exception as e:
    print(f"✗ Default read failed: {e}")

# Try 2: Skip header rows
print(f"\n{'-'*70}\n")
for skip in [0, 1, 2, 3, 4, 5]:
    try:
        df = pd.read_csv(csv_path, skiprows=skip, nrows=5, header=None)
        print(f"✓ skiprows={skip} successful! Shape: {df.shape}, Columns: {len(df.columns)}")
        if skip == 0:
            print(f"  First row: {df.iloc[0].tolist()}")
        break
    except Exception as e:
        print(f"✗ skiprows={skip} failed: {str(e)[:60]}")

# Try 3: Different delimiters
print(f"\n{'-'*70}\n")
for delim_name, delim in [('comma', ','), ('tab', '\t'), ('space', ' ')]:
    try:
        df = pd.read_csv(csv_path, sep=delim, nrows=5)
        print(f"✓ Delimiter={delim_name} successful! Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Delimiter={delim_name} failed")

# Try 4: Comment lines
print(f"\n{'-'*70}\n")
for comment_char in ['#', '%', '/']:
    try:
        df = pd.read_csv(csv_path, comment=comment_char, nrows=5)
        print(f"✓ Comment char='{comment_char}' successful! Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Comment char='{comment_char}' failed")

print(f"\n{'='*70}")
print("Recommendation:")
print("="*70)
print("\nBased on the analysis above, update the CSV reading code with:")
print("  - Correct skiprows value (if there are header/comment lines)")
print("  - Correct delimiter (comma, tab, space)")
print("  - Correct comment character (if any)")
