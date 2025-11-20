#!/usr/bin/env python3
"""
QTCR-Net Preprocessing Pipeline
Converts DVS128 CSV event streams to voxel grid tensors for neural network training.

Input: CSV files with format: timestamp_us, x, y, polarity
Output: Voxel grids [2, T, H, W] saved as .npy files + manifest CSV

Author: QTCR-Net Research Team
Date: 2025
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import yaml
import warnings
warnings.filterwarnings('ignore')


class DVSEventPreprocessor:
    """
    Preprocessor for DVS128 event camera data.
    Converts raw CSV event streams into voxel grid representations.
    """

    def __init__(self, config: dict):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.data_config = config['data']
        self.sensor_config = self.data_config['sensor']
        self.window_config = self.data_config['window']
        self.spatial_config = self.data_config['spatial']
        self.norm_config = self.data_config['normalization']

        # Sensor dimensions
        self.sensor_width = self.sensor_config['width']
        self.sensor_height = self.sensor_config['height']
        self.num_polarities = self.sensor_config['polarities']

        # Window parameters
        self.window_duration_sec = self.window_config['duration_sec']
        self.window_duration_us = int(self.window_duration_sec * 1e6)
        self.temporal_bins = self.window_config['temporal_bins']
        self.overlap = self.window_config['overlap']

        # Spatial pooling
        self.patch_size = self.spatial_config['patch_size']
        self.spatial_height = self.sensor_height // self.patch_size
        self.spatial_width = self.sensor_width // self.patch_size

        # Paths
        self.csv_dir = Path(self.data_config['csv_dir']).expanduser()
        self.processed_dir = Path(self.data_config['processed_dir'])
        self.manifest_path = Path(self.data_config['manifest_path'])

        # Create output directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[DVSEventPreprocessor] Initialized")
        print(f"  Sensor: {self.sensor_width}x{self.sensor_height}, {self.num_polarities} polarities")
        print(f"  Window: {self.window_duration_sec}s, {self.temporal_bins} temporal bins")
        print(f"  Spatial pooling: {self.patch_size}x{self.patch_size} → {self.spatial_width}x{self.spatial_height}")
        print(f"  Output dir: {self.processed_dir}")

    def load_csv_metadata(self, csv_path: Path) -> Dict[str, str]:
        """
        Extract metadata (waveform, voltage) from CSV filename.

        Expected filename formats:
        - {waveform}-{voltage}.csv  (e.g., sine-300mV.csv)
        - {waveform}_{voltage}.csv  (e.g., sine_300mV.csv)

        Args:
            csv_path: Path to CSV file

        Returns:
            Dictionary with 'waveform' and 'voltage' keys
        """
        filename = csv_path.stem  # Remove .csv extension

        # Try splitting by hyphen first, then underscore
        if '-' in filename:
            parts = filename.split('-')
        else:
            parts = filename.split('_')

        # Try to extract waveform and voltage from filename
        waveform = None
        voltage = None

        for part in parts:
            if part.lower() in self.data_config['waveform_classes']:
                waveform = part.lower()
            if 'mv' in part.lower():
                voltage = part

        if waveform is None or voltage is None:
            print(f"[WARNING] Could not parse metadata from filename: {filename}")
            print(f"  Using defaults: waveform='unknown', voltage='unknown'")
            waveform = waveform or 'unknown'
            voltage = voltage or 'unknown'

        return {'waveform': waveform, 'voltage': voltage}

    def read_csv_events(self, csv_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Read DVS128 CSV file with event data.

        Expected columns: timestamp_us, x, y, polarity

        Args:
            csv_path: Path to CSV file
            max_rows: Maximum number of rows to read (for testing)

        Returns:
            DataFrame with columns: timestamp_us, x, y, polarity
        """
        print(f"[Reading CSV] {csv_path.name}")

        # Read CSV, skipping comment lines starting with #
        # Force numeric types to avoid string contamination
        try:
            df = pd.read_csv(
                csv_path,
                comment='#',
                nrows=max_rows,
                dtype={
                    'timestamp': np.int64,
                    'timeStamp': np.int64,
                    'x': np.int32,
                    'y': np.int32,
                    'polarity': np.int32
                },
                skipinitialspace=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to read CSV: {e}")
            raise

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()

        # Map common column name variations
        column_mapping = {
            'timestamp': 'timestamp_us',
            'time': 'timestamp_us',
            't': 'timestamp_us',
            'pol': 'polarity',
            'p': 'polarity'
        }

        df.rename(columns=column_mapping, inplace=True)

        # Validate required columns
        required_cols = ['timestamp_us', 'x', 'y', 'polarity']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}. Available: {df.columns.tolist()}")

        # Ensure correct data types with error handling
        try:
            df['timestamp_us'] = pd.to_numeric(df['timestamp_us'], errors='coerce').astype(np.int64)
            df['x'] = pd.to_numeric(df['x'], errors='coerce').astype(np.int32)
            df['y'] = pd.to_numeric(df['y'], errors='coerce').astype(np.int32)
            df['polarity'] = pd.to_numeric(df['polarity'], errors='coerce').astype(np.int32)

            # Drop any rows with NaN values (from coercion failures)
            df = df.dropna()
        except Exception as e:
            print(f"[ERROR] Type conversion failed: {e}")
            print(f"Column types: {df.dtypes}")
            print(f"First few rows:\n{df.head()}")
            raise

        # Validate coordinates
        df = df[(df['x'] >= 0) & (df['x'] < self.sensor_width) &
                (df['y'] >= 0) & (df['y'] < self.sensor_height)]

        # Map polarity to 0/1 if needed
        unique_polarities = df['polarity'].unique()
        if set(unique_polarities) == {-1, 1}:
            df['polarity'] = (df['polarity'] + 1) // 2  # -1→0, 1→1
        elif set(unique_polarities) <= {0, 1}:
            pass  # Already in correct format
        else:
            print(f"[WARNING] Unexpected polarity values: {unique_polarities}")

        # Sort by timestamp
        df.sort_values('timestamp_us', inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"  Loaded {len(df):,} events, duration: {(df['timestamp_us'].iloc[-1] - df['timestamp_us'].iloc[0]) / 1e6:.2f}s")

        return df

    def events_to_voxel_grid(self, events: pd.DataFrame, start_time_us: int) -> np.ndarray:
        """
        Convert event stream to voxel grid representation.

        Args:
            events: DataFrame with columns [timestamp_us, x, y, polarity]
            start_time_us: Start timestamp of the window

        Returns:
            Voxel grid of shape [num_polarities, temporal_bins, spatial_height, spatial_width]
        """
        # Filter events within window
        end_time_us = start_time_us + self.window_duration_us
        window_events = events[
            (events['timestamp_us'] >= start_time_us) &
            (events['timestamp_us'] < end_time_us)
        ].copy()

        if len(window_events) == 0:
            # Return empty voxel grid
            return np.zeros((self.num_polarities, self.temporal_bins,
                           self.spatial_height, self.spatial_width), dtype=np.float32)

        # Extract raw numpy arrays immediately - NO pandas operations after this point
        timestamp_array = np.array(window_events['timestamp_us'].tolist(), dtype=np.int64)
        x_array = np.array(window_events['x'].tolist(), dtype=np.int32)
        y_array = np.array(window_events['y'].tolist(), dtype=np.int32)
        pol_array = np.array(window_events['polarity'].tolist(), dtype=np.int32)

        # Normalize timestamps to [0, 1] within window
        t_norm = (timestamp_array.astype(np.float64) - float(start_time_us)) / float(self.window_duration_us)
        t_norm = np.clip(t_norm, 0.0, 0.9999)

        # Compute temporal bin indices
        t_bin = (t_norm * float(self.temporal_bins)).astype(np.int32)
        t_bin = np.clip(t_bin, 0, self.temporal_bins - 1)

        # Compute spatial bin indices (pooling)
        x_bin = (x_array // int(self.patch_size)).astype(np.int32)
        x_bin = np.clip(x_bin, 0, self.spatial_width - 1)

        y_bin = (y_array // int(self.patch_size)).astype(np.int32)
        y_bin = np.clip(y_bin, 0, self.spatial_height - 1)

        # Initialize voxel grid
        voxel_grid = np.zeros((self.num_polarities, self.temporal_bins,
                               self.spatial_height, self.spatial_width), dtype=np.float32)

        # Accumulate events into voxel grid using vectorized approach
        # This avoids any loop-based indexing and type issues
        for pol_val in range(self.num_polarities):
            # Get events for this polarity
            pol_mask = pol_array == pol_val

            if np.any(pol_mask):
                t_pol = t_bin[pol_mask]
                y_pol = y_bin[pol_mask]
                x_pol = x_bin[pol_mask]

                # Debug first polarity
                if pol_val == 0:
                    print(f"    t_pol dtype: {t_pol.dtype}, sample: {t_pol[:3]}")
                    print(f"    y_pol dtype: {y_pol.dtype}, sample: {y_pol[:3]}")
                    print(f"    x_pol dtype: {x_pol.dtype}, sample: {x_pol[:3]}")

                # Use np.add.at for safe in-place accumulation
                # Convert to flat indices for the 3D slice
                flat_indices = (t_pol.astype(np.int64) * self.spatial_height * self.spatial_width +
                               y_pol.astype(np.int64) * self.spatial_width +
                               x_pol.astype(np.int64))

                if pol_val == 0:
                    print(f"    flat_indices dtype: {flat_indices.dtype}, sample: {flat_indices[:3]}")
                    raveled = voxel_grid[pol_val].ravel()
                    print(f"    voxel_grid[{pol_val}].ravel() dtype: {raveled.dtype}, shape: {raveled.shape}")
                    print(f"    voxel_grid dtype: {voxel_grid.dtype}")

                print(f"    About to call np.add.at for polarity {pol_val}")
                np.add.at(voxel_grid[pol_val].ravel(), flat_indices.astype(np.int64), np.float32(1.0))
                print(f"    Completed np.add.at for polarity {pol_val}")

        return voxel_grid

    def normalize_voxel_grid(self, voxel_grid: np.ndarray) -> np.ndarray:
        """
        Apply normalization to voxel grid.

        Args:
            voxel_grid: Raw voxel grid [C, T, H, W]

        Returns:
            Normalized voxel grid
        """
        # Log normalization: log(1 + x)
        log_eps = float(self.norm_config['log_eps'])
        voxel_grid = np.log1p(voxel_grid + log_eps)

        # Per-window normalization
        if self.norm_config['per_window_norm']:
            mean = voxel_grid.mean()
            std = voxel_grid.std() + 1e-8
            voxel_grid = (voxel_grid - mean) / std

        # Clip extreme values
        clip_value = float(self.norm_config['clip_value'])
        voxel_grid = np.clip(voxel_grid, -clip_value, clip_value)

        return voxel_grid

    def process_csv_file(self, csv_path: Path, max_windows: Optional[int] = None) -> List[Dict]:
        """
        Process a single CSV file: slice into windows and convert to voxel grids.

        Args:
            csv_path: Path to CSV file
            max_windows: Maximum number of windows to process (for testing)

        Returns:
            List of dictionaries with window metadata
        """
        # Extract metadata from filename
        metadata = self.load_csv_metadata(csv_path)
        waveform = metadata['waveform']
        voltage = metadata['voltage']

        # Read events
        events = self.read_csv_events(csv_path)

        if len(events) == 0:
            print(f"[WARNING] No events in {csv_path.name}, skipping")
            return []

        # Compute time range
        start_time_us = events['timestamp_us'].iloc[0]
        end_time_us = events['timestamp_us'].iloc[-1]
        total_duration_us = end_time_us - start_time_us

        # Compute number of windows
        stride_us = int(self.window_duration_us * (1 - self.overlap))
        num_windows = int((total_duration_us - self.window_duration_us) / stride_us) + 1

        if max_windows:
            num_windows = min(num_windows, max_windows)

        print(f"  Processing {num_windows} windows ({self.window_duration_sec}s each)")

        # Process each window
        window_metadata = []
        for i in tqdm(range(num_windows), desc=f"  {csv_path.stem}", leave=False):
            window_start_us = start_time_us + i * stride_us

            # Convert to voxel grid
            voxel_grid = self.events_to_voxel_grid(events, window_start_us)

            # Skip empty windows
            if voxel_grid.sum() == 0:
                continue

            # Normalize
            voxel_grid = self.normalize_voxel_grid(voxel_grid)

            # Save voxel grid
            npy_filename = f"{csv_path.stem}_window_{i:04d}.npy"
            npy_path = self.processed_dir / npy_filename
            np.save(npy_path, voxel_grid)

            # Store metadata
            window_metadata.append({
                'npy_path': str(npy_path),
                'waveform_label': waveform,
                'voltage_label': voltage,
                'original_csv': str(csv_path),
                'window_index': i,
                'window_start_time_us': window_start_us,
                'window_start_time_sec': window_start_us / 1e6,
                'num_events': int((voxel_grid > 0).sum()),
                'voxel_shape': str(voxel_grid.shape)
            })

        print(f"  Saved {len(window_metadata)} windows")
        return window_metadata

    def process_all_csvs(self, max_files: Optional[int] = None, max_windows_per_file: Optional[int] = None) -> pd.DataFrame:
        """
        Process all CSV files in the input directory.

        Args:
            max_files: Maximum number of CSV files to process (for testing)
            max_windows_per_file: Maximum windows per file (for testing)

        Returns:
            DataFrame with manifest of all processed windows
        """
        # Find all CSV files
        csv_files = sorted(self.csv_dir.glob("*.csv"))

        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {self.csv_dir}")

        print(f"\n[Processing] Found {len(csv_files)} CSV files")

        if max_files:
            csv_files = csv_files[:max_files]
            print(f"  Limited to {max_files} files for testing")

        # Process each file
        all_metadata = []
        for csv_path in csv_files:
            try:
                metadata = self.process_csv_file(csv_path, max_windows=max_windows_per_file)
                all_metadata.extend(metadata)
            except Exception as e:
                print(f"[ERROR] Failed to process {csv_path.name}: {e}")
                continue

        # Create manifest DataFrame
        manifest_df = pd.DataFrame(all_metadata)

        # Save manifest
        manifest_df.to_csv(self.manifest_path, index=False)
        print(f"\n[Manifest] Saved to {self.manifest_path}")
        print(f"  Total windows: {len(manifest_df)}")

        # Print label distribution
        print("\n[Label Distribution]")
        print("Waveform:")
        print(manifest_df['waveform_label'].value_counts())
        print("\nVoltage:")
        print(manifest_df['voltage_label'].value_counts())

        return manifest_df

    def get_statistics(self, manifest_df: pd.DataFrame) -> Dict:
        """
        Compute dataset statistics.

        Args:
            manifest_df: Manifest DataFrame

        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_windows': len(manifest_df),
            'waveform_distribution': manifest_df['waveform_label'].value_counts().to_dict(),
            'voltage_distribution': manifest_df['voltage_label'].value_counts().to_dict(),
            'unique_csvs': manifest_df['original_csv'].nunique(),
            'window_duration_sec': self.window_duration_sec,
            'temporal_bins': self.temporal_bins,
            'spatial_dims': f"{self.spatial_height}x{self.spatial_width}",
            'patch_size': self.patch_size
        }

        return stats


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="QTCR-Net DVS128 Preprocessing Pipeline")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of CSV files to process (for testing)')
    parser.add_argument('--max_windows', type=int, default=None,
                       help='Maximum windows per file (for testing)')
    parser.add_argument('--stats_only', action='store_true',
                       help='Only compute statistics from existing manifest')

    args = parser.parse_args()

    # Load configuration
    print(f"[Config] Loading from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize preprocessor
    preprocessor = DVSEventPreprocessor(config)

    if args.stats_only:
        # Load existing manifest
        if not preprocessor.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {preprocessor.manifest_path}")

        manifest_df = pd.read_csv(preprocessor.manifest_path)
        stats = preprocessor.get_statistics(manifest_df)

        print("\n[Dataset Statistics]")
        print(json.dumps(stats, indent=2))

    else:
        # Process all CSV files
        manifest_df = preprocessor.process_all_csvs(
            max_files=args.max_files,
            max_windows_per_file=args.max_windows
        )

        # Compute and save statistics
        stats = preprocessor.get_statistics(manifest_df)

        stats_path = preprocessor.processed_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n[Statistics] Saved to {stats_path}")
        print(json.dumps(stats, indent=2))

    print("\n[Preprocessing] Complete!")


if __name__ == '__main__':
    main()
