#!/usr/bin/env python3
"""
QTCR-Net PyTorch Dataset and DataLoader
Loads preprocessed DVS128 voxel grids for training and evaluation.

Author: QTCR-Net Research Team
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import yaml


class DVSVoxelDataset(Dataset):
    """
    PyTorch Dataset for DVS128 voxel grid data.
    Loads preprocessed .npy files and returns tensors with labels.
    """

    def __init__(self,
                 manifest_path: str,
                 waveform_classes: List[str],
                 voltage_classes: List[str],
                 transform=None,
                 augment: bool = False):
        """
        Initialize dataset.

        Args:
            manifest_path: Path to manifest CSV file
            waveform_classes: List of waveform class names
            voltage_classes: List of voltage class names
            transform: Optional transform to apply to data
            augment: Whether to apply data augmentation
        """
        self.manifest_path = Path(manifest_path)
        self.waveform_classes = waveform_classes
        self.voltage_classes = voltage_classes
        self.transform = transform
        self.augment = augment

        # Load manifest
        self.manifest = pd.read_csv(manifest_path)

        # Create label mappings
        self.waveform_to_idx = {name: idx for idx, name in enumerate(waveform_classes)}
        self.voltage_to_idx = {name: idx for idx, name in enumerate(voltage_classes)}

        print(f"[DVSVoxelDataset] Loaded {len(self.manifest)} samples")
        print(f"  Waveform classes: {waveform_classes}")
        print(f"  Voltage classes: {voltage_classes}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Args:
            idx: Index of sample

        Returns:
            Tuple of (voxel_tensor, labels_dict) where:
                - voxel_tensor: [C, T, H, W] float32 tensor
                - labels_dict: {'waveform': int64, 'voltage': int64}
        """
        # Get sample metadata
        sample = self.manifest.iloc[idx]
        npy_path = sample['npy_path']
        waveform_label = sample['waveform_label']
        voltage_label = sample['voltage_label']

        # Load voxel grid
        voxel_grid = np.load(npy_path)  # [C, T, H, W]

        # Convert to tensor
        voxel_tensor = torch.from_numpy(voxel_grid).float()

        # Apply augmentation if enabled
        if self.augment:
            voxel_tensor = self._augment(voxel_tensor)

        # Apply transform if provided
        if self.transform:
            voxel_tensor = self.transform(voxel_tensor)

        # Convert labels to indices
        waveform_idx = self.waveform_to_idx.get(waveform_label, -1)
        voltage_idx = self.voltage_to_idx.get(voltage_label, -1)

        if waveform_idx == -1:
            print(f"[WARNING] Unknown waveform label: {waveform_label}")
            waveform_idx = 0

        if voltage_idx == -1:
            print(f"[WARNING] Unknown voltage label: {voltage_label}")
            voltage_idx = 0

        labels = {
            'waveform': torch.tensor(waveform_idx, dtype=torch.long),
            'voltage': torch.tensor(voltage_idx, dtype=torch.long)
        }

        return voxel_tensor, labels

    def _augment(self, voxel_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to voxel tensor.

        Args:
            voxel_tensor: Input voxel tensor [C, T, H, W]

        Returns:
            Augmented voxel tensor
        """
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            voxel_tensor = torch.flip(voxel_tensor, dims=[3])  # Flip W dimension

        # Random vertical flip
        if torch.rand(1) > 0.5:
            voxel_tensor = torch.flip(voxel_tensor, dims=[2])  # Flip H dimension

        # Random polarity swap (swap channels 0 and 1)
        if torch.rand(1) > 0.5:
            voxel_tensor = torch.flip(voxel_tensor, dims=[0])  # Flip C dimension

        # Random temporal reverse (causal modeling, use with caution)
        # if torch.rand(1) > 0.9:  # Low probability
        #     voxel_tensor = torch.flip(voxel_tensor, dims=[1])  # Flip T dimension

        # Add small Gaussian noise
        if torch.rand(1) > 0.7:
            noise = torch.randn_like(voxel_tensor) * 0.01
            voxel_tensor = voxel_tensor + noise

        return voxel_tensor

    def get_label_distributions(self) -> Dict:
        """
        Get distribution of labels in the dataset.

        Returns:
            Dictionary with label counts
        """
        waveform_dist = self.manifest['waveform_label'].value_counts().to_dict()
        voltage_dist = self.manifest['voltage_label'].value_counts().to_dict()

        return {
            'waveform': waveform_dist,
            'voltage': voltage_dist
        }


def create_dataloaders_csv_split(config: dict,
                                 train_augment: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders with time-based splitting.

    IMPORTANT: Splits by window index within each CSV to ensure:
        - All voltage classes in all splits
        - All waveform classes in all splits
        - Temporal separation to minimize data leakage

    Split ratios (by window index within each CSV):
        - Train: first 70% of windows
        - Val: next 15% of windows
        - Test: last 15% of windows

    Args:
        config: Configuration dictionary
        train_augment: Whether to apply augmentation to training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config['data']
    training_config = config['training']

    # Load manifest
    manifest_path = Path(data_config['manifest_path'])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)

    # Time-based split within each CSV
    train_indices = []
    val_indices = []
    test_indices = []

    # Group by original CSV and split by window index
    for csv_name, group in manifest_df.groupby('original_csv'):
        # Sort by window index
        group = group.sort_values('window_index')
        indices = group.index.tolist()
        n = len(indices)

        # Calculate split points
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])

    # Create split manifests
    train_manifest = manifest_df.loc[train_indices]
    val_manifest = manifest_df.loc[val_indices]
    test_manifest = manifest_df.loc[test_indices]

    print(f"\n[Time-Based Dataset Split]")
    print(f"  Strategy: Split each CSV by window index (70/15/15)")
    print(f"  Total CSVs: {manifest_df['original_csv'].nunique()}")
    print(f"  Total samples: {len(manifest_df)}")

    # Save split manifests
    processed_dir = Path(data_config['processed_dir'])
    train_manifest.to_csv(processed_dir / 'manifest_train.csv', index=False)
    val_manifest.to_csv(processed_dir / 'manifest_val.csv', index=False)
    test_manifest.to_csv(processed_dir / 'manifest_test.csv', index=False)

    # Create datasets
    train_dataset = DVSVoxelDataset(
        manifest_path=processed_dir / 'manifest_train.csv',
        waveform_classes=data_config['waveform_classes'],
        voltage_classes=data_config['voltage_classes'],
        augment=train_augment
    )

    val_dataset = DVSVoxelDataset(
        manifest_path=processed_dir / 'manifest_val.csv',
        waveform_classes=data_config['waveform_classes'],
        voltage_classes=data_config['voltage_classes'],
        augment=False
    )

    test_dataset = DVSVoxelDataset(
        manifest_path=processed_dir / 'manifest_test.csv',
        waveform_classes=data_config['waveform_classes'],
        voltage_classes=data_config['voltage_classes'],
        augment=False
    )

    print(f"\n[Sample Counts]")
    print(f"  Train: {len(train_dataset)} samples (70%)")
    print(f"  Val:   {len(val_dataset)} samples (15%)")
    print(f"  Test:  {len(test_dataset)} samples (15%)")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_dataloaders(config: dict,
                      train_augment: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        config: Configuration dictionary
        train_augment: Whether to apply augmentation to training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config['data']
    training_config = config['training']

    # Load manifest
    manifest_path = Path(data_config['manifest_path'])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Create full dataset
    full_dataset = DVSVoxelDataset(
        manifest_path=manifest_path,
        waveform_classes=data_config['waveform_classes'],
        voltage_classes=data_config['voltage_classes'],
        augment=False  # Will be enabled per-split
    )

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(training_config['train_split'] * total_size)
    val_size = int(training_config['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    # Set random seed for reproducibility
    torch.manual_seed(training_config['random_seed'])

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

    print(f"\n[Dataset Splits]")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Enable augmentation for training dataset
    if train_augment:
        # Wrap train dataset to enable augmentation
        original_dataset = train_dataset.dataset

        class AugmentedDataset(Dataset):
            def __init__(self, subset, augment=True):
                self.subset = subset
                self.augment = augment

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                real_idx = self.subset.indices[idx]
                voxel, labels = self.subset.dataset.manifest.iloc[real_idx], labels

                # Load and augment
                sample = self.subset.dataset.manifest.iloc[real_idx]
                npy_path = sample['npy_path']
                voxel_grid = np.load(npy_path)
                voxel_tensor = torch.from_numpy(voxel_grid).float()

                if self.augment:
                    voxel_tensor = self.subset.dataset._augment(voxel_tensor)

                # Get labels
                waveform_label = sample['waveform_label']
                voltage_label = sample['voltage_label']
                waveform_idx = self.subset.dataset.waveform_to_idx.get(waveform_label, 0)
                voltage_idx = self.subset.dataset.voltage_to_idx.get(voltage_label, 0)

                labels = {
                    'waveform': torch.tensor(waveform_idx, dtype=torch.long),
                    'voltage': torch.tensor(voltage_idx, dtype=torch.long)
                }

                return voxel_tensor, labels

        train_dataset_aug = AugmentedDataset(train_dataset, augment=True)
    else:
        train_dataset_aug = train_dataset

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset_aug if train_augment else train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        prefetch_factor=config['hardware']['prefetch_factor'],
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        prefetch_factor=config['hardware']['prefetch_factor'],
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        prefetch_factor=config['hardware']['prefetch_factor'],
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )

    return train_loader, val_loader, test_loader


def test_dataset():
    """Test dataset loading."""
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create dataset
    manifest_path = config['data']['manifest_path']

    if not Path(manifest_path).exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("  Run preprocess.py first to generate the manifest.")
        return

    dataset = DVSVoxelDataset(
        manifest_path=manifest_path,
        waveform_classes=config['data']['waveform_classes'],
        voltage_classes=config['data']['voltage_classes'],
        augment=True
    )

    print(f"\n[Dataset Test]")
    print(f"  Total samples: {len(dataset)}")

    # Get first sample
    voxel, labels = dataset[0]
    print(f"\n[Sample 0]")
    print(f"  Voxel shape: {voxel.shape}")
    print(f"  Voxel dtype: {voxel.dtype}")
    print(f"  Voxel range: [{voxel.min():.3f}, {voxel.max():.3f}]")
    print(f"  Waveform label: {labels['waveform'].item()} ({config['data']['waveform_classes'][labels['waveform']]})")
    print(f"  Voltage label: {labels['voltage'].item()} ({config['data']['voltage_classes'][labels['voltage']]})")

    # Test DataLoader
    print(f"\n[DataLoader Test]")
    train_loader, val_loader, test_loader = create_dataloaders(config, train_augment=True)

    # Get a batch
    batch_voxels, batch_labels = next(iter(train_loader))
    print(f"  Batch shape: {batch_voxels.shape}")
    print(f"  Batch dtype: {batch_voxels.dtype}")
    print(f"  Waveform labels: {batch_labels['waveform']}")
    print(f"  Voltage labels: {batch_labels['voltage']}")

    # Show label distributions
    print(f"\n[Label Distributions]")
    distributions = dataset.get_label_distributions()
    print("Waveform:")
    for label, count in distributions['waveform'].items():
        print(f"  {label}: {count}")
    print("Voltage:")
    for label, count in distributions['voltage'].items():
        print(f"  {label}: {count}")

    print("\n[Dataset Test] Complete!")


if __name__ == '__main__':
    test_dataset()
