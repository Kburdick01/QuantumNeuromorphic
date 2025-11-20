#!/usr/bin/env python3
"""
QTCR-Net Installation Test
Verifies that all dependencies are correctly installed.

Author: QTCR-Net Research Team
Date: 2025
"""

import sys

def test_imports():
    """Test all required imports."""
    print("Testing QTCR-Net installation...\n")

    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA: {torch.version.cuda}")
        else:
            print("  WARNING: CUDA not available. Training will be slow.")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    # Test PennyLane
    try:
        import pennylane as qml
        print(f"✓ PennyLane {qml.__version__}")

        # Test quantum device
        try:
            dev = qml.device('default.qubit', wires=2)
            print("  Quantum device: default.qubit ✓")
        except Exception as e:
            print(f"  WARNING: Could not create quantum device: {e}")

        # Test lightning device if available
        try:
            import pennylane_lightning
            dev = qml.device('lightning.qubit', wires=2)
            print("  Quantum device: lightning.qubit ✓ (faster)")
        except ImportError:
            print("  INFO: pennylane-lightning not found (optional, but recommended for speed)")

    except ImportError as e:
        print(f"✗ PennyLane import failed: {e}")
        return False

    # Test other dependencies
    deps = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
        'tensorboard': 'TensorBoard'
    }

    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            return False

    print("\n✓ All dependencies installed successfully!")
    return True


def test_model_creation():
    """Test QTCR-Net model creation."""
    print("\nTesting QTCR-Net model creation...\n")

    try:
        import torch
        import yaml
        from qtcr_model import QTCRNet

        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        print("Creating QTCR-Net model...")
        model = QTCRNet(config)

        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        batch_size = 2
        C = config['data']['sensor']['polarities']
        T = config['data']['window']['temporal_bins']
        H = config['data']['sensor']['height'] // config['data']['spatial']['patch_size']
        W = config['data']['sensor']['width'] // config['data']['spatial']['patch_size']

        x = torch.randn(batch_size, C, T, H, W).to(device)

        print(f"Test input shape: {x.shape}")

        with torch.no_grad():
            waveform_logits, voltage_logits = model(x)

        print(f"Waveform output shape: {waveform_logits.shape}")
        print(f"Voltage output shape: {voltage_logits.shape}")

        print("\n✓ QTCR-Net model creation and forward pass successful!")
        return True

    except Exception as e:
        print(f"\n✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset loading."""
    print("\nTesting dataset loading...\n")

    try:
        from pathlib import Path
        import yaml

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        manifest_path = Path(config['data']['manifest_path'])

        if not manifest_path.exists():
            print(f"INFO: Manifest not found at {manifest_path}")
            print("Run preprocess.py first to generate the dataset.")
            return True

        from dataset import DVSVoxelDataset

        dataset = DVSVoxelDataset(
            manifest_path=str(manifest_path),
            waveform_classes=config['data']['waveform_classes'],
            voltage_classes=config['data']['voltage_classes']
        )

        print(f"Dataset size: {len(dataset)}")

        voxel, labels = dataset[0]
        print(f"Sample voxel shape: {voxel.shape}")
        print(f"Sample labels: waveform={labels['waveform']}, voltage={labels['voltage']}")

        print("\n✓ Dataset loading successful!")
        return True

    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("QTCR-Net Installation Test")
    print("="*70)

    # Test imports
    if not test_imports():
        print("\n✗ Installation test FAILED. Please install missing dependencies.")
        sys.exit(1)

    # Test model creation
    if not test_model_creation():
        print("\n✗ Model creation test FAILED. Check error messages above.")
        sys.exit(1)

    # Test dataset (optional)
    test_dataset()

    print("\n" + "="*70)
    print("✓ QTCR-Net installation test PASSED!")
    print("="*70)
    print("\nYou can now:")
    print("  1. Preprocess data: python preprocess.py --config config.yaml")
    print("  2. Train model: python train.py --config config.yaml")
    print("  3. Run demo: jupyter notebook QTCR_Net_Demo.ipynb")


if __name__ == '__main__':
    main()
