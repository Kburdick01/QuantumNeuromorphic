# Your DVS128 Data - Quick Start Guide

## 📁 Your Data Location

```
~/Desktop/QuantumNetwork/data/raw_truncated/
├── burst-200mV.csv
├── burst-300mV.csv
├── burst-400mV.csv
├── burst-500mV.csv
├── sine-200mV.csv
├── sine-300mV.csv
├── sine-400mV.csv
├── sine-500mV.csv
├── square-200mV.csv
├── square-300mV.csv
├── square-400mV.csv
├── square-500mV.csv
├── triangle-200mV.csv
├── triangle-300mV.csv
├── triangle-400mV.csv
└── triangle-500mV.csv
```

**Total: 16 CSV files**
- 4 waveforms: burst, sine, square, triangle
- 4 voltages: 200mV, 300mV, 400mV, 500mV

---

## 🚀 Step-by-Step Workflow

### Step 0: Verify Everything is Set Up

```bash
cd ~/MatlabToolBox
source venv/bin/activate
python setup_my_data.py
```

This will:
- ✓ Check that all 16 CSV files are accessible
- ✓ Verify CSV format
- ✓ Show file sizes and estimated event counts
- ✓ Confirm config.yaml is correct

---

### Step 1: Visualize Your Data (Optional but Recommended)

Before preprocessing, visualize some events to understand your data:

```bash
# Quick visualization of sine-300mV (default)
python visualize_my_data.py

# Visualize a specific file
python visualize_my_data.py burst-200mV

# All visualization types
python visualize_my_data.py sine-400mV --all

# Or using the full script with custom parameters
python visualize_events_3d.py ~/Desktop/QuantumNetwork/data/raw_truncated/sine-300mV.csv \
    --max_events 15000 \
    --time_start 0 \
    --time_end 3 \
    --all
```

**What you'll see:**
- 3D plot: Time × X × Y with blue (ON) and red (OFF) events
- Spatial distribution (X vs Y)
- Density heatmaps
- Temporal evolution slices

---

### Step 2: Test Preprocessing (Quick Test)

Process just 2 files with 10 windows each to verify everything works:

```bash
python preprocess.py --config config.yaml --max_files 2 --max_windows 10
```

**Expected output:**
```
[Reading CSV] burst-200mV.csv
  Loaded 123,456 events, duration: 900.00s
  Processing 10 windows (1.0s each)
  Saved 10 windows

[Reading CSV] burst-300mV.csv
  Loaded 234,567 events, duration: 900.00s
  Processing 10 windows (1.0s each)
  Saved 10 windows

[Manifest] Saved to ./data/processed/manifest.csv
  Total windows: 20
```

**Check the output:**
```bash
ls data/processed/
cat data/processed/manifest.csv | head -5
```

---

### Step 3: Process One Complete File

Process one full CSV file (all ~900 windows):

```bash
python preprocess.py --config config.yaml --max_files 1
```

This will take a few minutes and create ~900 window samples from one CSV.

---

### Step 4: Process All Files (Full Dataset)

**WARNING: This will take 1-4 hours depending on your data size!**

```bash
# Process all 16 CSV files
python preprocess.py --config config.yaml

# Monitor progress (run in another terminal)
watch -n 5 'ls data/processed/*.npy | wc -l'
```

**Expected result:**
- Input: 16 CSV files (~900 seconds each)
- Output: ~14,400 training samples (16 files × 900 windows)
- Each sample: `.npy` file with shape `[2, 128, 16, 16]`

---

### Step 5: Verify Preprocessed Data

Check the manifest and statistics:

```bash
python preprocess.py --config config.yaml --stats_only
```

**Expected output:**
```json
{
  "total_windows": 14400,
  "waveform_distribution": {
    "burst": 3600,
    "sine": 3600,
    "square": 3600,
    "triangle": 3600
  },
  "voltage_distribution": {
    "200mV": 3600,
    "300mV": 3600,
    "400mV": 3600,
    "500mV": 3600
  }
}
```

---

### Step 6: Test the Dataset Loader

```bash
python dataset.py
```

This will:
- Load the manifest
- Create train/val/test splits
- Show a sample batch
- Verify shapes and labels

---

### Step 7: Quick Training Test (5 Epochs)

Test that training works before running full training:

Edit `config.yaml` temporarily:
```yaml
training:
  num_epochs: 5  # Change from 100 to 5 for quick test
  batch_size: 16  # Smaller batch if GPU memory is tight
```

Then run:
```bash
python train.py --config config.yaml
```

**Expected:**
- 5 epochs should complete in ~5-10 minutes
- You'll see training and validation metrics
- Checkpoint saved to `checkpoints/`

**Monitor with TensorBoard:**
```bash
tensorboard --logdir runs/
# Open browser to http://localhost:6006
```

---

### Step 8: Full Training (100 Epochs)

Reset `num_epochs` back to 100 in `config.yaml`, then:

```bash
# Full training (will take 4-12 hours)
python train.py --config config.yaml

# Monitor in another terminal
tensorboard --logdir runs/
```

---

### Step 9: Evaluate the Trained Model

```bash
python eval.py --config config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --save_plots \
    --save_predictions
```

**Outputs:**
- `results/confusion_matrices.png`
- `results/per_class_metrics.png`
- `results/predictions.csv`
- Terminal output with accuracy, precision, recall, F1

---

### Step 10: Interactive Demo

```bash
jupyter notebook QTCR_Net_Demo.ipynb
```

Run through the notebook to see:
- Data loading
- Visualization
- Model architecture
- Training demo
- Evaluation results

---

## 📊 Expected Timeline

| Task | Duration |
|------|----------|
| Setup & verification | 2 min |
| Visualization (1 file) | 1 min |
| Test preprocessing (2 files) | 2 min |
| Full preprocessing (16 files) | 1-4 hours |
| Test training (5 epochs) | 5-10 min |
| Full training (100 epochs) | 4-12 hours |
| Evaluation | 5 min |

---

## 💾 Disk Space Requirements

- **Raw CSV files:** ~1-2 GB (your current data)
- **Preprocessed .npy files:** ~2-4 GB (14,400 windows)
- **Checkpoints:** ~500 MB - 1 GB
- **Logs & results:** ~100 MB

**Total: ~4-8 GB**

---

## 🎯 Quick Commands Cheat Sheet

```bash
# 1. Setup
cd ~/MatlabToolBox
source venv/bin/activate
python setup_my_data.py

# 2. Visualize
python visualize_my_data.py sine-300mV --all

# 3. Preprocess (test)
python preprocess.py --config config.yaml --max_files 2 --max_windows 10

# 4. Preprocess (full)
python preprocess.py --config config.yaml

# 5. Train (test)
python train.py --config config.yaml  # (set num_epochs=5 first)

# 6. Train (full)
python train.py --config config.yaml  # (set num_epochs=100)

# 7. Evaluate
python eval.py --config config.yaml --checkpoint checkpoints/best_model.pth --save_plots

# 8. Demo
jupyter notebook QTCR_Net_Demo.ipynb
```

---

## 🐛 Troubleshooting

### "No CSV files found"
- Check path: `ls ~/Desktop/QuantumNetwork/data/raw_truncated/`
- Verify config.yaml has correct `csv_dir`

### "Could not parse metadata from filename"
- This is now fixed! Files with hyphens (e.g., `sine-300mV.csv`) are handled correctly

### "Out of memory" during preprocessing
- Reduce `max_windows` to process in smaller chunks
- Close other applications

### "CUDA out of memory" during training
- Reduce `batch_size` in config.yaml (try 8 or 16)
- Reduce `num_groups` or `qubits_per_group`

### Preprocessing is slow
- This is normal for large files (200-540M events each)
- Use `--max_files` to process incrementally
- Consider running overnight for full dataset

---

## 📈 Expected Results

Based on your 16-file dataset:

### Dataset Split (70/15/15)
- **Training:** ~10,080 samples
- **Validation:** ~2,160 samples
- **Test:** ~2,160 samples

### Expected Performance
- **Waveform accuracy:** 85-95%
- **Voltage accuracy:** 80-90%

### Training Time (RTX 5070)
- **Per epoch:** ~3-7 minutes
- **100 epochs:** ~5-12 hours

---

## 🎉 You're Ready!

Your data is all set up and ready to go. Start with:

```bash
python setup_my_data.py
```

Then follow the steps above. Good luck with your research! 🚀
