# QTCR-Net Quick Start Guide

Get up and running with QTCR-Net in 10 minutes!

## 1. Installation (2 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/qtcr-net.git
cd qtcr-net

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

## 2. Prepare Your Data (5 minutes)

### Option A: Use Your DVS128 CSV Files

1. Place your CSV files in `~/Desktop/QuantumNetwork/CSVs/`
2. Ensure CSVs have columns: `timestamp_us, x, y, polarity`
3. Update paths in `config.yaml` if needed

### Option B: Test with Sample Data

```bash
# Generate synthetic test data (if you don't have real DVS data yet)
python generate_test_data.py  # Create a few sample CSV files
```

## 3. Preprocess Data (2 minutes for test, longer for full dataset)

```bash
# Quick test with 2 files, 10 windows each
python preprocess.py --config config.yaml --max_files 2 --max_windows 10

# Full preprocessing (may take hours for large datasets)
python preprocess.py --config config.yaml
```

## 4. Train QTCR-Net (1 minute for demo, hours for full training)

```bash
# Quick demo training (5 epochs)
python train.py --config config.yaml  # Edit num_epochs in config.yaml first

# Full training (100 epochs)
python train.py --config config.yaml
```

**Monitor training:**
```bash
tensorboard --logdir runs/
```

## 5. Evaluate Model

```bash
python eval.py --config config.yaml \
               --checkpoint checkpoints/best_model.pth \
               --save_plots \
               --save_predictions
```

## 6. Interactive Demo

```bash
jupyter notebook QTCR_Net_Demo.ipynb
```

---

## Configuration Tips

### For Quick Testing (Laptop/Small GPU)
```yaml
# config.yaml
training:
  batch_size: 8          # Reduce batch size
  num_epochs: 10         # Few epochs

model:
  quantum_reservoir:
    num_groups: 4        # Fewer quantum groups
    qubits_per_group: 4  # Fewer qubits
```

### For Full Training (RTX 5070)
```yaml
# config.yaml (default)
training:
  batch_size: 24
  num_epochs: 100

model:
  quantum_reservoir:
    num_groups: 6
    qubits_per_group: 6
```

---

## Troubleshooting

### Out of Memory?
- Reduce `batch_size` in config.yaml
- Reduce `num_groups` or `qubits_per_group`

### Slow Training?
- Install `pennylane-lightning`: `pip install pennylane-lightning`
- Enable mixed precision (already enabled by default)
- Reduce `num_workers` if CPU is bottleneck

### No Data?
- Check CSV paths in config.yaml
- Run preprocessing first
- Verify manifest.csv exists: `data/processed/manifest.csv`

---

## Expected Timeline

| Task | Quick Test | Full Run |
|------|------------|----------|
| Installation | 2 min | 2 min |
| Preprocessing | 2 min | 1-4 hours |
| Training | 5 min | 4-12 hours |
| Evaluation | 1 min | 5 min |

---

## Next Steps

1. ✅ **Run test installation**: `python test_installation.py`
2. ✅ **Preprocess small dataset**: `python preprocess.py --max_files 2`
3. ✅ **Train for 5 epochs**: Edit config.yaml, run `python train.py`
4. ✅ **Evaluate**: `python eval.py --checkpoint checkpoints/best_model.pth`
5. ✅ **Explore notebook**: `jupyter notebook QTCR_Net_Demo.ipynb`
6. 🚀 **Full training**: Scale up to full dataset and 100 epochs
7. 📊 **Experiments**: Try different hyperparameters, compare baselines

---

## Help

- 📖 **Full documentation**: See `README.md`
- 🐛 **Issues**: Open an issue on GitHub
- 💬 **Questions**: Check FAQ in README or contact the team

---

**Ready? Let's start!**

```bash
python test_installation.py
```
