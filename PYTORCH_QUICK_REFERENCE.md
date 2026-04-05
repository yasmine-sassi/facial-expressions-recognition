# 🎯 Quick Reference: PyTorch GPU Setup

Fast commands for your workflow.

## 🚀 Start Here (First Time Only)

```bash
# 1. Create environment
conda create -n pytorch_gpu python=3.10 -y
conda activate pytorch_gpu

# 2. Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements_pytorch.txt

# 4. Verify GPU
python -c "import torch; print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '✗ Not available')"

# 5. Start Jupyter
jupyter notebook
```

---

## 📖 Daily Workflow

### Activate Environment (Every Time)

```bash
conda activate pytorch_gpu
```

### Start Jupyter for Notebooks

```bash
jupyter notebook
```

### Train Baseline Model

```
Open: notebooks/04_pytorch_baseline.ipynb
Select Kernel: PyTorch GPU
Run All Cells
```

### Train Advanced Models

```
Open: notebooks/05_pytorch_advanced.ipynb
Select Kernel: PyTorch GPU
Run All Cells
```

### Use Trained Model in Streamlit

```bash
streamlit run frontend/front_pytorch.py
```

---

## 📊 File Structure

**New PyTorch Files:**

```
src/
├── pytorch_models.py      # Models: BaselineCNN, AdvancedCNN, ResNet
├── pytorch_train.py       # Training utilities with GPU support
└── pytorch_evaluate.py    # Evaluation & visualization

notebooks/
├── 04_pytorch_baseline.ipynb    # Baseline CNN training
└── 05_pytorch_advanced.ipynb    # Advanced CNN & ResNet training

frontend/
└── front_pytorch.py       # Streamlit with PyTorch support

saved_models/
├── pytorch_baseline_cnn_best.pt
├── pytorch_advanced_cnn_best.pt
├── pytorch_resnet_emotion_best.pt
└── pytorch_best_model.pt
```

---

## ⚡ Common Commands

### Check GPU Status

```bash
nvidia-smi
```

### While Training (GPU Monitoring)

```bash
nvidia-smi -l 1  # Refresh every 1 second
```

### Clear GPU Cache (If OOM Error)

```python
import torch
torch.cuda.empty_cache()
```

### Test GPU in Python

```python
import torch
print(torch.cuda.is_available())           # Should be True
print(torch.cuda.get_device_name(0))       # GPU name
print(torch.cuda.get_device_properties(0)) # GPU specs
```

---

## 🎯 Quick Training

### Baseline Only (~5-10 min on GPU)

1. Open `notebooks/04_pytorch_baseline.ipynb`
2. Run all cells
3. Model saved: `saved_models/pytorch_baseline_cnn_best.pt`

### Full Comparison (~25 min on GPU)

1. Open `notebooks/05_pytorch_advanced.ipynb`
2. Run all cells
3. Best model auto-selected and saved

---

## 🖥️ Expected Output

### Training Output

```
======================================================
Training pytorch_baseline_cnn
Device: cuda
======================================================

Epoch [5/100] - Train Loss: 1.8432, Train Acc: 0.4521 | Val Loss: 1.7623, Val Acc: 0.4832 | LR: 1.0e-03
Epoch [10/100] - Train Loss: 1.5234, Train Acc: 0.5123 | Val Loss: 1.4532, Val Acc: 0.5312 | LR: 1.0e-03
...
Early stopping at epoch 65
Best model saved to saved_models/pytorch_baseline_cnn_best.pt
```

### GPU Monitoring (nvidia-smi)

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 566.07       Driver Version: 566.07       CUDA Version: 12.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce RTX 3050  On   | 00:1F.0     Off |                  N/A |
| N/A   45C    P2    12W /  60W |   1200MiB /  4096MiB |    80%      Default |
+-------------------------------+----------------------+----------------------+
```

---

## ✅ Checklist

Before each training session:

- [ ] `conda activate pytorch_gpu` (shows `pytorch_gpu` in prompt)
- [ ] `nvidia-smi` shows GPU memory available
- [ ] Jupyter kernel set to "PyTorch GPU"
- [ ] Data exists: `../data/preprocessed/` ✓
- [ ] Sufficient GPU memory (~1-2GB per model)

---

## 🔧 Troubleshooting Quick Fixes

| Problem                  | Fix                                                                                           |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| CUDA not available       | Run: `pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall` |
| CUDA out of memory       | Reduce batch_size from 32 to 16, then `torch.cuda.empty_cache()`                              |
| Kernel not found         | Stop Jupyter, reinstall: `python -m ipykernel install --user --name=pytorch_gpu --force`      |
| Wrong environment active | Run: `conda activate pytorch_gpu`                                                             |
| GPU memory not freeing   | Restart Jupyter kernel (Kernel → Restart)                                                     |

---

## 📈 Performance Guidelines

**Training Time (100 epochs, batch=32):**

- GPU: 5-15 min (depending on model)
- CPU: 45-60 min (6-9x slower!)

**GPU Memory Usage:**

- During training: 800MB - 1.5GB
- Available on RTX 3050: 4GB total
- Safety margin: ~2x headroom

**Inference Speed (per image):**

- GPU: ~10-20ms
- CPU: ~150-300ms (15-30x slower!)

---

## 📚 Key Files

| File                                  | Purpose                  |
| ------------------------------------- | ------------------------ |
| `PYTORCH_GPU_SETUP.md`                | Detailed GPU setup guide |
| `PYTORCH_MIGRATION_GUIDE.md`          | Full Keras→PyTorch guide |
| `notebooks/04_pytorch_baseline.ipynb` | Train baseline model     |
| `notebooks/05_pytorch_advanced.ipynb` | Compare 3 models         |
| `src/pytorch_models.py`               | Model definitions        |
| `src/pytorch_train.py`                | Training code            |
| `src/pytorch_evaluate.py`             | Evaluation code          |
| `frontend/front_pytorch.py`           | Streamlit inference app  |

---

## 🚀 Your First Run

```bash
# Terminal
conda activate pytorch_gpu
jupyter notebook

# Browser: Open notebooks/04_pytorch_baseline.ipynb
# Kernel: Select "PyTorch GPU"
# Run: Cell → Run All
# Wait: ~5-10 minutes
# Check: Look for "Training complete!" message

# Result: Model saved to saved_models/pytorch_baseline_cnn_best.pt
```

---

## 💡 Pro Tips

1. **Monitor GPU**: Open second terminal, run `nvidia-smi -l 1`
2. **Early Stop**: Training often stops at epoch 40-50 (that's normal!)
3. **Batch Size**: Smaller is safer, larger is faster
4. **Learning Rate**: Default 0.001 works well, adjust if needed
5. **GPU Cleanup**: After training ends, GPU memory is freed automatically

---

## 🎓 Resources

| Resource          | Link                              |
| ----------------- | --------------------------------- |
| PyTorch Tutorials | https://pytorch.org/tutorials/    |
| CUDA Debugging    | https://docs.nvidia.com/cuda/     |
| Colab (Free GPU)  | https://colab.research.google.com |

---

**You're all set! Start training! 🔥**

Last updated: 2024-2025
