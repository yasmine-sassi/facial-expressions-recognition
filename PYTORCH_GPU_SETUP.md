# 🚀 PyTorch GPU Setup Guide

Complete guide to setting up PyTorch with GPU acceleration on your RTX 3050.

## ✅ Hardware Check

Your system has:

- **GPU:** NVIDIA GeForce RTX 3050 (4GB VRAM)
- **Driver:** Version 566.07
- **CUDA Capability:** 12.7 (supports CUDA 12.1)

---

## 📋 Installation Steps

### Step 1: Create Fresh Python Environment

```bash
# Create new conda environment
conda create -n pytorch_gpu python=3.10 -y

# Activate environment
conda activate pytorch_gpu
```

**Not using Conda?** Use venv instead:

```bash
python -m venv pytorch_gpu
.\pytorch_gpu\Scripts\activate  # Windows
```

---

### Step 2: Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Key Points:**

- Uses **CUDA 12.1** (compatible with your system)
- Downloads ~2.5GB
- Includes PyTorch, TorchVision, TorchAudio

**Alternative (if download fails):** Install step-by-step

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 3: Install Additional Dependencies

```bash
# From project directory
cd projet_traitement_img

# Install all requirements
pip install -r requirements_pytorch.txt
```

---

### Step 4: Verify GPU Setup

Run this **immediately after installation:**

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB" if torch.cuda.is_available() else "N/A")
```

**Expected Output:**

```
PyTorch version: 2.1.0
CUDA available: True
CUDA version: 12.1
GPU name: NVIDIA GeForce RTX 3050
GPU Memory: 4.0 GB
```

❌ **If CUDA says False:** See troubleshooting section below

---

### Step 5: Install Jupyter for Notebook Development

```bash
pip install jupyter ipykernel

# Register kernel
python -m ipykernel install --user --name=pytorch_gpu --display-name="PyTorch GPU"
```

---

### Step 6: Start Jupyter and Select Kernel

```bash
jupyter notebook
```

Then in any notebook:

1. Kernel → Change Kernel → **PyTorch GPU**
2. Run test cell (see Step 4 above)

---

## 🧪 Quick GPU Test

**Python script test:**

```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

**Should print:** `GPU: NVIDIA GeForce RTX 3050`

---

## 🐛 Troubleshooting

### Problem: `CUDA available: False`

**Most Common Causes:**

1. **Wrong PyTorch version installed**

   ```bash
   pip list | grep torch
   ```

   Check if version includes `+cu121` or `cu121`

2. **Installed CPU-only version**

   ```bash
   # Uninstall and reinstall
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **NVIDIA driver outdated**
   - Check: Settings → About → Device Specifications
   - Expected: Driver version 566.07 ✓
   - Update: https://www.nvidia.com/Download/driverDetails.aspx

4. **Jupyter kernel not updated**
   ```bash
   # Reinstall kernel after PyTorch
   python -m ipykernel install --user --name=pytorch_gpu --force
   # Then restart Jupyter
   ```

---

### Problem: `RuntimeError: CUDA out of memory`

**Solutions:**

1. Reduce batch size (e.g., 32 → 16)
2. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
3. Check GPU usage:
   ```bash
   nvidia-smi
   ```
4. Close other GPU-using programs

---

### Problem: `ImportError: No module named 'torch'`

```bash
# Verify environment is activated
conda activate pytorch_gpu

# Reinstall PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

---

## 📊 Monitor GPU Usage

### Real-time GPU Monitoring

While training:

```bash
# Live GPU stats (refresh every 1 sec)
nvidia-smi -l 1
```

Look for:

- Process name: `python`
- GPU Memory: Should increase during training
- GPU-Util: Should be 10-100%

### In Python

```python
import torch

def check_gpu():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")

    # During training, allocated memory should grow

check_gpu()
```

---

## ⚡ Performance Tips

### Maximize GPU Utilization

1. **Always move data to GPU:**

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   X = X.to(device)  # ← Don't forget this!
   ```

2. **Optimal batch size for 4GB GPU:** 32-64
   - Smaller = slower but safer
   - Larger = faster but OOM risk

3. **Use mixed precision (speeds up by ~1.5x):**

   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast():
       output = model(X)
   ```

4. **Pin memory for faster CPU→GPU transfer:**
   ```python
   train_loader = DataLoader(dataset, pin_memory=True)
   ```

---

## 📈 Expected Performance Improvements

**RTX 3050 (GPU) vs CPU:**

| Task                               | CPU Time | GPU Time                   | Speedup           |
| ---------------------------------- | -------- | -------------------------- | ----------------- |
| Baseline CNN Training (100 epochs) | ~45 min  | ~5-8 min                   | **6-9x faster**   |
| Image Inference (1000 images)      | ~30s     | ~1-2s                      | **15-30x faster** |
| Data Loading                       | N/A      | Faster (GPU preprocessing) | **2-3x**          |

---

## 🎯 Next Steps

After verification:

1. **Run baseline training:** `python notebooks/04_pytorch_baseline.ipynb`
2. **Compare models:** `python notebooks/05_pytorch_advanced.ipynb`
3. **Evaluate results:** Check saved models in `saved_models/`
4. **Deploy with Streamlit:** `streamlit run frontend/front.py`

---

## 📚 Resources

- **PyTorch Docs:** https://pytorch.org/docs/stable/index.html
- **CUDA Debugging:** https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- **GPU Memory Management:** https://pytorch.org/docs/stable/notes/cuda.html

---

## ✅ Checklist

Before starting training:

- [ ] Activated `pytorch_gpu` environment
- [ ] PyTorch installed with CUDA 12.1
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] `nvidia-smi` shows 4GB GPU memory
- [ ] Jupyter kernel is `PyTorch GPU`
- [ ] Test cell runs successfully
- [ ] Prerequisites data in `../data/preprocessed/` directory

---

## 💬 Common Questions

**Q: Will my RTX 3050 be enough?**
A: Yes! Perfect for:

- CNNs (MNIST, CIFAR, FER2013)
- Vision Transformers (small)
- RNNs, LSTMs
- Fine-tuning pre-trained models

Not recommended for:

- Large VGG/ResNet variants on ImageNet
- Large language models
- Heavy augmentation + large batches

**Q: Do I need to rewrite my code?**
A: Minimal changes needed:

```python
# Before (TensorFlow)
model.fit(X_train, y_train)

# After (PyTorch)
model.to(device)  # Move to GPU
X_train.to(device)
outputs = model(X_train)
```

**Q: Can I use both GPU and CPU?**
A: Yes! PyTorch handles it:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
# Code works on both
```

**Q: How do I update NVIDIA driver?**
A:

1. Visit https://www.nvidia.com/Download/index.aspx
2. Select your GPU model
3. Download + Install
4. Restart computer

---

**You're all set! Welcome to GPU-accelerated deep learning! 🔥**
