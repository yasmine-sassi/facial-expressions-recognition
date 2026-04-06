# Facial Emotion Recognition (FER2013) - PyTorch Edition

A deep learning project for recognizing facial emotions using PyTorch Convolutional Neural Networks (CNN) on the FER2013 dataset with GPU acceleration support.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Hardware & System Setup](#hardware--system-setup)
3. [Installation & Setup](#installation--setup)
4. [Project Structure](#project-structure)
5. [Dataset Overview](#dataset-overview)
6. [Architecture & Models](#architecture--models)
7. [Preprocessing Pipeline](#preprocessing-pipeline)
8. [Quick Start Guide](#quick-start-guide)
9. [Running the Project](#running-the-project)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project implements a complete facial emotion recognition pipeline using PyTorch with GPU acceleration:

- **Dataset**: FER2013 (35,887 images across 7 emotion classes)
- **Framework**: PyTorch 2.1+ with CUDA 12.1
- **GPU**: NVIDIA GeForce RTX 3050 (4GB VRAM)
- **Models**: Baseline CNN, Advanced CNN, ResNet-based architectures
- **Performance**: Class-balanced training with early stopping

---

## ⚙️ Hardware & System Setup

### Your Hardware

- **GPU**: NVIDIA GeForce RTX 3050 (4GB VRAM)
- **CUDA Capability**: 12.7 (supports CUDA 12.1)
- **Driver Version**: 566.07+
- **Python**: 3.10

### GPU Verification

Check your GPU status anytime:

```bash
nvidia-smi
```

---

## 🚀 Installation & Setup

### Step 1: Create Python Environment

**Using Conda (Recommended):**

```bash
conda create -n pytorch_gpu python=3.10 -y
conda activate pytorch_gpu
```

**Or Using venv:**

```bash
python -m venv pytorch_gpu
.\pytorch_gpu\Scripts\activate  # Windows
```

### Step 2: Install PyTorch with GPU Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

This installs PyTorch with CUDA 12.1 (optimized for RTX 3050).

### Step 3: Install Project Dependencies

```bash
pip install -r requirements_pytorch.txt
```

### Step 4: Verify GPU Setup

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```

**Expected Output:**

```
PyTorch version: 2.1.0
CUDA available: True
CUDA version: 12.1
GPU name: NVIDIA GeForce RTX 3050
GPU Memory: 4.0 GB
```

### Step 5: Install Jupyter (Optional, for notebooks)

```bash
pip install jupyter notebook
```

---

## 📁 Project Structure

```
projet_traitement_img/
├── data/                         # Dataset and preprocessed data
│   ├── train/                    # Training images (3,171 images)
│   ├── test/                     # Test images
│   └── preprocessed/             # Generated .npy files
│       ├── X_train.npy          # Training images (22,968, 48, 48, 1)
│       ├── y_train.npy          # Training labels
│       ├── X_val.npy            # Validation images (5,741, 48, 48, 1)
│       ├── y_val.npy            # Validation labels
│       ├── X_test.npy           # Test images (7,178, 48, 48, 1)
│       ├── y_test.npy           # Test labels
│       ├── class_weights.json    # Balanced class weights
│       └── class_mapping.json    # Emotion label mapping
│
├── notebooks/                    # Jupyter notebooks (PyTorch)
│   ├── 01_Preprocessing.ipynb    # Data loading & preprocessing
│   ├── 04_pytorch_baseline.ipynb # Baseline CNN training
│   └── 05_pytorch_advanced.ipynb # Advanced architectures (ResNet)
│
├── src/                          # Reusable PyTorch modules
│   ├── __init__.py
│   ├── pytorch_models.py         # CNN architectures
│   ├── pytorch_train.py          # Training utilities with GPU support
│   └── pytorch_evaluate.py       # Evaluation metrics & visualization
│
├── frontend/
│   └── front_pytorch.py          # Streamlit demo with PyTorch
│
├── results/                      # Generated visualizations
├── saved_models/                 # Trained PyTorch models (.pt)
│   ├── pytorch_baseline_cnn_*
│   ├── pytorch_advanced_cnn_*
│   └── pytorch_resnet_*
│
├── requirements_pytorch.txt      # PyTorch dependencies
└── README.md                     # This file
```

---

## 📊 Dataset Overview

### Total Images: **35,887**

| Emotion  | Train | Test  | Total | Class Weight |
| -------- | ----- | ----- | ----- | ------------ |
| Angry    | 3,995 | 958   | 4,953 | 1.0266×      |
| Disgust  | 436   | 111   | 547   | 9.4016× ⚠️   |
| Fear     | 4,097 | 1,024 | 5,121 | 1.0010×      |
| Happy    | 7,215 | 1,774 | 8,989 | 0.5685×      |
| Neutral  | 4,965 | 1,233 | 6,198 | 0.8261×      |
| Sad      | 4,830 | 1,247 | 6,077 | 0.8492×      |
| Surprise | 3,171 | 831   | 4,002 | 1.2933×      |

### Key Challenge

Class imbalance ratio: **16.5×** (Happy vs Disgust)

- **Solution**: Balanced class weights during training
- **Quality**: 99.96% valid images (35,874/35,887)

---

## 🏗️ Architecture & Models

### Available Models

All models implemented in `src/pytorch_models.py`:

#### 1. **Baseline CNN**

- 3 convolutional blocks
- Batch normalization & dropout
- Simple fully connected head
- **Best for**: Quick training & validation
- **Training time**: ~5-10 minutes (GPU)

```python
from src.pytorch_models import BaselineCNN
model = BaselineCNN()
```

#### 2. **Advanced CNN**

- 4 convolutional blocks
- Progressive feature expansion
- Enhanced dropout & regularization
- **Best for**: Balanced accuracy & training speed
- **Training time**: ~15-20 minutes (GPU)

```python
from src.pytorch_models import AdvancedCNN
model = AdvancedCNN()
```

#### 3. **ResNet-Based Model**

- Residual connections
- Deeper architecture
- Transfer learning potential
- **Best for**: Maximum accuracy
- **Training time**: ~30-40 minutes (GPU)

```python
from src.pytorch_models import ResNetEmotion
model = ResNetEmotion()
```

### Model Specifications

- **Input**: 48×48 grayscale images
- **Output**: 7-dimensional emotion logits
- **Framework**: Pure PyTorch (no .keras dependencies)
- **Device**: Automatically uses GPU if available

---

## 🔄 Preprocessing Pipeline

Complete 3-step workflow in `notebooks/01_Preprocessing.ipynb`:

### Step 1: Data Cleaning

✅ **Quality Assessment**:

- Corrupted file detection: 0 files
- Dimension validation (min 40×40): 0 files
- Contrast filtering (std dev ≥ 10): 13 anomalies removed
- **Overall quality: 99.96%**

### Step 2: Standardization

✅ **Resizing**:

- Output: 48×48 pixels (uniform)
- Color mode: Grayscale (1 channel)
- Interpolation: Linear
- Result: All images as (48, 48, 1) tensors

### Step 3: Normalization & Enhancement

✅ **Normalization**:

```
rescale = 1.0 / 255
Pixel range: [0, 255] → [0, 1]
Mean: 0.50 | Std: 0.25
```

---

## ⚡ Quick Start Guide

### Fastest Path (5 Minutes)

```bash
# 1. Activate environment
conda activate pytorch_gpu

# 2. Start Jupyter
jupyter notebook

# 3. Open notebooks/04_pytorch_baseline.ipynb
# 4. Run all cells to train baseline model
```

### Training Options

**Option A: Baseline Model** (Quick)

```
Open: notebooks/04_pytorch_baseline.ipynb
Time: ~5-10 minutes
Accuracy: ~65-70%
```

**Option B: Advanced Models** (Comprehensive)

```
Open: notebooks/05_pytorch_advanced.ipynb
Time: ~30-50 minutes
Accuracy: ~72-75%
Includes: Advanced CNN + ResNet
```

---

## 🎮 Running the Project

### Train a Model

```bash
# Activate environment
conda activate pytorch_gpu

# Start Jupyter
jupyter notebook

# Open notebooks/04_pytorch_baseline.ipynb or 05_pytorch_advanced.ipynb
```

### Use Trained Model in Web Demo

```bash
streamlit run frontend/front_pytorch.py
```

### Evaluate Model Performance

In Jupyter notebook:

```python
from src.pytorch_evaluate import evaluate_model
from src.pytorch_models import BaselineCNN
import torch

# Load trained model
model = BaselineCNN()
model.load_state_dict(torch.load('saved_models/pytorch_baseline_cnn_best.pt'))

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(metrics)
```

### Monitor GPU Usage During Training

In a separate terminal:

```bash
watch -n 1 nvidia-smi
```

Or one-time check:

```bash
nvidia-smi
```

---

## 🔧 Common Workflow

### Every Time You Start

```bash
conda activate pytorch_gpu
jupyter notebook
```

### After Installing New Packages

```bash
pip install -r requirements_pytorch.txt
```

### Check GPU Status

```bash
nvidia-smi
```

### Run Specific Training Script

```python
# In Jupyter cell or Python script
from src.pytorch_train import train_model
from src.pytorch_models import BaselineCNN

model = BaselineCNN()
trained_model = train_model(model, X_train, y_train, X_val, y_val, epochs=50)
```

---

## ❌ Troubleshooting

### GPU Not Detected (CUDA available: False)

**Solution:**

```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### OutOfMemory Error on GPU

**Solutions:**

1. Reduce batch size in training script
2. Reduce image resolution
3. Use gradient accumulation
4. Check GPU memory with `nvidia-smi`

### Notebook Kernel Not Found

**Solution:**

```bash
# Install Jupyter in current environment
pip install jupyter notebook
```

### Slow Training without GPU

**Verify:**

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GeForce RTX 3050
```

If False, reinstall PyTorch with CUDA support (see GPU Not Detected).

---

## 📞 Key Dependencies

- **PyTorch**: 2.1.0+
- **TorchVision**: Latest
- **NumPy**: Latest
- **Matplotlib**: Latest
- **Scikit-learn**: Latest
- **Streamlit**: Latest

See `requirements_pytorch.txt` for complete list.

---

## 📝 Notes

- This project uses **PyTorch only** (Keras/TensorFlow removed)
- GPU acceleration is optional but significantly speeds up training
- Notebooks auto-save results to `results/` and `saved_models/`
- All models use 7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

---

**Happy training! 🚀**
