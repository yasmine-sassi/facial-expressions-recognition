# 🚀 TensorFlow to PyTorch Migration Guide

Complete guide for migrating your facial emotion recognition project from TensorFlow/Keras to PyTorch with GPU acceleration.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [What's Changed](#whats-changed)
3. [Installation & Setup](#installation--setup)
4. [Running the Training](#running-the-training)
5. [Model Architecture Comparison](#model-architecture-comparison)
6. [Performance Comparison](#performance-comparison)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Start

**TL;DR - Get PyTorch running in 5 minutes:**

```bash
# 1. Create environment
conda create -n pytorch_gpu python=3.10 -y
conda activate pytorch_gpu

# 2. Install PyTorch with GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install -r requirements_pytorch.txt

# 4. Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# 5. Run training notebook
jupyter notebook notebooks/04_pytorch_baseline.ipynb
```

---

## 📊 What's Changed

### Project Structure

**Before (TensorFlow):**

```
src/
├── model.py              # Keras models
├── train.py              # Keras training
├── evaluate.py           # Keras evaluation
└── predict.py

notebooks/
├── 02_baseline_cnn.ipynb         # Keras baseline training
├── 03_advanced_model.ipynb       # Keras advanced training
└── 04_evaluation.ipynb
```

**After (PyTorch):**

```
src/
├── pytorch_models.py    # 🆕 PyTorch models (Baseline, Advanced, ResNet)
├── pytorch_train.py     # 🆕 PyTorch training utilities
├── pytorch_evaluate.py  # 🆕 PyTorch evaluation utilities
├── model.py              # (Old) Keras models - still available
├── train.py              # (Old) Keras training - still available
└── evaluate.py

notebooks/
├── 02_baseline_cnn.ipynb              # (Old) Keras training
├── 03_advanced_model.ipynb            # (Old) Keras training
├── 04_pytorch_baseline.ipynb          # 🆕 PyTorch baseline
├── 05_pytorch_advanced.ipynb          # 🆕 PyTorch advanced & ResNet
└── 04_evaluation.ipynb

frontend/
├── front.py                # (Old) Keras-only Streamlit
├── front_pytorch.py        # 🆕 Dual framework Streamlit
└── requirements.txt

saved_models/
├── baseline_cnn_best.keras         # Keras model
├── pytorch_baseline_cnn_best.pt    # 🆕 PyTorch model
├── pytorch_advanced_cnn_best.pt    # 🆕 PyTorch model
├── pytorch_resnet_emotion_best.pt  # 🆕 PyTorch model
└── pytorch_best_model.pt           # 🆕 Best PyTorch model for deployment
```

---

## 🔧 Installation & Setup

### Step 1: Read the GPU Setup Guide

👉 See: `PYTORCH_GPU_SETUP.md` for detailed GPU configuration

### Step 2: Environment Activation

```bash
# Activate environment
conda activate pytorch_gpu

# Verify activation (should show pytorch_gpu)
conda info | grep active
```

### Step 3: Verify Installation

```python
import torch
import sys
sys.path.insert(0, '.')

# Test GPU
print("✓ GPU Available:", torch.cuda.is_available())
print("✓ GPU Name:", torch.cuda.get_device_name(0))
print("✓ PyTorch Version:", torch.__version__)

# Test models
from src.pytorch_models import BaselineCNN, AdvancedCNN, ResNetEmotion
model = BaselineCNN()
print("✓ Models loaded successfully")
```

---

## 🏃 Running the Training

### Option 1: Baseline Model Training (Recommended for First Run)

```bash
# Terminal 1: Start Jupyter
jupyter notebook

# Browser: Navigate to notebooks/04_pytorch_baseline.ipynb
# Run all cells to train the baseline model
```

**What happens:**

1. Loads preprocessed data from `../data/preprocessed/`
2. Creates DataLoaders for GPU
3. Builds Baseline CNN (2.7M parameters)
4. Trains with early stopping and LR scheduling
5. Saves best model to `saved_models/pytorch_baseline_cnn_best.pt`
6. Evaluates on test set
7. Generates visualizations (confusion matrix, per-class metrics)

**Expected runtime:**

- GPU (4GB RTX 3050): **5-10 minutes**
- CPU: ~45 minutes (significant difference!)

---

### Option 2: Advanced Models Comparison

```bash
# Run notebook: 05_pytorch_advanced.ipynb

# This trains:
# 1. Advanced CNN (7.2M parameters) - ~8-12 min
# 2. ResNet (4M parameters) - ~10-15 min
# Then compares both on test set
```

---

## 🔄 Model Architecture Comparison

### Baseline CNN

**Keras Version:**

```python
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(48,48,1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(7)
])
```

**PyTorch Equivalent:**

```python
from src.pytorch_models import BaselineCNN
model = BaselineCNN(num_classes=7)

# Move to GPU
device = torch.device('cuda')
model = model.to(device)
```

✅ **Same architecture, faster training!**

---

### Advanced CNN

**PyTorch Only** (New model):

```python
from src.pytorch_models import AdvancedCNN
model = AdvancedCNN(num_classes=7)

# 4 conv blocks with double convolutions
# Global average pooling
# 7.2M parameters
# Better accuracy expectations
```

---

### ResNet (Bonus)

**PyTorch Only** (New experimental model):

```python
from src.pytorch_models import ResNetEmotion
model = ResNetEmotion(num_classes=7)

# ResNet with skip connections
# Better for deep networks
# 4M+ parameters
```

---

## 📈 Performance Comparison

### Training Speed (100 epochs, 32 batch size)

| Model        | CPU     | GPU (RTX 3050) | Speedup  |
| ------------ | ------- | -------------- | -------- |
| Baseline CNN | ~45 min | 5-8 min        | **6-9x** |
| Advanced CNN | ~60 min | 8-12 min       | **6-8x** |
| ResNet       | ~50 min | 10-15 min      | **4-6x** |

### Memory Usage

| Model    | GPU VRAM | CPU RAM |
| -------- | -------- | ------- |
| Baseline | ~800 MB  | 2.5 GB  |
| Advanced | ~1.2 GB  | 2.8 GB  |
| ResNet   | ~1.0 GB  | 2.7 GB  |

**Note:** RTX 3050 has 4GB total, so all models fit comfortably!

### Expected Model Accuracy (Test Set)

| Model    | Framework | Accuracy | F1-Score   |
| -------- | --------- | -------- | ---------- |
| Baseline | Keras     | 49.5%    | 0.47       |
| Baseline | PyTorch   | ~50-52%  | ~0.49-0.51 |
| Advanced | PyTorch   | ~52-55%  | ~0.51-0.54 |
| ResNet   | PyTorch   | ~51-54%  | ~0.50-0.53 |

_Variations due to random initialization, but generally similar or better than Keras_

---

## 🚀 Deployment

### Option 1: Use Trained PyTorch Model with Streamlit

```bash
# Run the PyTorch-compatible Streamlit app
streamlit run frontend/front_pytorch.py

# Select "PyTorch (GPU)" from sidebar
# Use webcam or upload video for detection
```

### Option 2: Load Trained Model for Inference

```python
import torch
from src.pytorch_models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model('advanced', num_classes=7, device=device)
model.load_state_dict(torch.load('saved_models/pytorch_advanced_cnn_best.pt', map_location=device))
model.eval()

# Prepare image
import cv2
import numpy as np

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (48, 48))
img_normalized = img_resized.astype('float32') / 255.0
img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(img_tensor)
    emotion = np.argmax(output.cpu().numpy())
    confidence = torch.softmax(output, dim=1)[0, emotion].item()

print(f"Emotion: {emotion}, Confidence: {confidence:.2%}")
```

### Option 3: Convert PyTorch to ONNX (For Cross-Platform Deployment)

```python
import torch
from src.pytorch_models import get_model

device = torch.device("cpu")  # Use CPU for export
model = get_model('advanced', num_classes=7, device=device)
model.load_state_dict(torch.load('saved_models/pytorch_advanced_cnn_best.pt', map_location=device))

# Export to ONNX
dummy_input = torch.randn(1, 1, 48, 48)
torch.onnx.export(
    model,
    dummy_input,
    "saved_models/emotion_model.onnx",
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={'image': {0: 'batch_size'}}
)

print("✓ Model exported to ONNX format")
```

**ONNX can be run on:**

- Android/iOS (ONNX Mobile)
- Web (ONNX.js)
- Edge devices
- Any platform with ONNX runtime

---

## 🆚 Keras vs PyTorch: Key Differences

### Data Loading

**Keras:**

```python
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=32, epochs=100)
```

**PyTorch:**

```python
train_loader = DataLoader(train_dataset, batch_size=32)
for epoch in range(100):
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)  # ← Manual GPU transfer
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
```

### Training Loop

**Keras:**

```python
# Automatic training loop
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
history = model.fit(X_train, y_train, ...)
```

**PyTorch:**

```python
# Manual control (more flexibility)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

### GPU Usage

**Keras:**

```python
# Automatic with GPU
# (But sometimes fragile on Windows)
```

**PyTorch:**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X = X.to(device)  # ← Explicit, clear control
```

---

## 🐛 Common Issues & Solutions

### Issue 1: "CUDA available: False"

```python
# Check:
import torch
print(torch.cuda.is_available())  # Should be True

# Fix:
# See PYTORCH_GPU_SETUP.md for detailed troubleshooting
```

### Issue 2: "CUDA out of memory"

```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Or clear cache
torch.cuda.empty_cache()
```

### Issue 3: Model predictions different from Keras

✅ **This is expected!** Different random initialization can cause slight differences.

To get identical results:

```python
import torch
import numpy as np

# Set seeds BEFORE creating model
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

model = YourModel()
```

### Issue 4: "RuntimeError: expected scalar type Long but found Float"

```python
# Fix: Labels must be Long (int64), not Float
y_batch = y_batch.long()  # Convert
```

---

## 📚 Learning Resources

### PyTorch Basics

- Official Tutorials: https://pytorch.org/tutorials/
- 60-Min Blitz: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

### CNN in PyTorch

- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

### GPU Optimization

- https://pytorch.org/docs/stable/notes/cuda.html
- Mixed Precision Training: https://pytorch.org/docs/stable/amp.html

---

## ✅ Checklist: Migration Complete

- [ ] PyTorch installed with CUDA support
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] `notebooks/04_pytorch_baseline.ipynb` runs successfully
- [ ] Model saved to `saved_models/pytorch_baseline_cnn_best.pt`
- [ ] Test accuracy > 50%
- [ ] `notebooks/05_pytorch_advanced.ipynb` completes
- [ ] `frontend/front_pytorch.py` loads PyTorch models
- [ ] Streamlit app shows "PyTorch (GPU)" option
- [ ] Webcam detection works with PyTorch model

---

## 🎯 Next Steps

1. **Complete Training:**
   - Run all PyTorch notebooks
   - Compare all 3 models (Baseline, Advanced, ResNet)
   - Select best model

2. **Deploy:**
   - Use best model for Streamlit inference
   - Test with webcam and video uploads
   - Verify inference speed

3. **Optimize (Optional):**
   - Use mixed precision training (torch.cuda.amp)
   - Try quantization for mobile deployment
   - Export to ONNX for cross-platform use

4. **Experiment:**
   - Try different architectures
   - Fine-tune hyperparameters
   - Use data augmentation
   - Implement attention mechanisms

---

## 💬 FAQ

**Q: Should I delete my Keras models?**
A: No! Keep both. They co-exist. Use PyTorch for new experiments, Keras for comparison.

**Q: Can I mix Keras and PyTorch in one app?**
A: Yes, but not recommended. Pick one framework for production.

**Q: How do I convert Keras weights to PyTorch?**
A: Manual conversion needed. Recommended to retrain with PyTorch.

**Q: Will PyTorch models run on CPU?**
A: Yes, just change `device = torch.device("cpu")`. But it's slow.

**Q: How do I save for inference without GPU?**
A: Load on CPU: `model.load_state_dict(torch.load(..., map_location='cpu'))`

---

## 🚀 You're Ready!

Your project is now running on **GPU with PyTorch**.

**Next:** Run `notebooks/04_pytorch_baseline.ipynb` to start training! 🎉

---

**Questions?** Check `PYTORCH_GPU_SETUP.md` or the notebooks for examples.
