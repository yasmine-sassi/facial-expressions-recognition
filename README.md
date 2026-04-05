# Facial Emotion Recognition (FER2013)

A deep learning project for recognizing facial emotions using Convolutional Neural Networks (CNN) on the FER2013 dataset.

## Project Structure

```
projet_traitement_img/
├── data/                         # Dataset and preprocessed data
│   ├── train/                    # Training images organized by emotion 3,171 img
│   ├── test/                     # Test images organized by emotion
│   └── preprocessed/             # Generated .npy files
│       ├── X_train.npy          # Training images (22,968, 48, 48, 1)
│       ├── y_train.npy          # Training labels (22,968,)
│       ├── X_val.npy            # Validation images (5,741, 48, 48, 1)
│       ├── y_val.npy            # Validation labels (5,741,)
│       ├── X_test.npy           # Test images (7,178, 48, 48, 1)
│       ├── y_test.npy           # Test labels (7,178,)
│       ├── class_weights.json    # Balanced class weights for training
│       └── class_mapping.json    # Emotion label mapping
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_preprocessing.ipynb    # Data loading, cleaning & preprocessing
│   ├── 02_baseline_cnn.ipynb     # Baseline CNN model
│   ├── 03_advanced_model.ipynb   # Advanced architectures
│   └── 04_evaluation.ipynb       # Results analysis and comparison
│
├── src/                          # Reusable modules
│   ├── preprocessing.py          # Data preprocessing functions
│   ├── model.py                  # CNN architectures
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Metrics and confusion matrix
│   └── predict.py                # Inference functions
│
├── frontend/
│   └── front.py                  # Web demo interface
│
├── results/                      # Generated visualizations
│
├── saved_models/                 # Trained models (.h5/.keras)
├── requirements.txt
└── README.md
```

---

## Dataset Overview

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

**Key Challenge**: Class imbalance ratio of **16.5×** (Happy vs Disgust) handled via balanced class weights.

---

## Preprocessing Pipeline

Complete 3-step preprocessing workflow implemented in `01_preprocessing.ipynb`:

### **Step 1: Nettoyage des données (Data Cleaning)**

✅ **Quality Assessment**:

- Corrupted file detection (cv2.imread validation): **0 files**
- Dimension validation (minimum 40×40 px): **0 files**
- Contrast filtering (std dev threshold = 10): **13 anomalies**
- **Overall quality: 99.96% valid** (35,874/35,887 images)

---

### **Step 2: Redimensionnement (Resizing)**

✅ **Standardization**:

- **Output size**: 48×48 pixels (uniform for all images)
- **Color mode**: Grayscale (single channel: reduces 3×computation)
- **Interpolation**: Linear (quality-preserving)
- **Result**: All images reshaped to (48, 48, 1) tensor

---

### **Step 3: Filtrage et amélioration (Filtering & Enhancement)**

#### **3a) Normalization**

```
rescale = 1.0 / 255
Pixel range: [0, 255] → [0, 1]
Verified mean: 0.50 | std: 0.25
```

#### **3b) Data Augmentation** (Applied during training only)

- Rotation: ±15°
- Horizontal flip: 50% probability
- Zoom: ±10%
- Width/Height shift: ±10%
- Purpose: Increase diversity, improve generalization

#### **3c) Class Balancing**

Balanced weights applied during `model.fit()` to handle class imbalance:

```python
class_weight_dict = {
    0: 1.0266,   # Angry
    1: 9.4016,   # Disgust (most underrepresented - 9.40× weight)
    2: 1.0010,   # Fear
    3: 0.5685,   # Happy (most overrepresented - 0.57× weight)
    4: 0.8261,   # Neutral
    5: 0.8492,   # Sad
    6: 1.2933    # Surprise
}
```

---

## Data Split & Loading

### **Train/Validation/Test Distribution**

```
28,709 training images
├── Training set (80%):   22,968 images → 359 batches
└── Validation set (20%):  5,741 images → 90 batches

Separate test set:        7,178 images → 113 batches
```

**All data exported to .npy format for fast loading (10-100× faster than imread)**

---

## Setup

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Organize dataset:
   - Place images in `data/train/{emotion}/` and `data/test/{emotion}/`
   - Emotion folders: angry, disgust, fear, happy, neutral, sad, surprise

---

## Workflow

1. **Preprocessing** ✅ COMPLETE
   - Run `01_preprocessing.ipynb`
   - Output: 6 .npy files + 2 JSON metadata + 10 PNG visualizations
   - Quality: 99.96% valid images
   - Time: ~60 seconds

2. **Baseline Model**
   - Run `02_baseline_cnn.ipynb`
   - Simple CNN with class weights & data augmentation

3. **Advanced Model**
   - Run `03_advanced_model.ipynb`
   - Batch norm, dropout, deeper architecture

4. **Evaluation**
   - Run `04_evaluation.ipynb`
   - Confusion matrix, precision/recall/F1, error analysis

5. **Demo**
   - Run `python -m streamlit run frontend/front.py`
   - Real-time emotion prediction

---

## Emotion Classes

| Index | Class    |
| ----- | -------- |
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Neutral  |
| 5     | Sad      |
| 6     | Surprise |

---

## Quality Metrics

| Metric           | Value           | Status       |
| ---------------- | --------------- | ------------ |
| Total images     | 35,887          | ✅           |
| Valid images     | 99.96%          | ✅ EXCELLENT |
| Corrupted files  | 0               | ✅ CLEAN     |
| Normalization    | [0, 1] verified | ✅           |
| Class balance    | 9.40× handled   | ✅           |
| Image uniformity | 48×48 grayscale | ✅           |

---

## Requirements

- Python 3.8+
- TensorFlow/Keras 2.10+
- NumPy, Pandas
- OpenCV (cv2)
- scikit-learn
- Matplotlib, Seaborn

Install with:

```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Application

### Basic Usage

```bash
# From the frontend directory
streamlit run frontend/front.py
```

The application will open in your default browser at `http://localhost:8501`

## License

FER2013 Dataset: https://www.kaggle.com/datasets/deadskull7/fer2013
