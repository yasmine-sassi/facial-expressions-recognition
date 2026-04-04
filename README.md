# Facial Emotion Recognition

A deep learning project for recognizing facial emotions using Convolutional Neural Networks (CNN).

## Project Structure

```
facial-emotion-recognition/
├── data/                          # Dataset and preprocessed data
│   ├── train/                    # Training images organized by emotion
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   ├── test/                     # Test images organized by emotion
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── preprocessed/             # Generated .npy files
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── FER2013_Preprocessing.ipynb    # Data loading & preprocessing
│   ├── 02_baseline_cnn.ipynb     # Baseline CNN model
│   ├── 03_advanced_model.ipynb   # Advanced architecture
│   └── 04_evaluation.ipynb       # Results analysis
│
├── src/                          # Reusable modules
│   ├── preprocessing.py          # Data preprocessing functions
│   ├── model.py                  # CNN architectures
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Metrics and confusion matrix
│   └── predict.py                # Inference
│
├── frontend/
│   └── front.py
│
├── saved_models/                 # Trained models (.h5/.keras)
├── results/                      # Graphs and generated plots
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Organize your dataset in the folder structure:
   - Place images in `data/train/{emotion}/` and `data/test/{emotion}/`
   - Emotion folders: angry, disgust, fear, happy, neutral, sad, surprise

## Workflow

1. **Preprocessing**: Run `FER2013_Preprocessing.ipynb` to load and preprocess the dataset
   - Generates normalized `.npy` files in `data/preprocessed/`
   - Computes class weights
   - Saves metadata (class mapping, weights)

2. **Baseline Model**: Run `02_baseline_cnn.ipynb` to train a baseline CNN
3. **Advanced Model**: Run `03_advanced_model.ipynb` for improved architectures
4. **Evaluation**: Run `04_evaluation.ipynb` to analyze and compare results
5. **Demo**: Run `python demo/webcam_demo.py` for real-time emotion prediction

## Emotion Classes

0. Angry
1. Disgust
2. Fear
3. Happy
4. Neutral
5. Sad
6. Surprise

## Requirements

- Python 3.8+
- TensorFlow/Keras
- NumPy, Pandas
- OpenCV
- Matplotlib, Seaborn

## Author

Yasmine Sassi
