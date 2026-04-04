"""Data preprocessing module for facial emotion recognition."""

import numpy as np
import pandas as pd
from pathlib import Path
import os
import json
import cv2
from sklearn.model_selection import train_test_split


EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def load_fer2013(data_root='data', data_dir='data/preprocessed'):
    """Load FER2013 dataset from organized folder structure.
    
    Expected structure:
        data/
            train/
                angry/
                disgust/
                fear/
                happy/
                neutral/
                sad/
                surprise/
            test/
                angry/
                disgust/
                fear/
                happy/
                neutral/
                sad/
                surprise/
    
    Args:
        data_root: Root directory containing train/ and test/ folders
        data_dir: Directory to save preprocessed files
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Create output directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    
    # Load training data from folders
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')
    
    print("Loading training data...")
    for emotion_idx, emotion in enumerate(EMOTION_LABELS):
        emotion_train_path = os.path.join(train_dir, emotion)
        
        if os.path.exists(emotion_train_path):
            image_files = [f for f in os.listdir(emotion_train_path) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(emotion_train_path, img_file)
                
                # Load image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize to 48x48
                    img = cv2.resize(img, (48, 48))
                    # Normalize
                    img = img.astype('float32') / 255.0
                    X_train_list.append(img)
                    y_train_list.append(emotion_idx)
            
            print(f"  {emotion.capitalize()}: {len(image_files)} images")
    
    print("\nLoading test data...")
    for emotion_idx, emotion in enumerate(EMOTION_LABELS):
        emotion_test_path = os.path.join(test_dir, emotion)
        
        if os.path.exists(emotion_test_path):
            image_files = [f for f in os.listdir(emotion_test_path) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(emotion_test_path, img_file)
                
                # Load image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize to 48x48
                    img = cv2.resize(img, (48, 48))
                    # Normalize
                    img = img.astype('float32') / 255.0
                    X_test_list.append(img)
                    y_test_list.append(emotion_idx)
            
            print(f"  {emotion.capitalize()}: {len(image_files)} images")
    
    # Convert to numpy arrays
    X_train_full = np.array(X_train_list)
    y_train_full = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    # Split training data into train and validation (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    # Reshape for CNN (add channel dimension)
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_val = X_val.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)
    
    # Save preprocessed data
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    
    print(f"\n✓ Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"✓ X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"✓ X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_preprocessed(data_dir='data/preprocessed'):
    """Load preprocessed .npy files.
    
    Args:
        data_dir: Directory containing preprocessed files
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_class_weights(y_train):
    """Compute class weights to handle imbalance.
    
    Args:
        y_train: Training labels
    
    Returns:
        Dictionary of class_weight -> weight mapping
    """
    from collections import Counter
    
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    
    class_weights = {int(label): total / (len(unique) * count) 
                    for label, count in zip(unique, counts)}
    
    return class_weights


def augment_data(X_train, y_train):
    """Apply data augmentation using Keras ImageDataGenerator.
    
    Args:
        X_train: Training images
        y_train: Training labels
    
    Returns:
        DataGenerator for training
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    augmentation = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    return augmentation.flow(X_train, y_train, batch_size=32)
