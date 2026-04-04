"""Training module for emotion recognition models."""

import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
                class_weights=None, model_name='emotion_model'):
    """Train a CNN model.
    
    Args:
        model: Compiled Keras model
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        class_weights: Dictionary of class weights
        model_name: Name for saving the model
    
    Returns:
        History object from training
    """
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                                   min_lr=1e-7, verbose=1)
    
    # Create saved_models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    checkpoint = ModelCheckpoint(f'saved_models/{model_name}_best.keras',
                                 monitor='val_accuracy', save_best_only=True, 
                                 verbose=1)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Save final model
    model.save(f'saved_models/{model_name}_final.keras')
    
    return history


def plot_training_history(history, save_path='results/training_history.png'):
    """Plot training and validation metrics.
    
    Args:
        history: History object from model training
        save_path: Path to save the plot
    """
    os.makedirs('results', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()
