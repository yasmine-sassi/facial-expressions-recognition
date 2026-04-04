"""Evaluation module for emotion recognition models."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def evaluate_model(model, X_test, y_test, model_name='emotion_model'):
    """Evaluate model on test set.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels
        model_name: Name for saving results
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': test_accuracy,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(y_test, y_pred, model_name='emotion_model'):
    """Plot and save confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name for saving the plot
    """
    os.makedirs('results', exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, 
                yticklabels=EMOTION_LABELS,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to results/{model_name}_confusion_matrix.png")
    plt.close()


def plot_per_class_metrics(y_test, y_pred, model_name='emotion_model'):
    """Plot per-class precision, recall, and F1-score.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name for saving the plot
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    os.makedirs('results', exist_ok=True)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    
    x = np.arange(len(EMOTION_LABELS))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Emotion Class', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Per-Class Metrics', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_per_class_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Per-class metrics saved to results/{model_name}_per_class_metrics.png")
    plt.close()


def plot_prediction_distribution(y_pred_proba, y_test, model_name='emotion_model'):
    """Plot distribution of prediction confidence.
    
    Args:
        y_pred_proba: Model probability predictions
        y_test: True labels
        model_name: Name for saving the plot
    """
    os.makedirs('results', exist_ok=True)
    
    max_proba = np.max(y_pred_proba, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(max_proba, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Maximum Prediction Probability', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Distribution of Prediction Confidence', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Confidence distribution saved to results/{model_name}_confidence_distribution.png")
    plt.close()
