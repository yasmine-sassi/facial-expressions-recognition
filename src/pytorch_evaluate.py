"""PyTorch evaluation utilities for emotion recognition models."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def evaluate_model(model, X_test, y_test, device='cuda', model_name='emotion_model'):
    """
    Evaluate PyTorch model on test set.
    
    Args:
        model: Trained PyTorch model
        X_test: Test images (numpy array)
        y_test: Test labels (numpy array)
        device: 'cuda' or 'cpu'
        model_name: Name for results
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Convert to tensor
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    
    # Predictions
    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
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


def plot_confusion_matrix(y_test, y_pred, model_name='emotion_model', save_path='results/pytorch_confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name for the plot
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, 
                yticklabels=EMOTION_LABELS,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_per_class_metrics(y_test, y_pred, model_name='emotion_model', 
                          save_path='results/pytorch_per_class_metrics.png'):
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name for the plot
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    
    x = np.arange(len(EMOTION_LABELS))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Emotion Class', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{model_name} - Per-Class Metrics', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics plot saved to {save_path}")
    plt.close()


def plot_prediction_distribution(y_pred_proba, model_name='emotion_model',
                                save_path='results/pytorch_pred_distribution.png'):
    """
    Plot distribution of prediction confidence.
    
    Args:
        y_pred_proba: Prediction probabilities (N, num_classes)
        model_name: Name for the plot
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    max_proba = np.max(y_pred_proba, axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.hist(max_proba, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Maximum Prediction Confidence', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title(f'{model_name} - Prediction Confidence Distribution', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction distribution plot saved to {save_path}")
    plt.close()


def compare_model_results(results_list, model_names, save_path='results/pytorch_model_comparison_results.png'):
    """
    Compare results of multiple models side-by-side.
    
    Args:
        results_list: List of results dictionaries from evaluate_model
        model_names: List of model names
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    accuracies = [r['accuracy'] for r in results_list]
    
    fig, axes = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 4))
    
    if len(model_names) == 1:
        axes = [axes]
    
    for idx, (results, name, ax) in enumerate(zip(results_list, model_names, axes)):
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=EMOTION_LABELS,
                   yticklabels=EMOTION_LABELS,
                   cbar_kws={'label': 'Count'})
        ax.set_title(f'{name}\nAccuracy: {results["accuracy"]:.4f}', fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison results saved to {save_path}")
    plt.close()


def create_evaluation_report(model, X_test, y_test, device='cuda', model_name='emotion_model'):
    """
    Create comprehensive evaluation report.
    
    Args:
        model: Trained PyTorch model
        X_test: Test images
        y_test: Test labels
        device: 'cuda' or 'cpu'
        model_name: Name for saving outputs
    
    Returns:
        Dictionary with complete evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Main evaluation
    results = evaluate_model(model, X_test, y_test, device=device, model_name=model_name)
    
    # Visualizations
    plot_confusion_matrix(y_test, results['y_pred'], model_name=model_name,
                         save_path=f'results/pytorch_{model_name}_confusion_matrix.png')
    
    plot_per_class_metrics(y_test, results['y_pred'], model_name=model_name,
                          save_path=f'results/pytorch_{model_name}_per_class_metrics.png')
    
    plot_prediction_distribution(results['y_pred_proba'], model_name=model_name,
                               save_path=f'results/pytorch_{model_name}_confidence.png')
    
    print(f"\nEvaluation complete for {model_name}")
    
    return results
