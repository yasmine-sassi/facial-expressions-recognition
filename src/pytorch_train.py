"""PyTorch training utilities for emotion recognition models."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt


class EarlyStoppingCallback:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=15, verbose=True, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping at epoch {epoch} (best: {self.best_epoch})")
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


def get_class_weights(y_train, num_classes=7):
    """Compute class weights to handle imbalance."""
    class_counts = np.bincount(y_train, minlevel=num_classes)
    total_samples = len(y_train)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.tensor(class_weights, dtype=torch.float32)


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32, device='cuda'):
    """Create PyTorch DataLoaders from numpy arrays."""
    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for X_batch, y_batch in progress_bar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == y_batch).sum().item()
        total_samples += y_batch.size(0)
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    
    return avg_loss, avg_accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)
    
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    
    return avg_loss, avg_accuracy


def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001,
                device='cuda', model_name='emotion_model', class_weights=None):
    """
    Train PyTorch model with early stopping and learning rate scheduling.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        device: 'cuda' or 'cpu'
        model_name: Name for saving model checkpoints
        class_weights: Class weights for imbalanced data
    
    Returns:
        Dictionary with training history
    """
    os.makedirs('saved_models', exist_ok=True)
    
    # Loss function
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                  patience=5, min_lr=1e-7, verbose=True)
    
    # Early stopping
    early_stopper = EarlyStoppingCallback(patience=15)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        # Save best model
        if epoch == 0 or val_loss < min(history['val_loss'][:-1]):
            best_loss = val_loss
            torch.save(model.state_dict(), f'saved_models/{model_name}_best.pt')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopper(val_loss, epoch)
        if early_stopper.early_stop:
            print(f"\nTraining stopped at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'saved_models/{model_name}_best.pt'))
    torch.save(model.state_dict(), f'saved_models/{model_name}_final.pt')
    
    print(f"\nBest model saved to saved_models/{model_name}_best.pt")
    print(f"Final model saved to saved_models/{model_name}_final.pt")
    
    return history


def plot_training_history(history, model_name='emotion_model', save_path='results/pytorch_training_history.png'):
    """Plot training history."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def compare_models_history(histories, model_names, save_path='results/pytorch_model_comparison.png'):
    """Compare training histories of multiple models."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (history, name) in enumerate(zip(histories, model_names)):
        # Accuracy
        axes[0, 0].plot(history['val_accuracy'], label=f'{name} (Val)', 
                       linewidth=2, color=colors[idx], linestyle='-')
        axes[0, 0].plot(history['train_accuracy'], label=f'{name} (Train)', 
                       linewidth=1, color=colors[idx], linestyle='--', alpha=0.7)
        
        # Loss
        axes[0, 1].plot(history['val_loss'], label=f'{name} (Val)', 
                       linewidth=2, color=colors[idx], linestyle='-')
        axes[0, 1].plot(history['train_loss'], label=f'{name} (Train)', 
                       linewidth=1, color=colors[idx], linestyle='--', alpha=0.7)
    
    # Val Accuracy Comparison
    for idx, (history, name) in enumerate(zip(histories, model_names)):
        axes[1, 0].plot(history['val_accuracy'], label=name, linewidth=2, color=colors[idx])
    
    # Val Loss Comparison
    for idx, (history, name) in enumerate(zip(histories, model_names)):
        axes[1, 1].plot(history['val_loss'], label=name, linewidth=2, color=colors[idx])
    
    # Formatting
    axes[0, 0].set_title('Accuracy Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Loss Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Validation Accuracy', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Loss', fontweight='bold')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {save_path}")
    plt.close()
