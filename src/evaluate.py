"""
Evaluation Module for Plant Leaf Disease Detection.
Generates accuracy/loss plots, confusion matrix, and classification report.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.config import (
    MODEL_SAVE_PATH_KERAS, RESULTS_DIR, DATASET_DIR,
    IMG_SIZE, BATCH_SIZE, RANDOM_SEED
)


def load_training_history():
    """Load training history from saved JSON file."""
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, 'r') as f:
        history = json.load(f)
    print(f"✓ Training history loaded from: {history_path}")
    return history


def plot_training_curves(history):
    """
    Plot training & validation accuracy and loss curves.
    
    Args:
        history: Dictionary with training history
    """
    print("\nGenerating training curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # --- Accuracy Plot ---
    axes[0].plot(epochs, history['accuracy'], 'b-o', label='Training Accuracy', 
                 linewidth=2, markersize=4)
    axes[0].plot(epochs, history['val_accuracy'], 'r-o', label='Validation Accuracy', 
                 linewidth=2, markersize=4)
    axes[0].set_title('Model Accuracy', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Add Phase 1/Phase 2 divider
    if len(epochs) > 5:
        axes[0].axvline(x=5, color='green', linestyle='--', alpha=0.7, label='Phase 1→2')
        axes[0].text(5.2, 0.1, 'Fine-tuning\nstarts', fontsize=9, color='green')
    
    # --- Loss Plot ---
    axes[1].plot(epochs, history['loss'], 'b-o', label='Training Loss', 
                 linewidth=2, markersize=4)
    axes[1].plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', 
                 linewidth=2, markersize=4)
    axes[1].set_title('Model Loss', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    if len(epochs) > 5:
        axes[1].axvline(x=5, color='green', linestyle='--', alpha=0.7)
    
    plt.suptitle('Plant Disease Detection — Training History', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot a confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # Normalize the confusion matrix
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_sum = np.where(cm_sum == 0, 1, cm_sum)
    cm_normalized = cm.astype('float') / cm_sum
    
    # Create a larger figure for 38 classes
    fig, ax = plt.subplots(figsize=(28, 24))
    
    # Shorten class names for readability
    short_names = [name.replace('___', '\n').replace('_(including_sour)', '')
                   .replace('_(maize)', '').replace(',_bell', '')
                   for name in class_names]
    
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='YlOrRd',
        xticklabels=short_names, 
        yticklabels=short_names,
        ax=ax,
        linewidths=0.5,
        vmin=0, 
        vmax=1,
        annot_kws={"size": 6}
    )
    
    ax.set_title('Confusion Matrix (Normalized)', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {save_path}")
    
    return cm


def generate_classification_report(y_true, y_pred, class_names):
    """
    Generate and save a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\nGenerating classification report...")
    
    report = classification_report(
        y_true, y_pred, 
        labels=range(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    # Print to console
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(report)
    
    # Save to file
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("PLANT LEAF DISEASE DETECTION — CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    print(f"✓ Classification report saved to: {report_path}")
    
    # Also save as dict for programmatic access
    report_dict = classification_report(
        y_true, y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_json_path = os.path.join(RESULTS_DIR, "classification_report.json")
    with open(report_json_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    return report_dict


def plot_per_class_accuracy(y_true, y_pred, class_names):
    """
    Plot per-class accuracy as a horizontal bar chart.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\nGenerating per-class accuracy chart...")
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_sum = cm.sum(axis=1)
    cm_sum = np.where(cm_sum == 0, 1, cm_sum)
    per_class_acc = cm.diagonal() / cm_sum
    
    # Sort by accuracy
    sorted_indices = np.argsort(per_class_acc)
    sorted_acc = per_class_acc[sorted_indices]
    sorted_names = [class_names[i].replace('___', ' → ') for i in sorted_indices]
    
    # Color based on accuracy
    colors = ['#e74c3c' if acc < 0.8 else '#f39c12' if acc < 0.9 else '#27ae60' 
              for acc in sorted_acc]
    
    fig, ax = plt.subplots(figsize=(14, 16))
    bars = ax.barh(range(len(sorted_names)), sorted_acc, color=colors, edgecolor='white', height=0.7)
    
    # Add value labels
    for bar, acc in zip(bars, sorted_acc):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.1%}', va='center', fontsize=8, fontweight='bold')
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    ax.set_xlim([0, 1.15])
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, "per_class_accuracy.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Per-class accuracy chart saved to: {save_path}")


def evaluate():
    """
    Main evaluation function. Loads the trained model and generates all reports.
    """
    print("\n" + "=" * 60)
    print("🌿 PLANT LEAF DISEASE DETECTION — EVALUATION")
    print("=" * 60)
    
    # Load the trained model
    print("\nLoading trained model...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH_KERAS)
    print(f"✓ Model loaded from: {MODEL_SAVE_PATH_KERAS}")
    
    # Load class names
    class_names_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH_KERAS), "class_names.json")
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    print(f"✓ {len(class_names)} class names loaded")
    
    # Load test dataset
    print("\nLoading test dataset...")
    val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False  # Important: don't shuffle for evaluation
    )
    
    val_test_size = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_size = val_test_size // 2
    test_ds = val_test_ds.skip(val_size)
    
    # No normalization needed — model handles preprocessing internally via preprocess_input
    
    # Get predictions
    print("\nGenerating predictions on test set...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall accuracy
    overall_accuracy = np.mean(y_true == y_pred)
    print(f"\n{'='*60}")
    print(f"  OVERALL TEST ACCURACY: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    # Generate all plots and reports
    # 1. Training curves
    try:
        history = load_training_history()
        plot_training_curves(history)
    except FileNotFoundError:
        print("⚠ Training history not found, skipping training curves")
    
    # 2. Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # 3. Classification report
    report = generate_classification_report(y_true, y_pred, class_names)
    
    # 4. Per-class accuracy
    plot_per_class_accuracy(y_true, y_pred, class_names)
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"All results saved in: {RESULTS_DIR}")
    print(f"\nFiles generated:")
    print(f"  📊 training_curves.png")
    print(f"  📊 confusion_matrix.png")
    print(f"  📊 per_class_accuracy.png")
    print(f"  📄 classification_report.txt")
    print(f"  📄 classification_report.json")
    
    return overall_accuracy, report


if __name__ == "__main__":
    evaluate()
