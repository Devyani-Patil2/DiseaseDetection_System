"""
Training Pipeline for Plant Leaf Disease Detection.
Single-phase training for custom CNN built from scratch.
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from src.config import (
    EPOCHS_PHASE1, MODEL_SAVE_PATH, MODEL_SAVE_PATH_KERAS,
    MODELS_DIR, RESULTS_DIR,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR
)
from src.data_preprocessing import get_data
from src.model import build_model


def get_callbacks():
    """
    Create training callbacks.
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Save best model based on validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH_KERAS,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOP_PATIENCE,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateauing
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        ),
    ]
    
    return callbacks


def train():
    """
    Main training function. Trains the custom CNN from scratch.
    
    Single-phase training:
        - All layers are trainable from the start
        - Learning rate starts at 1e-3 and is reduced dynamically
        - EarlyStopping prevents overfitting
    """
    print("\n" + "=" * 60)
    print("PLANT LEAF DISEASE DETECTION — TRAINING PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # ========================================
    # Step 1: Load and preprocess data
    # ========================================
    train_ds, val_ds, test_ds, class_names, class_weights = get_data()
    
    # ========================================
    # Step 2: Build custom CNN model
    # ========================================
    model = build_model(num_classes=len(class_names))
    
    # ========================================
    # Step 3: Train the model
    # ========================================
    total_epochs = EPOCHS_PHASE1
    print("\n" + "=" * 60)
    print(f"TRAINING CUSTOM CNN ({total_epochs} epochs)")
    print("=" * 60)
    print("All layers are trainable (no frozen layers)...\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )
    
    # Print training results
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"\n  Training Complete")
    print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # ========================================
    # Step 4: Evaluate on Test Set
    # ========================================
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # ========================================
    # Step 5: Save model and training history
    # ========================================
    print("\n" + "=" * 60)
    print("SAVING MODEL & HISTORY")
    print("=" * 60)
    
    # Save the model
    model.save(MODEL_SAVE_PATH_KERAS)
    print(f"  Model saved to: {MODEL_SAVE_PATH_KERAS}")
    
    # Also save in H5 format
    try:
        model.save(MODEL_SAVE_PATH)
        print(f"  Model saved to: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"  Note: Could not save .h5 format: {e}")
    
    # Save history as JSON
    full_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(full_history, f, indent=2)
    print(f"  Training history saved to: {history_path}")
    
    # Save class names
    class_names_path = os.path.join(MODELS_DIR, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"  Class names saved to: {class_names_path}")
    
    # ========================================
    # Summary
    # ========================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"  Model saved at: {MODEL_SAVE_PATH_KERAS}")
    print(f"\nNext steps:")
    print(f"  1. Run evaluate.py to generate plots & reports")
    print(f"  2. Run predict.py to test on individual images")
    print(f"  3. Run app.py to launch the Streamlit web app")
    
    return model, full_history, test_accuracy


if __name__ == "__main__":
    model, history, test_acc = train()
