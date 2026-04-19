"""
Training Pipeline for Plant Leaf Disease Detection.
Orchestrates the complete training process with two-phase training.
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from src.config import (
    EPOCHS_PHASE1, EPOCHS_PHASE2, MODEL_SAVE_PATH, MODEL_SAVE_PATH_KERAS,
    MODELS_DIR, RESULTS_DIR,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR
)
from src.data_preprocessing import get_data
from src.model import build_model, unfreeze_model


def get_callbacks(phase="phase1"):
    """
    Create training callbacks.
    
    Args:
        phase: 'phase1' or 'phase2'
    
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
    Main training function. Executes the complete two-phase training pipeline.
    
    Phase 1: Feature Extraction
        - Frozen MobileNetV2 base
        - Train only the custom classification head
        - Higher learning rate (1e-3)
    
    Phase 2: Fine-Tuning
        - Unfreeze top 30 layers of MobileNetV2
        - Train with very low learning rate (1e-5)
        - Fine-tune pre-trained features for our specific task
    """
    print("\n" + "=" * 60)
    print("🌿 PLANT LEAF DISEASE DETECTION — TRAINING PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # ========================================
    # Step 1: Load and preprocess data
    # ========================================
    train_ds, val_ds, test_ds, class_names, class_weights = get_data()
    
    # ========================================
    # Step 2: Build model
    # ========================================
    model, base_model = build_model(num_classes=len(class_names))
    
    # ========================================
    # Step 3: Phase 1 — Feature Extraction
    # ========================================
    print("\n" + "=" * 60)
    print(f"PHASE 1: FEATURE EXTRACTION ({EPOCHS_PHASE1} epochs)")
    print("=" * 60)
    print("Training only the classification head (base frozen)...\n")
    
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        class_weight=class_weights,
        callbacks=get_callbacks("phase1"),
        verbose=1
    )
    
    # Print Phase 1 results
    p1_train_acc = history_phase1.history['accuracy'][-1]
    p1_val_acc = history_phase1.history['val_accuracy'][-1]
    print(f"\n✓ Phase 1 Complete")
    print(f"  Training Accuracy:   {p1_train_acc:.4f} ({p1_train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {p1_val_acc:.4f} ({p1_val_acc*100:.2f}%)")
    
    # ========================================
    # Step 4: Phase 2 — Fine-Tuning
    # ========================================
    model = unfreeze_model(model, base_model)
    
    print(f"\nPhase 2: Fine-tuning ({EPOCHS_PHASE2} epochs)")
    print("Training top layers of MobileNetV2 + classification head...\n")
    
    # Total epochs = Phase1 + Phase2
    total_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2
    
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=len(history_phase1.history['accuracy']),
        class_weight=class_weights,
        callbacks=get_callbacks("phase2"),
        verbose=1
    )
    
    # Print Phase 2 results
    p2_train_acc = history_phase2.history['accuracy'][-1]
    p2_val_acc = history_phase2.history['val_accuracy'][-1]
    print(f"\n✓ Phase 2 Complete")
    print(f"  Training Accuracy:   {p2_train_acc:.4f} ({p2_train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {p2_val_acc:.4f} ({p2_val_acc*100:.2f}%)")
    
    # ========================================
    # Step 5: Evaluate on Test Set
    # ========================================
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # ========================================
    # Step 6: Save model and training history
    # ========================================
    print("\n" + "=" * 60)
    print("SAVING MODEL & HISTORY")
    print("=" * 60)
    
    # Save the model
    model.save(MODEL_SAVE_PATH_KERAS)
    print(f"✓ Model saved to: {MODEL_SAVE_PATH_KERAS}")
    
    # Also save in H5 format
    try:
        model.save(MODEL_SAVE_PATH)
        print(f"✓ Model saved to: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"  Note: Could not save .h5 format: {e}")
    
    # Merge training histories
    full_history = {}
    for key in history_phase1.history:
        full_history[key] = history_phase1.history[key] + history_phase2.history[key]
    
    # Save history as JSON
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    # Convert numpy values to Python floats for JSON serialization
    serializable_history = {k: [float(v) for v in vals] for k, vals in full_history.items()}
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")
    
    # Save class names
    class_names_path = os.path.join(MODELS_DIR, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"✓ Class names saved to: {class_names_path}")
    
    # ========================================
    # Summary
    # ========================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
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
