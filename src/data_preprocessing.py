"""
Data Preprocessing Module for Plant Leaf Disease Detection.
Handles data loading, augmentation, splitting, and class weight computation.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from src.config import (
    DATASET_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_SEED,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    ZOOM_RANGE, HORIZONTAL_FLIP, BRIGHTNESS_RANGE, FILL_MODE
)


def load_dataset():
    """
    Load the PlantVillage dataset and split into train, validation, and test sets.
    
    Returns:
        train_ds: Training dataset (80%)
        val_ds: Validation dataset (10%)
        test_ds: Test dataset (10%)
        class_names: List of class names
    """
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    print(f"Dataset path: {DATASET_DIR}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Load training + validation set (90% of data)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,    # 20% goes to val+test
        subset="training",       # 80% for training
        seed=RANDOM_SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=True
    )
    
    # Load the remaining 20% and split into val (10%) and test (10%)
    # IMPORTANT: shuffle=False ensures consistent take/skip split with evaluate.py
    val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",     # 20% for val+test
        seed=RANDOM_SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False
    )
    
    # Get class names
    class_names = train_ds.class_names
    print(f"\nTotal classes found: {len(class_names)}")
    
    # Split val_test into validation and test (50/50 = 10% each of total)
    val_test_size = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_size = val_test_size // 2
    
    val_ds = val_test_ds.take(val_size)
    test_ds = val_test_ds.skip(val_size)
    
    # Print split info
    train_size = tf.data.experimental.cardinality(train_ds).numpy()
    val_size_actual = tf.data.experimental.cardinality(val_ds).numpy()
    test_size_actual = tf.data.experimental.cardinality(test_ds).numpy()
    
    print(f"\nDataset split (in batches of {BATCH_SIZE}):")
    print(f"  Training batches:   {train_size}")
    print(f"  Validation batches: {val_size_actual}")
    print(f"  Test batches:       {test_size_actual}")
    
    return train_ds, val_ds, test_ds, class_names


def create_augmentation_layer():
    """
    Create a data augmentation layer using Keras preprocessing layers.
    Applied only during training.
    
    Returns:
        data_augmentation: Sequential augmentation layer
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(ROTATION_RANGE / 360),  # Convert degrees to fraction
        tf.keras.layers.RandomZoom(ZOOM_RANGE),
        tf.keras.layers.RandomTranslation(
            height_factor=HEIGHT_SHIFT_RANGE,
            width_factor=WIDTH_SHIFT_RANGE
        ),
        tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(factor=0.2),
    ], name='data_augmentation')
    
    print("\n✓ Data augmentation layer created")
    print("  Augmentations: RandomFlip, RandomRotation, RandomZoom,")
    print("  RandomTranslation, RandomBrightness, RandomContrast")
    
    return data_augmentation


def compute_class_weights(train_ds):
    """
    Compute class weights to handle class imbalance.
    Classes with fewer samples get higher weights.
    
    Args:
        train_ds: Training dataset
        
    Returns:
        class_weight_dict: Dictionary mapping class index to weight
    """
    print("\n" + "=" * 60)
    print("COMPUTING CLASS WEIGHTS")
    print("=" * 60)
    
    # Extract all labels from training dataset
    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    
    # Compute balanced class weights
    unique_classes = np.unique(all_labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=all_labels
    )
    
    class_weight_dict = dict(zip(unique_classes.astype(int), weights))
    
    # Print some stats
    min_weight_class = min(class_weight_dict, key=class_weight_dict.get)
    max_weight_class = max(class_weight_dict, key=class_weight_dict.get)
    print(f"  Min weight: {class_weight_dict[min_weight_class]:.3f} (class {min_weight_class})")
    print(f"  Max weight: {class_weight_dict[max_weight_class]:.3f} (class {max_weight_class})")
    print(f"  Weight range ratio: {class_weight_dict[max_weight_class]/class_weight_dict[min_weight_class]:.1f}x")
    print("✓ Class weights computed successfully")
    
    return class_weight_dict


def optimize_dataset(train_ds, val_ds, test_ds):
    """
    Optimize datasets for performance using prefetching.
    
    NOTE: No pixel normalization here! Images stay in [0, 255] range.
    The model handles preprocessing internally via MobileNetV2's preprocess_input.
    
    Args:
        train_ds, val_ds, test_ds: Raw datasets
        
    Returns:
        Optimized (train_ds, val_ds, test_ds)
    """
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Cast to float32 (images from image_dataset_from_directory are already float32 in [0,255])
    # No Rescaling — model's preprocess_input handles normalization
    # Note: train_ds is already shuffled by image_dataset_from_directory(shuffle=True)
    
    # Prefetch for performance (removed .cache() to prevent Out Of Memory errors)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    print("\n✓ Datasets optimized (shuffled, prefetched) — images in [0, 255] range")
    
    return train_ds, val_ds, test_ds


def get_data():
    """
    Main function to get all preprocessed data ready for training.
    
    Returns:
        train_ds: Optimized training dataset
        val_ds: Optimized validation dataset
        test_ds: Optimized test dataset
        class_names: List of class names
        class_weights: Dictionary of class weights
    """
    # Load and split
    train_ds, val_ds, test_ds, class_names = load_dataset()
    
    # Compute class weights before optimization
    class_weights = compute_class_weights(train_ds)
    
    # Optimize datasets
    train_ds, val_ds, test_ds = optimize_dataset(train_ds, val_ds, test_ds)
    
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING COMPLETE ✓")
    print("=" * 60)
    
    return train_ds, val_ds, test_ds, class_names, class_weights


if __name__ == "__main__":
    # Test the data pipeline
    train_ds, val_ds, test_ds, class_names, class_weights = get_data()
    
    print(f"\nClass names ({len(class_names)}):")
    for i, name in enumerate(class_names):
        print(f"  {i:2d}: {name}")
    
    # Check a batch
    for images, labels in train_ds.take(1):
        print(f"\nSample batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Pixel value range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
