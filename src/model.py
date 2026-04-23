"""
Model Architecture Module for Plant Leaf Disease Detection.
Custom CNN built from scratch — no pre-trained models used.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from src.config import (
    INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE, DENSE_UNITS,
    LEARNING_RATE_PHASE1
)
from src.data_preprocessing import create_augmentation_layer


def build_model(num_classes=NUM_CLASSES):
    """
    Build a custom CNN model from scratch for plant disease classification.
    
    Architecture:
        Input (224x224x3)
            -> Rescaling (0-255 to 0-1)
            -> Data Augmentation (training only)
            -> Conv Block 1: Conv2D(32) + BatchNorm + ReLU + MaxPool
            -> Conv Block 2: Conv2D(64) + BatchNorm + ReLU + MaxPool
            -> Conv Block 3: Conv2D(128) + BatchNorm + ReLU + Conv2D(128) + BatchNorm + ReLU + MaxPool
            -> Conv Block 4: Conv2D(256) + BatchNorm + ReLU + Conv2D(256) + BatchNorm + ReLU + MaxPool
            -> Conv Block 5: Conv2D(512) + BatchNorm + ReLU + Conv2D(512) + BatchNorm + ReLU + MaxPool
            -> Global Average Pooling
            -> Dense(512, ReLU) + Dropout(0.5)
            -> Dense(256, ReLU) + Dropout(0.3)
            -> Dense(38, Softmax)
    
    Args:
        num_classes: Number of output classes (default: 38)
    
    Returns:
        model: Compiled Keras model
    """
    print("\n" + "=" * 60)
    print("BUILDING CUSTOM CNN ARCHITECTURE")
    print("=" * 60)
    
    # Data augmentation layer
    data_augmentation = create_augmentation_layer()
    
    # Build the model from scratch
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    
    # Normalize pixel values from [0, 255] to [0, 1]
    x = layers.Rescaling(1./255, name='rescaling')(inputs)
    
    # Apply data augmentation (only during training)
    x = data_augmentation(x)
    
    # ===== Block 1: 32 filters =====
    x = layers.Conv2D(32, (3, 3), padding='same', name='block1_conv1')(x)
    x = layers.BatchNormalization(name='block1_bn1')(x)
    x = layers.Activation('relu', name='block1_relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='block1_pool')(x)
    
    # ===== Block 2: 64 filters =====
    x = layers.Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)
    x = layers.BatchNormalization(name='block2_bn1')(x)
    x = layers.Activation('relu', name='block2_relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='block2_pool')(x)
    
    # ===== Block 3: 128 filters (double conv for deeper features) =====
    x = layers.Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)
    x = layers.BatchNormalization(name='block3_bn1')(x)
    x = layers.Activation('relu', name='block3_relu1')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
    x = layers.BatchNormalization(name='block3_bn2')(x)
    x = layers.Activation('relu', name='block3_relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='block3_pool')(x)
    
    # ===== Block 4: 256 filters =====
    x = layers.Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)
    x = layers.BatchNormalization(name='block4_bn1')(x)
    x = layers.Activation('relu', name='block4_relu1')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', name='block4_conv2')(x)
    x = layers.BatchNormalization(name='block4_bn2')(x)
    x = layers.Activation('relu', name='block4_relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='block4_pool')(x)
    
    # ===== Block 5: 512 filters =====
    x = layers.Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = layers.BatchNormalization(name='block5_bn1')(x)
    x = layers.Activation('relu', name='block5_relu1')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = layers.BatchNormalization(name='block5_bn2')(x)
    x = layers.Activation('relu', name='block5_relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='block5_pool')(x)
    
    # ===== Classification Head =====
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(512, activation='relu', name='dense_512')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout_1')(x)
    x = layers.Dense(DENSE_UNITS, activation='relu', name='dense_256')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create the model
    model = Model(inputs, outputs, name='PlantDiseaseDetector_CustomCNN')
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print(f"\n  Model built successfully")
    print(f"  Input shape: {INPUT_SHAPE}")
    print(f"  Output classes: {num_classes}")
    print(f"  Architecture: Custom 5-Block CNN (built from scratch)")
    
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    print(f"  Learning rate: {LEARNING_RATE_PHASE1}")
    
    return model


def get_model_summary(model):
    """Print a formatted model summary."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    model.summary()


if __name__ == "__main__":
    # Test model creation
    model = build_model()
    get_model_summary(model)
