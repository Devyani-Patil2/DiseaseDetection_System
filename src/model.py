import tensorflow as tf
from tensorflow.keras import layers, Model
from src.config import (
    INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE, DENSE_UNITS,
    LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2, FINE_TUNE_AT
)
from src.data_preprocessing import create_augmentation_layer


def build_model(num_classes=NUM_CLASSES):
    """
    Build a MobileNetV2-based model for plant disease classification.
    
    Architecture:
        Input (224x224x3)
            → Data Augmentation (training only)
            → MobileNetV2 (pre-trained, frozen)
            → Global Average Pooling
            → Dense(256, ReLU)
            → Dropout(0.5)
            → Dense(38, Softmax)
    
    Args:
        num_classes: Number of output classes (default: 38)
    
    Returns:
        model: Compiled Keras model (Phase 1 - feature extraction)
    """
    print("\n" + "=" * 60)
    print("BUILDING MODEL ARCHITECTURE")
    print("=" * 60)
    
    # Data augmentation layer
    data_augmentation = create_augmentation_layer()
    
    # Load MobileNetV2 pre-trained on ImageNet (without top classification layer)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,          # Remove the classification head
        weights='imagenet'          # Pre-trained weights
    )
    
    # Freeze the base model (Phase 1: Feature Extraction)
    base_model.trainable = False
    
    print(f"\n✓ MobileNetV2 base model loaded")
    print(f"  Total layers: {len(base_model.layers)}")
    print(f"  Trainable: False (frozen for Phase 1)")
    
    # Build the complete model
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    
    # Apply augmentation (only during training) — images are in [0, 255]
    x = data_augmentation(inputs)
    
    # MobileNetV2 preprocess_input scales [0, 255] → [-1, 1]
    # Images are already in [0, 255] range from the data pipeline
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Pass through base model (training=False keeps BatchNorm in inference mode,
    # which is recommended for transfer learning to prevent destabilizing updates)
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(DENSE_UNITS, activation='relu', name='dense_features')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create the model
    model = Model(inputs, outputs, name='PlantDiseaseDetector')
    
    # Compile for Phase 1 (Feature Extraction)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print(f"\n✓ Model built successfully")
    print(f"  Input shape: {INPUT_SHAPE}")
    print(f"  Output classes: {num_classes}")
    print(f"  Dense units: {DENSE_UNITS}")
    print(f"  Dropout rate: {DROPOUT_RATE}")
    
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    print(f"  Phase 1 learning rate: {LEARNING_RATE_PHASE1}")
    
    return model, base_model


def unfreeze_model(model, base_model):
    """
    Unfreeze the top layers of MobileNetV2 for fine-tuning (Phase 2).
    
    Args:
        model: The compiled model
        base_model: The MobileNetV2 base model
    
    Returns:
        model: Re-compiled model ready for fine-tuning
    """
    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING")
    print("=" * 60)
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze everything before FINE_TUNE_AT layer
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    
    unfrozen_layers = len(base_model.layers) - FINE_TUNE_AT
    print(f"  Unfreezing top {unfrozen_layers} layers of MobileNetV2")
    print(f"  Frozen layers: 0 to {FINE_TUNE_AT - 1}")
    print(f"  Trainable layers: {FINE_TUNE_AT} to {len(base_model.layers) - 1}")
    
    # Re-compile with a much lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"  New learning rate: {LEARNING_RATE_PHASE2}")
    print(f"  New trainable parameters: {trainable_params:,}")
    print("✓ Model unfrozen and re-compiled for fine-tuning")
    
    return model


def get_model_summary(model):
    """Print a formatted model summary."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    model.summary()


if __name__ == "__main__":
    # Test model creation
    model, base_model = build_model()
    get_model_summary(model)
    
    # Test fine-tuning preparation
    model = unfreeze_model(model, base_model)
