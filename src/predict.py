"""
Prediction Module for Plant Leaf Disease Detection.
Predicts disease from a single leaf image with confidence scores.
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import (
    MODEL_SAVE_PATH_KERAS, MODELS_DIR, IMG_SIZE, DISEASE_INFO
)


def load_model_and_classes():
    """
    Load the trained model and class names.
    
    Returns:
        model: Loaded Keras model
        class_names: List of class names
    """
    # Load model
    model = tf.keras.models.load_model(MODEL_SAVE_PATH_KERAS)
    
    # Load class names
    class_names_path = os.path.join(MODELS_DIR, "class_names.json")
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    return model, class_names


def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        img_array: Preprocessed image array ready for prediction
        img_display: Original image for display
    """
    # Load and resize image
    img = Image.open(image_path).convert('RGB')
    img_display = img.copy()
    
    # Resize to model input size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array — keep in [0, 255] range
    # Model handles normalization internally via MobileNetV2 preprocess_input
    img_array = np.array(img, dtype=np.float32)
    
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img_display


def predict(image_path, top_k=3):
    """
    Predict the disease from a leaf image.
    
    Args:
        image_path: Path to the leaf image
        top_k: Number of top predictions to return
    
    Returns:
        results: List of dicts with class_name, confidence, plant, disease info
    """
    # Load model and class names
    model, class_names = load_model_and_classes()
    
    # Preprocess image
    img_array, img_display = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    
    # Get top-k predictions
    top_indices = np.argsort(predictions[0])[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        class_name = class_names[idx]
        confidence = float(predictions[0][idx])
        
        # Get disease info
        info = DISEASE_INFO.get(class_name, {
            'plant': class_name.split('___')[0] if '___' in class_name else 'Unknown',
            'disease': class_name.split('___')[1] if '___' in class_name else class_name,
            'description': 'No description available.',
            'remedy': 'Consult an agricultural expert.'
        })
        
        results.append({
            'class_name': class_name,
            'confidence': confidence,
            'plant': info['plant'],
            'disease': info['disease'],
            'description': info['description'],
            'remedy': info['remedy']
        })
    
    return results, img_display


def predict_and_display(image_path, top_k=3, save_path=None):
    """
    Predict disease and display results with the image.
    
    Args:
        image_path: Path to the leaf image
        top_k: Number of top predictions to show
        save_path: Optional path to save the result image
    """
    print(f"\n{'='*60}")
    print(f"🌿 PREDICTING DISEASE FOR: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    results, img_display = predict(image_path, top_k)
    
    # Print results
    print(f"\n{'─'*50}")
    for i, result in enumerate(results):
        marker = "🏆" if i == 0 else f"#{i+1}"
        print(f"\n  {marker} {result['plant']} — {result['disease']}")
        print(f"     Confidence: {result['confidence']*100:.2f}%")
        if i == 0:  # Show details only for top prediction
            print(f"     Description: {result['description']}")
            print(f"     Remedy: {result['remedy']}")
    print(f"\n{'─'*50}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Left: Image
    axes[0].imshow(img_display)
    axes[0].set_title('Input Leaf Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Right: Top predictions bar chart
    names = [f"{r['plant']}\n{r['disease']}" for r in results]
    confidences = [r['confidence'] * 100 for r in results]
    colors = ['#27ae60' if r['disease'] == 'Healthy' else '#e74c3c' for r in results]
    
    bars = axes[1].barh(range(len(names)), confidences, color=colors, edgecolor='white', height=0.5)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=10)
    axes[1].set_xlabel('Confidence (%)', fontsize=12)
    axes[1].set_title('Top Predictions', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0, 110])
    axes[1].invert_yaxis()
    
    for bar, conf in zip(bars, confidences):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{conf:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.suptitle('🌿 Plant Disease Detection Result', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Result saved to: {save_path}")
    
    plt.close()
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path>")
        print("Example: python -m src.predict test_leaf.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    from src.config import RESULTS_DIR
    save_path = os.path.join(RESULTS_DIR, "prediction_result.png")
    results = predict_and_display(image_path, top_k=3, save_path=save_path)
