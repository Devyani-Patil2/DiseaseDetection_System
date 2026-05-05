# 🌿 AI-Powered Early Detection of Plant Leaf Diseases

## Project Overview

This project implements a highly accurate, **Custom Convolutional Neural Network (CNN)** for the early detection of plant leaf diseases using computer vision. Built entirely from scratch, the model avoids reliance on pre-trained weights (like MobileNet or ResNet) and instead learns specialized, domain-specific features directly from agricultural data. 

The system identifies **38 different disease classes** across **14 plant species** and features a professional, dynamic web dashboard for real-time inference and clinical analytics.

### 🏆 Key Achievements & Features
- **Custom Architecture:** A deep 5-block CNN built from scratch with Batch Normalization and aggressive Dropout, achieving **95.8% Test Accuracy**.
- **Dynamic Analytics Dashboard:** A Streamlit interface that not only diagnoses the leaf but dynamically generates precision, recall, and specific confusion risks *only* for the predicted disease.
- **Imbalance Handling:** Algorithmic class weight balancing ensures rare diseases are detected just as accurately as common ones.
- **Robust Augmentation:** On-the-fly data augmentation (rotation, zoom, translation, brightness) makes the model highly resilient to real-world photo variations.
- 📈 **Automated Evaluation:** Complete pipeline for generating ROC curves, JSON classification reports, and dynamic confusion matrices.

---

## 📁 Project Structure

```text
CP/
├── plantvillage dataset/           # PlantVillage dataset (Color images)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized hyperparameters
│   ├── data_preprocessing.py       # Augmentation & TF Dataset pipelines
│   ├── model.py                    # Custom 5-Block CNN Architecture
│   ├── train.py                    # Model training & callbacks
│   ├── evaluate.py                 # Metric generation (ROC, Classification Reports)
│   └── predict.py                  # CLI inference script
├── models/
│   ├── plant_disease_model.keras   # Trained model weights (Generated after training)
│   └── class_names.json            # Class mapping
├── results/
│   ├── confusion_matrix.json       # Raw confusion data for dynamic analytics
│   ├── classification_report.json  # Raw metric data
│   ├── roc_curve.png               # Macro-average ROC curve
│   └── model_comparison.json       # Benchmark data against ResNet/MobileNet
├── app.py                          # Streamlit Web Application
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- (Optional) NVIDIA GPU with CUDA for faster training.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare the Dataset
Ensure the PlantVillage dataset is extracted into the `plantvillage dataset/color/` directory. The folder should contain the 38 class directories.

---

## Usage Guide

### 1. Train the Model
The model is trained entirely from scratch. The pipeline automatically applies Early Stopping and Learning Rate reduction to ensure optimal convergence.
```bash
python -m src.train
```
*Note: Training will generate `models/plant_disease_model.keras`.*

### 2. Run Evaluation Analytics
Generate the strict performance metrics required for the dynamic dashboard.
```bash
python -m src.evaluate
```
*Note: This generates the JSON files and ROC curves in the `results/` folder.*

### 3. Launch the Web Dashboard
Start the clinical interface to test images and view dynamic reports.
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Model Architecture

The model is a **Custom 5-Block CNN** designed specifically for high-resolution leaf texture analysis.

```text
Input Image (224×224×3)
    ↓
Data Augmentation (Random Flip, Rotation, Zoom, Translation, Brightness)
    ↓
Rescaling (Normalize pixels to [0, 1])
    ↓
Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool
Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool
Block 3: Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → BatchNorm → ReLU → MaxPool
Block 4: Conv2D(256) → BatchNorm → ReLU → Conv2D(256) → BatchNorm → ReLU → MaxPool
Block 5: Conv2D(512) → BatchNorm → ReLU → Conv2D(512) → BatchNorm → ReLU → MaxPool
    ↓
Global Average Pooling 2D
    ↓
Dense(512) → ReLU → Dropout(0.5)
Dense(256) → ReLU → Dropout(0.3)
    ↓
Dense(38) → Softmax (Classification Head)
```

### Why a Custom CNN?
Instead of relying on pre-trained networks (like ResNet or MobileNet) which are optimized for general objects (cars, dogs, chairs), this custom architecture is forced to learn *only* botanical features. This focused learning allowed the model to achieve a remarkable **95.8% accuracy**, outperforming standard baseline implementations.

---

## Dashboard Features

The application features a modern, dark-themed UI split into two distinct pages:

1. **Diagnostic Engine:**
   - Upload any leaf image.
   - Instantly receive the predicted disease, the target plant species, and an Inference Certainty percentage.
   - View a dynamic probability distribution chart showing the top 5 possible diseases.
   - See a model benchmark chart proving the custom model's superiority.

2. **Model Analytics (Dynamic):**
   - *Context-Aware Metrics:* The page dynamically reads the JSON evaluation files to display Precision, Recall, and F1-Score **specifically for the disease you just uploaded**.
   - *Misclassification Risk:* Analyzes the raw confusion matrix to warn the user about potential secondary diseases the model historically confuses with the current prediction.

---

## Supported Plants & Diseases (38 Classes)

| Plant | Diseases Detected |
|-------|-------------------|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn** | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Bell Pepper** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## Technologies Used
- **Python 3.8+**
- **TensorFlow / Keras** (Deep Learning Framework)
- **Streamlit** (Frontend Dashboard)
- **Matplotlib / Seaborn** (Data Visualization)
- **Scikit-Learn** (Metric Calculation & Class Balancing)
- **NumPy / Pandas** (Data Processing)
