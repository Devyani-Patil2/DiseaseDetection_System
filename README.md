# 🌿 AI-Powered Early Detection of Plant Leaf Diseases Using Image Processing

## Project Overview

This project implements an **AI-powered system** for the early detection of plant leaf diseases using **deep learning** and **image processing**. The system can identify **38 different disease classes** across **14 plant species** from leaf images with high accuracy.

### Key Features
- 🧠 **Transfer Learning** with MobileNetV2 (pre-trained on ImageNet)
- 📊 **38 Disease Classes** covering 14 plant species
- 📈 **Two-Phase Training** — Feature extraction + Fine-tuning
- ⚖️ **Class Imbalance Handling** with computed class weights
- 🎨 **Data Augmentation** for robust model performance
- 🌐 **Streamlit Web App** for real-time disease detection
- 📋 **Comprehensive Evaluation** — Confusion matrix, classification report, per-class accuracy

---

## 📁 Project Structure

```
CP/
├── plantvillage dataset/           # PlantVillage dataset
│   ├── color/                      # RGB images (used for training)
│   ├── grayscale/                  # Grayscale version
│   └── segmented/                  # Segmented version
├── src/
│   ├── __init__.py                 # Package init
│   ├── config.py                   # Configuration & hyperparameters
│   ├── data_preprocessing.py       # Data loading & augmentation
│   ├── model.py                    # MobileNetV2 model architecture
│   ├── train.py                    # Training pipeline
│   ├── evaluate.py                 # Evaluation & metrics
│   └── predict.py                  # Single image prediction
├── models/                         # Saved trained models
│   ├── plant_disease_model.keras   # Best model
│   └── class_names.json            # Class name mapping
├── results/                        # Evaluation results
│   ├── training_curves.png         # Accuracy/loss plots
│   ├── confusion_matrix.png        # Confusion matrix heatmap
│   ├── per_class_accuracy.png      # Per-class accuracy chart
│   └── classification_report.txt   # Precision/recall/F1 report
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA for faster training

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Dataset
Ensure the PlantVillage dataset is in the `plantvillage dataset/color/` directory with 38 class folders.

---

## 🚀 Usage

### 1. Train the Model

```bash
python -m src.train
```

This will:
- Load and preprocess the PlantVillage dataset (54,305 images)
- Split into train (80%), validation (10%), and test (10%)
- Phase 1: Train classification head (5 epochs, frozen MobileNetV2)
- Phase 2: Fine-tune top layers (10 epochs, unfrozen MobileNetV2)
- Save the best model to `models/plant_disease_model.keras`

**Expected training time:**
- GPU: ~15-30 minutes  
- CPU: ~2-4 hours

### 2. Evaluate the Model

```bash
python -m src.evaluate
```

Generates:
- Training accuracy/loss curves
- Confusion matrix heatmap
- Per-class accuracy chart
- Classification report (precision, recall, F1-score)

### 3. Predict on a Single Image

```bash
python -m src.predict path/to/leaf_image.jpg
```

### 4. Launch the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload any leaf image to get instant disease detection.

---

## 🧠 Model Architecture

### MobileNetV2 + Custom Classification Head

```
Input Image (224×224×3)
    ↓
Data Augmentation (training only)
    - Random Flip, Rotation, Zoom
    - Random Translation, Brightness, Contrast
    ↓
MobileNetV2 (pre-trained on ImageNet)
    - 155 layers
    - Phase 1: Fully frozen
    - Phase 2: Top 30 layers unfrozen
    ↓
Global Average Pooling 2D
    ↓
Dense Layer (256 units, ReLU activation)
    ↓
Dropout (0.5)
    ↓
Dense Layer (38 units, Softmax activation)
    ↓
Output: Disease Class + Confidence Score
```

### Why MobileNetV2?
- **Lightweight** — Only 3.4M parameters (efficient for deployment)
- **Accurate** — Strong performance on image classification tasks
- **Transfer Learning** — Pre-trained features reduce training time dramatically
- **Mobile-Ready** — Can be deployed on edge devices and drones

### Training Strategy
| Phase | Epochs | Learning Rate | What's Trained |
|-------|--------|--------------|----------------|
| Phase 1 (Feature Extraction) | 5 | 1e-3 | Only classification head |
| Phase 2 (Fine-Tuning) | 10 | 1e-5 | Top 30 MobileNetV2 layers + head |

---

## 📊 Dataset

### PlantVillage Dataset
- **Total Images:** 54,305
- **Total Classes:** 38
- **Image Size:** 256×256 (resized to 224×224)
- **Format:** RGB (JPG)

### Supported Plants & Diseases

| Plant | Diseases Detected |
|-------|-------------------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Orange | Huanglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Bell Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🔧 Configuration

All hyperparameters are centralized in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_SIZE` | 224 | Input image resolution |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS_PHASE1` | 5 | Feature extraction epochs |
| `EPOCHS_PHASE2` | 10 | Fine-tuning epochs |
| `LEARNING_RATE_PHASE1` | 1e-3 | Phase 1 learning rate |
| `LEARNING_RATE_PHASE2` | 1e-5 | Phase 2 learning rate |
| `DROPOUT_RATE` | 0.5 | Dropout rate |
| `DENSE_UNITS` | 256 | Dense layer units |
| `FINE_TUNE_AT` | 125 | Layer to start unfreezing |

---

## 📚 Technologies Used

- **Python 3.8+** — Programming language
- **TensorFlow / Keras** — Deep learning framework
- **MobileNetV2** — Pre-trained CNN for transfer learning
- **OpenCV** — Image processing
- **Scikit-learn** — Evaluation metrics
- **Matplotlib / Seaborn** — Visualization
- **Streamlit** — Web application framework
- **NumPy / Pandas** — Data processing

---

## 👥 Team

**Project:** AI-Powered Early Detection of Plant Leaf Diseases Using Image Processing  
**Organization:** Aerobetics

---
