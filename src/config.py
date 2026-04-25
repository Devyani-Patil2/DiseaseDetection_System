"""
Configuration file for Plant Leaf Disease Detection Project.
All hyperparameters, paths, and constants are defined here.
"""

import os

# ============================================================
# PATHS
# ============================================================
# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path (using the 'color' version of PlantVillage)
DATASET_DIR = os.path.join(BASE_DIR, "plantvillage dataset", "color")

# Output directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create output directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model save path
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "plant_disease_model.h5")
MODEL_SAVE_PATH_KERAS = os.path.join(MODELS_DIR, "plant_disease_model.keras")

# ============================================================
# IMAGE PARAMETERS
# ============================================================
IMG_SIZE = 224          # Input image size
IMG_CHANNELS = 3        # RGB
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 32
EPOCHS_PHASE1 = 40      # More epochs needed when training from scratch
LEARNING_RATE_PHASE1 = 5e-4
DROPOUT_RATE = 0.5
DENSE_UNITS = 256

# ============================================================
# DATA SPLIT
# ============================================================
VALIDATION_SPLIT = 0.1  # 10% for validation
TEST_SPLIT = 0.1        # 10% for testing
TRAIN_SPLIT = 0.8       # 80% for training
RANDOM_SEED = 42

# ============================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================
ROTATION_RANGE = 30
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False
BRIGHTNESS_RANGE = (0.8, 1.2)
FILL_MODE = "nearest"



# ============================================================
# CALLBACKS
# ============================================================
EARLY_STOP_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# ============================================================
# CLASS NAMES (38 classes from PlantVillage)
# ============================================================
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

NUM_CLASSES = len(CLASS_NAMES)  # 38

# ============================================================
# DISEASE INFO (for the Streamlit app)
# ============================================================
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'plant': 'Apple',
        'disease': 'Apple Scab',
        'description': 'A fungal disease caused by Venturia inaequalis. Appears as olive-green to brown lesions on leaves.',
        'remedy': 'Apply fungicides like captan or myclobutanil. Remove fallen leaves. Plant resistant varieties.'
    },
    'Apple___Black_rot': {
        'plant': 'Apple',
        'disease': 'Black Rot',
        'description': 'Caused by the fungus Botryosphaeria obtusa. Shows brown rotting spots with concentric rings.',
        'remedy': 'Prune dead branches. Apply fungicides during growing season. Remove mummified fruits.'
    },
    'Apple___Cedar_apple_rust': {
        'plant': 'Apple',
        'disease': 'Cedar Apple Rust',
        'description': 'Caused by Gymnosporangium juniperi-virginianae. Orange-yellow spots on leaves.',
        'remedy': 'Remove nearby cedar trees. Apply fungicides in spring. Plant resistant varieties.'
    },
    'Apple___healthy': {
        'plant': 'Apple',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Continue regular care and monitoring.'
    },
    'Blueberry___healthy': {
        'plant': 'Blueberry',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Maintain proper soil pH (4.5-5.5).'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'plant': 'Cherry',
        'disease': 'Powdery Mildew',
        'description': 'White powdery coating on leaves caused by Podosphaera clandestina.',
        'remedy': 'Apply sulfur-based fungicides. Improve air circulation. Avoid overhead watering.'
    },
    'Cherry_(including_sour)___healthy': {
        'plant': 'Cherry',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Continue regular pruning and care.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'plant': 'Corn',
        'disease': 'Cercospora Leaf Spot / Gray Leaf Spot',
        'description': 'Rectangular gray-brown lesions on leaves caused by Cercospora zeae-maydis.',
        'remedy': 'Rotate crops. Use resistant hybrids. Apply foliar fungicides if severe.'
    },
    'Corn_(maize)___Common_rust_': {
        'plant': 'Corn',
        'disease': 'Common Rust',
        'description': 'Reddish-brown pustules on leaf surfaces caused by Puccinia sorghi.',
        'remedy': 'Plant resistant varieties. Apply fungicides early. Monitor fields regularly.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'plant': 'Corn',
        'disease': 'Northern Leaf Blight',
        'description': 'Long cigar-shaped gray-green lesions caused by Exserohilum turcicum.',
        'remedy': 'Use resistant hybrids. Apply foliar fungicides. Practice crop rotation.'
    },
    'Corn_(maize)___healthy': {
        'plant': 'Corn',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Maintain proper nutrition and irrigation.'
    },
    'Grape___Black_rot': {
        'plant': 'Grape',
        'disease': 'Black Rot',
        'description': 'Brown circular lesions with black borders caused by Guignardia bidwellii.',
        'remedy': 'Remove mummified berries. Apply fungicides from bud break. Prune for air circulation.'
    },
    'Grape___Esca_(Black_Measles)': {
        'plant': 'Grape',
        'disease': 'Esca (Black Measles)',
        'description': 'Tiger-stripe patterns on leaves. Complex disease involving multiple fungi.',
        'remedy': 'No effective cure. Remove severely affected vines. Apply wound protectants after pruning.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'plant': 'Grape',
        'disease': 'Leaf Blight (Isariopsis Leaf Spot)',
        'description': 'Dark brown spots with yellow halos on leaves.',
        'remedy': 'Apply copper-based fungicides. Improve air circulation. Remove infected leaves.'
    },
    'Grape___healthy': {
        'plant': 'Grape',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Continue regular vineyard management.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'plant': 'Orange',
        'disease': 'Huanglongbing (Citrus Greening)',
        'description': 'Caused by Candidatus Liberibacter. Yellowing of leaves in asymmetric patterns.',
        'remedy': 'No cure exists. Control Asian citrus psyllid. Remove infected trees. Plant disease-free stock.'
    },
    'Peach___Bacterial_spot': {
        'plant': 'Peach',
        'disease': 'Bacterial Spot',
        'description': 'Small dark spots on leaves caused by Xanthomonas campestris.',
        'remedy': 'Apply copper sprays. Plant resistant varieties. Avoid overhead irrigation.'
    },
    'Peach___healthy': {
        'plant': 'Peach',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Continue proper orchard management.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'plant': 'Bell Pepper',
        'disease': 'Bacterial Spot',
        'description': 'Water-soaked spots on leaves caused by Xanthomonas campestris.',
        'remedy': 'Use disease-free seeds. Apply copper bactericides. Practice crop rotation.'
    },
    'Pepper,_bell___healthy': {
        'plant': 'Bell Pepper',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Maintain proper watering and nutrition.'
    },
    'Potato___Early_blight': {
        'plant': 'Potato',
        'disease': 'Early Blight',
        'description': 'Dark brown spots with concentric rings (target pattern) caused by Alternaria solani.',
        'remedy': 'Apply fungicides (chlorothalonil, mancozeb). Practice crop rotation. Remove plant debris.'
    },
    'Potato___Late_blight': {
        'plant': 'Potato',
        'disease': 'Late Blight',
        'description': 'Water-soaked dark lesions, rapidly spreading. Caused by Phytophthora infestans.',
        'remedy': 'Apply fungicides immediately. Destroy infected plants. Use resistant varieties.'
    },
    'Potato___healthy': {
        'plant': 'Potato',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Monitor for signs of blight regularly.'
    },
    'Raspberry___healthy': {
        'plant': 'Raspberry',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Maintain proper pruning and drainage.'
    },
    'Soybean___healthy': {
        'plant': 'Soybean',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Continue regular crop management.'
    },
    'Squash___Powdery_mildew': {
        'plant': 'Squash',
        'disease': 'Powdery Mildew',
        'description': 'White powdery spots on leaf surfaces caused by Erysiphe cichoracearum.',
        'remedy': 'Apply neem oil or sulfur fungicides. Improve spacing. Water at soil level.'
    },
    'Strawberry___Leaf_scorch': {
        'plant': 'Strawberry',
        'disease': 'Leaf Scorch',
        'description': 'Purple-red spots on leaves that may merge, caused by Diplocarpon earlianum.',
        'remedy': 'Remove infected leaves. Apply fungicides. Avoid overhead watering.'
    },
    'Strawberry___healthy': {
        'plant': 'Strawberry',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Maintain proper mulching and care.'
    },
    'Tomato___Bacterial_spot': {
        'plant': 'Tomato',
        'disease': 'Bacterial Spot',
        'description': 'Small dark raised spots on leaves caused by Xanthomonas species.',
        'remedy': 'Apply copper-based sprays. Use disease-free seeds. Practice crop rotation.'
    },
    'Tomato___Early_blight': {
        'plant': 'Tomato',
        'disease': 'Early Blight',
        'description': 'Dark concentric ring spots (bullseye pattern) caused by Alternaria solani.',
        'remedy': 'Apply fungicides. Mulch around plants. Remove lower infected leaves.'
    },
    'Tomato___Late_blight': {
        'plant': 'Tomato',
        'disease': 'Late Blight',
        'description': 'Large water-soaked brown patches caused by Phytophthora infestans.',
        'remedy': 'Apply fungicides immediately. Remove infected plants. Avoid wet conditions.'
    },
    'Tomato___Leaf_Mold': {
        'plant': 'Tomato',
        'disease': 'Leaf Mold',
        'description': 'Yellow spots on upper leaf surface, olive-green mold underneath. Caused by Passalora fulva.',
        'remedy': 'Improve ventilation. Reduce humidity. Apply fungicides. Use resistant varieties.'
    },
    'Tomato___Septoria_leaf_spot': {
        'plant': 'Tomato',
        'disease': 'Septoria Leaf Spot',
        'description': 'Small circular spots with dark borders and gray centers caused by Septoria lycopersici.',
        'remedy': 'Remove infected leaves. Apply fungicides. Avoid overhead watering.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'plant': 'Tomato',
        'disease': 'Spider Mites (Two-spotted)',
        'description': 'Tiny yellow spots on leaves, fine webbing. Caused by Tetranychus urticae.',
        'remedy': 'Spray with insecticidal soap or neem oil. Increase humidity. Introduce predatory mites.'
    },
    'Tomato___Target_Spot': {
        'plant': 'Tomato',
        'disease': 'Target Spot',
        'description': 'Brown spots with concentric rings and yellow halos caused by Corynespora cassiicola.',
        'remedy': 'Apply fungicides. Improve air circulation. Remove infected leaves.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'plant': 'Tomato',
        'disease': 'Yellow Leaf Curl Virus',
        'description': 'Upward curling of leaves with yellowing. Transmitted by whiteflies.',
        'remedy': 'Control whiteflies. Remove infected plants. Use resistant varieties. Apply reflective mulch.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'plant': 'Tomato',
        'disease': 'Mosaic Virus',
        'description': 'Mottled green-yellow pattern on leaves. Highly contagious viral disease.',
        'remedy': 'No cure. Remove infected plants. Disinfect tools. Use resistant varieties.'
    },
    'Tomato___healthy': {
        'plant': 'Tomato',
        'disease': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'remedy': 'No treatment needed. Continue regular care and staking.'
    },
}
