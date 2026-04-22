"""
Plant Pathology Diagnostic System
AI-Powered Early Detection of Plant Leaf Diseases Using Image Processing
"""

import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Pathology Diagnostics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean Professional CSS ---
st.markdown("""
<style>
    /* Professional Dark Dashboard Styling */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Header styling */
    .main-header {
        background-color: #1e293b;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Result cards */
    .result-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.5rem 0 1rem 0;
        border: 1px solid #334155;
    }
    
    .result-card.healthy {
        border-top: 4px solid #10b981;
    }
    
    .result-card.diseased {
        border-top: 4px solid #ef4444;
    }
    
    .disease-name {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 0.8rem 0;
        letter-spacing: 0.025em;
    }
    
    .confidence-high {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .confidence-medium {
        background-color: rgba(245, 158, 11, 0.1);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .info-section {
        background-color: #1e293b;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.8rem 0;
        border: 1px solid #334155;
    }
    
    .info-section h4 {
        color: #f8fafc;
        margin-bottom: 0.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.3rem;
    }
    
    .info-section p {
        color: #cbd5e1;
        line-height: 1.5;
        margin: 0;
        font-size: 0.95rem;
    }
    
    /* Stats cards */
    .stat-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 0.8rem 0.5rem;
        text-align: center;
        border: 1px solid #334155;
        margin-bottom: 0.5rem;
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #3b82f6;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-top: 0.2rem;
        font-weight: 600;
    }
    
    /* Upload area formatting */
    .upload-prompt {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "plant_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
IMG_SIZE = 224

# Disease info dictionary (imported from config)
from src.config import DISEASE_INFO


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        return None
    with open(CLASS_NAMES_PATH, 'r') as f:
        return json.load(f)


def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)  # Keep [0, 255] — model handles normalization
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_prediction(model, img_array, class_names, top_k=5):
    predictions = model.predict(img_array, verbose=0)
    top_indices = np.argsort(predictions[0])[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        class_name = class_names[idx]
        confidence = float(predictions[0][idx])
        info = DISEASE_INFO.get(class_name, {
            'plant': class_name.split('___')[0].replace('_', ' ') if '___' in class_name else 'Unknown',
            'disease': class_name.split('___')[1].replace('_', ' ') if '___' in class_name else class_name.replace('_', ' '),
            'description': 'Description data unavailable.',
            'remedy': 'Consult a certified agronomist for further instructions.'
        })
        results.append({
            'class_name': class_name,
            'confidence': confidence,
            **info
        })
    return results


# ===== MAIN APP =====
def main():
    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>Plant Pathology Diagnostic System</h1>
        <p>Automated feature-extraction utilizing centralized architecture</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar (Telemetry Panel) ---
    with st.sidebar:
        st.markdown("### System Telemetry")
        st.markdown("""
        <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem;">
        Neural network parameters trained extensively via the PlantVillage corpus.
        </div>
        """, unsafe_allow_html=True)
        
        # Stats
        st.markdown("<h4 style='color: #f8fafc; font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase;'>Dataset Parameters</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">38</div>
                <div class="stat-label">Classes</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">14</div>
                <div class="stat-label">Species</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">54K</div>
                <div class="stat-label">Images</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">224px</div>
                <div class="stat-label">Tensor</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<h4 style='color: #f8fafc; font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase;'>Diagnostic Species List</h4>", unsafe_allow_html=True)
        
        # Clean two-column list
        scol1, scol2 = st.columns(2)
        with scol1:
            st.markdown("""
            <div style="color: #cbd5e1; font-size: 0.85rem;">
            Apple<br>Blueberry<br>Cherry<br>Corn<br>Grape<br>Orange<br>Peach
            </div>
            """, unsafe_allow_html=True)
        with scol2:
            st.markdown("""
            <div style="color: #cbd5e1; font-size: 0.85rem;">
            Bell Pepper<br>Potato<br>Raspberry<br>Soybean<br>Squash<br>Strawberry<br>Tomato
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("""
        <div style="color: #64748b; font-size: 0.8rem;">
        SYSTEM PROTOCOL: Upload specimen sample. Await inference pipeline completion. Review localized treatment protocol.
        </div>
        """, unsafe_allow_html=True)
    
    # --- Main Content ---
    col_upload, col_result = st.columns([1, 1], gap="medium")
    
    with col_upload:
        st.markdown("### Image Input Stream")
        st.markdown('<div class="upload-prompt">Upload visual subject telemetry in standard encoding (JPG/PNG).</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            label="Upload visual subject telemetry",
            type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Subject Captured ({image.size[0]}x{image.size[1]}px)", use_container_width=True)
    
    with col_result:
        st.markdown("### Diagnostic Readout")
        
        if uploaded_file is not None:
            model = load_model()
            class_names = load_class_names()
            
            if model is None or class_names is None:
                st.error("System configuration error: Neural Network assets missing.")
                return
            
            with st.spinner("Processing visual inference..."):
                img_array = preprocess_image(image)
                results = get_prediction(model, img_array, class_names, top_k=5)
            
            top = results[0]
            is_healthy = top['disease'].lower() == 'healthy'
            card_class = 'healthy' if is_healthy else 'diseased'
            status_color = '#10b981' if is_healthy else '#ef4444'
            conf_class = 'confidence-high' if top['confidence'] > 0.8 else 'confidence-medium'
            
            # Clinical Result Block
            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="disease-name" style="color: {status_color};">
                    Primary Diagnosis: {top['disease']}
                </div>
                <div style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 0.5rem; letter-spacing: 0.5px;">
                    Target Specimen: <strong style="color: #f8fafc;">{top['plant']}</strong>
                </div>
                <div class="confidence-badge {conf_class}">
                    Inference Certainty: {top['confidence']*100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical Detailed Breakdowns
            st.markdown(f"""
            <div class="info-section">
                <h4>Pathology Origin & Characteristics</h4>
                <p>{top['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-section">
                <h4>Agricultural Action Protocol</h4>
                <p>{top['remedy']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simple, Clean Matplotlib Chart
            st.markdown("#### Probability Vector Analysis")
            chart_data = {
                f"{r['plant']} - {r['disease']}": r['confidence'] * 100 
                for r in results
            }
            
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Dark theme, robust rendering
            fig, ax = plt.subplots(figsize=(6, 2.5))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#1e293b')
            
            names = list(chart_data.keys())
            values = list(chart_data.values())
            colors = ['#10b981' if 'Healthy' in n else '#ef4444' for n in names]
            
            bars = ax.barh(range(len(names)), values, color=colors, edgecolor='none', height=0.4)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', color='#f8fafc', fontweight='bold', fontsize=9)
            
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, color='#cbd5e1', fontsize=9)
            ax.set_xlabel('Classification Probability (%)', color='#94a3b8', fontsize=10)
            ax.set_xlim([0, max(100, max(values) * 1.2)])
            ax.invert_yaxis()
            
            # Very standard axis cleaning
            ax.tick_params(colors='#64748b', length=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_visible(False)
            ax.grid(axis='x', alpha=0.5, color='#334155', linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        else:
            st.markdown("""
            <div style="background-color: #1e293b; border: 1px dashed #334155; 
                        border-radius: 8px; padding: 3rem; text-align: center; margin-top: 1rem;">
                <div style="color: #64748b; font-size: 1.2rem; font-weight: 500; margin-bottom: 0.5rem;">
                    Awaiting Input Data
                </div>
                <div style="color: #475569; font-size: 0.9rem;">
                    The diagnostic pipeline is idle. Input image required.
                </div>
            </div>
            """, unsafe_allow_html=True)
            

if __name__ == "__main__":
    main()
