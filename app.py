"""
Plant Disease Detection System — Streamlit Web Application
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
    page_title="Plant Diagnostics Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Advanced CSS for Premium UI ---
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Core Background with deep space gradient */
    .stApp {
        background: radial-gradient(circle at 15% 50%, rgba(4, 21, 38, 1) 0%, rgba(2, 6, 12, 1) 100%);
        background-attachment: fixed;
    }
    
    /* Hide Default Streamlit Elements */
    #MainMenu, footer, header {visibility: hidden;}

    /* Top Navigation Tabs Styling (making them look like a floating nav) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: rgba(13, 24, 42, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 10px 20px;
        border-radius: 50px;
        border: 1px solid rgba(255,255,255,0.05);
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        background-color: transparent !important;
        border: none !important;
        height: 45px;
        border-radius: 30px;
        padding: 0 25px;
        font-weight: 500;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        color: #00ff88 !important;
        background: rgba(0, 255, 136, 0.1) !important;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.15);
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(13, 24, 42, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
        color: #e2e8f0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeIn 0.8s ease-out forwards;
        opacity: 0;
        margin-bottom: 25px;
    }

    .glass-card:hover {
        transform: translateY(-8px);
        border-color: rgba(0, 255, 136, 0.3);
        box-shadow: 0 25px 45px rgba(0, 255, 136, 0.15);
    }
    
    /* Result Specific Colors */
    .healthy-card {
        border-top: 4px solid #00ff88;
        background: linear-gradient(180deg, rgba(0, 255, 136, 0.05) 0%, rgba(13, 24, 42, 0.45) 100%);
    }
    
    .diseased-card {
        border-top: 4px solid #ff3366;
        background: linear-gradient(180deg, rgba(255, 51, 102, 0.05) 0%, rgba(13, 24, 42, 0.45) 100%);
    }

    /* Glowing Text */
    .glow-healthy {
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.6);
        font-weight: 700;
        font-size: 1.8rem;
        margin-bottom: 5px;
    }

    .glow-diseased {
        color: #ff3366;
        text-shadow: 0 0 20px rgba(255, 51, 102, 0.6);
        font-weight: 700;
        font-size: 1.8rem;
        margin-bottom: 5px;
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 50px;
        padding: 8px 20px;
        font-size: 1rem;
        color: #a0aec0;
        margin-top: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .metric-high {
        background: rgba(0, 255, 136, 0.1);
        border-color: rgba(0, 255, 136, 0.3);
        color: #00ff88;
    }
    
    /* Headers & Text */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    p {
        color: #94a3b8;
        line-height: 1.7;
    }

    .subtitle {
        color: #00ff88;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 15px;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stSpinner > div > div {
        border-color: #00ff88 transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "plant_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
IMG_SIZE = 224

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
    img_array = np.array(img, dtype=np.float32)
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
            'description': 'No clinical description parameters available.',
            'remedy': 'Please route to agricultural specialists for advanced diagnosis.'
        })
        results.append({
            'class_name': class_name,
            'confidence': confidence,
            **info
        })
    return results


# ===== MAIN APP =====
def main():
    
    # Custom Navigation Tabs
    tab_dashboard, tab_model_info = st.tabs(["AI Diagnostics Hub", "System Architecture Overview"])
    
    with tab_dashboard:
        # Header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; animation: fadeIn 0.5s ease-out;">
            <h1 style="font-size: 3rem; margin-bottom: 0;">Diagnostics Engine</h1>
            <p style="font-size: 1.2rem; color: #00ff88; letter-spacing: 1px;">Powered by MobileNetV2 Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)

        col_upload, col_result = st.columns([1, 1.4], gap="large")
        
        with col_upload:
            st.markdown("""
            <div class="glass-card" style="animation-delay: 0.1s;">
                <div class="subtitle">Input Data</div>
                <h3 style="margin-top: 0; margin-bottom: 15px;">Image Uploader</h3>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                label="Select visual subject", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True, caption="Visual Subject Captured")
                
                # Image metadata table
                st.markdown(f"""
                <div style="margin-top: 20px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px;">
                    <p style="margin: 0; font-size: 0.9rem;"><strong>Dimensions:</strong> {image.size[0]}px × {image.size[1]}px</p>
                    <p style="margin: 0; font-size: 0.9rem;"><strong>Color Space:</strong> {image.mode}</p>
                    <p style="margin: 0; font-size: 0.9rem;"><strong>Format:</strong> {image.format or 'RAW'}</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_result:
            if uploaded_file is not None:
                model = load_model()
                class_names = load_class_names()
                
                if model is None or class_names is None:
                    st.error("Engine failure: Neural network weights missing.")
                    return
                
                # Predict
                with st.spinner("Executing neural inference..."):
                    img_array = preprocess_image(image)
                    results = get_prediction(model, img_array, class_names, top_k=5)
                
                # Core result routing
                top = results[0]
                is_healthy = top['disease'].lower() == 'healthy'
                card_style = 'healthy-card' if is_healthy else 'diseased-card'
                text_style = 'glow-healthy' if is_healthy else 'glow-diseased'
                conf_high = 'metric-high' if top['confidence'] > 0.8 else ''
                
                # Report Core Block
                st.markdown(f"""
                <div class="glass-card {card_style}" style="animation-delay: 0.2s;">
                    <div class="subtitle">Assessment Result</div>
                    <div class="{text_style}">{top['disease']}</div>
                    <p style="color: #ffffff; font-size: 1.2rem; margin: 0; padding-bottom: 10px;">Subject: <strong>{top['plant']}</strong></p>
                    <div class="metric-badge {conf_high}">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 5px;"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>
                        Network Certainty Level: <strong>{top['confidence']*100:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_desc, col_treat = st.columns(2)
                with col_desc:
                    st.markdown(f"""
                    <div class="glass-card" style="animation-delay: 0.3s; height: 100%;">
                        <div class="subtitle">Pathology Details</div>
                        <p style="font-size: 0.95rem;">{top['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_treat:
                    st.markdown(f"""
                    <div class="glass-card" style="animation-delay: 0.4s; height: 100%;">
                        <div class="subtitle">Suggested Protocol</div>
                        <p style="font-size: 0.95rem;">{top['remedy']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability Distribution Chart Customizer
                st.markdown("""
                <div class="glass-card" style="animation-delay: 0.5s; padding-bottom: 10px;">
                    <div class="subtitle">Neural Probability Topology</div>
                """, unsafe_allow_html=True)
                
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                chart_data = {
                    f"{r['plant'].split()[0]} - {r['disease'][:15]}": r['confidence'] * 100 
                    for r in results
                }
                
                fig, ax = plt.subplots(figsize=(8, 2.5))
                # Make background totally transparent to let glassmorphism show
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                
                names = list(chart_data.keys())
                values = list(chart_data.values())
                colors = ['#00ff88' if 'Healthy' in n else '#3a4b60' for n in names]
                colors[0] = '#ff3366' if not is_healthy and colors[0] != '#00ff88' else colors[0]
                
                bars = ax.barh(range(len(names)), values, color=colors, height=0.4, edgecolor='none')
                
                for bar, val in zip(bars, values):
                    # Add value string
                    ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                           f'{val:.1f}%', va='center', color='#ffffff', fontweight='bold', fontsize=9)
                
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, color='#cbd5e1', fontsize=9)
                ax.set_xlabel('Probability (%)', color='#64748b', fontsize=9)
                ax.set_xlim([0, max(100, max(values) * 1.2)])
                ax.invert_yaxis()
                ax.tick_params(colors='#1e293b', length=0)
                
                # Clean up borders
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_color('#ffffff')
                ax.spines['bottom'].set_alpha(0.1)
                ax.grid(axis='x', alpha=0.1, color='#ffffff', linestyle=':')
                
                plt.tight_layout()
                st.pyplot(fig, transparent=True)
                plt.close()
                st.markdown("</div>", unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 100px 20px;">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#25354e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 20px;"><path d="M21.2 15c.7-1.2 1-2.5 .7-3.9-.6-2-2.4-3.5-4.4-3.5h-1.2c-.7-3-3.2-5.2-6.2-5.6-3-.3-5.9 1.3-7.3 4-1.2 2.5-1 6.5.5 8.8m8.5 7v-12m-4 4l4-4 4 4"></path></svg>
                    <h3 style="color: #64748b;">Awaiting Visual Telemetry</h3>
                    <p style="color: #475569; font-size: 1rem;">Please input a plant tissue sample via the upload interface.</p>
                </div>
                """, unsafe_allow_html=True)
                
    with tab_model_info:
        # About section styled with premium cards
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2.5rem; animation: fadeIn 0.5s ease-out;">
            <div class="subtitle">System Details</div>
            <h1 style="font-size: 2.5rem; margin-bottom: 0;">Architecture & Telemetry</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats Row
        col1, col2, col3, col4 = st.columns(4)
        stats = [
            ("38", "Network Taxonomy Classes", col1),
            ("14", "Distinct Agronomic Species", col2),
            ("54.3K", "Training Inferences", col3),
            ("224p", "Tensor Resolution Depth", col4)
        ]
        
        delay = 0.1
        for val, label, col in stats:
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 25px; animation-delay: {delay}s;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: #00ff88; text-shadow: 0 0 15px rgba(0, 255, 136, 0.4);">{val}</div>
                    <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            delay += 0.1
            
        st.markdown("""
        <div class="glass-card" style="animation-delay: 0.5s; display: flex; align-items: center; justify-content: center; text-align: center; padding: 50px;">
            <div>
                <h3 style="color: #00ff88; margin-bottom: 10px;">MobileNetV2 Processing Pipeline</h3>
                <p style="font-size: 1.1rem; max-width: 700px; margin: 0 auto;">
                    This system operates on a state-of-the-art convolutional neural network utilizing transfer learning. 
                    Feature extraction is localized across distinct botany phenotypes to precisely isolate geometric and chromatic pathological identifiers.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_cat1, col_cat2 = st.columns(2)
        with col_cat1:
            st.markdown("""
            <div class="glass-card" style="animation-delay: 0.6s;">
                <div class="subtitle">Covered Flora (Set A)</div>
                <div style="color: #cbd5e1; line-height: 2; font-size: 1.05rem;">
                • Apple &nbsp; • Blueberry &nbsp; • Cherry <br>
                • Corn (Maize) &nbsp; • Grape &nbsp; • Orange <br>
                • Peach
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_cat2:
            st.markdown("""
            <div class="glass-card" style="animation-delay: 0.7s;">
            <div class="subtitle">Covered Flora (Set B)</div>
                <div style="color: #cbd5e1; line-height: 2; font-size: 1.05rem;">
                • Bell Pepper &nbsp; • Potato &nbsp; • Raspberry <br>
                • Soybean &nbsp; • Squash &nbsp; • Strawberry <br>
                • Tomato
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
