# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import requests

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI MHT-CET College Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM STYLING (Gradients, Glassmorphism, Animations, Mobile Responsive)
# -----------------------------------------------------------------------------
custom_css = """
<style>
/* Font Imports */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background gradient for main content */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
    color: #f8fafc;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px 0 rgba(112, 0, 255, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.25);
}

/* Prediction Result Highlighting */
.result-header {
    font-size: 0.95rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #a855f7;
    margin-bottom: 6px;
}

.result-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0px;
}

.result-card-inst {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
    border-left: 5px solid #6366f1;
}

.result-card-course {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
    border-left: 5px solid #10b981;
}

/* Rounded Buttons */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    font-size: 1.05rem;
    font-weight: 700;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
    box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
    transform: translateY(-2px);
}

/* Custom Header Banner */
.hero-title {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0px;
}

/* Footer Styling */
.footer {
    position: relative;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 20px 0;
    font-size: 0.85rem;
    color: #94a3b8;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 40px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS & ASSETS LOADERS
# -----------------------------------------------------------------------------
def load_lottie_url(url: str):
    """Fetch Lottie animation JSON directly via HTTP."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

@st.cache_resource
def load_artifacts():
    """Load model and encoders safely."""
    artifacts = {}
    files = {
        'model': 'collegename_model.pkl',
        'gender_encoder': 'gender_encoder.pkl',
        'category_encoder': 'category_encoder.pkl',
        'seat_encoder': 'seat_encoder.pkl',
        'target_encoder': 'target_encoder.pkl'
    }
    
    missing_files = []
    for key, filename in files.items():
        if os.path.exists(filename):
            try:
                artifacts[key] = joblib.load(filename)
            except Exception as e:
                return None, f"Error loading {filename}: {str(e)}"
        else:
            missing_files.append(filename)
            
    if missing_files:
        return None, f"Missing required file(s): {', '.join(missing_files)}"
        
    return artifacts, None

# -----------------------------------------------------------------------------
# APPLICATION HEADER
# -----------------------------------------------------------------------------
hero_col1, hero_col2 = st.columns([3, 1])

with hero_col1:
    st.markdown('<h1 class="hero-title">AI MHT-CET College Predictor</h1>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Smart Machine Learning Admission Predictions for Engineering Seats in Maharashtra</p>", unsafe_allow_html=True)

with hero_col2:
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path)
        st.image(logo_img, width=120)
    else:
        st.markdown("<h3>🎓 MHT-CET</h3>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# LOAD MODEL & ENCODERS
# -----------------------------------------------------------------------------
artifacts, error_msg = load_artifacts()

if error_msg:
    st.error(f"⚠️ App Setup Error: {error_msg}")
    st.info("Ensure all `.pkl` files are placed in the root directory alongside `app.py`.")
    st.stop()

model = artifacts['model']
gender_encoder = artifacts['gender_encoder']
category_encoder = artifacts['category_encoder']
seat_encoder = artifacts['seat_encoder']
target_encoder = artifacts['target_encoder']

# -----------------------------------------------------------------------------
# SIDEBAR INPUT FORM
# -----------------------------------------------------------------------------
st.sidebar.title("📌 Student Details")
st.sidebar.markdown("Enter your rank & seat preferences to predict admission results.")

# Input fields
merit_number = st.sidebar.number_input("Merit Number", min_value=1, max_value=300000, value=15000, step=1)
percentile = st.sidebar.number_input("MHTCET Percentile", min_value=0.0, max_value=100.0, value=92.50, step=0.01, format="%.2f")

# Extract categorical choices safely from encoders
gender_options = list(gender_encoder.classes_)
category_options = list(category_encoder.classes_)
seat_options = list(seat_encoder.classes_)

gender = st.sidebar.selectbox("Gender", options=gender_options)
category = st.sidebar.selectbox("Category", options=category_options)
seat_alloted = st.sidebar.selectbox("Seat Alloted", options=seat_options)

predict_btn = st.sidebar.button("✨ Predict College")

# Sidebar Information
st.sidebar.markdown("---")
st.sidebar.caption("⚡ Powered by Scikit-Learn & Streamlit")

# -----------------------------------------------------------------------------
# MAIN CONTENT / TABS
# -----------------------------------------------------------------------------
tab_predict, tab_about, tab_stats = st.tabs(["🎯 Prediction", "ℹ️ About Project", "📊 Model Info & Analytics"])

with tab_predict:
    if predict_btn:
        with st.spinner("Analyzing cutoff trends and making predictions..."):
            try:
                # Encode inputs using loaded LabelEncoders
                gender_enc = gender_encoder.transform([gender])[0]
                category_enc = category_encoder.transform([category])[0]
                seat_enc = seat_encoder.transform([seat_alloted])[0]

                # Match model input feature order
                if hasattr(model, "feature_names_in_"):
                    feature_names = list(model.feature_names_in_)
                    input_dict = {
                        'Merit Number': merit_number,
                        'MHTCET Percentile': percentile,
                        'Gender': gender_enc,
                        'Category': category_enc,
                        'Seat Alloted': seat_enc
                    }
                    input_df = pd.DataFrame([[input_dict.get(col, 0) for col in feature_names]], columns=feature_names)
                else:
                    input_df = pd.DataFrame([[merit_number, percentile, gender_enc, category_enc, seat_enc]])

                # Predict target index
                pred_raw = model.predict(input_df)
                
                # Inverse transform prediction target
                full_prediction = target_encoder.inverse_transform(pred_raw)[0]

                # Split prediction into Institute Name and Course Name
                if " | " in full_prediction:
                    institute, course = full_prediction.split(" | ", 1)
                else:
                    institute = full_prediction
                    course = "General / Unspecified"

                st.balloons()
                st.success("Prediction Generated Successfully!")

                # Render Prediction Cards
                st.markdown(f"""
                <div class="glass-card result-card-inst">
                    <div class="result-header">🎓 Predicted Institute</div>
                    <div class="result-title">{institute}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="glass-card result-card-course">
                    <div class="result-header">📚 Predicted Course</div>
                    <div class="result-title">{course}</div>
                </div>
                """, unsafe_allow_html=True)

                # Input Summary Metrics
                st.markdown("### 📋 Submitted Profile Summary")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("MHTCET Percentile", f"{percentile:.2f}%")
                m_col2.metric("Merit Rank", f"#{merit_number:,}")
                m_col3.metric("Category", category)
                m_col4.metric("Seat Type", seat_alloted)

            except Exception as e:
                st.error("🚨 An error occurred during input processing or prediction.")
                st.exception(e)

    else:
        # Default state before clicking predict button
        st.info("👈 Enter your academic details in the sidebar and click **Predict College**.")
        
        # Optionally display Lottie Animation
        lottie_json = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_qqwqzz9d.json")
        if lottie_json:
            try:
                from streamlit_lottie import st_lottie
                st_lottie(lottie_json, height=260, key="welcome_lottie")
            except ImportError:
                pass

        st.markdown("""
        <div class="glass-card">
            <h3>How to use this tool?</h3>
            <ol>
                <li>Input your <b>MHT-CET Percentile</b> and official <b>Merit Rank</b> in the sidebar.</li>
                <li>Select your designated <b>Gender</b>, <b>Reservation Category</b>, and <b>Quota/Seat Category</b>.</li>
                <li>Click on <b>Predict College</b> to obtain predicted institute and engineering branch.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

with tab_about:
    st.markdown("""
    <div class="glass-card">
        <h3>About MHT-CET College & Course Predictor</h3>
        <p>The <b>AI-Based MHT-CET College & Course Predictor</b> is a Machine Learning application engineered to assist engineering aspirants across Maharashtra in assessing college options.</p>
        <p>By learning from historical Centralized Admission Process (CAP) cutoffs, the model evaluates candidate merit ranks, reservation details, and percentile metrics to output the most probable allotment.</p>
        <h4>Key Features</h4>
        <ul>
            <li><b>Dual Prediction Engine:</b> Predicts targeted college and specific course branch simultaneously.</li>
            <li><b>Encapsulated Processing:</b> Utilizes custom LabelEncoders for inputs and target outputs.</li>
            <li><b>User-Centric UI:</b> Modern glassmorphism dashboard built for speed and clarity.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab_stats:
    st.markdown("### 📊 Model Architecture & System Metrics")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="glass-card">
            <h4>Algorithm Specifications</h4>
            <p><b>Model Type:</b> Scikit-Learn Classifier Pipeline</p>
            <p><b>Target Format:</b> Combined Label (<code>Institute Name | Course Name</code>)</p>
            <p><b>Feature Count:</b> 5 Input Variables</p>
            <p><b>Deployment:</b> Streamlit Cloud Native</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="glass-card">
            <h4>Input Feature Weights & Encoders</h4>
            <p><b>Merit Number:</b> Continuous Numerical Input</p>
            <p><b>MHTCET Percentile:</b> Continuous Floating Point Input</p>
            <p><b>Categorical Encoders:</b> Gender, Category, Seat Allotted</p>
            <p><b>Target Classes:</b> Automatically Inverted via <code>target_encoder.pkl</code></p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    AI-Based MHT-CET College & Course Predictor | Production Ready Deployment
</div>
""", unsafe_allow_html=True)
