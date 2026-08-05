# app.py
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & INITIAL SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MHT-CET College & Course Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS (Glassmorphism, Animations, Theme Styles)
# -----------------------------------------------------------------------------
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #0f172a);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #f8fafc;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.25);
    border: 1px solid rgba(129, 140, 248, 0.3);
}

/* Result Cards Specific Gradient Border */
.result-card-inst {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(147, 51, 234, 0.15));
    border-left: 6px solid #6366f1;
}

.result-card-course {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(59, 130, 246, 0.15));
    border-left: 6px solid #10b981;
}

/* Hero Title Styling */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(to right, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* Custom Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    font-size: 1rem;
    font-weight: 600;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
    box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
    transform: scale(1.02);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. RESOURCE LOADING & VALIDATION
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_artifacts():
    """Loads model and categorical encoders safely."""
    files = {
        "model": "collegename_model.pkl",
        "gender": "gender_encoder.pkl",
        "category": "category_encoder.pkl",
        "seat": "seat_encoder.pkl",
        "target": "target_encoder.pkl",
    }
    
    artifacts = {}
    missing_files = []

    for key, filename in files.items():
        if os.path.exists(filename):
            artifacts[key] = joblib.load(filename)
        else:
            missing_files.append(filename)

    return artifacts, missing_files


artifacts, missing = load_all_artifacts()

if missing:
    st.error(f"❌ Missing required model files: {', '.join(missing)}")
    st.info("Please ensure all `.pkl` files are located in the project root directory.")
    st.stop()

model = artifacts["model"]
gender_encoder = artifacts["gender"]
category_encoder = artifacts["category"]
seat_encoder = artifacts["seat"]
target_encoder = artifacts["target"]


# -----------------------------------------------------------------------------
# 4. SIDEBAR - USER INPUTS & LOGO
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", use_container_width=True)
    else:
        st.markdown("## 🎯 MHT-CET Predictor")

    st.markdown("### 📋 Enter Candidate Details")
    st.caption("Provide exact merit and category details for higher accuracy.")

    merit_number = st.number_input(
        "Merit Number",
        min_value=1,
        max_value=300000,
        value=5000,
        step=1,
        help="Enter your state-level merit rank.",
    )

    percentile = st.number_input(
        "MHTCET Percentile",
        min_value=0.0,
        max_value=100.0,
        value=98.50,
        step=0.01,
        format="%.2f",
        help="Enter overall percentile score.",
    )

    # Extract encoder classes safely
    gender_options = list(gender_encoder.classes_)
    category_options = list(category_encoder.classes_)
    seat_options = list(seat_encoder.classes_)

    gender = st.selectbox("Gender", options=gender_options)
    category = st.selectbox("Category", options=category_options)
    seat_alloted = st.selectbox("Seat Allotted Type", options=seat_options)

    predict_btn = st.button("✨ Predict College & Course")


# -----------------------------------------------------------------------------
# 5. MAIN CONTENT - HERO & DASHBOARD
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">MHT-CET Admission Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Machine Learning-Powered College & Branch Seat Allocation Insights</div>',
    unsafe_allow_html=True,
)

# Top Metrics Banner
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="Model Architecture", value="Random Forest")
with col_m2:
    st.metric(label="Selected Percentile", value=f"{percentile:.2f}%")
with col_m3:
    st.metric(label="Selected Category", value=str(category))
with col_m4:
    st.metric(label="Merit Rank", value=f"#{merit_number}")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. PREDICTION LOGIC & RESULT DISPLAY
# -----------------------------------------------------------------------------
if predict_btn:
    # Input Validation
    if percentile < 0.0 or percentile > 100.0:
        st.error("⚠️ Invalid Percentile entered. Please enter a value between 0.00 and 100.00.")
    elif merit_number <= 0:
        st.error("⚠️ Merit Number must be greater than zero.")
    else:
        try:
            with st.spinner("🔍 Analyzing cutoffs and matching institutes..."):
                # Encode Categorical Variables
                gender_enc = gender_encoder.transform([gender])[0]
                category_enc = category_encoder.transform([category])[0]
                seat_enc = seat_encoder.transform([seat_alloted])[0]

                # Prepare feature array matching model's expected features
                # Order: [Merit Number, MHTCET Percentile, Gender, Category, Seat Alloted]
                features = np.array([[merit_number, percentile, gender_enc, category_enc, seat_enc]])

                # Predict
                pred_encoded = model.predict(features)
                prediction_str = target_encoder.inverse_transform(pred_encoded)[0]

                # Split target string
                if " | " in prediction_str:
                    institute, course = prediction_str.split(" | ", 1)
                else:
                    institute = prediction_str
                    course = "General / Unspecified Branch"

            # Display Results Cards
            st.markdown("### 🎯 Predicted Allocation")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(
                    f"""
                    <div class="glass-card result-card-inst">
                        <p style="color: #818cf8; font-weight: 600; font-size: 0.9rem; margin-bottom: 4px;">🎓 PREDICTED INSTITUTE</p>
                        <h3 style="color: #ffffff; margin: 0; font-size: 1.3rem;">{institute}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown(
                    f"""
                    <div class="glass-card result-card-course">
                        <p style="color: #34d399; font-weight: 600; font-size: 0.9rem; margin-bottom: 4px;">📚 PREDICTED COURSE / BRANCH</p>
                        <h3 style="color: #ffffff; margin: 0; font-size: 1.3rem;">{course}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.success("✅ Seat allocation prediction generated successfully!")

        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {str(e)}")
            st.caption("Please verify that categorical values match the trained encoder labels.")

# -----------------------------------------------------------------------------
# 7. ABOUT & FEATURE HIGHLIGHTS SECTION
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 💡 Platform Capabilities")

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown(
        """
        <div class="glass-card">
            <h4>⚡ Machine Learning</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Powered by an ensemble <b>RandomForestClassifier</b> trained on past MHT-CET CAP round allotment datasets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feat_col2:
    st.markdown(
        """
        <div class="glass-card">
            <h4>🎯 Multi-Feature Matching</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Evaluates complex relationships across Percentiles, Merit Ranks, Gender, Categories, and Seat Allotment Types.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feat_col3:
    st.markdown(
        """
        <div class="glass-card">
            <h4>🛡️ Robust Validation</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Built-in input validation and safe label encoding prevent system runtime crashes during deployment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Expander for additional project details
with st.expander("ℹ️ About the Project & Model Architecture"):
    st.write(
        """
        This intelligent predictor analyzes historical CAP round seat allotment data for engineering colleges under MHT-CET. 
        By framing college name and branch as a combined target variable (`Institute | Course`), the multi-class model 
        provides holistic allocations based on candidate merit profiles.
        
        * **Framework:** Scikit-Learn Ensemble Learning
        * **Model Type:** Random Forest Classifier
        * **Target:** Combined Target Vector (`Institute Name | Course Name`)
        """
    )

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #64748b; font-size: 0.85rem;">MHT-CET College Predictor Dashboard | Designed for Streamlit Cloud Deployment</p>',
    unsafe_allow_html=True,
)
