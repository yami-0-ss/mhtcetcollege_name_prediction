import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MHT-CET College & Course Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS (Glassmorphism & Theme Styling)
# -----------------------------------------------------------------------------
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3);
    }

    /* Animated Result Cards */
    .result-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.02));
        border-left: 5px solid #FF4B4B;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        animation: fadeIn 0.8s ease-in-out;
    }

    .result-card-course {
        border-left-color: #1E88E5;
    }

    .result-header {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
        margin-bottom: 5px;
    }

    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
    }

    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(45deg, #FF4B4B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Animation Keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Custom Button */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
        font-weight: 600;
        background: linear-gradient(45deg, #FF4B4B, #FF7B54);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(45deg, #FF7B54, #FF4B4B);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5);
        transform: translateY(-2px);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. RESOURCE LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("collegename_model.pkl")
        gender_enc = joblib.load("gender_encoder.pkl")
        category_enc = joblib.load("category_encoder.pkl")
        seat_enc = joblib.load("seat_encoder.pkl")
        target_enc = joblib.load("target_encoder.pkl")
        return model, gender_enc, category_enc, seat_enc, target_enc
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models or encoders: {str(e)}")
        st.stop()

model, gender_encoder, category_encoder, seat_encoder, target_encoder = load_assets()

# -----------------------------------------------------------------------------
# 4. SIDEBAR INPUT FORM
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", use_container_width=True)
    else:
        st.title("🎓 MHT-CET Portal")

    st.header("📋 Candidate Profile")
    
    merit_no = st.number_input("Merit Number", min_value=1, value=1000, step=1)
    percentile = st.number_input("MHTCET Percentile", min_value=0.0, max_value=100.0, value=95.0, step=0.01)
    
    gender_options = list(gender_encoder.classes_)
    gender = st.selectbox("Gender", options=gender_options)
    
    category_options = list(category_encoder.classes_)
    category = st.selectbox("Category", options=category_options)
    
    seat_options = list(seat_encoder.classes_)
    seat_alloted = st.selectbox("Seat Allotted", options=seat_options)
    
    predict_btn = st.button("🔮 Predict College")

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown("<h1 class='gradient-text'>MHT-CET College & Course Predictor</h1>", unsafe_allow_html=True)
st.caption("Leverage Machine Learning to estimate your optimal admission outcome based on previous trends.")

# Hero & Overview
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Model Architecture", value="Random Forest")
with col2:
    st.metric(label="Percentile Input", value=f"{percentile:.2f} %ile")
with col3:
    st.metric(label="Merit Rank Input", value=f"#{merit_no}")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. INFERENCE & RESULTS
# -----------------------------------------------------------------------------
if predict_btn:
    # Input Validation
    if percentile <= 0 or percentile > 100:
        st.error("Please enter a valid MHTCET percentile between 0 and 100.")
    elif merit_no <= 0:
        st.error("Please enter a valid positive Merit Number.")
    else:
        with st.spinner("Analyzing historical data and running prediction..."):
            try:
                # Encoding categorical features
                gender_val = gender_encoder.transform([gender])[0]
                category_val = category_encoder.transform([category])[0]
                seat_val = seat_encoder.transform([seat_alloted])[0]

                # Model Expects 4 features matching feature_names_in_:
                # ["MHTCET Percentile", "Gender", "Category", "Seat Alloted"]
                input_data = pd.DataFrame([{
                    "MHTCET Percentile": percentile,
                    "Gender": gender_val,
                    "Category": category_val,
                    "Seat Alloted": seat_val
                }])

                # Prediction
                raw_pred = model.predict(input_data)
                prediction_str = target_encoder.inverse_transform(raw_pred)[0]

                # Parsing Target string
                if " | " in prediction_str:
                    institute, course = prediction_str.split(" | ", 1)
                else:
                    institute = prediction_str
                    course = "General / Unspecified"

                # Display Results
                st.subheader("🎉 Prediction Results")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-header">🎓 Predicted Institute</div>
                            <div class="result-title">{institute}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with res_col2:
                    st.markdown(f"""
                        <div class="result-card result-card-course">
                            <div class="result-header">📚 Predicted Course</div>
                            <div class="result-title">{course}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                st.success("Prediction generated successfully!")

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

# -----------------------------------------------------------------------------
# 7. ABOUT & FEATURE CARDS
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️ About the Model & System", expanded=False):
    st.write("""
        This machine learning system utilizes an ensemble **Random Forest Classifier** trained on MHT-CET admission data. 
        It evaluates historical allotment cutoffs across various candidate parameters (Percentile, Category, Allotment Type, and Gender) 
        to project the most likely engineering institute and program allocation.
    """)

st.markdown("---")
st.caption("MHT-CET Admission Predictor • Streamlit Deployment")
