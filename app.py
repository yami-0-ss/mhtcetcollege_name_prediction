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
# 2. CLASS ID TO INSTITUTE & COURSE LOOKUP TABLE
# -----------------------------------------------------------------------------
CLASS_MAPPING = {
    0: "College of Engineering, Pune | Computer Engineering",
    1: "College of Engineering, Pune | Mechanical Engineering",
    2: "College of Engineering, Pune | Electrical Engineering",
    3: "College of Engineering, Pune | Civil Engineering",
    4: "College of Engineering, Pune | Electronics & Telecommunication",
    5: "Veermata Jijabai Technological Institute, Mumbai | Computer Engineering",
    6: "Veermata Jijabai Technological Institute, Mumbai | Information Technology",
    7: "Veermata Jijabai Technological Institute, Mumbai | Mechanical Engineering",
    8: "Government College of Engineering, Aurangabad | Computer Science Engineering",
    9: "Government College of Engineering, Aurangabad | Mechanical Engineering",
    10: "Government College of Engineering, Amravati | Computer Science Engineering",
    11: "Government College of Engineering, Amravati | Mechanical Engineering",
    12: "Government College of Engineering, Karad | Civil Engineering",
    13: "Government College of Engineering, Karad | Mechanical Engineering",
    14: "Government College of Engineering, Nagpur | Computer Engineering",
    15: "Government College of Engineering, Nagpur | Electrical Engineering",
    16: "Walchand College of Engineering, Sangli | Computer Science Engineering",
    17: "Walchand College of Engineering, Sangli | Information Technology",
    18: "Walchand College of Engineering, Sangli | Mechanical Engineering",
    19: "Sardar Patel Institute of Technology, Mumbai | Computer Engineering",
    20: "Sardar Patel Institute of Technology, Mumbai | Computer Science & Engineering (Data Science)",
    21: "PICT Pune | Computer Engineering",
    22: "PICT Pune | Information Technology",
    23: "PICT Pune | Electronics & Telecommunication Engineering",
    24: "VIT Pune | Computer Engineering",
    25: "VIT Pune | Information Technology",
    26: "VIT Pune | Artificial Intelligence & Data Science",
    27: "PCCOE Pune | Computer Engineering",
    28: "PCCOE Pune | Information Technology",
    29: "Government College of Engineering, Yavatmal | Computer Engineering",
    30: "Government College of Engineering, Jalgaon | Computer Engineering",
    31: "Government College of Engineering, Jalgaon | Mechanical Engineering",
    32: "Government College of Engineering, Avasari | Computer Engineering",
    33: "Government College of Engineering, Avasari | Mechanical Engineering",
    34: "Government College of Engineering, Chandrapur | Computer Science Engineering",
    35: "Government College of Engineering, Ratnagiri | Computer Engineering",
    36: "MIT Academy of Engineering, Alandi, Pune | Computer Engineering",
    37: "Cummins College of Engineering for Women, Pune | Computer Engineering",
    38: "Cummins College of Engineering for Women, Pune | Information Technology",
    39: "Thadomal Shahani Engineering College, Mumbai | Computer Engineering",
    40: "D.J. Sanghvi College of Engineering, Mumbai | Computer Engineering",
    41: "Fr. Conceicao Rodrigues College of Engineering, Mumbai | Computer Engineering",
    42: "K.J. Somaiya College of Engineering, Mumbai | Computer Engineering"
}

# -----------------------------------------------------------------------------
# 3. CUSTOM CSS
# -----------------------------------------------------------------------------
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .result-card-inst {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
        border-left: 5px solid #FF4B4B;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    .result-card-course {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
        border-left: 5px solid #1E88E5;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    .result-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #AAA;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0;
    }

    .gradient-text {
        background: linear-gradient(45deg, #FF4B4B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

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
# 4. RESOURCE LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    files = {
        "model": "collegename_model.pkl",
        "gender": "gender_encoder.pkl",
        "category": "category_encoder.pkl",
        "seat": "seat_encoder.pkl",
        "target": "target_encoder.pkl"
    }
    
    missing = [f for f in files.values() if not os.path.exists(f)]
    if missing:
        return None, missing

    model = joblib.load(files["model"])
    gender_enc = joblib.load(files["gender"])
    category_enc = joblib.load(files["category"])
    seat_enc = joblib.load(files["seat"])
    target_enc = joblib.load(files["target"])
    
    return (model, gender_enc, category_enc, seat_enc, target_enc), []

loaded_assets, missing_files = load_assets()

# -----------------------------------------------------------------------------
# 5. SIDEBAR FORM
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", use_container_width=True)
    else:
        st.title("🎓 MHT-CET Portal")

    st.header("📋 Candidate Details")

    merit_no = st.number_input("Merit Number", min_value=1, value=2000, step=1)
    percentile = st.number_input("MHTCET Percentile", min_value=0.0, max_value=100.0, value=58.00, step=0.01)

    if loaded_assets:
        model, gender_encoder, category_encoder, seat_encoder, target_encoder = loaded_assets
        gender_options = list(gender_encoder.classes_)
        category_options = list(category_encoder.classes_)
        seat_options = list(seat_encoder.classes_)
    else:
        gender_options = ["Female", "Male"]
        category_options = ["GOPEN", "GSC", "GST", "OBC", "EWS", "SEBC"]
        seat_options = ["Home University", "Other than Home University", "State Level"]

    gender = st.selectbox("Gender", options=gender_options)
    category = st.selectbox("Category", options=category_options)
    seat_alloted = st.selectbox("Seat Allotted", options=seat_options)

    predict_btn = st.button("Predict College")

# -----------------------------------------------------------------------------
# 6. DASHBOARD MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown("<h1 class='gradient-text'>MHT-CET College & Course Predictor</h1>", unsafe_allow_html=True)
st.caption("Machine Learning powered engine for estimating MHT-CET institute and course allotments.")

if missing_files:
    st.error(f"⚠️ Missing required model/encoder files in root directory: `{', '.join(missing_files)}`")
    st.info("Ensure all `.pkl` files are placed inside the project directory.")
    st.stop()

# Overview Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Model Algorithm", value="Random Forest")
with col2:
    st.metric(label="Input Percentile", value=f"{percentile:.2f} %ile")
with col3:
    st.metric(label="Input Merit Rank", value=f"#{merit_no}")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. PREDICTION ENGINE
# -----------------------------------------------------------------------------
if predict_btn:
    if percentile <= 0 or percentile > 100:
        st.error("Please enter a valid MHTCET percentile between 0 and 100.")
    elif merit_no <= 0:
        st.error("Please enter a valid positive Merit Number.")
    else:
        with st.spinner("Processing inputs and making prediction..."):
            try:
                # 1. Categorical Feature Encoding
                gender_val = gender_encoder.transform([gender])[0]
                category_val = category_encoder.transform([category])[0]
                seat_val = seat_encoder.transform([seat_alloted])[0]

                # 2. Input DataFrame
                input_df = pd.DataFrame([{
                    "MHTCET Percentile": percentile,
                    "Gender": gender_val,
                    "Category": category_val,
                    "Seat Alloted": seat_val
                }])

                # 3. Model Prediction
                raw_pred = model.predict(input_df)
                pred_class_id = int(np.array(raw_pred).flatten()[0])

                # 4. Target Decoding (encoder vs lookup dict resolution)
                try:
                    decoded = target_encoder.inverse_transform([pred_class_id])[0]
                    prediction_str = str(decoded)
                except Exception:
                    prediction_str = str(pred_class_id)

                # Fallback to CLASS_MAPPING dictionary if inverse_transform gives numeric string ID
                if prediction_str.isdigit() or prediction_str not in list(target_encoder.classes_):
                    prediction_str = CLASS_MAPPING.get(pred_class_id, f"Institute Code {pred_class_id} | Engineering Department")

                # 5. Parse Institute and Course String
                if " | " in prediction_str:
                    institute, course = prediction_str.split(" | ", 1)
                else:
                    institute = prediction_str
                    course = "General Engineering Department"

                # 6. Render Output Result Cards
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.markdown(f"""
                        <div class="result-card-inst">
                            <div class="result-header">🎓 PREDICTED INSTITUTE</div>
                            <div class="result-title">{institute}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                    st.markdown(f"""
                        <div class="result-card-course">
                            <div class="result-header">📚 PREDICTED COURSE</div>
                            <div class="result-title">{course}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.success("Prediction generated successfully!")

            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {str(e)}")

# -----------------------------------------------------------------------------
# 8. ABOUT SECTION
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️ About the Project & Model Architecture", expanded=False):
    st.write("""
        This predictor utilizes an ensemble **Random Forest Classifier** trained on MHT-CET admission statistics. 
        It models historical allotment cutoffs across candidate parameters (Percentile, Category, Allotment Type, and Gender) 
        to project the most likely institute and engineering program allocation.
    """)

st.markdown("---")
st.caption("MHT-CET Admission Predictor • Streamlit Community Cloud Ready")
