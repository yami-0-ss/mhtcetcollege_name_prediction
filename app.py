import os
import joblib
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="MHT-CET College & Course Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load Pre-trained Artifacts
@st.cache_resource
def load_all_artifacts():
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
    st.info("Ensure all .pkl files are committed to your GitHub repository.")
    st.stop()

model = artifacts["model"]
gender_encoder = artifacts["gender"]
category_encoder = artifacts["category"]
seat_encoder = artifacts["seat"]
target_encoder = artifacts["target"]

# Sidebar Inputs
st.sidebar.title("📋 Candidate Details")
merit_number = st.sidebar.number_input(
    "Merit Number", min_value=1, max_value=300000, value=5000
)
percentile = st.sidebar.number_input(
    "MHTCET Percentile", min_value=0.0, max_value=100.0, value=98.50, format="%.2f"
)

gender = st.sidebar.selectbox("Gender", options=list(gender_encoder.classes_))
category = st.sidebar.selectbox(
    "Category", options=list(category_encoder.classes_)
)
seat_alloted = st.sidebar.selectbox(
    "Seat Allotted Type", options=list(seat_encoder.classes_)
)

# Prediction Logic
if st.sidebar.button("Predict College"):
    try:
        gender_enc = gender_encoder.transform([gender])[0]
        category_enc = category_encoder.transform([category])[0]
        seat_enc = seat_encoder.transform([seat_alloted])[0]

        features = np.array(
            [[percentile, gender_enc, category_enc, seat_enc]]
        )
        pred = model.predict(features)
        prediction = target_encoder.inverse_transform(pred)[0]

        if " | " in prediction:
            institute, course = prediction.split(" | ", 1)
        else:
            institute, course = prediction, "General"

        st.markdown("### 🎓 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Predicted Institute:**\n\n{institute}")
        with col2:
            st.info(f"**Predicted Course:**\n\n{course}")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
