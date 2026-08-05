import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="MHT-CET College & Department Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        padding: 0.6rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    .result-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">🎓 MHT-CET Admission Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict potential college & department allocations based on MHT-CET cutoff trends</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_data():
    csv_path = "CAP_Seat_Allocation (v2).xlsx - CAP I - Maharashtra (MHTCET) (1).csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

model = load_model()
df = load_data()

# Layout Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Input Details")
    
    percentile = st.number_input(
        "MHTCET Percentile",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=0.01,
        help="Enter your overall MHT-CET percentile score."
    )
    
    gender_options = ["G", "F"] if df is None or "Gender" not in df.columns else sorted(df["Gender"].dropna().unique())
    gender = st.selectbox("Gender", options=gender_options)
    
    category_options = ["OPEN", "OBC", "SC", "ST", "VJ", "NT1", "NT2", "NT3", "EWS", "TFWS"] if df is None or "Category" not in df.columns else sorted(df["Category"].dropna().unique())
    category = st.selectbox("Category", options=category_options)
    
    predict_btn = st.button("Predict Allocation")

with col2:
    st.subheader("🎯 Prediction Results")
    
    if predict_btn:
        if model is not None:
            try:
                # Prepare input array/DataFrame matching model feature names
                input_data = pd.DataFrame({
                    'MHTCET Percentile': [percentile],
                    'Gender': [gender],
                    'Category': [category]
                })
                
                prediction = model.predict(input_data)[0]
                
                st.markdown(f"""
                    <div class="result-card">
                        <h4 style="margin:0; color:#1E3A8A;">Top Predicted Institution</h4>
                        <p style="font-size: 1.2rem; font-weight: 600; margin-top: 0.5rem; color: #111827;">
                            {prediction}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                # Fallback if categorical encoding was not directly handled by pipeline
                st.error(f"Error during prediction: {e}")
                st.info("Tip: Ensure model preprocessing matches input data formats (e.g. LabelEncoder/OneHotEncoder).")
        else:
            st.warning("`model.pkl` file not found in the directory.")

    # Historical Data Filter Section
    st.markdown("---")
    st.subheader("🔍 Historical Cutoff Search")
    if df is not None:
        filtered_df = df[
            (df['MHTCET Percentile'] <= percentile + 2.0) & 
            (df['MHTCET Percentile'] >= percentile - 5.0)
        ] if 'MHTCET Percentile' in df.columns else df
        
        st.write(f"Displaying top matching records near **{percentile} percentile**:")
        st.dataframe(filtered_df.head(10), use_container_width=True)
    else:
        st.info("Upload the dataset CSV to enable raw cutoff browsing.")
