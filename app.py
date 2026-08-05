import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="MHT-CET College Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Modern Custom CSS Styling
st.markdown(
    """
    <style>
    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2.5rem 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Card Component */
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.75rem;
    }
    
    /* Prediction Output Box */
    .result-box {
        background: #F0F9FF;
        border-left: 5px solid #0284C7;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-box h4 {
        margin: 0;
        color: #0369A1;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-box p {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0C4A6E;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Primary Action Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">🎓 MHT-CET Admission Predictor</h1>
        <p class="header-subtitle">Analyze historical cutoff data & machine learning models to forecast potential college allocations.</p>
    </div>
""",
    unsafe_allow_html=True,
)


# 3. Model & Data Loaders with Caching
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_data():
    csv_path = (
        "CAP_Seat_Allocation (v2).xlsx - CAP I - Maharashtra (MHTCET) (1).csv"
    )
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


model = load_model()
df = load_data()

# 4. User Interface Layout
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Enter Candidate Metrics")

    percentile = st.number_input(
        "MHT-CET Score / Percentile",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=0.01,
        help="Enter your official MHT-CET examination percentile.",
    )

    gender_list = (
        ["G", "F"]
        if df is None or "Gender" not in df.columns
        else sorted(df["Gender"].dropna().unique())
    )
    gender = st.selectbox("Select Gender", options=gender_list)

    category_list = (
        ["OPEN", "OBC", "SC", "ST", "VJ", "NT1", "NT2", "NT3", "EWS", "TFWS"]
        if df is None or "Category" not in df.columns
        else sorted(df["Category"].dropna().unique())
    )
    category = st.selectbox("Select Category", options=category_list)

    predict_btn = st.button("🚀 Run Admission Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Projected Seat Allocation")

    if predict_btn:
        if model is not None:
            try:
                # Structure input features to match the classifier requirements
                input_frame = pd.DataFrame(
                    {
                        "MHTCET Percentile": [percentile],
                        "Gender": [gender],
                        "Category": [category],
                    }
                )

                # Execute ML Inference
                predicted_institution = model.predict(input_frame)[0]

                st.markdown(
                    f"""
                    <div class="result-box">
                        <h4>Predicted College & Department</h4>
                        <p>{predicted_institution}</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            except Exception as error_msg:
                st.error(f"Inference Failure: {error_msg}")
                st.info(
                    "Note: Ensure input features match the column names and data types expected by your trained scikit-learn model."
                )
        else:
            st.error("Error: `model.pkl` could not be loaded from the project directory.")
    else:
        st.info("Fill out your details on the left and click **Run Admission Prediction**.")

    st.markdown("</div>", unsafe_allow_html=True)

# 5. Historical Data Explorer
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🔍 Cutoff Search Engine")

if df is not None:
    # Look up column dynamically to guard against small dataset header variations
    pct_col = next((c for c in df.columns if "percentile" in c.lower()), None)

    if pct_col:
        filtered_df = df[
            (df[pct_col] <= percentile + 2.0) & (df[pct_col] >= percentile - 5.0)
        ]
        st.caption(
            f"Showing options matching between **{percentile - 5.0:.2f}** and **{percentile + 2.0:.2f}** percentile."
        )
        st.dataframe(filtered_df.head(25), use_container_width=True)
    else:
        st.dataframe(df.head(25), use_container_width=True)
else:
    st.warning("No CSV data source found. Verify dataset file presence in repository root.")
