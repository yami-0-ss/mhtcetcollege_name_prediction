import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. PAGE CONFIG & INSIGHTAI STUDIO UI THEME
# ==========================================
st.set_page_config(
    page_title="InsightAI Studio - College Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Global Background Canvas */
    header[data-testid="stHeader"] {
        background-color: #0b0f19 !important;
    }
    
    .stApp {
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }

    /* Global Typography Overrides */
    label, p, span, h1, h2, h3, h4, h5, h6, 
    div[data-testid="stWidgetLabel"] label, 
    div[data-testid="stMarkdownContainer"] p {
        color: #f1f5f9 !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* Hero Purple Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #4c1d95 0%, #312e81 100%);
        padding: 2.2rem 2.8rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(76, 29, 149, 0.4);
        margin-bottom: 1.5rem;
        width: 100%;
        text-align: center;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1 !important;
        margin: 0;
    }

    /* Welcome Alert Bar */
    .welcome-bar {
        background-color: #111e38;
        border: 1px solid #1e3a8a;
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        color: #60a5fa !important;
        font-weight: 500;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Predictor Form Container Card */
    .input-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
    }

    .input-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Colored Prediction Result Card */
    .result-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border: 2px solid #10b981;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
        margin-top: 1.5rem;
        text-align: center;
    }

    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #a7f3d0 !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-college {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff !important;
        line-height: 1.3;
        margin-bottom: 0.8rem;
    }

    .result-dept {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38bdf8 !important;
        background-color: #064e3b;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        display: inline-block;
        border: 1px solid #059669;
    }

    /* Custom Input Controls */
    div[data-baseweb="select"] > div {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }

    /* Action Button (Green Glow) */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border-radius: 10px !important;
        border: none !important;
        height: 3.4rem !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
        margin-top: 1rem !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. DATA LOADING & ML MODEL PIPELINE
# ==========================================

DATA_FILE = (
    "CAP_Seat_Allocation (v2).xlsx - CAP I - Maharashtra (MHTCET) (1).csv"
)


def simplify_category(cat):
    cat = str(cat).upper().strip()
    if "OPEN" in cat:
        return "OPEN"
    if "OBC" in cat:
        return "OBC"
    if "SC" in cat:
        return "SC"
    if "ST" in cat:
        return "ST"
    if "NT 1" in cat or "NT-1" in cat or "NT-B" in cat:
        return "NT-1"
    if "NT 2" in cat or "NT-2" in cat or "NT-C" in cat:
        return "NT-2"
    if "NT 3" in cat or "NT-3" in cat or "NT-D" in cat:
        return "NT-3"
    if "DT" in cat or "VJ" in cat:
        return "VJ/DT"
    if "SEBC" in cat:
        return "SEBC"
    if "SBC" in cat:
        return "SBC"
    return "OPEN"


@st.cache_data(ttl=3600)
def load_and_preprocess_data():
    if not os.path.exists(DATA_FILE):
        return None, None, None

    df = pd.read_csv(DATA_FILE)

    for col in ["Institute Name", "Course Name", "Category", "Gender"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["MHTCET Percentile"] = pd.to_numeric(
        df["MHTCET Percentile"], errors="coerce"
    )

    df_clean = df.dropna(
        subset=[
            "MHTCET Percentile",
            "Gender",
            "Category",
            "Institute Name",
            "Course Name",
        ]
    ).copy()

    le_gender = LabelEncoder()
    df_clean["gender_enc"] = le_gender.fit_transform(df_clean["Gender"])

    df_clean["cat_clean"] = df_clean["Category"].apply(simplify_category)
    le_cat = LabelEncoder()
    df_clean["cat_enc"] = le_cat.fit_transform(df_clean["cat_clean"])

    return df_clean, le_gender, le_cat


@st.cache_resource
def build_knn_model(df_clean):
    if df_clean is None or df_clean.empty:
        return None
    X = df_clean[["MHTCET Percentile", "gender_enc", "cat_enc"]]
    knn = NearestNeighbors(n_neighbors=25, algorithm="auto")
    knn.fit(X)
    return knn


df_data, le_gender, le_cat = load_and_preprocess_data()
knn_model = build_knn_model(df_data)

# ==========================================
# 3. MAIN PAGE LAYOUT & FRONT FORM
# ==========================================

# Hero Banner
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">InsightAI Studio</div>
        <div class="hero-subtitle">Keyless Enterprise AI Platform for Interactive Data Analytics & AutoML</div>
    </div>
""",
    unsafe_allow_html=True,
)

# Welcome Notice Bar
st.markdown(
    """
    <div class="welcome-bar">
        👉 <b>Welcome!</b> Configure your score parameters below and click <b>Predict Allocation</b> to generate your prediction.
    </div>
""",
    unsafe_allow_html=True,
)

# FRONT PAGE PREDICTOR PANEL
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="input-header">⚙️ Predictor Parameters</div>',
    unsafe_allow_html=True,
)
st.markdown("<hr style='border-color: #1f2937;'>", unsafe_allow_html=True)

percentile = st.slider(
    "Percentile Score",
    min_value=0.0,
    max_value=100.0,
    value=48.75,
    step=0.01,
)

gender_input = st.radio(
    "Gender", options=["Female (F)", "Male (M)"], horizontal=True
)
gender = "F" if "Female" in gender_input else "M"

available_categories = (
    sorted(df_data["cat_clean"].unique())
    if df_data is not None
    else ["OPEN", "OBC", "SC", "ST"]
)
category = st.selectbox(
    "Reservation Quota",
    options=available_categories,
    index=(
        available_categories.index("OBC")
        if "OBC" in available_categories
        else 0
    ),
)

predict_btn = st.button(
    "⚡ Predict Allocation", type="primary", use_container_width=True
)
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 4. PREDICTION INFERENCE & OUTPUT
# ==========================================

if predict_btn:
    if df_data is not None and knn_model is not None:
        gender_encoded = le_gender.transform([gender])[0]
        cat_encoded = le_cat.transform([category])[0]

        user_vector = np.array([[percentile, gender_encoded, cat_encoded]])
        distances, indices = knn_model.kneighbors(user_vector)

        matched_df = df_data.iloc[indices[0]].copy()

        top_match = matched_df.iloc[0]
        predicted_college = top_match["Institute Name"]
        predicted_department = top_match["Course Name"]

        # DIRECT PREDICTION RESULT CARD
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">OPTIMAL PREDICTED ALLOTMENT</div>
                <div class="result-college">🏛️ {predicted_college}</div>
                <div class="result-dept">🎓 Department: {predicted_department}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    else:
        st.error("Error: CSV Dataset file not found in working directory.")
