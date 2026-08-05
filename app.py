import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. PAGE CONFIG & GLASSMORPHIC THEME
# ==========================================
st.set_page_config(
    page_title="MHT-CET Admission Analytics Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Global Background Canvas */
    header[data-testid="stHeader"] {
        background-color: #0d1117 !important;
    }
    
    .stApp {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1000px !important;
        margin: 0 auto !important;
    }

    /* Global Typography Overrides */
    label, p, span, h1, h2, h3, h4, h5, h6, 
    div[data-testid="stWidgetLabel"] label, 
    div[data-testid="stMarkdownContainer"] p {
        color: #e6edf3 !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* Modern Top Header Bar */
    .app-header {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        backdrop-filter: blur(10px);
        padding: 1.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }

    .app-title {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2dd4bf 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .app-subtitle {
        font-size: 0.85rem;
        color: #8b949e !important;
        margin-top: 0.2rem;
    }

    /* Glassmorphic Form Card */
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid #30363d;
        backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
    }

    .card-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #2dd4bf !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* High-Impact Result Card */
    .prediction-card {
        background: linear-gradient(135deg, #0d2b26 0%, #064e3b 100%);
        border: 2px solid #10b981;
        padding: 2.2rem;
        border-radius: 20px;
        box-shadow: 0 12px 35px rgba(16, 185, 129, 0.25);
        margin-top: 1.5rem;
        text-align: center;
    }

    .prediction-badge {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: #6ee7b7 !important;
        font-weight: 800;
        background: rgba(16, 185, 129, 0.2);
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 1rem;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .prediction-college {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff !important;
        line-height: 1.35;
        margin-bottom: 1rem;
    }

    .prediction-dept {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38bdf8 !important;
        background: rgba(15, 23, 42, 0.6);
        padding: 0.8rem 1.6rem;
        border-radius: 12px;
        display: inline-block;
        border: 1px solid #1e293b;
    }

    /* Form Select Controls Styling */
    div[data-baseweb="select"] > div {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }

    /* Neon Emerald Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 12px !important;
        border: none !important;
        height: 3.5rem !important;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
        margin-top: 1rem !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5) !important;
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
# 3. HEADER & FRONT DASHBOARD
# ==========================================

# Modern Top Bar Banner
st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="app-title">🎓 Admission Analytics Studio</div>
            <div class="app-subtitle">KNN Machine Learning Engine • MHT-CET Seat Allocation Matrix</div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# FRONT PAGE PREDICTOR CARD
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="card-header">⚙️ Student Parameters</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="medium")

with col1:
    percentile = st.slider(
        "MHT-CET Percentile Score",
        min_value=0.0,
        max_value=100.0,
        value=48.75,
        step=0.01,
        help="Adjust slider to select student percentile.",
    )

    gender_input = st.radio(
        "Gender Classification",
        options=["Female (F)", "Male (M)"],
        horizontal=True,
    )
    gender = "F" if "Female" in gender_input else "M"

with col2:
    available_categories = (
        sorted(df_data["cat_clean"].unique())
        if df_data is not None
        else ["OPEN", "OBC", "SC", "ST"]
    )
    category = st.selectbox(
        "Reservation Quota Category",
        options=available_categories,
        index=(
            available_categories.index("OBC")
            if "OBC" in available_categories
            else 0
        ),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(
        "⚡ Predict College & Department", type="primary", use_container_width=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 4. PREDICTION INFERENCE & OUTPUT
# ==========================================

if predict_btn:
    if df_data is not None and knn_model is not None:
        with st.spinner("Processing KNN vector space & calculating cutoffs..."):
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
            <div class="prediction-card">
                <div class="prediction-badge">OPTIMAL ALLOTMENT MATCH</div>
                <div class="prediction-college">🏛️ {predicted_college}</div>
                <div class="prediction-dept">🎓 Department: {predicted_department}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    else:
        st.error("Error: CSV Dataset file not found in working directory.")
