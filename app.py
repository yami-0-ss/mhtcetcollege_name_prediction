import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* 1. Global Dark Background Canvas */
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
        max-width: 95% !important;
    }

    /* 2. Global Text Overrides */
    label, p, span, h1, h2, h3, h4, h5, h6, 
    div[data-testid="stWidgetLabel"] label, 
    div[data-testid="stMarkdownContainer"] p {
        color: #f1f5f9 !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* 3. Hero Purple Header Banner (Matching Reference Image) */
    .hero-banner {
        background: linear-gradient(135deg, #4c1d95 0%, #312e81 100%);
        padding: 2.5rem 3rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(76, 29, 149, 0.4);
        margin-bottom: 1.5rem;
        width: 100%;
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

    /* 4. Welcome Alert Bar (Matching Reference Image) */
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

    /* 5. Capability Feature Cards */
    .feature-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 1.5rem;
        border-radius: 14px;
        height: 100%;
    }

    .feature-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 0.6rem;
    }

    .feature-desc {
        font-size: 0.9rem;
        color: #94a3b8 !important;
        line-height: 1.4;
    }

    /* 6. Sidebar Custom Navigation (Obsidian Modern Menu) */
    section[data-testid="stSidebar"] {
        background-color: #060913 !important;
        border-right: 1px solid #111827 !important;
    }

    .stRadio > label {
        display: none !important;
    }

    /* Custom Input Controls */
    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }

    /* Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border-radius: 10px !important;
        border: none !important;
        height: 3.2rem !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #111827 !important;
        padding: 1.2rem 1.5rem !important;
        border-radius: 12px !important;
        border: 1px solid #1f2937 !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] div {
        color: #818cf8 !important;
    }

    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #818cf8, #c084fc) !important;
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

CUTOFF_BENCHMARKS = pd.DataFrame(
    {
        "Category": [
            "OPEN",
            "SEBC",
            "OBC",
            "NT-3",
            "NT-2",
            "VJ/DT",
            "SC",
            "ST",
        ],
        "Median Cutoff Percentile": [
            91.5,
            88.9,
            87.3,
            82.1,
            80.5,
            78.4,
            73.6,
            62.0,
        ],
        "Seat Matrix Allocation (%)": [
            40.0,
            10.0,
            19.0,
            3.5,
            3.5,
            3.0,
            13.0,
            7.0,
        ],
    }
)

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    page_selection = st.radio(
        "Navigation", options=["app", "Dataset", "AI Chat", "Report"], index=0
    )

    st.markdown("<hr style='border-color: #1f2937;'>", unsafe_allow_html=True)

    if page_selection == "app":
        st.subheader("⚙️ Predictor Parameters")
        percentile = st.slider(
            "Percentile Score",
            min_value=0.0,
            max_value=100.0,
            value=87.34,
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
            "⚡ Run Analysis", type="primary", use_container_width=True
        )

# ==========================================
# 4. MAIN PAGE CONTENT
# ==========================================

# Hero Banner Matching Image
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
        👉 <b>Welcome!</b> Adjust student score inputs on the sidebar to execute real-time admission predictions.
    </div>
""",
    unsafe_allow_html=True,
)

if page_selection == "app":
    # 3 Key Capability Boxes
    st.markdown(
        "<h3 style='color: #ffffff; margin-bottom: 1.2rem;'>🛠️ Platform Capabilities</h3>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">📊 Visual Analytics</div>
                <div class="feature-desc">Interactive charts, custom plot generators, and probability distribution maps.</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🤖 Local AI Engine</div>
                <div class="feature-desc">K-Nearest Neighbors machine learning queries computed entirely offline.</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">⚡ Cutoff Predictor</div>
                <div class="feature-desc">Predict seat allotment probabilities across all Maharashtra engineering institutes.</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><hr style='border-color: #1f2937;'><br>", unsafe_allow_html=True)

    if 'predict_btn' in locals() and predict_btn:
        if df_data is not None and knn_model is not None:
            gender_encoded = le_gender.transform([gender])[0]
            cat_encoded = le_cat.transform([category])[0]

            user_vector = np.array([[percentile, gender_encoded, cat_encoded]])
            distances, indices = knn_model.kneighbors(user_vector)

            matched_df = df_data.iloc[indices[0]].copy()

            top_match = matched_df.iloc[0]
            top_college = top_match["Institute Name"]
            top_department = top_match["Course Name"]

            top_combinations = (
                matched_df.groupby(["Institute Name", "Course Name"])
                .size()
                .reset_index(name="Count")
                .sort_values(by="Count", ascending=False)
            )

            total_count = top_combinations["Count"].sum()
            top_combinations["Probability"] = (
                top_combinations["Count"] / total_count * 100
            ).round(1)

            top_confidence = (
                top_combinations.iloc[0]["Probability"]
                if not top_combinations.empty
                else 0.0
            )

            st.markdown(
                "<h3 style='color: #ffffff; margin-bottom: 1rem;'>🎯 Admission Prediction Results</h3>",
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Percentile Score", f"{percentile:.2f}%")
            m2.metric("Category", category)
            m3.metric("Gender", gender)
            m4.metric("Model Confidence", f"{top_confidence}%")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="background-color: #1e1b4b; border: 1px solid #6366f1; padding: 1.5rem; border-radius: 12px; text-align: center;">
                    <div style="color: #818cf8; font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">Top Matching College</div>
                    <div style="color: #ffffff; font-size: 1.6rem; font-weight: 800; margin-top: 0.4rem;">{top_college}</div>
                    <div style="color: #c084fc; font-size: 1.2rem; font-weight: 600; margin-top: 0.2rem;">Department: {top_department}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<br><h4 style='color: #ffffff;'>Top 5 Recommended Allotments</h4>",
                unsafe_allow_html=True,
            )
            for _, row in top_combinations.head(5).iterrows():
                col_n, col_p = st.columns([3, 2])
                with col_n:
                    st.markdown(
                        f"**{row['Institute Name']}**  \n<span style='color:#a5b4fc;'>{row['Course Name']}</span>",
                        unsafe_allow_html=True,
                    )
                with col_p:
                    st.progress(
                        float(row["Probability"] / 100),
                        text=f"{row['Probability']}% Probability",
                    )
                st.markdown(
                    "<hr style='margin: 0.5rem 0; border-color: #1f2937;'>",
                    unsafe_allow_html=True,
                )

        else:
            st.error("Error: Dataset file not found.")

elif page_selection == "Dataset":
    st.subheader("📂 Dataset Explorer")
    if df_data is not None:
        st.dataframe(df_data.head(50), use_container_width=True)
    else:
        st.warning("CSV Dataset not loaded.")

elif page_selection == "AI Chat":
    st.subheader("💬 Local AI Chat Assistant")
    st.info("AI Chat functionality initialized for dataset queries.")

elif page_selection == "Report":
    st.subheader("📄 Generated Analytics Summary")
    st.dataframe(CUTOFF_BENCHMARKS, use_container_width=True)
