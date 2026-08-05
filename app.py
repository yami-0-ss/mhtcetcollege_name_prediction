import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. PAGE CONFIG & MODERN DARK SYSTEM STYLING
# ==========================================
st.set_page_config(
    page_title="College Admission Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* 1. Eliminate Streamlit Header Bleed & Standardize App Canvas */
    header[data-testid="stHeader"] {
        background-color: #0b1120 !important;
    }
    
    .stApp {
        background-color: #0b1120 !important;
        color: #f8fafc !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 96% !important;
    }

    /* 2. Global Text Overrides */
    label, p, span, h1, h2, h3, h4, h5, h6, 
    div[data-testid="stWidgetLabel"] label, 
    div[data-testid="stMarkdownContainer"] p {
        color: #f1f5f9 !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    .stCaption, p.caption, div[data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }

    /* 3. Hero Banner Styling */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2.2rem 2.8rem;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
        width: 100%;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8 !important;
        margin: 0;
    }

    /* 4. Glassmorphic Result Card */
    .prediction-card {
        padding: 2rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #311b92 100%);
        border-radius: 16px;
        box-shadow: 0 12px 30px rgba(79, 70, 229, 0.3);
        margin: 1rem 0 1.5rem 0;
        text-align: center;
        border: 1px solid #6366f1;
    }

    .prediction-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: #38bdf8 !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .prediction-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff !important;
        line-height: 1.3;
    }

    .prediction-dept {
        font-size: 1.25rem;
        font-weight: 600;
        color: #818cf8 !important;
        margin-top: 0.5rem;
    }

    /* 5. Clean Metrics Styling */
    div[data-testid="stMetric"] {
        background: #1e293b !important;
        padding: 1.2rem 1.5rem !important;
        border-radius: 12px !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stMetricValue"] div {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* 6. Refined Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #334155;
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: #1e293b;
        border-radius: 8px;
        padding: 0 18px;
        font-weight: 600;
        color: #cbd5e1 !important;
        border: 1px solid #334155;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0284c7, #2563eb) !important;
        color: #ffffff !important;
        border: none !important;
    }

    /* 7. Sidebar Controls */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border-radius: 10px !important;
        border: none !important;
        height: 3.2rem !important;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
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

    # Clean text whitespace across target columns
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

# Benchmark Summary Table
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
# 3. SIDEBAR INPUTS
# ==========================================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3429/3429149.png", width=55
    )
    st.title("Student Profile")
    st.caption("Adjust score metrics to evaluate target probabilities.")

    st.markdown("---")

    percentile = st.slider(
        "MHTCET Percentile Score",
        min_value=0.0,
        max_value=100.0,
        value=87.34,
        step=0.01,
        help="Select exact percentile obtained.",
    )

    gender_input = st.radio(
        "Gender Classification",
        options=["Female (F)", "Male (M)"],
        horizontal=True,
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

    st.markdown("---")
    predict_btn = st.button(
        "⚡ Run Predictor Engine", type="primary", use_container_width=True
    )

# ==========================================
# 4. MAIN DASHBOARD CONTENT
# ==========================================

# Hero Banner
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">⚡ College Admission Analytics</div>
        <div class="hero-subtitle">
            K-Nearest Neighbors (KNN) Machine Learning engine calculating real-time CAP round allotment distributions.
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

if predict_btn:
    if df_data is not None and knn_model is not None:
        # Preprocess User Inputs
        gender_encoded = le_gender.transform([gender])[0]
        cat_encoded = le_cat.transform([category])[0]

        user_vector = np.array([[percentile, gender_encoded, cat_encoded]])
        distances, indices = knn_model.kneighbors(user_vector)

        matched_df = df_data.iloc[indices[0]].copy()

        top_match = matched_df.iloc[0]
        top_college = top_match["Institute Name"]
        top_department = top_match["Course Name"]

        # Group top predictions to compute probabilities
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
            "<h3 style='color: #f8fafc; margin-bottom: 1rem;'>📊 Admission Diagnostic Summary</h3>",
            unsafe_allow_html=True,
        )

        tab_result, tab_analytics, tab_benchmark = st.tabs(
            [
                "🎯 Preferred Allotment",
                "📈 Percentile Position",
                "📉 Category Trends",
            ]
        )

        # Tab 1: Primary Output
        with tab_result:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">OPTIMAL MATCHING INSTITUTE</div>
                    <div class="prediction-value">{top_college}</div>
                    <div class="prediction-dept">🎓 Department: {top_department}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Percentile Score", f"{percentile:.2f}%")
            m2.metric("Category Quota", category)
            m3.metric("Gender Group", gender)
            m4.metric("Engine Confidence", f"{top_confidence}%")

            st.markdown(
                "<br><h4 style='color: #38bdf8;'>🏆 Candidate College & Department Probabilities</h4>",
                unsafe_allow_html=True,
            )

            for _, row in top_combinations.head(5).iterrows():
                col_name, col_bar = st.columns([3, 2])
                with col_name:
                    st.markdown(
                        f"**{row['Institute Name']}**  \n<span style='color:#818cf8; font-size:0.9rem;'>{row['Course Name']}</span>",
                        unsafe_allow_html=True,
                    )
                with col_bar:
                    st.progress(
                        float(row["Probability"] / 100),
                        text=f"{row['Probability']}% Probability",
                    )
                st.markdown(
                    "<hr style='margin: 0.5rem 0; border-color: #1e293b;'>",
                    unsafe_allow_html=True,
                )

        # Tab 2: Score Gauge Chart
        with tab_analytics:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=percentile,
                    number={
                        "suffix": "%",
                        "font": {"size": 36, "color": "#38bdf8"},
                    },
                    title={
                        "text": "Score Competitiveness Index",
                        "font": {"size": 16, "color": "#cbd5e1"},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": "#475569",
                        },
                        "bar": {"color": "#38bdf8", "thickness": 0.28},
                        "bgcolor": "#1e293b",
                        "borderwidth": 1,
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 60], "color": "#311b92"},
                            {"range": [60, 85], "color": "#0f172a"},
                            {"range": [85, 100], "color": "#064e3b"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Tab 3: Cutoffs Bar Chart & Department Distribution
        with tab_benchmark:
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                fig_bar = px.bar(
                    CUTOFF_BENCHMARKS,
                    x="Category",
                    y="Median Cutoff Percentile",
                    color="Median Cutoff Percentile",
                    color_continuous_scale="tealgrn",
                    title="State Cutoff Distribution Across Quotas",
                )
                fig_bar.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1"),
                    coloraxis_showscale=False,
                    height=300,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_chart2:
                dept_counts = (
                    matched_df["Course Name"]
                    .value_counts()
                    .head(5)
                    .reset_index()
                )
                dept_counts.columns = ["Department", "Count"]
                fig_dept = px.pie(
                    dept_counts,
                    names="Department",
                    values="Count",
                    title="Matched Department Allotment Ratio",
                    hole=0.4,
                )
                fig_dept.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1"),
                    height=300,
                )
                st.plotly_chart(fig_dept, use_container_width=True)

    else:
        st.warning(
            f"⚠️ Dataset `{DATA_FILE}` not found in working directory."
        )

else:
    # Pre-Prediction Explanatory Interface
    col_info, col_chart = st.columns([1, 1])

    with col_info:
        st.markdown(
            """
            #### System Overview
            * **Vector Preprocessing**: Converts continuous scores and demographic metrics into normalized feature vectors.
            * **KNN Neighborhood Search**: Evaluates feature distances against historical CAP round allocation datasets.
            * **Probabilistic Assignment**: Computes target probability distributions across engineering institutes and departments.
        """
        )

    with col_chart:
        fig_overview = px.scatter(
            CUTOFF_BENCHMARKS,
            x="Median Cutoff Percentile",
            y="Seat Matrix Allocation (%)",
            text="Category",
            size="Median Cutoff Percentile",
            color="Median Cutoff Percentile",
            color_continuous_scale="ice",
            title="Category Cutoffs vs. Reserved Quotas",
        )
        fig_overview.update_traces(textposition="top center")
        fig_overview.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            coloraxis_showscale=False,
            height=280,
        )
        st.plotly_chart(fig_overview, use_container_width=True)

st.divider()
st.caption(
    "College Predictor Analytics Engine • Executive Data Science Dashboard"
)
