import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Load dataset
df = pd.read_csv("CAP_Seat_Allocation_short.csv")

# 2. Combine Institute and Course names to create the Target column
df["Target"] = df["Institute Name"] + " | " + df["Course Name"]

# 3. Fit LabelEncoders on categorical columns
gender_encoder = LabelEncoder().fit(df["Gender"])
category_encoder = LabelEncoder().fit(df["Category"])
seat_encoder = LabelEncoder().fit(df["Seat Alloted"])
target_encoder = LabelEncoder().fit(df["Target"])

# 4. Save the fitted encoders to pickle files
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(category_encoder, "category_encoder.pkl")
joblib.dump(seat_encoder, "seat_encoder.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print(
    "✅ Success: gender_encoder.pkl, category_encoder.pkl, seat_encoder.pkl, and target_encoder.pkl created!"
)
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
# 2. CUSTOM CSS (Glassmorphism & Custom Styling)
# -----------------------------------------------------------------------------
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Animated Gradient Background */
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

/* Glassmorphism Styling */
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

/* Result Cards Specific Gradient Borders */
.result-card-inst {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(147, 51, 234, 0.15));
    border-left: 6px solid #6366f1;
}

.result-card-course {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(59, 130, 246, 0.15));
    border-left: 6px solid #10b981;
}

/* Title & Subtitle Styling */
.hero-title {
    font-size: 2.5rem;
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

/* Custom Primary Button Styling */
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

/* Sidebar Background Adjustment */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

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
    st.info(
        "Please place all required `.pkl` files into the root project directory and restart."
    )
    st.stop()

model = artifacts["model"]
gender_encoder = artifacts["gender"]
category_encoder = artifacts["category"]
seat_encoder = artifacts["seat"]
target_encoder = artifacts["target"]


# -----------------------------------------------------------------------------
# 4. SIDEBAR - INPUT CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", use_container_width=True)
    else:
        st.markdown("## 🎯 MHT-CET Predictor")

    st.markdown("### 📋 Student Credentials")
    st.caption("Provide exact candidate merit details for precise allocation.")

    merit_number = st.number_input(
        "Merit Number",
        min_value=1,
        max_value=300000,
        value=5000,
        step=1,
        help="Enter your state rank from your MHT-CET score report.",
    )

    percentile = st.number_input(
        "MHTCET Percentile",
        min_value=0.0,
        max_value=100.0,
        value=98.50,
        step=0.01,
        format="%.2f",
        help="Enter overall MHT-CET percentile score.",
    )

    # Load encoder option classes dynamically
    gender_options = list(gender_encoder.classes_)
    category_options = list(category_encoder.classes_)
    seat_options = list(seat_encoder.classes_)

    gender = st.selectbox("Gender", options=gender_options)
    category = st.selectbox("Category", options=category_options)
    seat_alloted = st.selectbox("Seat Allotted Type", options=seat_options)

    predict_btn = st.button("Predict College")


# -----------------------------------------------------------------------------
# 5. DASHBOARD MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="hero-title">AI-Based MHT-CET College & Course Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">Machine Learning Powered Engineering Seat Allotment Analytics</div>',
    unsafe_allow_html=True,
)

# Metric Summary Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Algorithm", value="Random Forest")
with col2:
    st.metric(label="Selected Percentile", value=f"{percentile:.2f}%")
with col3:
    st.metric(label="Category", value=str(category))
with col4:
    st.metric(label="Merit Rank", value=f"#{merit_number}")

st.markdown("---")


# -----------------------------------------------------------------------------
# 6. PREDICTION INFERENCE & RESULT DISPLAY
# -----------------------------------------------------------------------------
if predict_btn:
    # Validate inputs
    if percentile < 0.0 or percentile > 100.0:
        st.error("⚠️ Invalid percentile score! Enter a value between 0.0 and 100.0.")
    elif merit_number <= 0:
        st.error("⚠️ Merit number must be greater than zero.")
    else:
        try:
            with st.spinner("🔍 Matching cutoffs and predicting college..."):
                # Encode input categorical values
                gender_encoded = gender_encoder.transform([gender])[0]
                category_encoded = category_encoder.transform([category])[0]
                seat_encoded = seat_encoder.transform([seat_alloted])[0]

                # Features shape: [MHTCET Percentile, Gender, Category, Seat Alloted]
                input_features = np.array(
                    [[percentile, gender_encoded, category_encoded, seat_encoded]]
                )

                # Make prediction and inverse transform label
                pred = model.predict(input_features)
                prediction = target_encoder.inverse_transform(pred)[0]

                # Separate Institute Name and Course Name
                if " | " in prediction:
                    institute, course = prediction.split(" | ", 1)
                else:
                    institute = prediction
                    course = "Unspecified Branch"

            # Display prediction result cards
            st.markdown("### 🎓 Allocation Result")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown(
                    f"""
                    <div class="glass-card result-card-inst">
                        <p style="color: #818cf8; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">🎓 PREDICTED INSTITUTE</p>
                        <h3 style="color: #ffffff; margin: 0; font-size: 1.25rem;">{institute}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown(
                    f"""
                    <div class="glass-card result-card-course">
                        <p style="color: #34d399; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">📚 PREDICTED COURSE</p>
                        <h3 style="color: #ffffff; margin: 0; font-size: 1.25rem;">{course}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.success("✅ Prediction calculated successfully!")

        except Exception as e:
            st.error(f"⚠️ Error performing prediction: {str(e)}")
            st.caption(
                "Ensure that selected values match the trained encoder classes."
            )

# -----------------------------------------------------------------------------
# 7. FEATURE HIGHLIGHTS & ABOUT SECTION
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 💡 Model & Platform Overview")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown(
        """
        <div class="glass-card">
            <h4>⚡ Ensemble Model</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Trained on official CAP round historical seat allotment data using a multi-class RandomForestClassifier.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with f2:
    st.markdown(
        """
        <div class="glass-card">
            <h4>🎯 Multi-Category Support</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Considers seat allotment categories (OPEN, OBC, SC, ST, EWS, TFWS) and gender reservation criteria.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with f3:
    st.markdown(
        """
        <div class="glass-card">
            <h4>🛡️ Fault Tolerant</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Includes robust exception handling, dynamic encoder options, and safe input verification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("ℹ️ Project Architecture Details"):
    st.write(
        """
        * **Target Combination:** `df["Target"] = df["Institute Name"] + " | " + df["Course Name"]`
        * **Inference Pipeline:** Inputs are encoded with `LabelEncoder` pickles before passing to `collegename_model.pkl`.
        * **Outputs:** Decoded target strings are split using `prediction.split(" | ")` to populate the result cards.
        """
    )

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #64748b; font-size: 0.85rem;">MHT-CET College & Course Predictor Dashboard</p>',
    unsafe_allow_html=True,
)
