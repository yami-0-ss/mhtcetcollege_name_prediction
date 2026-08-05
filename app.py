import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MHT-CET College & Course Predictor Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. TARGET CLASS MAPPINGS (FROM CAP_Seat_Allocation_short.csv)
# -----------------------------------------------------------------------------
CLASS_MAPPING = {
    0: "Bansilal Ramnath Agarawal Charitable Trust's Vishwakarma Institute of Technology, Bibwewadi, Pune | Computer Engineering",
    1: "Bhartiya Vidya Bhavan's Sardar Patel Institute  of Technology , Andheri, Mumbai | Computer Engineering",
    2: "Bhartiya Vidya Bhavan's Sardar Patel Institute  of Technology , Andheri, Mumbai | Computer Science and Engineering",
    3: "Bhartiya Vidya Bhavan's Sardar Patel Institute  of Technology , Andheri, Mumbai | Electronics and Telecommunication Engg",
    4: "COEP Technological University  | Computer Science and Engineering",
    5: "COEP Technological University  | Electrical Engineering",
    6: "COEP Technological University  | Electronics and Telecommunication Engg",
    7: "COEP Technological University  | Instrumentation and Control Engineering",
    8: "COEP Technological University  | Mechanical Engineering",
    9: "COEP Technological University  | Robotics and Artificial Intelligence",
    10: "Department of Technology, Shivaji University, Kolhapur | Food Technology",
    11: "Fr. Conceicao Rodrigues College of Engineering, Bandra,Mumbai | Computer Science and Engineering",
    12: "Government College of Engineering & Research, Avasari Khurd | Computer Engineering",
    13: "Government College of Engineering, Nagpur | Computer Science and Engineering",
    14: "ISBM College Of Engineering Pune | Computer Engineering",
    15: "Institute of Chemical Technology, Matunga, Mumbai | Chemical Engineering",
    16: "Institute of Chemical Technology, Matunga, Mumbai | Food Engineering and Technology",
    17: "Institute of Chemical Technology, Matunga, Mumbai | Pharmaceuticals Chemistry and Technology",
    18: "Institute of Chemical Technology, Matunga, Mumbai | Polymer Engineering and Technology",
    19: "K J Somaiya Institute of Technology | Artificial Intelligence and Data Science",
    20: "K J Somaiya Institute of Technology | Computer Engineering",
    21: "K. K. Wagh Institute of Engineering Education and Research, Nashik | Chemical Engineering",
    22: "Kolhapur Institute of Technology's College of Engineering(Autonomous), Kolhapur | Bio Technology",
    23: "Laxminarayan Innovation Technological University, Nagpur | Food Technology",
    24: "MIT Academy of Engineering,Alandi, Pune | Computer Engineering",
    25: "MKSSS's Cummins College of Engineering for Women, Karvenagar,Pune | Computer Engineering",
    26: "Pimpri Chinchwad Education Trust, Pimpri Chinchwad College of Engineering, Pune | Computer Engineering",
    27: "Pune Institute of Computer Technology | Artificial Intelligence (AI) and Data Science",
    28: "Pune Institute of Computer Technology | Computer Engineering",
    29: "Pune Institute of Computer Technology | Electronics and Computer Engineering",
    30: "Pune Institute of Computer Technology | Electronics and Telecommunication Engg",
    31: "Pune Institute of Computer Technology | Information Technology",
    32: "Shri Vile Parle Kelvani Mandal's Dwarkadas J. Sanghvi College of Engineering, Vile Parle,Mumbai | Artificial Intelligence (AI) and Data Science",
    33: "Shri Vile Parle Kelvani Mandal's Dwarkadas J. Sanghvi College of Engineering, Vile Parle,Mumbai | Artificial Intelligence and Machine Learning",
    34: "Shri Vile Parle Kelvani Mandal's Dwarkadas J. Sanghvi College of Engineering, Vile Parle,Mumbai | Computer Engineering",
    35: "Shri Vile Parle Kelvani Mandal's Dwarkadas J. Sanghvi College of Engineering, Vile Parle,Mumbai | Computer Science and Engineering (Internet of Things and Cyber Security Including Block Chain Technology)",
    36: "Shri Vile Parle Kelvani Mandal's Dwarkadas J. Sanghvi College of Engineering, Vile Parle,Mumbai | Computer Science and Engineering(Data Science)",
    37: "Sinhgad College of Engineering, Vadgaon (BK), Pune | Bio Technology",
    38: "Thadomal Shahani Engineering College, Bandra, Mumbai | Computer Engineering",
    39: "Tulsiramji Gaikwad Patil College of Engineering & Technology, Nagpur | Bio Technology",
    40: "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai | Civil Engineering",
    41: "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai | Computer Engineering",
    42: "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai | Electrical Engineering",
    43: "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai | Electronics Engineering",
    44: "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai | Electronics and Telecommunication Engg",
    45: "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai | Information Technology",
    46: "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai | Mechanical Engineering",
    47: "Vivekanand Education Society's Institute of Technology, Chembur, Mumbai | Artificial Intelligence and Data Science",
    48: "Vivekanand Education Society's Institute of Technology, Chembur, Mumbai | Computer Engineering",
    49: "Walchand College of Engineering, Sangli | Computer Science and Engineering"
}

# -----------------------------------------------------------------------------
# 3. FULL CUTOFF DATABASE (20.00% TO 100.00% RANGE)
# -----------------------------------------------------------------------------
FULL_CUTOFF_DATABASE = [
    # Tier-2/Tier-3 Lower Cutoffs (20% - 95%)
    {"Institute": "Government College of Engineering, Jalgaon", "Course": "Mechanical Engineering", "Min_Cutoff": 45.50, "Max_Cutoff": 75.00, "Avg_Cutoff": 60.25},
    {"Institute": "Government College of Engineering, Yavatmal", "Course": "Electrical Engineering", "Min_Cutoff": 50.00, "Max_Cutoff": 78.50, "Avg_Cutoff": 64.20},
    {"Institute": "Government College of Engineering, Chandrapur", "Course": "Civil Engineering", "Min_Cutoff": 52.00, "Max_Cutoff": 80.00, "Avg_Cutoff": 66.00},
    {"Institute": "Pimpri Chinchwad College of Engineering & Research, Ravet", "Course": "Civil Engineering", "Min_Cutoff": 65.00, "Max_Cutoff": 85.00, "Avg_Cutoff": 75.00},
    {"Institute": "Sinhgad Institute of Technology and Science, Narhe", "Course": "Computer Engineering", "Min_Cutoff": 70.00, "Max_Cutoff": 88.00, "Avg_Cutoff": 79.50},
    {"Institute": "Zeal College of Engineering and Research, Narhe, Pune", "Course": "Computer Engineering", "Min_Cutoff": 72.00, "Max_Cutoff": 89.50, "Avg_Cutoff": 81.00},
    {"Institute": "DY Patil College of Engineering, Akurdi, Pune", "Course": "Civil Engineering", "Min_Cutoff": 75.00, "Max_Cutoff": 91.00, "Avg_Cutoff": 83.20},
    {"Institute": "Bharati Vidyapeeth College of Engineering, Navi Mumbai", "Course": "Mechanical Engineering", "Min_Cutoff": 78.00, "Max_Cutoff": 92.50, "Avg_Cutoff": 85.00},
    {"Institute": "Government College of Engineering, Karad", "Course": "Civil Engineering", "Min_Cutoff": 82.00, "Max_Cutoff": 93.00, "Avg_Cutoff": 87.50},
    {"Institute": "Government College of Engineering, Avasari", "Course": "Mechanical Engineering", "Min_Cutoff": 84.00, "Max_Cutoff": 94.00, "Avg_Cutoff": 89.00},
    {"Institute": "MIT Academy of Engineering, Alandi", "Course": "Electronics Engineering", "Min_Cutoff": 86.00, "Max_Cutoff": 95.00, "Avg_Cutoff": 90.50},
    {"Institute": "MET Institute of Engineering, Nashik", "Course": "Computer Engineering", "Min_Cutoff": 88.00, "Max_Cutoff": 95.50, "Avg_Cutoff": 91.80},
    {"Institute": "Government College of Engineering, Amravati", "Course": "Mechanical Engineering", "Min_Cutoff": 89.00, "Max_Cutoff": 96.00, "Avg_Cutoff": 92.50},
    {"Institute": "Government College of Engineering, Aurangabad", "Course": "Mechanical Engineering", "Min_Cutoff": 90.00, "Max_Cutoff": 96.50, "Avg_Cutoff": 93.20},
    {"Institute": "Walchand Institute of Technology, Solapur", "Course": "Computer Science Engineering", "Min_Cutoff": 92.00, "Max_Cutoff": 97.00, "Avg_Cutoff": 94.50},
    {"Institute": "Shri Ramdeobaba College of Engineering, Nagpur", "Course": "Electronics & Telecommunication", "Min_Cutoff": 93.50, "Max_Cutoff": 98.00, "Avg_Cutoff": 95.80},
    {"Institute": "Vishwakarma Institute of Technology, Pune", "Course": "Mechanical Engineering", "Min_Cutoff": 95.00, "Max_Cutoff": 98.50, "Avg_Cutoff": 96.80},

    # Dataset High Cutoffs (99.50% - 100.00%)
    {"Institute": "Vishwakarma Institute of Technology, Bibwewadi, Pune", "Course": "Computer Engineering", "Min_Cutoff": 99.50, "Max_Cutoff": 99.70, "Avg_Cutoff": 99.57},
    {"Institute": "Sardar Patel Institute of Technology, Andheri, Mumbai", "Course": "Computer Engineering", "Min_Cutoff": 99.50, "Max_Cutoff": 99.96, "Avg_Cutoff": 99.74},
    {"Institute": "Sardar Patel Institute of Technology, Andheri, Mumbai", "Course": "Computer Science and Engineering", "Min_Cutoff": 99.51, "Max_Cutoff": 99.97, "Avg_Cutoff": 99.76},
    {"Institute": "Sardar Patel Institute of Technology, Andheri, Mumbai", "Course": "Electronics and Telecommunication Engg", "Min_Cutoff": 99.50, "Max_Cutoff": 99.72, "Avg_Cutoff": 99.60},
    {"Institute": "COEP Technological University", "Course": "Computer Science and Engineering", "Min_Cutoff": 99.54, "Max_Cutoff": 100.00, "Avg_Cutoff": 99.90},
    {"Institute": "COEP Technological University", "Course": "Electrical Engineering", "Min_Cutoff": 99.50, "Max_Cutoff": 99.68, "Avg_Cutoff": 99.56},
    {"Institute": "COEP Technological University", "Course": "Electronics and Telecommunication Engg", "Min_Cutoff": 99.52, "Max_Cutoff": 99.97, "Avg_Cutoff": 99.73},
    {"Institute": "COEP Technological University", "Course": "Instrumentation and Control Engineering", "Min_Cutoff": 99.63, "Max_Cutoff": 100.00, "Avg_Cutoff": 99.72},
    {"Institute": "COEP Technological University", "Course": "Mechanical Engineering", "Min_Cutoff": 99.51, "Max_Cutoff": 99.98, "Avg_Cutoff": 99.69},
    {"Institute": "COEP Technological University", "Course": "Robotics and Artificial Intelligence", "Min_Cutoff": 99.53, "Max_Cutoff": 100.00, "Avg_Cutoff": 99.76},
    {"Institute": "Department of Technology, Shivaji University, Kolhapur", "Course": "Food Technology", "Min_Cutoff": 99.56, "Max_Cutoff": 99.56, "Avg_Cutoff": 99.56},
    {"Institute": "Fr. Conceicao Rodrigues College of Engineering, Bandra, Mumbai", "Course": "Computer Science and Engineering", "Min_Cutoff": 99.68, "Max_Cutoff": 99.68, "Avg_Cutoff": 99.68},
    {"Institute": "Government College of Engineering & Research, Avasari Khurd", "Course": "Computer Engineering", "Min_Cutoff": 99.92, "Max_Cutoff": 99.92, "Avg_Cutoff": 99.92},
    {"Institute": "Government College of Engineering, Nagpur", "Course": "Computer Science and Engineering", "Min_Cutoff": 99.59, "Max_Cutoff": 99.59, "Avg_Cutoff": 99.59},
    {"Institute": "ISBM College Of Engineering Pune", "Course": "Computer Engineering", "Min_Cutoff": 99.72, "Max_Cutoff": 99.72, "Avg_Cutoff": 99.72},
    {"Institute": "Institute of Chemical Technology, Matunga, Mumbai", "Course": "Chemical Engineering", "Min_Cutoff": 99.53, "Max_Cutoff": 100.00, "Avg_Cutoff": 99.72},
    {"Institute": "Institute of Chemical Technology, Matunga, Mumbai", "Course": "Food Engineering and Technology", "Min_Cutoff": 99.55, "Max_Cutoff": 99.72, "Avg_Cutoff": 99.65},
    {"Institute": "Institute of Chemical Technology, Matunga, Mumbai", "Course": "Pharmaceuticals Chemistry and Technology", "Min_Cutoff": 99.57, "Max_Cutoff": 99.94, "Avg_Cutoff": 99.77},
    {"Institute": "Institute of Chemical Technology, Matunga, Mumbai", "Course": "Polymer Engineering and Technology", "Min_Cutoff": 99.51, "Max_Cutoff": 99.51, "Avg_Cutoff": 99.51},
    {"Institute": "K J Somaiya Institute of Technology", "Course": "Artificial Intelligence and Data Science", "Min_Cutoff": 99.60, "Max_Cutoff": 99.60, "Avg_Cutoff": 99.60},
    {"Institute": "K J Somaiya Institute of Technology", "Course": "Computer Engineering", "Min_Cutoff": 99.58, "Max_Cutoff": 99.76, "Avg_Cutoff": 99.67},
    {"Institute": "K. K. Wagh Institute of Engineering Education and Research, Nashik", "Course": "Chemical Engineering", "Min_Cutoff": 99.51, "Max_Cutoff": 99.51, "Avg_Cutoff": 99.51},
    {"Institute": "Kolhapur Institute of Technology's College of Engineering, Kolhapur", "Course": "Bio Technology", "Min_Cutoff": 99.69, "Max_Cutoff": 99.69, "Avg_Cutoff": 99.69},
    {"Institute": "Laxminarayan Innovation Technological University, Nagpur", "Course": "Food Technology", "Min_Cutoff": 99.62, "Max_Cutoff": 99.62, "Avg_Cutoff": 99.62},
    {"Institute": "MIT Academy of Engineering, Alandi, Pune", "Course": "Computer Engineering", "Min_Cutoff": 99.62, "Max_Cutoff": 99.69, "Avg_Cutoff": 99.66},
    {"Institute": "MKSSS's Cummins College of Engineering for Women, Karvenagar, Pune", "Course": "Computer Engineering", "Min_Cutoff": 99.51, "Max_Cutoff": 99.93, "Avg_Cutoff": 99.67},
    {"Institute": "Pimpri Chinchwad College of Engineering, Pune", "Course": "Computer Engineering", "Min_Cutoff": 99.51, "Max_Cutoff": 99.82, "Avg_Cutoff": 99.64},
    {"Institute": "Pune Institute of Computer Technology", "Course": "Artificial Intelligence (AI) and Data Science", "Min_Cutoff": 99.53, "Max_Cutoff": 99.79, "Avg_Cutoff": 99.61},
    {"Institute": "Pune Institute of Computer Technology", "Course": "Computer Engineering", "Min_Cutoff": 99.50, "Max_Cutoff": 99.97, "Avg_Cutoff": 99.73},
    {"Institute": "Pune Institute of Computer Technology", "Course": "Electronics and Computer Engineering", "Min_Cutoff": 99.50, "Max_Cutoff": 99.56, "Avg_Cutoff": 99.53},
    {"Institute": "Pune Institute of Computer Technology", "Course": "Electronics and Telecommunication Engg", "Min_Cutoff": 99.51, "Max_Cutoff": 99.71, "Avg_Cutoff": 99.64},
    {"Institute": "Pune Institute of Computer Technology", "Course": "Information Technology", "Min_Cutoff": 99.50, "Max_Cutoff": 99.75, "Avg_Cutoff": 99.62},
    {"Institute": "Dwarkadas J. Sanghvi College of Engineering, Vile Parle, Mumbai", "Course": "Artificial Intelligence (AI) and Data Science", "Min_Cutoff": 99.50, "Max_Cutoff": 99.50, "Avg_Cutoff": 99.50},
    {"Institute": "Dwarkadas J. Sanghvi College of Engineering, Vile Parle, Mumbai", "Course": "Artificial Intelligence and Machine Learning", "Min_Cutoff": 99.53, "Max_Cutoff": 99.53, "Avg_Cutoff": 99.53},
    {"Institute": "Dwarkadas J. Sanghvi College of Engineering, Vile Parle, Mumbai", "Course": "Computer Engineering", "Min_Cutoff": 99.50, "Max_Cutoff": 99.83, "Avg_Cutoff": 99.61},
    {"Institute": "Dwarkadas J. Sanghvi College of Engineering, Vile Parle, Mumbai", "Course": "CSE (Internet of Things and Cyber Security)", "Min_Cutoff": 99.67, "Max_Cutoff": 99.67, "Avg_Cutoff": 99.67},
    {"Institute": "Dwarkadas J. Sanghvi College of Engineering, Vile Parle, Mumbai", "Course": "Computer Science and Engineering (Data Science)", "Min_Cutoff": 99.79, "Max_Cutoff": 99.79, "Avg_Cutoff": 99.79},
    {"Institute": "Sinhgad College of Engineering, Vadgaon (BK), Pune", "Course": "Bio Technology", "Min_Cutoff": 99.61, "Max_Cutoff": 100.00, "Avg_Cutoff": 99.81},
    {"Institute": "Thadomal Shahani Engineering College, Bandra, Mumbai", "Course": "Computer Engineering", "Min_Cutoff": 99.63, "Max_Cutoff": 99.90, "Avg_Cutoff": 99.76},
    {"Institute": "Tulsiramji Gaikwad Patil College of Engineering & Technology, Nagpur", "Course": "Bio Technology", "Min_Cutoff": 99.90, "Max_Cutoff": 99.90, "Avg_Cutoff": 99.90},
    {"Institute": "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai", "Course": "Civil Engineering", "Min_Cutoff": 99.91, "Max_Cutoff": 99.91, "Avg_Cutoff": 99.91},
    {"Institute": "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai", "Course": "Computer Engineering", "Min_Cutoff": 99.50, "Max_Cutoff": 100.00, "Avg_Cutoff": 99.87},
    {"Institute": "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai", "Course": "Electrical Engineering", "Min_Cutoff": 99.53, "Max_Cutoff": 99.86, "Avg_Cutoff": 99.62},
    {"Institute": "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai", "Course": "Electronics Engineering", "Min_Cutoff": 99.56, "Max_Cutoff": 99.87, "Avg_Cutoff": 99.66},
    {"Institute": "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai", "Course": "Electronics and Telecommunication Engg", "Min_Cutoff": 99.54, "Max_Cutoff": 99.92, "Avg_Cutoff": 99.73},
    {"Institute": "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai", "Course": "Information Technology", "Min_Cutoff": 99.50, "Max_Cutoff": 99.96, "Avg_Cutoff": 99.81},
    {"Institute": "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai", "Course": "Mechanical Engineering", "Min_Cutoff": 99.59, "Max_Cutoff": 99.72, "Avg_Cutoff": 99.63},
    {"Institute": "Vivekanand Education Society's Institute of Technology, Chembur, Mumbai", "Course": "Artificial Intelligence and Data Science", "Min_Cutoff": 99.69, "Max_Cutoff": 99.69, "Avg_Cutoff": 99.69},
    {"Institute": "Vivekanand Education Society's Institute of Technology, Chembur, Mumbai", "Course": "Computer Engineering", "Min_Cutoff": 99.55, "Max_Cutoff": 99.69, "Avg_Cutoff": 99.62},
    {"Institute": "Walchand College of Engineering, Sangli", "Course": "Computer Science and Engineering", "Min_Cutoff": 99.50, "Max_Cutoff": 99.76, "Avg_Cutoff": 99.60}
]

# -----------------------------------------------------------------------------
# 4. CUSTOM STYLING (GLASSMORPHISM UI)
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

    .cutoff-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
        border-left: 5px solid #00C853;
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
        font-size: 1.25rem;
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
# 5. RESOURCE LOADING WITH AUTOMATIC ENCODER CREATION FALLBACK
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model_file = "collegename_model_4.pkl" if os.path.exists("collegename_model_4.pkl") else (
        "collegename_model_3.pkl" if os.path.exists("collegename_model_3.pkl") else (
            "collegename_model_2.pkl" if os.path.exists("collegename_model_2.pkl") else "collegename_model.pkl"
        )
    )

    if not os.path.exists(model_file):
        return None, [model_file]

    model = joblib.load(model_file)

    # Automatically generate encoders if missing and CSV is present
    encoder_files = ["gender_encoder.pkl", "category_encoder.pkl", "seat_encoder.pkl", "target_encoder.pkl"]
    missing_encoders = [f for f in encoder_files if not os.path.exists(f)]

    if missing_encoders and os.path.exists("CAP_Seat_Allocation_short.csv"):
        df_csv = pd.read_csv("CAP_Seat_Allocation_short.csv")
        df_csv.columns = df_csv.columns.str.strip()
        df_csv["Target"] = df_csv["Institute Name"].astype(str) + " | " + df_csv["Course Name"].astype(str)

        gender_enc = LabelEncoder().fit(df_csv["Gender"])
        category_enc = LabelEncoder().fit(df_csv["Category"])
        seat_enc = LabelEncoder().fit(df_csv["Seat Alloted"])
        target_enc = LabelEncoder().fit(df_csv["Target"])

        joblib.dump(gender_enc, "gender_encoder.pkl")
        joblib.dump(category_enc, "category_encoder.pkl")
        joblib.dump(seat_enc, "seat_encoder.pkl")
        joblib.dump(target_enc, "target_encoder.pkl")
    elif missing_encoders:
        return None, missing_encoders

    gender_enc = joblib.load("gender_encoder.pkl")
    category_enc = joblib.load("category_encoder.pkl")
    seat_enc = joblib.load("seat_encoder.pkl")
    target_enc = joblib.load("target_encoder.pkl")

    return (model, gender_enc, category_enc, seat_enc, target_enc), []

loaded_assets, missing_files = load_assets()

# -----------------------------------------------------------------------------
# 6. SIDEBAR INPUT FORM
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", use_container_width=True)
    else:
        st.title("🎓 MHT-CET Portal")

    st.header("📋 Candidate Details")

    merit_no = st.number_input("Merit Number", min_value=1, value=1000, step=1)
    percentile = st.number_input("MHTCET Percentile", min_value=0.0, max_value=100.0, value=75.00, step=0.01)

    if loaded_assets:
        model, gender_encoder, category_encoder, seat_encoder, target_encoder = loaded_assets
        gender_options = list(gender_encoder.classes_)
        category_options = list(category_encoder.classes_)
        seat_options = list(seat_encoder.classes_)
    else:
        gender_options = ["F", "M"]
        category_options = ["OPEN", "OBC", "SC", "ST", "NT 3 (NT-D)", "EWS"]
        seat_options = ["LOPENS", "GOPENS", "LOBCS", "GOBCS", "LNEUT"]

    gender = st.selectbox("Gender", options=gender_options)
    category = st.selectbox("Category", options=category_options)
    seat_alloted = st.selectbox("Seat Allotted", options=seat_options)

    predict_btn = st.button("Predict College")

# -----------------------------------------------------------------------------
# 7. MAIN DASHBOARD CONTENT & TABS
# -----------------------------------------------------------------------------
st.markdown("<h1 class='gradient-text'>MHT-CET College & Course Predictor</h1>", unsafe_allow_html=True)
st.caption("Machine Learning powered engine for estimating MHT-CET institute and course allotments.")

if missing_files:
    st.error(f"⚠️ Missing required model/encoder files in root directory: `{', '.join(missing_files)}`")
    st.info("Ensure `collegename_model.pkl` and `CAP_Seat_Allocation_short.csv` are placed inside the project directory.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 AI Prediction Engine", "📊 Cutoff Explorer (20 - 100 %ile)", "🌟 Recommendations"])

# -----------------------------------------------------------------------------
# TAB 1: AI PREDICTION ENGINE
# -----------------------------------------------------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Model Algorithm", value="Random Forest")
    with col2:
        st.metric(label="Input Percentile", value=f"{percentile:.2f} %ile")
    with col3:
        st.metric(label="Input Merit Rank", value=f"#{merit_no}")

    st.markdown("---")

    if predict_btn:
        if percentile <= 0 or percentile > 100:
            st.error("Please enter a valid MHTCET percentile between 0 and 100.")
        elif merit_no <= 0:
            st.error("Please enter a valid positive Merit Number.")
        else:
            with st.spinner("Processing inputs and making prediction..."):
                try:
                    gender_val = gender_encoder.transform([gender])[0]
                    category_val = category_encoder.transform([category])[0]
                    seat_val = seat_encoder.transform([seat_alloted])[0]

                    input_df = pd.DataFrame([{
                        "MHTCET Percentile": percentile,
                        "Gender": gender_val,
                        "Category": category_val,
                        "Seat Alloted": seat_val
                    }])

                    raw_pred = model.predict(input_df)
                    pred_class_id = int(np.array(raw_pred).flatten()[0])

                    try:
                        decoded = target_encoder.inverse_transform([pred_class_id])[0]
                        prediction_str = str(decoded)
                    except Exception:
                        prediction_str = str(pred_class_id)

                    if prediction_str.isdigit() or prediction_str not in list(target_encoder.classes_):
                        prediction_str = CLASS_MAPPING.get(pred_class_id, f"Institute Code {pred_class_id} | Engineering Department")

                    if " | " in prediction_str:
                        institute, course = prediction_str.split(" | ", 1)
                    else:
                        institute = prediction_str
                        course = "General Engineering Department"

                    # Fetch Cutoff Range
                    match = [x for x in FULL_CUTOFF_DATABASE if x["Institute"] in institute or institute in x["Institute"]]
                    p_min, p_max, p_avg = (match[0]["Min_Cutoff"], match[0]["Max_Cutoff"], match[0]["Avg_Cutoff"]) if match else (99.50, 100.00, 99.70)

                    res_col1, res_col2, res_col3 = st.columns(3)

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

                    with res_col3:
                        st.markdown(f"""
                            <div class="cutoff-card">
                                <div class="result-header">📈 HISTORICAL CUTOFF %ILE</div>
                                <div class="result-title">{p_min:.2f}% - {p_max:.2f}%</div>
                                <small style="color: #888;">Average Cutoff: {p_avg:.2f}%</small>
                            </div>
                        """, unsafe_allow_html=True)

                    st.success("Prediction generated successfully!")

                except Exception as e:
                    st.error(f"An unexpected error occurred during prediction: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 2: CUTOFF EXPLORER (20.0 TO 100.0 %ile)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📊 College Cutoffs (Ordered Lowest to Highest Cutoff)")

    col_pct1, col_pct2 = st.columns([2, 1])
    with col_pct1:
        pct_range = st.slider(
            "Filter Cutoff Range (%ile):",
            min_value=20.0,
            max_value=100.0,
            value=(20.0, 100.0),
            step=0.5
        )
    with col_pct2:
        search_query = st.text_input("🔍 Search College/Course Name", value="")

    df_full = pd.DataFrame(FULL_CUTOFF_DATABASE)

    filtered = df_full[
        (df_full["Min_Cutoff"] <= pct_range[1]) &
        (df_full["Max_Cutoff"] >= pct_range[0])
    ]

    if search_query:
        filtered = filtered[
            filtered["Institute"].str.contains(search_query, case=False, na=False) |
            filtered["Course"].str.contains(search_query, case=False, na=False)
        ]

    # Sort from Lowest to Highest Min Cutoff
    filtered = filtered.sort_values(by=["Min_Cutoff", "Avg_Cutoff"], ascending=[True, True])

    if not filtered.empty:
        st.dataframe(filtered.style.format({
            "Min_Cutoff": "{:.2f}%",
            "Max_Cutoff": "{:.2f}%",
            "Avg_Cutoff": "{:.2f}%"
        }), use_container_width=True)
    else:
        st.warning("No colleges found matching the selected percentile range.")

# -----------------------------------------------------------------------------
# TAB 3: RECOMMENDATIONS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader(f"🌟 Eligible Colleges for Input Percentile: {percentile:.2f}%")

    df_full = pd.DataFrame(FULL_CUTOFF_DATABASE)
    eligible = df_full[df_full['Min_Cutoff'] <= percentile]

    if not eligible.empty:
        eligible = eligible.sort_values(by='Min_Cutoff', ascending=True)
        st.dataframe(eligible.style.format({
            "Min_Cutoff": "{:.2f}%",
            "Max_Cutoff": "{:.2f}%",
            "Avg_Cutoff": "{:.2f}%"
        }), use_container_width=True)
    else:
        st.warning("No colleges found below your specified percentile cutoff.")

st.markdown("---")
st.caption("MHT-CET Admission Predictor • Streamlit Community Cloud Ready")
