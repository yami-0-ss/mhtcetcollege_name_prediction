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
    page_title="MHT-CET College & Course Predictor Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. EXACT 50 TARGET CLASS MAPPINGS & HISTORICAL CUTOFF DATA
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

# (Min Cutoff %, Max Cutoff %, Average Cutoff %) from CAP_Seat_Allocation_short.csv
CUTOFF_STATS = {
    0: (99.50, 99.70, 99.57),
    1: (99.50, 99.96, 99.74),
    2: (99.51, 99.97, 99.76),
    3: (99.50, 99.72, 99.60),
    4: (99.54, 100.00, 99.90),
    5: (99.50, 99.68, 99.56),
    6: (99.52, 99.97, 99.73),
    7: (99.63, 100.00, 99.72),
    8: (99.51, 99.98, 99.69),
    9: (99.53, 100.00, 99.76),
    10: (99.56, 99.56, 99.56),
    11: (99.68, 99.68, 99.68),
    12: (99.92, 99.92, 99.92),
    13: (99.59, 99.59, 99.59),
    14: (99.72, 99.72, 99.72),
    15: (99.53, 100.00, 99.72),
    16: (99.55, 99.72, 99.65),
    17: (99.57, 99.94, 99.77),
    18: (99.51, 99.51, 99.51),
    19: (99.60, 99.60, 99.60),
    20: (99.58, 99.76, 99.67),
    21: (99.51, 99.51, 99.51),
    22: (99.69, 99.69, 99.69),
    23: (99.62, 99.62, 99.62),
    24: (99.62, 99.69, 99.66),
    25: (99.51, 99.93, 99.67),
    26: (99.51, 99.82, 99.64),
    27: (99.53, 99.79, 99.61),
    28: (99.50, 99.97, 99.73),
    29: (99.50, 99.56, 99.53),
    30: (99.51, 99.71, 99.64),
    31: (99.50, 99.75, 99.62),
    32: (99.50, 99.50, 99.50),
    33: (99.53, 99.53, 99.53),
    34: (99.50, 99.83, 99.61),
    35: (99.67, 99.67, 99.67),
    36: (99.79, 99.79, 99.79),
    37: (99.61, 100.00, 99.81),
    38: (99.63, 99.90, 99.76),
    39: (99.90, 99.90, 99.90),
    40: (99.91, 99.91, 99.91),
    41: (99.50, 100.00, 99.87),
    42: (99.53, 99.86, 99.62),
    43: (99.56, 99.87, 99.66),
    44: (99.54, 99.92, 99.73),
    45: (99.50, 99.96, 99.81),
    46: (99.59, 99.72, 99.63),
    47: (99.69, 99.69, 99.69),
    48: (99.55, 99.69, 99.62),
    49: (99.50, 99.76, 99.60)
}

# -----------------------------------------------------------------------------
# 3. CUSTOM STYLING (Glassmorphism UI)
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
# 4. RESOURCE LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model_file = "collegename_model_3.pkl" if os.path.exists("collegename_model_3.pkl") else (
        "collegename_model_2.pkl" if os.path.exists("collegename_model_2.pkl") else "collegename_model.pkl"
    )
    
    files = {
        "model": model_file,
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

    merit_no = st.number_input("Merit Number", min_value=1, value=1000, step=1)
    percentile = st.number_input("MHTCET Percentile", min_value=0.0, max_value=100.0, value=99.60, step=0.01)

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
# 6. DASHBOARD MAIN HEADER & TABS
# -----------------------------------------------------------------------------
st.markdown("<h1 class='gradient-text'>MHT-CET College & Course Predictor</h1>", unsafe_allow_html=True)
st.caption("Machine Learning powered engine for estimating MHT-CET institute and course allotments.")

if missing_files:
    st.error(f"⚠️ Missing required model/encoder files in root directory: `{', '.join(missing_files)}`")
    st.info("Ensure all `.pkl` files are placed inside the project directory.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 AI Prediction Engine", "📊 Cutoff Explorer", "🌟 Recommendations"])

# -----------------------------------------------------------------------------
# TAB 1: PREDICTION ENGINE WITH CUTOFF STATS
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

                    # Get Cutoff Stats
                    p_min, p_max, p_avg = CUTOFF_STATS.get(pred_class_id, (0.0, 0.0, 0.0))

                    # Render 3 Result Cards (Institute, Course, and College Cutoff)
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
                                <small style="color: #888;">Average: {p_avg:.2f}%</small>
                            </div>
                        """, unsafe_allow_html=True)

                    st.success("Prediction generated successfully!")

                except Exception as e:
                    st.error(f"An unexpected error occurred during prediction: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 2: CUTOFF EXPLORER
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Explore All 50 Colleges & Department Cutoffs")
    
    if os.path.exists("CAP_Seat_Allocation_short.csv"):
        df_csv = pd.read_csv("CAP_Seat_Allocation_short.csv")
        df_csv['Target'] = df_csv['Institute Name'].astype(str) + " | " + df_csv['Course Name'].astype(str)
        
        search_query = st.text_input("🔍 Search by College or Course Name", value="")

        summary_data = []
        for class_id, target_name in CLASS_MAPPING.items():
            sub = df_csv[df_csv['Target'] == target_name]
            if not sub.empty:
                inst, dept = target_name.split(" | ", 1)
                if search_query.lower() in inst.lower() or search_query.lower() in dept.lower():
                    summary_data.append({
                        "Class ID": class_id,
                        "Institute Name": inst,
                        "Course Name": dept,
                        "Min Cutoff %ile": float(sub['MHTCET Percentile'].min()),
                        "Max Cutoff %ile": float(sub['MHTCET Percentile'].max()),
                        "Avg Percentile": float(sub['MHTCET Percentile'].mean())
                    })
        
        res_df = pd.DataFrame(summary_data)
        st.dataframe(res_df.style.format({
            "Min Cutoff %ile": "{:.2f}%",
            "Max Cutoff %ile": "{:.2f}%",
            "Avg Percentile": "{:.2f}%"
        }), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: TOP RECOMMENDATIONS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader(f"🌟 Eligible Colleges for Percentile: {percentile:.2f}%")
    
    if os.path.exists("CAP_Seat_Allocation_short.csv"):
        df_csv = pd.read_csv("CAP_Seat_Allocation_short.csv")
        eligible = df_csv[df_csv['MHTCET Percentile'] <= percentile]
        
        if not eligible.empty:
            recs = eligible.groupby(['Institute Name', 'Course Name'])['MHTCET Percentile'].agg(['min', 'max']).reset_index()
            recs = recs.rename(columns={'min': 'Min Cutoff %ile', 'max': 'Max Cutoff %ile'}).sort_values(by='Max Cutoff %ile', ascending=False)
            st.dataframe(recs.style.format({
                "Min Cutoff %ile": "{:.2f}%",
                "Max Cutoff %ile": "{:.2f}%"
            }), use_container_width=True)
        else:
            st.warning("No colleges found below your specified percentile cutoff.")

st.markdown("---")
st.caption("MHT-CET Admission Predictor • Streamlit Community Cloud Ready")
