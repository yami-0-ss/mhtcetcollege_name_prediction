import os
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MHT-CET Department Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Styling (CSS Scaffolding)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
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
    .chance-high {
        color: #15803D;
        font-weight: 700;
        background-color: #DCFCE7;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .chance-mod {
        color: #B45309;
        font-weight: 700;
        background-color: #FEF3C7;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .chance-low {
        color: #B91C1C;
        font-weight: 700;
        background-color: #FEE2E2;
        padding: 3px 8px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. App Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🎓 MHT-CET College & Department Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze cutoff statistics & predict department options based on CAP I allocation records</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Data Loading & Sanitization
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    csv_path = "CAP_Seat_Allocation (v2).xlsx - CAP I - Maharashtra (MHTCET) (1).csv"
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    # Strip whitespace from string columns to avoid lookup mismatches
    str_cols = ['Institute Name', 'Course Name', 'Category', 'Gender', 'Seat Alloted']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Ensure Percentile is float and clean NaNs
    df['MHTCET Percentile'] = pd.to_numeric(df['MHTCET Percentile'], errors='coerce')
    df = df.dropna(subset=['MHTCET Percentile'])
    
    # Clean special symbols ($ and @) from Category names
    df['Clean_Category'] = df['Category'].str.replace(r'[\$\@]', '', regex=True).str.strip()
    return df

df = load_data()

if df is None:
    st.error("Dataset Error: File `CAP_Seat_Allocation (v2).xlsx - CAP I - Maharashtra (MHTCET) (1).csv` was not found in the root directory.")
    st.info("Please place your CSV dataset in the same directory as `app.py`.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. User Input & Layout Section
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Candidate Input")
    
    percentile = st.number_input(
        "MHT-CET Percentile",
        min_value=0.0,
        max_value=100.0,
        value=92.0,
        step=0.01,
        help="Enter your MHT-CET overall score/percentile."
    )
    
    g_col, c_col = st.columns(2)
    with g_col:
        genders = ["All"] + sorted(df["Gender"].unique().tolist())
        selected_gender = st.selectbox("Gender", options=genders)
        
    with c_col:
        categories = ["All"] + sorted(df["Clean_Category"].unique().tolist())
        default_idx = categories.index("OPEN") if "OPEN" in categories else 0
        selected_category = st.selectbox("Category", options=categories, index=default_idx)

    st.subheader("🏫 Target College")
    all_colleges = ["All Colleges"] + sorted(df["Institute Name"].unique().tolist())
    selected_institute = st.selectbox("Select College / Institute", options=all_colleges)
    
    predict_btn = st.button("Predict Departments")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Results & Department Prediction Processing
# -----------------------------------------------------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Projected Department Allocations")
    
    if predict_btn or selected_institute:
        filtered_df = df.copy()
        
        # Apply Gender Filter
        if selected_gender != "All":
            filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
            
        # Apply Category Filter
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['Clean_Category'] == selected_category]
            
        # Case 1: Specific College Selected
        if selected_institute != "All Colleges":
            filtered_df = filtered_df[filtered_df['Institute Name'] == selected_institute]
            
            if filtered_df.empty:
                st.warning(f"No records matching category '{selected_category}' for this college. Showing all category records for {selected_institute}.")
                filtered_df = df[df['Institute Name'] == selected_institute]

            # Aggregate stats per department
            dept_summary = filtered_df.groupby('Course Name')['MHTCET Percentile'].agg(
                Min_Cutoff='min',
                Median_Cutoff='median',
                Max_Cutoff='max',
                Seats_Allocated='count'
            ).reset_index()

            # Classification logic
            def evaluate_chance(min_val, median_val):
                if percentile >= median_val:
                    return "High Chance"
                elif percentile >= min_val:
                    return "Moderate Chance"
                elif percentile >= (min_val - 2.5):
                    return "Reach / Low Chance"
                else:
                    return "Unlikely"

            dept_summary['Chance'] = dept_summary.apply(
                lambda row: evaluate_chance(row['Min_Cutoff'], row['Median_Cutoff']), axis=1
            )
            
            # Sort departments by cutoff score
            dept_summary = dept_summary.sort_values(by='Median_Cutoff', ascending=False)
            
            st.write(f"Department cutoff predictions for **{selected_institute}**:")
            
            for _, row in dept_summary.iterrows():
                chance_status = row['Chance']
                if chance_status == "High Chance":
                    badge = f'<span class="chance-high">High Chance</span>'
                elif chance_status == "Moderate Chance":
                    badge = f'<span class="chance-mod">Moderate Chance</span>'
                else:
                    badge = f'<span class="chance-low">{chance_status}</span>'
                    
                st.markdown(f"""
                **{row['Course Name']}**  
                Status: {badge} | Min Cutoff: **{row['Min_Cutoff']:.2f}** | Median: **{row['Median_Cutoff']:.2f}**  
                ---
                """, unsafe_allow_html=True)
                
        # Case 2: "All Colleges" Selected
        else:
            eligible = filtered_df[filtered_df['MHTCET Percentile'] <= percentile + 1.0]
            top_colleges = eligible.groupby(['Institute Name', 'Course Name'])['MHTCET Percentile'].agg(
                Cutoff='max', Count='count'
            ).reset_index().sort_values(by='Cutoff', ascending=False)
            
            st.write(f"Top matching Colleges & Departments near **{percentile} percentile**:")
            st.dataframe(top_colleges.head(20), use_container_width=True)
            
    else:
        st.info("Select a college and click **Predict Departments** to view cutoff breakdowns.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. Raw Data Cutoff Explorer
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Cutoff Search Table")

explorer_df = df.copy()
if selected_institute != "All Colleges":
    explorer_df = explorer_df[explorer_df['Institute Name'] == selected_institute]

explorer_df = explorer_df[
    (explorer_df['MHTCET Percentile'] <= percentile + 2.0) & 
    (explorer_df['MHTCET Percentile'] >= percentile - 5.0)
]

st.dataframe(
    explorer_df[['Merit Number', 'MHTCET Percentile', 'Institute Name', 'Course Name', 'Category', 'Seat Alloted']].head(25),
    use_container_width=True
)
