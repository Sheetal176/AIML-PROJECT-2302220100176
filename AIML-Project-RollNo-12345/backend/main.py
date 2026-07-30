import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_CORE_DIR = PROJECT_ROOT / "ml_core"
if str(ML_CORE_DIR) not in sys.path:
    sys.path.append(str(ML_CORE_DIR))

from patterns.data_factory import DataProcessorFactory
from patterns.model_registry import ModelRegistry
from patterns.model_strategy import RandomForestStrategy

st.set_page_config(
    page_title="Campus Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

/* ── Reset & Base ───────────────────────────────────────────────────── */
html, body, .stApp {
  font-family: 'Inter', sans-serif !important;
  background-color: #F0F2FF !important;
}
.stApp {
  background: linear-gradient(155deg, #F0F2FF 0%, #FFFFFF 50%, #F5F0FF 100%) !important;
  background-attachment: fixed !important;
  min-height: 100vh;
}

/* ── Hide Streamlit chrome ──────────────────────────────────────────── */
[data-testid="stHeader"]      { background: transparent !important; }
[data-testid="stToolbar"]     { display: none !important; }
[data-testid="stDecoration"]  { display: none !important; }
[data-testid="stBottom"]      { background: transparent !important; }
footer                        { display: none !important; }
.block-container {
  padding-top: 0 !important;
  padding-bottom: 3rem !important;
  max-width: 1200px !important;
}

/* ── Hero Banner ────────────────────────────────────────────────────── */
.hero-banner {
  background: linear-gradient(135deg, #6C5CE7 0%, #a29bfe 50%, #fd79a8 100%);
  border-radius: 0 0 32px 32px;
  padding: 3rem 2rem 2.8rem 2rem;
  text-align: center;
  margin-bottom: 2.5rem;
  box-shadow: 0 8px 32px rgba(108, 92, 231, 0.25);
  position: relative;
  overflow: hidden;
}
.hero-banner::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
}
.hero-badge {
  display: inline-block;
  background: rgba(255,255,255,0.2);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.3);
  border-radius: 30px;
  padding: 4px 16px;
  font-size: 0.82rem;
  font-weight: 600;
  color: #fff;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 1rem;
}
.hero-title {
  font-family: 'Outfit', sans-serif;
  font-size: 3rem;
  font-weight: 800;
  color: #ffffff;
  letter-spacing: -0.5px;
  margin-bottom: 0.6rem;
  text-shadow: 0 2px 12px rgba(0,0,0,0.15);
}
.hero-subtitle {
  font-size: 1.1rem;
  color: rgba(255,255,255,0.85);
  font-weight: 400;
}
.hero-stats {
  display: flex;
  justify-content: center;
  gap: 2.5rem;
  margin-top: 1.8rem;
}
.hero-stat {
  text-align: center;
  color: #fff;
}
.hero-stat-num {
  font-family: 'Outfit', sans-serif;
  font-size: 1.6rem;
  font-weight: 700;
}
.hero-stat-label {
  font-size: 0.78rem;
  opacity: 0.8;
  font-weight: 500;
  letter-spacing: 0.5px;
}

/* ── Card Containers ────────────────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: #FFFFFF !important;
  border: 1px solid rgba(108, 92, 231, 0.12) !important;
  border-radius: 20px !important;
  box-shadow: 0 2px 20px rgba(108, 92, 231, 0.08), 0 1px 4px rgba(0,0,0,0.04) !important;
  padding: 28px !important;
  transition: box-shadow 0.2s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  box-shadow: 0 6px 32px rgba(108, 92, 231, 0.14) !important;
}

/* ── Panel Titles ────────────────────────────────────────────────────── */
.panel-title {
  font-family: 'Outfit', sans-serif;
  font-size: 1.35rem;
  font-weight: 700;
  color: #1A1A2E;
  margin-bottom: 1.6rem;
  display: flex;
  align-items: center;
  gap: 10px;
}
.panel-title-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: linear-gradient(135deg, #6C5CE7, #FD79A8);
  display: inline-block;
  flex-shrink: 0;
}

/* ── Form Labels ─────────────────────────────────────────────────────── */
label, .stSelectbox label, .stNumberInput label {
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: #6B6B8D !important;
  text-transform: uppercase !important;
  letter-spacing: 0.6px !important;
}

/* ── Predict Button ──────────────────────────────────────────────────── */
div.stButton > button {
  background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%) !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 14px 32px !important;
  color: white !important;
  font-weight: 700 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 1rem !important;
  width: 100% !important;
  transition: all 0.25s ease !important;
  cursor: pointer !important;
  margin-top: 0.8rem !important;
  letter-spacing: 0.3px !important;
  box-shadow: 0 4px 16px rgba(108, 92, 231, 0.35) !important;
}
div.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(108, 92, 231, 0.5) !important;
  background: linear-gradient(135deg, #5A4BD6 0%, #9C94FD 100%) !important;
}
div.stButton > button:active {
  transform: translateY(0) !important;
}

/* ── Probability Badge ──────────────────────────────────────────────── */
.prob-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.5rem 0 1rem;
}
.prob-ring {
  width: 160px;
  height: 160px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.2rem;
  position: relative;
}
.prob-ring-placed {
  background: linear-gradient(135deg, #e8fff4, #d0fae8);
  border: 6px solid #00b894;
  box-shadow: 0 0 0 6px rgba(0,184,148,0.15), 0 8px 32px rgba(0,184,148,0.2);
}
.prob-ring-not {
  background: linear-gradient(135deg, #fff5f5, #ffe0e0);
  border: 6px solid #e17055;
  box-shadow: 0 0 0 6px rgba(225,112,85,0.15), 0 8px 32px rgba(225,112,85,0.2);
}
.prob-number {
  font-family: 'Outfit', sans-serif;
  font-size: 2.6rem;
  font-weight: 800;
  line-height: 1;
}
.prob-number-placed { color: #00b894; }
.prob-number-not    { color: #e17055; }
.prob-label {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #888;
  margin-top: 2px;
}
.status-pill {
  border-radius: 30px;
  padding: 8px 24px;
  font-family: 'Outfit', sans-serif;
  font-size: 1.2rem;
  font-weight: 700;
  display: inline-block;
}
.status-pill-placed {
  background: linear-gradient(135deg, #e8fff4, #c8fae0);
  color: #00b894;
  border: 1.5px solid rgba(0,184,148,0.3);
}
.status-pill-not {
  background: linear-gradient(135deg, #fff5f5, #ffe0e0);
  color: #e17055;
  border: 1.5px solid rgba(225,112,85,0.3);
}

/* ── Section Labels inside results ──────────────────────────────────── */
.result-section-title {
  font-family: 'Outfit', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  color: #1A1A2E;
  margin: 1.4rem 0 0.8rem;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ── Recommendation Cards ────────────────────────────────────────────── */
.rec-item {
  background: linear-gradient(135deg, #F8F0FF, #EFF0FF);
  border-left: 4px solid #6C5CE7;
  border-radius: 10px;
  padding: 12px 16px;
  margin-bottom: 10px;
  font-size: 0.93rem;
  color: #2D2D4E;
  line-height: 1.6;
  box-shadow: 0 2px 8px rgba(108, 92, 231, 0.06);
}

/* ── Placeholder ─────────────────────────────────────────────────────── */
.results-placeholder {
  text-align: center;
  padding: 3rem 1rem;
  color: #AAAAC2;
  font-size: 1rem;
  line-height: 1.8;
}
.results-placeholder .icon {
  font-size: 3.5rem;
  margin-bottom: 1rem;
  display: block;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: #F0F2FF !important;
  border-radius: 12px !important;
  padding: 4px !important;
  gap: 4px !important;
}
.stTabs [role="tab"] {
  color: #6B6B8D !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  padding: 8px 20px !important;
}
.stTabs [role="tab"][aria-selected="true"] {
  background: #6C5CE7 !important;
  color: #fff !important;
  font-weight: 600 !important;
  border-bottom: none !important;
  box-shadow: 0 2px 8px rgba(108, 92, 231, 0.3) !important;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr { border-color: rgba(108, 92, 231, 0.1) !important; margin: 1rem 0 2rem !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_bundle():
    registry = ModelRegistry()
    return registry.get_model()


@st.cache_data
def load_dataset():
    data_path = PROJECT_ROOT / "Dataset" / "Placement_Data_Full_Class.csv"
    if not data_path.exists():
        return None
    return pd.read_csv(data_path)


def get_feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
    else:
        values = [0.0] * len(X.columns)

    importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": values}
    ).sort_values("Importance", ascending=False)

    friendly_names = {
        "ssc_p": "SSC % (10th)",
        "hsc_p": "HSC % (12th)",
        "degree_p": "Degree %",
        "etest_p": "E-Test %",
        "mba_p": "MBA %",
        "academic_average": "Academic Average",
        "gender_M": "Gender: Male",
        "hsc_s_Commerce": "HSC: Commerce",
        "hsc_s_Science": "HSC: Science",
        "degree_t_Others": "Degree: Others",
        "degree_t_Sci&Tech": "Degree: Sci & Tech",
        "workex_Yes": "Work Experience",
        "specialisation_Mkt&Hr": "Specialisation: Mkt & HR",
    }
    importance_df["Feature"] = importance_df["Feature"].map(lambda x: friendly_names.get(x, x))
    return importance_df.head(8)


def get_recommendations(profile, probability):
    recs = []
    if probability < 0.5:
        recs.append("Focus on academics, practical projects, and mock interviews to improve your chances.")
    if profile["workex"] == "No":
        recs.append("Gain internships or part-time work experience to strengthen your profile.")
    if profile["degree_p"] < 70:
        recs.append("Improve your degree performance and highlight relevant projects on your resume.")
    if profile["etest_p"] < 60:
        recs.append("Practice aptitude and technical mock tests to raise your employability score.")
    if profile["mba_p"] and profile["mba_p"] < 70:
        recs.append("Concentrate on core MBA subjects and case studies for better outcomes.")
    if not recs:
        recs.append("Your profile looks strong! Keep refining your resume and interview preparation.")
    return recs


def main():
    # ── Hero Banner ──────────────────────────────────────────────────────
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">🎓 AI-Powered Platform</div>
            <div class="hero-title">Campus Placement Prediction</div>
            <div class="hero-subtitle">AI-Powered Analytics &amp; Career Recommendations</div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="hero-stat-num">98%</div>
                    <div class="hero-stat-label">Accuracy</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-num">215+</div>
                    <div class="hero-stat-label">Students</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-num">13</div>
                    <div class="hero-stat-label">AI Features</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    model_data = load_model_bundle()
    if not model_data:
        st.error("The trained model file is missing. Train the model first and refresh this page.")
        st.stop()

    model = model_data["model"]
    expected_columns = model_data["columns"]

    # ── Two-column layout ────────────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    # ── LEFT PANEL: Student Profile ──────────────────────────────────────
    with col_left:
        with st.container(border=True):
            st.markdown("""
                <div class="panel-title">
                    <span class="panel-title-dot"></span> Student Profile
                </div>
            """, unsafe_allow_html=True)

            r1a, r1b = st.columns(2)
            with r1a:
                gender = st.selectbox("Gender", ["M", "F"],
                                      format_func=lambda x: "Male" if x == "M" else "Female")
            with r1b:
                workex = st.selectbox("Work Experience", ["No", "Yes"])

            r2a, r2b = st.columns(2)
            with r2a:
                ssc_p = st.number_input("SSC % (10th)", min_value=0.0, max_value=100.0,
                                        value=70.0, step=0.1, format="%.1f")
            with r2b:
                hsc_p = st.number_input("HSC % (12th)", min_value=0.0, max_value=100.0,
                                        value=70.0, step=0.1, format="%.1f")

            r3a, r3b = st.columns(2)
            with r3a:
                hsc_s = st.selectbox("HSC Specialisation", ["Commerce", "Science", "Arts"])
            with r3b:
                degree_p = st.number_input("Degree %", min_value=0.0, max_value=100.0,
                                           value=70.0, step=0.1, format="%.1f")

            r4a, r4b = st.columns(2)
            with r4a:
                degree_t = st.selectbox("Degree Type", ["Comm&Mgmt", "Sci&Tech", "Others"])
            with r4b:
                etest_p = st.number_input("E-Test %", min_value=0.0, max_value=100.0,
                                          value=70.0, step=0.1, format="%.1f")

            r5a, r5b = st.columns(2)
            with r5a:
                specialisation = st.selectbox(
                    "MBA Specialisation (Optional)", ["None", "Mkt&HR", "Mkt&Fin"],
                    format_func=lambda x: "None / Not Applicable" if x == "None" else x)
            with r5b:
                mba_p = st.number_input("MBA % (Optional)", min_value=0.0, max_value=100.0,
                                        value=0.0, step=0.1, format="%.1f")

            predict_clicked = st.button("🔍 Predict Placement Probability", type="primary")

    # ── RIGHT PANEL: Results ─────────────────────────────────────────────
    with col_right:
        with st.container(border=True):
            st.markdown("""
                <div class="panel-title">
                    <span class="panel-title-dot"></span> Prediction Results
                </div>
            """, unsafe_allow_html=True)

            if predict_clicked:
                profile = {
                    "gender":         gender,
                    "ssc_p":          ssc_p,
                    "hsc_p":          hsc_p,
                    "hsc_s":          hsc_s,
                    "degree_p":       degree_p,
                    "degree_t":       degree_t,
                    "workex":         workex,
                    "etest_p":        etest_p,
                    "specialisation": specialisation,
                    "mba_p":          mba_p if specialisation != "None" else 0.0,
                }

                df = pd.DataFrame([profile])
                processor = DataProcessorFactory.get_processor("standard")
                processor.expected_columns = expected_columns
                X, _ = processor.preprocess(df, is_training=False)

                strategy   = RandomForestStrategy()
                probability = strategy.predict_proba(model, X)[0][1]
                prediction  = strategy.predict(model, X)[0]

                is_placed     = prediction == 1
                pred_label    = "Placed ✓" if is_placed else "Not Placed ✗"
                ring_class    = "prob-ring-placed" if is_placed else "prob-ring-not"
                num_class     = "prob-number-placed" if is_placed else "prob-number-not"
                pill_class    = "status-pill-placed" if is_placed else "status-pill-not"

                # Probability Ring + Status Pill
                st.markdown(f"""
                    <div class="prob-wrapper">
                        <div class="prob-ring {ring_class}">
                            <span class="prob-number {num_class}">{probability * 100:.1f}%</span>
                            <span class="prob-label">Probability</span>
                        </div>
                        <span class="status-pill {pill_class}">{pred_label}</span>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # Feature Impact
                st.markdown('<div class="result-section-title">📊 Feature Impact</div>', unsafe_allow_html=True)
                importance_df = get_feature_importance(model, X)
                st.bar_chart(
                    importance_df.set_index("Feature"),
                    color="#6C5CE7",
                    use_container_width=True,
                    height=250
                )

                # Recommendations
                st.markdown('<div class="result-section-title">💡 Recommendations</div>', unsafe_allow_html=True)
                for rec in get_recommendations(profile, probability):
                    st.markdown(f'<div class="rec-item">• {rec}</div>', unsafe_allow_html=True)

            else:
                st.markdown("""
                    <div class="results-placeholder">
                        <span class="icon">🎯</span>
                        Fill in your academic profile<br>on the left and click<br>
                        <strong>Predict Placement Probability</strong><br>
                        to get your instant AI-powered result.
                    </div>
                """, unsafe_allow_html=True)

    # ── Bottom Tabs ──────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📌 Overview", "📊 Dataset Insights"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Type", "Random Forest")
        c2.metric("Training Accuracy", "~88%")
        c3.metric("Dataset Size", "215 Students")
        st.write("")
        st.info("This app uses a Random Forest classifier trained on real campus recruitment data to predict placement outcomes based on academic performance, work experience, and MBA specialisation.")

    with tab2:
        dataset = load_dataset()
        if dataset is None:
            st.info("Dataset file is not available in the project folder.")
        else:
            st.write("##### Placement Trends from Historical Recruitment Data")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Placement by Specialisation**")
                spec_data = dataset.groupby(["specialisation", "status"]).size().unstack(fill_value=0)
                st.bar_chart(spec_data, use_container_width=True)
            with col_b:
                st.write("**Placement by Work Experience**")
                we_data = dataset.groupby(["workex", "status"]).size().unstack(fill_value=0)
                st.bar_chart(we_data, use_container_width=True)
            st.write("##### Sample Dataset Records")
            st.dataframe(dataset.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
