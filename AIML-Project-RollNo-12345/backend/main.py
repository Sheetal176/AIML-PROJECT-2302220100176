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

# ── Custom CSS for exact React/Vite replica (Glassmorphism & styling) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;700&display=swap');

/* ─── Global Styling ────────────────────────────────────────────── */
.stApp {
  background: linear-gradient(135deg, #0F2027, #203A43, #2C5364) !important;
  background-attachment: fixed !important;
  font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit elements */
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stBottom"] { background: transparent !important; }

/* Center block container */
.block-container {
  padding-top: 2rem !important;
  padding-bottom: 2rem !important;
  max-width: 1200px !important;
}

/* ─── Site Header ────────────────────────────────────────────────── */
.site-header {
  text-align: center;
  padding: 1rem 0 2.5rem 0;
}
.site-title {
  font-family: 'Outfit', sans-serif;
  font-size: 3.2rem;
  font-weight: 700;
  background: linear-gradient(45deg, #FD79A8, #A29BFE);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.5rem;
  line-height: 1.15;
}
.site-subtitle {
  font-size: 1.1rem;
  color: rgba(248, 249, 250, 0.75);
}

/* ─── Native st.container (border=True) Glassmorphism styling ─── */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255, 255, 255, 0.06) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  border-radius: 16px !important;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
  padding: 24px !important;
  margin-bottom: 1.5rem !important;
}

.panel-title {
  font-family: 'Outfit', sans-serif;
  font-size: 1.6rem;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding-bottom: 0.5rem;
}

/* ─── Inputs and Selectors ───────────────────────────────────────── */
label {
  font-weight: 600 !important;
  color: #cccccc !important;
}

/* ─── Primary Button (.btn-primary) ───────────────────────────────── */
div.stButton > button {
  background: linear-gradient(90deg, #6C5CE7, #A29BFE) !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 12px 24px !important;
  color: white !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 1.05rem !important;
  width: 100% !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
  cursor: pointer !important;
  margin-top: 1rem !important;
}
div.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 4px 15px rgba(108, 92, 231, 0.5) !important;
  color: white !important;
}

/* ─── Probability Badge / Circle ─────────────────────────────────── */
.prob-badge-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 1.5rem 0;
}
.prob-circle {
  width: 140px;
  height: 140px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.2rem;
  font-weight: 700;
  color: #ffffff;
  font-family: 'Outfit', sans-serif;
  margin-bottom: 1rem;
}
.prob-circle-placed {
  border: 8px solid #2ecc71;
  box-shadow: 0 0 24px rgba(46, 204, 113, 0.4);
  background: rgba(46, 204, 113, 0.05);
}
.prob-circle-not {
  border: 8px solid #e74c3c;
  box-shadow: 0 0 24px rgba(231, 76, 60, 0.4);
  background: rgba(231, 76, 60, 0.05);
}
.status-text {
  font-family: 'Outfit', sans-serif;
  font-size: 1.6rem;
  font-weight: 700;
}
.status-placed { color: #2ecc71; }
.status-not-placed { color: #e74c3c; }

/* ─── Actionable Recommendations ─────────────────────────────────── */
.rec-item {
  background: rgba(255, 255, 255, 0.05);
  border-left: 4px solid #A29BFE;
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 12px;
  font-size: 0.95rem;
  color: #F8F9FA;
  line-height: 1.5;
}

/* ─── Results Placeholder ────────────────────────────────────────── */
.results-placeholder {
  text-align: center;
  margin: 4rem 0;
  color: rgba(255, 255, 255, 0.5);
  font-size: 1.05rem;
  line-height: 1.6;
}

/* ─── Tabs styling ───────────────────────────────────────────────── */
.stTabs [role="tab"] {
  color: rgba(255, 255, 255, 0.6) !important;
  font-size: 1rem !important;
}
.stTabs [role="tab"][aria-selected="true"] {
  color: #FD79A8 !important;
  border-bottom: 2px solid #FD79A8 !important;
}
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
        {"feature": X.columns, "importance": values}
    ).sort_values("importance", ascending=True)

    # Friendly names for features
    friendly_names = {
        "ssc_p": "SSC % (10th)",
        "hsc_p": "HSC % (12th)",
        "degree_p": "Degree %",
        "etest_p": "E-Test %",
        "mba_p": "MBA %",
        "academic_average": "Academic Avg",
        "gender_M": "Gender: Male",
        "hsc_s_Commerce": "HSC: Commerce",
        "hsc_s_Science": "HSC: Science",
        "degree_t_Others": "Degree: Others",
        "degree_t_Sci&Tech": "Degree: Sci & Tech",
        "workex_Yes": "Work Experience",
        "specialisation_Mkt&Hr": "MBA Spec: Mkt & HR",
        "specialisation_Mkt&Fin": "MBA Spec: Mkt & Fin"
    }
    importance_df["feature"] = importance_df["feature"].map(lambda x: friendly_names.get(x, x))
    return importance_df.tail(8)


def get_recommendations(profile, probability):
    recommendations = []
    if probability < 0.5:
        recommendations.append("Focus on academics, practical projects, and mock interviews to improve your chances.")
    if profile["workex"] == "No":
        recommendations.append("Gain internships or part-time work experience to strengthen your profile.")
    if profile["degree_p"] < 70:
        recommendations.append("Improve your degree performance and highlight relevant projects on your resume.")
    if profile["etest_p"] < 60:
        recommendations.append("Practice aptitude and technical mock tests to raise your employability score.")
    if profile["mba_p"] and profile["mba_p"] < 70:
        recommendations.append("Concentrate on core MBA subjects and case studies for better outcomes.")
    if not recommendations:
        recommendations.append("Your profile looks strong. Keep refining your resume and interview preparation.")
    return recommendations


def main():
    # ── Site Header ──────────────────────────────────────────────────────────
    st.markdown("""
        <div class="site-header">
            <div class="site-title">Campus Placement Prediction</div>
            <div class="site-subtitle">AI-Powered Analytics &amp; Career Recommendations</div>
        </div>
    """, unsafe_allow_html=True)

    model_data = load_model_bundle()
    if not model_data:
        st.error("The trained model file is missing. Train the model first and refresh this page.")
        st.stop()

    model = model_data["model"]
    expected_columns = model_data["columns"]

    # ── Two-column layout ───────────────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    # ── LEFT PANEL: Student Profile ──────────────────────────────────────────
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Student Profile</div>', unsafe_allow_html=True)

            r1a, r1b = st.columns(2)
            with r1a:
                gender = st.selectbox("Gender", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
            with r1b:
                workex = st.selectbox("Work Experience", ["No", "Yes"])

            r2a, r2b = st.columns(2)
            with r2a:
                ssc_p = st.number_input("SSC % (10th)", min_value=0.0, max_value=100.0, value=70.0, step=0.1, format="%.1f")
            with r2b:
                hsc_p = st.number_input("HSC % (12th)", min_value=0.0, max_value=100.0, value=70.0, step=0.1, format="%.1f")

            r3a, r3b = st.columns(2)
            with r3a:
                hsc_s = st.selectbox("HSC Specialisation", ["Commerce", "Science", "Arts"])
            with r3b:
                degree_p = st.number_input("Degree %", min_value=0.0, max_value=100.0, value=70.0, step=0.1, format="%.1f")

            r4a, r4b = st.columns(2)
            with r4a:
                degree_t = st.selectbox("Degree Type", ["Comm&Mgmt", "Sci&Tech", "Others"])
            with r4b:
                etest_p = st.number_input("E-Test %", min_value=0.0, max_value=100.0, value=70.0, step=0.1, format="%.1f")

            r5a, r5b = st.columns(2)
            with r5a:
                specialisation = st.selectbox("MBA Specialisation (Optional)", ["None", "Mkt&HR", "Mkt&Fin"],
                                              format_func=lambda x: "None / Not Applicable" if x == "None" else x)
            with r5b:
                mba_p = st.number_input("MBA % (Optional)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")

            predict_clicked = st.button("Predict Placement Probability", type="primary")

    # ── RIGHT PANEL: Prediction Results ──────────────────────────────────────
    with col_right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Prediction Results</div>', unsafe_allow_html=True)

            if predict_clicked:
                profile = {
                    "gender":        gender,
                    "ssc_p":         ssc_p,
                    "hsc_p":         hsc_p,
                    "hsc_s":         hsc_s,
                    "degree_p":      degree_p,
                    "degree_t":      degree_t,
                    "workex":        workex,
                    "etest_p":       etest_p,
                    "specialisation": specialisation,
                    "mba_p":         mba_p if specialisation != "None" else 0.0,
                }

                df = pd.DataFrame([profile])
                processor = DataProcessorFactory.get_processor("standard")
                processor.expected_columns = expected_columns
                X, _ = processor.preprocess(df, is_training=False)

                strategy = RandomForestStrategy()
                probability = strategy.predict_proba(model, X)[0][1]
                prediction  = strategy.predict(model, X)[0]

                prediction_label  = "Placed" if prediction == 1 else "Not Placed"
                circle_class = "prob-circle-placed" if prediction == 1 else "prob-circle-not"
                status_class = "status-placed"      if prediction == 1 else "status-not-placed"

                # Status + Circular Badge
                st.markdown(f"""
                    <div class="prob-badge-container">
                        <div class="prob-circle {circle_class}">{probability * 100:.1f}%</div>
                        <div class="status-text {status_class}">Status: {prediction_label}</div>
                    </div>
                """, unsafe_allow_html=True)

                # Feature Impact chart
                st.markdown("#### 📊 Explainable AI (Feature Impact)")
                importance_df = get_feature_importance(model, X)
                
                # Render native interactive Streamlit horizontal bar chart matching the page style
                st.bar_chart(
                    importance_df, 
                    x="feature", 
                    y="importance", 
                    color="#FD79A8",
                    use_container_width=True
                )

                # Recommendations
                st.markdown("#### 💡 Actionable Recommendations")
                for rec in get_recommendations(profile, probability):
                    st.markdown(f'<div class="rec-item">• {rec}</div>', unsafe_allow_html=True)

            else:
                st.markdown("""
                    <div class="results-placeholder">
                        Enter your profile details and click<br>
                        <strong>Predict Placement Probability</strong><br>
                        to see your placement probability and AI recommendations.
                    </div>
                """, unsafe_allow_html=True)

    # ── Bottom Tabs ──────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📌 Overview", "📊 Dataset Insights"])

    with tab1:
        st.write("This application combines machine learning analytics with interactive career recommendations to evaluate student employability.")
        st.write("It uses Random Forest classification trained on historical campus recruitment data.")

    with tab2:
        dataset = load_dataset()
        if dataset is None:
            st.info("Dataset file is not available in the project folder.")
        else:
            st.write("##### Placement Trends from Historical Recruitment Data")
            st.dataframe(dataset.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
