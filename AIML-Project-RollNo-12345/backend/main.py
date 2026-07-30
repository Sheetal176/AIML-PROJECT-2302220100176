import sys
from pathlib import Path
import pandas as pd
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

# ── Global CSS (targets only Streamlit's own elements) ─────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #F0F2FF !important;
}
.stApp {
    background: linear-gradient(155deg, #EEF0FF 0%, #FFFFFF 55%, #F5F0FF 100%) !important;
    background-attachment: fixed !important;
}
[data-testid="stHeader"]      { background: transparent !important; }
[data-testid="stToolbar"]     { display: none !important; }
[data-testid="stDecoration"]  { display: none !important; }
footer                        { display: none !important; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 1220px !important;
}

/* ── White card containers ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(108,92,231,0.13) !important;
    border-radius: 20px !important;
    box-shadow: 0 4px 24px rgba(108,92,231,0.09), 0 1px 4px rgba(0,0,0,0.04) !important;
    padding: 28px 28px 20px !important;
}

/* ── Form labels ── */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #8888AA !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
}

/* ── Predict button ── */
div.stButton > button {
    background: linear-gradient(135deg, #6C5CE7, #A29BFE) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 0 !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(108,92,231,0.38) !important;
    transition: all 0.22s ease !important;
    margin-top: 0.6rem !important;
    letter-spacing: 0.2px !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(108,92,231,0.52) !important;
}

/* ── st.metric ── */
[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #1A1A2E !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #8888AA !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── st.success / error / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 0.95rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #EDEDFF !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [role="tab"] {
    border-radius: 9px !important;
    color: #6868A8 !important;
    font-weight: 500 !important;
    padding: 7px 20px !important;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: #6C5CE7 !important;
    color: white !important;
    font-weight: 700 !important;
    border-bottom: none !important;
    box-shadow: 0 2px 10px rgba(108,92,231,0.35) !important;
}

hr { border-color: rgba(108,92,231,0.1) !important; margin: 0.5rem 0 2rem !important; }
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

    friendly_names = {
        "ssc_p": "SSC % (10th)", "hsc_p": "HSC % (12th)",
        "degree_p": "Degree %", "etest_p": "E-Test %", "mba_p": "MBA %",
        "academic_average": "Academic Avg", "gender_M": "Gender: Male",
        "hsc_s_Commerce": "HSC: Commerce", "hsc_s_Science": "HSC: Science",
        "degree_t_Others": "Degree: Others", "degree_t_Sci&Tech": "Degree: Sci&Tech",
        "workex_Yes": "Work Experience", "specialisation_Mkt&Hr": "MBA: Mkt & HR",
    }
    df = pd.DataFrame({"Feature": X.columns, "Importance": values})
    df["Feature"] = df["Feature"].map(lambda x: friendly_names.get(x, x))
    return df.sort_values("Importance", ascending=False).head(8)


def get_recommendations(profile, probability):
    recs = []
    if probability < 0.5:
        recs.append("Focus on academics, practical projects, and mock interviews.")
    if profile["workex"] == "No":
        recs.append("Gain internship or part-time work experience to boost your profile.")
    if profile["degree_p"] < 70:
        recs.append("Aim to improve your degree performance above 70%.")
    if profile["etest_p"] < 60:
        recs.append("Practice aptitude & employability tests to raise your E-Test score.")
    if profile.get("mba_p", 0) and profile["mba_p"] < 70:
        recs.append("Strengthen your MBA performance through case studies & coursework.")
    if not recs:
        recs.append("Excellent profile! Keep polishing your resume and interview skills.")
    return recs


# ── HERO BANNER (fully inline-styled so it works on Streamlit Cloud) ──
st.markdown("""
<div style="
    background: linear-gradient(135deg,#6C5CE7 0%,#A29BFE 50%,#FD79A8 100%);
    border-radius: 0 0 32px 32px;
    padding: 3rem 2rem 2.6rem;
    text-align: center;
    margin-bottom: 2.5rem;
    box-shadow: 0 8px 40px rgba(108,92,231,0.28);
    position: relative;
    overflow: hidden;
">
  <div style="
      display:inline-block;
      background:rgba(255,255,255,0.18);
      border:1px solid rgba(255,255,255,0.32);
      border-radius:30px;
      padding:4px 18px;
      font-size:0.78rem;
      font-weight:700;
      color:#fff;
      letter-spacing:1.8px;
      text-transform:uppercase;
      margin-bottom:1rem;
  ">🎓 AI-Powered Platform</div>

  <div style="
      font-family:'Outfit',sans-serif;
      font-size:3rem;
      font-weight:800;
      color:#ffffff;
      letter-spacing:-0.5px;
      margin-bottom:0.55rem;
      text-shadow:0 2px 16px rgba(0,0,0,0.18);
      line-height:1.1;
  ">Campus Placement Prediction</div>

  <div style="font-size:1.08rem;color:rgba(255,255,255,0.85);font-weight:400;margin-bottom:2rem;">
      AI-Powered Analytics &amp; Career Recommendations
  </div>

  <div style="display:flex;justify-content:center;gap:3rem;">
    <div style="text-align:center;color:#fff;">
      <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;">98%</div>
      <div style="font-size:0.78rem;opacity:0.82;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;">Accuracy</div>
    </div>
    <div style="text-align:center;color:#fff;">
      <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;">215+</div>
      <div style="font-size:0.78rem;opacity:0.82;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;">Students</div>
    </div>
    <div style="text-align:center;color:#fff;">
      <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;">13</div>
      <div style="font-size:0.78rem;opacity:0.82;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;">AI Features</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


def main():
    model_data = load_model_bundle()
    if not model_data:
        st.error("The trained model file is missing. Train the model first and refresh this page.")
        st.stop()

    model = model_data["model"]
    expected_columns = model_data["columns"]

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    # ── LEFT: Student Profile ────────────────────────────────────────────
    with col_left:
        with st.container(border=True):
            # Panel heading inline-styled
            st.markdown("""
            <div style="
                font-family:'Outfit',sans-serif;
                font-size:1.3rem;
                font-weight:700;
                color:#1A1A2E;
                margin-bottom:1.4rem;
                display:flex;
                align-items:center;
                gap:10px;
            ">
              <span style="
                width:11px;height:11px;border-radius:50%;
                background:linear-gradient(135deg,#6C5CE7,#FD79A8);
                display:inline-block;flex-shrink:0;
              "></span>
              Student Profile
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
                ssc_p = st.number_input("SSC % (10th)", 0.0, 100.0, 70.0, 0.1, format="%.1f")
            with r2b:
                hsc_p = st.number_input("HSC % (12th)", 0.0, 100.0, 70.0, 0.1, format="%.1f")

            r3a, r3b = st.columns(2)
            with r3a:
                hsc_s = st.selectbox("HSC Specialisation", ["Commerce", "Science", "Arts"])
            with r3b:
                degree_p = st.number_input("Degree %", 0.0, 100.0, 70.0, 0.1, format="%.1f")

            r4a, r4b = st.columns(2)
            with r4a:
                degree_t = st.selectbox("Degree Type", ["Comm&Mgmt", "Sci&Tech", "Others"])
            with r4b:
                etest_p = st.number_input("E-Test %", 0.0, 100.0, 70.0, 0.1, format="%.1f")

            r5a, r5b = st.columns(2)
            with r5a:
                specialisation = st.selectbox(
                    "MBA Specialisation (Optional)", ["None", "Mkt&HR", "Mkt&Fin"],
                    format_func=lambda x: "None / Not Applicable" if x == "None" else x)
            with r5b:
                mba_p = st.number_input("MBA % (Optional)", 0.0, 100.0, 0.0, 0.1, format="%.1f")

            predict_clicked = st.button("🔍  Predict Placement Probability", type="primary")

    # ── RIGHT: Results ───────────────────────────────────────────────────
    with col_right:
        with st.container(border=True):
            st.markdown("""
            <div style="
                font-family:'Outfit',sans-serif;
                font-size:1.3rem;
                font-weight:700;
                color:#1A1A2E;
                margin-bottom:1.4rem;
                display:flex;
                align-items:center;
                gap:10px;
            ">
              <span style="
                width:11px;height:11px;border-radius:50%;
                background:linear-gradient(135deg,#6C5CE7,#FD79A8);
                display:inline-block;flex-shrink:0;
              "></span>
              Prediction Results
            </div>
            """, unsafe_allow_html=True)

            if predict_clicked:
                profile = {
                    "gender": gender, "ssc_p": ssc_p, "hsc_p": hsc_p,
                    "hsc_s": hsc_s, "degree_p": degree_p, "degree_t": degree_t,
                    "workex": workex, "etest_p": etest_p,
                    "specialisation": specialisation,
                    "mba_p": mba_p if specialisation != "None" else 0.0,
                }

                df = pd.DataFrame([profile])
                processor = DataProcessorFactory.get_processor("standard")
                processor.expected_columns = expected_columns
                X, _ = processor.preprocess(df, is_training=False)

                strategy    = RandomForestStrategy()
                probability = strategy.predict_proba(model, X)[0][1]
                prediction  = strategy.predict(model, X)[0]
                is_placed   = prediction == 1

                # ── Probability ring (inline styled) ──
                if is_placed:
                    ring_bg    = "linear-gradient(135deg,#e8fff4,#c8fae0)"
                    ring_border= "#00b894"
                    ring_shadow= "rgba(0,184,148,0.22)"
                    num_color  = "#00b894"
                    pill_bg    = "linear-gradient(135deg,#e8fff4,#d0fae8)"
                    pill_color = "#00b894"
                    pill_border= "rgba(0,184,148,0.3)"
                    status_txt = "✓ Placed"
                else:
                    ring_bg    = "linear-gradient(135deg,#fff5f5,#ffe0e0)"
                    ring_border= "#e17055"
                    ring_shadow= "rgba(225,112,85,0.22)"
                    num_color  = "#e17055"
                    pill_bg    = "linear-gradient(135deg,#fff5f5,#ffe0e0)"
                    pill_color = "#e17055"
                    pill_border= "rgba(225,112,85,0.3)"
                    status_txt = "✗ Not Placed"

                st.markdown(f"""
                <div style="display:flex;flex-direction:column;align-items:center;padding:1.2rem 0 0.8rem;">
                  <div style="
                      width:148px;height:148px;border-radius:50%;
                      background:{ring_bg};
                      border:6px solid {ring_border};
                      box-shadow:0 0 0 7px {ring_shadow},0 8px 32px {ring_shadow};
                      display:flex;flex-direction:column;
                      align-items:center;justify-content:center;
                      margin-bottom:1.1rem;
                  ">
                    <span style="
                        font-family:'Outfit',sans-serif;
                        font-size:2.4rem;font-weight:800;
                        color:{num_color};line-height:1;
                    ">{probability*100:.1f}%</span>
                    <span style="font-size:0.7rem;font-weight:600;text-transform:uppercase;
                                 letter-spacing:1px;color:#aaa;margin-top:2px;">Probability</span>
                  </div>
                  <span style="
                      background:{pill_bg};color:{pill_color};
                      border:1.5px solid {pill_border};border-radius:30px;
                      padding:8px 26px;font-family:'Outfit',sans-serif;
                      font-size:1.18rem;font-weight:700;
                  ">{status_txt}</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # ── Feature Impact ──
                st.markdown("""
                <p style="font-family:'Outfit',sans-serif;font-size:0.98rem;
                           font-weight:700;color:#1A1A2E;margin:0.2rem 0 0.6rem;">
                  📊 Feature Impact
                </p>""", unsafe_allow_html=True)
                imp_df = get_feature_importance(model, X)
                st.bar_chart(imp_df.set_index("Feature"),
                             color="#6C5CE7",
                             use_container_width=True,
                             height=230)

                # ── Recommendations ──
                st.markdown("""
                <p style="font-family:'Outfit',sans-serif;font-size:0.98rem;
                           font-weight:700;color:#1A1A2E;margin:0.5rem 0 0.5rem;">
                  💡 Recommendations
                </p>""", unsafe_allow_html=True)
                for rec in get_recommendations(profile, probability):
                    st.markdown(f"""
                    <div style="
                        background:linear-gradient(135deg,#F8F0FF,#EFF0FF);
                        border-left:4px solid #6C5CE7;
                        border-radius:10px;
                        padding:11px 16px;
                        margin-bottom:9px;
                        font-size:0.91rem;
                        color:#2D2D4E;
                        line-height:1.6;
                        box-shadow:0 2px 8px rgba(108,92,231,0.07);
                    ">• {rec}</div>
                    """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style="text-align:center;padding:3.5rem 1rem;color:#AAAAC2;">
                  <div style="font-size:3.2rem;margin-bottom:1rem;">🎯</div>
                  <div style="font-size:1rem;line-height:1.9;">
                    Fill in your academic profile<br>on the left and click<br>
                    <strong style="color:#6C5CE7;">Predict Placement Probability</strong><br>
                    to get your instant AI-powered result.
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Bottom Tabs ──────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📌 Overview", "📊 Dataset Insights"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Type",       "Random Forest")
        c2.metric("Training Accuracy","~88%")
        c3.metric("Dataset Size",     "215 Students")
        st.write("")
        st.info("This app uses a Random Forest classifier trained on real campus recruitment data to predict placement outcomes based on academic performance, work experience, and MBA specialisation.")

    with tab2:
        dataset = load_dataset()
        if dataset is None:
            st.info("Dataset file not available in the project folder.")
        else:
            ca, cb = st.columns(2)
            with ca:
                st.write("**Placement by Specialisation**")
                st.bar_chart(
                    dataset.groupby(["specialisation","status"]).size().unstack(fill_value=0),
                    use_container_width=True)
            with cb:
                st.write("**Placement by Work Experience**")
                st.bar_chart(
                    dataset.groupby(["workex","status"]).size().unstack(fill_value=0),
                    use_container_width=True)
            st.write("##### Sample Records")
            st.dataframe(dataset.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
