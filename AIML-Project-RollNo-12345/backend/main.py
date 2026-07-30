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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

/* ── Reset ─────────────────────────────────────────────────────── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #F4F5FF !important;
}
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stToolbar"],
footer { display: none !important; }

/* ── Full-width, zero wasted space ──────────────────────────────── */
.block-container {
    padding: 0 !important;
    margin: 0 auto !important;
    max-width: 100% !important;
}
section[data-testid="stMain"] > div {
    padding: 0 !important;
}

/* ── Card panels ─────────────────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(108,92,231,0.14) !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 16px rgba(108,92,231,0.09), 0 1px 3px rgba(0,0,0,0.05) !important;
    padding: 20px 22px 16px !important;
}

/* ── Compact field labels ─────────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    color: #8888AA !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    margin-bottom: 0 !important;
}

/* Tighter input row gaps */
[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
    gap: 10px !important;
}
[data-testid="element-container"] { margin-bottom: 0 !important; }

/* ── Predict button ──────────────────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #6C5CE7, #A29BFE) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 11px 0 !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.97rem !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(108,92,231,0.36) !important;
    transition: all 0.2s ease !important;
    margin-top: 0.4rem !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(108,92,231,0.5) !important;
}

/* ── Metric cards ────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #1A1A2E !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #8888AA !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #EDEDFF !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 3px !important;
}
.stTabs [role="tab"] {
    border-radius: 8px !important;
    color: #6868A8 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 6px 18px !important;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: #6C5CE7 !important;
    color: white !important;
    font-weight: 700 !important;
    border-bottom: none !important;
    box-shadow: 0 2px 8px rgba(108,92,231,0.3) !important;
}
hr { border-color: rgba(108,92,231,0.1) !important; margin: 0.2rem 0 1rem !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_bundle():
    return ModelRegistry().get_model()


@st.cache_data
def load_dataset():
    p = PROJECT_ROOT / "Dataset" / "Placement_Data_Full_Class.csv"
    return pd.read_csv(p) if p.exists() else None


def get_feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = abs(model.coef_[0])
    else:
        vals = [0.0] * len(X.columns)
    names = {
        "ssc_p": "SSC %", "hsc_p": "HSC %", "degree_p": "Degree %",
        "etest_p": "E-Test %", "mba_p": "MBA %", "academic_average": "Acad. Avg",
        "gender_M": "Male", "hsc_s_Commerce": "Commerce", "hsc_s_Science": "Science",
        "degree_t_Others": "Deg: Others", "degree_t_Sci&Tech": "Sci&Tech",
        "workex_Yes": "Work Exp", "specialisation_Mkt&Hr": "Mkt&HR",
    }
    df = pd.DataFrame({"Feature": X.columns, "Importance": vals})
    df["Feature"] = df["Feature"].map(lambda x: names.get(x, x))
    return df.sort_values("Importance", ascending=False).head(7)


def get_recommendations(profile, prob):
    r = []
    if prob < 0.5:
        r.append("Work on projects and attend mock interviews to improve your chances.")
    if profile["workex"] == "No":
        r.append("Gain internship or part-time experience to strengthen your profile.")
    if profile["degree_p"] < 70:
        r.append("Aim to improve your degree performance above 70%.")
    if profile["etest_p"] < 60:
        r.append("Practice aptitude tests to raise your employability score.")
    if profile.get("mba_p", 0) and profile["mba_p"] < 70:
        r.append("Focus on core MBA subjects and case-study preparation.")
    if not r:
        r.append("Great profile! Keep polishing your resume and interview skills.")
    return r


# ── COMPACT HERO ──────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(120deg, #6C5CE7 0%, #A29BFE 55%, #FD79A8 100%);
  padding: 1.6rem 2.5rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0;
  box-shadow: 0 4px 24px rgba(108,92,231,0.25);
">
  <div>
    <div style="
      font-size: 0.7rem; font-weight: 700; letter-spacing: 2px;
      text-transform: uppercase; color: rgba(255,255,255,0.75); margin-bottom: 0.3rem;
    ">🎓 AI-Powered Platform</div>
    <div style="
      font-family:'Outfit',sans-serif; font-size: 1.9rem; font-weight: 800;
      color: #fff; line-height: 1.1; letter-spacing: -0.5px;
    ">Campus Placement Prediction</div>
    <div style="font-size: 0.88rem; color: rgba(255,255,255,0.8); margin-top: 0.25rem;">
      AI-Powered Analytics &amp; Career Recommendations
    </div>
  </div>
  <div style="display:flex; gap: 2.5rem; align-items: center;">
    <div style="text-align:center; color:#fff;">
      <div style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;line-height:1;">98%</div>
      <div style="font-size:0.68rem;opacity:0.78;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;">Accuracy</div>
    </div>
    <div style="text-align:center; color:#fff;">
      <div style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;line-height:1;">215+</div>
      <div style="font-size:0.68rem;opacity:0.78;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;">Students</div>
    </div>
    <div style="text-align:center; color:#fff;">
      <div style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;line-height:1;">13</div>
      <div style="font-size:0.68rem;opacity:0.78;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;">Features</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN CONTENT ──────────────────────────────────────────────────────
# Wrap in padded container
st.markdown('<div style="padding: 1.2rem 1.8rem 1rem;">', unsafe_allow_html=True)


def main():
    model_data = load_model_bundle()
    if not model_data:
        st.error("Model file missing. Please train the model first.")
        st.stop()

    model            = model_data["model"]
    expected_columns = model_data["columns"]

    # Equal 50/50 columns for proper screen fit
    col_left, col_right = st.columns([1, 1], gap="medium")

    # ── LEFT: Student Profile ────────────────────────────────────────
    with col_left:
        with st.container(border=True):
            st.markdown("""
            <div style="font-family:'Outfit',sans-serif;font-size:1.15rem;font-weight:700;
                        color:#1A1A2E;margin-bottom:1rem;display:flex;align-items:center;gap:8px;">
              <span style="width:10px;height:10px;border-radius:50%;flex-shrink:0;
                           background:linear-gradient(135deg,#6C5CE7,#FD79A8);display:inline-block;">
              </span> Student Profile
            </div>
            """, unsafe_allow_html=True)

            a1, a2 = st.columns(2)
            with a1:
                gender = st.selectbox("Gender", ["M", "F"],
                                      format_func=lambda x: "Male" if x=="M" else "Female")
            with a2:
                workex = st.selectbox("Work Experience", ["No", "Yes"])

            b1, b2 = st.columns(2)
            with b1:
                ssc_p = st.number_input("SSC % (10th)", 0.0, 100.0, 70.0, 0.5, format="%.1f")
            with b2:
                hsc_p = st.number_input("HSC % (12th)", 0.0, 100.0, 70.0, 0.5, format="%.1f")

            c1, c2 = st.columns(2)
            with c1:
                hsc_s = st.selectbox("HSC Specialisation", ["Commerce", "Science", "Arts"])
            with c2:
                degree_p = st.number_input("Degree %", 0.0, 100.0, 70.0, 0.5, format="%.1f")

            d1, d2 = st.columns(2)
            with d1:
                degree_t = st.selectbox("Degree Type", ["Comm&Mgmt", "Sci&Tech", "Others"])
            with d2:
                etest_p = st.number_input("E-Test %", 0.0, 100.0, 70.0, 0.5, format="%.1f")

            e1, e2 = st.columns(2)
            with e1:
                specialisation = st.selectbox(
                    "MBA Specialisation", ["None", "Mkt&HR", "Mkt&Fin"],
                    format_func=lambda x: "Not Applicable" if x=="None" else x)
            with e2:
                mba_p = st.number_input("MBA %", 0.0, 100.0, 0.0, 0.5, format="%.1f")

            predict_clicked = st.button("🔍  Predict Placement Probability", type="primary")

    # ── RIGHT: Prediction Results ────────────────────────────────────
    with col_right:
        with st.container(border=True):
            st.markdown("""
            <div style="font-family:'Outfit',sans-serif;font-size:1.15rem;font-weight:700;
                        color:#1A1A2E;margin-bottom:1rem;display:flex;align-items:center;gap:8px;">
              <span style="width:10px;height:10px;border-radius:50%;flex-shrink:0;
                           background:linear-gradient(135deg,#6C5CE7,#FD79A8);display:inline-block;">
              </span> Prediction Results
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

                # Colours
                if is_placed:
                    c_main, c_bg, c_shadow = "#00b894", "rgba(0,184,148,0.08)", "rgba(0,184,148,0.22)"
                    pill_bg = "linear-gradient(135deg,#e8fff4,#d0fae8)"
                    status  = "✓ Placed"
                else:
                    c_main, c_bg, c_shadow = "#e17055", "rgba(225,112,85,0.08)", "rgba(225,112,85,0.22)"
                    pill_bg = "linear-gradient(135deg,#fff5f5,#ffe0e0)"
                    status  = "✗ Not Placed"

                # ── Horizontal layout: ring LEFT, pill+chart RIGHT ──
                ring_col, info_col = st.columns([1, 1.4], gap="small")

                with ring_col:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:center;align-items:center;
                                height:100%;padding-top:0.5rem;">
                      <div style="
                        width:120px;height:120px;border-radius:50%;
                        background:{c_bg};
                        border:6px solid {c_main};
                        box-shadow:0 0 0 6px {c_shadow},0 6px 24px {c_shadow};
                        display:flex;flex-direction:column;
                        align-items:center;justify-content:center;
                      ">
                        <span style="font-family:'Outfit',sans-serif;font-size:1.9rem;
                                     font-weight:800;color:{c_main};line-height:1;">
                          {probability*100:.1f}%
                        </span>
                        <span style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                                     letter-spacing:1px;color:#aaa;margin-top:2px;">
                          Probability
                        </span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                with info_col:
                    st.markdown(f"""
                    <div style="padding-top:0.4rem;">
                      <span style="
                        background:{pill_bg};color:{c_main};
                        border:1.5px solid {c_shadow};border-radius:24px;
                        padding:6px 18px;font-family:'Outfit',sans-serif;
                        font-size:1.05rem;font-weight:700;display:inline-block;
                        margin-bottom:0.8rem;
                      ">{status}</span>
                      <div style="font-size:0.82rem;color:#555;line-height:1.7;">
                        <div>📊 <b>SSC</b>: {ssc_p}% &nbsp;|&nbsp; <b>HSC</b>: {hsc_p}%</div>
                        <div>🎓 <b>Degree</b>: {degree_p}% &nbsp;|&nbsp; <b>E-Test</b>: {etest_p}%</div>
                        <div>💼 <b>Work Exp</b>: {workex}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)

                # Feature Impact chart — compact
                st.markdown("""
                <p style="font-family:'Outfit',sans-serif;font-size:0.88rem;font-weight:700;
                           color:#1A1A2E;margin:0 0 0.3rem;">
                  📊 Feature Impact
                </p>""", unsafe_allow_html=True)
                imp_df = get_feature_importance(model, X)
                st.bar_chart(imp_df.set_index("Feature"),
                             color="#6C5CE7",
                             use_container_width=True,
                             height=185)

                # Recommendations — compact
                st.markdown("""
                <p style="font-family:'Outfit',sans-serif;font-size:0.88rem;font-weight:700;
                           color:#1A1A2E;margin:0.3rem 0 0.4rem;">
                  💡 Recommendations
                </p>""", unsafe_allow_html=True)
                for rec in get_recommendations(profile, probability):
                    st.markdown(f"""
                    <div style="
                      background:linear-gradient(135deg,#F8F0FF,#EFF0FF);
                      border-left:3px solid #6C5CE7;border-radius:8px;
                      padding:8px 12px;margin-bottom:7px;
                      font-size:0.84rem;color:#2D2D4E;line-height:1.55;
                    ">• {rec}</div>
                    """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style="text-align:center;padding:2.5rem 1rem;color:#AAAAC2;">
                  <div style="font-size:2.8rem;margin-bottom:0.8rem;">🎯</div>
                  <div style="font-size:0.95rem;line-height:1.9;">
                    Fill in your academic profile on the left<br>and click
                    <strong style="color:#6C5CE7;">Predict Placement Probability</strong><br>
                    to see your AI-powered result instantly.
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── BOTTOM TABS ──────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📌 Overview", "📊 Dataset Insights"])

    with tab1:
        m1, m2, m3 = st.columns(3)
        m1.metric("Model", "Random Forest")
        m2.metric("Training Accuracy", "~88%")
        m3.metric("Dataset Size", "215 Students")
        st.write("")
        st.info("This app uses a Random Forest classifier trained on real campus placement data to predict employment outcomes using 13 academic & profile features.")

    with tab2:
        dataset = load_dataset()
        if dataset is None:
            st.info("Dataset file not found.")
        else:
            t1, t2 = st.columns(2)
            with t1:
                st.write("**Placement by Specialisation**")
                st.bar_chart(
                    dataset.groupby(["specialisation", "status"]).size().unstack(fill_value=0),
                    use_container_width=True, height=220)
            with t2:
                st.write("**Placement by Work Experience**")
                st.bar_chart(
                    dataset.groupby(["workex", "status"]).size().unstack(fill_value=0),
                    use_container_width=True, height=220)
            st.write("##### Sample Records")
            st.dataframe(dataset.head(8), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
