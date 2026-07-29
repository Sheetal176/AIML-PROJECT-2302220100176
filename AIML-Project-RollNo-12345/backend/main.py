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

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="wide")


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
    ).sort_values("importance", ascending=False)

    return importance_df.head(10)


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
    if profile["mba_p"] < 70:
        recommendations.append("Concentrate on core MBA subjects and case studies for better outcomes.")

    if not recommendations:
        recommendations.append("Your profile looks strong. Keep refining your resume and interview preparation.")

    return recommendations


def main():
    st.title("🎓 Student Placement Predictor")
    st.caption("A simple Streamlit app that predicts whether a student is likely to be placed based on academic and profile data.")

    model_data = load_model_bundle()
    if not model_data:
        st.error("The trained model file is missing. Train the model first and refresh this page.")
        st.stop()

    model = model_data["model"]
    expected_columns = model_data["columns"]

    with st.sidebar:
        st.header("Student profile")
        gender = st.selectbox("Gender", ["M", "F"])
        ssc_p = st.slider("SSC percentage", 0, 100, 70)
        hsc_p = st.slider("HSC percentage", 0, 100, 75)
        hsc_s = st.selectbox("HSC specialization", ["Commerce", "Science", "Arts"])
        degree_p = st.slider("Degree percentage", 0, 100, 75)
        degree_t = st.selectbox("Degree type", ["Sci&Tech", "Comm&Mgmt", "Others"])
        workex = st.selectbox("Work experience", ["Yes", "No"])
        etest_p = st.slider("E-test percentage", 0, 100, 70)
        specialisation = st.selectbox("MBA specialization", ["Mkt&HR", "Mkt&Fin"])
        mba_p = st.slider("MBA percentage", 0, 100, 75)

        st.markdown("---")
        st.write("Deploy this app on Streamlit Cloud or a server with:")
        st.code("streamlit run app.py", language="bash")

    profile = {
        "gender": gender,
        "ssc_p": ssc_p,
        "hsc_p": hsc_p,
        "hsc_s": hsc_s,
        "degree_p": degree_p,
        "degree_t": degree_t,
        "workex": workex,
        "etest_p": etest_p,
        "specialisation": specialisation,
        "mba_p": mba_p,
    }

    if st.button("Predict placement", type="primary"):
        df = pd.DataFrame([profile])
        processor = DataProcessorFactory.get_processor("standard")
        processor.expected_columns = expected_columns
        X, _ = processor.preprocess(df, is_training=False)

        strategy = RandomForestStrategy()
        probability = strategy.predict_proba(model, X)[0][1]
        prediction = strategy.predict(model, X)[0]

        prediction_label = "Placed" if prediction == 1 else "Not Placed"

        col1, col2, col3 = st.columns(3)
        col1.metric("Placement probability", f"{probability * 100:.2f}%")
        col2.metric("Prediction", prediction_label)
        col3.metric("Confidence", "High" if probability > 0.6 else "Moderate" if probability > 0.4 else "Low")

        st.success(f"The model predicts: {prediction_label} with {probability * 100:.2f}% probability.")

        with st.expander("Why this prediction?", expanded=True):
            importance_df = get_feature_importance(model, X)
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

        st.subheader("Recommended next steps")
        for item in get_recommendations(profile, probability):
            st.write(f"- {item}")

    tab1, tab2 = st.tabs(["Overview", "Dataset insights"])
    with tab1:
        st.write("Use the sidebar to enter a student profile and get an instant placement prediction.")
        st.write("The app reuses the trained model and preprocessing logic from the original project.")

    with tab2:
        dataset = load_dataset()
        if dataset is None:
            st.info("Dataset file is not available in the project folder.")
        else:
            st.write("Placement trends from the training data")
            status_counts = dataset["status"].value_counts().rename(index={1: "Placed", 0: "Not Placed"})
            st.bar_chart(status_counts)

            placement_by_specialisation = (
                dataset.groupby(["specialisation", "status"]).size().unstack(fill_value=0)
            )
            placement_by_specialisation = placement_by_specialisation.rename(index={1: "Placed", 0: "Not Placed"})
            st.bar_chart(placement_by_specialisation)

            st.dataframe(dataset.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
