"""Standalone Streamlit app — loads model artifacts directly (no FastAPI required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from dropout.features.engineer import add_derived_features_dict

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


@st.cache_resource
def load_predictor():
    model = joblib.load(ARTIFACT_DIR / "model.pkl")
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.pkl")
    le = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
    return model, preprocessor, le


def predict(features: dict, model, preprocessor, le) -> dict:
    features = add_derived_features_dict(features)
    df = pd.DataFrame([features])
    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    label = le.classes_[pred_idx]
    dropout_prob = float(proba[list(le.classes_).index("Dropout")])

    if dropout_prob >= 0.6:
        risk_tier = "High"
    elif dropout_prob >= 0.35:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    return {
        "predicted_outcome": label,
        "dropout_probability": round(dropout_prob, 4),
        "risk_tier": risk_tier,
        "class_probabilities": {
            cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)
        },
    }


st.set_page_config(page_title="Student Dropout Risk Predictor", layout="wide")
st.title("Student Dropout Risk Prediction")
st.markdown("Enter student details below to predict dropout risk.")

model, preprocessor, le = load_predictor()

with st.form("student_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        marital_status = st.selectbox("Marital Status", [1, 2, 3, 4, 5, 6])
        gender = st.selectbox(
            "Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male"
        )
        age_at_enrollment = st.number_input(
            "Age at Enrollment", min_value=15, max_value=80, value=20
        )
        nationality = st.number_input("Nationality", min_value=1, max_value=100, value=1)
        international = st.selectbox(
            "International", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col2:
        st.subheader("Academic Background")
        application_mode = st.number_input("Application Mode", min_value=1, max_value=50, value=1)
        application_order = st.number_input("Application Order", min_value=0, max_value=10, value=1)
        course = st.number_input("Course", min_value=1, max_value=50, value=1)
        daytime_evening_attendance = st.selectbox(
            "Daytime/Evening", [0, 1], format_func=lambda x: "Daytime" if x == 1 else "Evening"
        )
        previous_qualification = st.number_input(
            "Previous Qualification", min_value=1, max_value=50, value=1
        )
        previous_qualification_grade = st.number_input(
            "Previous Qualification Grade", min_value=0.0, max_value=200.0, value=130.0
        )
        admission_grade = st.number_input(
            "Admission Grade", min_value=0.0, max_value=200.0, value=130.0
        )
        displaced = st.selectbox(
            "Displaced", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )
        educational_special_needs = st.selectbox(
            "Special Needs", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col3:
        st.subheader("Socioeconomic & Financial")
        mothers_qualification = st.number_input(
            "Mother's Qualification", min_value=1, max_value=50, value=1
        )
        fathers_qualification = st.number_input(
            "Father's Qualification", min_value=1, max_value=50, value=1
        )
        mothers_occupation = st.number_input(
            "Mother's Occupation", min_value=1, max_value=50, value=1
        )
        fathers_occupation = st.number_input(
            "Father's Occupation", min_value=1, max_value=50, value=1
        )
        debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        tuition_fees_up_to_date = st.selectbox(
            "Tuition Fees Up to Date", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )
        scholarship_holder = st.selectbox(
            "Scholarship Holder", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )

    st.subheader("Semester 1")
    col4, col5, col6 = st.columns(3)
    with col4:
        curricular_units_1st_sem_credited = st.number_input(
            "1st Sem Credited", min_value=0, max_value=20, value=0
        )
        curricular_units_1st_sem_enrolled = st.number_input(
            "1st Sem Enrolled", min_value=0, max_value=20, value=6
        )
        curricular_units_1st_sem_evaluations = st.number_input(
            "1st Sem Evaluations", min_value=0, max_value=20, value=6
        )
    with col5:
        curricular_units_1st_sem_approved = st.number_input(
            "1st Sem Approved", min_value=0, max_value=20, value=6
        )
        curricular_units_1st_sem_grade = st.number_input(
            "1st Sem Grade", min_value=0.0, max_value=20.0, value=12.0
        )
        curricular_units_1st_sem_without_evaluations = st.number_input(
            "1st Sem Without Evaluations", min_value=0, max_value=20, value=0
        )

    st.subheader("Semester 2")
    col7, col8, col9 = st.columns(3)
    with col7:
        curricular_units_2nd_sem_credited = st.number_input(
            "2nd Sem Credited", min_value=0, max_value=20, value=0
        )
        curricular_units_2nd_sem_enrolled = st.number_input(
            "2nd Sem Enrolled", min_value=0, max_value=20, value=6
        )
        curricular_units_2nd_sem_evaluations = st.number_input(
            "2nd Sem Evaluations", min_value=0, max_value=20, value=6
        )
    with col8:
        curricular_units_2nd_sem_approved = st.number_input(
            "2nd Sem Approved", min_value=0, max_value=20, value=6
        )
        curricular_units_2nd_sem_grade = st.number_input(
            "2nd Sem Grade", min_value=0.0, max_value=20.0, value=13.0
        )
        curricular_units_2nd_sem_without_evaluations = st.number_input(
            "2nd Sem Without Evaluations", min_value=0, max_value=20, value=0
        )

    st.subheader("Macroeconomic Factors")
    col10, col11, col12 = st.columns(3)
    with col10:
        unemployment_rate = st.number_input(
            "Unemployment Rate", min_value=0.0, max_value=50.0, value=10.0
        )
    with col11:
        inflation_rate = st.number_input(
            "Inflation Rate", min_value=-5.0, max_value=20.0, value=1.5
        )
    with col12:
        gdp = st.number_input("GDP", min_value=-5.0, max_value=10.0, value=1.0)

    submitted = st.form_submit_button("Predict Dropout Risk")

if submitted:
    features = {
        "Marital Status": marital_status,
        "Gender": gender,
        "Age at enrollment": age_at_enrollment,
        "Nacionality": nationality,
        "International": international,
        "Application mode": application_mode,
        "Application order": application_order,
        "Course": course,
        "Daytime/evening attendance": daytime_evening_attendance,
        "Previous qualification": previous_qualification,
        "Previous qualification (grade)": previous_qualification_grade,
        "Admission grade": admission_grade,
        "Displaced": displaced,
        "Mother's qualification": mothers_qualification,
        "Father's qualification": fathers_qualification,
        "Mother's occupation": mothers_occupation,
        "Father's occupation": fathers_occupation,
        "Educational special needs": educational_special_needs,
        "Debtor": debtor,
        "Tuition fees up to date": tuition_fees_up_to_date,
        "Scholarship holder": scholarship_holder,
        "Curricular units 1st sem (credited)": curricular_units_1st_sem_credited,
        "Curricular units 1st sem (enrolled)": curricular_units_1st_sem_enrolled,
        "Curricular units 1st sem (evaluations)": curricular_units_1st_sem_evaluations,
        "Curricular units 1st sem (approved)": curricular_units_1st_sem_approved,
        "Curricular units 1st sem (grade)": curricular_units_1st_sem_grade,
        "Curricular units 1st sem (without evaluations)": curricular_units_1st_sem_without_evaluations,
        "Curricular units 2nd sem (credited)": curricular_units_2nd_sem_credited,
        "Curricular units 2nd sem (enrolled)": curricular_units_2nd_sem_enrolled,
        "Curricular units 2nd sem (evaluations)": curricular_units_2nd_sem_evaluations,
        "Curricular units 2nd sem (approved)": curricular_units_2nd_sem_approved,
        "Curricular units 2nd sem (grade)": curricular_units_2nd_sem_grade,
        "Curricular units 2nd sem (without evaluations)": curricular_units_2nd_sem_without_evaluations,
        "Unemployment rate": unemployment_rate,
        "Inflation rate": inflation_rate,
        "GDP": gdp,
    }

    try:
        result = predict(features, model, preprocessor, le)

        st.divider()
        st.subheader("Prediction Result")

        outcome = result["predicted_outcome"]
        dropout_prob = result["dropout_probability"]
        risk_tier = result["risk_tier"]

        if risk_tier == "High":
            st.error(f"Risk Tier: {risk_tier} | Dropout Probability: {dropout_prob:.2%}")
        elif risk_tier == "Medium":
            st.warning(f"Risk Tier: {risk_tier} | Dropout Probability: {dropout_prob:.2%}")
        else:
            st.success(f"Risk Tier: {risk_tier} | Dropout Probability: {dropout_prob:.2%}")

        st.write(f"**Predicted Outcome:** {outcome}")

        st.subheader("Class Probabilities")
        for cls, prob in result["class_probabilities"].items():
            st.progress(float(prob), text=f"{cls}: {prob:.2%}")

    except Exception as exc:
        st.error(f"Prediction error: {exc}")
