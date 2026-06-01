"""Streamlit UI for the Student Dropout Prediction API."""

import requests
import streamlit as st

FASTAPI_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Student Dropout Risk Predictor", layout="wide")

st.title("Student Dropout Risk Prediction")
st.markdown("Enter student details below to predict dropout risk.")


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
    payload = {
        "marital_status": marital_status,
        "gender": gender,
        "age_at_enrollment": age_at_enrollment,
        "nationality": nationality,
        "international": international,
        "application_mode": application_mode,
        "application_order": application_order,
        "course": course,
        "daytime_evening_attendance": daytime_evening_attendance,
        "previous_qualification": previous_qualification,
        "previous_qualification_grade": previous_qualification_grade,
        "admission_grade": admission_grade,
        "displaced": displaced,
        "mothers_qualification": mothers_qualification,
        "fathers_qualification": fathers_qualification,
        "mothers_occupation": mothers_occupation,
        "fathers_occupation": fathers_occupation,
        "educational_special_needs": educational_special_needs,
        "debtor": debtor,
        "tuition_fees_up_to_date": tuition_fees_up_to_date,
        "scholarship_holder": scholarship_holder,
        "curricular_units_1st_sem_credited": curricular_units_1st_sem_credited,
        "curricular_units_1st_sem_enrolled": curricular_units_1st_sem_enrolled,
        "curricular_units_1st_sem_evaluations": curricular_units_1st_sem_evaluations,
        "curricular_units_1st_sem_approved": curricular_units_1st_sem_approved,
        "curricular_units_1st_sem_grade": curricular_units_1st_sem_grade,
        "curricular_units_1st_sem_without_evaluations": (
            curricular_units_1st_sem_without_evaluations
        ),
        "curricular_units_2nd_sem_credited": curricular_units_2nd_sem_credited,
        "curricular_units_2nd_sem_enrolled": curricular_units_2nd_sem_enrolled,
        "curricular_units_2nd_sem_evaluations": curricular_units_2nd_sem_evaluations,
        "curricular_units_2nd_sem_approved": curricular_units_2nd_sem_approved,
        "curricular_units_2nd_sem_grade": curricular_units_2nd_sem_grade,
        "curricular_units_2nd_sem_without_evaluations": (
            curricular_units_2nd_sem_without_evaluations
        ),
        "unemployment_rate": unemployment_rate,
        "inflation_rate": inflation_rate,
        "gdp": gdp,
    }

    try:
        response = requests.post(FASTAPI_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

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
        probs = result["class_probabilities"]
        for cls, prob in probs.items():
            st.progress(float(prob), text=f"{cls}: {prob:.2%}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI backend. Ensure the server is running on localhost:8000."
        )
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error: {exc.response.text}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
