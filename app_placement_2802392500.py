import streamlit as st
import joblib
import pandas as pd

# Load the machine learning model and encode
model_class = joblib.load('artifacts/placement_prediction_pipeline.pkl')
model_reg = joblib.load('artifacts/salary_regression_pipeline.pkl')


def main():
    st.set_page_config(page_title="Placement & Salary Predictor", layout="wide")
    st.title('Placement and Salary Prediction Model Deployment')
    st.write("Stevanus Gerald Marconus - 2802392500")

    with st.sidebar:
        st.header("Input Guide")
        st.write("Silahkan mengisi Data Mahasiswa, lalu klik **Make Prediction** untuk melihat hasil.")
        st.markdown(
            "NOTES:\n"
            "- Gunakan nilai akademik dalam skala 0-100\n"
            "- CGPA berada pada rentang 4.0-10.0\n"
        )
        st.info("Model akan memprediksi placement lebih dulu, lalu mengestimasi salary jika kandidat diprediksi placed (Mendapat pekerjaan).")

    st.subheader("Candidate Profile")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            ssc_percentage = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, step=0.1)
            hsc_percentage = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, step=0.1)
            degree_percentage = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, step=0.1)
            cgpa = st.number_input("CGPA", min_value=4.0, max_value=10.0, step=0.1)
            entrance_exam_score = st.number_input("Entrance Exam Score", min_value=0.0, max_value=100.0, step=0.1)
            attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, step=0.1)

        with col2:
            technical_skill_score = st.number_input("Technical Skill Score", min_value=0.0, max_value=100.0, step=0.1)
            soft_skill_score = st.number_input("Soft Skill Score", min_value=0.0, max_value=100.0, step=0.1)
            internship_count = st.number_input("Internship Count", min_value=0, max_value=10, step=1)
            live_projects = st.number_input("Live Project Count", min_value=0, max_value=10, step=1)
            work_experience_months = st.number_input("Work Experience (months)", min_value=0, max_value=100, step=1)
            certifications = st.number_input("Certification Count", min_value=0, max_value=20, step=1)
            backlogs = st.number_input("Backlog Count", min_value=0, max_value=20, step=1)
            extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])

        submitted = st.form_submit_button("Make Prediction", use_container_width=True)

    data = {
        "gender": gender,
        "ssc_percentage": ssc_percentage,
        "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage,
        "cgpa": cgpa,
        "entrance_exam_score": entrance_exam_score,
        "technical_skill_score": technical_skill_score,
        "soft_skill_score": soft_skill_score,
        "internship_count": internship_count,
        "live_projects": live_projects,
        "work_experience_months": work_experience_months,
        "certifications": certifications,
        "attendance_percentage": attendance_percentage,
        "backlogs": backlogs,
        "extracurricular_activities": extracurricular_activities,
    }

    df = pd.DataFrame([data])

    st.divider()
    st.subheader("Quick Profile Snapshot")
    viz_df = pd.DataFrame(
        {
            "Score": [ssc_percentage, hsc_percentage, degree_percentage, entrance_exam_score, technical_skill_score, soft_skill_score, attendance_percentage, cgpa * 10]
        },
        index=["SSC", "HSC", "Degree", "Entrance", "Technical", "Soft Skill", "Attendance", "CGPA",
        ],
    )
    st.bar_chart(viz_df)

    if submitted:
        prediction = model_class.predict(df)[0]
        if prediction == 1:
            salary_pred = model_reg.predict(df)[0]
            st.success("Predicted: Placed")
            st.metric("Estimated Salary Package (LPA)", f"{salary_pred:.2f}")
        else:
            st.warning("Predicted: Not Placed")


if __name__ == "__main__":
    main()

