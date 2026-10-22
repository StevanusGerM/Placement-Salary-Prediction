# 🎓 Student Placement & Salary Package Prediction System
*Mid-Term Exam Model Deployment Project*

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen?logo=streamlit)](https://placement-salary-prediction-sgm-2802392500-mdmidexam.streamlit.app/)

## 🔍 Project Overview
This repository contains the complete workflow for a machine learning model deployment project focused on student employability and career analytics. The objective is to analyze historical student performance, academic metrics, and skill assessments to predict two major outcomes:
1. **Placement Status** (Binary Classification: Placed / Not Placed)[cite: 2]
2. **Salary Package in LPA (Lakhs Per Annum)** (Regression / Package Estimation)[cite: 2]

---

## 📊 Dataset Summary
* **Source File:** `B.csv`[cite: 2]
* **Dimensions:** 5,000 rows × 18 columns[cite: 2]
* **Features Include:**
  * **Academic Performance:** SSC percentage, HSC percentage, degree percentage, CGPA, and entrance exam scores[cite: 2].
  * **Skills & Experience:** Technical skill score, soft skill score, internship count, live projects, work experience in months, and certifications[cite: 2].
  * **Behavioral & Extracurricular:** Attendance percentage, active backlogs, and extracurricular activities participation[cite: 2].
  * **Target Variables:** Placement status (`placement_status`) and annual salary package (`salary_package_lpa`)[cite: 2].

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python[cite: 2]
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn[cite: 2]
* **Machine Learning (`scikit-learn`, `xgboost`):**
  * *Preprocessing:* `LabelEncoder`, `OneHotEncoder`, `StandardScaler`, `RobustScaler`[cite: 2]
  * *Algorithms:* Logistic Regression, Linear Regression, Random Forest Classifier/Regressor, Support Vector Regressor (SVR), XGBoost Classifier[cite: 2]
  * *Model Evaluation & Validation:* Accuracy, F1-Score, Precision, Recall, ROC-AUC, Mean Squared Error (MSE), $R^2$ Score, Stratified K-Fold Cross-Validation[cite: 2]

---

## 🚀 Deployment & Presentation Links
* **Live Web Application:** [Streamlit App Link](https://placement-salary-prediction-sgm-2802392500-mdmidexam.streamlit.app/)[cite: 2]
* **Video Presentation & Walkthrough:** [YouTube Video Link](https://youtu.be/RuZc80pSx7Y?si=qyTyN5SGTwAePJmR)[cite: 2]
