import streamlit as st
import pandas as pd
import joblib

# Load the trained model and column names
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# App title
st.title("Salary Prediction App")

# User inputs
age = st.number_input("Age", min_value=18, max_value=70, value=30)
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
job_title = st.selectbox("Job Title", [
    'Accountant', 'Administrative Assistant', 'Back end Developer',
    'Business Analyst', 'Business Development Manager',
    'Data Analyst', 'Data Engineer', 'Data Scientist',
    'Database Administrator', 'Front end Developer', 'Full Stack Developer',
    'HR Manager', 'IT Support', 'ML Engineer', 'Marketing Analyst',
    'Network Engineer', 'Project Manager', 'Sales Associate',
    'Software Developer', 'Software Engineer', 'System Administrator',
    'Senior Manager', 'Director', 'Marketing Coordinator',
    'Sales Manager', 'Senior Scientist', 'Financial Analyst',
    'Customer Service Rep', 'Operations Manager', 'Marketing Manager',
    'Senior Engineer', 'Data Entry Clerk', 'Sales Director',
    'VP of Operations', 'Recruiter', 'Product Manager'
])

# One-hot encode input
input_data = {
    'Age': age,
    'Years of Experience': experience,
    **{col: 0 for col in model_columns if col not in ['Age', 'Years of Experience']}
}

# Set selected features to 1
if f'Gender_{gender}' in input_data:
    input_data[f'Gender_{gender}'] = 1
if f"Education Level_{education}" in input_data:
    input_data[f"Education Level_{education}"] = 1
if f"Job Title_{job_title}" in input_data:
    input_data[f"Job Title_{job_title}"] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Reorder columns to match model
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict Salary"):
    salary = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ₹{salary:,.2f}")
