import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Student Academic Performance Predictor")
st.markdown("""
This application predicts whether a student will Graduate, Dropout, or remain Enrolled 
based on their academic, demographic, and economic factors.
""")

# Sidebar for input
st.sidebar.header("Student Information")
st.sidebar.markdown("Please fill in all the fields below:")

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        encoder = joblib.load("models/label_encoder.pkl")
        return model, scaler, encoder
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'models' folder contains best_model.pkl, scaler.pkl, and label_encoder.pkl")
        return None, None, None

# Create input fields function
def create_input_fields():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographic Information")
        marital_status = st.selectbox("Marital Status", [1, 2, 3, 4, 5, 6], format_func=lambda x: {1:"Single", 2:"Married", 3:"Divorced", 4:"Widowed", 5:"Separated", 6:"Other"}.get(x, str(x)))
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        age_at_enrollment = st.number_input("Age at Enrollment", min_value=17, max_value=100, value=20, step=1)
        nationality = st.number_input("Nationality (Code)", min_value=1, max_value=200, value=1, step=1)
        international = st.selectbox("International Student", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        
        st.subheader("Financial Information")
        debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        tuition_fees_up_to_date = st.selectbox("Tuition Fees Up to Date", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        scholarship_holder = st.selectbox("Scholarship Holder", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        
        st.subheader("Academic Background")
        application_mode = st.number_input("Application Mode (Code)", min_value=1, max_value=60, value=17, step=1)
        application_order = st.number_input("Application Order", min_value=0, max_value=10, value=1, step=1)
        course = st.number_input("Course (Code)", min_value=33, max_value=10000, value=9500, step=100)
        daytime_evening = st.selectbox("Daytime/Evening Attendance", [0, 1], format_func=lambda x: "Evening" if x==0 else "Daytime")
        previous_qualification = st.number_input("Previous Qualification (Code)", min_value=1, max_value=50, value=1, step=1)
        previous_qualification_grade = st.number_input("Previous Qualification Grade", min_value=50.0, max_value=200.0, value=130.0, step=1.0)
        admission_grade = st.number_input("Admission Grade", min_value=50.0, max_value=200.0, value=130.0, step=1.0)
        
        displaced = st.selectbox("Displaced", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        educational_special_needs = st.selectbox("Educational Special Needs", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    
    with col2:
        st.subheader("Family Background")
        mother_qualification = st.number_input("Mother's Qualification (Code)", min_value=1, max_value=45, value=19, step=1)
        father_qualification = st.number_input("Father's Qualification (Code)", min_value=1, max_value=45, value=19, step=1)
        mother_occupation = st.number_input("Mother's Occupation (Code)", min_value=0, max_value=200, value=9, step=1)
        father_occupation = st.number_input("Father's Occupation (Code)", min_value=0, max_value=200, value=9, step=1)
        
        st.subheader("First Semester Performance")
        curricular_units_1st_sem_credited = st.number_input("1st Sem - Units Credited", min_value=0, max_value=20, value=0, step=1)
        curricular_units_1st_sem_enrolled = st.number_input("1st Sem - Units Enrolled", min_value=0, max_value=25, value=6, step=1)
        curricular_units_1st_sem_evaluations = st.number_input("1st Sem - Evaluations", min_value=0, max_value=35, value=6, step=1)
        curricular_units_1st_sem_approved = st.number_input("1st Sem - Units Approved", min_value=0, max_value=25, value=5, step=1)
        curricular_units_1st_sem_grade = st.number_input("1st Sem - Average Grade", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
        curricular_units_1st_sem_without_evaluations = st.number_input("1st Sem - Without Evaluations", min_value=0, max_value=15, value=0, step=1)
    
    with col3:
        st.subheader("Second Semester Performance")
        curricular_units_2nd_sem_credited = st.number_input("2nd Sem - Units Credited", min_value=0, max_value=20, value=0, step=1)
        curricular_units_2nd_sem_enrolled = st.number_input("2nd Sem - Units Enrolled", min_value=0, max_value=25, value=6, step=1)
        curricular_units_2nd_sem_evaluations = st.number_input("2nd Sem - Evaluations", min_value=0, max_value=35, value=6, step=1)
        curricular_units_2nd_sem_approved = st.number_input("2nd Sem - Units Approved", min_value=0, max_value=25, value=5, step=1)
        curricular_units_2nd_sem_grade = st.number_input("2nd Sem - Average Grade", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
        curricular_units_2nd_sem_without_evaluations = st.number_input("2nd Sem - Without Evaluations", min_value=0, max_value=15, value=0, step=1)
        
        st.subheader("Economic Indicators")
        unemployment_rate = st.number_input("Unemployment Rate", min_value=5.0, max_value=20.0, value=11.0, step=0.5)
        inflation_rate = st.number_input("Inflation Rate", min_value=-2.0, max_value=5.0, value=1.0, step=0.5)
        gdp = st.number_input("GDP", min_value=-5.0, max_value=5.0, value=0.5, step=0.5)
    
    # Create feature array in the exact order as training
    features = [
        marital_status, application_mode, application_order, course, daytime_evening,
        previous_qualification, previous_qualification_grade, nationality, mother_qualification,
        father_qualification, mother_occupation, father_occupation, admission_grade, displaced,
        educational_special_needs, debtor, tuition_fees_up_to_date, gender, scholarship_holder,
        age_at_enrollment, international, curricular_units_1st_sem_credited,
        curricular_units_1st_sem_enrolled, curricular_units_1st_sem_evaluations,
        curricular_units_1st_sem_approved, curricular_units_1st_sem_grade,
        curricular_units_1st_sem_without_evaluations, curricular_units_2nd_sem_credited,
        curricular_units_2nd_sem_enrolled, curricular_units_2nd_sem_evaluations,
        curricular_units_2nd_sem_approved, curricular_units_2nd_sem_grade,
        curricular_units_2nd_sem_without_evaluations, unemployment_rate, inflation_rate, gdp
    ]
    
    return np.array(features).reshape(1, -1)

# Prediction function
def predict(features, model, scaler, encoder):
    # Scale the features
    features_scaled = scaler.transform(features)
    # Make prediction
    prediction_encoded = model.predict(features_scaled)
    # Convert back to original label
    prediction = encoder.inverse_transform(prediction_encoded)
    return prediction[0]

# Main app
def main():
    # Load models
    model, scaler, encoder = load_models()
    
    if model is None:
        st.warning("Please ensure you have trained and saved your models first!")
        st.info("Run your Jupyter notebook to generate the model files in the 'models' folder.")
        return
    
    # Create input fields
    features = create_input_fields()
    
    # Prediction button
    st.markdown("---")
    col_button, col_result = st.columns([1, 2])
    
    with col_button:
        predict_button = st.button("Predict Student Outcome", type="primary", use_container_width=True)
    
    with col_result:
        if predict_button:
            with st.spinner("Analyzing student data..."):
                result = predict(features, model, scaler, encoder)
                
                # Display result with appropriate styling
                if result == "Graduate":
                    st.success(f"PREDICTION: {result}")
                    st.markdown("This student is likely to graduate!")
                elif result == "Dropout":
                    st.error(f"PREDICTION: {result}")
                    st.markdown("This student may be at risk of dropping out. Early intervention recommended.")
                else:  # Enrolled
                    st.warning(f"PREDICTION: {result}")
                    st.markdown("This student is currently enrolled but progress needs monitoring.")
    
    # Feature importance note
    st.markdown("---")
    st.caption("Note: This prediction is based on a Random Forest model trained on historical student data. Results should be used as a guideline, not as absolute determination.")

# Run the app
if __name__ == "__main__":
    main()