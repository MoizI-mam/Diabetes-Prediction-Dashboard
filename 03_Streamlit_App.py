import streamlit as st
import pandas as pd
import joblib
# 1.Page Config
st.set_page_config(
    page_title = "Diabetes Predictor",
    page_icon  = "🩺",
    layout     = "centered"
)

# 2.Load Model and Columns
model = joblib.load("Best_Model.pkl")
columns = joblib.load("columns.pkl")

# 3.Title
st.title("🩺 Diabetes Prediction Dashboard")
st.markdown("Developed by Moiz Imam | ML Engineer")
st.divider()

# 4.Now Making a new Layout
# Session State 
if "step" not in st.session_state:
    st.session_state.step = 1

# Step Indicator 
st.markdown(f"**Step {st.session_state.step} of 3**")
st.progress(st.session_state.step / 3)

# Step: 1 Personal Info
if st.session_state.step == 1:
    st.subheader("👤 Step 1 — Personal Information")
    age = st.number_input("Age",min_value = 1, max_value = 100, value = 30)
    gender = st.selectbox("Gender",["Female","Male","Other"])
    smoking_history = st.selectbox("Smoking History",   ["never", "former", "unknown", "current"])

    if st.button("Next →", use_container_width = True):
        # Save to session state
        st.session_state.age = age
        st.session_state.gender = gender
        st.session_state.smoking_history = smoking_history
        st.session_state.step = 2
        st.rerun()
# Step: 2 Medical Information        
elif st.session_state.step == 2:
    st.subheader("🏥 Step 2 — Medical Information")

    bmi = st.number_input("BMI",min_value = 10.0,max_value = 60.0, value = 25.0)
    hba1c = st.number_input("HbA1c Level",min_value = 3.0,max_value = 15.0,value = 5.0)
    blood_glucose = st.number_input("Blood Glucose Level",min_value = 50,max_value = 300,value = 100)
    hypertension = st.selectbox("Hypertension",["No","Yes"])
    heart_disease = st.selectbox("Heart Disease",["No", "Yes"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back",use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("🔍 Predict", use_container_width=True):
            # save to session state    
            st.session_state.bmi = bmi
            st.session_state.hba1c = hba1c
            st.session_state.blood_glucose = blood_glucose
            st.session_state.hypertension = hypertension
            st.session_state.heart_disease = heart_disease
            st.session_state.step = 3
            st.rerun()
# Step: 3 Prediction Result
# Auto Calculate Engineered Features 
elif st.session_state.step == 3:
    st.subheader("📝 Prediction Result")
    glucose_hba1c_ratio = st.session_state.blood_glucose / st.session_state.hba1c
    bmi_age_ratio       = st.session_state.bmi / st.session_state.age

    # Auto Calculate BMI Category
    bmi = st.session_state.bmi
    if bmi <= 18.5:
        bmi_category = "underweight"
    elif bmi <= 24.9:
        bmi_category = "normal"
    elif bmi <= 29.9:
        bmi_category = "overweight"
    else:
        bmi_category = "obese"

    # Auto Calculate Age Group
    age = st.session_state.age
    if age <= 18:
        age_group = "teen"
    elif age <= 35:
        age_group = "young_adult"
    elif age <= 50:
        age_group = "middle_aged"
    elif age <= 65:
        age_group = "senior"
    else:
        age_group = "elderly"
    # Build Input Dict with all zeros first    
    input_dict = {col: 0 for col in columns}
    # ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'glucose_hba1c_ratio',
    #  'bmi_age_ratio', 'gender_Male', 'gender_Other', 'smoking_history_former', 'smoking_history_never', 'smoking_history_unknown',
    #  'bmi_category_normal', 'bmi_category_overweight', 'bmi_category_obese', 'age_group_young_adult', 'age_group_middle_aged',
    #  'age_group_senior', 'age_group_elderly']

    # Fill Numerical Feature
    input_dict["age"] = st.session_state.age
    input_dict["bmi"] = st.session_state.bmi
    input_dict["HbA1c_level"] = st.session_state.hba1c
    input_dict["blood_glucose_level"] = st.session_state.blood_glucose
    input_dict["hypertension"] = 1 if st.session_state.hypertension == "Yes" else 0
    input_dict["heart_disease"] = 1 if st.session_state.heart_disease == "Yes" else 0
    input_dict["glucose_hba1c_ratio"] = glucose_hba1c_ratio
    input_dict["bmi_age_ratio"] = bmi_age_ratio
    # Encoding Gender
    if st.session_state.gender == "Male":
        input_dict["gender_Male"] = 1
    elif st.session_state.gender == "Other":
        input_dict["gender_Other"]

    # Encoding Smoking History 
    if st.session_state.smoking_history == "former":
        input_dict["smoking_history_former"]  = 1
    elif st.session_state.smoking_history == "never":
        input_dict["smoking_history_never"] = 1
    elif st.session_state.smoking_history == "unknown":
        input_dict["smoking_history_unknown"] = 1

    # Encoding BMI Category 
    if bmi_category == "normal":
        input_dict["bmi_category_normal"] = 1
    elif bmi_category == "overweight":
        input_dict["bmi_category_overweight"] = 1
    elif bmi_category == "obese":
        input_dict["bmi_category_obese"] = 1
    # Encoding Age Group    
    if age_group == "young_adult":
        input_dict["age_group_young_adult"] = 1
    elif age_group == "middle_aged":
        input_dict["age_group_middle_aged"] = 1
    elif age_group == "senior":
        input_dict["age_group_senior"] = 1
    elif age_group == "elderly":
        input_dict["age_group_elderly"] = 1
    # Prediction
    input_df   = pd.DataFrame([input_dict])[columns]
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    # Showing Result
    if prediction == 1:
        st.error("🔴 Diabetic")
        st.metric("Diabetes Probability", f"{proba[1]*100:.1f}%")
    else:
        st.success("🟢 Not Diabetic")
        st.metric("Diabetes Probability", f"{proba[1]*100:.1f}%")
    # Start Over Button    
    if st.button("🔄 Start Over", use_container_width=True):
        st.session_state.step = 1
        st.rerun()        
   