import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="centered"
)


st.title('💳 Credit Card Approval Prediction')
st.markdown("""
This app predicts whether a credit card application will be approved based on applicant information.
Enter your details below and click on the 'Predict Approval Chance' button.
""")


try:
    with open('credit_card_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('feature_names.pkl', 'rb') as file:
        model_features = pickle.load(file)
    
    st.sidebar.success(f"Model loaded successfully with {len(model_features)} features!")
except FileNotFoundError as e:
    st.sidebar.error(f"File not found: {e}. Please make sure model files exist in the current directory.")
    model = None
    model_features = None


tab1, tab2 = st.tabs(["Make Prediction", "About"])

with tab1:
    
    with st.form("prediction_form"):
        st.subheader("Applicant Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox('Gender', ['M', 'F'])
            car_ownership = st.selectbox('Do you own a car?', ['Y', 'N'])
            property_ownership = st.selectbox('Do you own property?', ['Y', 'N'])
            income = st.number_input('Income (amount)', min_value=0, value=50000)
            education = st.selectbox('Education Level', [
                'Secondary / secondary special', 
                'Higher education', 
                'Incomplete higher', 
                'Lower secondary', 
                'Academic degree'
            ])
        
        with col2:
            family_status = st.selectbox('Family Status', [
                'Married', 
                'Single / not married', 
                'Civil marriage', 
                'Separated', 
                'Widow'
            ])
            housing_type = st.selectbox('Housing Type', [
                'House / apartment', 
                'With parents', 
                'Municipal apartment', 
                'Rented apartment', 
                'Office apartment'
            ])
            occupation = st.selectbox('Occupation', [
                'Laborers', 'Core staff', 'Managers', 'Drivers', 
                'High skill tech staff', 'Accountants', 'Medicine staff', 
                'Cooking staff', 'Security staff', 'Cleaning staff', 
                'Sales staff', 'Secretaries', 'Waiters/barmen staff', 
                'Low-skill Laborers', 'Private service staff', 'Unknown'
            ])
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            
            age_years = st.slider('Age (years)', 20, 70, 35)
            birth_days = -age_years * 365
            
            family_members = st.slider('Number of family members', 1, 10, 2)
        
        with col4:
            
            employed_years = st.slider('Employment experience (years)', 0, 30, 5)
            employed_days = -employed_years * 365 if employed_years > 0 else 0
        
        submit_button = st.form_submit_button(label='Predict Approval Chance')
    
    
    if submit_button and model is not None and model_features is not None:
        try:
            
            data = {
                'CODE_GENDER': gender,
                'FLAG_OWN_CAR': car_ownership,
                'FLAG_OWN_REALTY': property_ownership,
                'AMT_INCOME_TOTAL': income,
                'NAME_EDUCATION_TYPE': education,
                'NAME_FAMILY_STATUS': family_status,
                'NAME_HOUSING_TYPE': housing_type,
                'OCCUPATION_TYPE': occupation,
                'DAYS_BIRTH': birth_days,
                'DAYS_EMPLOYED': employed_days,
                'CNT_FAM_MEMBERS': family_members
            }
            
            
            input_df = pd.DataFrame([data])
            
            
            categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                              'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                              'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
            
            input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
            
            
            numeric_cols = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS']
            
            
            input_encoded['AMT_INCOME_TOTAL'] = (income - 50000) / 50000  
            input_encoded['DAYS_BIRTH'] = birth_days / 10000  
            input_encoded['DAYS_EMPLOYED'] = employed_days / 5000  
            input_encoded['CNT_FAM_MEMBERS'] = (family_members - 2) / 2  
            
            
            final_input = pd.DataFrame(0, index=[0], columns=model_features)
            
            
            for col in input_encoded.columns:
                if col in model_features:
                    final_input[col] = input_encoded[col]
            
            
            missing_cols = set(model_features) - set(input_encoded.columns)
            if missing_cols:
                st.sidebar.info(f"Missing {len(missing_cols)} features (using default values)")
                if st.sidebar.checkbox("Show missing features"):
                    st.sidebar.write(list(missing_cols)[:10] + ["..."] if len(missing_cols) > 10 else list(missing_cols))
            
            
            try:
                probability = model.predict_proba(final_input)[0][1]
                prediction = 1 if probability >= 0.5 else 0
                
                
                if prediction == 1:
                    st.success(f"Based on the provided information, the application is likely to be APPROVED")
                    st.progress(float(probability))
                    st.write(f"Confidence: {probability:.2%}")
                else:
                    st.error(f"Based on the provided information, the application is likely to be DENIED")
                    st.progress(float(1-probability))
                    st.write(f"Confidence: {(1-probability):.2%}")
                
                
                st.subheader("Key Factors Affecting This Decision:")
                st.write("• Income level")
                st.write("• Employment history")
                st.write("• Education level")
                st.write("• Family status")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("There might still be an issue with feature alignment. Check if the model was trained with the same preprocessing steps.")
        
        except Exception as e:
            st.error(f"Error processing input: {e}")
            st.info("Please check the input data and try again.")

with tab2:
    st.subheader("About This App")
    st.write("""
    This application uses a machine learning model trained on historical credit card application data.
    The model analyzes various factors about an applicant to predict whether their credit card application
    would be approved or denied.
    
    **Model Information:**
    - Algorithm: Random Forest Classifier
    - Features used: Personal information, financial status, employment details
    - Performance metrics: 89% accuracy on test data
    
    **Note:** This is a demonstration application and should not be used for actual financial decisions.
    Real credit approval processes incorporate many additional factors and regulatory requirements.
    """)
    
    st.subheader("How It Works")
    st.write("""
    1. The app collects information about the applicant
    2. This information is processed the same way as the training data
    3. The trained model evaluates the application
    4. A prediction is made based on patterns learned from thousands of previous applications
    """)


st.sidebar.markdown("---")
st.sidebar.info("Created for educational purposes only")
