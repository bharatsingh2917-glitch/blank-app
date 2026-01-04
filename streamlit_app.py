import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .healthy {
        color: #28a745;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .disease {
        color: #dc3545;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load and train model
@st.cache_resource
def load_model():
    # Load the dataset
    heart_data = pd.read_csv('data.csv')

    # Split features and target
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']

    # Split into training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, Y_train)

    # Calculate accuracy
    X_test_prediction = model.predict(X_test_scaled)
    accuracy = accuracy_score(X_test_prediction, Y_test)

    return model, scaler, accuracy

# Main app
def main():
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)

    # Load model
    model, scaler, accuracy = load_model()

    st.markdown(f"**Model Accuracy: {accuracy*100:.2f}%**")

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Patient Information")

        # Input fields
        age = st.slider("Age", 20, 100, 50)

        sex = st.selectbox("Sex", ["Female", "Male"])
        sex_val = 0 if sex == "Female" else 1

        cp = st.selectbox("Chest Pain Type",
                         ["Typical Angina (0)", "Atypical Angina (1)",
                          "Non-anginal Pain (2)", "Asymptomatic (3)"])
        cp_val = ["Typical Angina (0)", "Atypical Angina (1)",
                  "Non-anginal Pain (2)", "Asymptomatic (3)"].index(cp)

        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)

        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)

        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs_val = 0 if fbs == "No" else 1

    with col2:
        st.subheader("🔍 Additional Parameters")

        restecg = st.selectbox("Resting ECG Results",
                              ["Normal (0)", "ST-T Wave Abnormality (1)",
                               "Left Ventricular Hypertrophy (2)"])
        restecg_val = ["Normal (0)", "ST-T Wave Abnormality (1)",
                       "Left Ventricular Hypertrophy (2)"].index(restecg)

        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang_val = 0 if exang == "No" else 1

        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)

        slope = st.selectbox("Slope of Peak Exercise ST Segment",
                           ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
        slope_val = ["Upsloping (0)", "Flat (1)", "Downsloping (2)"].index(slope)

        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)

        thal = st.selectbox("Thalassemia",
                           ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)", "Not Described (3)"])
        thal_val = ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)", "Not Described (3)"].index(thal)

    # Prediction button
    if st.button("🔮 Predict Heart Disease", type="primary", use_container_width=True):
        # Prepare input data
        input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                               restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Display result
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("📋 Prediction Result")

        if prediction[0] == 0:
            st.markdown('<p class="healthy">✅ The patient does NOT have heart disease</p>', unsafe_allow_html=True)
            st.info("Based on the provided parameters, the model predicts a healthy heart. However, please consult with a medical professional for accurate diagnosis.")
        else:
            st.markdown('<p class="disease">⚠️ The patient HAS heart disease</p>', unsafe_allow_html=True)
            st.warning("Based on the provided parameters, the model predicts heart disease. Please consult with a medical professional immediately for proper diagnosis and treatment.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This is a machine learning prediction tool for educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.")

if __name__ == "__main__":
    main()
