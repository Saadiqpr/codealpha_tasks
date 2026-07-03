import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model/best_heart_disease_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Prediction System")

st.write(
    "Predict whether a patient is likely to have heart disease "
    "using a trained Random Forest Machine Learning model."
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About")

st.sidebar.info(
    """
    **Machine Learning Internship Project**

    Model: Random Forest

    Dataset: UCI Heart Disease Dataset

    Accuracy: **88.52%**
    """
)

# -----------------------------
# Input Fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.number_input("Age", 20, 100, 45)

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [1, 2, 3, 4]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        80,
        220,
        120
    )

    chol = st.number_input(
        "Cholesterol",
        100,
        600,
        200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2]
    )

with col2:

    thalach = st.number_input(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        0.0,
        10.0,
        1.0
    )

    slope = st.selectbox(
        "Slope",
        [1, 2, 3]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thal",
        [3, 6, 7]
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Heart Disease", use_container_width=True):

    patient = np.array([[
        age,
        1 if sex == "Male" else 0,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("❤️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    confidence = max(probability) * 100

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.progress(float(confidence / 100))

st.divider()

st.caption(
    "⚠️ This application is for educational purposes only and should "
    "not be used as a substitute for professional medical advice."
)