# ==========================================
# MediAssist AI - Intelligent Healthcare Assistant
# app.py
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    # If model files already exist, load them
    if os.path.exists("disease_model.pkl") and os.path.exists("label_encoder.pkl"):

        model = joblib.load("disease_model.pkl")
        encoder = joblib.load("label_encoder.pkl")

        return model, encoder

    # ---------------------------------
    # Train model automatically
    # ---------------------------------

    df = pd.read_csv("disease_dataset.csv")

    X = df.drop("Disease", axis=1)

    y = df["Disease"]

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(
        model,
        "disease_model.pkl"
    )

    joblib.dump(
        encoder,
        "label_encoder.pkl"
    )

    return model, encoder

model, encoder = load_model()

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_medicine_data():
    try:
        return pd.read_csv("medicines.csv")
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_blood_ranges():
    try:
        return pd.read_csv("blood_ranges.csv")
    except Exception:
        return pd.DataFrame()


medicine_df = load_medicine_data()
blood_df = load_blood_ranges()

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main{
    background-color:#F4F9FF;
}

.title{
    text-align:center;
    color:#0F4C81;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.card{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

.metric-box{
    background:#E8F3FF;
    padding:15px;
    border-radius:10px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

if "patient_age" not in st.session_state:
    st.session_state.patient_age = 18

if "patient_gender" not in st.session_state:
    st.session_state.patient_gender = "Male"

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🏥 MediAssist AI")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "👤 Patient Details",
        "🤖 Disease Prediction",
        "⚖ BMI Calculator",
        "🩸 Blood Report Analyzer",
        "💊 Medicine Search",
        "📄 Prescription Reader",
        "❤️ Health Tips",
        "🤖 Health Chatbot",
        "ℹ About"
    ]
)

# ==========================================
# HOME PAGE
# ==========================================

if menu == "🏠 Home":

    st.markdown(
        "<div class='title'>🏥 MediAssist AI</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Intelligent Healthcare Assistant</div>",
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🤖 AI Disease Prediction")

    with col2:
        st.success("⚖ BMI Calculator")

    with col3:
        st.warning("🩸 Blood Report Analysis")

    st.write("")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.info("💊 Medicine Information")

    with col5:
        st.success("📄 Prescription OCR")

    with col6:
        st.warning("🤖 Health Chatbot")

    st.divider()

    st.subheader("✨ Features")

    st.markdown("""
    - AI Disease Prediction using Random Forest
    - BMI Calculator
    - Blood Report Analysis
    - Medicine Information Search
    - Prescription OCR
    - AI Healthcare Chatbot
    - Health Recommendations
    """)

    st.divider()

    

# ==========================================
# PATIENT DETAILS
# ==========================================

elif menu == "👤 Patient Details":

    st.title("👤 Patient Details")

    st.session_state.patient_name = st.text_input(
        "Patient Name"
    )

    st.session_state.patient_age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=20
    )

    st.session_state.patient_gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    height = st.number_input(
        "Height (cm)",
        min_value=50,
        max_value=250,
        value=170
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=10,
        max_value=250,
        value=65
    )

    blood_group = st.selectbox(
        "Blood Group",
        [
            "A+", "A-",
            "B+", "B-",
            "AB+", "AB-",
            "O+", "O-"
        ]
    )

    allergies = st.text_area(
        "Known Allergies"
    )

    history = st.text_area(
        "Medical History"
    )

    if st.button("Save Details"):

        st.success("Patient details saved successfully")

        st.write("### Patient Summary")

        st.write(
            f"**Name:** {st.session_state.patient_name}"
        )

        st.write(
            f"**Age:** {st.session_state.patient_age}"
        )

        st.write(
            f"**Gender:** {st.session_state.patient_gender}"
        )

        st.write(
            f"**Blood Group:** {blood_group}"
        )

        st.write(
            f"**Allergies:** {allergies}"
        )

        st.write(
            f"**Medical History:** {history}"
        )

# ==========================================
# AI DISEASE PREDICTION
# ==========================================

elif menu == "🤖 Disease Prediction":

    st.title("🤖 AI Disease Prediction")

    st.write("### Select Your Symptoms")

    col1, col2 = st.columns(2)

    symptoms = {}

    with col1:
        symptoms["Fever"] = st.checkbox("Fever")
        symptoms["Cough"] = st.checkbox("Cough")
        symptoms["Headache"] = st.checkbox("Headache")
        symptoms["Vomiting"] = st.checkbox("Vomiting")
        symptoms["Fatigue"] = st.checkbox("Fatigue")
        symptoms["BodyPain"] = st.checkbox("Body Pain")
        symptoms["SoreThroat"] = st.checkbox("Sore Throat")
        symptoms["ChestPain"] = st.checkbox("Chest Pain")
        symptoms["RunnyNose"] = st.checkbox("Runny Nose")
        symptoms["Sneezing"] = st.checkbox("Sneezing")
        symptoms["Nausea"] = st.checkbox("Nausea")
        symptoms["Diarrhea"] = st.checkbox("Diarrhea")
        symptoms["AbdominalPain"] = st.checkbox("Abdominal Pain")
        symptoms["Dizziness"] = st.checkbox("Dizziness")
        symptoms["Chills"] = st.checkbox("Chills")
        symptoms["LossOfTaste"] = st.checkbox("Loss Of Taste")
        symptoms["LossOfSmell"] = st.checkbox("Loss Of Smell")
        symptoms["ShortnessOfBreath"] = st.checkbox("Shortness Of Breath")
        symptoms["JointPain"] = st.checkbox("Joint Pain")
        symptoms["Rash"] = st.checkbox("Skin Rash")

    with col2:
        symptoms["Itching"] = st.checkbox("Itching")
        symptoms["HighBP"] = st.checkbox("High Blood Pressure")
        symptoms["LowBP"] = st.checkbox("Low Blood Pressure")
        symptoms["HighSugar"] = st.checkbox("High Blood Sugar")
        symptoms["FrequentUrination"] = st.checkbox("Frequent Urination")
        symptoms["BlurredVision"] = st.checkbox("Blurred Vision")
        symptoms["WeightLoss"] = st.checkbox("Weight Loss")
        symptoms["WeightGain"] = st.checkbox("Weight Gain")
        symptoms["Swelling"] = st.checkbox("Swelling")
        symptoms["BackPain"] = st.checkbox("Back Pain")
        symptoms["EarPain"] = st.checkbox("Ear Pain")
        symptoms["EyePain"] = st.checkbox("Eye Pain")
        symptoms["NeckPain"] = st.checkbox("Neck Pain")
        symptoms["Anxiety"] = st.checkbox("Anxiety")
        symptoms["Depression"] = st.checkbox("Depression")
        symptoms["Insomnia"] = st.checkbox("Insomnia")
        symptoms["Palpitations"] = st.checkbox("Palpitations")
        symptoms["Wheezing"] = st.checkbox("Wheezing")
        symptoms["Constipation"] = st.checkbox("Constipation")
        symptoms["Acidity"] = st.checkbox("Acidity")

    st.divider()

    if st.button("Predict Disease", type="primary"):

        input_data = []

        feature_order = [
            "Fever", "Cough", "Headache", "Vomiting", "Fatigue",
            "BodyPain", "SoreThroat", "ChestPain", "RunnyNose",
            "Sneezing", "Nausea", "Diarrhea", "AbdominalPain",
            "Dizziness", "Chills", "LossOfTaste", "LossOfSmell",
            "ShortnessOfBreath", "JointPain", "Rash", "Itching",
            "HighBP", "LowBP", "HighSugar", "FrequentUrination",
            "BlurredVision", "WeightLoss", "WeightGain",
            "Swelling", "BackPain", "EarPain", "EyePain",
            "NeckPain", "Anxiety", "Depression", "Insomnia",
            "Palpitations", "Wheezing", "Constipation", "Acidity"
        ]

        for symptom in feature_order:
            if symptoms[symptom]:
                input_data.append(1)
            else:
                input_data.append(0)

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_order
        )

        # ==========================================
        # AI MODEL PREDICTION
        # ==========================================

        prediction = model.predict(input_df)

        predicted_disease = encoder.inverse_transform(prediction)[0]

        probability = model.predict_proba(input_df)

        confidence = np.max(probability) * 100

        # ==========================================
        # DOCTOR RECOMMENDATION
        # ==========================================

        doctor = {
            "Flu": "General Physician",
            "Common Cold": "General Physician",
            "COVID-19": "General Physician",
            "Typhoid": "General Physician",
            "Migraine": "Neurologist",
            "Food Poisoning": "General Physician",
            "Heart Disease": "Cardiologist",
            "Diabetes": "Endocrinologist",
            "Hypertension": "Cardiologist",
            "Asthma": "Pulmonologist",
            "Allergy": "Dermatologist / Allergist",
            "Arthritis": "Orthopedic Specialist",
            "Depression": "Psychiatrist",
            "Anxiety Disorder": "Psychologist",
            "GERD": "Gastroenterologist",
            "Constipation": "Gastroenterologist",
            "Ear Infection": "ENT Specialist",
            "Conjunctivitis": "Ophthalmologist",
            "Back Strain": "Orthopedic Specialist",
            "Dengue": "General Physician"
        }

        specialist = doctor.get(
            predicted_disease,
            "General Physician"
        )

        # ==========================================
        # PRECAUTIONS
        # ==========================================

        precautions = {

            "Flu": [
                "Drink plenty of fluids",
                "Take adequate rest",
                "Monitor temperature"
            ],

            "Common Cold": [
                "Drink warm fluids",
                "Take sufficient rest",
                "Avoid cold drinks"
            ],

            "COVID-19": [
                "Wear a mask",
                "Stay isolated if advised",
                "Consult a doctor"
            ],

            "Typhoid": [
                "Drink boiled water",
                "Eat hygienic food",
                "Complete prescribed medicines"
            ],

            "Migraine": [
                "Avoid bright lights",
                "Stay hydrated",
                "Get enough sleep"
            ],

            "Food Poisoning": [
                "Drink ORS",
                "Stay hydrated",
                "Avoid oily food"
            ],

            "Heart Disease": [
                "Seek immediate medical care",
                "Avoid strenuous activity",
                "Take prescribed medicines"
            ],

            "Diabetes": [
                "Monitor blood sugar",
                "Exercise regularly",
                "Reduce sugar intake"
            ],

            "Hypertension": [
                "Reduce salt intake",
                "Exercise daily",
                "Monitor blood pressure"
            ],

            "Asthma": [
                "Avoid dust",
                "Carry inhaler",
                "Consult pulmonologist"
            ],

            "Allergy": [
                "Avoid allergens",
                "Keep surroundings clean",
                "Consult doctor if severe"
            ],

            "Arthritis": [
                "Exercise regularly",
                "Maintain healthy weight",
                "Avoid joint strain"
            ],

            "Depression": [
                "Talk to a professional",
                "Exercise regularly",
                "Maintain healthy sleep"
            ],

            "Anxiety Disorder": [
                "Practice breathing exercises",
                "Reduce stress",
                "Seek counselling"
            ],

            "GERD": [
                "Avoid spicy food",
                "Eat smaller meals",
                "Don't lie down immediately"
            ],

            "Constipation": [
                "Drink water",
                "Increase fibre intake",
                "Exercise daily"
            ],

            "Ear Infection": [
                "Keep ears dry",
                "Avoid inserting objects",
                "Visit ENT specialist"
            ],

            "Conjunctivitis": [
                "Avoid touching eyes",
                "Wash hands frequently",
                "Do not share towels"
            ],

            "Back Strain": [
                "Avoid heavy lifting",
                "Use proper posture",
                "Apply cold or warm compress"
            ],

            "Dengue": [
                "Drink plenty of fluids",
                "Monitor platelet count",
                "Consult doctor immediately"
            ]
        }

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================

        st.success("Prediction Completed Successfully")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Disease",
                predicted_disease
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with col2:
            st.metric(
                "Recommended Specialist",
                specialist
            )

        st.divider()

        st.subheader("General Precautions")

        for item in precautions.get(
            predicted_disease,
            ["Consult a qualified doctor."]
        ):
            st.write("✅", item)

        st.divider()

        st.subheader("Selected Symptoms")

        selected = [
            symptom
            for symptom, value in symptoms.items()
            if value
        ]

        if selected:
            st.write(", ".join(selected))
        else:
            st.warning("No symptoms selected.")

        st.error(
            "⚠ This AI prediction is for educational purposes only. "
            "It is not a medical diagnosis. Always consult a qualified healthcare professional."
        )

# ==========================================
# BMI CALCULATOR
# ==========================================

elif menu == "⚖ BMI Calculator":

    st.title("⚖ BMI Calculator")

    col1, col2 = st.columns(2)

    with col1:
        height = st.number_input(
            "Height (cm)",
            min_value=50,
            max_value=250,
            value=170
        )

    with col2:
        weight = st.number_input(
            "Weight (kg)",
            min_value=10,
            max_value=250,
            value=65
        )

    if st.button("Calculate BMI"):

        bmi = weight / ((height / 100) ** 2)

        st.metric("BMI", f"{bmi:.2f}")

        if bmi < 18.5:
            status = "Underweight"
            color = "warning"
            tips = [
                "Increase protein intake",
                "Eat healthy meals",
                "Exercise regularly"
            ]

        elif bmi < 25:
            status = "Normal"
            color = "success"
            tips = [
                "Maintain healthy diet",
                "Continue regular exercise",
                "Drink enough water"
            ]

        elif bmi < 30:
            status = "Overweight"
            color = "warning"
            tips = [
                "Reduce junk food",
                "Walk 30 minutes daily",
                "Drink more water"
            ]

        else:
            status = "Obese"
            color = "error"
            tips = [
                "Consult a doctor",
                "Follow a calorie-controlled diet",
                "Exercise regularly"
            ]

        if color == "success":
            st.success(status)
        elif color == "warning":
            st.warning(status)
        else:
            st.error(status)

        st.subheader("Health Tips")

        for tip in tips:
            st.write("✅", tip)

        st.info(
            "BMI is only a screening tool and does not diagnose body fat or medical conditions."
        )

# ==========================================
# BLOOD REPORT ANALYZER
# ==========================================

elif menu == "🩸 Blood Report Analyzer":

    st.title("🩸 Blood Report Analyzer")

    st.write("Enter your blood test values.")

    col1, col2 = st.columns(2)

    with col1:
        hemoglobin = st.number_input(
            "Hemoglobin (g/dL)",
            0.0,
            25.0,
            13.5
        )

        wbc = st.number_input(
            "WBC Count (/µL)",
            0,
            30000,
            7000
        )

        rbc = st.number_input(
            "RBC Count (million/µL)",
            0.0,
            10.0,
            5.0
        )

    with col2:
        platelets = st.number_input(
            "Platelets (/µL)",
            0,
            1000000,
            250000
        )

        sugar = st.number_input(
            "Blood Sugar (mg/dL)",
            0,
            500,
            100
        )

    if st.button("Analyze Report"):

        st.subheader("Analysis Result")

        # Hemoglobin
        if hemoglobin < 12:
            st.error("🔴 Low Hemoglobin (Possible Anemia)")
        elif hemoglobin > 17:
            st.warning("🟡 High Hemoglobin")
        else:
            st.success("🟢 Hemoglobin Normal")

        # WBC
        if wbc < 4000:
            st.error("🔴 Low WBC Count")
        elif wbc > 11000:
            st.warning("🟡 High WBC Count")
        else:
            st.success("🟢 WBC Count Normal")

        # RBC
        if rbc < 4.2:
            st.error("🔴 Low RBC Count")
        elif rbc > 6.1:
            st.warning("🟡 High RBC Count")
        else:
            st.success("🟢 RBC Count Normal")

        # Platelets
        if platelets < 150000:
            st.error("🔴 Low Platelet Count")
        elif platelets > 450000:
            st.warning("🟡 High Platelet Count")
        else:
            st.success("🟢 Platelet Count Normal")

        # Blood Sugar
        if sugar < 70:
            st.error("🔴 Low Blood Sugar")
        elif sugar > 140:
            st.warning("🟡 High Blood Sugar")
        else:
            st.success("🟢 Blood Sugar Normal")

        st.divider()

        st.subheader("Doctor Recommendation")

        if sugar > 140:
            st.write("✅ Endocrinologist")

        if platelets < 150000:
            st.write("✅ General Physician")

        if hemoglobin < 12:
            st.write("✅ Hematologist")

        if wbc > 11000:
            st.write("✅ General Physician")

        if (
            12 <= hemoglobin <= 17
            and 4000 <= wbc <= 11000
            and 4.2 <= rbc <= 6.1
            and 150000 <= platelets <= 450000
            and 70 <= sugar <= 140
        ):
            st.success(
                "Your blood report appears to be within normal limits."
            )

        st.divider()

        st.warning(
            "This analysis is for educational purposes only and should not replace professional medical advice."
        )

# ==========================================
# MEDICINE SEARCH
# ==========================================

elif menu == "💊 Medicine Search":

    st.title("💊 Medicine Information")

    if medicine_df.empty:
        st.error("medicines.csv not found.")

    else:
        medicine_name = st.text_input(
            "Search Medicine"
        )

        if medicine_name:
            result = medicine_df[
                medicine_df["Medicine"].str.lower()
                ==
                medicine_name.lower()
            ]

            if result.empty:
                st.warning("Medicine not found.")

            else:
                st.success("Medicine Found")

                medicine = result.iloc[0]

                st.subheader(medicine["Medicine"])

                st.write("### Uses")
                st.info(medicine["Uses"])

                st.write("### Dosage")
                st.success(medicine["Dosage"])

                st.write("### Side Effects")
                st.warning(medicine["Side Effects"])

                st.write("### Warnings")
                st.error(medicine["Warnings"])

# ==========================================
# PRESCRIPTION READER (OCR)
# ==========================================

elif menu == "📄 Prescription Reader":

    import easyocr

    st.title("📄 AI Prescription Reader")

    uploaded_file = st.file_uploader(
        "Upload Prescription Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Prescription",
            use_container_width=True
        )

        if st.button("Read Prescription"):

            with st.spinner("Reading Prescription..."):

                reader = easyocr.Reader(['en'])

                result = reader.readtext(
                    np.array(image)
                )

                extracted_text = ""

                for item in result:
                    extracted_text += item[1] + "\n"

                st.subheader("Extracted Text")

                st.text_area(
                    "Result",
                    extracted_text,
                    height=250
                )

                st.download_button(
                    "Download Text",
                    extracted_text,
                    file_name="prescription.txt"
                )

# ==========================================
# HEALTH TIPS
# ==========================================

elif menu == "❤️ Health Tips":

    st.title("❤️ Daily Health Tips")

    tips = [
        "Drink at least 2-3 litres of water daily.",
        "Sleep for 7-8 hours every night.",
        "Exercise for at least 30 minutes.",
        "Eat fresh fruits and vegetables.",
        "Reduce sugar and salt intake.",
        "Avoid smoking and alcohol.",
        "Wash your hands regularly.",
        "Take medicines only as prescribed.",
        "Manage stress through meditation.",
        "Schedule regular health checkups."
    ]

    for tip in tips:
        st.success("✅ " + tip)

# ==========================================
# AI HEALTH CHATBOT
# ==========================================

elif menu == "🤖 Health Chatbot":

    st.title("🤖 AI Health Chatbot")

    st.info(
        "Ask simple health-related questions. "
        "This chatbot is for educational purposes only."
    )

    question = st.text_input(
        "Ask your question"
    )

    if st.button("Ask AI"):

        q = question.lower()

        answer = (
            "Please consult a qualified healthcare professional "
            "for accurate medical advice."
        )

        if "fever" in q:
            answer = (
                "Fever may indicate an infection. "
                "Drink plenty of fluids, take adequate rest, "
                "and consult a doctor if it lasts more than 2 days."
            )

        elif "cold" in q or "cough" in q:
            answer = (
                "A common cold usually improves with rest, warm fluids, "
                "and hydration. If symptoms become severe, visit a doctor."
            )

        elif "headache" in q:
            answer = (
                "Headaches can occur due to stress, dehydration, "
                "or lack of sleep. Drink water and rest."
            )

        elif "diabetes" in q:
            answer = (
                "Monitor your blood sugar regularly, "
                "follow a balanced diet, and exercise daily."
            )

        elif "blood pressure" in q:
            answer = (
                "Monitor your blood pressure regularly, "
                "reduce salt intake, and exercise."
            )

        elif "covid" in q:
            answer = (
                "If you have COVID-like symptoms, "
                "consult a healthcare provider and follow local guidelines."
            )

        elif "dengue" in q:
            answer = (
                "Drink plenty of fluids, monitor your platelet count, "
                "and seek medical care immediately."
            )

        elif "heart" in q:
            answer = (
                "Chest pain or symptoms related to the heart "
                "require immediate medical attention."
            )

        elif "vomiting" in q:
            answer = (
                "Drink ORS or other fluids to stay hydrated. "
                "Consult a doctor if vomiting is persistent."
            )

        elif "medicine" in q:
            answer = (
                "Always take medicines only as prescribed "
                "by a qualified healthcare professional."
            )

        st.success(answer)

# ==========================================
# ABOUT PAGE
# ==========================================

elif menu == "ℹ About":

    st.title("ℹ About MediAssist AI")

    st.markdown("""
# 🏥 MediAssist AI

An AI-powered healthcare assistant developed using:

- Streamlit
- Python
- Machine Learning
- Random Forest Classifier
- EasyOCR
- Pandas
- NumPy
- Scikit-Learn

## Features

✅ Disease Prediction

✅ BMI Calculator

✅ Blood Report Analysis

✅ Medicine Information

✅ Prescription OCR

✅ Health Chatbot

## Developer

Developed by:

**Dharshini Natarajan**

B.Sc Computer Science with Artificial Intelligence

## Version

Version 1.0
""")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "© 2026 MediAssist AI | "
    "Educational Project | "
    "Not a substitute for professional medical advice."
)
