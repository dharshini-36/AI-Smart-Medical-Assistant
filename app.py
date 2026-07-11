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


medicine_df = load_medicine_data()

# ==========================================
# HEALTH KNOWLEDGE BASE (used by chatbot)
# ==========================================

medical_data = {

    "fever":{

        "symptoms":[
            "High body temperature",
            "Chills",
            "Headache",
            "Body pain",
            "Weakness"
        ],

        "causes":[
            "Viral infection",
            "Bacterial infection",
            "Flu",
            "COVID-19"
        ],

        "treatment":[
            "Drink fluids",
            "Take rest",
            "Paracetamol (if prescribed)"
        ],

        "prevention":[
            "Wash hands",
            "Drink clean water",
            "Maintain hygiene"
        ],

        "doctor":"General Physician"

    },

    "diabetes":{

        "symptoms":[
            "Frequent urination",
            "Excessive thirst",
            "Blurred vision",
            "Weight loss"
        ],

        "causes":[
            "High blood sugar",
            "Insulin resistance"
        ],

        "treatment":[
            "Healthy diet",
            "Exercise",
            "Medicines prescribed by doctor"
        ],

        "prevention":[
            "Exercise",
            "Healthy food",
            "Maintain weight"
        ],

        "doctor":"Endocrinologist"

    },

    "flu": {
        "symptoms": [
            "High fever",
            "Cough",
            "Body pain",
            "Fatigue",
            "Headache",
            "Chills"
        ],
        "causes": [
            "Influenza virus"
        ],
        "treatment": [
            "Take adequate rest",
            "Drink plenty of fluids",
            "Take medicines prescribed by your doctor"
        ],
        "prevention": [
            "Wash hands frequently",
            "Cover mouth while coughing",
            "Annual flu vaccination"
        ],
        "doctor": "General Physician"
    },

    "common cold": {
        "symptoms": [
            "Runny nose",
            "Sneezing",
            "Sore throat",
            "Cough",
            "Headache"
        ],
        "causes": [
            "Rhinovirus infection"
        ],
        "treatment": [
            "Drink warm fluids",
            "Take rest",
            "Steam inhalation"
        ],
        "prevention": [
            "Wash hands",
            "Avoid close contact with infected people"
        ],
        "doctor": "General Physician"
    },

    "covid-19": {
        "symptoms": [
            "Fever",
            "Cough",
            "Loss of smell",
            "Loss of taste",
            "Shortness of breath",
            "Fatigue"
        ],
        "causes": [
            "SARS-CoV-2 virus"
        ],
        "treatment": [
            "Rest",
            "Drink fluids",
            "Consult doctor if symptoms worsen"
        ],
        "prevention": [
            "Wear a mask",
            "Wash hands",
            "Maintain social distancing"
        ],
        "doctor": "General Physician"
    },

    "typhoid": {
        "symptoms": [
            "High fever",
            "Headache",
            "Abdominal pain",
            "Weakness",
            "Loss of appetite"
        ],
        "causes": [
            "Salmonella Typhi bacteria"
        ],
        "treatment": [
            "Complete antibiotic course",
            "Drink clean water",
            "Eat soft foods"
        ],
        "prevention": [
            "Drink boiled water",
            "Eat hygienic food"
        ],
        "doctor": "General Physician"
    },

    "migraine": {
        "symptoms": [
            "Severe headache",
            "Nausea",
            "Vomiting",
            "Sensitivity to light",
            "Blurred vision"
        ],
        "causes": [
            "Stress",
            "Hormonal changes",
            "Lack of sleep"
        ],
        "treatment": [
            "Rest in a dark room",
            "Take prescribed medicine",
            "Stay hydrated"
        ],
        "prevention": [
            "Sleep well",
            "Avoid stress",
            "Avoid trigger foods"
        ],
        "doctor": "Neurologist"
    },

    "food poisoning": {
        "symptoms": [
            "Nausea",
            "Vomiting",
            "Diarrhea",
            "Abdominal pain",
            "Fever",
            "Dehydration"
        ],
        "causes": [
            "Contaminated food",
            "Bacteria",
            "Viruses",
            "Food toxins"
        ],
        "treatment": [
            "Drink ORS",
            "Stay hydrated",
            "Eat light foods",
            "Consult a doctor if symptoms are severe"
        ],
        "prevention": [
            "Eat freshly cooked food",
            "Wash hands before eating",
            "Store food properly"
        ],
        "doctor": "General Physician"
    },

    "heart disease": {
        "symptoms": [
            "Chest pain",
            "Shortness of breath",
            "Fatigue",
            "Palpitations",
            "Dizziness"
        ],
        "causes": [
            "High cholesterol",
            "High blood pressure",
            "Smoking",
            "Diabetes",
            "Obesity"
        ],
        "treatment": [
            "Take prescribed medicines",
            "Maintain a healthy diet",
            "Exercise regularly",
            "Regular medical checkups"
        ],
        "prevention": [
            "Avoid smoking",
            "Exercise daily",
            "Eat a heart-healthy diet",
            "Control blood pressure"
        ],
        "doctor": "Cardiologist"
    },

    "diabetes": {
        "symptoms": [
            "Frequent urination",
            "Excessive thirst",
            "Blurred vision",
            "Weight loss",
            "Fatigue"
        ],
        "causes": [
            "Insulin deficiency",
            "Insulin resistance",
            "Genetic factors"
        ],
        "treatment": [
            "Monitor blood sugar",
            "Exercise regularly",
            "Follow a healthy diet",
            "Take prescribed medicines"
        ],
        "prevention": [
            "Maintain healthy weight",
            "Exercise daily",
            "Reduce sugar intake"
        ],
        "doctor": "Endocrinologist"
    },

    "hypertension": {
        "symptoms": [
            "Headache",
            "Dizziness",
            "Chest pain",
            "Blurred vision",
            "Fatigue"
        ],
        "causes": [
            "High salt intake",
            "Stress",
            "Obesity",
            "Family history"
        ],
        "treatment": [
            "Take blood pressure medicines",
            "Reduce salt intake",
            "Exercise regularly"
        ],
        "prevention": [
            "Eat healthy foods",
            "Reduce stress",
            "Maintain healthy weight"
        ],
        "doctor": "Cardiologist"
    },

    "asthma": {
        "symptoms": [
            "Wheezing",
            "Shortness of breath",
            "Cough",
            "Chest tightness"
        ],
        "causes": [
            "Allergies",
            "Dust",
            "Cold air",
            "Respiratory infections"
        ],
        "treatment": [
            "Use inhaler",
            "Take prescribed medicines",
            "Avoid triggers"
        ],
        "prevention": [
            "Avoid dust",
            "Do not smoke",
            "Carry inhaler"
        ],
        "doctor": "Pulmonologist"
    },

    "allergy": {
        "symptoms": [
            "Skin rash",
            "Itching",
            "Sneezing",
            "Runny nose",
            "Watery eyes"
        ],
        "causes": [
            "Dust",
            "Pollen",
            "Pet dander",
            "Certain foods",
            "Medicines"
        ],
        "treatment": [
            "Take antihistamines",
            "Avoid allergens",
            "Consult a doctor if symptoms worsen"
        ],
        "prevention": [
            "Keep surroundings clean",
            "Avoid known allergens",
            "Wear a mask in dusty areas"
        ],
        "doctor": "Dermatologist / Allergist"
    },

    "arthritis": {
        "symptoms": [
            "Joint pain",
            "Joint stiffness",
            "Swelling",
            "Reduced movement"
        ],
        "causes": [
            "Age",
            "Joint wear and tear",
            "Autoimmune disorders"
        ],
        "treatment": [
            "Exercise regularly",
            "Pain relief medicines",
            "Physiotherapy"
        ],
        "prevention": [
            "Maintain healthy weight",
            "Exercise daily",
            "Avoid joint injuries"
        ],
        "doctor": "Orthopedic Specialist"
    },

    "depression": {
        "symptoms": [
            "Persistent sadness",
            "Loss of interest",
            "Fatigue",
            "Sleep problems",
            "Difficulty concentrating"
        ],
        "causes": [
            "Stress",
            "Genetics",
            "Chemical imbalance",
            "Life events"
        ],
        "treatment": [
            "Counselling",
            "Psychotherapy",
            "Medicines prescribed by psychiatrist"
        ],
        "prevention": [
            "Exercise regularly",
            "Maintain social connections",
            "Seek help early"
        ],
        "doctor": "Psychiatrist"
    },

    "anxiety disorder": {
        "symptoms": [
            "Excessive worry",
            "Restlessness",
            "Fast heartbeat",
            "Sweating",
            "Difficulty sleeping"
        ],
        "causes": [
            "Stress",
            "Genetics",
            "Traumatic experiences"
        ],
        "treatment": [
            "Counselling",
            "Breathing exercises",
            "Medicines if prescribed"
        ],
        "prevention": [
            "Meditation",
            "Regular exercise",
            "Adequate sleep"
        ],
        "doctor": "Psychologist / Psychiatrist"
    },

    "gerd": {
        "symptoms": [
            "Heartburn",
            "Acid reflux",
            "Chest discomfort",
            "Difficulty swallowing"
        ],
        "causes": [
            "Weak lower esophageal sphincter",
            "Obesity",
            "Spicy foods",
            "Large meals"
        ],
        "treatment": [
            "Take antacids or prescribed medicines",
            "Avoid spicy foods",
            "Eat smaller meals"
        ],
        "prevention": [
            "Maintain healthy weight",
            "Avoid lying down after meals",
            "Reduce caffeine intake"
        ],
        "doctor": "Gastroenterologist"
    },

    "constipation": {
        "symptoms": [
            "Infrequent bowel movements",
            "Hard stools",
            "Abdominal discomfort",
            "Bloating"
        ],
        "causes": [
            "Low fiber diet",
            "Not drinking enough water",
            "Lack of exercise"
        ],
        "treatment": [
            "Drink more water",
            "Eat fiber-rich foods",
            "Exercise regularly"
        ],
        "prevention": [
            "Eat fruits and vegetables",
            "Drink 2-3 litres of water daily",
            "Stay physically active"
        ],
        "doctor": "Gastroenterologist"
    },

    "ear infection": {
        "symptoms": [
            "Ear pain",
            "Difficulty hearing",
            "Fever",
            "Fluid discharge from ear"
        ],
        "causes": [
            "Bacterial infection",
            "Viral infection"
        ],
        "treatment": [
            "Take prescribed antibiotics",
            "Keep ear dry",
            "Consult an ENT specialist"
        ],
        "prevention": [
            "Maintain ear hygiene",
            "Avoid inserting objects into the ear"
        ],
        "doctor": "ENT Specialist"
    },

    "conjunctivitis": {
        "symptoms": [
            "Red eyes",
            "Itchy eyes",
            "Watery eyes",
            "Eye discharge"
        ],
        "causes": [
            "Virus",
            "Bacteria",
            "Allergy"
        ],
        "treatment": [
            "Use prescribed eye drops",
            "Keep eyes clean",
            "Avoid touching eyes"
        ],
        "prevention": [
            "Wash hands frequently",
            "Do not share towels",
            "Avoid rubbing eyes"
        ],
        "doctor": "Ophthalmologist"
    },

    "back strain": {
        "symptoms": [
            "Back pain",
            "Muscle stiffness",
            "Difficulty bending",
            "Muscle spasms"
        ],
        "causes": [
            "Heavy lifting",
            "Poor posture",
            "Sudden movements"
        ],
        "treatment": [
            "Take rest",
            "Apply ice or heat",
            "Pain relief medicines if prescribed"
        ],
        "prevention": [
            "Maintain proper posture",
            "Exercise regularly",
            "Lift heavy objects correctly"
        ],
        "doctor": "Orthopedic Specialist"
    },

    "dengue": {
        "symptoms": [
            "High fever",
            "Severe headache",
            "Body pain",
            "Joint pain",
            "Skin rash",
            "Nausea"
        ],
        "causes": [
            "Dengue virus spread by Aedes mosquitoes"
        ],
        "treatment": [
            "Drink plenty of fluids",
            "Take adequate rest",
            "Consult a doctor immediately"
        ],
        "prevention": [
            "Use mosquito nets",
            "Remove stagnant water",
            "Use mosquito repellent"
        ],
        "doctor": "General Physician"
    }
}

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
        "⚖ BMI Calculator",
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
        st.warning("💊 Medicine Information")

    st.write("")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.info("📄 Prescription Reader")

    with col5:
        st.success("❤️ Health Tips")

    with col6:
        st.warning("🤖 AI Health Chatbot")

    st.divider()

    st.subheader("✨ Features")

    st.markdown("""
### 🚀 Features

✅ Patient Details Management

✅ BMI Calculator

✅ Medicine Information Search

✅ AI Prescription Reader (OCR)

✅ Daily Health Tips

✅ AI Healthcare Chatbot

✅ Doctor Recommendation

✅ Disease Symptoms, Causes, Treatment & Prevention
""")

# ==========================================
# PATIENT DETAILS
# ==========================================

elif menu == "👤 Patient Details":

    st.title("👤 Patient Details")

    name = st.text_input("Full Name")

    age = st.number_input("Age", 1, 120)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    height = st.number_input(
        "Height (cm)",
        50,
        250
    )

    weight = st.number_input(
        "Weight (kg)",
        10,
        250
    )

    blood_group = st.selectbox(
        "Blood Group",
        [
            "A+","A-","B+","B-",
            "AB+","AB-","O+","O-"
        ]
    )

    allergies = st.text_area(
        "Known Allergies (Optional)"
    )

    if st.button("Save Details"):

        st.session_state["name"] = name
        st.session_state["age"] = age
        st.session_state["gender"] = gender
        st.session_state["height"] = height
        st.session_state["weight"] = weight
        st.session_state["blood_group"] = blood_group
        st.session_state["allergies"] = allergies

        st.success("✅ Patient details saved successfully!")

# ==========================================
# BMI CALCULATOR
# ==========================================

elif menu == "⚖ BMI Calculator":
    if "name" in st.session_state:
        st.write(
        f"### Patient : {st.session_state['name']}"
    )
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

        st.success(
    f"{st.session_state['name']}, your BMI is {bmi:.2f}"
)
        st.info(
            "BMI is only a screening tool and does not diagnose body fat or medical conditions."
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
    if "name" in st.session_state:
        st.write(
        f"### Personalized Tips for {st.session_state['name']}")

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

   if "name" in st.session_state:
       st.title(
        f"🤖 Hello {st.session_state['name']} 👋"
    )
   else:
       st.title("🤖 MediAssist AI Chatbot")

    st.write("Ask questions like:")
    st.write("- Symptoms of Fever")
    st.write("- Causes of Diabetes")
    st.write("- Treatment for Asthma")
    st.write("- Prevention of Dengue")
    st.write("- Doctor for Migraine")
    st.write("- Uses of Paracetamol")

    question = st.text_input("Ask your health question")

    if st.button("Ask"):

        q = question.lower()

        found = False

        # Greetings
        if q in ["hi", "hello", "hey", "good morning", "good evening"]:

            st.success("Hello 👋 Welcome to MediAssist AI!")

            st.info("You can ask about symptoms, causes, treatment, prevention, doctors or medicines.")

            found = True

        # Disease Knowledge
        for disease, info in medical_data.items():

            if disease in q:

                found = True

                if "symptom" in q:

                    st.subheader("Symptoms")

                    for item in info["symptoms"]:
                        st.write("✅", item)

                elif "cause" in q:

                    st.subheader("Causes")

                    for item in info["causes"]:
                        st.write("✅", item)

                elif "treatment" in q:

                    st.subheader("Treatment")

                    for item in info["treatment"]:
                        st.write("✅", item)

                elif "prevent" in q:

                    st.subheader("Prevention")

                    for item in info["prevention"]:
                        st.write("✅", item)

                elif "doctor" in q:

                    st.subheader("Recommended Specialist")

                    st.success(info["doctor"])

                else:

                    st.info("Please ask about symptoms, causes, treatment, prevention or doctor.")

                break

        # Medicine Search
        if not found and not medicine_df.empty:

            for _, row in medicine_df.iterrows():

                if row["Medicine"].lower() in q:

                    found = True

                    st.success(row["Medicine"])

                    st.write("### Uses")
                    st.write(row["Uses"])

                    st.write("### Dosage")
                    st.write(row["Dosage"])

                    st.write("### Side Effects")
                    st.write(row["Side Effects"])

                    st.write("### Warnings")
                    st.write(row["Warnings"])

                    break

        if not found:

            st.warning("Sorry, I couldn't understand your question.")

            st.info("""
Try asking:

• Symptoms of Fever

• Causes of Diabetes

• Treatment for Asthma

• Prevention of Dengue

• Doctor for Migraine

• Uses of Paracetamol
""")

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

""")
st.markdown("---")

st.caption(
    "© 2026 MediAssist AI | "
    "Educational Project | "
    "Not a substitute for professional medical advice."
)
