# 🏥 MediAssist AI - Smart Medical Assistant

An AI-powered healthcare assistant built using **Python**, **Streamlit**, and **Google Gemini API**. MediAssist AI provides essential healthcare tools such as a BMI calculator, medicine information search, AI-powered prescription reading, health tips, and a medical chatbot through a simple and user-friendly interface.

> **Note:** This project is developed for educational purposes and is not a substitute for professional medical advice.

---

## 📌 Features

- ⚖️ **BMI Calculator**
  - Calculates Body Mass Index (BMI)
  - Displays health category and recommendations

- 💊 **Medicine Information Search**
  - Search medicines from a built-in dataset
  - View uses, dosage, side effects, and warnings

- 📄 **AI Prescription Reader**
  - Upload a prescription image
  - Extract medicine names and prescription details using Google Gemini AI

- 🤖 **AI Health Chatbot**
  - Answers basic health-related questions
  - Provides information on symptoms, causes, treatments, prevention, and specialists

- ❤️ **Daily Health Tips**
  - Displays useful health and wellness tips

- 🎨 **Interactive Streamlit Interface**
  - Clean, responsive, and easy-to-use web application

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Google Gemini API
- Pandas
- NumPy
- Pillow
- CSV Dataset

---

## 📂 Project Structure

```
MediAssist-AI/
│
├── app.py
├── medicines.csv
├── requirements.txt
├── .streamlit/
│   └── secrets.toml
├── assets/
│   └── screenshots
└── README.md
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/your-username/MediAssist-AI.git
cd MediAssist-AI
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add your Gemini API Key

Create the following file:

```
.streamlit/secrets.toml
```

Add:

```toml
GEMINI_API_KEY="YOUR_API_KEY"
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

### Home Page
<img width="1366" height="625" alt="image" src="https://github.com/user-attachments/assets/ce7a7f42-148c-4d33-8962-29e03f0f65f1" />


### BMI Calculator

<img width="1272" height="620" alt="image" src="https://github.com/user-attachments/assets/65ca2f2f-9dd4-46fb-84af-4bda74d855a7" />

### Medicine Search

<img width="1364" height="622" alt="image" src="https://github.com/user-attachments/assets/5dccc60e-af9c-4283-91d2-8955a67d3e48" />

### Prescription Reader

<img width="1366" height="631" alt="image" src="https://github.com/user-attachments/assets/329af4d7-1246-49e7-9c2e-32da6b8c7aaa" />

### Health Chatbot

<img width="1366" height="640" alt="image" src="https://github.com/user-attachments/assets/bb1026a9-8a54-49c7-b875-b82241f07ab5" />

## 🌐 Live Demo

https://ai-smart-medical-assistant-project.streamlit.app/

---

## 📚 Future Improvements

- Voice Assistant
- Appointment Booking
- Disease Prediction
- Medicine Recommendation
- Nearby Hospital Locator
- Multi-language Support
- PDF Prescription Export

---

## 👩‍💻 Developer

**Dharshini Natarajan**

B.Sc. Computer Science with Artificial Intelligence

GitHub: https://github.com/dharshini-36

LinkedIn: *(Add your LinkedIn profile link)*

---

## 📄 License

This project is developed for educational and learning purposes.

---

⭐ If you found this project useful, please consider giving it a star!
