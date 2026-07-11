# ==========================================
# train_model.py
# AI Smart Medical Assistant
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("=" * 50)
print("AI SMART MEDICAL ASSISTANT")
print("Disease Prediction Model Training")
print("=" * 50)

# -----------------------------------------
# Load Dataset
# -----------------------------------------

try:
    df = pd.read_csv("disease_dataset.csv")
    print("\nDataset Loaded Successfully.")
except FileNotFoundError:
    print("\nERROR: disease_dataset.csv not found.")
    exit()

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows")
print(df.head())

# -----------------------------------------
# Check Missing Values
# -----------------------------------------

print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------------------
# Features & Target
# -----------------------------------------

X = df.drop("Disease", axis=1)
y = df["Disease"]

print("\nFeatures:", len(X.columns))
print("Target Classes:", y.nunique())

# -----------------------------------------
# Encode Disease Labels
# -----------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nDisease Labels")

for i, disease in enumerate(label_encoder.classes_):
    print(i, "->", disease)

# -----------------------------------------
# Train Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -----------------------------------------
# Random Forest Model
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Completed.")

# -----------------------------------------
# Prediction
# -----------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy")

print(round(accuracy * 100, 2), "%")

# -----------------------------------------
# Classification Report
# -----------------------------------------

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# -----------------------------------------
# Confusion Matrix
# -----------------------------------------

print("\nConfusion Matrix")

print(confusion_matrix(y_test, y_pred))

# -----------------------------------------
# Feature Importance
# -----------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Symptoms")

print(importance.head(15))

# -----------------------------------------
# Save Model
# -----------------------------------------

joblib.dump(model, "disease_model.pkl")

joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel Saved Successfully")

print("File : disease_model.pkl")

print("Label Encoder Saved")

print("File : label_encoder.pkl")

# -----------------------------------------
# Sample Prediction
# -----------------------------------------

print("\nTesting Sample Prediction")

sample = X.iloc[[0]]

prediction = model.predict(sample)

disease = label_encoder.inverse_transform(prediction)

probability = model.predict_proba(sample)

confidence = probability.max() * 100

print("\nPredicted Disease :", disease[0])

print("Confidence :", round(confidence, 2), "%")

print("\nTraining Completed Successfully!")

print("=" * 50)
