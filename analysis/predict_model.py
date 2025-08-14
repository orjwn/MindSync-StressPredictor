import pandas as pd
import joblib
from scipy.sparse import hstack

# === Load model, vectorizer, and scaler ===
model = joblib.load("MindSync-StressPredictor/pickle/rf_model.p")
vectorizer = joblib.load("MindSync-StressPredictor/pickle/vectorizer_summary.p")
scaler = joblib.load("MindSync-StressPredictor/pickle/scaler.p")

# === Load test data ===
df = pd.read_csv("MindSync-StressPredictor/data/dataset.csv")  
df = df.dropna(subset=["summary", "location", "duration", "morning"])

# === Extract features ===
summary_text = df["summary"].fillna("")
summary_features = vectorizer.transform(summary_text)

# Normalize numeric features using the same scaler used during training
X_numeric = df[["location", "duration", "morning"]]
X_numeric_scaled = scaler.transform(X_numeric)

# Combine features
X_combined = hstack([summary_features, X_numeric_scaled])

# === Predict ===
predictions = model.predict(X_combined)
df["predicted_stress_level"] = predictions

# === Save ===
df.to_csv("MindSync-StressPredictor/output/predicted_output.csv", index=False)
print(" Predictions completed. Output saved to output/predicted_output.csv")
