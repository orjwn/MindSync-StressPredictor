import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib

# === 1. Load Data ===
df = pd.read_csv("MindSync-StressPredictor/data/dataset.csv")
df_cleaned = df.dropna(subset=["stress_level"])

# === 2. Prepare Features ===
summary_text = df_cleaned["summary"].fillna("")
vectorizer = CountVectorizer()
summary_features = vectorizer.fit_transform(summary_text)

X_numeric = df_cleaned[["location", "duration", "morning"]]
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

X_combined = hstack([summary_features, X_numeric_scaled])
y = df_cleaned["stress_level"].astype(int)

# === 3. Split into Train/Test (80/20) ===
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, stratify=y, random_state=42)

# === 4. Train the Model ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === 5. Save Model and Tools ===
joblib.dump(clf, "MindSync-StressPredictor/pickle/rf_model.p")
joblib.dump(vectorizer, "MindSync-StressPredictor/pickle/vectorizer_summary.p")
joblib.dump(scaler, "MindSync-StressPredictor/pickle/scaler.p")
print("Model trained successfully and saved.")

