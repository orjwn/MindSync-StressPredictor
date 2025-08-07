import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import joblib

# === Load dataset and model ===
df = pd.read_csv("MindSync-StressPredictor/data/dataset.csv").dropna(subset=["stress_level"])
model = joblib.load("MindSync-StressPredictor/pickle/rf_model.p")
vectorizer = joblib.load("MindSync-StressPredictor/pickle/vectorizer_summary.p")

# === Prepare feature matrix (required if using model later, but not needed for this plot) ===
summary_features = vectorizer.transform(df["summary"].fillna(""))
X_numeric = df.drop(columns=["stress_level", "event_id", "summary", "start_date_time", "end_date_time"])
X_combined = hstack([summary_features, X_numeric])

# === Plot class distribution ===
plt.figure(figsize=(7, 5))
class_counts = df["stress_level"].value_counts().sort_index()
sns.barplot(x=class_counts.index, y=class_counts.values, palette="pastel", edgecolor="black")

plt.title("Class Distribution in Training Data")
plt.xlabel("Stress Level")
plt.ylabel("Number of Samples")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("MindSync-StressPredictor/output/class_distribution.png", bbox_inches="tight")
plt.close()

print("✅ class_distribution.png saved successfully.")
