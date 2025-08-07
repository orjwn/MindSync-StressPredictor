import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# === Load Data and Model ===
df = pd.read_csv("MindSync-StressPredictor/data/dataset.csv").dropna(subset=["stress_level"])
model = joblib.load("MindSync-StressPredictor/pickle/rf_model.p")
vectorizer = joblib.load("MindSync-StressPredictor/pickle/vectorizer_summary.p")

# === Prepare Features ===
summary_features = vectorizer.transform(df["summary"].fillna(""))
X_numeric = df.drop(columns=["stress_level", "event_id", "summary", "start_date_time", "end_date_time"])
X_combined = hstack([summary_features, X_numeric])

y_true = df["stress_level"].astype(int)
y_pred = model.predict(X_combined)

# Parse classification report
report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose().iloc[:5][["precision", "recall", "f1-score"]]

# Plot class-wise precision, recall, f1-score
plt.figure(figsize=(10, 6))
report_df.plot(kind='bar', ylim=(0, 1), colormap='Set2', edgecolor='black')
plt.title("Class-wise Precision, Recall, and F1-score")
plt.ylabel("Score")
plt.xlabel("Stress Level")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("MindSync-StressPredictor/output/class_metrics_bar.png", bbox_inches="tight")
plt.close()
