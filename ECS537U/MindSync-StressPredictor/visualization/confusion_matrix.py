import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack
import joblib
import numpy as np

# Reload dataset and model
df = pd.read_csv("MindSync-StressPredictor/data/dataset.csv")
model = joblib.load("MindSync-StressPredictor/pickle/rf_model.p")
vectorizer = joblib.load("MindSync-StressPredictor/pickle/vectorizer_summary.p")

# Prepare features
df_cleaned = df.dropna(subset=["stress_level"])
summary_features = vectorizer.transform(df_cleaned["summary"].fillna(""))
X_numeric = df_cleaned.drop(columns=["stress_level", "event_id", "summary", "start_date_time", "end_date_time"])
X_combined = hstack([summary_features, X_numeric])
y_true = df_cleaned["stress_level"].astype(int)
y_pred = model.predict(X_combined)

# Confusion Matrix
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("MindSync-StressPredictor/output/confusion_matrix.png", bbox_inches="tight")
plt.close()

