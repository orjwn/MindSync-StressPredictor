import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === Load Actual and Predicted Data ===
actual_df = pd.read_csv("MindSync-StressPredictor/data/dataset.csv")
predicted_df = pd.read_csv("MindSync-StressPredictor/output/predicted_output.csv")

# === Ensure Common Event IDs ===
common_ids = set(actual_df["event_id"]).intersection(set(predicted_df["event_id"]))
actual_df = actual_df[actual_df["event_id"].isin(common_ids)].sort_values("event_id")
predicted_df = predicted_df[predicted_df["event_id"].isin(common_ids)].sort_values("event_id")

# === Match True and Predicted Labels ===
y_true = actual_df["stress_level"].astype(int).values
y_pred = predicted_df["predicted_stress_level"].astype(int).values

# === Evaluate ===
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)

# === Print Evaluation Results ===
print(f"✅ Accuracy: {round(accuracy * 100, 2)}%")
print("\n📊 Classification Report:\n", report)
print("\n🧮 Confusion Matrix:\n", conf_matrix)
