import joblib
import numpy as np
import pandas as pd

# === Load the trained model and vectorizer ===
model = joblib.load("MindSync-StressPredictor/pickle/rf_model.p")
vectorizer = joblib.load("MindSync-StressPredictor/pickle/vectorizer_summary.p")

# === Define your numeric feature names manually ===
numeric_feature_names = ["location", "duration", "morning"]

# === Get text feature names from the CountVectorizer ===
text_feature_names = vectorizer.get_feature_names_out()

# === Combine all feature names ===
all_feature_names = np.concatenate([text_feature_names, numeric_feature_names])

# === Get feature importances from the RandomForest model ===
importances = model.feature_importances_

# === Build DataFrame of importances ===
feature_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# === Display top N features (optional) ===
print("Top 20 Important Features:")
print(feature_df.head(20))

# === Save full list to CSV ===
feature_df.to_csv("MindSync-StressPredictor/data/model_feature_importances.csv", index=False)
print("✅ Feature importances saved to MindSync-StressPredictor/data/model_feature_importances.csv")

