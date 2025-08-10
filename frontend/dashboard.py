
import streamlit as st
import pandas as pd
import os

# === Local copy of classify_score ===
def classify_score(score):
    if score <= 1:
        return "Calm", "✅ You’re having a light day. Stay focused and enjoy it!"
    elif score <= 4:
        return "Mild", "🙂 A manageable schedule. Remember to take short breaks."
    elif score <= 7:
        return "Moderate", "🧘‍♀️ Your day is a bit packed. Try a breathing exercise or stretch."
    elif score <= 10:
        return "High", "🚶‍♂️ It's a demanding day. Take a proper break or short walk if possible."
    else:
        return "Very High", "⚠️ Your schedule looks intense. Consider rescheduling or adding recovery time."

# === File paths ===
BACKEND_SCRIPT = "MindSync-StressPredictor/googleAPI/fetch_predict.py"
DATA_PATH = "MindSync-StressPredictor/output/calendar_events_today.csv"

st.set_page_config(page_title="MindSync Dashboard", layout="centered")
st.title("🧠 MindSync Stress Dashboard")

# === Button to refresh calendar + prediction ===
if st.button("🔁 Refresh Calendar "):
    result = os.system(f"python {BACKEND_SCRIPT}")
    if result == 0:
        st.success(" Calendar and predictions updated.")
    else:
        st.error("❌ Failed to run backend script. Check paths and environment.")

# === Load and display predictions ===
if not os.path.exists(DATA_PATH):
    st.warning("No data file found. Please run the backend at least once.")
else:
    df = pd.read_csv(DATA_PATH)

    st.subheader("📅 Today's Calendar Events") 
      # Prepare display DataFrame
    df_display = df.rename(columns={
        "summary": "event",
        "start_date_time": "starts at",
        "end_date_time": "ends",
        "predicted_stress_level": "stress level"
    }).drop(columns=["location", "morning"])

    # Reorder and show
    st.dataframe(df_display[["event", "starts at", "ends", "duration", "stress level"]])

    
    
    total_score = df["predicted_stress_level"].sum()
    level, suggestion = classify_score(total_score)

    # === Compute total stress and display ===
if df.empty:
    st.subheader("📈 Stress Evaluation")
    st.success("🎉 No events today!")
    st.info(" Enjoy your day. No stress detected.")
else:
    total_score = df["predicted_stress_level"].sum()
    level, suggestion = classify_score(total_score)

    st.subheader("📈 Stress Evaluation")
    st.metric("Total Stress Score", total_score)
    st.success(f"Level: {level}")
    st.info(f"Suggestion: {suggestion}")

