import os
import pickle
import datetime
import pandas as pd
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import joblib

# === Setup ===
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
CREDENTIALS_PATH = 'MindSync-StressPredictor/googleAPI/credentials.json'
TOKEN_PATH = 'MindSync-StressPredictor/googleAPI/token.pickle'
OUTPUT_CSV = 'MindSync-StressPredictor/output/calendar_events_today.csv'
MODEL_PATH = 'MindSync-StressPredictor/pickle/rf_model.p'
VECTORIZER_PATH = 'MindSync-StressPredictor/pickle/vectorizer_summary.p'
SCALER_PATH = 'MindSync-StressPredictor/pickle/scaler.p'  # ✅ NEW

# === Authenticate ===
creds = None
if os.path.exists(TOKEN_PATH):
    with open(TOKEN_PATH, 'rb') as token:
        creds = pickle.load(token)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        creds = flow.run_local_server(port=0)
    with open(TOKEN_PATH, 'wb') as token:
        pickle.dump(creds, token)

# === Connect to Calendar ===
service = build('calendar', 'v3', credentials=creds)

now = datetime.datetime.utcnow()
start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat() + 'Z'

events_result = service.events().list(
    calendarId='primary', timeMin=start_of_day, timeMax=end_of_day,
    maxResults=100, singleEvents=True, orderBy='startTime').execute()

events = events_result.get('items', [])

# === Extract and Format Events ===
processed = []
for event in events:
    summary = event.get('summary', '')
    location = event.get('location', 'online')
    start = event['start'].get('dateTime', event['start'].get('date'))
    end = event['end'].get('dateTime', event['end'].get('date'))

    try:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        duration = (end_dt - start_dt).total_seconds() / 60
        morning = 1 if start_dt.hour < 12 else 0
    except:
        continue

    processed.append({
        'summary': summary,
        'location': 0 if 'online' in location.lower() else 1,
        'start_date_time': start_dt,
        'end_date_time': end_dt,
        'duration': duration,
        'morning': morning
    })

df = pd.DataFrame(processed)
if df.empty:
    print("⚠️ No events found today.")
    df = pd.DataFrame(columns=[
        "summary", "location", "start_date_time", "end_date_time",
        "duration", "morning", "predicted_stress_level"
    ])
    df.to_csv(OUTPUT_CSV, index=False)
    exit()

# === Load Model, Vectorizer, Scaler ===
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)  # ✅ NEW

# === Predict Stress Levels ===
summary_features = vectorizer.transform(df["summary"].fillna(""))
numeric_features = df[["location", "duration", "morning"]]

# ✅ Normalize numeric features
numeric_scaled = scaler.transform(numeric_features)

X_combined = hstack([summary_features, numeric_scaled])
df["predicted_stress_level"] = model.predict(X_combined)

# === Compute Total Score and Suggestion ===
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

total_score = df["predicted_stress_level"].sum()
stress_level, suggestion = classify_score(total_score)

# === Output Results ===
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Fetched {len(df)} events.")
print(f"🧠 Total Stress Score: {total_score}")
print(f"📈 Stress Level: {stress_level}")
print(f"💡 Suggestion: {suggestion}")
