import os
import json
import pickle
import datetime
import pandas as pd
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import joblib

# === Paths & setup ===
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
CREDENTIALS_PATH = 'MindSync-StressPredictor/googleAPI/credentials.json'
TOKEN_PATH = 'MindSync-StressPredictor/googleAPI/token.pickle'
OUTPUT_CSV = 'MindSync-StressPredictor/output/calendar_events_today.csv'

MODEL_PATH = 'MindSync-StressPredictor/pickle/rf_model.p'
VECTORIZER_PATH = 'MindSync-StressPredictor/pickle/vectorizer_summary.p'
SCALER_PATH = 'MindSync-StressPredictor/pickle/scaler.p'

FALLBACK_JSON_PATH = 'MindSync-StressPredictor/data/fallback_events.json'  

# === Helpers ===
def process_events(raw_events):
    """Turn raw Google events into a clean DataFrame."""
    processed = []
    for event in raw_events:
        summary = event.get('summary', '')
        location = event.get('location', 'online') or 'online'
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))

        try:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            duration = (end_dt - start_dt).total_seconds() / 60.0
            morning = 1 if start_dt.hour < 12 else 0
        except Exception:
            continue

        processed.append({
            'summary': summary,
            'location': 0 if 'online' in str(location).lower() else 1,
            'start_date_time': start_dt,
            'end_date_time': end_dt,
            'duration': duration,
            'morning': morning
        })

    return pd.DataFrame(processed)

def load_fallback_df():
    """Load fallback JSON and return a DataFrame in the same schema as process_events()."""
    if not os.path.exists(FALLBACK_JSON_PATH):
        raise FileNotFoundError(f"Fallback file not found: {FALLBACK_JSON_PATH}")

    with open(FALLBACK_JSON_PATH, 'r') as f:
        data = json.load(f)

    # Ensure consistent types
    rows = []
    for ev in data:
        rows.append({
            'summary': ev.get('summary', ''),
            'location': int(ev.get('location', 0)),
            'start_date_time': pd.to_datetime(ev.get('start_date_time')),
            'end_date_time': pd.to_datetime(ev.get('end_date_time')),
            'duration': float(ev.get('duration', 0)),
            'morning': int(ev.get('morning', 0)),
        })
    return pd.DataFrame(rows)

# === Try API first ===
df = pd.DataFrame()
used_fallback = False

try:
    # Authenticate
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

    # Connect & fetch today's events
    service = build('calendar', 'v3', credentials=creds)
    now = datetime.datetime.utcnow()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
    end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat() + 'Z'

    events_result = service.events().list(
        calendarId='primary',
        timeMin=start_of_day,
        timeMax=end_of_day,
        maxResults=100,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])
    df = process_events(events)

    # If API returned nothing, switch to fallback
    if df.empty:
        df = load_fallback_df()
        used_fallback = True

except Exception as e:
    print(f"⚠️ API error: {e}\n➡️ Using fallback JSON instead.")
    df = load_fallback_df()
    used_fallback = True

# If still empty after fallback, write empty CSV and exit
if df.empty:
    print("⚠️ No events available (API and fallback empty).")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out = pd.DataFrame(columns=[
        "summary", "location", "start_date_time", "end_date_time",
        "duration", "morning", "predicted_stress_level"
    ])
    df_out.to_csv(OUTPUT_CSV, index=False)
    raise SystemExit()

# === Load Model, Vectorizer, Scaler ===
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)

# === Predict ===
summary_features = vectorizer.transform(df["summary"].fillna(""))
numeric_features = df[["location", "duration", "morning"]]
numeric_scaled = scaler.transform(numeric_features)

X_combined = hstack([summary_features, numeric_scaled])
df["predicted_stress_level"] = model.predict(X_combined)

# === Score & Suggest ===
def classify_score(score):
    if score <= 1:
        return "Calm", " You’re having a light day. Stay focused and enjoy it!"
    elif score <= 4:
        return "Mild", "🙂 A manageable schedule. Remember to take short breaks."
    elif score <= 7:
        return "Moderate", "🧘‍♀️ Your day is a bit packed. Try a breathing exercise or stretch."
    elif score <= 10:
        return "High", "🚶‍♂️ It's a demanding day. Take a proper break or short walk if possible."
    else:
        return "Very High", "⚠️ Your schedule looks intense. Consider rescheduling or adding recovery time."

total_score = int(df["predicted_stress_level"].sum())
stress_level, suggestion = classify_score(total_score)

# === Output ===
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

source = "Fallback JSON" if used_fallback else "Google Calendar API"
print(f" Source: {source}")
print(f" Events processed: {len(df)}")
print(f"🧠 Total Stress Score: {total_score}")
print(f"📈 Stress Level: {stress_level}")
print(f"💡 Suggestion: {suggestion}")
