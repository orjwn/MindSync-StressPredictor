# 🧠 MindSync - Stress Prediction System

MindSync is a smart stress prediction system that uses machine learning to predict a user’s stress level based on calendar events.  
It supports:  
- **Live prediction** using Google Calendar API  
- **Fallback mode** using a static JSON file (`fallback_events.json`) when the API is unavailable.

---

## 📂 Project Structure

```
MindSync-StressPredictor/
├── analysis/
│   ├── train_model.py
│   ├── predict_model.py
│   ├── evaluate_predictions.py
│   └── check.py
├── data/
│   ├── dataset.csv
│   └── fallback_events.json   
├── googleAPI/
│   ├── credentials.json
│   ├── token.pickle
│   └── fetch_predict.py       
├── pickle/
│   ├── rf_model.p
│   ├── vectorizer_summary.p
│   └── scaler.p
├── output/
│   ├── predicted_output.csv
│   ├── calendar_events_today.csv
│   ├── *.png (visuals)
├── frontend/
│   └── dashboard.py

```

---

## ⚙️ Installation

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate the environment:**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. 🔮 Train the Model
```bash
python analysis/train_model.py
```

This will:
- Train a Random Forest model on `dataset.csv`
- Save the trained model in `pickle/rf_model.p`
- Save the summary vectorizer and scaler

---

### 2. ✅ Evaluate Predictions on Test Data
```bash
python analysis/evaluate_predictions.py
```
- Compares the model's predictions (`predicted_output.csv`) against actual labels
- Outputs accuracy, precision, recall, F1-score, and confusion matrix

---

### 3. 📊 Visualizations
Run any of the following to generate visuals saved in `output/`:
```bash
python visualization/class_distribution.py
python visualization/confusion_matrix.py
python visualization/class_metrics_bar.py
```

---

### 4. 📡 Live Calendar Stress Prediction
**Set up Google API credentials** in `googleAPI/credentials.json`.

Then run:
```bash
python googleAPI/fetch_predict.py
```

This will:
- Connect to today's Google Calendar events
- Predict stress levels using the trained model
- Save results to `output/calendar_events_today.csv`
- If the API fails or returns no events, automatically load data/fallback_events.json


---

### 5. 🖥️ Launch Dashboard (Streamlit)
```bash
streamlit run frontend/dashboard.py
```

---

### 6. 🧪 (Optional) Offline Prediction
Use this to predict stress on static CSV data :
```bash
python analysis/predict_model.py
```
Input: data/dataset.csv
Output: Results are saved to output/predicted_output.csv

This mode is useful for testing the trained model offline using static CSV data.
---

## 🧾 Requirements

All dependencies are listed in `requirements.txt`. Major libraries include:
- `scikit-learn`
- `pandas`
- `streamlit`
- `google-api-python-client`
- `google-auth`, `oauthlib`
- `matplotlib`, `seaborn`

---

## 📝 Notes
- `fetch_predict.py` requires valid OAuth setup with Google Calendar.
- `predict_model.py` is useful for evaluation without using live calendar data.
- Stress scores are based on event duration, time of day, location, and summary content.

---

Author:
-------
Orjuwan Almotarafi
