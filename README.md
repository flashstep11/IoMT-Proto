# Zzz

Lightweight prototype for an IoT + EHR medical monitoring demo (Streamlit).

## What this repo contains

- A Streamlit app: `app.py` — runs the demo UI that combines static EHR risk predictors with a live IoT vitals dashboard.
- Pre-trained model artifacts in `models/` (joblib files) used by the app.
- Example/utility scripts such as `train_fever_model.py`, `collect_fever_data.py`, and `simulate_fever_spike.py`.
- Notebooks (`*.ipynb`) under the repo root that show model training/evaluation and end-to-end demos.

## Quick start (Windows PowerShell)

These steps will get the app running locally on Windows using PowerShell.

1. Install Python 3.9+ if you don't already have it.
2. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies. There is no requirements file in this repository; install the packages used by the app directly:

```powershell
pip install --upgrade pip
pip install streamlit pandas numpy scikit-learn joblib requests
```

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

Optional: To run on a specific port (for example 8501):

```powershell
streamlit run app.py --server.port 8501
```

The app will open in your default browser. If it doesn't, Streamlit prints a local URL you can visit (usually http://localhost:8501).

## Models & data

- Model artifact files are expected in the `models/` folder (e.g. `diabetes_prediction_model.joblib`, `hypertension_model.joblib`, `ckd_model.joblib`, `fever_model.joblib`). If a model file is missing the app will show an error message in the UI and that feature will be disabled.
- Training data and cleaned CSVs reside in `data/processed/`. Several training notebooks and scripts rely on those CSVs.

If you need to regenerate models, check the training scripts (for example `train_fever_model.py`) or the Jupyter notebooks. Training may require additional dependencies and can take time.

## App tabs (what each tab does)

The Streamlit application (`app.py`) exposes two primary tabs in the UI. This section summarizes what each tab does so you know where to look and what to expect.

- Tab 1 — "EHR Risk Predictor":
	- This tab provides static Electronic Health Record (EHR) based risk prediction for three conditions: Type-2 Diabetes, Hypertension, and Chronic Kidney Disease (CKD).
	- Inputs are provided in the sidebar (age, BMI, blood pressure, cholesterol, smoking status, diabetes/hypertension history, etc.).
	- Each condition uses a pre-trained joblib model in `models/` (e.g., `diabetes_prediction_model.joblib`). If a model file is missing the UI will show an error message for that predictor.
	- The Hypertension predictor includes a small "What-If" analysis: when you change the Smoking Status input, the UI displays risk score deltas for alternate smoking states so you can see how that single factor affects predicted risk.

- Tab 2 — "Live Patient Monitor" (Fever prediction + Dashboard):
	- This tab fetches recent IoT sensor data from ThingSpeak and computes engineered features over a 5-minute lookback window (mean, std, trend) for key signals such as temperature and heart rate.
	- A persisted fever prediction model (`fever_model.joblib`) is loaded and used to predict the likelihood of a fever spike (> 38.0 °C) in the next 15 minutes.
	- The tab also embeds a Chart.js-based dashboard (via Streamlit components) that auto-refreshes to show the latest sensor streams (HR, SpO₂, temperature, activity, sound level).
	- Key Performance Indicators (KPIs) such as Latest Temperature, Latest Heart Rate, and Latest SpO₂ are shown alongside the model's confidence/risk score.
	- Important: If no live ThingSpeak data is available (for example, data collection from the ESP32 is currently paused), the prediction section will not produce a fever prediction and the UI will display a warning like "Waiting for sufficient live data from ThingSpeak...". The dashboard charts may still attempt to load but will be empty or show placeholders until data becomes available.

## Notebooks — DO NOT EDIT / DO NOT RUN (unless you have the datasets)

The repository includes several Jupyter notebooks (e.g., `01_end_to_end_diabetes_model.ipynb`, `02_hypertension_model.ipynb`, `03_ckd_model.ipynb`). These are included for demonstration and documentation only.

- Do NOT modify these notebooks unless you know what you are doing.
- Important: the notebooks require the datasets in `data/processed/` and other preprocessing steps; many cells will fail if those datasets are missing. They are provided for explanation and reproducibility, not as a runnable part of the app in this checkout.

## Troubleshooting

- "Model file not found" or similar errors in the Streamlit UI: confirm `models/` contains the `.joblib` files. If not present, run the training scripts or copy the model files into that folder.
- Network/timeouts when the dashboard pulls ThingSpeak data: ensure outbound HTTP is allowed from your machine and that ThingSpeak channel details in `app.py` (channel ID and read key) are valid.
- If the app fails to start with import errors, re-check the installed packages and that the virtual environment is activated.

## Notes

- The UI is built with Streamlit. The live dashboard embeds a Chart.js widget which uses CDN-hosted JS; no extra install is needed for the charts.
- This repo is a prototype/demo: predictions are for demonstration only and must not be used for real clinical decision making.

---
