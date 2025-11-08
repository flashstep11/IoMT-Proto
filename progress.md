# Project Progress

Date: 2025-11-09

## What we have done

- Integrated an ESP32 microcontroller with multiple sensors and created a stable hardware setup:
  - Temperature: DS18B20
  - Heart rate / SpO2: MAX30105
  - Gyroscope / accelerometer: MPU6050

- Collected sensor data and uploaded it to ThingSpeak for remote access and visualization.

- Developed Python scripts that process raw, timestamped sensor streams and engineer predictive features over a 5-minute lookback window. Engineered features include:
  - Mean
  - Standard deviation
  - Linear trend (using `np.polyfit`)

- Created, trained, and saved a RandomForestClassifier model that predicts the classification task: a fever spike (temperature > 38.0 °C) within the next 15 minutes. The model is persisted as a `.joblib` artifact.

- Implemented a live monitoring Streamlit application that:
  - Continuously fetches the latest data window from ThingSpeak
  - Runs the loaded AI model on engineered features
  - Displays the prediction and confidence score

- Embedded an auto-refreshing Chart.js dashboard in the Streamlit app to visualize live sensor data (Heart Rate, SpO₂, Temperature, Activity, Sound level) fetched from ThingSpeak.

- Integrated Key Performance Indicators (KPIs) into the monitoring interface showing the latest sensor readings and the AI model's confidence/risk score (e.g., Latest Temp, HR, SpO₂, Risk Score).

- Built a multi-tab Streamlit web application that unifies static EHR risk prediction (offline inputs) with dynamic IoT data monitoring (live inputs) in a single UI.

- Developed, trained, and integrated three separate ML models (persisted as joblib artifacts) for predicting chronic disease risk from simulated EHR inputs:
  - Type-2 Diabetes
  - Hypertension
  - Chronic Kidney Disease (CKD)

- Implemented a What-If analysis feature in the Hypertension predictor to show how changing a single risk factor (Smoking Status) alters the model's risk score dynamically.


## Key design notes

- Feature engineering for live monitoring is implemented in `create_live_features` and is deliberately structured to be extensible. The function computes per-sensor aggregates and trends that match the features used for training the fever model.

- Model loading and inference use `joblib` artifacts and are integrated into the Streamlit app with caching to avoid repeated disk I/O during a session.

- The ThingSpeak integration is generalized using a small field map and configurable channel/read key values so the dashboard can be pointed at different channels easily.


## Future scope

- Integrate an ECG sensor for non-invasive blood pressure estimation (a regression task). This will involve designing a PTT (Pulse Transit Time) pipeline and incorporating ECG-derived features.

- Expand feature engineering and model pipelines to support PTT and other ECG-derived features required for robust BP regression models.

- Explore acoustic modalities using the sound sensor for additional clinical signals and context-aware event detection.

- Onboard a BP regression model into the same app architecture (joblib loading, new tab and UI elements) so clinicians/researchers can compare classification and regression outputs from the same interface.

- Harden the production readiness of the app: configuration files, clearer error handling for missing models, a `requirements.txt` or environment spec, and optional containerization for reproducible deployment.


## Notes / Next steps

- The existing codebase and model pipeline are designed to be modular—adding a new regression model (e.g., for BP) should require minimal structural changes.

- Recommended immediate tasks:
  1. Add `requirements.txt` and a small `run` helper (PowerShell script) to simplify launching the app for other team members.
  2. Add unit tests for the `create_live_features` pipeline to ensure feature parity between training and inference.
  3. Prototype a minimal ECG + PTT feature extraction script and collect labeled BP data for model training.


---

If you want, I can also commit a `requirements.txt` and a `run.ps1` helper script, or add the unit tests for the feature pipeline next. Which should I do first?
