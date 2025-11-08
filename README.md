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
