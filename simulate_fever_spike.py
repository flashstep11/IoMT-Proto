import pandas as pd
import numpy as np
import sys 

# --- CONFIGURATION ---
TRAINING_DATA_PATH = "./data/processed/fever_training_data.csv"
FEVER_PEAK_TEMP = 38.5  # The peak temperature of the simulated fever
FEVER_DURATION_MINUTES = 20 # A shorter, more realistic spike duration

print("--- Starting Robust Fever Spike Simulation Utility ---")

try:
    df = pd.read_csv(TRAINING_DATA_PATH, index_col='created_at', parse_dates=True)
    print(f"✅ Loaded existing training data with {len(df)} records.")
except FileNotFoundError:
    print(f"❌ ERROR: Training data not found at {TRAINING_DATA_PATH}. Please run collect_fever_data.py first.")
    sys.exit(1) 

# Check if the dataset is long enough
MINIMUM_RECORDS = 2 * 60 * 4 # ~2 hours of data
if len(df) < MINIMUM_RECORDS:
    print(f"❌ ERROR: Your dataset is too short ({len(df)} records). Please collect at least {MINIMUM_RECORDS} records (approx. 2 hours) before simulating a fever.")
    sys.exit(1)
else:
    print(f"-> Simulating a {FEVER_DURATION_MINUTES}-minute fever spike...")

    midpoint_index = len(df) // 2
    fever_records_count = FEVER_DURATION_MINUTES * 4 # Assuming 15s interval
    start_index = midpoint_index - (fever_records_count // 2)
    end_index = midpoint_index + (fever_records_count // 2)

    fever_curve = np.sin(np.linspace(0, np.pi, fever_records_count))
    original_temps = df['temperature'].iloc[start_index:end_index].copy()
    simulated_temps = original_temps + (fever_curve * (FEVER_PEAK_TEMP - original_temps.mean()))

    df.loc[df.index[start_index:end_index], 'temperature'] = simulated_temps
    print("-> Fever spike successfully added.")

    df.to_csv(TRAINING_DATA_PATH)
    print(f"✅ Modified training data saved back to {TRAINING_DATA_PATH}")