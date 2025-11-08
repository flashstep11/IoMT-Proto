import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# --- MODEL & DATA CONFIGURATION ---
TRAINING_DATA_PATH = "./data/processed/fever_training_data.csv"
MODEL_OUTPUT_PATH = "./models/fever_model.joblib"
FEVER_THRESHOLD = 38.0 

# Time window settings
LOOKBACK_WINDOW = pd.Timedelta(minutes=5)
FORECAST_WINDOW = pd.Timedelta(minutes=15)
SLIDE_INTERVAL = pd.Timedelta(minutes=1)


def create_features_and_target(df):
    print("Creating time-series features and target...")
    feature_list = []
    target_list = []

    for start_time in tqdm(pd.date_range(df.index.min(), df.index.max() - LOOKBACK_WINDOW - FORECAST_WINDOW, freq=SLIDE_INTERVAL)):
        end_lookback = start_time + LOOKBACK_WINDOW
        end_forecast = end_lookback + FORECAST_WINDOW

        lookback_df = df.loc[start_time:end_lookback]
        forecast_df = df.loc[end_lookback:end_forecast]

        if len(lookback_df) < 2 or forecast_df.empty:
            continue

        features = {}
        time_diff = (lookback_df.index - lookback_df.index[0]).total_seconds()

        # Feature Engineering (must match features in app.py)
        features['temp_mean'] = lookback_df['temperature'].mean()
        features['temp_std'] = lookback_df['temperature'].std()
        features['temp_trend'] = np.polyfit(time_diff, lookback_df['temperature'], 1)[0]
        features['hr_mean'] = lookback_df['heart_rate'].mean()
        features['hr_std'] = lookback_df['heart_rate'].std()
        features['hr_trend'] = np.polyfit(time_diff, lookback_df['heart_rate'], 1)[0]
        features['spo2_mean'] = lookback_df['spo2'].mean()
        features['activity_mean'] = lookback_df['activity'].mean()
        features['sound_level_mean'] = lookback_df['sound_level'].mean()

        # Target Creation
        max_future_temp = forecast_df['temperature'].max()
        target = 1 if max_future_temp > FEVER_THRESHOLD else 0

        feature_list.append(features)
        target_list.append(target)

    feature_df = pd.DataFrame(feature_list).fillna(0)
    target_series = pd.Series(target_list)

    print(f"✅ Created {len(feature_df)} training examples.")
    if len(feature_df) > 0:
        print(f"   Positive (fever spike) class count: {target_series.sum()}")
    return feature_df, target_series


def train_model(X, y):
    print("\n--- Training Fever Prediction Model ---")

    if len(X) < 10 or y.nunique() < 2:
        print("❌ Not enough data or only one class present. Cannot train model.")
        return None

    stratify_option = y
    if y.value_counts().min() < 2:
        print("⚠ WARNING: Smallest class has < 2 members. Disabling stratification.")
        stratify_option = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_option)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100))
    ])

    pipeline.fit(X_train, y_train)

    print("\n--- Model Performance ---")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    if len(np.unique(y_test)) > 1:
        print(f"AUC-ROC Score: {roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]):.4f}")
    else:
        print("AUC-ROC Score: Not applicable (only one class in the test set).")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\n✅ Model saved successfully to {MODEL_OUTPUT_PATH}")
    return pipeline


if __name__ == "__main__":
    try:
        historical_df = pd.read_csv(TRAINING_DATA_PATH, index_col='created_at', parse_dates=True)
    except FileNotFoundError:
        print(f"❌ ERROR: Training data not found at {TRAINING_DATA_PATH}. Please run collect_fever_data.py first.")
        exit()

    X, y = create_features_and_target(historical_df)
    trained_model = train_model(X, y)