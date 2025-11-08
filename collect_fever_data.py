import requests
import pandas as pd
import os

# --- CONFIGURATION: FILL THESE IN! ---
THINGSPEAK_CHANNEL_ID = "3110381"
THINGSPEAK_READ_API_KEY = "X998UI7V7LF1X3Z9"
HOURS_TO_FETCH = 4 # How many hours of historical data to download
OUTPUT_CSV_PATH = "./data/processed/fever_training_data.csv"

FIELD_MAP = {
    'field1': 'heart_rate', 'field2': 'spo2', 'field3': 'activity',
    'field4': 'sound_level', 'field5': 'temperature'
}

def fetch_thingspeak_history():
    print(f"Attempting to fetch the last {HOURS_TO_FETCH} hours of data...")
    num_records = HOURS_TO_FETCH * 60 * 4 # Assuming 15s interval
    
    # Cap at ThingSpeak's 8000 record limit
    if num_records > 8000:
        print(f"Warning: Requested records ({num_records}) exceeds 8000. Capping at 8000.")
        num_records = 8000
        
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"
    params = {"api_key": THINGSPEAK_READ_API_KEY, "results": num_records}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()['feeds']
        
        if not data:
            print("❌ No data returned from ThingSpeak. Check your Channel ID and API Key.")
            return None
            
        df = pd.DataFrame(data)
        print(f"✅ Successfully fetched {len(df)} records.")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data from ThingSpeak: {e}")
        return None

def process_and_save_data(df):
    if df is None: return
    df = df.rename(columns=FIELD_MAP)
    
    required_cols = list(FIELD_MAP.values()) + ['created_at']
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols]
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.set_index('created_at')
    
    for col in FIELD_MAP.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.ffill().dropna() # Forward fill and drop any remaining NaNs
    
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH)
    print(f"✅ Clean training data saved successfully to {OUTPUT_CSV_PATH}")
    print(f"   Data shape: {df.shape}")
    print(f"   Time range: {df.index.min()} to {df.index.max()}")

if __name__ == "__main__":
    raw_df = fetch_thingspeak_history()
    process_and_save_data(raw_df)