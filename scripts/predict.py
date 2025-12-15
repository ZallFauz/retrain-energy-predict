import os
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import time
import utils

MODEL_PATH = "models/svr_energy.pkl"


# -------------------------------------------------
# Load trained model
# -------------------------------------------------
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["scaler"]


# -------------------------------------------------
# Read recent data from ThingSpeak
# -------------------------------------------------
def read_recent_thingspeak():
    url = (
        f"https://api.thingspeak.com/channels/{utils.CHANNEL_ID}/feeds.json"
        f"?api_key={utils.READ_API_KEY}&results=8000"
    )

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    df = pd.DataFrame(resp.json()["feeds"])

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).set_index("created_at")

    df.rename(columns={
        "field1": "voltage",
        "field2": "current",
        "field3": "power",
        "field4": "energy_kwh",
        "field5": "temperature",
        "field6": "humidity",
        "field7": "power_factor",
        "field8": "frequency",
    }, inplace=True)

    df = df[utils.FIELDS].astype(float)
    return df


# -------------------------------------------------
# Preprocess (must match training)
# -------------------------------------------------
def preprocess(df):
    df = df.resample("1H").mean().ffill()
    df["energy_delta"] = df["energy_kwh"].diff()
    return df.dropna()


# -------------------------------------------------
# Build future prediction
# -------------------------------------------------
def predict_future(df, hours=168):
    model, scaler = load_model()

    history = df.copy()
    future_rows = []

    for step in range(hours):
        row = history.iloc[-1]

        X = np.array([[
            row["voltage"],
            row["current"],
            row["power"],
            row["temperature"],
            row["humidity"],
            row["power_factor"],
            row["frequency"],
            history["energy_delta"].iloc[-1],
            history["energy_delta"].iloc[-24],
        ]])

        Xs = scaler.transform(X)
        y_pred = model.predict(Xs)[0]

        next_time = history.index[-1] + timedelta(hours=1)

        next_row = row.copy()
        next_row.name = next_time
        next_row["timestamp"] = next_time
        next_row["energy_delta"] = max(y_pred, 0.0)

        history = pd.concat([history, next_row.to_frame().T])
        future_rows.append(next_row)

    future_df = pd.DataFrame(future_rows)
    return future_df


# -------------------------------------------------
# Convert to cumulative kWh
# -------------------------------------------------
def make_cumulative(df, last_energy_kwh):
    df = df.copy()
    df["predicted_energy_kwh"] = last_energy_kwh + df["energy_delta"].cumsum()
    return df

def clear_channel():
    url = f"https://api.thingspeak.com/channels/{utils.WRITE_CHANNEL_ID}/feeds.json"
    r = requests.delete(url, params={"api_key": utils.USER_API_KEY})
    if r.status_code != 200:
        raise RuntimeError("Failed to clear channel")
     
def push_predictions(df):
    url = "https://api.thingspeak.com/update.json"
    
    for _, row in df.iterrows():
        payload = {
            "api_key": utils.WRITE_API_KEY,
            "field1": row["energy_delta"],
            "created_at": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        }

        r = requests.post(url, data=payload)

        if r.status_code != 200 or r.text == "0":
            print("Push failed:", r.text)
        else:
            print("Pushed:", row.name)

        time.sleep(16)  # REQUIRED
        
def bulk_push_predictions(df):
    # df["timestamp"] = df.name
    url = "https://api.thingspeak.com/channels/{}/bulk_update.json".format(
        utils.WRITE_CHANNEL_ID
    )

    payload = {
        "write_api_key": utils.WRITE_API_KEY,
        "updates": []
    }

    for _, row in df.iterrows():
        payload["updates"].append({
            "created_at": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "field1": float(row["energy_delta"])
        })
    
    print("Pushing to channel:", utils.WRITE_CHANNEL_ID)

    r = requests.post(url, json=payload, timeout=30)
    
    print(r.status_code, r.text)

    if r.status_code != 202:
        raise RuntimeError(f"Bulk update failed: {r.text}")

    print(f"Bulk pushed {len(payload['updates'])} predictions")     
def publish_7day_forecast(future_df):
    clear_channel()
    time.sleep(5)
    push_predictions(future_df)

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    df = read_recent_thingspeak()
    df = preprocess(df)

    last_energy = df["energy_kwh"].iloc[-1]

    future_df = predict_future(df, hours=168)
    future_df = make_cumulative(future_df, last_energy)

    print(future_df.head())
    print("Prediction complete.")
    
    clear_channel()
    publish_7day_forecast(future_df)
    
    print("push complete")
    


if __name__ == "__main__":
    main()
