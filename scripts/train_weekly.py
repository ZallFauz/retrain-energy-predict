import os
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import utils


DATA_PATH = "data/energy_history.csv"
MODEL_PATH = "models/svr_energy.pkl"


def read_recent_thingspeak():
    url = (
        f"https://api.thingspeak.com/channels/{utils.CHANNEL_ID}/feeds.json"
        f"?api_key={utils.READ_API_KEY}&results=8000"
    )

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    data = resp.json()["feeds"]
    df = pd.DataFrame(data)

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])
    df = df.set_index("created_at")

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


def preprocess(df):
    # 1-hour aggregation
    df = df.resample("1H").mean()
    df = df.ffill()

    # Hourly energy usage (kWh)
    df["energy_delta"] = df[utils.ENERGY_KWH].diff()

    # Remove first NaN after diff
    df = df.dropna()

    return df


def update_dataset_from_thingspeak():
    df_new = read_recent_thingspeak()
    df_new = preprocess(df_new)

    df_new = df_new.reset_index().rename(columns={"created_at": "time"})

    if os.path.exists(DATA_PATH):
        df_old = pd.read_csv(DATA_PATH, parse_dates=["time"])
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time")
    else:
        df = df_new

    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    return df


def build_features(df):
    X = np.column_stack([
        df["voltage"],
        df["current"],
        df["power"],
        df["temperature"],
        df["humidity"],
        df["power_factor"],
        df["frequency"],
        df["energy_delta"].shift(1),
        df["energy_delta"].shift(24),
    ])

    y = df["energy_delta"].values

    # Drop rows with lag NaNs
    X = X[24:]
    y = y[24:]

    return X, y


def train_model(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = SVR(
        kernel="rbf",
        C=10.0,
        gamma=0.1,
        epsilon=0.0001
    )

    model.fit(Xs, y)
    return model, scaler


def export_model(model, scaler):
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler},
        MODEL_PATH
    )
    
def export_to_onnx(model, scaler, X_sample):
    initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
    
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model',model)
    ])
    
    onx = convert_sklearn(pipeline, initial_types=initial_type)

    with open("models/energy_svr.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    
    print("Model exported successfully")

def main():
    df = update_dataset_from_thingspeak()
    X, y = build_features(df)

    if len(X) < 100:
        raise RuntimeError("Not enough data to train SVR reliably")

    model, scaler = train_model(X, y)
    export_model(model, scaler)
    export_to_onnx(model, scaler, X)

    print("Training complete. Model exported.")


if __name__ == "__main__":
    main()
