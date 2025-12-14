
import os
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


READ_API_KEY = os.environ["READ_API_KEY"]
WRITE_API_KEY = os.environ["WRITE_API_KEY"]
CHANNEL_ID = os.environ["CHANNEL_ID"]


VOLTAGE = 'voltage'
CURRENT = 'current'
POWER = 'power'
ENERGY_KWH = 'energy_kwh'
TEMPERATURE = 'temperature'
HUMIDITY = 'humidity'
POWER_FACTOR = 'power_factor'
FREQUENCY = 'frequency'

FIELDS = [VOLTAGE,
          CURRENT,
          POWER,
          ENERGY_KWH,
          TEMPERATURE,
          HUMIDITY,
          POWER_FACTOR,
          FREQUENCY
          ]


FEATURES = [VOLTAGE,
          CURRENT,
          POWER,
          TEMPERATURE,
          HUMIDITY,
          POWER_FACTOR,
          FREQUENCY
          ]


PRED_CHANNEL_FIELD = 1  # where prediction is written


def read_recent_thingspeak():
    url = (
        f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
        f"?api_key={READ_API_KEY}&results=8000"
    )
    data = requests.get(url).json()["feeds"]
    df = pd.DataFrame(data)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.set_index("created_at")
    df.rename(columns={
        'field1': 'voltage',
        'field2': 'current',
        'field3': 'power',
        'field4': 'energy_kwh',
        'field5': 'temperature',
        'field6': 'humidity',
        'field7': 'power_factor',
        'field8': 'frequency',
    }, inplace=True)

    df = df[FIELDS].astype(float)
    return df


def prepocess(df):
    df = df.resample("1h").mean()
    df = df.ffill()

    df["energy_delta"] = df[ENERGY_KWH].diff()
    df = df.dropna()

    return df


def update_dataset_from_thingspeak():
    df_new = read_recent_thingspeak()
    df_new = prepocess(df_new)

    try:
        df_old = pd.read_csv("data/energy_history.csv", parse_dates=["time"])
        df = pd.concat([df_old, df_new]).drop_duplicates("time")
    except:
        df = df_new

    df.to_csv("data/energy_history.csv", index=False)
    return df


def build_feature(df):
    X = np.column_stack([
        df[VOLTAGE],
        df[CURRENT],
        df[POWER],
        df[TEMPERATURE],
        df[HUMIDITY],
        df[POWER_FACTOR],
        df[FREQUENCY],
        df["energy_delta"].shift(1),
        df["energy_delta"].shift(24)
    ])

    y = df["energy_delta"].values

    X = X[24:]
    y = y[24:]

    return X, y


def train_model(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = SVR(kernel="rbf", C=10, gamma=0.1, epsilon=0.0001)
    model.fit(Xs, y)

    return model, scaler


def export_models(model, scaler):
    ts = datetime.now().strftime("%Y%m%d")

    joblib.dump((model, scaler), f"svr_energy_{ts}.pkl")

    initial_type = [("input", FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(f"models/energy_svr.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


def predict_next_week(model, scaler, df):
    last = df.iloc[-1]

    preds = []
    prev = last["energy_delta"]

    for _ in range(168):
        x = np.array([[
            last[VOLTAGE],
            last[CURRENT],
            last[POWER],
            last[TEMPERATURE],
            last[HUMIDITY],
            last[POWER_FACTOR],
            last[FREQUENCY],
            prev, prev
            ]])
        x = scaler.transform(x)
        p = model.predict(x)[0]
        preds.append(p)
        prev = p
        
    return np.cumsum(preds)
    
    


def main():
    df = update_dataset_from_thingspeak()
    X, y = build_feature(df)
    model, scaler = train_model(X, y)
    
    export_models(model, scaler)    
    forecast = predict_next_week(model, scaler, df)
    print(forecast)



if __name__ == "__main__":
    main()
