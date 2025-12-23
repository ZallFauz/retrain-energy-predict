import pandas as pd
import numpy as np
import onnxruntime as ort
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # 1. Load the most recent data
    if not os.path.exists("data/energy_history.csv"):
        print("No data found to evaluate.")
        return

    df = pd.read_csv("data/energy_history.csv")
    
    # 2. Build the same features used in training
    X = np.column_stack([
        df["voltage"], df["current"], df["power"],
        df["temperature"], df["humidity"],
        df["power_factor"], df["frequency"],
        df["energy_delta"].shift(1),
        df["energy_delta"].shift(24),
    ])
    y = df["energy_delta"].values

    # Drop NaNs from shifting
    X = X[24:]
    y = y[24:]

    # 3. Load ONNX Session
    session = ort.InferenceSession("models/energy_svr.onnx")
    input_name = session.get_inputs()[0].name

    # 4. Run Inference (Predict)
    # Ensure X is float32 for ONNX
    y_pred_onnx = session.run(None, {input_name: X.astype(np.float32)})[0]
    
    # Flatten y_pred because ONNX returns [N, 1]
    y_pred_onnx = y_pred_onnx.flatten()

    # 5. Metrics
    mae = mean_absolute_error(y, y_pred_onnx)
    rmse = np.sqrt(mean_squared_error(y, y_pred_onnx))
    r2 = r2_score(y, y_pred_onnx)

    print("=== ONNX PRODUCTION MODEL EVALUATION ===")
    print(f"Total Samples Tested: {len(y)}")
    print(f"MAE:  {mae:.4f} kWh")
    print(f"RMSE: {rmse:.4f} kWh")
    print(f"R2:   {r2:.4f}")

    # # Check for anomalies (e.g., negative predictions)
    # neg_count = np.sum(y_pred_onnx < 0)
    # if neg_count > 0:
    #     print(f"WARNING: {neg_count} negative predictions detected in test set.")

if __name__ == "__main__":
    import os
    main()