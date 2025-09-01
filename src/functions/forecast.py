import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import shap
import json
from datetime import timedelta
from src.functions.utils import load_json, targets, load_dataset, actual_column, error_with_df
from src.functions.forecast_plot import plot_forecast_by_field
from src.models.models import ErrorModel

XGB_MODEL_PATH = "./src/created_model/xgb_model.pkl"
train_history = "./src/resource/train_history1.xlsx"
forecast_xgb = "./src/resource/forecast_xgb.xlsx"
synthetic_production_numeric = "./src/resource/synthetic_production_numeric.xlsx"
forecast_production_numeric = "./src/resource/forecast_production_numeric.xlsx"
#change to the forecast data
forecast_data= "./src/resource/synthetic_production_numeric.xlsx"

LSTM_MODEL_PATH = "./src/created_model/lstm_model.h5"
SCALER_X_PATH = "./src/created_model/scaler_X.pkl"
SCALER_Y_PATH = "./src/created_model/scaler_y.pkl"

forecast_lstm = "./src/resource/forecast_lstm.xlsx"
future_forecast_lstm = "./src/resource/future_forecast_lstm.xlsx"
output_excel_lstm = "./src/resource/output_excel.xlsx"
FEATURES_PATH = "./src/resource/features.json"

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

    # Features and Targets
def features():

    return load_json(FEATURES_PATH)

def xgb_forecast():
    # Load saved model
    
    xgb_loaded = joblib.load(XGB_MODEL_PATH)
    data = load_dataset(forecast_data)

    # Use only features (X)
    X_new = data["X"]

    # Forecast
    y_forecast = xgb_loaded.predict(X_new)

    save_forecast(y_forecast, data, forecast_xgb)

def lstm_forecast():
    # Load trained model
    model = load_model(LSTM_MODEL_PATH, compile=False)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # Load new data
    data = load_dataset(forecast_data)

    # Preprocess (must use the same scaler as training!)
    X_new = data["X"]
    X_new_scaled = scaler_X.transform(X_new)

    # Make sequences (30-day window)
    seq_length = 30
    X_seq = []
    for i in range(len(X_new_scaled) - seq_length):
        X_seq.append(X_new_scaled[i:i+seq_length])
    X_seq = np.array(X_seq)

    # Forecast
    y_pred_scaled = model.predict(X_seq, seq_length)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    save_forecast(y_pred, data, future_forecast_lstm)

def save_forecast(y_pred, data, filepath):
    # Save forecast
    forecast_df = pd.DataFrame(y_pred, columns=targets())
    forecast_df["oil_rate_bopd (BOPD)"] = data["df"]["oil_rate_bopd (BOPD)"]
    forecast_df["gas_rate_mscf_day (MSCF/day)"] = data["df"]["gas_rate_mscf_day (MSCF/day)"]
    forecast_df["water_rate_bwpd (BWPD)"] = data["df"]["water_rate_bwpd (BWPD)"]  
    forecast_df["field_name_label"] = data["df"]["field_name_label"]
    forecast_df["well_name_label"] = data["df"]["well_name_label"]
    forecast_df["timestamp"] = data["df"]["timestamp"]

    forecast_df.to_excel(filepath, index=False)
    

def xgb_plot():
    # Load test dataset
    data = load_dataset(forecast_xgb)
    df = data["df"]
    # df = pd.read_excel(forecast_production_numeric)
    # Load saved model
    xgb_loaded = joblib.load(XGB_MODEL_PATH)

    # Features and Targets
    X = data["X"]
    y = data["y"]

    # Forecast
    y_pred = xgb_loaded.predict(X)

    # Convert to DataFrame
    forecast_df = pd.DataFrame(y_pred, columns=y.columns)
    forecast_df["timestamp"] = df["timestamp"]

    # Plot comparison for each target
    for col in y.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], y[col], label=f"Actual {col}", color="blue")
        plt.plot(forecast_df["timestamp"], forecast_df[col], label=f"Forecast {col}", color="red", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.title(f"Forecast vs Actual - {col}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fname = os.path.join(PLOT_DIR, f"dashboard_{col}.png")
        plt.savefig(fname)
        plt.show()

def lstm_forecast_plot():
    # Load scalers
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # Load actual data (from training history)
    df = pd.read_excel(forecast_data)
    y_true = df[targets()]
    timestamps = df["timestamp"]

    # Load forecast results (previously generated Excel or predictions)
    forecast_df = pd.read_excel(future_forecast_lstm)  # <-- replace with your forecast file
    y_pred = forecast_df[targets()]

    # --- Plot each target ---
    for col in y_true.columns:
        plt.figure(figsize=(10,5))
        plt.plot(timestamps, y_true[col], label=f"Actual {col}", color="blue")
        plt.plot(forecast_df["timestamp"], y_pred[col], label=f"Forecast {col}", color="red", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.title(f"Forecast vs Actual - {col}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fname = os.path.join(PLOT_DIR, f"forecast_vs_actual_{col}.png")
        plt.savefig(fname)
        plt.show()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def lstm_explanation():
    # 1) Load history
    # df = pd.read_excel(forecast_data)
    # df["timestamp"] = pd.to_datetime(df["timestamp"])
    data = load_dataset(forecast_data)
    df = data["df"]

    # 2) Load model + scalers
    model = load_model(LSTM_MODEL_PATH, compile=False)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    SEQ_LEN = 30

    # 3) Prepare data
    X = scaler_X.transform(data["X"])

    # Predict one step ahead
    y_pred_scaled = model.predict(np.array([X[-SEQ_LEN:]]))  
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Save forecast (last timestamp + 1 day)
    forecast_time = df["timestamp"].iloc[-1] + timedelta(days=1)
    out = pd.DataFrame({"timestamp": [forecast_time]})
    out[targets()] = y_pred
    out.to_excel(output_excel_lstm, index=False)

    # Background sequences (for SHAP)
    background = np.array([X[i-SEQ_LEN:i] for i in range(SEQ_LEN, SEQ_LEN+20)])
    background_flat = background.reshape(background.shape[0], -1)

    # Sequence we want to explain
    sample_seq = np.array([X[-SEQ_LEN:]])   # (1,30,features)
    sample_seq_flat = sample_seq.reshape(sample_seq.shape[0], -1)

    # 4) SHAP explanations per target
    for t, target in enumerate(targets()):
        def predict_fn_target(data, t=t):
            data_reshaped = data.reshape(data.shape[0], SEQ_LEN, len(features()))
            return model.predict(data_reshaped)[:, t]  # only one output

        explainer = shap.KernelExplainer(predict_fn_target, background_flat)
        shap_vals = explainer.shap_values(sample_seq_flat, nsamples=100)

        # shap_vals is already (seq_len*features,)
        vals = np.array(shap_vals)
        if vals.size != SEQ_LEN * len(features()):
            print(f"Skipping {target}, unexpected SHAP size {vals.size}")
            continue

        vals = vals.reshape(SEQ_LEN, len(features()))   # (30, features)
        mean_vals = np.abs(vals).mean(axis=0)          # average abs shap values
        mean_vals = mean_vals / mean_vals.sum() * 100  # normalize %

        # Plot
        plt.figure(figsize=(6, 4))
        plt.bar(features(), mean_vals)
        plt.title(f"Feature Importance for {target} Forecast")
        plt.ylabel("Relative Influence (%)")
        plt.xticks(rotation=30)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        fname = os.path.join(PLOT_DIR, f"shap_importance_{target}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

def plot_forecast():
    data = load_dataset(forecast_xgb)
    #data = load_dataset(future_forecast_lstm)
    df = data["df"]
    wells = ["Well_A", "Well_B", "Well_C"]
    fields = ["Field_1", "Field_2"]
    # features = ["oil_rate_bopd (BOPD)", "gas_rate_mscf_day (MSCF/day)", "water_rate_bwpd (BWPD)"]
    features_oil = ["oil_rate_bopd (BOPD)"]
    features_gas = ["gas_rate_mscf_day (MSCF/day)"]
    features_water = ["water_rate_bwpd (BWPD)"]

    for well in wells:
        plot_forecast_by_field(df, fields[0], well,  features_oil, "Forecast")

def error_forecast():
    data = load_dataset(forecast_xgb)

    actual = actual_column()

    results = []
    for i in range(len(actual)):
        actual_column_name = actual_column()[i]
        pred_column_name = targets()[i]
        result = error_with_df(data["df"], actual_column_name, pred_column_name)
        results.append(result)

    results_str = json.dumps(results)
    # Compute errors
    return results_str

    # return { "mae": result["mae"], "rmse": result["rmse"],
    #         "r2": result["r2"], "mape": result["mape"], "mse":  result["mse"] }