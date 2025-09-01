import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
import json
from src.functions.utils import load_dataset

LSTM_MODEL_PATH = "./src/created_model/lstm_model.h5"
SCALER_X_PATH = "./src/created_model/scaler_X.pkl"
SCALER_Y_PATH = "./src/created_model/scaler_y.pkl"
forecast_lstm = "./src/resource/forecast_lstm.xlsx"
FEATURES_PATH = "./src/resource/features.json"

train_history = "./src/resource/train_history1.xlsx"
sample_path = "./src/resource/forecast_input_sample_extended.xlsx"
synthetic_production_numeric = "./src/resource/synthetic_production_numeric.xlsx"


def scale_data(X, y):

    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Save scalers to file
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    return {
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "X_scaled": X_scaled,
        "y_scaled": y_scaled
    }

# Create sequences (30-day history → next-day prediction)
def create_sequences(X, y, seq_length=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model_forecast():
    data = load_dataset(synthetic_production_numeric)
    scaled_data = scale_data(data["X"], data["y"])
    seq_length=30
    X_seq, y_seq = create_sequences(scaled_data["X_scaled"], scaled_data["y_scaled"], seq_length)

    # Train-test split
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(32),
        Dense(y_train.shape[1])  # multi-output
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)
    model.save(LSTM_MODEL_PATH)

    feature_order = list(data["X"].columns)
    Path(FEATURES_PATH).write_text(json.dumps(feature_order, indent=2))

    # # Forecast
    # y_pred_scaled = model.predict(X_test)
    # y_pred = scaled_data["scaler_y"].inverse_transform(y_pred_scaled)
