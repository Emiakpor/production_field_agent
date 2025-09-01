# train_model.py
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import losses, metrics, optimizers
# ---------------------------
# Config
# ---------------------------
INPUT_EXCEL = "./src/resource/oilfield_sample_data_with_units.xlsx"
INPUT_EXCEL_LSTM = "./src/resource/Volve_Oil_Production.xlsx"
# INPUT_EXCEL = "./src/resource/monthly_sum_excel.xlsx"
# INPUT_EXCEL = "./src/resource/forecast_input_sample_extended.xlsx"
MODEL_PATH = "./src/created_model/xgb_predict_model.joblib"
LSTM_MODEL_PATH = "./src/created_model/lstm_predict_model.h5"
LSTM_MODEL_PATH_VOLVE = "./src/created_model/volve_lstm_predict_model.h5"

FEATURES_PATH = "./src/resource/feature_order.json"
LSTM_FEATURES_PATH = "./src/resource/lstm_feature_order.json"

TARGET_COLS = [
    "oil_rate_bopd (BOPD)",
    "gas_rate_mscf_day (MSCF/day)",
    "water_rate_bwpd (BWPD)",
]

LSTM_TARGET_COLS = [
    "oil_rate_bopd (BOPD)",
    "gas_rate_mscf_day (MSCF/day)",
    "water_rate_bwpd (BWPD)",
]

# ---------------------------
# Helpers
# ---------------------------
def normalize_col(col: str) -> str:
    """
    Strip units in parentheses and whitespace, convert to snake_case.
    """
    base = re.sub(r"\s*\(.*?\)\s*", "", col).strip()
    return base.lower().replace(" ", "_")

def load_dataset():
    # ---------------------------
    # Load dataset
    # ---------------------------
    df = pd.read_excel(INPUT_EXCEL)

    # Normalize column names
    df_norm = df.rename(columns={c: normalize_col(c) for c in df.columns})

    # Features and target
    y = df_norm[[normalize_col(c) for c in TARGET_COLS]]
    X = df_norm.drop(columns=[normalize_col(c) for c in TARGET_COLS])

    return {"X": X, "y": y}

def load_dataset_lstm():
    # ---------------------------
    # Load dataset
    # ---------------------------
    df = pd.read_excel(INPUT_EXCEL_LSTM)

    # Normalize column names
    df_norm = df.rename(columns={c: normalize_col(c) for c in df.columns})

    # Features and target
    y = df_norm[[normalize_col(c) for c in LSTM_TARGET_COLS]]
    X = df_norm.drop(columns=[normalize_col(c) for c in LSTM_TARGET_COLS])

    return {"X": X, "y": y}


def identify_categorical_vs_numeric(X):
    # Identify categorical vs numeric
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype == "bool"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    return {"categorical_cols": categorical_cols, "numeric_cols": numeric_cols}

def build_xbg_model_predict():
    df = load_dataset()
    data = identify_categorical_vs_numeric(df["X"])
    categorical_cols = data["categorical_cols"]
    numeric_cols = data["numeric_cols"]
    # ---------------------------
    # Preprocessing
    # ---------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # ---------------------------
    # Model: Multi-target XGBoost
    # ---------------------------
    base_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base_model, n_jobs=-1)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


    # ---------------------------
    # Train-test split
    # ---------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        df["X"], df["y"], test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    # ---------------------------
    # Evaluate
    # ---------------------------

    y_pred = pipeline.predict(X_val)
    print("Validation R²:", r2_score(y_val, y_pred, multioutput="raw_values"))
    print("Validation MAE:", mean_absolute_error(y_val, y_pred, multioutput="raw_values"))

    # ---------------------------
    # Save artifacts
    # ---------------------------
    dump(pipeline, MODEL_PATH)
    

    # Save feature order (after preprocessing, but store raw cols for input mapping)
    feature_order = list(df["X"].columns)
    Path(FEATURES_PATH).write_text(json.dumps(feature_order, indent=2))
    

def build_lstm_model_predict():

    df = load_dataset()

    feature_order = list(df["X"].columns)
    Path(LSTM_FEATURES_PATH).write_text(json.dumps(feature_order, indent=2))

    lookback: int = 30
    n_features = len(feature_order)
    horizon = 7  # forecast length for LSTM

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
        LSTM(32),
        Dense(horizon)
    ])
    model.compile(optimizer="adam", loss="mse")

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanSquaredError()]
    )

    model.save(LSTM_MODEL_PATH_VOLVE)
    return model