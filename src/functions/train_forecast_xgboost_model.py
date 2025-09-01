import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
from pathlib import Path
import json
from src.functions.utils import load_dataset

XGB_MODEL_PATH = "./src/created_model/xgb_model.pkl"
FEATURES_PATH = "./src/resource/features.json"

forecast_xgb = "./src/resource/forecast_xgb.xlsx"
synthetic_production_numeric = "./src/resource/synthetic_production_numeric.xlsx"

def train_test(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return {
        "X_train": X_train,
        "X_test": X_test, 
        "y_train": y_train,
        "y_test": y_test
    }

def build_xgboost_model_forecast():
    data = load_dataset(synthetic_production_numeric)
    train_data = train_test(data["X"], data["y"])
    # Train XGBoost model (multi-output)
    xgb = MultiOutputRegressor(XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5))
    xgb.fit(train_data["X_train"], train_data["y_train"])

    joblib.dump(xgb, XGB_MODEL_PATH)

    # Forecast
    y_pred = xgb.predict(train_data["X_test"])

    feature_order = list(data["X"].columns)
    Path(FEATURES_PATH).write_text(json.dumps(feature_order, indent=2))

    # Calculate MSE for each target
    # mse_scores = {}
    # for i, col in enumerate(data["y"].columns):
    #     mse_scores[col] = mean_squared_error(train_data["y_test"].iloc[:, i], y_pred[:, i])

    # Evaluate
    # print("RMSE:", np.sqrt(mean_squared_error(train_data["y_test"], y_pred)))
