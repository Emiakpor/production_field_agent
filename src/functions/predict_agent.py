# batch_predict.py (no argparse version)
import re
import pandas as pd
from joblib import load
import json
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from tensorflow.keras.models import load_model
from tensorflow.keras import losses, metrics

from src.models.models import ForecastRequest
from src.functions.utils import load_json, actual_column, predict_column, error, get_monthly_average, aggregate_excel_monthly_by_well

MODEL_PATH = "./src/created_model/xgb_predict_model.joblib"
LSTM_MODEL_PATH = "./src/created_model/lstm_predict_model.h5"
FEATURES_PATH = "./src/resource/feature_order.json"

# Default paths
# INPUT_EXCEL = "./src/resource/oilfield_sample_data_with_units.xlsx"
INPUT_EXCEL = "./src/resource/oilfield_sample_data_with_units.xlsx"
OUTPUT_EXCEL = "./src/resource/predictions.xlsx"
train_history = "./src/resource/train_history.xlsx"
monthly_sum_excel = "./src/resource/monthly_sum_excel.xlsx"

# Load model + feature names
pipeline = load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    feature_order = json.load(f)

def getFeature():    
    return load_json(FEATURES_PATH)

def load_data():
    df = pd.read_excel(INPUT_EXCEL)

    # Normalize headers to match training step
    df_norm = df.rename(columns={c: normalize_col(c) for c in df.columns})

    # Drop targets if present (we're predicting them)
    for target in ["oil_rate_bopd", "gas_rate_mscf_day", "water_rate_bwpd"]:
        if target in df_norm.columns:
            df_norm = df_norm.drop(columns=[target])

    return {
        "df": df,
        "df_norm": df_norm
    }

def get_monthly_average_data():
    result = get_monthly_average(INPUT_EXCEL, monthly_sum_excel, "timestamp", ["well_id"])

    return result

def convert_features_to_numeric(df, exclude_cols=None):
    
    if exclude_cols is None:
        exclude_cols = []

    # Columns to convert
    cols_to_convert = [col for col in df.columns if col not in exclude_cols]

    # Convert selected columns to numeric (invalid parsing becomes NaN)
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    monthly_agg = df.groupby(["well_id", "day_of_week"])[cols_to_convert].mean().reset_index()
    
    out = pd.concat([monthly_agg], axis=1)
    out.to_excel(monthly_sum_excel, index=False)
    return df

# Helper to normalize Excel columns
def normalize_col(col: str) -> str:
    base = re.sub(r"\s*\(.*?\)\s*", "", col).strip()
    return base.lower().replace(" ", "_")

def decision_rules(preds: pd.DataFrame):
    actions = []
    for _, row in preds.iterrows():
        if row["water_rate_bwpd_pred"] > 5000:  
            actions.append("High water cut detected – recommend water shutoff treatment")
        elif row["gas_rate_mscf_day_pred"] < 200:
            actions.append("Low gas rate – possible gas lift issue")
        else:
            actions.append("Operating normally")
    return actions

def explain( input_df: pd.DataFrame, sample_idx=0):
    """Return SHAP explanations for a sample"""
    model = pipeline.named_steps["model"].estimators_[0]  # oil model example
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pipeline.named_steps["preprocess"].transform(input_df))

    # -------------------------
    # Local (Waterfall Plot)
    # -------------------------
    buf_local = BytesIO()
    fig_local = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(buf_local, format="png", bbox_inches="tight")
    plt.close(fig_local)
    buf_local.seek(0)
    shap_local_b64 = base64.b64encode(buf_local.read()).decode("utf-8")

    explanation = dict(zip(
        pipeline.named_steps["preprocess"].get_feature_names_out(),
        shap_values[0].values.tolist()
    ))

    return {
        "explanation": explanation,
        "shap_local_b64": shap_local_b64,   # base64 image string
    # "shap_global_b64": shap_global_b64 ,  # base64 image string
        # "time_series_plot": plot_path  # file path to saved image (if generated)
    }

def xgb_predict():

    data = load_data()

    df = data["df"]
    df_norm = data["df_norm"]
    # Run prediction on DataFrame
    y_pred = pipeline.predict(df_norm)

    # Build prediction DataFrame
    target_names = ["oil_rate_bopd_pred", "gas_rate_mscf_day_pred", "water_rate_bwpd_pred"][: y_pred.shape[1]]
    df_pred = pd.DataFrame(y_pred, columns=target_names)

    actions = decision_rules(df_pred)
    df_pred["action"] = actions

    # Merge original + predictions
    out = pd.concat([df, df_pred], axis=1)
    out.to_excel(OUTPUT_EXCEL, index=False)

    data = explain(df_norm, sample_idx=0)

    return {
        "explanation": data["explanation"],
        "shap_local_b64": data["shap_local_b64"], 
        # "shap_global_b64": data["shap_global_b64"]
        # "time_series_plot": data["plot_path"]  # file path to saved image (if generated) 
    }

def lstm_predict():
    data = load_data()  
    df= data["df_norm"]
    
    lookback: int = 30     
    horizon = 7

    model = load_model(LSTM_MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mse")
    features = getFeature()
    # prepare input sequence
    last_seq = df[features].tail(lookback).values.reshape(1, lookback, len(features))
    preds = model.predict(last_seq)

    # Build forecast dataframe
    future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=horizon+1, freq="D")[1:]
    forecast_df = pd.DataFrame({
        "timestamp": future_dates,
        "oil_rate_bopd_pred": preds[0]  # here assume 1 target (can expand to multi-output)
    })

    preds_df = forecast_df.to_dict(orient="records")
    return {
        "forecast_df": forecast_df,
        "preds_df": preds_df
    }

def error_predict():
    actual = actual_column()

    results = {}
    for i in range(len(actual)):
        actual_column_name = actual_column()[i]
        pred_column_name = predict_column()[i]
        result = error(OUTPUT_EXCEL, actual_column_name, pred_column_name)
        results[actual_column_name] = result

    return results
    



