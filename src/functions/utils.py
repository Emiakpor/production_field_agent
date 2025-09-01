import re
import json
import os
import matplotlib.pyplot as plt
import shap
import pandas as pd
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

features_targets_path = "./src/resource/features_target.json"
synthetic_production_numeric = "./src/resource/synthetic_production_numeric.xlsx"

features = []

def features_df(df: pd.DataFrame):
    features_ = [col for col in df.columns if col not in targets() + id_col()]

    return features_

def id_col():
    return ["field_name_label", "well_name_label", "timestamp"]

def actual_column():
    return [
        "oil_rate_bopd (BOPD)",
        "gas_rate_mscf_day (MSCF/day)",
        "water_rate_bwpd (BWPD)"
    ]

def predict_column():
    return [
        "oil_rate_bopd_pred",
        "gas_rate_mscf_day_pred",
        "water_rate_bwpd_pred"
    ]

def targets():
    return [
        "pred_oil_rate_bopd",
        "pred_gas_rate_mscf_day",
        "pred_water_rate_bwpd"
    ]

def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
        
    return data

def load_dataset(excel_file):
    # Load dataset
    df = pd.read_excel(excel_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    features = features_df(df)
    

    X = df[features]
    y = df[targets()]
    return {
        "df": df,
        "X": X,
        "y": y
    }

def clean_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)

def normalize_col(col: str) -> str:
    base = re.sub(r"\s*\(.*?\)\s*", "", col).strip()
    return base.lower().replace(" ", "_")

def clean_inputs(df: pd.DataFrame):
    # Normalize headers to match training step
    return df.rename(columns={c: normalize_col(c) for c in df.columns})

def shap_time_series(X, explainer, pipeline, model, df_timeseries, feature_order, time_col="timestamp"):
    """Compute SHAP values over time for multiple rows (timestamps)."""

    X_transformed = pipeline.named_steps["preprocess"].transform(X)

    shap_values = explainer(X_transformed)
    shap_df = pd.DataFrame(
        shap_values.values,
        columns=pipeline.named_steps["preprocess"].get_feature_names_out()
    )
    shap_df[time_col] = df_timeseries[time_col].values
    return shap_df


def save_shap_time_series_plot(shap_df, feature_list, out_dir="plots", filename="shap_timeseries.png", time_col="timestamp"):
    """Save SHAP values for selected features over time as PNG."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    plt.figure(figsize=(12,6))
    for f in feature_list:
        if f in shap_df.columns:
            plt.plot(shap_df[time_col], shap_df[f], label=f, linewidth=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("SHAP Contribution")
    plt.title("Feature Influence Over Time")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path

def error(filepath, actual_column_name, pred_column_name):
    df = pd.read_excel(filepath)
    # Compute errors
    mae = mean_absolute_error(df[actual_column_name], df[pred_column_name])
    mse = root_mean_squared_error(df[actual_column_name], df[pred_column_name])
    r2 = r2_score(df[actual_column_name], df[pred_column_name])
    rmse = mse ** 0.5
    mape = (abs((df[actual_column_name] - df[pred_column_name]) / df[actual_column_name]).mean()) * 100

    return { "mae": mae, "rmse": rmse,
            "r2": r2, "mape": mape, "mse":  mse }

def error_with_df(df: pd.DataFrame, actual_column_name, pred_column_name):
    # Compute errors
    mae = mean_absolute_error(df[actual_column_name], df[pred_column_name])
    mse = root_mean_squared_error(df[actual_column_name], df[pred_column_name])
    r2 = r2_score(df[actual_column_name], df[pred_column_name])
    rmse = mse ** 0.5
    mape = (abs((df[actual_column_name] - df[pred_column_name]) / df[actual_column_name]).mean()) * 100

    return { "mae": mae, "rmse": rmse,
            "r2": r2, "mape": mape, "mse":  mse }

def get_monthly_average(input_path, result_path, date_col, group_cols):
    monthly_df = aggregate_excel_monthly_by_well(
        file_path=input_path,
        sheet_name=0,
        date_col= date_col,
        agg_func="mean",
        group_cols=group_cols,  # aggregate per well
    )

    out = pd.concat([monthly_df], axis=1)
    out.to_excel(result_path, index=False)

    return {"result": "Created"}


def get_mixe_column(df):
    columns = []
    for col in df.columns:
        if df[col].dtype == "object":
            columns.append(col)
    
    return columns

def aggregate_excel_monthly_full(file_path, sheet_name=0, date_col="date", agg_func="mean"):
    """
    Read Excel, aggregate numeric columns monthly, and retain all categorical columns.

    Parameters:
        file_path (str): Path to Excel file.
        sheet_name (str or int): Sheet name or index.
        date_col (str): Datetime column name.
        agg_func (str or dict): Aggregation function for numeric columns.
        exclude_cols (list, optional): Columns to exclude from numeric conversion/aggregation.
        
    Returns:
        pd.DataFrame: Monthly aggregated DataFrame with categorical columns retained.
    """
    exclude_cols = get_mixe_column(df)

    if exclude_cols is None:
        exclude_cols = []

    # 1️⃣ Read Excel
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 2️⃣ Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # 3️⃣ Convert numeric columns
    cols_to_convert = [col for col in df.columns if col not in exclude_cols + [date_col]]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    # 4️⃣ Create year/month for grouping
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month

    # 5️⃣ Identify numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["year", "month"] + exclude_cols]

    # 6️⃣ Identify categorical columns to retain
    # categorical_cols = [col for col in df.columns if col not in numeric_cols + [date_col]]
    categorical_cols = [col for col in df.columns if col not in numeric_cols + ["year", "month", date_col]]
    print(categorical_cols)
    # 7️⃣ Aggregate numeric columns
    monthly_numeric = df.groupby(["year", "month"] + categorical_cols)[numeric_cols].agg(agg_func).reset_index()

    # 8️⃣ Optional: add month_start
    monthly_numeric["month_start"] = pd.to_datetime(
        monthly_numeric["year"].astype(str) + "-" + monthly_numeric["month"].astype(str) + "-01"
    )

    return monthly_numeric


def aggregate_excel_monthly_by_well(file_path, sheet_name=0, date_col="date", agg_func="mean", group_cols=None, exclude_cols=None):
    """
    Read Excel, aggregate numeric columns monthly by specified group columns (e.g., well_name), retaining categorical columns.
    
    Parameters:
        file_path (str): Path to Excel file.
        sheet_name (str or int): Sheet name or index.
        date_col (str): Datetime column name.
        agg_func (str or dict): Aggregation function for numeric columns.
        group_cols (list, optional): Categorical columns to group by (e.g., ["well_name"]).
        exclude_cols (list, optional): Columns to exclude from numeric conversion/aggregation.
        
    Returns:
        pd.DataFrame: Monthly aggregated DataFrame grouped by specified columns.
    """
    if exclude_cols is None:
        exclude_cols = []
    if group_cols is None:
        group_cols = []

    # 1️⃣ Read Excel
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 2️⃣ Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # 3️⃣ Convert numeric columns
    cols_to_convert = [col for col in df.columns if col not in exclude_cols + [date_col]]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    # 4️⃣ Create year/month for grouping
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month

    # 5️⃣ Identify numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["year", "month"] + exclude_cols + group_cols]

    # 6️⃣ Aggregate numeric columns by year, month, and group_cols
    monthly_agg = df.groupby(["year", "month"] + group_cols)[numeric_cols].agg(agg_func).reset_index()

    # 7️⃣ Add month_start for convenience
    monthly_agg[date_col] = pd.to_datetime(
        monthly_agg["year"].astype(str) + "-" + monthly_agg["month"].astype(str) + "-01"
    )

    return monthly_agg

