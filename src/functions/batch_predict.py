# batch_predict.py (no argparse version)
import re
import pandas as pd
from joblib import load
import json

MODEL_PATH = "./src/created_model/model_xgb_multi.joblib"
FEATURES_PATH = "./src/resource/feature_order.json"

# Default paths
INPUT_EXCEL = "./src/resource/oilfield_sample_data_with_units.xlsx"
OUTPUT_EXCEL = "./src/resource/predictions.xlsx"

# Load model + feature names
pipeline = load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    feature_order = json.load(f)

# Helper to normalize Excel columns
def normalize_col(col: str) -> str:
    base = re.sub(r"\s*\(.*?\)\s*", "", col).strip()
    return base.lower().replace(" ", "_")

def predict():
    df = pd.read_excel(INPUT_EXCEL)

    # Normalize headers to match training step
    df_norm = df.rename(columns={c: normalize_col(c) for c in df.columns})

    # Drop targets if present (we're predicting them)
    for target in ["oil_rate_bopd", "gas_rate_mscf_day", "water_rate_bwpd"]:
        if target in df_norm.columns:
            df_norm = df_norm.drop(columns=[target])

    # Run prediction on DataFrame
    y_pred = pipeline.predict(df_norm)

    # Build prediction DataFrame
    target_names = ["oil_rate_bopd_pred", "gas_rate_mscf_day_pred", "water_rate_bwpd_pred"][: y_pred.shape[1]]
    df_pred = pd.DataFrame(y_pred, columns=target_names)

    # Merge original + predictions
    out = pd.concat([df, df_pred], axis=1)
    out.to_excel(OUTPUT_EXCEL, index=False)

    print(f"✅ Predictions saved to {OUTPUT_EXCEL}")
