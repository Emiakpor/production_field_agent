import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from src.functions.forecast_plot import plot_forecast_by_well

OUTPUT_EXCEL = "./src/resource/predictions.xlsx"

# --------------------------
# Helper: safe filenames
# --------------------------
def clean_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)

# --------------------------
# Output directory
# --------------------------
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --------------------------
# Load dataset
# --------------------------
df = pd.read_excel(OUTPUT_EXCEL)

# detect timestamp
time_col = None
for cand in ["timestamp", "datetime", "date", "time"]:
    if cand in df.columns:
        time_col = cand
        break

if time_col:
    df[time_col] = pd.to_datetime(df[time_col])
else:
    df["timestamp"] = pd.date_range(start="2023-01-01", periods=len(df), freq="h")
    time_col = "timestamp"

# Ensure well_id exists
if "well_id" not in df.columns:
    df["well_id"] = "Well-1"

# --------------------------
# Targets
# --------------------------
targets = [
    ("oil_rate_bopd (BOPD)", "oil_rate_bopd_pred"),
    ("gas_rate_mscf_day (MSCF/day)", "gas_rate_mscf_day_pred"),
    ("water_rate_bwpd (BWPD)", "water_rate_bwpd_pred"),
]

# --------------------------
# Function: dashboard for one well
# --------------------------
def plot_dashboard_for_well(well_id, df):
    df_well = df[df["well_id"] == well_id].copy()
    if df_well.empty:
        print(f"sNo data found for {well_id}")
        return

    fig, axes = plt.subplots(nrows=len(targets), ncols=1, figsize=(12, 10), sharex=True)

    for i, (actual, pred) in enumerate(targets):
        ax = axes[i]
        if actual not in df_well.columns or pred not in df_well.columns:
            ax.set_visible(False)
            continue
        ax.plot(df_well[time_col], df_well[actual], label=f"Actual {actual}", linewidth=2)
        ax.plot(df_well[time_col], df_well[pred], label=f"Predicted {pred}", linestyle="--", linewidth=2)
        ax.set_ylabel(actual)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("Timestamp")
    fig.suptitle(f"Forecast vs Actual Over Time for {well_id}", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fname = os.path.join(PLOT_DIR, f"dashboard_{clean_filename(well_id)}.png")
    plt.savefig(fname)
    plt.close()
    print(f"📊 Saved dashboard for {well_id}: {fname}")


def plot_predict1():
    # --------------------------
    # Generate one dashboard per well
    # --------------------------
    for well in df["well_id"].unique():
        plot_dashboard_for_well(well, df)

def plot_predict():
    # data = load_dataset(forecast_xgb)
    df = pd.read_excel(OUTPUT_EXCEL)
    df = df.rename(columns={"timestamp (ISO8601)": "timestamp" })

    wells = ["Well-1", "Well-2", "Well-3"]
    # features = ["oil_rate_bopd (BOPD)", "gas_rate_mscf_day (MSCF/day)", "water_rate_bwpd (BWPD)"]
    for well in wells:
        # features = ["oil_rate_bopd (BOPD)"]
        # features = ["gas_rate_mscf_day (MSCF/day)"]
        features = ["water_rate_bwpd (BWPD)"]
        plot_forecast_by_well(df, well,  features, "Predict", "2025-01-01T00:00:00", "2025-12-31T18:00:00")