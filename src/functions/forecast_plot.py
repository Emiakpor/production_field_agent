import matplotlib.pyplot as plt
import os
import pandas as pd

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_forecast_by_field(df, field_name, well_name, features, plot_type_label):
    # filter data
    df_plot = df[(df["field_name_label"] == field_name) & (df["well_name_label"] == well_name)]
    
    if df_plot.empty:
        print(f"No data found for {field_name} / {well_name}")
        return
    
    plt.figure(figsize=(10, 6))
    
    for feature in features:
        # forecast column name based on your naming convention
        forecast_col = None
        if feature == "oil_rate_bopd (BOPD)":
            forecast_col = "pred_oil_rate_bopd"
        elif feature == "gas_rate_mscf_day (MSCF/day)":
            forecast_col = "pred_gas_rate_mscf_day"
        elif feature == "water_rate_bwpd (BWPD)":
            forecast_col = "pred_water_rate_bwpd"
        else:
            print(f"Unknown feature '{feature}' — skipping.")
            continue

        # plot actual
        plt.plot(df_plot["timestamp"], df_plot[feature], label=f"{feature} - Actual")
        # plot forecast
        plt.plot(df_plot["timestamp"], df_plot[forecast_col], label=f"{feature} - {plot_type_label}", linestyle="--")
    
    plt.xlabel("Time")
    plt.ylabel("Production Rate")
    plt.title(f"{plot_type_label} vs Actual - {field_name} / {well_name}")
    plt.legend()
    filename = f"{field_name}_{well_name}_{plot_type_label}.png".replace(" ", "_")
    # filepath = os.path.join(save_dir, filename)
    fname = os.path.join(PLOT_DIR, filename)
    plt.savefig(fname)
    plt.grid(True)
        
    # plt.savefig(filepath, dpi=300, bbox_inches="tight")
    
    plt.show()

def plot_forecast_by_well(df, well_name, features, plot_type_label, start_date=None, end_date=None):
    # filter data
    df_plot = df[(df["well_id"] == well_name)]
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

    if start_date:
        df_plot = df_plot[df_plot["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df_plot = df_plot[df_plot["timestamp"] <= pd.to_datetime(end_date)]

    if df_plot.empty:
        print(f"No data found for {well_name}")
        return
    
    plt.figure(figsize=(10, 6))
    
    for feature in features:
        # forecast column name based on your naming convention
        forecast_col = None
        if feature == "oil_rate_bopd (BOPD)":
            forecast_col = "oil_rate_bopd_pred"
        elif feature == "gas_rate_mscf_day (MSCF/day)":
            forecast_col = "gas_rate_mscf_day_pred"
        elif feature == "water_rate_bwpd (BWPD)":
            forecast_col = "water_rate_bwpd_pred"
        else:
            print(f"Unknown feature '{feature}' — skipping.")
            continue

        # plot actual
        plt.plot(df_plot["timestamp"], df_plot[feature], label=f"{feature} - Actual")
        # plot forecast
        plt.plot(df_plot["timestamp"], df_plot[forecast_col], label=f"{feature} - {plot_type_label}", linestyle="--")
    
    plt.xlabel("Time")
    plt.ylabel("Production Rate")
    plt.title(f"{plot_type_label} vs Actual - {well_name}")
    plt.legend()
    filename = f"{well_name}_{plot_type_label}.png".replace(" ", "_")
    # filepath = os.path.join(save_dir, filename)
    fname = os.path.join(PLOT_DIR, filename)
    plt.savefig(fname)
    plt.grid(True)
        
    # plt.savefig(filepath, dpi=300, bbox_inches="tight")
    
    plt.show()

def filter_production_data(df, field_name=None, well_name=None, 
                           start_date=None, end_date=None, min_oil_rate=None):
       
    df_filtered = df.copy()

    # Field filter
    if field_name:
        df_filtered = df_filtered[df_filtered["field_name"] == field_name]
    
    # Well filter (single or list)
    if well_name:
        if isinstance(well_name, list):
            df_filtered = df_filtered[df_filtered["well_name"].isin(well_name)]
        else:
            df_filtered = df_filtered[df_filtered["well_name"] == well_name]
    
    # Date range filter
    if start_date:
        df_filtered = df_filtered[df_filtered["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered["timestamp"] <= pd.to_datetime(end_date)]
    
    # Minimum oil rate filter
    if min_oil_rate:
        df_filtered = df_filtered[df_filtered["oil_rate_bopd"] >= min_oil_rate]
    
    return df_filtered

