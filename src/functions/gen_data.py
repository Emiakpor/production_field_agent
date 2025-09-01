import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

sample_path = "./src/resource/forecast_input_sample_extended.xlsx"
history_path = "./src/resource/train_history1.xlsx"
history_path2 = "./src/resource/train_history2.xlsx"
synthetic_production_numeric = "./src/resource/synthetic_production_numeric.xlsx"
forecast_production_numeric = "./src/resource/forecast_production_numeric.xlsx"

# Generate 7 days of synthetic data
def generate_template_data():
    dates = pd.date_range(start="2025-09-01", periods=7, freq="D")
    np.random.seed(42)

    data = {
        "timestamp (ISO8601)": dates.astype(str),

        # Operational
        "choke_size_in (inches)": np.random.choice([0.5, 0.6, 0.7], size=7),
        "choke_size_percent (%)": np.random.randint(40, 80, 7),
        "gas_lift_rate_mscf_day (MSCF/day)": np.random.randint(100, 500, 7),
        "pump_frequency_hz (Hz)": np.random.randint(40, 60, 7),
        "pump_discharge_pressure_psi (psi)": np.random.randint(1000, 2000, 7),
        "tubing_head_pressure_psi (psi)": np.random.randint(500, 1500, 7),
        "casing_head_pressure_psi (psi)": np.random.randint(300, 1200, 7),

        # Reservoir & Wellbore
        "reservoir_pressure_initial (psi)": np.linspace(4000, 3950, 7),
        "bubble_point_pressure (psi)": np.random.randint(1800, 2200, 7),
        "permeability_md (mD)": np.random.uniform(10, 200, 7),
        "porosity_percent (%)": np.random.uniform(10, 25, 7),
        "net_pay_thickness_ft (ft)": np.random.randint(20, 50, 7),
        "drawdown_psi (psi)": np.random.randint(100, 300, 7),
        "productivity_index_bbl_day_psi (bbl/day/psi)": np.random.uniform(0.5, 2.0, 7),

        # Surface & Process
        "separator_inlet_pressure_psi (psi)": np.random.randint(300, 800, 7),
        "separator_inlet_temperature_c (°C)": np.random.uniform(25, 40, 7),
        "compressor_suction_psi (psi)": np.random.randint(200, 600, 7),
        "compressor_discharge_psi (psi)": np.random.randint(800, 1500, 7),
        "chemical_injection_rate_lph (L/h)": np.random.uniform(5, 15, 7),
        "dp_across_choke_psi (psi)": np.random.randint(20, 100, 7),

        # Monitoring & Reliability
        "esp_vibration_mm_s (mm/s)": np.random.uniform(0.1, 0.5, 7),
        "esp_current_amp (A)": np.random.randint(50, 100, 7),
        "pump_efficiency_bbl_per_kw (bbl/kW)": np.random.uniform(5, 15, 7),
        "maintenance_event (bool)": np.random.choice([0, 1], size=7),
        "shutdown_flag (bool)": np.random.choice([0, 1], size=7),

        # Environmental
        "ambient_temp_c (°C)": np.random.uniform(25, 35, 7),
        "ambient_pressure_psi (psi)": np.random.uniform(14, 15, 7),
        "humidity_percent (%)": np.random.randint(40, 90, 7),
        "grid_voltage_stability_percent (%)": np.random.uniform(90, 100, 7),

        # Targets
        "oil_rate_bopd (BOPD)": np.random.randint(800, 2000, 7),
        "gas_rate_mscf_day (MSCF/day)": np.random.randint(300, 1200, 7),
        "water_rate_bwpd (BWPD)": np.random.randint(100, 600, 7),
        "gor_scf_bbl (SCF/bbl)": np.random.randint(500, 1500, 7),
        "water_cut_percent (%)": np.random.randint(10, 60, 7),
    }

    df_sample = pd.DataFrame(data)

    # Save sample filled data to Excel
    df_sample.to_excel(sample_path, index=False)

    return sample_path

def generate_history_data():
   # -----------------------------
   # 1. Setup
   # -----------------------------
   np.random.seed(42)
   num_days = 365
   wells = ["Well_A", "Well_B", "Well_C"]
   fields = ["Field_1", "Field_2"]
   
   dates = pd.date_range(start="2024-01-01", periods=num_days, freq='D')
   
   # -----------------------------
   # 2. Generate Data
   # -----------------------------
   df_list = []
   for field in fields:
       for well in wells:
           df_well = pd.DataFrame({'timestamp': dates})
           df_well['field_name_label'] = field
           df_well['well_name_label'] = well

           df_well['field_name'] = field
           df_well['well_name'] = well

           # Base production
           df_well['oil_rate_bopd (BOPD)'] = np.random.uniform(80, 120, size=num_days)
           df_well['gas_rate_mscf_day (MSCF/day)'] = np.random.uniform(50, 100, size=num_days)
           df_well['water_rate_bwpd (BWPD)'] = np.random.uniform(20, 40, size=num_days)
   
           # Operational parameters
           df_well['pressure (psi)'] = np.random.uniform(2500, 3000, size=num_days)
           df_well['choke (%)'] = np.random.uniform(70, 100, size=num_days)
           df_well['water_cut (%)'] = df_well['water_rate_bwpd (BWPD)'] / \
               (df_well['oil_rate_bopd (BOPD)'] + df_well['water_rate_bwpd (BWPD)']) * 100
           df_well['temperature (°C)'] = np.random.uniform(60, 90, size=num_days)
   
           # Operational events
           df_well['maintenance_event'] = np.random.choice([0,1], size=num_days, p=[0.95,0.05])
           df_well['shutdown_flag'] = np.random.choice([0,1], size=num_days, p=[0.98,0.02])
           df_well['derate_factor'] = 1.0
           df_well.loc[df_well['maintenance_event']==1, 'derate_factor'] = np.random.uniform(0.7, 0.9)
           df_well.loc[df_well['shutdown_flag']==1, 'derate_factor'] = 0.0
   
           # Apply derate
           df_well['oil_rate_bopd (BOPD)'] *= df_well['derate_factor']
           df_well['gas_rate_mscf_day (MSCF/day)'] *= df_well['derate_factor']
           df_well['water_rate_bwpd (BWPD)'] *= df_well['derate_factor']
   
           # Temporal features
           df_well['day_of_week'] = df_well['timestamp'].dt.dayofweek
           df_well['month'] = df_well['timestamp'].dt.month
           df_well['day_of_year'] = df_well['timestamp'].dt.dayofyear
           df_well['is_weekend'] = df_well['day_of_week'].isin([5,6]).astype(int)
           df_well['season'] = df_well['month'].map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
                                                     6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'})
   
           # Lag & rolling features
           df_well['oil_rate_lag_1'] = df_well['oil_rate_bopd (BOPD)'].shift(1).fillna(method='bfill')
           df_well['gas_rate_lag_1'] = df_well['gas_rate_mscf_day (MSCF/day)'].shift(1).fillna(method='bfill')
           df_well['water_rate_lag_1'] = df_well['water_rate_bwpd (BWPD)'].shift(1).fillna(method='bfill')
           df_well['rolling_avg_oil_3'] = df_well['oil_rate_bopd (BOPD)'].rolling(3, min_periods=1).mean()
           df_well['rolling_avg_gas_7'] = df_well['gas_rate_mscf_day (MSCF/day)'].rolling(7, min_periods=1).mean()
   
           # Derived ratios
           df_well['GOR'] = df_well['gas_rate_mscf_day (MSCF/day)'] / df_well['oil_rate_bopd (BOPD)']
           df_well['water_cut_ratio'] = df_well['water_rate_bwpd (BWPD)'] / \
               (df_well['oil_rate_bopd (BOPD)'] + df_well['water_rate_bwpd (BWPD)'])
           df_well['choke_efficiency'] = df_well['choke (%)'] / 100
           df_well['production_efficiency'] = df_well['derate_factor']
   
           # Metadata
           df_well['well_age_days'] = df_well['day_of_year']
           df_well['cumulative_oil_produced_bbl'] = df_well['oil_rate_bopd (BOPD)'].cumsum()
           df_well['field_type'] = 'Oil'
           df_well['well_type'] = 'Production'
           df_well['drilling_method'] = 'Horizontal'
           df_well['completion_type'] = 'Cased'
           df_well['is_holiday'] = df_well['timestamp'].isin(pd.to_datetime(['2024-01-01','2024-12-25'])).astype(int)
           df_well['special_operation'] = np.random.choice([0,1], size=num_days, p=[0.97,0.03])
   
           # Environmental
           df_well['weather_temp'] = np.random.uniform(20,35,size=num_days)
           df_well['rain_mm'] = np.random.uniform(0,20,size=num_days)
   
           # Targets
        #    df_well['target_oil_next_day'] = df_well['oil_rate_bopd (BOPD)'].shift(-1)
        #    df_well['target_gas_next_day'] = df_well['gas_rate_mscf_day (MSCF/day)'].shift(-1)
        #    df_well['target_water_next_day'] = df_well['water_rate_bwpd (BWPD)'].shift(-1)
   
           df_well['pred_oil_rate_bopd'] = df_well['oil_rate_bopd (BOPD)'].shift(-1)
           df_well['pred_gas_rate_mscf_day'] = df_well['gas_rate_mscf_day (MSCF/day)'].shift(-1)
           df_well['pred_water_rate_bwpd'] = df_well['water_rate_bwpd (BWPD)'].shift(-1)

           df_list.append(df_well)
   
   # -----------------------------
   # 3. Combine all wells
   # -----------------------------
   df_all = pd.concat(df_list, ignore_index=True)
   
   # Fill last-day targets
   df_all[['pred_oil_rate_bopd','pred_gas_rate_mscf_day','pred_water_rate_bwpd']] = \
       df_all[['pred_oil_rate_bopd','pred_gas_rate_mscf_day','pred_water_rate_bwpd']].fillna(0)
   
   # -----------------------------
   # 4. Encode categorical columns numerically
   # -----------------------------
   for col in ['field_name', 'well_name', 'season', 'field_type', 'well_type', 'drilling_method', 'completion_type']:
       le = LabelEncoder()
       df_all[col] = le.fit_transform(df_all[col])
   
   # Drop timestamp (already converted to numeric features)
#    df_all = df_all.drop(columns=['timestamp'])
   
   # -----------------------------
   # 5. Save Excel
   # -----------------------------
   df_all.to_excel(synthetic_production_numeric, index=False)

def generate_history_data1():
    import pandas as pd
    import numpy as np

    # --- Settings ---
    n_days = 365   # 1 year of history
    n_wells = 3    # number of wells
    fields = ["Field_A", "Field_B"]
    wells = [f"Well_{i+1}" for i in range(n_wells)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    # --- Generate synthetic data ---
    data = []
    for field in fields:
        for well in wells:
            base_oil = np.random.randint(800, 1500)
            base_gas = np.random.randint(5000, 12000)
            base_water = np.random.randint(200, 500)

            for day, ts in enumerate(dates):
                # Production decline + random noise
                oil_rate = base_oil * np.exp(-0.0005 * day) + np.random.normal(0, 30)
                gas_rate = base_gas * np.exp(-0.0003 * day) + np.random.normal(0, 100)
                water_rate = base_water * (1 + 0.002 * day) + np.random.normal(0, 20)

                # Surface measurements
                pressure = np.random.uniform(1500, 3000)
                choke = np.random.uniform(20, 80)
                water_cut = min(100, (water_rate / (oil_rate + water_rate)) * 100)
                temperature = np.random.uniform(60, 120)

                # Operational events
                maintenance_event = np.random.choice([0, 1], p=[0.97, 0.03])
                shutdown_flag = 1 if np.random.rand() < 0.01 else 0

                # Engineered features
                gor = gas_rate / oil_rate if oil_rate > 0 else np.nan
                wor = water_rate / oil_rate if oil_rate > 0 else np.nan
                water_fraction = water_rate / (oil_rate + water_rate + 1e-6)

                # Time-based features
                month = ts.month
                day_of_week = ts.dayofweek
                is_weekend = 1 if day_of_week >= 5 else 0
                season = (
                    "Winter" if month in [12, 1, 2]
                    else "Spring" if month in [3, 4, 5]
                    else "Summer" if month in [6, 7, 8]
                    else "Autumn"
                )

                data.append([
                    field, well, ts,
                    pressure, choke, water_cut, temperature,
                    oil_rate, gas_rate, water_rate,
                    maintenance_event, shutdown_flag,
                    gor, wor, water_fraction,
                    month, day_of_week, is_weekend, season
                ])

    # --- Build DataFrame ---
    df = pd.DataFrame(data, columns=[
        "field_name", "well_name", "timestamp",
        "pressure (psi)", "choke (%)", "water_cut (%)", "temperature (°C)",
        "oil_rate_bopd (BOPD)", "gas_rate_mscf_day (MSCF/day)", "water_rate_bwpd (BWPD)",
        "maintenance_event", "shutdown_flag",
        "GOR", "WOR", "water_fraction",
        "month", "day_of_week", "is_weekend", "season"
    ])

    # --- Add rolling stats (last 7 days per well) ---
    for col in ["oil_rate_bopd (BOPD)", "gas_rate_mscf_day (MSCF/day)", "water_rate_bwpd (BWPD)"]:
        df[f"{col}_7d_avg"] = df.groupby("well_name")[col].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # --- Add target columns (forecast next-day values) ---
    df["target_oil_next_day"] = df.groupby("well_name")["oil_rate_bopd (BOPD)"].shift(-1)
    df["target_gas_next_day"] = df.groupby("well_name")["gas_rate_mscf_day (MSCF/day)"].shift(-1)
    df["target_water_next_day"] = df.groupby("well_name")["water_rate_bwpd (BWPD)"].shift(-1)

    # Drop last row per well (no target)
    df = df.dropna().reset_index(drop=True)

    # --- Save to Excel ---
    df.to_excel(forecast_production_numeric, index=False)

