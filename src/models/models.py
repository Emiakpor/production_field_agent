# models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class Static(BaseModel):
    depth_md: float
    depth_tvd: float
    reservoir_pressure_initial: float
    bubble_point_pressure: float
    permeability_md: float
    porosity_percent: float
    net_pay_thickness_ft: float
    completion_type: str
    lift_type: str
    region: str

class Controls(BaseModel):
    choke_size_percent: float = Field(..., ge=0, le=100)
    gas_lift_rate_mscf_day: float
    pump_frequency_hz: float
    pump_intake_pressure_psi: float
    pump_discharge_pressure_psi: float
    compressor_suction_psi: float
    compressor_discharge_psi: float
    chemical_injection_rate_lph: Optional[float]

class Sensors(BaseModel):
    tubing_head_pressure_psi: float
    casing_head_pressure_psi: float
    bottomhole_pressure_psi: Optional[float]
    wellhead_temperature_c: float
    separator_inlet_pressure_psi: float
    separator_inlet_temperature_c: float
    annulus_pressure_psi: float
    dp_across_choke_psi: float
    esp_vibration_mm_s: Optional[float]
    esp_current_amp: Optional[float]

class Production(BaseModel):
    oil_rate_bopd: float
    gas_rate_mscf_day: float
    water_rate_bwpd: float
    water_cut_percent: float = Field(..., ge=0, le=100)
    gor_scf_bbl: float
    liquid_rate_bpd: float

class Derived(BaseModel):
    drawdown_psi: float
    productivity_index_bbl_day_psi: float
    pump_efficiency_bbl_per_kw: Optional[float]
    normalized_oil_rate: Optional[float]
    reynolds_number: Optional[float]

class Environment(BaseModel):
    ambient_temp_c: float
    ambient_pressure_psi: float
    humidity_percent: float = Field(..., ge=0, le=100)
    grid_voltage_stability_percent: float
    maintenance_event: bool
    shutdown_flag: bool

class Time(BaseModel):
    timestamp: datetime
    day_of_week: str
    day_of_year: int
    shift: str

class ForecastInput(BaseModel):
    well_id: str
    static: Static
    controls: Controls
    sensors: Sensors
    production: Production
    derived: Derived
    environment: Environment
    time: Time

class ForecastRequest(BaseModel):
    mode: str  # "scenario" or "lstm"
    data: Optional[List[Dict]] = None     # for scenario-based
    horizon: Optional[int] = 7            # forecast length for LSTM
    feature_order: List[str]
    target_columns: List[str]

class ErrorModel(BaseModel):
    mae: float 
    rmse: float
    r2: float  
    mape: float
    mse: float

# return { "mae": result["mae"], "rmse": result["rmse"],
    #         "r2": result["r2"], "mape": result["mape"], "mse":  result["mse"] }

