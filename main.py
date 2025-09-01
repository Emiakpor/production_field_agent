from fastapi import FastAPI
from src.models.models import ForecastInput
from src.functions.train_predict_model import build_xbg_model_predict, build_lstm_model_predict
from src.functions.predict_agent import xgb_predict, lstm_predict, error_predict, get_monthly_average_data
from src.functions.predict_plot import plot_predict 
from src.functions.gen_data import generate_template_data, generate_history_data
from src.functions.train_forecast_xgboost_model import build_xgboost_model_forecast
from src.functions.train_forecast_lstm_model import build_lstm_model_forecast
from src.functions.forecast import xgb_forecast, lstm_forecast, lstm_explanation, lstm_forecast_plot, plot_forecast, error_forecast
from src.functions.gen_lstm_model import build_model
from src.functions.predict_with_lstm import predict_production, plot_history, plot_scatter, plot_test, plot_train

app = FastAPI(title="Oilfield Forecasting Predictor", version="1.0")
@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/gen_model_forecast")
def gen_model():    
    build_xgboost_model_forecast()
    build_lstm_model_forecast()
    return {f"message: created forecast model"}

@app.get("/gen_model_predict")
def gen_model():    
    build_xbg_model_predict()
    build_lstm_model_predict()
    return {f"message: created predict model"}

@app.get("/gen_lstm_model")
def gen_model1():    
    build_model()
    return {f"message: created predict model"}


@app.get("/xgb_predict")
def forecast():

    data = xgb_predict()

    return {
        "explanation": data["explanation"],
        "shap_local_b64": data["shap_local_b64"],
        # "shap_global_b64": data["shap_global_b64"],
        # "time_series_plot": data["plot_path"]  # file path to saved image (if generated) 
    }

@app.get("/lstm_predict")
def forecast():

    data = lstm_predict()

    return {
        "forecast_df": data["forecast_df"],
        "preds_df": data["preds_df"]
    }

@app.get("/predict_production")
def predict_data():
    errors = predict_production()

    return errors

@app.get("/xgb_forecast")
def forecast():
    xgb_forecast()   

@app.get("/lstm_forecast")
def forecast():
    lstm_forecast()    

@app.get("/gen_data")
def forecast():
    # generate_template_data()
    generate_history_data()

@app.get("/plot_predict")
def forecast():
    plot_predict()

    return {"Plot created"}

@app.get("/plot_forecast")
def forecast():
    # plot()
    #xgb_plot()
    # plot_data()
    plot_forecast()
    #lstm_forecast_plot()
    #lstm_explanation()

@app.get("/lstm_forecast_plot")
def forecast():
    # plot_forecast()
    #plot_history()
    plot_test()
    # plot_train()


@app.get("/error_predict")
def forecast():
    result = error_predict()

    return { "result": result}


@app.get("/error_forecast")
def forecast():
    result = error_forecast()

    return { "result": result}


@app.get("/get_monthly_average_data")
def get_monthly():
    result = get_monthly_average_data()

    return {"Monthly data created"}

