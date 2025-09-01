import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import load_model
import pickle

from src.functions.gen_lstm_model import import_data, train_test_prod, prepare_lstm_input, load_excel_data

LSTM_MODEL_PATH = './src/created_model/lstm.h5'
LOOKBACK = 14
X_TRAIN_EXCEL = "./src/resource/x_train_data.xlsx"
X_TEST_EXCEL = "./src/resource/x_test_data.xlsx"
TRAINING_HISTORY_PATH = "./src/created_model/training_history.pkl"
ERROR_PATH = "./src/created_model/actual_predicted_error.pkl"

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def predict_production():
    df, well_codes = import_data()
    train_oil_prod, test_oil_prod = train_test_prod(df, well_codes)

    X_train, y_train, scaler = prepare_lstm_input(train_oil_prod, LOOKBACK)
    X_test, y_test, _ = prepare_lstm_input(test_oil_prod, LOOKBACK, scaler)

    # Load the trained LSTM model
    model = load_model(LSTM_MODEL_PATH, custom_objects={"mse": MeanSquaredError()})

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Inverse transform both actual and predicted
    y_test_inv = scaler.inverse_transform(np.reshape(y_test, (-1, 1)))
    y_pred_inv = scaler.inverse_transform(y_pred)

    # out = pd.concat([
    #     pd.DataFrame(y_test_inv, columns=["actual"]),
    #     pd.DataFrame(y_pred_inv, columns=["predicted"]),
    #     pd.DataFrame(y_test_inv - y_pred_inv, columns=["error"]),
    # ], axis=1)
    # out.to_excel(X_TEST_EXCEL, index=False, engine="openpyxl")

    errors = evaluate_predictions(y_test_inv, y_pred_inv)

    shap_values = explain_with_shap_deep(
        model,
        X_train, X_test,
        lookback=LOOKBACK,
        feature_names=["CUM_OIL_PROD", "DAY_OF_PROD", "WELL_CODE"],   # or list of all features
        nsamples=10,
        background_size=30
    )
    return errors, shap_values

def evaluate_predictions(y_true, y_pred, save_path=None, plot=True):
    
    # Flatten arrays (in case they are 2D from model)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Safe MAPE
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE (%)": float(mape),
        "R²": float(r2)
    }

    return metrics

def explain_with_shap_deep(model, X_train, X_test, lookback, feature_names, nsamples=10, background_size=50):

    # --- Pick background for DeepExplainer ---
    background = X_train[np.random.choice(X_train.shape[0], background_size, replace=False)]

    # --- Initialize DeepExplainer ---
    explainer = shap.DeepExplainer(model, background)

    # --- Select test samples ---
    test_samples = X_test[:nsamples]

    # --- Compute SHAP values ---
    shap_values = explainer.shap_values(test_samples)

    # If single output → shap_values is list with one element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # --- Flatten time and feature dimensions ---
    shap_values_flat = shap_values.reshape(nsamples, lookback * len(feature_names))
    X_test_flat = test_samples.reshape(nsamples, lookback * len(feature_names))

    # Expand feature names by timestep: CUM_OIL_PROD_t-0, CUM_OIL_PROD_t-1, ...
    feature_names_expanded = [f"{f}_t-{t}" for t in range(lookback) for f in feature_names]

    # --- Convert SHAP values to DataFrame ---
    shap_df = pd.DataFrame(shap_values_flat, columns=feature_names_expanded)
    shap_df["Sample"] = np.arange(len(shap_df))

    # Save SHAP values to Excel
    shap_df.to_excel(os.path.join(PLOT_DIR, "shap_values.xlsx"), index=False)

    # --- SHAP Summary Plot ---
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values_flat, X_test_flat, feature_names=feature_names_expanded, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "shap_summary.png"))
    plt.close()

    # --- SHAP Bar Plot (Feature Importance) ---
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values_flat, X_test_flat, feature_names=feature_names_expanded, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "shap_bar.png"))
    plt.close()

    return shap_values, shap_df

def plot_history():
    with open(TRAINING_HISTORY_PATH, "rb") as f:
        hist = pickle.load(f)

    plot_data(
        hist['loss'],
        hist['val_loss'],
        'Learning Curves',
        'Epoch',
        'MSE',
        'Train Loss',
        'Validation',
        '',
        'history'
    )

def plot_test():
    df = load_excel_data(X_TEST_EXCEL)

    plot_scatter(
        df['actual'],
        df['predicted'],
        'Actual vs Predicted',
        'Test Data',
        'Predicted',
        'test_data_scatter',
    )

    plot_data(
        df['actual'],
        df['predicted'],
        'Cumulative Oil Prod. (Norm.)',
        '',
        '',
        'Test Data',
        'Prediction',
        'best',
        'test_data_curve'
    )

def plot_train():
    df = load_excel_data(X_TRAIN_EXCEL)

    plot_scatter(
        df['actual'],
        df['predicted'],
        f'{df.columns[0]} vs {df.columns[1]}',
        'Test Data',
        'Predicted',
        'train_data_scatter',
    )

    plot_data(
        df['actual'],
        df['predicted'],
        'Cumulative Oil Prod. (Norm.)',
        '',
        '',
        'Train Data',
        'Prediction',
        'best',
        'train_data_curve'
    )

def plot_data(x_data, y_data, title, xlabel, ylabel, series1_label, series2_label, legend, filename):
   # Plot learning curves
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x_data, label=series1_label)
    plt.plot(y_data, label=series2_label)
    plt.legend(loc=legend)
    fname = os.path.join(PLOT_DIR, f"{filename}.png")
    plt.savefig(fname)
    plt.close()
    plt.show()

def plot_scatter(x_data, y_data, title, xlabel, ylabel, filename):
   # Plot learning curves
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x_data, y_data)
    fname = os.path.join(PLOT_DIR, f"{filename}.png")
    plt.savefig(fname)
    plt.close()
    plt.show()
