
import os
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import metrics
import pickle

import joblib 

OIL_PRODUCTION_DATA_PATH = "./src/resource/oil_production_data.xlsx"
LSTM_MODEL_PATH = './src/created_model/lstm.h5'
SCALER_PATH = "./src/created_model/scaler.pkl"
TRAINING_HISTORY_PATH = "./src/created_model/training_history.pkl"
HISTORY_PATH = "./src/resource/lstm_model_history.xlsx"

LOOKBACK = 14

def load_excel_data(filepath):
    df = pd.read_excel(filepath)
    return df

def import_data():
    # df = pd.read_csv(OIL_PRODUCTION_DATA_PATH)
    df = pd.read_excel(OIL_PRODUCTION_DATA_PATH)

    well_codes = df.WELL_CODE.unique()
    df = df.pivot(index='DAY_OF_PROD', columns='WELL_CODE', values=['OIL_PROD_VOL', 'CUM_OIL_PROD'])
    # df.plot(y='OIL_PROD_VOL', title='Daily Oil Production')
    # df.plot(y='CUM_OIL_PROD', title='Cum. Oil Production')
    return  df, well_codes

def train_test_prod(df, well_codes):
    train_oil_prod = df['CUM_OIL_PROD', well_codes[0]].values
    test_oil_prod = df['CUM_OIL_PROD', well_codes[1]].values

    return  train_oil_prod, test_oil_prod


def prepare_lstm_input(prod_data, lookback, scaler=None):
    data = np.array(prod_data).reshape(-1, 1)
    if scaler == None:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    X = []
    y = []
    num_idxs = len(data)
    for i in range(lookback, num_idxs):
        i_start = i - lookback
        i_end = i - 1
        X.append([data[i] for i in range(i_start, i_end + 1)])
        y.append(data[i])
    num_y = len(y)
    X = np.array(X).reshape(num_y, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y, scaler

def build_model():
    df, well_codes =import_data()
    train_oil_prod, test_oil_prod = train_test_prod(df, well_codes)
    X_train, y_train, scaler = prepare_lstm_input(train_oil_prod, LOOKBACK)
    # X_test, y_test, _ = prepare_lstm_input(test_oil_prod, LOOKBACK, scaler)

    joblib.dump(scaler, SCALER_PATH)

    NUM_HIDDEN = 28
    VALID_FRAC = 0.1
    BATCH_SIZE = 32
    # Define a LSTM model
    model = Sequential()
    model.add(LSTM(NUM_HIDDEN, activation="relu", return_sequences=True, input_shape=(LOOKBACK, 1)))
    model.add(LSTM(NUM_HIDDEN, activation="relu", return_sequences=False))
    model.add(Dense(NUM_HIDDEN))
    model.add(Dense(1))
    model.compile(optimizer='Adadelta', loss="mse",metrics=[metrics.MeanSquaredError()])
    # Early-stopping callback
    earlystop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.0001,
        patience=50)
    # Save the best model
    ckpt_callback = ModelCheckpoint(filepath=LSTM_MODEL_PATH, mode='min', 
                                    monitor='val_loss', verbose=1, 
                                    save_best_only=True)
    # Train the model
    history = model.fit(X_train, y_train, epochs=500, batch_size=BATCH_SIZE, 
                        validation_split=VALID_FRAC, 
                        callbacks=[earlystop_callback, ckpt_callback], 
                        verbose=1)
    
    with open(TRAINING_HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)


