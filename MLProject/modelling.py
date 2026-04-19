"""
Bitcoin Price Prediction with MLflow Tracking
==============================================
Trains Linear Regression, Random Forest, and LSTM models on Bitcoin historical data.
All runs are tracked with MLflow.
"""

import os
import gc
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_FILE = "bitcoinhistoricaldata_raw.csv"
DEFAULT_MAX_SAMPLES = 15_000
DEFAULT_LOOK_BACK = 60
DEFAULT_EPOCHS = 5  # Reduced for faster CI
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_SIZE = 0.2
MLFLOW_DB_PATH = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Bitcoin Price Prediction"

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
TARGET = "High"

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_data(data_file: str, max_samples: int) -> pd.DataFrame:
    """Load and clean the raw CSV data."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"CSV file not found at: {os.path.abspath(data_file)}")
        
    df = pd.read_csv(data_file)
    if "Volume_(BTC)" in df.columns and "Volume" not in df.columns:
        df = df.rename(columns={"Volume_(BTC)": "Volume"})

    df = df[FEATURES].dropna()
    if len(df) > max_samples:
        df = df.head(max_samples)
        print(f"Dataset truncated to {max_samples} rows.")

    print(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def build_flat_splits(df: pd.DataFrame, test_size: float):
    X = df[["Open", "Low", "Close", "Volume"]]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def build_sequence_splits(df: pd.DataFrame, look_back: int, test_size: float):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[FEATURES].values)
    high_idx = FEATURES.index(TARGET)
    
    X_seq, y_seq = [], []
    for i in range(len(scaled) - look_back):
        X_seq.append(scaled[i : i + look_back])
        y_seq.append(scaled[i + look_back, high_idx])
    
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    split = int(len(X_seq) * (1 - test_size))
    return X_seq[:split], X_seq[split:], y_seq[:split], y_seq[split:], scaler, high_idx

# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, X_test, y_train, y_test, experiment_id: str):
    mlflow.sklearn.autolog(log_models=True)
    # FIXED: Removed extra parenthesis here
    with mlflow.start_run(run_name="LinearRegression", experiment_id=experiment_id, nested=True):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mlflow.log_metrics({"test_rmse": rmse, "test_r2_score": r2})
        print(f"[LinearRegression]  RMSE={rmse:.6f}  R²={r2:.6f}")
    return model

def train_random_forest(X_train, X_test, y_train, y_test, experiment_id: str):
    mlflow.sklearn.autolog(log_models=True)
    # FIXED: Added nested=True
    with mlflow.start_run(run_name="RandomForestRegressor", experiment_id=experiment_id, nested=True):
        model = RandomForestRegressor(random_state=42, n_estimators=50) # Smaller for CI speed
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mlflow.log_metrics({"test_rmse_rf": rmse, "test_r2_score_rf": r2})
        print(f"[RandomForest]      RMSE={rmse:.6f}  R²={r2:.6f}")
    return model

def train_lstm(X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler, high_idx, look_back, epochs, batch_size, experiment_id):
    mlflow.tensorflow.autolog(log_models=True)
    # FIXED: Added nested=True
    with mlflow.start_run(run_name="LSTM", experiment_id=experiment_id, nested=True):
        n_features = X_train_seq.shape[2]
        model = Sequential([
            LSTM(50, activation="relu", input_shape=(look_back, n_features)),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred_scaled = model.predict(X_test_seq)
        dummy = np.zeros((len(y_pred_scaled), len(FEATURES)))
        dummy[:, high_idx] = y_pred_scaled.flatten()
        y_pred = scaler.inverse_transform(dummy)[:, high_idx]
        
        dummy_true = np.zeros((len(y_test_seq), len(FEATURES)))
        dummy_true[:, high_idx] = y_test_seq.flatten()
        y_true = scaler.inverse_transform(dummy_true)[:, high_idx]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mlflow.log_metrics({"test_rmse_lstm": rmse})
        print(f"[LSTM]              RMSE={rmse:.6f}")
    return model

# ---------------------------------------------------------------------------
# MLflow setup & CLI
# ---------------------------------------------------------------------------

def setup_mlflow(db_path: str, experiment_name: str) -> str:
    mlflow.set_tracking_uri(db_path)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI : {db_path}")
    print(f"Experiment          : {experiment_name}  (id={exp_id})")
    return exp_id

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default=DEFAULT_DATA_FILE)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--look_back", type=int, default=DEFAULT_LOOK_BACK)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--mlflow_db", type=str, default=MLFLOW_DB_PATH)
    parser.add_argument("--experiment", type=str, default=EXPERIMENT_NAME)
    return parser.parse_args()

def main():
    args = parse_args()
    exp_id = setup_mlflow(args.mlflow_db, args.experiment)
    df = load_data(args.data_file, args.max_samples)

    # Flat Models
    X_train, X_test, y_train, y_test = build_flat_splits(df, args.test_size)
    train_linear_regression(X_train, X_test, y_train, y_test, exp_id)
    train_random_forest(X_train, X_test, y_train, y_test, exp_id)

    # LSTM Model
    X_tr_s, X_te_s, y_tr_s, y_te_s, sc, h_idx = build_sequence_splits(df, args.look_back, args.test_size)
    train_lstm(X_tr_s, X_te_s, y_tr_s, y_te_s, sc, h_idx, args.look_back, args.epochs, args.batch_size, exp_id)

if __name__ == "__main__":
    main()
