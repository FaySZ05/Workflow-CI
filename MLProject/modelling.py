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
MLFLOW_DB_PATH = "sqlite:///mlruns/mlflow.db"
EXPERIMENT_NAME = "Bitcoin Price Prediction"
DEFAULT_DATA_FILE = "MLProject/preprocessed_bitcoin_data.csv"
DEFAULT_MAX_SAMPLES = 15_000
DEFAULT_LOOK_BACK = 60
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_SIZE = 0.2
EXPERIMENT_NAME = "Bitcoin Price Prediction"

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
TARGET = "High"
db_dir = "mlruns"
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def _is_lfs_pointer_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(200)
        return b"git-lfs.github.com/spec/v1" in head
    except OSError:
        return False


def load_data(data_file: str, max_samples: int) -> pd.DataFrame:
    """
    Load the dataset from CSV.

    In CI, the repository may be checked out without Git LFS objects (e.g. due to LFS quota limits).
    In that case, `data_file` may exist but be a Git LFS pointer file; we fall back to synthetic data
    so the pipeline can still run.
    """
    if (not data_file) or (not os.path.exists(data_file)) or _is_lfs_pointer_file(data_file):
        print(
            f"Warning: data file '{data_file}' missing or unavailable (LFS pointer). "
            "Falling back to synthetic data for CI."
        )
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5000, 4))
        y = X @ np.array([0.4, -0.2, 0.1, 0.05]) + rng.normal(scale=0.1, size=(5000,))
        df = pd.DataFrame(X, columns=["Open", "Low", "Close", "Volume"])
        df["High"] = y
        return df.head(max_samples) if max_samples else df

    df = pd.read_csv(data_file)
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples).copy()
    return df


def build_flat_splits(df: pd.DataFrame, test_size: float):
    """Build flat (non-sequential) train/test splits for sklearn models."""
    X = df[["Open", "Low", "Close", "Volume"]]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def create_sequences(scaled: np.ndarray, look_back: int, high_idx: int):
    """Convert a 2-D scaled array into (X_seq, y_seq) for sequence models."""
    X_seq, y_seq = [], []
    for i in range(len(scaled) - look_back):
        X_seq.append(scaled[i : i + look_back])
        y_seq.append(scaled[i + look_back, high_idx])
    return np.array(X_seq), np.array(y_seq)


def build_sequence_splits(df: pd.DataFrame, look_back: int, test_size: float):
    """Scale data and build sequential train/test splits for LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[FEATURES].values)

    high_idx = FEATURES.index(TARGET)
    X_seq, y_seq = create_sequences(scaled, look_back, high_idx)

    split = int(len(X_seq) * (1 - test_size))
    return (
        X_seq[:split], X_seq[split:],
        y_seq[:split], y_seq[split:],
        scaler, high_idx,
    )


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, X_test, y_train, y_test, experiment_id: str):
    mlflow.sklearn.autolog(log_models=True)
    with mlflow.start_run(run_name="LinearRegression", experiment_id=experiment_id):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metrics({"test_rmse": rmse, "test_r2_score": r2})
        mlflow.log_param("model_type", "LinearRegression")

        print(f"[LinearRegression]  RMSE={rmse:.6f}  R²={r2:.6f}")
    return model


def train_random_forest(X_train, X_test, y_train, y_test, experiment_id: str):
    mlflow.sklearn.autolog(log_models=True)
    with mlflow.start_run(run_name="RandomForestRegressor", experiment_id=experiment_id):
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metrics({"test_rmse_rf": rmse, "test_r2_score_rf": r2})
        mlflow.log_param("model_type", "RandomForestRegressor")

        print(f"[RandomForest]      RMSE={rmse:.6f}  R²={r2:.6f}")
    return model


def train_lstm(
    X_train_seq, X_test_seq,
    y_train_seq, y_test_seq,
    scaler, high_idx: int,
    look_back: int, epochs: int, batch_size: int,
    experiment_id: str,
):
    mlflow.tensorflow.autolog(log_models=True)
    with mlflow.start_run(run_name="LSTM", experiment_id=experiment_id):
        n_features = X_train_seq.shape[2]

        model = Sequential([
            LSTM(50, activation="relu", input_shape=(look_back, n_features)),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")

        model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
        )

        # Predict and inverse-transform
        y_pred_scaled = model.predict(X_test_seq)

        def inverse_single_col(arr_1d):
            dummy = np.zeros((len(arr_1d), len(FEATURES)))
            dummy[:, high_idx] = arr_1d
            return scaler.inverse_transform(dummy)[:, high_idx]

        y_pred = inverse_single_col(y_pred_scaled.flatten())
        y_true = inverse_single_col(y_test_seq.flatten())

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        mlflow.log_metrics({"test_rmse_lstm": rmse, "test_r2_score_lstm": r2})
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("look_back_window", look_back)

        print(f"[LSTM]              RMSE={rmse:.6f}  R²={r2:.6f}")
    return model


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def setup_mlflow(db_path: str, experiment_name: str) -> str:
    if db_path.startswith("sqlite:///"):
        db_dir = db_path.replace("sqlite:///", "").rsplit("/", 1)[0]
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    # Setup Tracking URI
    mlflow.set_tracking_uri(db_path)
    
    # Cleaning Environment Variables to Prevent Conflict with Previous Runs
    for _env in ("MLFLOW_RUN_ID", "MLFLOW_PARENT_RUN_ID"):
        os.environ.pop(_env, None)

    # So TensorFlow Can Recognize TensorBoard 
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

    # 5. Inisialisasi Eksperimen
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return exp_id


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Bitcoin price prediction models.")
    parser.add_argument("--data_file", type=str, default=DEFAULT_DATA_FILE)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 1. Setup
    eid = setup_mlflow(MLFLOW_DB_PATH, EXPERIMENT_NAME)
    df = load_data(args.data_file, args.max_samples)
    
    # 2. Sklearn Models (Linear & Random Forest)
    X_train, X_test, y_train, y_test = build_flat_splits(df, DEFAULT_TEST_SIZE)
    train_linear_regression(X_train, X_test, y_train, y_test, eid)
    train_random_forest(X_train, X_test, y_train, y_test, eid)
    
    # 3. Deep Learning Model (LSTM)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler, h_idx = build_sequence_splits(
        df, DEFAULT_LOOK_BACK, DEFAULT_TEST_SIZE
    )
    train_lstm(X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler, h_idx, 
               DEFAULT_LOOK_BACK, args.epochs, DEFAULT_BATCH_SIZE, eid)
    
    print("All models trained and tracked successfully!")
