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

DEFAULT_DATA_FILE = "data/btcusd_1-min_data.csv"
DEFAULT_MAX_SAMPLES = 15_000
DEFAULT_LOOK_BACK = 60
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_SIZE = 0.2
MLFLOW_DB_PATH = "sqlite:///mlruns/mlflow.db"
EXPERIMENT_NAME = "Bitcoin Price Prediction"

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
TARGET = "High"


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_data(data_file: str, max_samples: int) -> pd.DataFrame:
    """Load and clean the raw CSV data."""
    df = pd.read_csv(data_file)

    # Normalise volume column name (dataset uses 'Volume_(BTC)' in some versions)
    if "Volume_(BTC)" in df.columns and "Volume" not in df.columns:
        df = df.rename(columns={"Volume_(BTC)": "Volume"})

    df = df[FEATURES].dropna()

    if len(df) > max_samples:
        df = df.head(max_samples)
        print(f"Dataset truncated to {max_samples} rows.")

    print(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
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
    with mlflow.start_run(run_name="LinearRegression", experiment_id=experiment_id, nested=True)):
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
    with mlflow.start_run(run_name="RandomForestRegressor", experiment_id=experiment_id, nested=True)):
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
    with mlflow.start_run(run_name="LSTM", experiment_id=experiment_id, nested=True)):
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
    # Ensure the mlruns directory exists so SQLite doesn't fail
    if not os.path.exists("mlruns"):
        os.makedirs("mlruns")
        
    mlflow.set_tracking_uri(db_path)
    
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
    parser.add_argument("--data_file", type=str, default=DEFAULT_DATA_FILE,
                        help="Path to the CSV data file.")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help="Maximum number of rows to use (RAM guard).")
    parser.add_argument("--look_back", type=int, default=DEFAULT_LOOK_BACK,
                        help="Sequence length for LSTM input.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Training epochs for LSTM.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for LSTM training.")
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE,
                        help="Fraction of data reserved for testing.")
    parser.add_argument("--mlflow_db", type=str, default=MLFLOW_DB_PATH,
                        help="MLflow backend store URI.")
    parser.add_argument("--experiment", type=str, default=EXPERIMENT_NAME,
                        help="MLflow experiment name.")
    parser.add_argument("--skip_lstm", action="store_true",
                        help="Skip LSTM training (faster, no TensorFlow needed).")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_id = setup_mlflow(args.mlflow_db, args.experiment)
    
    df = load_data(args.data_file, args.max_samples)
    
    # Run Scikit-learn models
    X_train, X_test, y_train, y_test = build_flat_splits(df, args.test_size)
    print(f"Flat splits — train: {X_train.shape}  test: {X_test.shape}")
    train_linear_regression(X_train, X_test, y_train, y_test, exp_id)
    train_random_forest(X_train, X_test, y_train, y_test, exp_id)
    
    # Run LSTM
    X_tr_s, X_te_s, y_tr_s, y_te_s, sc, h_idx = build_sequence_splits(df, args.look_back, args.test_size)
    train_lstm(X_tr_s, X_te_s, y_tr_s, y_te_s, sc, h_idx, args.look_back, args.epochs, args.batch_size, exp_id)

    # ── MLflow ──────────────────────────────────────────────────────────────
    os.makedirs("mlruns", exist_ok=True)
    exp_id = setup_mlflow(args.mlflow_db, args.experiment)

    # ── Data ────────────────────────────────────────────────────────────────
    df = load_data(args.data_file, args.max_samples)

    # Flat splits (sklearn models)
    X_train, X_test, y_train, y_test = build_flat_splits(df, args.test_size)
    print(f"Flat splits  — train: {X_train.shape}  test: {X_test.shape}")

    # ── Train sklearn models ─────────────────────────────────────────────────
    train_linear_regression(X_train, X_test, y_train, y_test, exp_id)
    train_random_forest(X_train, X_test, y_train, y_test, exp_id)

    # Free flat arrays before sequence work
    del X_train, X_test, y_train, y_test
    gc.collect()

    # ── Train LSTM ──────────────────────────────────────────────────────────
    if not args.skip_lstm:
        (X_train_seq, X_test_seq,
         y_train_seq, y_test_seq,
         scaler, high_idx) = build_sequence_splits(df, args.look_back, args.test_size)

        print(f"Sequence splits — train: {X_train_seq.shape}  test: {X_test_seq.shape}")

        train_lstm(
            X_train_seq, X_test_seq,
            y_train_seq, y_test_seq,
            scaler, high_idx,
            look_back=args.look_back,
            epochs=args.epochs,
            batch_size=args.batch_size,
            experiment_id=exp_id,
        )
    else:
        print("Skipping LSTM training (--skip_lstm flag set).")

    print("\nAll runs complete. Start the MLflow UI with:")
    print(f"  mlflow ui --backend-store-uri {args.mlflow_db}")


if __name__ == "__main__":
    main()
