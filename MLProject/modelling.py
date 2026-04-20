"""
Bitcoin Price Prediction with MLflow Tracking
==============================================
Trains Linear Regression, Random Forest, and LSTM models on Bitcoin historical data.
All runs are tracked with MLflow.
"""

import os
import gc
import argparse
from typing import Sequence

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from modelling_tuning import (
    load_dataframe,
    ohlcv_sequence_columns,
    prepare_sequence_data,
    prepare_tabular_regression,
)

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

TABULAR_FEATURES = ["Open", "Low", "Close", "Volume"]
TARGET = "High"


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
    inverse_columns: Sequence[str],
    look_back: int, epochs: int, batch_size: int,
    experiment_id: str,
):
    mlflow.tensorflow.autolog(log_models=True)
    with mlflow.start_run(run_name="LSTM", experiment_id=experiment_id):
        n_features = X_train_seq.shape[2]
        n_cols = len(inverse_columns)

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
            dummy = np.zeros((len(arr_1d), n_cols))
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

    # ── MLflow ──────────────────────────────────────────────────────────────
    os.makedirs("mlruns", exist_ok=True)
    exp_id = setup_mlflow(args.mlflow_db, args.experiment)

    # ── Data (same preprocessing as modelling_tuning.py) ────────────────────
    df = load_dataframe(args.data_file)

    tabular = prepare_tabular_regression(
        df,
        feature_cols=TABULAR_FEATURES,
        target_col=TARGET,
        max_samples=args.max_samples,
        test_size=args.test_size,
        random_state=42,
    )
    X_train, X_test = tabular.X_train, tabular.X_test
    y_train, y_test = tabular.y_train, tabular.y_test
    print(f"Flat splits  — train: {X_train.shape}  test: {X_test.shape}")

    # ── Train sklearn models ─────────────────────────────────────────────────
    train_linear_regression(X_train, X_test, y_train, y_test, exp_id)
    train_random_forest(X_train, X_test, y_train, y_test, exp_id)

    # Free flat arrays before sequence work
    del X_train, X_test, y_train, y_test
    gc.collect()

    # ── Train LSTM ──────────────────────────────────────────────────────────
    if not args.skip_lstm:
        seq_cols = ohlcv_sequence_columns(df)
        if len(seq_cols) >= 2 and TARGET in seq_cols:
            seq_data = prepare_sequence_data(
                df,
                features=seq_cols,
                target_feature=TARGET,
                look_back=args.look_back,
                max_samples=args.max_samples,
                train_fraction=1.0 - args.test_size,
            )
            high_idx = list(seq_data.features).index(TARGET)
            inv_cols = list(seq_data.features)

            print(f"Sequence splits — train: {seq_data.X_train_seq.shape}  test: {seq_data.X_test_seq.shape}")

            train_lstm(
                seq_data.X_train_seq, seq_data.X_test_seq,
                seq_data.y_train_seq, seq_data.y_test_seq,
                seq_data.scaler, high_idx,
                inverse_columns=inv_cols,
                look_back=args.look_back,
                epochs=args.epochs,
                batch_size=args.batch_size,
                experiment_id=exp_id,
            )
        else:
            print(
                "Skipping LSTM: need OHLCV columns for sequences; "
                f"got {seq_cols}. Align CSV with modelling_tuning preprocessing."
            )
    else:
        print("Skipping LSTM training (--skip_lstm flag set).")

    print("\nAll runs complete. Start the MLflow UI with:")
    print(f"  mlflow ui --backend-store-uri {args.mlflow_db}")


if __name__ == "__main__":
    main()
