from __future__ import annotations
import argparse
import logging
import os
import shutil
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Menangani TensorFlow untuk menghindari error TensorBoard di CI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None

LOGGER = logging.getLogger(__name__)

# --- Konfigurasi Default ---
DEFAULT_MLFLOW_DB = "sqlite:///mlruns/mlflow.db"
DEFAULT_EXP_NAME = "Bitcoin Price Tuning"

@dataclass(frozen=True)
class PreparedTabularData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

# ... (Fungsi load_dataframe dan prepare_tabular_regression tetap sama) ...

def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Setup MLflow dengan pembuatan direktori otomatis."""
    if mlflow is None:
        LOGGER.error("MLflow tidak terinstal.")
        return

    # Pastikan folder database ada sebelum inisialisasi
    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    
    # Membersihkan environment variabel untuk CI
    for env_var in ["MLFLOW_RUN_ID", "MLFLOW_PARENT_RUN_ID"]:
        os.environ.pop(env_var, None)

    mlflow.set_experiment(experiment_name)
    LOGGER.info("MLflow Tracking URI: %s", tracking_uri)

def tune_random_forest(
    data: PreparedTabularData,
    *,
    n_estimators_grid: Sequence[int],
    max_depth_grid: Sequence[Optional[int]],
) -> None:
    """Melakukan Hyperparameter Tuning dan mencatat setiap kombinasi ke MLflow."""
    LOGGER.info("Memulai tuning Random Forest...")
    
    combinations = list(product(n_estimators_grid, max_depth_grid))
    
    for n_est, depth in combinations:
        run_name = f"RF_n{n_est}_d{depth if depth else 'None'}"
        
        # Menggunakan helper yang sudah ada
        train_and_log_random_forest(
            data,
            n_estimators=n_est,
            max_depth=depth,
            run_name=run_name
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="MLProject/preprocessed_bitcoin_data.csv")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--mlflow_db", type=str, default=DEFAULT_MLFLOW_DB)
    args = parser.parse_args()

    _configure_logging("INFO")
    
    # 1. Setup MLflow
    setup_mlflow(args.mlflow_db, DEFAULT_EXP_NAME)
    
    # 2. Load & Prepare Data
    df = load_dataframe(args.data_file)
    feature_cols = ["Open", "Low", "Close", "Volume"]
    target_col = "High"
    
    try:
        data = prepare_tabular_regression(
            df, 
            feature_cols=feature_cols, 
            target_col=target_col, 
            max_samples=args.max_samples,
            test_size=0.2,
            random_state=42
        )
        
        # 3. Baseline Linear Regression
        train_and_log_linear_regression(data, run_name="Baseline_Linear")
        
        # 4. Hyperparameter Tuning Random Forest
        tune_random_forest(
            data,
            n_estimators_grid=[50, 100],
            max_depth_grid=[10, 20, None]
        )
        
        print("Tuning selesai! Semua hasil tersimpan di MLflow.")
        
    except Exception as e:
        LOGGER.error("Terjadi kesalahan: %s", e)
        raise
if __name__ == "__main__":
    main()
