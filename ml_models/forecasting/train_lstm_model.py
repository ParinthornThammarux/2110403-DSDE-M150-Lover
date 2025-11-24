"""
RandomForest Time-Series Forecasting Model for Urban Complaint Prediction
Forecasts daily complaint volumes using tree-based model instead of LSTM
(Compatible with Python 3.13 – no TensorFlow required)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

import matplotlib.pyplot as plt
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplaintForecaster:
    """
    Tree-based forecasting system for urban complaints

    - lookback_days: ใช้ข้อมูลกี่วันย้อนหลังเป็น feature
    - forecast_horizon: ทำนายกี่วันล่วงหน้า (multi-step forecast)
    """

    def __init__(self, lookback_days=30, forecast_horizon=7):
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model: RandomForestRegressor | None = None

    # ------------------------------------------------------------------
    # DATA PREPARATION
    # ------------------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame, target_col: str = "complaint_count"):
        """
        เตรียมข้อมูล time-series สำหรับการ forecast

        ขั้นตอน:
        1. แปลง timestamp เป็น DatetimeIndex
        2. resample เป็นรายวัน (count จำนวนเคสต่อวัน)
        3. เติมวันที่ขาดให้เป็น 0
        4. scale ด้วย MinMaxScaler
        """
        logger.info(
            f"Preparing data with lookback={self.lookback_days}, "
            f"horizon={self.forecast_horizon}"
        )

        # Ensure datetime index
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")

        # Use complaint_count if already aggregated, otherwise count rows
        if target_col in df.columns:
            daily_counts = (
                df[target_col]
                .resample("D")
                .sum()
                .to_frame(name=target_col)
            )
        else:
            daily_counts = df.resample("D").size().to_frame(name=target_col)

        # Fill missing dates with 0
        date_range = pd.date_range(
            start=daily_counts.index.min(),
            end=daily_counts.index.max(),
            freq="D",
        )
        daily_counts = daily_counts.reindex(date_range, fill_value=0)

        # Scale data
        values = daily_counts[target_col].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)

        logger.info(
            f"Prepared daily series from {daily_counts.index.min().date()} "
            f"to {daily_counts.index.max().date()} "
            f"({len(daily_counts)} days)"
        )

        return scaled, daily_counts

    def create_sequences(self, data: np.ndarray):
        """
        แปลง sequence univariate ให้เป็น supervised learning dataset

        input:  data shape (T, 1)  # T = number of days
        output: X shape (N, lookback_days, 1)
                y shape (N, forecast_horizon)
        """
        X, y = [], []
        T = len(data)

        for i in range(T - self.lookback_days - self.forecast_horizon + 1):
            X.append(data[i : i + self.lookback_days])  # window ย้อนหลัง
            y.append(
                data[
                    i
                    + self.lookback_days : i
                    + self.lookback_days
                    + self.forecast_horizon
                ].flatten()  # horizon ล่วงหน้า
            )

        X = np.array(X)  # (N, lookback, 1)
        y = np.array(y)  # (N, horizon)

        logger.info(f"Created sequences: X{X.shape}, y{y.shape}")
        return X, y

    # ------------------------------------------------------------------
    # MODEL TRAINING
    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        เทรน RandomForestRegressor แบบ multi-output

        - flatten window (lookback_days * features) ให้เป็น feature vector
        """
        logger.info("Training RandomForestRegressor model...")

        n_samples, lookback, n_features = X_train.shape
        X_train_flat = X_train.reshape(n_samples, lookback * n_features)

        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_train_flat, y_train)
        logger.info("RandomForest training completed.")

    # ------------------------------------------------------------------
    # PREDICTION / EVALUATION
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray):
        """
        ทำนาย sequence แล้ว inverse scale กลับเป็นจำนวน complaint จริง
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        n_samples, lookback, n_features = X.shape
        X_flat = X.reshape(n_samples, lookback * n_features)

        preds_scaled = self.model.predict(X_flat)  # (N, horizon)

        # inverse scale: flatten ทั้ง matrix แล้ว inverse, จากนั้น reshape กลับ
        preds_flat = preds_scaled.reshape(-1, 1)
        preds_inv = self.scaler.inverse_transform(preds_flat).reshape(
            preds_scaled.shape
        )

        return preds_inv

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        คำนวณ MAE, RMSE, MAPE บน scale จริง (ไม่ใช่ normalized)
        """
        logger.info("Evaluating model...")

        predictions = self.predict(X_test)

        # inverse scale y_test
        y_flat = y_test.reshape(-1, 1)
        y_inv = self.scaler.inverse_transform(y_flat).reshape(y_test.shape)

        mae = mean_absolute_error(y_inv, predictions)
        mse = mean_squared_error(y_inv, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_inv, predictions) * 100

        logger.info(f"MAE:  {mae:.2f}")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"MAPE: {mape:.2f}%")

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "predictions": predictions,
            "actuals": y_inv,
        }

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def plot_predictions(self, actuals, predictions):
        """
        วาดกราฟเปรียบเทียบ actual vs predicted
        - แสดงหลาย ๆ window ทับกัน (เหมือน version LSTM เดิม)
        """
        Path("ml_models/forecasting/outputs").mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(15, 6))
        n_samples = min(100, len(actuals))

        for i in range(n_samples):
            if i == 0:
                plt.plot(
                    range(self.forecast_horizon),
                    actuals[i],
                    "b-",
                    alpha=0.3,
                    label="Actual",
                )
                plt.plot(
                    range(self.forecast_horizon),
                    predictions[i],
                    "r--",
                    alpha=0.3,
                    label="Predicted",
                )
            else:
                plt.plot(
                    range(self.forecast_horizon),
                    actuals[i],
                    "b-",
                    alpha=0.1,
                )
                plt.plot(
                    range(self.forecast_horizon),
                    predictions[i],
                    "r--",
                    alpha=0.1,
                )

        plt.title(
            f"Complaint Volume Forecasts ({self.forecast_horizon}-day ahead)"
        )
        plt.xlabel("Days Ahead")
        plt.ylabel("Number of Complaints")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = "ml_models/forecasting/outputs/predictions_rf.png"
        plt.savefig(out_path, dpi=300)
        logger.info(f"Predictions plot saved to {out_path}")

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save_model(self, path="ml_models/forecasting/models/rf_forecaster.pkl"):
        """เซฟ model + scaler ด้วย joblib"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler},
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_model(self, path="ml_models/forecasting/models/rf_forecaster.pkl"):
        """โหลด model + scaler"""
        obj = joblib.load(path)
        self.model = obj["model"]
        self.scaler = obj["scaler"]
        logger.info(f"Model loaded from {path}")


# ----------------------------------------------------------------------
# MAIN: train forecasting model using actual CSV file
# ----------------------------------------------------------------------
def main():
    logger.info("=" * 80)
    logger.info("RandomForest Complaint Forecasting Model Training")
    logger.info("=" * 80)

    # สร้างโฟลเดอร์ที่ต้องใช้
    Path("ml_models/forecasting/models").mkdir(parents=True, exist_ok=True)
    Path("ml_models/forecasting/outputs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # READ CSV (ไม่ clean เพิ่มใน main – assume clean แล้ว)
    # ------------------------------------------------------------------
    csv_path = "clean_data.csv"  # เปลี่ยนชื่อให้ตรงไฟล์ที่คุณใช้
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # PREPARE DATA
    # ------------------------------------------------------------------
    forecaster = ComplaintForecaster(lookback_days=30, forecast_horizon=7)
    scaled_data, daily_counts = forecaster.prepare_data(df)

    X, y = forecaster.create_sequences(scaled_data)

    # train / val / test split (80 / 10 / 10)
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))  # ยังไม่ได้ใช้ val ตรง ๆ แต่กันไว้ได้

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]  # เผื่อเอาไป extend ทีหลัง

    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    logger.info(
        f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    forecaster.train(X_train, y_train)

    # ------------------------------------------------------------------
    # EVALUATE & PLOT
    # ------------------------------------------------------------------
    results = forecaster.evaluate(X_test, y_test)
    forecaster.plot_predictions(results["actuals"], results["predictions"])

    # ------------------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------------------
    forecaster.save_model()

    logger.info("=" * 80)
    logger.info("Forecasting pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
