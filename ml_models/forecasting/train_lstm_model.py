"""
RandomForest Time-Series Forecasting Model for Urban Complaint Prediction
(Refactored Version with:
 - Time features & rolling features
 - GridSearchCV + TimeSeriesSplit
 - Single multi-output RandomForest model
 - Baselines (naive & moving average)
 - Per-horizon metrics & plots
)

Compatible with Python 3.13, no TensorFlow required.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import matplotlib.pyplot as plt
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# HELPER: METRICS
# ----------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, per_horizon: bool = True):
    """
    คำนวณ MAE / RMSE / MAPE รวมทุก horizon
    และ (option) แยก per horizon
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }

    if per_horizon:
        H = y_true.shape[1]
        mae_h, rmse_h, mape_h = [], [], []
        for h in range(H):
            mae_h.append(mean_absolute_error(y_true[:, h], y_pred[:, h]))
            mse_h = mean_squared_error(y_true[:, h], y_pred[:, h])
            rmse_h.append(np.sqrt(mse_h))
            mape_h.append(
                mean_absolute_percentage_error(
                    y_true[:, h], y_pred[:, h]
                )
                * 100
            )

        metrics["per_horizon"] = {
            "mae": np.array(mae_h),
            "rmse": np.array(rmse_h),
            "mape": np.array(mape_h),
        }

    return metrics


# ----------------------------------------------------------------------
# CLASS: COMPLAINT FORECASTER
# ----------------------------------------------------------------------
class ComplaintForecaster:
    """
    Tree-based forecasting system for urban complaints

    - lookback_days: ใช้ข้อมูลกี่วันย้อนหลังเป็น feature
    - forecast_horizon: ทำนายกี่วันล่วงหน้า (multi-step forecast)
    - ใช้ RandomForest ตัวเดียวแบบ multi-output (output = horizon vector)
    """

    def __init__(self, lookback_days: int = 30, forecast_horizon: int = 7):
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon
        self.model: RandomForestRegressor | None = None

    # ------------------------------------------------------------------
    # DATA PREPARATION
    # ------------------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame, target_col: str = "complaint_count"):
        """
        เตรียมข้อมูล time-series สำหรับการ forecast

        ขั้นตอน:
        1. แปลง timestamp เป็น DatetimeIndex
        2. resample เป็นรายวัน (count จำนวนเคสต่อวัน ถ้าไม่มี complaint_count)
        3. เติมวันที่ขาดให้เป็น 0
        4. เพิ่ม time features + rolling features
        5. คืนค่า:
            - values: numpy array (T, n_features)
            - daily_counts: DataFrame (index = วันที่)
        """
        logger.info(
            f"Preparing data with lookback={self.lookback_days}, "
            f"horizon={self.forecast_horizon}"
        )

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")

        # ถ้ามี column complaint_count อยู่แล้ว ใช้ sum, ไม่งั้น count rows
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

        # ---- เพิ่ม time features ----
        daily_counts["dayofweek"] = daily_counts.index.dayofweek
        daily_counts["month"] = daily_counts.index.month
        daily_counts["is_weekend"] = daily_counts["dayofweek"].isin([5, 6]).astype(int)

        # ---- rolling features ----
        daily_counts["roll_mean_7"] = (
            daily_counts[target_col].rolling(window=7, min_periods=1).mean()
        )
        daily_counts["roll_std_7"] = (
            daily_counts[target_col].rolling(window=7, min_periods=1).std().fillna(0)
        )
        daily_counts["roll_mean_30"] = (
            daily_counts[target_col].rolling(window=30, min_periods=1).mean()
        )

        feature_cols = [
            target_col,       # index 0 = target
            "dayofweek",
            "month",
            "is_weekend",
            "roll_mean_7",
            "roll_std_7",
            "roll_mean_30",
        ]

        values = daily_counts[feature_cols].values  # shape (T, n_features)

        logger.info(
            f"Prepared daily series from {daily_counts.index.min().date()} "
            f"to {daily_counts.index.max().date()} "
            f"({len(daily_counts)} days, {len(feature_cols)} features)"
        )

        return values, daily_counts

    def create_sequences(self, data: np.ndarray):
        """
        แปลง sequence multivariate ให้เป็น supervised learning dataset

        Parameters
        ----------
        data: shape (T, n_features)
              โดย feature index 0 = target (complaint_count)

        Returns
        -------
        X: shape (N, lookback_days, n_features)
        y: shape (N, forecast_horizon)   # target ล้วน ๆ
        """
        X, y = [], []
        T, n_features = data.shape

        for i in range(T - self.lookback_days - self.forecast_horizon + 1):
            # window ย้อนหลัง
            X_window = data[i : i + self.lookback_days, :]
            # horizon ล่วงหน้า: เฉพาะ target (column 0)
            y_window = data[
                i + self.lookback_days : i + self.lookback_days + self.forecast_horizon,
                0,
            ]
            X.append(X_window)
            y.append(y_window)

        X = np.array(X)  # (N, lookback_days, n_features)
        y = np.array(y)  # (N, forecast_horizon)

        logger.info(f"Created sequences: X{X.shape}, y{y.shape}")
        return X, y

    # ------------------------------------------------------------------
    # MODEL TRAINING
    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        เทรน RandomForestRegressor แบบ multi-output

        - flatten window (lookback_days * n_features) ให้เป็น feature vector
        - ใช้ GridSearchCV + TimeSeriesSplit (optimize จาก horizon แรก)
        """
        logger.info("Training RandomForestRegressor (multi-output) with GridSearch...")

        n_samples, lookback, n_features = X_train.shape
        X_train_flat = X_train.reshape(n_samples, lookback * n_features)

        tscv = TimeSeriesSplit(n_splits=3)

        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            verbose=1,
            n_jobs=-1,
        )

        # ใช้ horizon แรกเป็น target ตอนหา best params
        grid.fit(X_train_flat, y_train[:, 0])

        logger.info(f"Best params from GridSearch: {grid.best_params_}")

        self.model = RandomForestRegressor(
            **grid.best_params_,
            random_state=42,
            n_jobs=-1,
        )

        # fit ใหม่ด้วย multi-output y_train (ทุก horizon)
        self.model.fit(X_train_flat, y_train)
        logger.info("RandomForest training completed with best parameters.")

    # ------------------------------------------------------------------
    # PREDICTION / EVALUATION
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray):
        """
        ทำนาย sequence (multi-step)
        Output อยู่ใน scale จริงของ complaint_count
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        n_samples, lookback, n_features = X.shape
        X_flat = X.reshape(n_samples, lookback * n_features)

        preds = self.model.predict(X_flat)  # (N, forecast_horizon)

        return preds

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        คำนวณ MAE, RMSE, MAPE (รวม + per horizon)
        """
        logger.info("Evaluating model...")
        predictions = self.predict(X_test)

        metrics = compute_metrics(y_test, predictions, per_horizon=True)

        logger.info(f"MAE  (overall): {metrics['mae']:.2f}")
        logger.info(f"RMSE (overall): {metrics['rmse']:.2f}")
        logger.info(f"MAPE (overall): {metrics['mape']:.2f}%")

        logger.info("Per-horizon MAE: " + ", ".join(
            [f"h+{i+1}={v:.2f}" for i, v in enumerate(metrics["per_horizon"]["mae"])]
        ))

        return {
            "metrics": metrics,
            "predictions": predictions,
            "actuals": y_test,
        }

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def plot_predictions(self, actuals: np.ndarray, predictions: np.ndarray):
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
                    alpha=0.4,
                    label="Actual",
                )
                plt.plot(
                    range(self.forecast_horizon),
                    predictions[i],
                    "r--",
                    alpha=0.4,
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
        plt.close()
        logger.info(f"Predictions plot saved to {out_path}")

    def plot_horizon_mae(
        self,
        model_metrics: dict,
        naive_metrics: dict,
        ma7_metrics: dict,
    ):
        """
        วาด bar chart ของ MAE ต่อ horizon เปรียบเทียบ:
        - RandomForest
        - Naive baseline
        - 7-day moving average baseline
        """
        Path("ml_models/forecasting/outputs").mkdir(parents=True, exist_ok=True)

        mae_model = model_metrics["per_horizon"]["mae"]
        mae_naive = naive_metrics["per_horizon"]["mae"]
        mae_ma7 = ma7_metrics["per_horizon"]["mae"]

        H = self.forecast_horizon
        x = np.arange(H)

        width = 0.25

        plt.figure(figsize=(12, 6))
        plt.bar(x - width, mae_model, width, label="RandomForest")
        plt.bar(x, mae_naive, width, label="Naive (last value)")
        plt.bar(x + width, mae_ma7, width, label="7-day MA")

        plt.xticks(x, [f"h+{i+1}" for i in range(H)])
        plt.ylabel("MAE")
        plt.title("Per-horizon MAE Comparison")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out_path = "ml_models/forecasting/outputs/horizon_mae_comparison.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Horizon MAE comparison plot saved to {out_path}")

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save_model(self, path="ml_models/forecasting/models/rf_forecaster.pkl"):
        """เซฟ model + config ด้วย joblib"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "lookback_days": self.lookback_days,
                "forecast_horizon": self.forecast_horizon,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_model(self, path="ml_models/forecasting/models/rf_forecaster.pkl"):
        """โหลด model + config"""
        obj = joblib.load(path)
        self.model = obj["model"]
        self.lookback_days = obj["lookback_days"]
        self.forecast_horizon = obj["forecast_horizon"]
        logger.info(f"Model loaded from {path}")


# ----------------------------------------------------------------------
# BASELINES
# ----------------------------------------------------------------------
def compute_baseline_predictions(
    X: np.ndarray,
    forecast_horizon: int,
):
    """
    สร้าง baseline prediction 2 แบบ:
    - naive: ใช้ค่าล่าสุดของ window (day t) เป็นค่าทุก horizon
    - ma7: ใช้ค่าเฉลี่ย 7 วันล่าสุดของ window เป็นค่าทุก horizon

    Parameters
    ----------
    X: shape (N, lookback_days, n_features)  (feature index 0 = target)
    """
    N, lookback, n_features = X.shape

    naive_preds = np.zeros((N, forecast_horizon))
    ma7_preds = np.zeros((N, forecast_horizon))

    for i in range(N):
        history = X[i, :, 0]  # target history
        last_val = history[-1]
        ma7 = history[-7:].mean() if lookback >= 7 else history.mean()

        naive_preds[i, :] = last_val
        ma7_preds[i, :] = ma7

    return naive_preds, ma7_preds


# ----------------------------------------------------------------------
# MAIN: train forecasting model using actual CSV file
# ----------------------------------------------------------------------
def main():
    logger.info("=" * 80)
    logger.info("RandomForest Complaint Forecasting Model Training (Refactored)")
    logger.info("=" * 80)

    Path("ml_models/forecasting/models").mkdir(parents=True, exist_ok=True)
    Path("ml_models/forecasting/outputs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # READ CSV (assume data cleaned แล้ว)
    # ------------------------------------------------------------------
    csv_path = "../data/clean_data.csv"  # เปลี่ยนชื่อให้ตรงไฟล์ที่คุณใช้
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # PREPARE DATA
    # ------------------------------------------------------------------
    lookback_days = 30
    forecast_horizon = 7
    forecaster = ComplaintForecaster(
        lookback_days=lookback_days,
        forecast_horizon=forecast_horizon,
    )

    values, daily_counts = forecaster.prepare_data(df, target_col="complaint_count")
    X, y = forecaster.create_sequences(values)

    # train / val / test split (80 / 10 / 10)
    N = len(X)
    train_size = int(0.8 * N)
    val_size = int(0.1 * N)  # ยังไม่ได้ใช้ val ตรง ๆ แต่กันไว้ได้

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]  # เผื่อใช้ขยายในอนาคต

    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    logger.info(
        f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} "
        f"(Total windows: {N})"
    )

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    forecaster.train(X_train, y_train)

    # ------------------------------------------------------------------
    # EVALUATE MODEL
    # ------------------------------------------------------------------
    results = forecaster.evaluate(X_test, y_test)
    model_metrics = results["metrics"]
    preds = results["predictions"]
    actuals = results["actuals"]

    # ------------------------------------------------------------------
    # BASELINES: NAIVE + MOVING AVERAGE
    # ------------------------------------------------------------------
    naive_preds, ma7_preds = compute_baseline_predictions(
        X_test, forecast_horizon=forecast_horizon
    )

    naive_metrics = compute_metrics(y_test, naive_preds, per_horizon=True)
    ma7_metrics = compute_metrics(y_test, ma7_preds, per_horizon=True)

    logger.info("-" * 80)
    logger.info("Baseline comparison:")
    logger.info(
        f"[RandomForest] MAE={model_metrics['mae']:.2f}, "
        f"RMSE={model_metrics['rmse']:.2f}, "
        f"MAPE={model_metrics['mape']:.2f}%"
    )
    logger.info(
        f"[Naive]       MAE={naive_metrics['mae']:.2f}, "
        f"RMSE={naive_metrics['rmse']:.2f}, "
        f"MAPE={naive_metrics['mape']:.2f}%"
    )
    logger.info(
        f"[7-day MA]    MAE={ma7_metrics['mae']:.2f}, "
        f"RMSE={ma7_metrics['rmse']:.2f}, "
        f"MAPE={ma7_metrics['mape']:.2f}%"
    )
    logger.info("-" * 80)

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    forecaster.plot_predictions(actuals, preds)
    forecaster.plot_horizon_mae(
        model_metrics=model_metrics,
        naive_metrics=naive_metrics,
        ma7_metrics=ma7_metrics,
    )

    # ------------------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------------------
    forecaster.save_model()

    logger.info("=" * 80)
    logger.info("Forecasting pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
