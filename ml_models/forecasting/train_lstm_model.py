"""
RandomForest Time-Series Forecasting Model (Clean RF-Only Version)
- ใช้ RandomForestRegressor ตัวเดียวแบบ multi-output
- ใช้ GridSearchCV + TimeSeriesSplit
- ใช้ time features + rolling features
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------
# HELPER: METRICS
# ----------------------------------------------
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {"mae": mae, "rmse": rmse, "mape": mape}


# ----------------------------------------------
# MODEL CLASS
# ----------------------------------------------
class ComplaintForecaster:

    def __init__(self, lookback_days=30, forecast_horizon=7):
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon
        self.model: RandomForestRegressor | None = None

    # ------------------------------------------
    # PREPARE DATA
    # ------------------------------------------
    def prepare_data(self, df: pd.DataFrame, target_col="complaint_count"):

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")

        # daily aggregation
        if target_col in df.columns:
            daily = df[target_col].resample("D").sum().to_frame(name=target_col)
        else:
            daily = df.resample("D").size().to_frame(name=target_col)

        # fill missing days
        idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
        daily = daily.reindex(idx, fill_value=0)

        # Time features
        daily["dayofweek"] = daily.index.dayofweek
        daily["month"] = daily.index.month
        daily["is_weekend"] = daily["dayofweek"].isin([5, 6]).astype(int)

        # Rolling features
        daily["roll_mean_7"] = daily[target_col].rolling(7, min_periods=1).mean()
        daily["roll_std_7"] = daily[target_col].rolling(7, min_periods=1).std().fillna(0)
        daily["roll_mean_30"] = daily[target_col].rolling(30, min_periods=1).mean()

        feature_cols = [
            target_col,
            "dayofweek",
            "month",
            "is_weekend",
            "roll_mean_7",
            "roll_std_7",
            "roll_mean_30",
        ]

        values = daily[feature_cols].values

        return values, daily

    # ------------------------------------------
    # CREATE SEQUENCES
    # ------------------------------------------
    def create_sequences(self, data: np.ndarray):

        X, y = [], []
        T = len(data)
        n_features = data.shape[1]

        for i in range(T - self.lookback_days - self.forecast_horizon + 1):
            X.append(data[i : i + self.lookback_days, :])
            y.append(
                data[
                    i + self.lookback_days : i + self.lookback_days + self.forecast_horizon,
                    0,
                ]
            )

        return np.array(X), np.array(y)

    # ------------------------------------------
    # TRAIN
    # ------------------------------------------
    def train(self, X_train, y_train):

        logger.info("Training RandomForest (multi-output) with GridSearch...")

        N, L, F = X_train.shape
        X_flat = X_train.reshape(N, L * F)

        tscv = TimeSeriesSplit(n_splits=3)

        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        grid = GridSearchCV(
            base_model,
            param_grid,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            verbose=1,
            n_jobs=-1,
        )

        grid.fit(X_flat, y_train[:, 0])  # optimize using 1-step horizon

        logger.info(f"Best params: {grid.best_params_}")

        # Train model using multi-output y_train
        self.model = RandomForestRegressor(
            **grid.best_params_,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_flat, y_train)
        logger.info("Training completed.")

    # ------------------------------------------
    # PREDICT
    # ------------------------------------------
    def predict(self, X):

        if self.model is None:
            raise RuntimeError("Model not trained")

        N, L, F = X.shape
        X_flat = X.reshape(N, L * F)

        return self.model.predict(X_flat)

    # ------------------------------------------
    # PLOT
    # ------------------------------------------
    def plot_predictions(self, actual, pred):


        plt.figure(figsize=(15, 6))
        n = min(100, len(actual))

        for i in range(n):
            plt.plot(pred[i], "r--", alpha=0.2)
            plt.plot(actual[i], "b-", alpha=0.2)

        plt.title("RandomForest Multi-step Predictions")
        plt.xlabel("Days Ahead")
        plt.ylabel("Complaint Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig("ml_models/forecasting/outputs/predictions.png", dpi=300)
        plt.close()

        logger.info("Saved: outputs/predictions.png")
    # ------------------------------------------
    # SaveModel
    # ------------------------------------------
    def save_model(self, path="ml_models/forecasting/outputs/model.pkl"):
        if self.model is None:
            raise RuntimeError("Model not trained")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, out_path)
        logger.info(f"Model saved to: {out_path}")


# ----------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------
def main():

    df = pd.read_csv("clean_data.csv")

    model = ComplaintForecaster(lookback_days=30, forecast_horizon=7)

    values, daily = model.prepare_data(df)
    X, y = model.create_sequences(values)

    # split
    N = len(X)
    train_size = int(N * 0.8)
    val_size = int(N * 0.1)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    # train
    model.train(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    print("=== Evaluation ===")
    print(metrics)

    # plot
    model.plot_predictions(y_test, preds)

    #save model
    model.save_model("forecasting/outputs/rf_model.pkl")


if __name__ == "__main__":
    main()