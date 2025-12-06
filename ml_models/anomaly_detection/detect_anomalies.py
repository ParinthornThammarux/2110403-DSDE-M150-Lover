"""
Anomaly Detection System for Urban Complaints
Model-focused version: fit Isolation Forest and save model file
(เวอร์ชัน: assume CSV ถูก clean มาแล้ว)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy import stats
import logging
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplaintAnomalyDetector:
    """Multi-method anomaly detection for complaint data"""

    def __init__(self, contamination=0.05, max_samples=10000, n_estimators=300, max_features=1.0):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.iso_forest = None

        # เก็บค่ากลางไว้ใช้ตอน predict
        self.medians_ = None
        # เก็บชื่อ feature ไว้สำหรับ ensure ลำดับ column ตอนใช้กับ data ใหม่
        self.feature_names_ = None

        # config ของ Isolation Forest
        self.if_params = {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "contamination": contamination,
            "random_state": 42,
            "n_jobs": -1,
        }

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for anomaly detection"""
        logger.info("Preparing features for anomaly detection...")

        features = pd.DataFrame(index=df.index)

        # Temporal features
        ts = pd.to_datetime(df["timestamp"])
        features["hour"] = ts.dt.hour
        features["day_of_week"] = ts.dt.dayofweek
        features["month"] = ts.dt.month

        # Include complaint_count if available
        if "complaint_count" in df.columns:
            features["complaint_count"] = df["complaint_count"]

        # Geospatial features (if available)
        if "lat" in df.columns and "lon" in df.columns:
            features["lat"] = df["lat"]
            features["lon"] = df["lon"]

        # Complaint characteristics
        if "solve_days" in df.columns:
            features["solve_days"] = df["solve_days"].fillna(0)

        # Category encoding (simple one-hot-ish flags)
        if "type" in df.columns:
            features["is_flood"] = df["type"].str.contains("น้ำท่วม", na=False).astype(int)
            features["is_traffic"] = df["type"].str.contains("จราจร|ถนน", na=False).astype(int)
            features["is_waste"] = df["type"].str.contains("ความสะอาด|ขยะ", na=False).astype(int)

        # Aggregated features (complaints per district per day)
        if "district" in df.columns:
            df_tmp = df.copy()
            df_tmp["date"] = pd.to_datetime(df_tmp["timestamp"]).dt.date
            district_daily = (
                df_tmp.groupby(["district", "date"])
                .size()
                .reset_index(name="daily_count")
            )

            # Merge back
            df_with_count = df_tmp.merge(district_daily, on=["district", "date"], how="left")
            features["district_daily_count"] = df_with_count["daily_count"]

        # Keep only numeric columns (กัน object/string หลุดเข้าไป)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_cols]

        logger.info(f"Prepared {len(features.columns)} features: {list(features.columns)}")
        return features

    # ------------------------------------------------------------------
    # ISOLATION FOREST CORE MODEL
    # ------------------------------------------------------------------
    def fit_model(self, features: pd.DataFrame):
        """Fit Isolation Forest model (train)"""
        logger.info("Fitting Isolation Forest model...")

        # Keep feature names (for later use with new data)
        self.feature_names_ = list(features.columns)

        # Handle missing values (fit medians from training data)
        self.medians_ = features.median(numeric_only=True)
        features_filled = features.fillna(self.medians_)

        # Scale features
        features_scaled = self.scaler.fit_transform(features_filled)

        # Train Isolation Forest
        self.iso_forest = IsolationForest(**self.if_params)
        self.iso_forest.fit(features_scaled)
        logger.info("Isolation Forest training completed.")

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Align incoming features with the feature_names_ used during training.
        """
        if self.feature_names_ is None:
            logger.warning("feature_names_ is None; using current features as-is.")
            return features

        # Add any missing columns with 0, and ensure order matches
        for col in self.feature_names_:
            if col not in features.columns:
                features[col] = 0

        features = features[self.feature_names_]
        return features

    def predict_isolation_forest(self, features: pd.DataFrame):
        """Predict anomalies using already-fitted Isolation Forest"""
        if self.iso_forest is None:
            raise RuntimeError("Model is not fitted yet. Call fit_model() first.")

        # Align feature columns with training schema
        features = self._align_features(features)

        # Use stored medians; fallback if somehow missing
        if self.medians_ is None:
            self.medians_ = features.median(numeric_only=True)

        features_filled = features.fillna(self.medians_)
        features_scaled = self.scaler.transform(features_filled)

        predictions = self.iso_forest.predict(features_scaled)   # -1 anomaly, 1 normal
        anomaly_scores = self.iso_forest.score_samples(features_scaled)

        n_anomalies = (predictions == -1).sum()
        logger.info(
            f"Isolation Forest predicted {n_anomalies} anomalies "
            f"({n_anomalies / len(predictions) * 100:.2f}%)"
        )

        return predictions, anomaly_scores

    # ------------------------------------------------------------------
    # MODEL PERSISTENCE
    # ------------------------------------------------------------------
    def save_model(self, path: str = "anomaly_if_model.pkl"):
        """Save Isolation Forest model + scaler to file"""
        if self.iso_forest is None:
            raise RuntimeError("No model to save. Fit the model first.")

        obj = {
            "scaler": self.scaler,
            "iso_forest": self.iso_forest,
            "contamination": self.contamination,
            "medians_": self.medians_,
            "feature_names_": self.feature_names_,
            "if_params": self.if_params,
        }
        joblib.dump(obj, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str = "anomaly_if_model.pkl"):
        """Load Isolation Forest model + scaler from file"""
        obj = joblib.load(path)
        self.scaler = obj["scaler"]
        self.iso_forest = obj["iso_forest"]
        self.contamination = obj.get("contamination", self.contamination)
        self.medians_ = obj.get("medians_", None)
        self.feature_names_ = obj.get("feature_names_", None)
        self.if_params = obj.get("if_params", self.if_params)
        logger.info(f"Model loaded from {path}")

    # ------------------------------------------------------------------
    # OTHER DETECTION METHODS (still usable to combine)
    # ------------------------------------------------------------------
    def detect_statistical(
        self,
        df: pd.DataFrame,
        column: str = "solve_days",
        threshold: float = 3.0,
    ) -> np.ndarray:
        """Detect anomalies using statistical Z-score method"""
        logger.info(f"Running statistical anomaly detection on '{column}'...")

        if column not in df.columns:
            logger.warning(f"Column '{column}' not found")
            return np.zeros(len(df), dtype=int)

        values = df[column].fillna(df[column].median())

        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(values))
        z_scores = np.nan_to_num(z_scores, nan=0.0)

        # Anomalies are points with |z-score| > threshold
        anomalies = z_scores > threshold

        n_anomalies = anomalies.sum()
        logger.info(
            f"Statistical method detected {n_anomalies} anomalies "
            f"({n_anomalies / len(df) * 100:.2f}%)"
        )

        return anomalies.astype(int)

    def detect_spatial_clusters(
        self,
        df: pd.DataFrame,
        eps=0.01,
        min_samples=5,
    ) -> np.ndarray:
        """Detect spatial anomalies using DBSCAN clustering"""
        logger.info("Running spatial anomaly detection with DBSCAN...")

        if "lat" not in df.columns or "lon" not in df.columns:
            logger.warning("Latitude/Longitude not found")
            return np.zeros(len(df), dtype=int)

        # Extract coordinates
        coords = df[["lat", "lon"]].dropna()

        if len(coords) == 0:
            logger.warning("No coordinates available for DBSCAN")
            return np.zeros(len(df), dtype=int)

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(coords)

        # Points with label -1 are noise/anomalies
        anomalies = (clusters == -1).astype(int)

        n_anomalies = anomalies.sum()
        logger.info(
            f"DBSCAN detected {n_anomalies} spatial anomalies "
            f"({n_anomalies / len(coords) * 100:.2f}%)"
        )

        # Align with original dataframe
        result = np.zeros(len(df), dtype=int)
        result[coords.index] = anomalies

        return result

    def detect_temporal_spikes(
        self,
        df: pd.DataFrame,
        window: int = 7,
        threshold_std: float = 3.0,
    ) -> np.ndarray:
        """Detect temporal spikes in complaint volume"""
        logger.info("Running temporal spike detection...")

        df_tmp = df.copy()
        df_tmp["date"] = pd.to_datetime(df_tmp["timestamp"]).dt.date

        # If complaint_count exists, use sum of it per day, otherwise count rows
        if "complaint_count" in df_tmp.columns:
            daily_counts = df_tmp.groupby("date")["complaint_count"].sum()
        else:
            daily_counts = df_tmp.groupby("date").size()

        # Calculate rolling statistics
        rolling_mean = daily_counts.rolling(window=window, center=True).mean()
        rolling_std = daily_counts.rolling(window=window, center=True).std()
        rolling_std_safe = rolling_std.replace(0, np.nan)

        # Detect spikes
        z_scores = (daily_counts - rolling_mean) / (rolling_std_safe + 1e-10)
        z_scores = z_scores.fillna(0.0)
        spike_dates = daily_counts[np.abs(z_scores) > threshold_std].index

        logger.info(f"Detected {len(spike_dates)} days with anomalous complaint volumes")

        # Map back to original dataframe
        anomalies = df_tmp["date"].isin(spike_dates).astype(int)

        return anomalies.values

    def combine_methods(self, anomalies_dict: dict, weights: dict = None):
        """Combine multiple anomaly detection methods using weighted voting"""
        if weights is None:
            # Equal weights by default
            weights = {method: 1.0 for method in anomalies_dict.keys()}

        logger.info(f"Combining {len(anomalies_dict)} detection methods with weights: {weights}")

        first_key = next(iter(anomalies_dict))
        n = len(anomalies_dict[first_key])
        combined_score = np.zeros(n, dtype=float)

        for method, anomalies in anomalies_dict.items():
            weight = weights.get(method, 1.0)
            anomalies = np.asarray(anomalies)

            # Convert -1/1 labels to 0/1 if needed
            anomalies_binary = (anomalies == -1).astype(int) if anomalies.min() < 0 else anomalies
            combined_score += weight * anomalies_binary

        # Normalize
        max_score = sum(weights.values())
        if max_score > 0:
            combined_score /= max_score

        # Threshold for final anomaly classification
        final_anomalies = (combined_score > 0.5).astype(int)

        n_final = final_anomalies.sum()
        logger.info(
            f"Combined method identified {n_final} final anomalies "
            f"({n_final / len(final_anomalies) * 100:.2f}%)"
        )

        return final_anomalies, combined_score

    # ------------------------------------------------------------------
    # NEW: EVALUATION METHODS (UNSUPERVISED)
    # ------------------------------------------------------------------
    def evaluate_score_separation(self, predictions: np.ndarray, scores: np.ndarray) -> float:
        """
        วิธีที่ 1: วัดการแยกตัวของ anomaly score ระหว่างกลุ่มปกติ (-1/1) โดยใช้ Silhouette score
        - ใช้ label จาก Isolation Forest (-1 anomaly, 1 normal)
        - ยิ่งค่าใกล้ 1 แปลว่า distribution ของ score แยกกลุ่มดี
        """
        # แปลงเป็นกลุ่ม 0/1 (0 = normal, 1 = anomaly)
        labels = (predictions == -1).astype(int)

        # ถ้ามีกลุ่มเดียวหรือกลุ่มเล็กเกินไป silhouette จะคำนวณไม่ได้
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2 or np.any(counts < 2):
            logger.warning("Not enough samples in one of the groups for silhouette_score.")
            return 0.0

        # IsolationForest: scores ปกติจะสูงกว่า anomaly -> แต่ silhouette ไม่แคร์ scale
        values = scores.reshape(-1, 1)

        try:
            s = silhouette_score(values, labels)
            logger.info(f"Silhouette score (score separation) = {s:.4f}")
            return float(s)
        except Exception as e:
            logger.warning(f"Silhouette score failed with error: {e}")
            return 0.0

    def evaluate_stability(self, features: pd.DataFrame, n_iter: int = 5, noise_std: float = 0.01) -> float:
        """
        วิธีที่ 2: วัดความเสถียรของโมเดลต่อ noise เล็กน้อยใน feature
        - ใช้โมเดลตัวเดิม (ไม่ retrain)
        - เพิ่ม noise ~ N(0, noise_std) ให้ features แล้วดู prediction เปลี่ยนไปมากน้อยแค่ไหน
        - ค่าใกล้ 1 = stable สูง, ค่าใกล้ 0 = prediction ไวต่อการเปลี่ยนแปลงเล็กน้อย
        """
        if self.iso_forest is None:
            raise RuntimeError("Model is not fitted yet. Call fit_model() first.")

        # base prediction (ไม่มี noise)
        base_preds, _ = self.predict_isolation_forest(features)
        base_binary = (base_preds == -1).astype(int)

        stability_scores = []

        for i in range(n_iter):
            # เพิ่ม noise ให้กับ features เฉพาะ column numeric
            noise = np.random.normal(loc=0.0, scale=noise_std, size=features.shape)
            noisy_values = features.values + noise
            noisy_df = pd.DataFrame(noisy_values, columns=features.columns, index=features.index)

            preds_i, _ = self.predict_isolation_forest(noisy_df)
            preds_binary = (preds_i == -1).astype(int)

            # วัดสัดส่วนที่ prediction ตรงกับ base
            same_ratio = np.mean(preds_binary == base_binary)
            stability_scores.append(same_ratio)

        stability_mean = float(np.mean(stability_scores)) if stability_scores else 0.0
        logger.info(f"Stability score (mean agreement over {n_iter} runs) = {stability_mean:.4f}")
        return stability_mean


# ----------------------------------------------------------------------
# MAIN: train Isolation Forest model using actual CSV file
# + แยก train/validation ตามเวลา (time-based split 80/20)
# + เพิ่ม evaluation วิธีที่ 1 และ 2 บน validation set
# ----------------------------------------------------------------------
def main():
    logger.info("=" * 80)
    logger.info("Train Anomaly Detection Model (Isolation Forest) with time-based train/validation split")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # READ CSV (คลีนมาแล้ว ใช้ได้เลย)
    # ------------------------------------------------------------------
    csv_path = "../../data/clean_data.csv"
    logger.info(f"Loading CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # TIME-BASED SPLIT (เช็คว่า timestamp เรียงตามเวลา)
    # ------------------------------------------------------------------
    if "timestamp" not in df.columns:
        raise ValueError("Input CSV must contain 'timestamp' column for time-based split.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    split_ratio = 0.8  # 80% train, 20% validation
    split_idx = int(len(df) * split_ratio)

    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()

    logger.info(f"Time-based split with ratio {split_ratio}")
    logger.info(f"Train set: {df_train.shape[0]} rows")
    logger.info(f"Validation set: {df_val.shape[0]} rows")

    # ------------------------------------------------------------------
    # INITIALIZE MODEL
    # ------------------------------------------------------------------
    detector = ComplaintAnomalyDetector(
        contamination=0.05,
        max_samples=30000,
        n_estimators=400,
        max_features=0.8,
    )

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING (ใช้ train/val แยกกัน)
    # ------------------------------------------------------------------
    features_train = detector.prepare_features(df_train)
    features_val = detector.prepare_features(df_val)

    # ------------------------------------------------------------------
    # TRAIN MODEL (ใช้เฉพาะ train)
    # ------------------------------------------------------------------
    detector.fit_model(features_train)

    # ------------------------------------------------------------------
    # VALIDATION (ดู distribution ของ anomaly บน val)
    # ------------------------------------------------------------------
    val_preds, val_scores = detector.predict_isolation_forest(features_val)
    val_anomaly_ratio = (val_preds == -1).mean()

    logger.info(
        f"Validation set anomaly ratio: {val_anomaly_ratio * 100:.2f}% "
        f"({(val_preds == -1).sum()} / {len(val_preds)} records)"
    )

    # ------------------------------------------------------------------
    # EVALUATION METHODS (บน validation set)
    # ------------------------------------------------------------------
    # วิธีที่ 1: วัดการแยกตัวของ score ระหว่าง anomaly vs normal
    score_sep = detector.evaluate_score_separation(val_preds, val_scores)
    logger.info(f"[Evaluation-1] Score separation (silhouette) on validation = {score_sep:.4f}")

    # วิธีที่ 2: วัดความเสถียรของ prediction ต่อ noise เล็กน้อย
    stability = detector.evaluate_stability(features_val, n_iter=5, noise_std=0.01)
    logger.info(f"[Evaluation-2] Stability on validation = {stability:.4f}")

    # ------------------------------------------------------------------
    # SAVE MODEL (เทรนจาก train แล้ว)
    # ------------------------------------------------------------------
    Path("models").mkdir(parents=True, exist_ok=True)
    model_path = "models/anomaly_if_model.pkl"

    detector.save_model(model_path)

    logger.info("=" * 80)
    logger.info(f"Training completed and model saved to: {model_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
