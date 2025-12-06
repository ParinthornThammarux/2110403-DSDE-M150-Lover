"""
ML Model Integration Module
Integrates trained models (RandomForest Forecaster and Isolation Forest Anomaly Detector)
"""

import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory (parent of visualization folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MLModelIntegrator:
    """Integrates all ML models for the dashboard"""

    def __init__(self):
        self.rf_model = None
        self.anomaly_model = None
        self.anomaly_detector = None
        self.outage_model = None
        # Forecasting model parameters
        self.lookback_days = 30
        self.forecast_horizon = 7

    def load_forecasting_model(self, model_path: str = None):
        """
        Load RandomForest forecasting model

        คำอธิบาย: โหลดโมเดลการพยากรณ์ที่ train ไว้แล้ว
        ใช้ทำนายจำนวน complaint ในอนาคต
        """
        if model_path is None:
            model_path = PROJECT_ROOT / 'ml_models' / 'forecasting' / 'models' / 'rf_forecaster.pkl'
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path

        try:
            if model_path.exists():
                # Load the model (stored as a dict with 'model', 'lookback_days', 'forecast_horizon')
                model_data = joblib.load(model_path)

                # Extract the actual model from the dictionary if needed
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.rf_model = model_data['model']
                    self.lookback_days = model_data.get('lookback_days', 30)
                    self.forecast_horizon = model_data.get('forecast_horizon', 7)
                    logger.info(f"Loaded forecasting model from {model_path}")
                    logger.info(f"Model type: {type(self.rf_model)}")
                    logger.info(f"Lookback days: {self.lookback_days}, Forecast horizon: {self.forecast_horizon}")
                else:
                    # Fallback for direct model format
                    self.rf_model = model_data
                    self.lookback_days = 30
                    self.forecast_horizon = 7
                    logger.info(f"Loaded direct forecasting model from {model_path}")
                    logger.info(f"Model type: {type(self.rf_model)}")
                return True
            else:
                logger.warning(f"Forecasting model not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading forecasting model: {e}")
            return False

    def load_anomaly_model(self, model_path: str = None):
        """
        Load Isolation Forest anomaly detection model

        คำอธิบาย: โหลดโมเดลตรวจจับความผิดปกติที่ train ไว้แล้ว
        ใช้หา complaint ที่มีพฤติกรรมผิดปกติ
        """
        if model_path is None:
            model_path = PROJECT_ROOT / 'ml_models' / 'anomaly_detection' / 'models' / 'anomaly_if_model.pkl'
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path

        try:
            if model_path.exists():
                self.anomaly_model = joblib.load(model_path)
                logger.info(f"Loaded anomaly detection model from {model_path}")
                return True
            else:
                logger.warning(f"Anomaly model not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading anomaly model: {e}")
            return False

    def load_outage_model(self, model_path: str = None):
        """
        Load K-Means outage clustering model

        คำอธิบาย: โหลดโมเดลจัดกลุ่มพฤติกรรมไฟดับที่ train ไว้แล้ว
        ใช้วิเคราะห์รูปแบบการเกิดไฟดับ
        """
        if model_path is None:
            model_path = PROJECT_ROOT / 'ml_models' / 'outage_model' / 'models' / 'outage_kmeans_model.pkl'
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path

        try:
            if model_path.exists():
                self.outage_model = joblib.load(model_path)
                logger.info(f"Loaded outage clustering model from {model_path}")
                return True
            else:
                logger.warning(f"Outage model not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading outage model: {e}")
            return False

    def generate_forecast(self, df: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
        """
        Generate complaint volume forecast

        คำอธิบาย: พยากรณ์จำนวน complaint ในอนาคต
        ใช้โมเดล RandomForest หรือสร้าง forecast จำลองถ้าไม่มีโมเดล
        """
        try:
            # Aggregate historical data
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date
            daily_counts = df_copy.groupby('date').size().reset_index(name='count')
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])

            # Generate future dates
            last_date = daily_counts['date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')

            if self.rf_model is not None:
                # Use the new trained model (sequence-based)
                predictions = self._predict_with_sequence_model(daily_counts, days_ahead)
            else:
                # No model available
                logger.error("No forecasting model available")
                raise ValueError("Forecasting model not loaded. Please ensure the model file exists.")

            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'predicted': predictions,
                'lower_bound': predictions * 0.85,  # 15% confidence interval
                'upper_bound': predictions * 1.15
            })

            return forecast_df

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate forecast: {e}")

    def _predict_with_sequence_model(self, daily_counts: pd.DataFrame, days_ahead: int) -> np.ndarray:
        """
        Generate predictions using the sequence-based RandomForest model

        The model expects sequences with lookback_days and returns forecast_horizon day predictions
        """
        lookback_days = self.lookback_days
        forecast_horizon = self.forecast_horizon

        # Prepare daily data with features (matching training format)
        daily_data = daily_counts.set_index('date')

        # Add time features
        daily_data['dayofweek'] = daily_data.index.dayofweek
        daily_data['month'] = daily_data.index.month
        daily_data['is_weekend'] = daily_data['dayofweek'].isin([5, 6]).astype(int)

        # Add rolling features
        daily_data['roll_mean_7'] = daily_data['count'].rolling(7, min_periods=1).mean()
        daily_data['roll_std_7'] = daily_data['count'].rolling(7, min_periods=1).std().fillna(0)
        daily_data['roll_mean_30'] = daily_data['count'].rolling(30, min_periods=1).mean()

        # Select feature columns in the same order as training
        feature_cols = ['count', 'dayofweek', 'month', 'is_weekend', 'roll_mean_7', 'roll_std_7', 'roll_mean_30']
        values = daily_data[feature_cols].values

        # Generate predictions iteratively for the required days_ahead
        all_predictions = []
        current_sequence = values[-lookback_days:]  # Start with the last 30 days

        # Number of iterations needed
        num_iterations = int(np.ceil(days_ahead / forecast_horizon))

        for i in range(num_iterations):
            # Prepare input (flatten the sequence)
            X = current_sequence.reshape(1, -1)

            # Predict next 7 days
            pred = self.rf_model.predict(X)[0]  # Returns array of 7 values

            # Take only what we need (in case last iteration predicts more than needed)
            remaining = days_ahead - len(all_predictions)
            pred_to_use = pred[:min(forecast_horizon, remaining)]
            all_predictions.extend(pred_to_use)

            # Update sequence for next iteration
            # Create new rows with predicted values and estimated features
            for j, pred_val in enumerate(pred):
                if len(all_predictions) >= days_ahead:
                    break

                # Calculate date for this prediction
                next_date = daily_data.index[-1] + timedelta(days=len(all_predictions))

                # Create new row with prediction and features
                new_row = np.array([
                    pred_val,  # count
                    next_date.dayofweek,  # dayofweek
                    next_date.month,  # month
                    int(next_date.dayofweek in [5, 6]),  # is_weekend
                    np.mean(current_sequence[-7:, 0]),  # roll_mean_7
                    np.std(current_sequence[-7:, 0]),  # roll_std_7
                    np.mean(current_sequence[-30:, 0])  # roll_mean_30
                ])

                # Shift sequence and add new prediction
                current_sequence = np.vstack([current_sequence[1:], new_row])

        return np.array(all_predictions[:days_ahead])

    def generate_backtest_predictions(self, df: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
        """
        Generate predictions for past dates to validate model accuracy

        คำอธิบาย: สร้างค่าพยากรณ์สำหรับวันที่ในอดีตเพื่อเปรียบเทียบกับข้อมูลจริง
        """
        try:
            # Aggregate historical data
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date
            daily_counts = df_copy.groupby('date').size().reset_index(name='count')
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            daily_counts = daily_counts.tail(lookback_days)

            logger.info(f"Generating backtest predictions for {len(daily_counts)} days")
            logger.info(f"RF model available: {self.rf_model is not None}")

            if self.rf_model is None:
                raise ValueError("Forecasting model not loaded. Cannot generate backtest predictions.")

            # Get all historical data (not just lookback_days) for proper feature preparation
            df_all = df.copy()
            df_all['date'] = pd.to_datetime(df_all['timestamp']).dt.date
            all_daily_counts = df_all.groupby('date').size().reset_index(name='count')
            all_daily_counts['date'] = pd.to_datetime(all_daily_counts['date'])

            # Prepare data with features
            daily_data = all_daily_counts.set_index('date')
            daily_data['dayofweek'] = daily_data.index.dayofweek
            daily_data['month'] = daily_data.index.month
            daily_data['is_weekend'] = daily_data['dayofweek'].isin([5, 6]).astype(int)
            daily_data['roll_mean_7'] = daily_data['count'].rolling(7, min_periods=1).mean()
            daily_data['roll_std_7'] = daily_data['count'].rolling(7, min_periods=1).std().fillna(0)
            daily_data['roll_mean_30'] = daily_data['count'].rolling(30, min_periods=1).mean()

            feature_cols = ['count', 'dayofweek', 'month', 'is_weekend', 'roll_mean_7', 'roll_std_7', 'roll_mean_30']
            values = daily_data[feature_cols].values

            # Generate predictions for each day in lookback period
            lookback_window = 30
            predictions = []
            dates = []

            # Start from day 30 onwards (need at least 30 days of history)
            start_idx = max(lookback_window, len(values) - lookback_days)

            for i in range(start_idx, len(values)):
                if i < lookback_window:
                    continue

                # Get sequence for this prediction
                sequence = values[i - lookback_window:i]
                X = sequence.reshape(1, -1)

                # Predict (will give 7 days, but we only use first day for backtest)
                pred = self.rf_model.predict(X)[0][0]
                predictions.append(pred)
                dates.append(daily_data.index[i])

            past_df = pd.DataFrame({
                'date': dates,
                'predicted': predictions
            })

            logger.info(f"RF sequence model predictions generated: {len(predictions)} values")

            result = past_df.dropna()
            logger.info(f"Returning {len(result)} backtest predictions after dropna()")
            return result

        except Exception as e:
            logger.error(f"Error generating backtest predictions: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in complaint data

        คำอธิบาย: ตรวจจับ complaint ที่มีพฤติกรรมผิดปกติ
        เช่น ใช้เวลาแก้ปัญหานานผิดปกติ หรือเกิดในพื้นที่ผิดปกติ
        """
        df_result = df.copy()

        if self.anomaly_model is not None:
            try:
                # Prepare features
                features = self._prepare_anomaly_features(df)

                # Use loaded model
                scaler = self.anomaly_model.get('scaler')
                iso_forest = self.anomaly_model.get('iso_forest')

                if scaler and iso_forest:
                    features_scaled = scaler.transform(features)
                    predictions = iso_forest.predict(features_scaled)
                    anomaly_scores = iso_forest.score_samples(features_scaled)

                    # Convert to 0-1 scale (lower score = more anomalous)
                    anomaly_scores_normalized = 1 - ((anomaly_scores - anomaly_scores.min()) /
                                                     (anomaly_scores.max() - anomaly_scores.min() + 1e-10))

                    df_result['is_anomaly'] = (predictions == -1).astype(int)
                    df_result['anomaly_score'] = anomaly_scores_normalized

                    logger.info(f"Detected {df_result['is_anomaly'].sum()} anomalies using Isolation Forest")
                else:
                    # Fallback to statistical method
                    df_result = self._statistical_anomaly_detection(df_result)
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                df_result = self._statistical_anomaly_detection(df_result)
        else:
            # Use statistical method
            df_result = self._statistical_anomaly_detection(df_result)

        return df_result

    def _prepare_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection matching training features"""
        features = pd.DataFrame(index=df.index)

        # Temporal features (vectorized)
        if 'hour' in df.columns:
            features['hour'] = df['hour'].values
        else:
            ts = pd.to_datetime(df['timestamp'], errors='coerce')
            features['hour'] = ts.dt.hour.fillna(0).astype(int).values

        if 'day_of_week' in df.columns:
            features['day_of_week'] = df['day_of_week'].values
        else:
            ts = pd.to_datetime(df['timestamp'], errors='coerce')
            features['day_of_week'] = ts.dt.dayofweek.fillna(0).astype(int).values

        if 'month' in df.columns:
            features['month'] = df['month'].values
        else:
            ts = pd.to_datetime(df['timestamp'], errors='coerce')
            features['month'] = ts.dt.month.fillna(1).astype(int).values

        # Geospatial features
        if 'lat' in df.columns and 'lon' in df.columns:
            features['lat'] = df['lat'].fillna(13.7563).values
            features['lon'] = df['lon'].fillna(100.5018).values

        # Solve days
        if 'solve_days' in df.columns:
            features['solve_days'] = df['solve_days'].fillna(0).values

        # Type features (optimized)
        if 'type' in df.columns:
            type_col = df['type'].astype(str)
            features['is_flood'] = type_col.str.contains('น้ำท่วม', na=False).astype(int).values
            features['is_traffic'] = type_col.str.contains('จราจร|ถนน', na=False, regex=True).astype(int).values
            features['is_waste'] = type_col.str.contains('ความสะอาด|ขยะ', na=False, regex=True).astype(int).values
        elif 'primary_type' in df.columns:
            type_col = df['primary_type'].astype(str)
            features['is_flood'] = type_col.str.contains('น้ำท่วม', na=False).astype(int).values
            features['is_traffic'] = type_col.str.contains('จราจร|ถนน', na=False, regex=True).astype(int).values
            features['is_waste'] = type_col.str.contains('ความสะอาด|ขยะ', na=False, regex=True).astype(int).values

        # District daily count (optimized with categorical groupby)
        if 'district' in df.columns and 'timestamp' in df.columns:
            try:
                df_tmp = pd.DataFrame({
                    'district': df['district'].values,
                    'date': pd.to_datetime(df['timestamp'], errors='coerce').dt.date
                })
                district_daily = df_tmp.groupby(['district', 'date'], observed=True).size()
                df_tmp['daily_count'] = df_tmp.apply(
                    lambda row: district_daily.get((row['district'], row['date']), 1), axis=1
                )
                features['district_daily_count'] = df_tmp['daily_count'].values
            except:
                # Fallback: use constant value
                features['district_daily_count'] = 1

        # Fill missing values with median
        for col in features.columns:
            if features[col].isna().any():
                features[col] = features[col].fillna(features[col].median())

        # Final safety check
        features = features.fillna(0)

        return features

    def _statistical_anomaly_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback statistical anomaly detection based on solve_days"""
        median_solve = df['solve_days'].median()
        std_solve = df['solve_days'].std()

        # Z-score based anomaly detection
        df['anomaly_score'] = np.abs(df['solve_days'] - median_solve) / (std_solve + 1)
        df['anomaly_score'] = df['anomaly_score'].clip(0, 1)
        df['is_anomaly'] = (df['anomaly_score'] > 0.7).astype(int)

        logger.info(f"Detected {df['is_anomaly'].sum()} anomalies using statistical method")

        return df


def plot_forecast_visualization(forecast_df: pd.DataFrame, historical_df: pd.DataFrame = None, lookback_days: int = 90, ml_integrator=None) -> go.Figure:
    """
    Plot forecast with confidence intervals and historical comparison

    คำอธิบาย: แสดงการพยากรณ์จำนวน complaint พร้อม confidence interval
    เส้นสีน้ำเงินคือข้อมูลจริง เส้นสีแดงคือค่าพยากรณ์ (ทั้งอดีตและอนาคต) พื้นที่สีเทาคือช่วงความเชื่อมั่น
    """
    fig = go.Figure()

    # Process historical data first to get actual values
    historical_actual = None
    mape = None
    rmse = None
    predicted_past_df = None

    if historical_df is not None:
        historical_df = historical_df.copy()
        historical_df['date'] = pd.to_datetime(historical_df['timestamp']).dt.date
        daily_counts = historical_df.groupby('date').size().reset_index(name='count')
        daily_counts = daily_counts.tail(lookback_days)  # Last N days
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        historical_actual = daily_counts

        # Exclude the last day to avoid showing incomplete data that looks like a drop
        daily_counts_display = daily_counts.iloc[:-1] if len(daily_counts) > 1 else daily_counts

        # Add historical actual data (excluding last day)
        fig.add_trace(go.Scatter(
            x=daily_counts_display['date'],
            y=daily_counts_display['count'],
            mode='lines',
            name='ข้อมูลจริง',
            line=dict(color='#1f77b4', width=2.5),
            hovertemplate='วันที่: %{x}<br>จำนวนจริง: %{y:,.0f}<extra></extra>'
        ))

        # Generate predictions for past dates using RF model
        try:
            if ml_integrator is not None:
                # Use RF model for past predictions
                predicted_past_df = ml_integrator.generate_backtest_predictions(historical_df, lookback_days)

                if len(predicted_past_df) > 0:
                    # Merge with actual data to calculate metrics
                    merged = pd.merge(
                        daily_counts[['date', 'count']],
                        predicted_past_df[['date', 'predicted']],
                        on='date',
                        how='inner'
                    )

                    if len(merged) > 0:
                        actual = merged['count']
                        predicted = merged['predicted']
                        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            else:
                # Fallback to moving average if no ML integrator
                window = 7
                daily_counts['predicted_past'] = daily_counts['count'].rolling(window=window, min_periods=1).mean().shift(1)
                predicted_past_df = daily_counts[['date', 'predicted_past']].copy()
                predicted_past_df = predicted_past_df.rename(columns={'predicted_past': 'predicted'})
                predicted_past_df = predicted_past_df.dropna()

                # Calculate accuracy metrics
                valid_indices = daily_counts['predicted_past'].notna()
                if valid_indices.sum() > 0:
                    actual = daily_counts.loc[valid_indices, 'count']
                    predicted = daily_counts.loc[valid_indices, 'predicted_past']
                    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        except Exception as e:
            logger.warning(f"Error generating past predictions: {e}")
            pass

    # Combine past predictions with future predictions into one continuous line
    if predicted_past_df is not None and len(predicted_past_df) > 0:
        logger.info(f"Combining {len(predicted_past_df)} past predictions with {len(forecast_df)} future predictions")

        # Combine past and future predictions
        combined_predictions = pd.concat([
            predicted_past_df[['date', 'predicted']],
            forecast_df[['date', 'predicted']]
        ], ignore_index=True)

        # Add single continuous prediction line (past + future)
        fig.add_trace(go.Scatter(
            x=combined_predictions['date'],
            y=combined_predictions['predicted'],
            mode='lines+markers',
            name='Predicted (model)',
            line=dict(color='red', width=2.5),
            marker=dict(size=5, symbol='circle'),
            hovertemplate='Date: %{x}<br>Predicted: %{y:.0f}<extra></extra>'
        ))
    else:
        # Only future predictions if no historical data
        logger.warning("No past predictions available, showing only future predictions")
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted'],
            mode='lines+markers',
            name='Predicted (model)',
            line=dict(color='red', width=2.5),
            marker=dict(size=5, symbol='circle'),
            hovertemplate='Date: %{x}<br>Predicted: %{y:.0f}<extra></extra>'
        ))

    # Add future confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['upper_bound'],
        mode='lines',
        name='Confidence Interval',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['lower_bound'],
        mode='lines',
        name='Confidence Interval',
        line=dict(width=0),
        fillcolor='rgba(255, 0, 0, 0.15)',
        fill='tonexty',
        showlegend=True,
        hovertemplate='Lower Bound: %{y:.0f}<extra></extra>'
    ))

    # Add vertical line to separate past and future
    if historical_actual is not None and len(historical_actual) > 1:
        # Use second-to-last date since we're excluding the last day from display
        display_last_date = historical_actual['date'].iloc[-2]

        fig.add_shape(
            type="line",
            x0=display_last_date,
            x1=display_last_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dash"),
            opacity=0.5
        )

        fig.add_annotation(
            x=display_last_date,
            y=1,
            yref="paper",
            text="วันนี้",
            showarrow=False,
            yshift=10,
            font=dict(size=10, color="gray")
        )

    # Add title with accuracy info if available
    title_text = 'Forecasted Number of Complaints (RandomForest Model)'
    if mape is not None and rmse is not None:
        title_text += f'<br><sub>Accuracy: MAPE={mape:.1f}%, RMSE={rmse:.0f}</sub>'

    fig.update_layout(
        title=title_text,
        xaxis_title='Date',
        yaxis_title='Number of Complaints',
        template='plotly_white',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_anomaly_scatter(
    df: pd.DataFrame,
    max_points: int = 800,
    max_normal_points: int = 600,
) -> go.Figure:
    """
    Fast scatter plot of anomalies with scores.

    - Downsample normal points
    - Limit anomaly points
    - Use vectorized operations for hover text
    """
    fig = go.Figure()

    # -------------------------------------------------
    # 1) Normal points (background, light gray)
    # -------------------------------------------------
    df_normal = df[df["anomaly_score"] <= 0.5]

    if not df_normal.empty:
        # Sample normal points for performance
        sample_size = min(len(df_normal), max_normal_points)
        df_normal_sampled = (
            df_normal.sample(n=sample_size, random_state=42)
            if len(df_normal) > sample_size
            else df_normal
        )

        fig.add_trace(
            go.Scatter(
                x=df_normal_sampled["timestamp"],
                y=df_normal_sampled["solve_days"],
                mode="markers",
                marker=dict(
                    size=4,
                    color="lightgray",
                    opacity=0.3,
                ),
                name="ปกติ",
                hovertemplate=(
                    "<b>ปกติ</b><br>"
                    "วันที่: %{x}<br>"
                    "ระยะเวลาแก้ปัญหา: %{y} วัน<extra></extra>"
                ),
            )
        )

    # -------------------------------------------------
    # 2) Anomaly points
    # -------------------------------------------------
    df_anomaly = df[df["anomaly_score"] > 0.5]

    # Limit number of anomaly points for performance
    if len(df_anomaly) > max_points:
        # Keep top anomalies by score (simple and fast)
        df_anomaly = df_anomaly.nlargest(max_points, "anomaly_score")

    # Prepare hover text in vectorized way (no iterrows)
    if not df_anomaly.empty:
        hover_text = (
            df_anomaly["district"].astype(str)
            + "<br>"
            + df_anomaly["primary_type"].astype(str)
        )

        fig.add_trace(
            go.Scatter(
                x=df_anomaly["timestamp"],
                y=df_anomaly["solve_days"],
                mode="markers",
                marker=dict(
                    size=df_anomaly["anomaly_score"].to_numpy() * 20.0,
                    color=df_anomaly["anomaly_score"].to_numpy(),
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(
                        title="Anomaly<br>Score",
                        x=1.15,
                        xanchor="left",
                        len=0.7,
                        y=0.5,
                    ),
                    line=dict(width=1, color="white"),
                    opacity=0.8,
                    cmin=0.5,
                    cmax=1.0,
                ),
                name="ผิดปกติ",
                text=hover_text,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "วันที่: %{x}<br>"
                    "ระยะเวลาแก้ปัญหา: %{y} วัน<br>"
                    "Anomaly Score: %{marker.color:.2f}"
                    "<extra></extra>"
                ),
            )
        )
    else:
        # Empty trace just to show colorbar (rare case)
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(
                    size=10,
                    color=[],
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(
                        title="Anomaly<br>Score",
                        x=1.15,
                        xanchor="left",
                        len=0.7,
                        y=0.5,
                    ),
                    cmin=0.5,
                    cmax=1.0,
                ),
                name="ผิดปกติ",
            )
        )

    # -------------------------------------------------
    # 3) Y-axis settings
    # -------------------------------------------------
    max_solve_days = float(df["solve_days"].max())

    # ticks every 5 days, up to 100
    tick_vals = list(range(0, 101, 5))
    tick_text = [str(i) for i in tick_vals]

    y_range = [0, min(max_solve_days * 1.05, 105)]

    if max_solve_days > 100:
        tick_vals.append(105)
        tick_text.append("100+")
        y_range = [0, 110]

    fig.update_layout(
        title=f"การตรวจจับความผิดปกติ (แสดง {len(df_anomaly):,} รายการที่ผิดปกติ)",
        xaxis_title="วันที่",
        yaxis_title="ระยะเวลาในการแก้ปัญหา (วัน)",
        yaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=y_range,
        ),
        template="plotly_white",
        height=500,
        hovermode="closest",
        margin=dict(r=200, t=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def plot_anomaly_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Distribution of anomalies by district and type (fast).
    """
    anomalies = df[df["is_anomaly"] == 1]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Anomalies ตามเขต", "Anomalies ตามประเภท"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    # By district (top 10)
    district_counts = anomalies["district"].value_counts().head(10)
    fig.add_trace(
        go.Bar(
            x=district_counts.index,
            y=district_counts.values,
            marker_color="indianred",
            hovertemplate="<b>%{x}</b><br>จำนวน: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # By type (top 10)
    type_counts = anomalies["primary_type"].value_counts().head(10)
    fig.add_trace(
        go.Bar(
            x=type_counts.index,
            y=type_counts.values,
            marker_color="lightcoral",
            hovertemplate="<b>%{x}</b><br>จำนวน: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)

    fig.update_layout(
        title_text="การกระจายของ Anomalies",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    return fig
