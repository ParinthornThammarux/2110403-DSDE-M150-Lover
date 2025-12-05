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


class MLModelIntegrator:
    """Integrates all ML models for the dashboard"""

    def __init__(self):
        self.rf_model = None
        self.anomaly_model = None
        self.anomaly_detector = None

    def load_forecasting_model(self, model_path: str = 'ml_models/forecasting/models/rf_forecaster.pkl'):
        """
        Load RandomForest forecasting model

        คำอธิบาย: โหลดโมเดลการพยากรณ์ที่ train ไว้แล้ว
        ใช้ทำนายจำนวน complaint ในอนาคต
        """
        try:
            if Path(model_path).exists():
                model_data = joblib.load(model_path)
                self.rf_model = model_data.get('model')
                logger.info(f"Loaded forecasting model from {model_path}")
                return True
            else:
                logger.warning(f"Forecasting model not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading forecasting model: {e}")
            return False

    def load_anomaly_model(self, model_path: str = 'ml_models/anomaly_detection/anomaly_if_model.pkl'):
        """
        Load Isolation Forest anomaly detection model

        คำอธิบาย: โหลดโมเดลตรวจจับความผิดปกติที่ train ไว้แล้ว
        ใช้หา complaint ที่มีพฤติกรรมผิดปกติ
        """
        try:
            if Path(model_path).exists():
                self.anomaly_model = joblib.load(model_path)
                logger.info(f"Loaded anomaly detection model from {model_path}")
                return True
            else:
                logger.warning(f"Anomaly model not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading anomaly model: {e}")
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
                # Use trained model if available
                # Create features for prediction
                future_df = pd.DataFrame({'date': future_dates})
                future_df['day_of_week'] = future_df['date'].dt.dayofweek
                future_df['month'] = future_df['date'].dt.month
                future_df['day'] = future_df['date'].dt.day

                # Predict
                try:
                    predictions = self.rf_model.predict(future_df[['day_of_week', 'month', 'day']])
                except:
                    # Fallback to simulated forecast
                    predictions = self._simulate_forecast(daily_counts, days_ahead)
            else:
                # Simulated forecast
                predictions = self._simulate_forecast(daily_counts, days_ahead)

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
            return self._fallback_forecast(days_ahead)

    def _simulate_forecast(self, daily_counts: pd.DataFrame, days_ahead: int) -> np.ndarray:
        """Generate simulated forecast based on historical patterns"""
        # Calculate trend and seasonality
        recent_mean = daily_counts['count'].tail(30).mean()
        recent_std = daily_counts['count'].tail(30).std()

        trend = np.linspace(recent_mean, recent_mean * 1.05, days_ahead)
        seasonality = recent_std * 0.3 * np.sin(2 * np.pi * np.arange(days_ahead) / 7)
        noise = np.random.normal(0, recent_std * 0.1, days_ahead)

        forecast = trend + seasonality + noise
        return np.maximum(forecast, 0)  # Ensure non-negative

    def _fallback_forecast(self, days_ahead: int) -> pd.DataFrame:
        """Fallback forecast when data is unavailable"""
        future_dates = pd.date_range(start=datetime.now(), periods=days_ahead, freq='D')
        predictions = 100 + 20 * np.sin(2 * np.pi * np.arange(days_ahead) / 7)

        return pd.DataFrame({
            'date': future_dates,
            'predicted': predictions,
            'lower_bound': predictions * 0.85,
            'upper_bound': predictions * 1.15
        })

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

            if self.rf_model is not None:
                # Create features for past dates
                past_df = daily_counts[['date']].copy()
                past_df['day_of_week'] = past_df['date'].dt.dayofweek
                past_df['month'] = past_df['date'].dt.month
                past_df['day'] = past_df['date'].dt.day

                # Predict using RF model
                try:
                    predictions = self.rf_model.predict(past_df[['day_of_week', 'month', 'day']])
                    past_df['predicted'] = predictions
                    logger.info(f"RF model predictions generated: {len(predictions)} values")
                except Exception as e:
                    logger.warning(f"RF prediction failed for backtest, using moving average: {e}")
                    # Fallback to moving average
                    past_df['predicted'] = daily_counts['count'].rolling(window=7, min_periods=1).mean().shift(1)
            else:
                # Use moving average as fallback
                logger.info("No RF model, using moving average for backtest")
                past_df = daily_counts[['date']].copy()
                past_df['predicted'] = daily_counts['count'].rolling(window=7, min_periods=1).mean().shift(1)

            result = past_df.dropna()
            logger.info(f"Returning {len(result)} backtest predictions after dropna()")
            return result

        except Exception as e:
            logger.error(f"Error generating backtest predictions: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def generate_synthetic_data_from_forecast(self, start_date: str = None, days: int = 90) -> pd.DataFrame:
        """
        Generate synthetic complaint data based on forecast model predictions

        คำอธิบาย: สร้างข้อมูล complaint สังเคราะห์จากโมเดลการพยากรณ์
        ใช้สำหรับการวิเคราะห์ความผิดปกติโดยไม่ใช้ข้อมูลจริง

        Args:
            start_date: วันที่เริ่มต้น (default: 90 days ago)
            days: จำนวนวันที่ต้องการสร้าง

        Returns:
            DataFrame with synthetic complaint data
        """
        try:
            if self.rf_model is None:
                logger.warning("No RF model available for synthetic data generation")
                return pd.DataFrame()

            # Set date range
            if start_date is None:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
            else:
                start_date = pd.to_datetime(start_date)
                end_date = start_date + timedelta(days=days)

            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Create feature dataframe
            forecast_df = pd.DataFrame({'date': date_range})
            forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
            forecast_df['month'] = forecast_df['date'].dt.month
            forecast_df['day'] = forecast_df['date'].dt.day

            # Generate predictions
            predictions = self.rf_model.predict(forecast_df[['day_of_week', 'month', 'day']])
            forecast_df['predicted_count'] = predictions.round().astype(int)

            logger.info(f"Generated {len(forecast_df)} days of predictions")

            # Create synthetic complaint records
            # For each date, create N rows based on predicted count
            synthetic_records = []

            # Common districts and types for synthetic data
            districts = ['คลองเตย', 'บางกอกใหญ่', 'ปทุมวัน', 'สาทร', 'ราชเทวี',
                        'ดินแดง', 'ห้วยขวาง', 'วัฒนา', 'บางรัก', 'ประเวศ']
            primary_types = ['ถนน/ทางเท้า', 'ขยะ', 'น้ำประปา/น้ำใช้', 'ไฟฟ้า/แสงสว่าง',
                           'การจราจร', 'ความสะอาด', 'สวนสาธารณะ', 'อื่นๆ']

            for _, row in forecast_df.iterrows():
                date = row['date']
                count = row['predicted_count']

                # Generate records for this date
                for i in range(count):
                    # Generate random but realistic features
                    hour = np.random.choice([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                           p=[0.05, 0.1, 0.15, 0.15, 0.1, 0.08, 0.08, 0.08, 0.05, 0.05, 0.05, 0.03, 0.02, 0.01])
                    minute = np.random.randint(0, 60)

                    timestamp = date + timedelta(hours=int(hour), minutes=int(minute))

                    # Random district and type
                    district = np.random.choice(districts)
                    primary_type = np.random.choice(primary_types)

                    # Generate solve_days with realistic distribution
                    # Most complaints solved within 30 days, but some take longer
                    solve_days = np.random.gamma(shape=2, scale=5)  # mean ~10 days
                    solve_days = min(solve_days, 180)  # cap at 180 days

                    # Random coordinates within Bangkok bounds
                    lat = 13.7 + np.random.uniform(-0.15, 0.15)
                    lon = 100.5 + np.random.uniform(-0.15, 0.15)

                    synthetic_records.append({
                        'timestamp': timestamp,
                        'date': date,
                        'hour': hour,
                        'day_of_week': date.dayofweek,
                        'month': date.month,
                        'district': district,
                        'primary_type': primary_type,
                        'solve_days': solve_days,
                        'lat': lat,
                        'lon': lon
                    })

            synthetic_df = pd.DataFrame(synthetic_records)
            logger.info(f"Generated {len(synthetic_df)} synthetic complaint records from {days} days of predictions")

            return synthetic_df

        except Exception as e:
            logger.error(f"Error generating synthetic data from forecast: {e}")
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

        # Add historical actual data
        fig.add_trace(go.Scatter(
            x=daily_counts['date'],
            y=daily_counts['count'],
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
            name='ค่าพยากรณ์ (โมเดล)',
            line=dict(color='red', width=2.5),
            marker=dict(size=5, symbol='circle'),
            hovertemplate='วันที่: %{x}<br>ค่าพยากรณ์: %{y:.0f}<extra></extra>'
        ))
    else:
        # Only future predictions if no historical data
        logger.warning("No past predictions available, showing only future predictions")
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted'],
            mode='lines+markers',
            name='ค่าพยากรณ์ (โมเดล)',
            line=dict(color='red', width=2.5),
            marker=dict(size=5, symbol='circle'),
            hovertemplate='วันที่: %{x}<br>ค่าพยากรณ์: %{y:.0f}<extra></extra>'
        ))

    # Add future confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['upper_bound'],
        mode='lines',
        name='ช่วงความเชื่อมั่น',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['lower_bound'],
        mode='lines',
        name='ช่วงความเชื่อมั่น',
        line=dict(width=0),
        fillcolor='rgba(255, 0, 0, 0.15)',
        fill='tonexty',
        showlegend=True,
        hovertemplate='ขอบล่าง: %{y:.0f}<extra></extra>'
    ))

    # Add vertical line to separate past and future
    if historical_actual is not None and len(historical_actual) > 0:
        last_date = historical_actual['date'].max()

        fig.add_shape(
            type="line",
            x0=last_date,
            x1=last_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dash"),
            opacity=0.5
        )

        fig.add_annotation(
            x=last_date,
            y=1,
            yref="paper",
            text="วันนี้",
            showarrow=False,
            yshift=10,
            font=dict(size=10, color="gray")
        )

    # Add title with accuracy info if available
    title_text = 'การพยากรณ์จำนวน Complaint (RandomForest Model)'
    if mape is not None and rmse is not None:
        title_text += f'<br><sub>ความแม่นยำ: MAPE={mape:.1f}%, RMSE={rmse:.0f}</sub>'

    fig.update_layout(
        title=title_text,
        xaxis_title='วันที่',
        yaxis_title='จำนวน Complaint',
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


def plot_anomaly_scatter(df: pd.DataFrame, max_points: int = 2000) -> go.Figure:
    """
    Scatter plot of anomalies with scores

    คำอธิบาย: แสดง complaint ที่ผิดปกติตามแกนเวลา
    ขนาดและสีของจุดแสดงระดับความผิดปกติ
    """
    fig = go.Figure()

    # Always show normal background points (sampled for performance)
    df_normal = df[df['anomaly_score'] <= 0.5].copy()
    if len(df_normal) > 0:
        # Sample normal points for performance
        sample_size = min(len(df_normal), 1000)
        df_normal_sampled = df_normal.sample(n=sample_size, random_state=42)

        fig.add_trace(go.Scatter(
            x=df_normal_sampled['timestamp'],
            y=df_normal_sampled['solve_days'],
            mode='markers',
            marker=dict(
                size=4,
                color='lightgray',
                opacity=0.3
            ),
            name='ปกติ',
            hovertemplate='<b>ปกติ</b><br>วันที่: %{x}<br>ระยะเวลาแก้ปัญหา: %{y} วัน<extra></extra>'
        ))

    # Get anomalous points (score > 0.5)
    df_anomaly = df[df['anomaly_score'] > 0.5].copy()

    # Limit points for performance - stratified sampling to show variety of scores
    if len(df_anomaly) > max_points:
        # Sample across different score ranges to show color variety
        high_scores = df_anomaly[df_anomaly['anomaly_score'] >= 0.85]
        mid_scores = df_anomaly[(df_anomaly['anomaly_score'] >= 0.7) & (df_anomaly['anomaly_score'] < 0.85)]
        low_scores = df_anomaly[(df_anomaly['anomaly_score'] > 0.5) & (df_anomaly['anomaly_score'] < 0.7)]

        # Distribute max_points across ranges proportionally
        n_high = min(len(high_scores), int(max_points * 0.4))
        n_mid = min(len(mid_scores), int(max_points * 0.3))
        n_low = min(len(low_scores), max_points - n_high - n_mid)

        sampled_parts = []
        if n_high > 0 and len(high_scores) > 0:
            sampled_parts.append(high_scores.sample(n=n_high, random_state=42) if len(high_scores) > n_high else high_scores)
        if n_mid > 0 and len(mid_scores) > 0:
            sampled_parts.append(mid_scores.sample(n=n_mid, random_state=42) if len(mid_scores) > n_mid else mid_scores)
        if n_low > 0 and len(low_scores) > 0:
            sampled_parts.append(low_scores.sample(n=n_low, random_state=42) if len(low_scores) > n_low else low_scores)

        if sampled_parts:
            df_anomaly = pd.concat(sampled_parts, ignore_index=True)
        else:
            # Fallback to top anomalies if stratified sampling fails
            df_anomaly = df_anomaly.nlargest(max_points, 'anomaly_score')

    # Always show anomaly trace even if empty (to ensure colorbar appears)
    if len(df_anomaly) > 0:
        fig.add_trace(go.Scatter(
            x=df_anomaly['timestamp'],
            y=df_anomaly['solve_days'],
            mode='markers',
            marker=dict(
                size=df_anomaly['anomaly_score'] * 20,
                color=df_anomaly['anomaly_score'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(
                    title='Anomaly<br>Score',
                    x=1.15,
                    xanchor='left',
                    len=0.7,
                    y=0.5
                ),
                line=dict(width=1, color='white'),
                opacity=0.8,
                cmin=0.5,
                cmax=1.0
            ),
            name='ผิดปกติ',
            text=[f"{row['district']}<br>{row['primary_type']}" for _, row in df_anomaly.iterrows()],
            hovertemplate='<b>%{text}</b><br>วันที่: %{x}<br>ระยะเวลาแก้ปัญหา: %{y} วัน<br>Anomaly Score: %{marker.color:.2f}<extra></extra>'
        ))
    else:
        # Add empty trace with colorbar to ensure it always shows
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='markers',
            marker=dict(
                size=10,
                color=[],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(
                    title='Anomaly<br>Score',
                    x=1.15,
                    xanchor='left',
                    len=0.7,
                    y=0.5
                ),
                cmin=0.5,
                cmax=1.0
            ),
            name='ผิดปกติ'
        ))

    df_plot = df_anomaly

    # Determine y-axis range and ticks
    max_solve_days = df['solve_days'].max()

    # Create tick values at 5-day intervals up to 100
    tick_vals = list(range(0, 101, 5))
    tick_text = [str(i) for i in range(0, 101, 5)]

    # Set y-axis range to cap at 105 to keep consistent spacing
    y_range = [0, min(max_solve_days * 1.05, 105)]

    # If max > 100, add 100+ label at position 105
    if max_solve_days > 100:
        tick_vals.append(105)
        tick_text.append('100+')
        y_range = [0, 110]

    fig.update_layout(
        title=f'การตรวจจับความผิดปกติ (แสดง {len(df_plot):,} รายการที่ผิดปกติ)',
        xaxis_title='วันที่',
        yaxis_title='ระยะเวลาในการแก้ปัญหา (วัน)',
        yaxis=dict(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            range=y_range
        ),
        template='plotly_white',
        height=500,
        hovermode='closest',
        margin=dict(r=200, t=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def plot_anomaly_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Distribution of anomalies by district and type

    คำอธิบาย: แสดงการกระจายของ complaint ที่ผิดปกติแยกตามเขตและประเภท
    """
    anomalies = df[df['is_anomaly'] == 1].copy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Anomalies ตามเขต', 'Anomalies ตามประเภท'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # By district
    district_counts = anomalies['district'].value_counts().head(10)
    fig.add_trace(
        go.Bar(x=district_counts.index, y=district_counts.values,
               marker_color='indianred',
               hovertemplate='<b>%{x}</b><br>จำนวน: %{y}<extra></extra>'),
        row=1, col=1
    )

    # By type
    type_counts = anomalies['primary_type'].value_counts().head(10)
    fig.add_trace(
        go.Bar(x=type_counts.index, y=type_counts.values,
               marker_color='lightcoral',
               hovertemplate='<b>%{x}</b><br>จำนวน: %{y}<extra></extra>'),
        row=1, col=2
    )

    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)

    fig.update_layout(
        title_text='การกระจายของ Anomalies',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig
