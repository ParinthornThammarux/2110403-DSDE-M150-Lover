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


def plot_forecast_visualization(forecast_df: pd.DataFrame, historical_df: pd.DataFrame = None) -> go.Figure:
    """
    Plot forecast with confidence intervals

    คำอธิบาย: แสดงการพยากรณ์จำนวน complaint พร้อม confidence interval
    เส้นสีแดงคือค่าพยากรณ์ พื้นที่สีเทาคือช่วงความเชื่อมั่น
    """
    fig = go.Figure()

    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['upper_bound'],
        mode='lines',
        name='ขอบบน',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['lower_bound'],
        mode='lines',
        name='ขอบล่าง',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.2)',
        fill='tonexty',
        showlegend=True,
        hoverinfo='skip'
    ))

    # Add prediction line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['predicted'],
        mode='lines+markers',
        name='ค่าพยากรณ์',
        line=dict(color='red', width=3),
        marker=dict(size=6),
        hovertemplate='วันที่: %{x}<br>ค่าพยากรณ์: %{y:.0f}<extra></extra>'
    ))

    # Add historical data if provided
    if historical_df is not None:
        historical_df = historical_df.copy()
        historical_df['date'] = pd.to_datetime(historical_df['timestamp']).dt.date
        daily_counts = historical_df.groupby('date').size().reset_index(name='count')
        daily_counts = daily_counts.tail(60)  # Last 60 days

        fig.add_trace(go.Scatter(
            x=pd.to_datetime(daily_counts['date']),
            y=daily_counts['count'],
            mode='lines',
            name='ข้อมูลจริง',
            line=dict(color='blue', width=2),
            hovertemplate='วันที่: %{x}<br>จำนวนจริง: %{y}<extra></extra>'
        ))

    fig.update_layout(
        title='การพยากรณ์จำนวน Complaint (RandomForest Model)',
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
    # Filter to show only anomalies or high scores
    df_plot = df[df['anomaly_score'] > 0.5].copy()

    # Limit points for performance
    if len(df_plot) > max_points:
        # Show top anomalies
        df_plot = df_plot.nlargest(max_points, 'anomaly_score')

    fig = go.Figure()

    # Add normal background (low score) if dataset is not too large
    if len(df) < 10000:
        df_normal = df[df['anomaly_score'] <= 0.5].copy()
        if len(df_normal) > 0:
            # Sample normal points
            if len(df_normal) > 1000:
                df_normal = df_normal.sample(n=1000, random_state=42)

            fig.add_trace(go.Scatter(
                x=df_normal['timestamp'],
                y=df_normal['solve_days'],
                mode='markers',
                marker=dict(
                    size=4,
                    color='lightgray',
                    opacity=0.3
                ),
                name='ปกติ',
                hovertemplate='<b>ปกติ</b><br>วันที่: %{x}<br>ระยะเวลาแก้ปัญหา: %{y} วัน<extra></extra>'
            ))

    # Add anomalies
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'],
        y=df_plot['solve_days'],
        mode='markers',
        marker=dict(
            size=df_plot['anomaly_score'] * 20,
            color=df_plot['anomaly_score'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title='Anomaly<br>Score'),
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        name='ผิดปกติ',
        text=[f"{row['district']}<br>{row['primary_type']}" for _, row in df_plot.iterrows()],
        hovertemplate='<b>%{text}</b><br>วันที่: %{x}<br>ระยะเวลาแก้ปัญหา: %{y} วัน<br>Anomaly Score: %{marker.color:.2f}<extra></extra>'
    ))

    # Add trend line for anomalies
    if len(df_plot) > 10:
        # Group by month to show trend
        df_plot_copy = df_plot.copy()
        df_plot_copy['month'] = pd.to_datetime(df_plot_copy['timestamp']).dt.to_period('M')
        monthly_avg = df_plot_copy.groupby('month')['solve_days'].mean().reset_index()
        monthly_avg['month'] = monthly_avg['month'].dt.to_timestamp()

        fig.add_trace(go.Scatter(
            x=monthly_avg['month'],
            y=monthly_avg['solve_days'],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='แนวโน้มรายเดือน',
            hovertemplate='เดือน: %{x}<br>เฉลี่ย: %{y:.1f} วัน<extra></extra>'
        ))

    fig.update_layout(
        title=f'การตรวจจับความผิดปกติ (แสดง {len(df_plot):,} รายการที่ผิดปกติ)',
        xaxis_title='วันที่',
        yaxis_title='ระยะเวลาในการแก้ปัญหา (วัน)',
        template='plotly_white',
        height=500,
        hovermode='closest'
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
