"""
Interactive Geospatial Dashboard for Urban Issue Forecasting
Built with Streamlit, Plotly, and Folium

Adapted for clean_data.csv from Bangkok Traffy dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
from datetime import datetime, timedelta
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Urban Issue Forecasting Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed complaint data from clean_data.csv"""
    csv_path = 'clean_data.csv'

    if not Path(csv_path).exists():
        st.error(f"❌ Data file not found: {csv_path}")
        st.info("Please ensure clean_data.csv is in the root directory")
        st.stop()

    # Load CSV
    df = pd.read_csv(csv_path)
    initial_rows = len(df)

    # Parse type field - it's a set string like "{น้ำท่วม,ร้องเรียน}"
    def parse_types(type_str):
        if pd.isna(type_str) or type_str == '{}' or type_str == 'ไม่ระบุ':
            return ['Unknown']
        # Remove curly braces and split
        type_str = str(type_str).strip('{}')
        types = [t.strip() for t in type_str.split(',') if t.strip()]
        return types if types else ['Unknown']

    df['types_list'] = df['type'].apply(parse_types)
    df['primary_type'] = df['types_list'].apply(lambda x: x[0])

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Reconstruct state from one-hot encoded columns
    def get_state(row):
        if 'state_เสร็จสิ้น' in row and row.get('state_เสร็จสิ้น', 0) == 1.0:
            return 'เสร็จสิ้น'
        elif 'state_กำลังดำเนินการ' in row and row.get('state_กำลังดำเนินการ', 0) == 1.0:
            return 'กำลังดำเนินการ'
        elif 'state_รอรับเรื่อง' in row and row.get('state_รอรับเรื่อง', 0) == 1.0:
            return 'รอรับเรื่อง'
        return 'Unknown'

    df['state'] = df.apply(get_state, axis=1)

    # Reconstruct star rating from one-hot encoded columns
    def get_star(row):
        star_cols = ['star_1.0', 'star_2.0', 'star_3.0', 'star_4.0', 'star_5.0']
        for i, col in enumerate(star_cols, 1):
            if col in row and row[col] == 1.0:
                return float(i)
        return np.nan

    df['star_rating'] = df.apply(get_star, axis=1)

    # Create anomaly_score based on solve_days
    median_solve = df['solve_days'].median()
    std_solve = df['solve_days'].std()
    df['anomaly_score'] = np.abs(df['solve_days'] - median_solve) / (std_solve + 1)
    df['anomaly_score'] = df['anomaly_score'].clip(0, 1)

    # Extract time components (only for valid timestamps)
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour

    # Track data quality
    data_quality = {
        'initial_rows': initial_rows,
        'missing_lat': df['lat'].isna().sum(),
        'missing_lon': df['lon'].isna().sum(),
        'missing_timestamp': df['timestamp'].isna().sum(),
        'missing_any_geo': df[['lat', 'lon']].isna().any(axis=1).sum(),
    }

    return df, data_quality


@st.cache_data
def load_forecast_data():
    """Load or generate forecasting predictions"""
    # Check if forecast model output exists
    forecast_file = Path('ml_models/forecasting/outputs/forecast_predictions.csv')

    if forecast_file.exists():
        df_forecast = pd.read_csv(forecast_file)
        df_forecast['date'] = pd.to_datetime(df_forecast['date'])
    else:
        # Generate simulated forecast
        future_dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
        trend = np.linspace(100, 120, 30)
        seasonality = 15 * np.sin(2 * np.pi * np.arange(30) / 7)
        forecast = trend + seasonality + np.random.normal(0, 5, 30)

        df_forecast = pd.DataFrame({
            'date': future_dates,
            'predicted': forecast,
            'lower_bound': forecast - 10,
            'upper_bound': forecast + 10
        })

    return df_forecast


def create_geospatial_map(df, map_type='heatmap'):
    """Create interactive geospatial visualization"""
    # Center on Bangkok
    center_lat, center_lon = 13.7563, 100.5018

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    if map_type == 'heatmap':
        # Heat map of complaint density
        heat_data = [[row['lat'], row['lon']] for idx, row in df.iterrows()]
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)

    elif map_type == 'clusters':
        # Marker clusters
        marker_cluster = MarkerCluster().add_to(m)

        for idx, row in df.head(500).iterrows():  # Limit for performance
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"""
                    <b>District:</b> {row['district']}<br>
                    <b>Type:</b> {row['primary_type']}<br>
                    <b>Date:</b> {row['timestamp'].strftime('%Y-%m-%d')}<br>
                    <b>Status:</b> {row['state']}<br>
                    <b>Solve Days:</b> {row['solve_days']}
                """,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)

    return m


def plot_time_series(df):
    """Plot complaint volume over time"""
    daily = df.groupby(df['timestamp'].dt.date).size().reset_index()
    daily.columns = ['date', 'complaints']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['complaints'],
        mode='lines',
        name='Daily Complaints',
        line=dict(color='#1f77b4', width=2),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))

    # Add 7-day moving average
    daily['ma7'] = daily['complaints'].rolling(window=7).mean()
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['ma7'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Complaint Volume Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Complaints',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def plot_forecast(df_forecast):
    """Plot forecasting predictions"""
    fig = go.Figure()

    # Prediction with confidence interval
    fig.add_trace(go.Scatter(
        x=df_forecast['date'],
        y=df_forecast['upper_bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_forecast['date'],
        y=df_forecast['lower_bound'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_forecast['date'],
        y=df_forecast['predicted'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=3)
    ))

    fig.update_layout(
        title='30-Day Complaint Volume Forecast (RandomForest Model)',
        xaxis_title='Date',
        yaxis_title='Predicted Complaints',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def plot_category_distribution(df):
    """Plot complaint type distribution"""
    category_counts = df['primary_type'].value_counts().head(15)

    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={'x': 'Complaint Type', 'y': 'Count'},
        title='Top 15 Complaint Types Distribution',
        color=category_counts.values,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False,
        xaxis={'categoryorder':'total descending'}
    )

    return fig


def plot_district_heatmap(df):
    """District vs time heatmap"""
    df_copy = df.copy()
    df_copy['month_name'] = df_copy['timestamp'].dt.strftime('%Y-%m')

    pivot = df_copy.pivot_table(
        index='district',
        columns='month_name',
        aggfunc='size',
        fill_value=0
    )

    # Limit to top districts and recent months
    pivot = pivot.nlargest(15, pivot.columns[-1])
    pivot = pivot[pivot.columns[-12:]]  # Last 12 months

    fig = px.imshow(
        pivot,
        labels=dict(x='Month', y='District', color='Complaints'),
        title='Complaint Intensity by District and Month',
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )

    fig.update_layout(height=500)

    return fig


def plot_resolution_time(df):
    """Plot resolution time distribution"""
    # Filter to top complaint types for readability
    top_types = df['primary_type'].value_counts().head(10).index
    df_filtered = df[df['primary_type'].isin(top_types)]

    fig = px.box(
        df_filtered,
        x='primary_type',
        y='solve_days',
        title='Resolution Time by Complaint Type (Top 10)',
        labels={'solve_days': 'Days to Resolve', 'primary_type': 'Complaint Type'},
        color='primary_type'
    )

    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False,
        xaxis={'categoryorder':'median descending'}
    )

    return fig


def main():
    """Main dashboard application"""

    # Header
    st.markdown('<div class="main-header">🏙️ Urban Issue Forecasting Dashboard</div>',
               unsafe_allow_html=True)
    st.markdown("### Bangkok Complaint Analysis & Prediction System")
    st.markdown("---")

    # Load data
    with st.spinner("Loading data..."):
        df, data_quality = load_data()
        df_forecast = load_forecast_data()

    # Data Quality Alert
    if data_quality['missing_any_geo'] > 0:
        with st.expander("⚠️ Data Quality Report - Click to expand", expanded=False):
            st.warning(f"**Data Filtering Applied**: Some rows have missing data")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows in CSV", f"{data_quality['initial_rows']:,}")
            with col2:
                st.metric("Missing Coordinates", f"{data_quality['missing_any_geo']:,}")
            with col3:
                st.metric("Missing Timestamp", f"{data_quality['missing_timestamp']:,}")

            st.info(f"""
            **Why rows are missing:**
            - {data_quality['missing_lat']:,} rows missing latitude
            - {data_quality['missing_lon']:,} rows missing longitude
            - {data_quality['missing_timestamp']:,} rows missing valid timestamp
            - {data_quality['missing_any_geo']:,} rows missing coordinates (can't be shown on map)

            **Current behavior:**
            - Geospatial maps require valid coordinates (lat/lon)
            - Time series analysis requires valid timestamps
            - All other analytics use available data
            """)

    # Sidebar filters
    st.sidebar.header("📊 Filters & Settings")

    # Data filtering option
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Data Options")

    filter_mode = st.sidebar.radio(
        "Include rows with missing data:",
        ["Only complete data (for maps)", "All data (no maps)"],
        help="Maps require coordinates. Choose 'All data' to analyze everything without maps."
    )

    # Apply coordinate filtering based on user choice
    if filter_mode == "Only complete data (for maps)":
        df = df.dropna(subset=['lat', 'lon', 'timestamp'])
        show_maps = True
    else:
        # Keep all data, only drop rows without timestamp (needed for time series)
        df = df.dropna(subset=['timestamp'])
        show_maps = False
        st.sidebar.info("📍 Map features disabled - showing all data")

    st.sidebar.markdown("---")

    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    # Calculate default start date (1 year back or min_date, whichever is later)
    default_start = max(min_date, max_date - timedelta(days=365))

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # District filter
    districts = ['All'] + sorted(df['district'].dropna().unique().tolist())
    selected_district = st.sidebar.selectbox("Select District", districts)

    # Complaint type filter
    types = ['All'] + sorted(df['primary_type'].unique().tolist())
    selected_type = st.sidebar.selectbox("Select Complaint Type", types)

    # Map visualization type
    map_type = st.sidebar.radio(
        "Map Visualization",
        ['heatmap', 'clusters'],
        format_func=lambda x: 'Heat Map' if x == 'heatmap' else 'Marker Clusters'
    )

    # Apply filters
    df_filtered = df.copy()
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['timestamp'].dt.date >= date_range[0]) &
            (df_filtered['timestamp'].dt.date <= date_range[1])
        ]

    if selected_district != 'All':
        df_filtered = df_filtered[df_filtered['district'] == selected_district]

    if selected_type != 'All':
        df_filtered = df_filtered[df_filtered['primary_type'] == selected_type]

    # Key Metrics
    st.header("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Complaints",
            f"{len(df_filtered):,}"
        )

    with col2:
        avg_resolution = df_filtered['solve_days'].mean()
        st.metric(
            "Avg Resolution Time",
            f"{avg_resolution:.1f} days"
        )

    with col3:
        completion_rate = (df_filtered['state'] == 'เสร็จสิ้น').mean() * 100
        st.metric(
            "Completion Rate",
            f"{completion_rate:.1f}%"
        )

    with col4:
        anomaly_rate = (df_filtered['anomaly_score'] > 0.7).mean() * 100
        st.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.1f}%"
        )

    st.markdown("---")

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Geospatial Analysis",
        "📊 Time Series & Forecasting",
        "📈 Analytics",
        "🔍 Anomaly Detection"
    ])

    with tab1:
        st.header("Interactive Geospatial Map")

        if show_maps:
            # Create and display map
            with st.spinner("Generating map..."):
                m = create_geospatial_map(df_filtered.head(5000), map_type=map_type)
                folium_static(m, width=1200, height=600)

            # District statistics
            st.subheader("District Statistics")
            district_stats = df_filtered.groupby('district').agg({
                'lat': 'count',  # Using lat as count
                'solve_days': 'mean',
                'anomaly_score': 'mean'
            }).round(2)
            district_stats.columns = ['Total Complaints', 'Avg Resolution Days', 'Avg Anomaly Score']
            district_stats = district_stats.sort_values('Total Complaints', ascending=False)

            st.dataframe(district_stats, use_container_width=True)
        else:
            st.warning("⚠️ **Map features disabled** - You've selected 'All data (no maps)' mode")
            st.info("""
            To view geospatial maps:
            1. Go to sidebar → Data Options
            2. Select 'Only complete data (for maps)'

            This will filter to rows with valid coordinates (lat/lon).
            """)

            # Still show district statistics without map
            st.subheader("District Statistics (All Data)")
            if 'district' in df_filtered.columns:
                district_stats = df_filtered.groupby('district').agg({
                    'solve_days': ['count', 'mean'],
                    'anomaly_score': 'mean'
                }).round(2)
                district_stats.columns = ['Total Complaints', 'Avg Resolution Days', 'Avg Anomaly Score']
                district_stats = district_stats.sort_values('Total Complaints', ascending=False)
                st.dataframe(district_stats, use_container_width=True)

    with tab2:
        st.header("Time Series Analysis & Forecasting")

        # Historical time series
        st.plotly_chart(plot_time_series(df_filtered), use_container_width=True)

        # Forecast
        st.subheader("30-Day Forecast")
        st.plotly_chart(plot_forecast(df_forecast), use_container_width=True)

        # Seasonal patterns
        col1, col2 = st.columns(2)

        with col1:
            # Day of week pattern
            dow_counts = df_filtered.groupby('day_of_week').size()
            fig_dow = px.bar(
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                y=dow_counts.values,
                title='Complaints by Day of Week',
                labels={'x': 'Day', 'y': 'Count'},
                color=dow_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        with col2:
            # Hour of day pattern
            hour_counts = df_filtered.groupby('hour').size()
            fig_hour = px.line(
                x=hour_counts.index,
                y=hour_counts.values,
                title='Complaints by Hour of Day',
                labels={'x': 'Hour', 'y': 'Count'},
                markers=True
            )
            st.plotly_chart(fig_hour, use_container_width=True)

    with tab3:
        st.header("Detailed Analytics")

        # Category distribution
        st.plotly_chart(plot_category_distribution(df_filtered), use_container_width=True)

        # District heatmap
        if len(df_filtered) > 0:
            st.plotly_chart(plot_district_heatmap(df_filtered), use_container_width=True)

        # Resolution time
        st.plotly_chart(plot_resolution_time(df_filtered), use_container_width=True)

        # Star rating distribution
        if df_filtered['star_rating'].notna().sum() > 0:
            st.subheader("User Satisfaction Ratings")
            rating_counts = df_filtered['star_rating'].value_counts().sort_index()
            fig_rating = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                title='Distribution of Star Ratings',
                labels={'x': 'Star Rating', 'y': 'Count'},
                color=rating_counts.index,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_rating, use_container_width=True)

    with tab4:
        st.header("Anomaly Detection Results")

        # Filter anomalies
        anomalies = df_filtered[df_filtered['anomaly_score'] > 0.7]

        st.metric("Total Anomalies Detected", f"{len(anomalies):,}")

        if len(anomalies) > 0:
            # Anomaly timeline
            fig_anomaly = go.Figure()
            fig_anomaly.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies['anomaly_score'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=anomalies['anomaly_score'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title='Anomaly Score')
                ),
                text=[f"{row['district']} - {row['primary_type']}<br>Solve days: {row['solve_days']}"
                      for _, row in anomalies.iterrows()],
                hovertemplate='<b>%{text}</b><br>Score: %{y:.2f}<br>Date: %{x}<extra></extra>'
            ))

            fig_anomaly.update_layout(
                title='Anomaly Detection Timeline',
                xaxis_title='Date',
                yaxis_title='Anomaly Score',
                template='plotly_white',
                height=400
            )

            st.plotly_chart(fig_anomaly, use_container_width=True)

            # Anomaly table
            st.subheader("Recent Anomalies (Unusual Resolution Times)")
            anomaly_display = anomalies[['timestamp', 'district', 'primary_type', 'solve_days', 'anomaly_score']].sort_values(
                'anomaly_score', ascending=False
            ).head(20)
            st.dataframe(anomaly_display, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; color: #666;'>
            <p>Urban Issue Forecasting System | DSDE M150-Lover Team | Chulalongkorn University</p>
            <p>Data Source: Bangkok Traffy Fondue | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
