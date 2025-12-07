"""
Main Streamlit Dashboard - Urban Issue Forecasting System
Bangkok Traffy Complaint Analysis & Prediction

ระบบวิเคราะห์และพยากรณ์ปัญหาในเขตกรุงเทพมหานคร
รวม ML models: RandomForest Forecaster, Isolation Forest Anomaly Detector, และ K-Means Outage Clustering

Run: streamlit run visualization/dashboard/main_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import pydeck as pdk
from viz_modules import plot_complaint_timeseries, plot_top_complaint_types
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from viz_modules import (
    plot_complaints_by_district,
    plot_complaint_distribution_across_districts,
    plot_top_complaint_districts,
    plot_top_complaint_types,
    plot_time_series_comparison,
    plot_hourly_pattern,
    plot_weekday_pattern,
)

from ml_integration import (
    MLModelIntegrator,
    plot_forecast_visualization,
    plot_anomaly_scatter,
    plot_anomaly_distribution
)

from outage_viz import (
    plot_cluster_distribution,
    plot_cluster_by_time,
    plot_cluster_characteristics,
    plot_cluster_by_district,
    plot_cluster_by_day,
    plot_cluster_weather_correlation,
    render_cluster_summary,
    prepare_outage_dataframe,
    plot_outage_duration_by_district,
    # plot_outage_timeline
)

# Page configuration
st.set_page_config(
    page_title="Urban Issue Dashboard - Bangkok",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: rgba(31, 119, 180, 0.1);
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
        color: inherit;
    }
    .info-box b {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    """
    Loading complaint data from clean_data.csv
    """
    csv_path = Path("../data/clean_data_sampled.csv")

    if not csv_path.exists():
        st.error(f"File not found: {csv_path}")
        st.info("Please place clean_data.csv in the root directory")
        st.stop()

    # Load CSV
    df = pd.read_csv(csv_path)

    st.sidebar.info(f"Loaded data: {len(df):,} rows")

    # Parse type field
    def parse_types(type_str):
        if pd.isna(type_str) or type_str == '{}' or type_str == 'ไม่ระบุ':
            return ['Unknown']
        type_str = str(type_str).strip('{}')
        types = [t.strip() for t in type_str.split(',') if t.strip()]
        return types if types else ['Unknown']

    df['types_list'] = df['type'].apply(parse_types)
    df['primary_type'] = df['types_list'].apply(lambda x: x[0])

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Reconstruct state from one-hot encoding
    def get_state(row):
        if 'state_Completed' in row and row.get('state_Completed    ', 0) == 1.0:
            return 'Completed'
        elif 'state_In Progress' in row and row.get('state_In Progress', 0) == 1.0:
            return 'In Progress'
        elif 'state_Pending' in row and row.get('state_Pending', 0) == 1.0:
            return 'Pending'
        return 'Unknown'

    df['state'] = df.apply(get_state, axis=1)

    # Reconstruct star rating from one-hot encoding
    def get_star(row):
        star_cols = ['star_1.0', 'star_2.0', 'star_3.0', 'star_4.0', 'star_5.0']
        for i, col in enumerate(star_cols, 1):
            if col in row and row[col] == 1.0:
                return float(i)
        return np.nan

    df['star_rating'] = df.apply(get_star, axis=1)

    # Extract time components
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour

    # Drop rows with missing critical data
    df = df.dropna(subset=['lat', 'lon', 'timestamp'])

    return df


@st.cache_data(ttl=3600)
def load_mea_outage_data():
    """
    Loading MEA power outage data from clean_scraping_data.csv
    """
    csv_path = Path("../data/clean_scraping_data.csv")

    if not csv_path.exists():
        return None

    # Load CSV
    df = pd.read_csv(csv_path)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return df

@st.cache_resource
def load_ml_models():
    """
    โหลด ML models ทั้งหมด

    คำอธิบาย: โหลดโมเดล RandomForest สำหรับพยากรณ์
    และ Isolation Forest สำหรับตรวจจับความผิดปกติ
    """
    integrator = MLModelIntegrator()

    # Load forecasting model
    rf_loaded = integrator.load_forecasting_model()

    # Load anomaly detection model
    anomaly_loaded = integrator.load_anomaly_model()

    # Load outage clustering model
    outage_loaded = integrator.load_outage_model()

    status_msg = []
    if rf_loaded:
        status_msg.append("\n[OK] RandomForest Forecaster (New Model)")
    else:
        status_msg.append("\n[ERROR] RandomForest Forecaster - MODEL REQUIRED")
        st.sidebar.error("\nWARNING: Forecasting model not found! Please train the model first.")

    if anomaly_loaded:
        status_msg.append("\n[OK] Isolation Forest Anomaly Detector")
    else:
        status_msg.append("\n[WARNING] Anomaly Detector (Model not found)")

    if outage_loaded:
        status_msg.append("\n[OK] K-Means Outage Clustering")
    else:
        status_msg.append("\n[WARNING] Outage Clustering (Model not found)")

    st.sidebar.info("ML Models Status:\n" + "\n".join(status_msg))

    return integrator


def create_geospatial_map(df, map_type='heatmap'):
    """
    สร้างแผนที่แสดงตำแหน่ง complaint

    คำอธิบาย: แสดงการกระจายตัวของ complaint บนแผนที่กรุงเทพ
    - Heatmap: แสดงความหนาแน่นของปัญหา
    - Clusters: แสดงจุดแต่ละ complaint พร้อมรายละเอียด
    """
    center_lat, center_lon = 13.7563, 100.5018

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    if map_type == 'heatmap':
        heat_data = [[row['lat'], row['lon']] for idx, row in df.head(10000).iterrows()]
        HeatMap(heat_data, 
            radius=20,           # Increase circle size
            blur=30,             # Increase blur
            max_zoom=15,         # Allow zooming in more before hiding
            min_opacity=0.3,     # Minimum transparency (0-1)
            gradient={0.2: 'blue', 0.3: 'green', 0.6: 'yellow', 0.7: 'orange', 1.0: 'red'}  # Custom color gradient
        ).add_to(m)

    elif map_type == 'clusters':
        marker_cluster = MarkerCluster().add_to(m)

        for idx, row in df.head(1000).iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"""
                    <b>เขต:</b> {row['district']}<br>
                    <b>ประเภท:</b> {row['primary_type']}<br>
                    <b>วันที่:</b> {row['timestamp'].strftime('%Y-%m-%d')}<br>
                    <b>สถานะ:</b> {row['state']}<br>
                    <b>ระยะเวลาแก้:</b> {row['solve_days']} วัน
                """,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)
            
    elif map_type == 'GridLayer':
        # Create pydeck GridLayer
        grid_data = df[['lat', 'lon']].head(10000).copy()
        grid_data.columns = ['latitude', 'longitude']
        
        grid_layer = pdk.Layer(
            "GridLayer",
            data=grid_data,
            get_position='[longitude, latitude]',
            cell_size=100,
            elevation_scale=20,
            extruded=True,
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=12,
            pitch=45,
        )

        deck = pdk.Deck(
            layers=[grid_layer], 
            initial_view_state=view_state,
            tooltip={'text': 'Cell count: {count}'}
        )
        return deck

    return m

def main():
    """Main dashboard application"""

    # Header
    st.markdown('<div class="main-header">Urban Issue Forecasting Dashboard</div>',
               unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ระบบวิเคราะห์และพยากรณ์ปัญหาเขตกรุงเทพมหานคร | Bangkok Traffy Data Analysis</div>',
               unsafe_allow_html=True)
    st.markdown("---")

    # Load data and models
    with st.spinner("กำลังโหลดข้อมูลและ ML models..."):
        df = load_data()
        df_mea_outage = load_mea_outage_data()
        ml_integrator = load_ml_models()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    date_range = st.sidebar.date_input(
        "เลือกช่วงเวลา",
        value=(date(2024, 10    , 1), max_date),
        min_value=min_date,
        max_value=max_date
    )

    # District filter
    districts = ['All'] + sorted(df['district'].dropna().unique().tolist())
    selected_district = st.sidebar.selectbox("เลือกเขต", districts)

    # Complaint type filter
    types = ['All'] + sorted(df['primary_type'].unique().tolist())
    selected_type = st.sidebar.selectbox("เลือกประเภท Complaint", types)

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
    st.header("ตัวชี้วัดหลัก (Key Metrics)")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "จำนวน Complaint ทั้งหมด",
            f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df):.0f}" if selected_district != 'All' or selected_type != 'All' else None
        )

    with col2:
        avg_resolution = df_filtered['solve_days'].mean()
        st.metric(
            "เวลาแก้ปัญหาเฉลี่ย",
            f"{avg_resolution:.1f} วัน"
        )

    with col3:
        completion_rate = (df_filtered['state'] == 'เสร็จสิ้น').mean() * 100
        st.metric(
            "อัตราการแก้ไขเสร็จสิ้น",
            f"{completion_rate:.1f}%"
        )

    with col4:
        unique_districts = df_filtered['district'].nunique()
        st.metric(
            "จำนวนเขต",
            f"{unique_districts}"
        )

    with col5:
        unique_types = df_filtered['primary_type'].nunique()
        st.metric(
            "ประเภทปัญหา",
            f"{unique_types}"
        )

    st.markdown("---")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Geospatial Map",
        "District and Type Analysis",
        "MEA power outage",
        "ML: Predictive Forecasting",
        "ML: Anomaly Detection",
        "ML: Power Outage Clustering",
        "Additional Analysis"
    ])

    # Tab 1: Geospatial Analysis
    with tab1:
        st.header("Geospatial Analysis")

        st.markdown("""
        <div class="info-box">
        <b>คำอธิบาย:</b> แผนที่แสดงการกระจายตัวของ complaint ในกรุงเทพมหานคร<br>
        - <b>Heat Map:</b> แสดงความหนาแน่นของปัญหาในแต่ละพื้นที่<br>
        - <b>Marker Clusters:</b> แสดงรายละเอียดของแต่ละ complaint
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Truffy Complaint Map")
        # Map visualization type
        map_type = st.radio(
            "Choose map type",
            ['heatmap', 'clusters', 'GridLayer'],
            format_func=lambda x: 'Heat Map (ความหนาแน่น)' if x == 'heatmap' else 'Marker Clusters (จุดแต่ละรายการ)' if x == 'clusters' else 'Grid Layer (3D Grid Visualization)'
        )
        st.markdown("---")

        with st.spinner("Loading map..."):
            if map_type == 'GridLayer':
                result = create_geospatial_map(df_filtered, map_type=map_type)
                st.pydeck_chart(result)
            else:
                m = create_geospatial_map(df_filtered, map_type=map_type)
                folium_static(m, width=1400, height=600)

        # District statistics table
        st.subheader("District Statistics")
        district_stats = df_filtered.groupby('district').agg({
            'lat': 'count',
            'solve_days': 'mean',
        }).round(2)
        district_stats.columns = ['Number of Complaints', 'Average Resolution Time (days)']
        district_stats = district_stats.sort_values('Number of Complaints', ascending=False)

        st.dataframe(district_stats, use_container_width=True, height=400)

    # Tab 2: District and Type Analysis
    with tab2:
        st.header("District and Type Analysis")

        # 1.) Top districts
        st.subheader("Top Districts by Number of Complaints")
        st.markdown("""
        <div class="info-box">
        <b>คำอธิบาย:</b> จัดอันดับเขตที่มีจำนวน complaint สูงสุด ช่วยระบุพื้นที่ที่ต้องให้ความสนใจเป็นพิเศษ
        </div>
        """, unsafe_allow_html=True)

        top_n = st.slider("จำนวนเขตที่ต้องการแสดง", 5, 30, 15, key="top_districts")
        st.subheader(f"Top {top_n} Districts by Number of Complaints")
        st.plotly_chart(plot_top_complaint_districts(df_filtered, top_n), use_container_width=True)
        
        # 2.) Complaints by district
        st.subheader("Complaints by District")
        st.markdown("""
        <div class="info-box">
        <b>คำอธิบาย:</b> แสดงการกระจายของ complaint แต่ละประเภทในแต่ละเขต
        ช่วยให้เห็นว่าแต่ละเขตมีปัญหาประเภทใดบ้าง
        </div>
        """, unsafe_allow_html=True)

        col_filter1, col_spacer1 = st.columns([2, 3])
        with col_filter1:
            complaint_filter_1 = st.selectbox(
                "กรองตามประเภท Complaint",
                ['All'] + sorted(df_filtered['primary_type'].unique().tolist()),
                key="complaint_by_district"
            )

        st.plotly_chart(plot_complaints_by_district(df_filtered, complaint_filter_1), use_container_width=True)

        st.markdown("---")
        
        
        # 3.) Additional visualizations
        st.subheader("Top Complaint Types")
        st.plotly_chart(plot_top_complaint_types(df_filtered, top_n=15), use_container_width=True)


        # 4.) Complaint distribution across districts
        st.subheader("Complaint Distribution Across Districts")
        st.markdown("""
        <div class="info-box">
        <b>คำอธิบาย:</b> แสดงว่า complaint แต่ละประเภทเกิดขึ้นในเขตใดบ้าง
        ช่วยระบุรูปแบบการกระจายของปัญหาแต่ละประเภท
        </div>
        """, unsafe_allow_html=True)

        col_filter2, col_spacer2 = st.columns([2, 3])
        with col_filter2:
            district_filter_1 = st.selectbox(
                "กรองตามเขต",
                ['All'] + sorted(df_filtered['district'].dropna().unique().tolist()),
                key="complaint_distribution"
            )

        st.plotly_chart(plot_complaint_distribution_across_districts(df_filtered, district_filter_1), use_container_width=True)

        st.markdown("---")

        # 5.) Time series: complaints over time with filters
        st.subheader("Time Series: จำนวน Complaint ตามเวลา")

        st.markdown("""
        <div class="info-box">
        <b>คำอธิบาย:</b> แสดงจำนวน complaint ต่อวัน โดยสามารถเลือกช่วงเวลาและจังหวัดได้ 
        เพื่อดูแนวโน้มการเกิดปัญหาในช่วงต่าง ๆ
        </div>
        """, unsafe_allow_html=True)

        # Ensure timestamp/date are in proper format
        df_ts = df_filtered.copy()
        if "timestamp" in df_ts.columns:
            # Always try to convert to datetime, safe even if already datetime
            df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"], errors="coerce")
            min_date = df_ts["timestamp"].dt.date.min()
            max_date = df_ts["timestamp"].dt.date.max()
        else:
            df_ts["date"] = pd.to_datetime(df_ts["date"], errors="coerce")
            min_date = df_ts["date"].dt.date.min()
            max_date = df_ts["date"].dt.date.max()


        # UI controls for time series (in main area, not sidebar)
        col1, col2 = st.columns(2)

        with col1:
            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

        with col2:
            if "province" in df_ts.columns:
                province_options = sorted(df_ts["province"].dropna().unique())
                selected_provinces = st.multiselect(
                    "Select provinces",
                    options=province_options,
                    default=province_options  # show all by default
                )
            else:
                selected_provinces = None

        # Apply filters
        start_date, end_date = date_range
        if "timestamp" in df_ts.columns:
            df_ts = df_ts[
                (df_ts["timestamp"].dt.date >= start_date)
                & (df_ts["timestamp"].dt.date <= end_date)
            ]
        else:
            df_ts = df_ts[
                (df_ts["date"].dt.date >= start_date)
                & (df_ts["date"].dt.date <= end_date)
            ]

        if selected_provinces is not None and len(selected_provinces) > 0:
            df_ts = df_ts[df_ts["province"].isin(selected_provinces)]

        if df_ts.empty:
            st.warning("ไม่มีข้อมูลในช่วงวันที่และจังหวัดที่เลือก")
        else:
                st.plotly_chart(plot_complaint_timeseries(df_ts), use_container_width=True)

    # Tab 3: MEA power outage
    with tab3:
        st.header("⚡ Outage Slots Visualization")

        st.markdown("""
        <div class="info-box">
        <b>คำอธิบาย:</b> แท็บนี้แสดงข้อมูลช่วงเวลาไฟดับตามตัวอย่างข้อมูล 
        โดยแสดงทั้งระยะเวลาไฟดับรวมในแต่ละเขต และไทม์ไลน์ของช่วงเวลาที่ไฟดับในแต่ละวัน
        </div>
        """, unsafe_allow_html=True)

        df_outage = df_mea_outage

        # Prepare dataframe (add start_dt, end_dt)
        df_outage_prepared = prepare_outage_dataframe(df_outage)

        # Bar chart: total duration by district
        st.subheader("Total outage duration by district")
        st.plotly_chart(
            plot_outage_duration_by_district(df_outage_prepared),
            use_container_width=True
        )

    # Tab 4: Forecasting
    with tab4:
        st.header("Predictive Modeling: Number of Complaints")

        st.markdown("""
        <div class="info-box">
        <b>คำอธิบาย:</b> ใช้โมเดล RandomForest ในการพยากรณ์จำนวน complaint<br>
        - <b>เส้นสีน้ำเงิน:</b> ข้อมูลจริง<br>
        - <b>เส้นสีแดง:</b> ค่าพยากรณ์ (ทั้งอดีตและอนาคต - เปรียบเทียบความแม่นยำและดูอนาคต)<br>
        - <b>พื้นที่สีเทา:</b> ช่วงความเชื่อมั่นสำหรับอนาคต (Confidence Interval)<br>
        - <b>เส้นประสีเทา:</b> แบ่งระหว่างอดีตและอนาคต (วันนี้)
        </div>
        """, unsafe_allow_html=True)

        forecast_days = st.slider("Number of days to predict", 7, 60, 30)

        # Prepare session_state for storing forecast results
        if "forecast_df" not in st.session_state:
            st.session_state["forecast_df"] = None
        if "forecast_days_used" not in st.session_state:
            st.session_state["forecast_days_used"] = None
    
        # Click button to run forecast
        run_forecast = st.button("Run forecast / Update prediction")
        if run_forecast:
            if df_filtered.empty:
                st.warning("No data available for forecasting with the current filters.")
            else:
                with st.spinner("Loading forecast data..."):
                    # Heavy computation: run ML model here only when button is pressed
                    forecast_df = ml_integrator.generate_forecast(
                        df_filtered,
                        days_ahead=forecast_days
                    )
                    # Save results to session_state
                    st.session_state["forecast_df"] = forecast_df
                    st.session_state["forecast_days_used"] = forecast_days
    
        # 4) Show forecast if available in session_state
        if st.session_state["forecast_df"] is not None:
            forecast_df = st.session_state["forecast_df"]

            st.caption(
                f"Display last run (days_ahead = "
                f"{st.session_state['forecast_days_used']} days)"
            )

            # Plot forecast visualization
            st.plotly_chart(
                plot_forecast_visualization(
                    forecast_df,
                    df_filtered,
                    ml_integrator=ml_integrator
                ),
                use_container_width=True
            )

            # Forecast statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "mean predicted",
                    f"{forecast_df['predicted'].mean():.0f} complaints/day"
                )

            with col2:
                st.metric(
                    "max predicted",
                    f"{forecast_df['predicted'].max():.0f} complaints/day"
                )

            with col3:
                st.metric(
                    "min predicted",
                    f"{forecast_df['predicted'].min():.0f} complaints/day"
                )

            # Show forecast data
            with st.expander("see forecast data table"):
                forecast_display = forecast_df.copy()
                forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
                forecast_display.columns = ['วันที่', 'ค่าพยากรณ์', 'ขอบล่าง', 'ขอบบน']
                st.dataframe(forecast_display, use_container_width=True, height=400)

        else:
            st.info("Please 'Run forecast / Update prediction'")

    # Tab 5: Anomaly Detection
    with tab5:
        st.header("การตรวจจับความผิดปกติด้วย Machine Learning")

        st.markdown("""
        <div class="info-box">
        <b>คำอธิบาย:</b> ใช้โมเดล Isolation Forest ในการตรวจจับ complaint ที่มีพฤติกรรมผิดปกติ<br>
        <b>ข้อมูลที่ใช้:</b> ข้อมูลจริงจาก clean_data.csv<br>
        <b>โมเดล:</b> IsolationForest <br>
        Anomaly Score สูง = ผิดปกติมาก (เช่น ใช้เวลาแก้ไขนานผิดปกติ หรือเกิดในพื้นที่/เวลาที่ผิดปกติ)
        </div>
        """, unsafe_allow_html=True)

        # Settings for data sampling
        st.markdown("##### settings for data sampling")

        col_setting1, col_setting2 = st.columns([2, 3])

        # Use filtered data from main dashboard
        df_for_anomaly = df_filtered.copy()
        total_data = len(df_for_anomaly)

        with col_setting1:
            # Allow sampling if dataset is large
            if total_data > 50000:
                sample_percentage = st.slider(
                    "Percentage of data to sample for anomaly detection",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=10,
                    format="%d%%",
                    help="ลดขนาดข้อมูลเพื่อลดเวลาในการประมวลผล",
                )
                sample_size = int(total_data * sample_percentage / 100)
                sample_size = max(5000, sample_size)
                df_for_anomaly = df_for_anomaly.sample(n=sample_size, random_state=42).copy()
            else:
                sample_percentage = 100

        with col_setting2:
            st.info(f"ใช้ข้อมูลจริง {len(df_for_anomaly):,} รายการ จาก clean_data.csv")

        # Detect anomalies
        st.markdown("---")
        progress_text = "Loading anomaly detection model..."
        progress_bar = st.progress(0, text=progress_text)

        @st.cache_data(ttl=3600, show_spinner=False)
        def detect_anomalies_cached(_ml_int, data_hash, size):
            return _ml_int.detect_anomalies(df_for_anomaly)

        try:
            progress_bar.progress(30, text="Preparing features...")

            # Create hash based on data
            data_hash = hash(str(len(df_for_anomaly)) + str(df_for_anomaly['timestamp'].min()) + str(df_for_anomaly['timestamp'].max()))

            progress_bar.progress(70, text="Processing with Isolation Forest model...")
            df_with_anomalies = detect_anomalies_cached(ml_integrator, data_hash, len(df_for_anomaly))

            progress_bar.progress(100, text="Completed!")
            progress_bar.empty()
        except Exception as e:
            progress_bar.empty()
            st.error(f"Error during anomaly detection: {str(e)}")
            st.stop()

        # Anomaly statistics
        total_anomalies = df_with_anomalies['is_anomaly'].sum()
        anomaly_rate = (total_anomalies / len(df_with_anomalies)) * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Number of Anomalies Detected",
                f"{total_anomalies:,}"
            )

        with col2:
            st.metric(
                "Anomaly Rate",
                f"{anomaly_rate:.2f}%"
            )

        with col3:
            if total_anomalies > 0:
                avg_anomaly_score = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]['anomaly_score'].mean()
                st.metric(
                    "Average Anomaly Score",
                    f"{avg_anomaly_score:.2f}"
                )
            else:
                st.metric(
                    "Average Anomaly Score",
                    "N/A"
                )

        # Data source info
        st.info(f"**Data Source:** Actual data from clean_data.csv ({len(df_with_anomalies):,} records)")

        # Anomaly scatter plot
        st.subheader("Anomaly Detection Timeline")
        st.plotly_chart(plot_anomaly_scatter(df_with_anomalies), use_container_width=True)

        # Anomaly distribution
        st.subheader("Anomaly Distribution by Type and District")
        st.plotly_chart(plot_anomaly_distribution(df_with_anomalies), use_container_width=True)

        # Anomaly table
        st.subheader("Detected Anomalies (Top 50)")
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1].copy()

        if len(anomalies) > 0:
            anomalies_display = anomalies[['timestamp', 'district', 'primary_type', 'solve_days', 'anomaly_score']].sort_values(
                'anomaly_score', ascending=False
            ).head(50)

            anomalies_display.columns = ['วันที่', 'เขต', 'ประเภท', 'ระยะเวลาแก้ (วัน)', 'Anomaly Score']
            st.dataframe(anomalies_display, use_container_width=True, height=400)
        else:
            st.info("ไม่พบความผิดปกติในข้อมูลที่เลือก")

    # Tab 6: Outage Clustering
    with tab6:
        st.header("K-Means Clustering: การจัดกลุ่มเหตุการณ์ไฟดับ")

        st.markdown("""
        <div class="info-box">
        <b>คำอธิบาย:</b> ใช้โมเดล K-Means ในการจัดกลุ่มเหตุการณ์ไฟดับตามพฤติกรรมที่คล้ายกัน<br>
        <b>ข้อมูลที่ใช้:</b> MEA <br>
        <b>Model:</b> K-Means Clustering <br>
        <b>Features:</b> วันในสัปดาห์, เขต, อุณหภูมิ, ปริมาณฝน, ความเร็วลม, เวลาเริ่ม, ระยะเวลา
        </div>
        """, unsafe_allow_html=True)

        if ml_integrator.outage_model is None:
            st.warning("WARNING: K-Means Clustering model is not available")
            st.info("Please train the model by running: `ml_models/outage_model/train_outage_model.py`")
        else:
            # Load outage data with clusters
            outage_data_path = Path("../ml_models/outage_model/models/power_outage_with_clusters.csv")

            if not outage_data_path.exists():
                st.error(f"Cluster data file not found: {outage_data_path}")
                st.info("Please train the model first to generate the cluster data file")
            else:
                with st.spinner("Loading cluster data..."):
                    df_outage = pd.read_csv(outage_data_path)

                st.success(f"Loaded data successfully: {len(df_outage):,} power outage events")

                # Show key metrics
                st.markdown("### Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Number of Clusters", f"{df_outage['cluster'].nunique()}")

                with col2:
                    avg_duration = df_outage['duration'].mean()
                    st.metric("Average Duration", f"{avg_duration:.0f} minutes")
                with col3:
                    total_outages = len(df_outage)
                    st.metric("Total Outages", f"{total_outages:,}")

                with col4:
                    unique_districts = df_outage['district'].nunique()
                    st.metric("Number of Districts", f"{unique_districts}")

                st.markdown("---")

                # Cluster distribution
                st.subheader("Distribution of Outages by Cluster")
                st.plotly_chart(plot_cluster_distribution(df_outage), use_container_width=True)

                st.markdown("---")

                # Cluster characteristics
                st.subheader("Average Characteristics of Each Cluster")
                st.plotly_chart(plot_cluster_characteristics(df_outage), use_container_width=True)

                st.markdown("---")

                # Time patterns
                st.subheader("Time Patterns of Power Outages by Cluster")
                st.plotly_chart(plot_cluster_by_time(df_outage), use_container_width=True)

                st.markdown("---")

                # Geographic and temporal patterns
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Geographic Distribution")
                    st.plotly_chart(plot_cluster_by_district(df_outage), use_container_width=True)

                with col2:
                    st.subheader("Distribution by Day of Week")
                    st.plotly_chart(plot_cluster_by_day(df_outage), use_container_width=True)

                st.markdown("---")

                # Weather correlation
                st.subheader("Weather Correlation with Clusters")
                st.plotly_chart(plot_cluster_weather_correlation(df_outage), use_container_width=True)

                st.markdown("---")

                # Cluster details
                st.subheader("Detailed Cluster Analysis")

                clusters = sorted(df_outage['cluster'].unique())
                selected_cluster = st.selectbox(
                    "Select a cluster to view details",
                    clusters,
                    format_func=lambda x: f"Cluster {x}"
                )

                render_cluster_summary(df_outage, selected_cluster)

                # Show sample data
                with st.expander("View sample data of the selected cluster"):
                    cluster_sample = df_outage[df_outage['cluster'] == selected_cluster].head(20)
                    display_cols = ['date', 'day_of_week', 'district', 'start', 'end',
                                   'duration', 'temp', 'rain', 'wind_gust', 'cluster']
                    st.dataframe(cluster_sample[display_cols], use_container_width=True)

    # Tab 7: Additional Analytics
    with tab7:
        st.header("การวิเคราะห์เพิ่มเติม")

        # Time patterns
        st.subheader("รูปแบบตามเวลา")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**รูปแบบตามช่วงเวลาในวัน**")
            st.plotly_chart(plot_hourly_pattern(df_filtered), use_container_width=True)

        with col2:
            st.markdown("**รูปแบบตามวันในสัปดาห์**")
            st.plotly_chart(plot_weekday_pattern(df_filtered), use_container_width=True)

        # Time series comparison
        st.subheader("เปรียบเทียบแนวโน้มแต่ละเขต")
        st.markdown("""
        <div class="info-box">
        <b>คำอธิบาย:</b> เปรียบเทียบแนวโน้มจำนวน complaint ของหลายเขตตามเวลา
        </div>
        """, unsafe_allow_html=True)

        top_districts_for_comparison = df_filtered['district'].value_counts().head(10).index.tolist()
        selected_districts = st.multiselect(
            "เลือกเขตที่ต้องการเปรียบเทียบ",
            top_districts_for_comparison,
            default=top_districts_for_comparison[:5]
        )

        if selected_districts:
            st.plotly_chart(plot_time_series_comparison(df_filtered, selected_districts), use_container_width=True)

        # Summary statistics
        st.subheader("สถิติสรุป")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Top 5 เขตที่มี Complaint มากที่สุด**")
            top_districts = df_filtered['district'].value_counts().head(5)
            for district, count in top_districts.items():
                st.write(f"- {district}: {count:,}")

        with col2:
            st.markdown("**Top 5 ประเภท Complaint**")
            top_types = df_filtered['primary_type'].value_counts().head(5)
            for ptype, count in top_types.items():
                st.write(f"- {ptype}: {count:,}")

        with col3:
            st.markdown("**สถิติเวลาแก้ปัญหา**")
            st.write(f"- เฉลี่ย: {df_filtered['solve_days'].mean():.1f} วัน")
            st.write(f"- มัธยฐาน: {df_filtered['solve_days'].median():.1f} วัน")
            st.write(f"- สูงสุด: {df_filtered['solve_days'].max():.0f} วัน")
            st.write(f"- ต่ำสุด: {max(0, df_filtered['solve_days'].min()):.0f} วัน")

    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p style='font-size: 1.2rem; font-weight: bold;'>Urban Issue Forecasting System</p>
            <p>DSDE M150-Lover Team | Chulalongkorn University</p>
            <p>Data Source: Bangkok Traffy Fondue | Data Rows: {len(df):,}</p>
            <p>ML Models: RandomForest Forecaster + Isolation Forest Anomaly Detector + K-Means Outage Clustering</p>
            <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
