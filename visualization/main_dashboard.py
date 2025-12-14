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
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Translation dictionary
TRANSLATIONS = {
    'en': {
        # Headers
        'main_header': 'Urban Issue Forecasting Dashboard',
        'sub_header': 'Bangkok Traffy Data Analysis & Prediction System',
        'loading_data': 'Loading data and ML models...',

        # Sidebar
        'filters': 'Filters',
        'select_date_range': 'Select Date Range',
        'select_district': 'Select District',
        'select_complaint_type': 'Select Complaint Type',
        'all': 'All',
        'language': 'Language',
        'sampled_records': 'Sampled {count:,} records',
        'loaded_data': 'Loaded data: {count:,} rows',

        # Key Metrics
        'key_metrics': 'Key Metrics',
        'total_complaints': 'Total Complaints',
        'avg_resolution_time': 'Avg Resolution Time',
        'completion_rate': 'Completion Rate',
        'num_districts': 'Number of Districts',
        'complaint_types': 'Complaint Types',
        'days': 'days',

        # Tabs
        'tab_geospatial': 'Geospatial Map',
        'tab_district_analysis': 'District and Type Analysis',
        'tab_mea_outage': 'MEA Power Outage',
        'tab_forecasting': 'ML: Predictive Forecasting',
        'tab_anomaly': 'ML: Anomaly Detection',
        'tab_clustering': 'ML: Power Outage Clustering',
        'tab_additional': 'Additional Analysis',

        # Tab 1: Geospatial
        'geospatial_analysis': 'Geospatial Analysis',
        'geospatial_desc': '<b>Description:</b> Map showing the distribution of complaints in Bangkok<br>- <b>Heat Map:</b> Shows density of issues in each area<br>- <b>Marker Clusters:</b> Shows details of each complaint',
        'traffy_complaint_map': 'Traffy Complaint Map',
        'choose_map_type': 'Choose map type',
        'heat_map': 'Heat Map (Density)',
        'marker_clusters': 'Marker Clusters (Individual Points)',
        'grid_layer': 'Grid Layer (3D Grid Visualization)',
        'loading_map': 'Loading map...',
        'district_statistics': 'District Statistics',
        'number_of_complaints': 'Number of Complaints',
        'avg_resolution_time_days': 'Average Resolution Time (days)',

        # Tab 2: District Analysis
        'district_type_analysis': 'District and Type Analysis',
        'top_districts_title': 'Top Districts by Number of Complaints',
        'top_districts_desc': '<b>Description:</b> Ranks districts with the highest number of complaints to identify areas requiring special attention',
        'num_districts_to_show': 'Number of districts to display',
        'complaints_by_district': 'Complaints by District',
        'complaints_by_district_desc': '<b>Description:</b> Shows the distribution of each complaint type in each district to understand what types of issues occur in each area',
        'filter_by_type': 'Filter by Complaint Type',
        'top_complaint_types': 'Top Complaint Types',
        'complaint_distribution': 'Complaint Distribution Across Districts',
        'complaint_distribution_desc': '<b>Description:</b> Shows which districts each complaint type occurs in to identify distribution patterns of each issue type',
        'filter_by_district': 'Filter by District',

        # Tab 3: MEA Outage
        'outage_slots_viz': '⚡ Outage Slots Visualization',
        'outage_desc': '<b>Description:</b> This tab displays power outage time slot data showing total outage duration in each district and timeline of outage periods each day',
        'total_outage_duration': 'Total outage duration by district',
        'outage_not_available': 'MEA outage data not available. Add clean_scraping_data.csv to data/',

        # Tab 4: Forecasting
        'predictive_modeling': 'Predictive Modeling: Number of Complaints',
        'forecasting_desc': '<b>Description:</b> Uses RandomForest model to forecast number of complaints<br>- <b>Blue line:</b> Actual data<br>- <b>Red line:</b> Predicted values (both past and future - compare accuracy and see future)<br>- <b>Gray area:</b> Confidence interval for future predictions<br>- <b>Gray dashed line:</b> Divides past and future (today)',
        'num_days_predict': 'Number of days to predict',
        'run_forecast': 'Run forecast / Update prediction',
        'no_data_forecast': 'No data available for forecasting with the current filters.',
        'loading_forecast': 'Loading forecast data...',
        'mean_predicted': 'mean predicted',
        'max_predicted': 'max predicted',
        'min_predicted': 'min predicted',
        'complaints_per_day': 'complaints/day',
        'see_forecast_data': 'see forecast data table',
        'please_run_forecast': "Please 'Run forecast / Update prediction'",
        'display_last_run': 'Display last run (days_ahead = {days} days)',
        'date': 'Date',
        'predicted': 'Predicted',
        'lower_bound': 'Lower Bound',
        'upper_bound': 'Upper Bound',

        # Tab 5: Anomaly Detection
        'anomaly_detection': 'Anomaly Detection with Machine Learning',
        'anomaly_desc': '<b>Description:</b> Uses Isolation Forest model to detect complaints with abnormal behavior<br><b>Data Source:</b> Real data from clean_data.csv<br><b>Model:</b> IsolationForest<br>High Anomaly Score = Highly abnormal (e.g., unusually long resolution time, or occurring in abnormal location/time)',
        'settings_for_sampling': 'settings for data sampling',
        'sample_percentage': 'Percentage of data to sample for anomaly detection',
        'reduce_data_help': 'Reduce data size to decrease processing time',
        'using_real_data': 'Using real data {count:,} records from clean_data.csv',
        'loading_anomaly': 'Loading anomaly detection model...',
        'preparing_features': 'Preparing features...',
        'processing_isolation': 'Processing with Isolation Forest model...',
        'completed': 'Completed!',
        'error_anomaly': 'Error during anomaly detection: {error}',
        'num_anomalies': 'Number of Anomalies Detected',
        'anomaly_rate': 'Anomaly Rate',
        'avg_anomaly_score': 'Average Anomaly Score',
        'data_source': 'Data Source',
        'actual_data': 'Actual data from clean_data.csv ({count:,} records)',
        'anomaly_timeline': 'Anomaly Detection Timeline',
        'anomaly_distribution_title': 'Anomaly Distribution by Type and District',
        'detected_anomalies': 'Detected Anomalies (Top 50)',
        'district': 'District',
        'type': 'Type',
        'resolution_days': 'Resolution Time (days)',
        'anomaly_score': 'Anomaly Score',
        'no_anomalies': 'No anomalies found in selected data',

        # Tab 6: Clustering
        'clustering_title': 'K-Means Clustering: Power Outage Event Grouping',
        'clustering_desc': '<b>Description:</b> Uses K-Means model to group power outage events by similar behavior<br><b>Data Source:</b> MEA<br><b>Model:</b> K-Means Clustering<br><b>Features:</b> Day of week, district, temperature, rainfall, wind speed, start time, duration',
        'clustering_warning': 'WARNING: K-Means Clustering model is not available',
        'train_model_info': 'Please train the model by running: `ml_models/outage_model/train_outage_model.py`',
        'cluster_file_not_found': 'Cluster data file not found: {path}',
        'train_first': 'Please train the model first to generate the cluster data file',
        'loading_cluster': 'Loading cluster data...',
        'loaded_successfully': 'Loaded data successfully: {count:,} power outage events',
        'summary_statistics': 'Summary Statistics',
        'num_clusters': 'Number of Clusters',
        'avg_duration': 'Average Duration',
        'minutes': 'minutes',
        'total_outages': 'Total Outages',
        'cluster_distribution': 'Distribution of Outages by Cluster',
        'cluster_characteristics': 'Average Characteristics of Each Cluster',
        'time_patterns': 'Time Patterns of Power Outages by Cluster',
        'geographic_distribution': 'Geographic Distribution',
        'distribution_by_day': 'Distribution by Day of Week',
        'weather_correlation': 'Weather Correlation with Clusters',
        'detailed_cluster': 'Detailed Cluster Analysis',
        'select_cluster': 'Select a cluster to view details',
        'cluster': 'Cluster {num}',
        'view_sample_data': 'View sample data of the selected cluster',

        # Tab 7: Additional
        'additional_analysis': 'Additional Analysis',
        'time_patterns_title': 'Time Patterns',
        'hourly_pattern': 'Hourly Pattern',
        'weekday_pattern': 'Weekday Pattern',
        'compare_trends': 'Compare Trends by District',
        'compare_trends_desc': '<b>Description:</b> Compare complaint trends of multiple districts over time',
        'select_districts_compare': 'Select districts to compare',
        'summary_stats': 'Summary Statistics',
        'top_5_districts': 'Top 5 Districts with Most Complaints',
        'top_5_types': 'Top 5 Complaint Types',
        'resolution_stats': 'Resolution Time Statistics',
        'average': 'Average',
        'median': 'Median',
        'maximum': 'Maximum',
        'minimum': 'Minimum',

        # Footer
        'footer_title': 'Urban Issue Forecasting System',
        'footer_team': 'DSDE M150-Lover Team | Chulalongkorn University',
        'data_source': 'Data Source',
        'data_rows': 'Data Rows',
        'ml_models': 'ML Models',
        'last_updated': 'Last Updated',

        # Map popup
        'popup_district': 'District',
        'popup_type': 'Type',
        'popup_date': 'Date',
        'popup_status': 'Status',
        'popup_resolution': 'Resolution Time',
    },
    'th': {
        # Headers
        'main_header': 'ระบบวิเคราะห์และพยากรณ์ปัญหาในเขตกรุงเทพมหานคร',
        'sub_header': 'ระบบวิเคราะห์และพยากรณ์ปัญหาเขตกรุงเทพมหานคร | Bangkok Traffy Data Analysis',
        'loading_data': 'กำลังโหลดข้อมูลและ ML models...',

        # Sidebar
        'filters': 'ตัวกรอง',
        'select_date_range': 'เลือกช่วงเวลา',
        'select_district': 'เลือกเขต',
        'select_complaint_type': 'เลือกประเภท Complaint',
        'all': 'ทั้งหมด',
        'language': 'ภาษา',
        'sampled_records': 'สุ่มตัวอย่าง {count:,} รายการ',
        'loaded_data': 'โหลดข้อมูล: {count:,} แถว',

        # Key Metrics
        'key_metrics': 'ตัวชี้วัดหลัก (Key Metrics)',
        'total_complaints': 'จำนวน Complaint ทั้งหมด',
        'avg_resolution_time': 'เวลาแก้ปัญหาเฉลี่ย',
        'completion_rate': 'อัตราการแก้ไขเสร็จสิ้น',
        'num_districts': 'จำนวนเขต',
        'complaint_types': 'ประเภทปัญหา',
        'days': 'วัน',

        # Tabs
        'tab_geospatial': 'แผนที่ภูมิศาสตร์',
        'tab_district_analysis': 'การวิเคราะห์ตามเขตและประเภท',
        'tab_mea_outage': 'ข้อมูลไฟดับ MEA',
        'tab_forecasting': 'ML: พยากรณ์แนวโน้ม',
        'tab_anomaly': 'ML: ตรวจจับความผิดปกติ',
        'tab_clustering': 'ML: จัดกลุ่มเหตุการณ์ไฟดับ',
        'tab_additional': 'การวิเคราะห์เพิ่มเติม',

        # Tab 1: Geospatial
        'geospatial_analysis': 'การวิเคราะห์เชิงพื้นที่',
        'geospatial_desc': '<b>คำอธิบาย:</b> แผนที่แสดงการกระจายตัวของ complaint ในกรุงเทพมหานคร<br>- <b>Heat Map:</b> แสดงความหนาแน่นของปัญหาในแต่ละพื้นที่<br>- <b>Marker Clusters:</b> แสดงรายละเอียดของแต่ละ complaint',
        'traffy_complaint_map': 'แผนที่ Traffy Complaint',
        'choose_map_type': 'เลือกประเภทแผนที่',
        'heat_map': 'Heat Map (ความหนาแน่น)',
        'marker_clusters': 'Marker Clusters (จุดแต่ละรายการ)',
        'grid_layer': 'Grid Layer (กราฟ 3D)',
        'loading_map': 'กำลังโหลดแผนที่...',
        'district_statistics': 'สถิติตามเขต',
        'number_of_complaints': 'จำนวน Complaint',
        'avg_resolution_time_days': 'เวลาแก้ปัญหาเฉลี่ย (วัน)',

        # Tab 2: District Analysis
        'district_type_analysis': 'การวิเคราะห์ตามเขตและประเภท',
        'top_districts_title': 'เขตที่มี Complaint มากที่สุด',
        'top_districts_desc': '<b>คำอธิบาย:</b> จัดอันดับเขตที่มีจำนวน complaint สูงสุด ช่วยระบุพื้นที่ที่ต้องให้ความสนใจเป็นพิเศษ',
        'num_districts_to_show': 'จำนวนเขตที่ต้องการแสดง',
        'complaints_by_district': 'Complaint แยกตามเขต',
        'complaints_by_district_desc': '<b>คำอธิบาย:</b> แสดงการกระจายของ complaint แต่ละประเภทในแต่ละเขต ช่วยให้เห็นว่าแต่ละเขตมีปัญหาประเภทใดบ้าง',
        'filter_by_type': 'กรองตามประเภท Complaint',
        'top_complaint_types': 'ประเภท Complaint ที่พบมากที่สุด',
        'complaint_distribution': 'การกระจายของ Complaint ในแต่ละเขต',
        'complaint_distribution_desc': '<b>คำอธิบาย:</b> แสดงว่า complaint แต่ละประเภทเกิดขึ้นในเขตใดบ้าง ช่วยระบุรูปแบบการกระจายของปัญหาแต่ละประเภท',
        'filter_by_district': 'กรองตามเขต',

        # Tab 3: MEA Outage
        'outage_slots_viz': '⚡ การแสดงผลช่วงเวลาไฟดับ',
        'outage_desc': '<b>คำอธิบาย:</b> แท็บนี้แสดงข้อมูลช่วงเวลาไฟดับตามตัวอย่างข้อมูล โดยแสดงทั้งระยะเวลาไฟดับรวมในแต่ละเขต และไทม์ไลน์ของช่วงเวลาที่ไฟดับในแต่ละวัน',
        'total_outage_duration': 'ระยะเวลาไฟดับรวมในแต่ละเขต',
        'outage_not_available': 'ไม่มีข้อมูลไฟดับ MEA กรุณาเพิ่มไฟล์ clean_scraping_data.csv ใน data/',

        # Tab 4: Forecasting
        'predictive_modeling': 'การพยากรณ์: จำนวน Complaint',
        'forecasting_desc': '<b>คำอธิบาย:</b> ใช้โมเดล RandomForest ในการพยากรณ์จำนวน complaint<br>- <b>เส้นสีน้ำเงิน:</b> ข้อมูลจริง<br>- <b>เส้นสีแดง:</b> ค่าพยากรณ์ (ทั้งอดีตและอนาคต - เปรียบเทียบความแม่นยำและดูอนาคต)<br>- <b>พื้นที่สีเทา:</b> ช่วงความเชื่อมั่นสำหรับอนาคต (Confidence Interval)<br>- <b>เส้นประสีเทา:</b> แบ่งระหว่างอดีตและอนาคต (วันนี้)',
        'num_days_predict': 'จำนวนวันที่ต้องการพยากรณ์',
        'run_forecast': 'พยากรณ์ / อัพเดทการพยากรณ์',
        'no_data_forecast': 'ไม่มีข้อมูลสำหรับการพยากรณ์ตามตัวกรองปัจจุบัน',
        'loading_forecast': 'กำลังโหลดข้อมูลการพยากรณ์...',
        'mean_predicted': 'ค่าเฉลี่ยที่พยากรณ์',
        'max_predicted': 'ค่าสูงสุดที่พยากรณ์',
        'min_predicted': 'ค่าต่ำสุดที่พยากรณ์',
        'complaints_per_day': 'complaints/วัน',
        'see_forecast_data': 'ดูตารางข้อมูลการพยากรณ์',
        'please_run_forecast': 'กรุณากด "พยากรณ์ / อัพเดทการพยากรณ์"',
        'display_last_run': 'แสดงผลการพยากรณ์ครั้งล่าสุด (days_ahead = {days} วัน)',
        'date': 'วันที่',
        'predicted': 'ค่าพยากรณ์',
        'lower_bound': 'ขอบล่าง',
        'upper_bound': 'ขอบบน',

        # Tab 5: Anomaly Detection
        'anomaly_detection': 'การตรวจจับความผิดปกติด้วย Machine Learning',
        'anomaly_desc': '<b>คำอธิบาย:</b> ใช้โมเดล Isolation Forest ในการตรวจจับ complaint ที่มีพฤติกรรมผิดปกติ<br><b>ข้อมูลที่ใช้:</b> ข้อมูลจริงจาก clean_data.csv<br><b>โมเดล:</b> IsolationForest <br>Anomaly Score สูง = ผิดปกติมาก (เช่น ใช้เวลาแก้ไขนานผิดปกติ หรือเกิดในพื้นที่/เวลาที่ผิดปกติ)',
        'settings_for_sampling': 'ตั้งค่าการสุ่มตัวอย่างข้อมูล',
        'sample_percentage': 'เปอร์เซ็นต์ข้อมูลที่ใช้ในการตรวจจับความผิดปกติ',
        'reduce_data_help': 'ลดขนาดข้อมูลเพื่อลดเวลาในการประมวลผล',
        'using_real_data': 'ใช้ข้อมูลจริง {count:,} รายการ จาก clean_data.csv',
        'loading_anomaly': 'กำลังโหลดโมเดลตรวจจับความผิดปกติ...',
        'preparing_features': 'กำลังเตรียม features...',
        'processing_isolation': 'กำลังประมวลผลด้วยโมเดล Isolation Forest...',
        'completed': 'เสร็จสิ้น!',
        'error_anomaly': 'เกิดข้อผิดพลาดในการตรวจจับความผิดปกติ: {error}',
        'num_anomalies': 'จำนวนความผิดปกติที่ตรวจพบ',
        'anomaly_rate': 'อัตราความผิดปกติ',
        'avg_anomaly_score': 'คะแนนความผิดปกติเฉลี่ย',
        'data_source': 'แหล่งข้อมูล',
        'actual_data': 'ข้อมูลจริงจาก clean_data.csv ({count:,} รายการ)',
        'anomaly_timeline': 'ไทม์ไลน์การตรวจจับความผิดปกติ',
        'anomaly_distribution_title': 'การกระจายความผิดปกติตามประเภทและเขต',
        'detected_anomalies': 'ความผิดปกติที่ตรวจพบ (50 อันดับแรก)',
        'district': 'เขต',
        'type': 'ประเภท',
        'resolution_days': 'ระยะเวลาแก้ (วัน)',
        'anomaly_score': 'คะแนนความผิดปกติ',
        'no_anomalies': 'ไม่พบความผิดปกติในข้อมูลที่เลือก',

        # Tab 6: Clustering
        'clustering_title': 'K-Means Clustering: การจัดกลุ่มเหตุการณ์ไฟดับ',
        'clustering_desc': '<b>คำอธิบาย:</b> ใช้โมเดล K-Means ในการจัดกลุ่มเหตุการณ์ไฟดับตามพฤติกรรมที่คล้ายกัน<br><b>ข้อมูลที่ใช้:</b> MEA <br><b>Model:</b> K-Means Clustering <br><b>Features:</b> วันในสัปดาห์, เขต, อุณหภูมิ, ปริมาณฝน, ความเร็วลม, เวลาเริ่ม, ระยะเวลา',
        'clustering_warning': 'คำเตือน: โมเดล K-Means Clustering ไม่พร้อมใช้งาน',
        'train_model_info': 'กรุณาเทรนโมเดลโดยรัน: `ml_models/outage_model/train_outage_model.py`',
        'cluster_file_not_found': 'ไม่พบไฟล์ข้อมูล cluster: {path}',
        'train_first': 'กรุณาเทรนโมเดลก่อนเพื่อสร้างไฟล์ข้อมูล cluster',
        'loading_cluster': 'กำลังโหลดข้อมูล cluster...',
        'loaded_successfully': 'โหลดข้อมูลสำเร็จ: {count:,} เหตุการณ์ไฟดับ',
        'summary_statistics': 'สถิติสรุป',
        'num_clusters': 'จำนวน Cluster',
        'avg_duration': 'ระยะเวลาเฉลี่ย',
        'minutes': 'นาที',
        'total_outages': 'จำนวนไฟดับทั้งหมด',
        'cluster_distribution': 'การกระจายของไฟดับในแต่ละ Cluster',
        'cluster_characteristics': 'ลักษณะเฉลี่ยของแต่ละ Cluster',
        'time_patterns': 'รูปแบบเวลาของไฟดับในแต่ละ Cluster',
        'geographic_distribution': 'การกระจายตามพื้นที่',
        'distribution_by_day': 'การกระจายตามวันในสัปดาห์',
        'weather_correlation': 'ความสัมพันธ์กับสภาพอากาศ',
        'detailed_cluster': 'การวิเคราะห์ Cluster แบบละเอียด',
        'select_cluster': 'เลือก cluster เพื่อดูรายละเอียด',
        'cluster': 'Cluster {num}',
        'view_sample_data': 'ดูตัวอย่างข้อมูลของ cluster ที่เลือก',

        # Tab 7: Additional
        'additional_analysis': 'การวิเคราะห์เพิ่มเติม',
        'time_patterns_title': 'รูปแบบตามเวลา',
        'hourly_pattern': 'รูปแบบตามช่วงเวลาในวัน',
        'weekday_pattern': 'รูปแบบตามวันในสัปดาห์',
        'compare_trends': 'เปรียบเทียบแนวโน้มแต่ละเขต',
        'compare_trends_desc': '<b>คำอธิบาย:</b> เปรียบเทียบแนวโน้มจำนวน complaint ของหลายเขตตามเวลา',
        'select_districts_compare': 'เลือกเขตที่ต้องการเปรียบเทียบ',
        'summary_stats': 'สถิติสรุป',
        'top_5_districts': 'Top 5 เขตที่มี Complaint มากที่สุด',
        'top_5_types': 'Top 5 ประเภท Complaint',
        'resolution_stats': 'สถิติเวลาแก้ปัญหา',
        'average': 'เฉลี่ย',
        'median': 'มัธยฐาน',
        'maximum': 'สูงสุด',
        'minimum': 'ต่ำสุด',

        # Footer
        'footer_title': 'ระบบวิเคราะห์และพยากรณ์ปัญหาในเขตกรุงเทพมหานคร',
        'footer_team': 'ทีม DSDE M150-Lover | จุฬาลงกรณ์มหาวิทยาลัย',
        'data_source': 'แหล่งข้อมูล',
        'data_rows': 'จำนวนแถวข้อมูล',
        'ml_models': 'โมเดล ML',
        'last_updated': 'อัพเดทล่าสุด',

        # Map popup
        'popup_district': 'เขต',
        'popup_type': 'ประเภท',
        'popup_date': 'วันที่',
        'popup_status': 'สถานะ',
        'popup_resolution': 'ระยะเวลาแก้',
    }
}

def t(key, lang='en', **kwargs):
    """Translation helper function"""
    text = TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text

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
    csv_path = Path("data/clean_data.csv")

    if not csv_path.exists():
        st.error(f"File not found: {csv_path.absolute()}")
        st.info(f"Please place clean_data.csv at: {csv_path.absolute()}")
        st.stop()

    # Load CSV
    df = pd.read_csv(csv_path)

    # SAMPLE DATA FOR DEPLOYMENT
    SAMPLE_SIZE = 100000  # Change this number if it lags
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        st.sidebar.info(f"Sampled {SAMPLE_SIZE:,} records")

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
    csv_path = Path("data/clean_scraping_data.csv")

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

    # Initialize session state for language
    if 'language' not in st.session_state:
        st.session_state.language = 'en'

    # Language toggle in sidebar (at the very top)
    lang_options = {'English': 'en', 'ไทย': 'th'}
    selected_lang_label = st.sidebar.selectbox(
        "🌐 Language / ภาษา",
        options=list(lang_options.keys()),
        index=0 if st.session_state.language == 'en' else 1
    )
    st.session_state.language = lang_options[selected_lang_label]
    lang = st.session_state.language

    # Header
    st.markdown(f'<div class="main-header">{t("main_header", lang)}</div>',
               unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{t("sub_header", lang)}</div>',
               unsafe_allow_html=True)
    st.markdown("---")

    # Load data and models
    with st.spinner(t("loading_data", lang)):
        df = load_data()
        df_mea_outage = load_mea_outage_data()
        ml_integrator = load_ml_models()

    # Sidebar filters
    st.sidebar.header(t("filters", lang))

    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    date_range = st.sidebar.date_input(
        t("select_date_range", lang),
        value=(date(2024, 10    , 1), max_date),
        min_value=min_date,
        max_value=max_date
    )

    # District filter
    districts = [t("all", lang)] + sorted(df['district'].dropna().unique().tolist())
    selected_district = st.sidebar.selectbox(t("select_district", lang), districts)

    # Complaint type filter
    types = [t("all", lang)] + sorted(df['primary_type'].unique().tolist())
    selected_type = st.sidebar.selectbox(t("select_complaint_type", lang), types)

    # Apply filters
    df_filtered = df.copy()
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['timestamp'].dt.date >= date_range[0]) &
            (df_filtered['timestamp'].dt.date <= date_range[1])
        ]

    if selected_district != t("all", lang):
        df_filtered = df_filtered[df_filtered['district'] == selected_district]

    if selected_type != t("all", lang):
        df_filtered = df_filtered[df_filtered['primary_type'] == selected_type]

    # Key Metrics
    st.header(t("key_metrics", lang))

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            t("total_complaints", lang),
            f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df):.0f}" if selected_district != t("all", lang) or selected_type != t("all", lang) else None
        )

    with col2:
        avg_resolution = df_filtered['solve_days'].mean()
        st.metric(
            t("avg_resolution_time", lang),
            f"{avg_resolution:.1f} {t('days', lang)}"
        )

    with col3:
        completion_rate = (df_filtered['state'] == 'เสร็จสิ้น').mean() * 100
        st.metric(
            t("completion_rate", lang),
            f"{completion_rate:.1f}%"
        )

    with col4:
        unique_districts = df_filtered['district'].nunique()
        st.metric(
            t("num_districts", lang),
            f"{unique_districts}"
        )

    with col5:
        unique_types = df_filtered['primary_type'].nunique()
        st.metric(
            t("complaint_types", lang),
            f"{unique_types}"
        )

    st.markdown("---")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        t("tab_geospatial", lang),
        t("tab_district_analysis", lang),
        t("tab_mea_outage", lang),
        t("tab_forecasting", lang),
        t("tab_anomaly", lang),
        t("tab_clustering", lang),
        t("tab_additional", lang)
    ])

    # Tab 1: Geospatial Analysis
    with tab1:
        st.header(t("geospatial_analysis", lang))

        st.markdown(f"""
        <div class="info-box">
        {t("geospatial_desc", lang)}
        </div>
        """, unsafe_allow_html=True)

        st.subheader(t("traffy_complaint_map", lang))
        # Map visualization type
        map_type = st.radio(
            t("choose_map_type", lang),
            ['heatmap', 'clusters', 'GridLayer'],
            format_func=lambda x: t('heat_map', lang) if x == 'heatmap' else t('marker_clusters', lang) if x == 'clusters' else t('grid_layer', lang)
        )
        st.markdown("---")

        with st.spinner(t("loading_map", lang)):
            if map_type == 'GridLayer':
                result = create_geospatial_map(df_filtered, map_type=map_type)
                st.pydeck_chart(result)
            else:
                m = create_geospatial_map(df_filtered, map_type=map_type)
                folium_static(m, width=1400, height=600)

        # District statistics table
        st.subheader(t("district_statistics", lang))
        district_stats = df_filtered.groupby('district').agg({
            'lat': 'count',
            'solve_days': 'mean',
        }).round(2)
        district_stats.columns = [t("number_of_complaints", lang), t("avg_resolution_time_days", lang)]
        district_stats = district_stats.sort_values(t("number_of_complaints", lang), ascending=False)

        st.dataframe(district_stats, use_container_width=True, height=400)

    # Tab 2: District and Type Analysis
    with tab2:
        st.header(t("district_type_analysis", lang))

        # 1.) Top districts
        st.subheader(t("top_districts_title", lang))
        st.markdown(f"""
        <div class="info-box">
        {t("top_districts_desc", lang)}
        </div>
        """, unsafe_allow_html=True)

        top_n = st.slider(t("num_districts_to_show", lang), 5, 30, 15, key="top_districts")
        st.subheader(f"Top {top_n} {t('top_districts_title', lang)}")
        st.plotly_chart(plot_top_complaint_districts(df_filtered, top_n), use_container_width=True)
        
        # 2.) Complaints by district
        st.subheader(t("complaints_by_district", lang))
        st.markdown(f"""
        <div class="info-box">
        {t("complaints_by_district_desc", lang)}
        </div>
        """, unsafe_allow_html=True)

        col_filter1, col_spacer1 = st.columns([2, 3])
        with col_filter1:
            complaint_filter_1 = st.selectbox(
                t("filter_by_type", lang),
                [t("all", lang)] + sorted(df_filtered['primary_type'].unique().tolist()),
                key="complaint_by_district"
            )

        st.plotly_chart(plot_complaints_by_district(df_filtered, complaint_filter_1 if complaint_filter_1 != t("all", lang) else 'All'), use_container_width=True)

        st.markdown("---")


        # 3.) Additional visualizations
        st.subheader(t("top_complaint_types", lang))
        st.plotly_chart(plot_top_complaint_types(df_filtered, top_n=15), use_container_width=True)


        # 4.) Complaint distribution across districts
        st.subheader(t("complaint_distribution", lang))
        st.markdown(f"""
        <div class="info-box">
        {t("complaint_distribution_desc", lang)}
        </div>
        """, unsafe_allow_html=True)

        col_filter2, col_spacer2 = st.columns([2, 3])
        with col_filter2:
            district_filter_1 = st.selectbox(
                t("filter_by_district", lang),
                [t("all", lang)] + sorted(df_filtered['district'].dropna().unique().tolist()),
                key="complaint_distribution"
            )

        st.plotly_chart(plot_complaint_distribution_across_districts(df_filtered, district_filter_1 if district_filter_1 != t("all", lang) else 'All'), use_container_width=True)

        #st.markdown("---")

        # # 5.) Time series: complaints over time with filters
        # st.subheader("Time Series: จำนวน Complaint ตามเวลา")

        # st.markdown("""
        # <div class="info-box">
        # <b>คำอธิบาย:</b> แสดงจำนวน complaint ต่อวัน โดยสามารถเลือกช่วงเวลาและจังหวัดได้ 
        # เพื่อดูแนวโน้มการเกิดปัญหาในช่วงต่าง ๆ
        # </div>
        # """, unsafe_allow_html=True)

        # # Ensure timestamp/date are in proper format
        # df_ts = df_filtered.copy()
        # if "timestamp" in df_ts.columns:
        #     # Always try to convert to datetime, safe even if already datetime
        #     df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"], errors="coerce")
        #     min_date = df_ts["timestamp"].dt.date.min()
        #     max_date = df_ts["timestamp"].dt.date.max()
        # else:
        #     df_ts["date"] = pd.to_datetime(df_ts["date"], errors="coerce")
        #     min_date = df_ts["date"].dt.date.min()
        #     max_date = df_ts["date"].dt.date.max()


        # # UI controls for time series (in main area, not sidebar)
        # col1, col2 = st.columns(2)

        # with col1:
        #     date_range = st.date_input(
        #         "Select date range",
        #         value=(min_date, max_date),
        #         min_value=min_date,
        #         max_value=max_date
        #     )

        # with col2:
        #     if "province" in df_ts.columns:
        #         province_options = sorted(df_ts["province"].dropna().unique())
        #         selected_provinces = st.multiselect(
        #             "Select provinces",
        #             options=province_options,
        #             default=province_options  # show all by default
        #         )
        #     else:
        #         selected_provinces = None

        # # Apply filters
        # start_date, end_date = date_range
        # if "timestamp" in df_ts.columns:
        #     df_ts = df_ts[
        #         (df_ts["timestamp"].dt.date >= start_date)
        #         & (df_ts["timestamp"].dt.date <= end_date)
        #     ]
        # else:
        #     df_ts = df_ts[
        #         (df_ts["date"].dt.date >= start_date)
        #         & (df_ts["date"].dt.date <= end_date)
        #     ]

        # if selected_provinces is not None and len(selected_provinces) > 0:
        #     df_ts = df_ts[df_ts["province"].isin(selected_provinces)]

        # if df_ts.empty:
        #     st.warning("ไม่มีข้อมูลในช่วงวันที่และจังหวัดที่เลือก")    
        # else:
        #         st.plotly_chart(plot_complaint_timeseries(df_ts), use_container_width=True)

    # Tab 3: MEA power outage
    with tab3:
        st.header(t("outage_slots_viz", lang))

        st.markdown(f"""
        <div class="info-box">
        {t("outage_desc", lang)}
        </div>
        """, unsafe_allow_html=True)

        df_outage = df_mea_outage

        if df_outage is not None:
            # Prepare dataframe (add start_dt, end_dt)
            df_outage_prepared = prepare_outage_dataframe(df_outage)

            # Bar chart: total duration by district
            st.subheader(t("total_outage_duration", lang))
            st.plotly_chart(
                plot_outage_duration_by_district(df_outage_prepared),
                use_container_width=True
            )
        else:
            st.warning(t("outage_not_available", lang))

    # Tab 4: Forecasting
    with tab4:
        st.header(t("predictive_modeling", lang))

        st.markdown(f"""
        <div class="info-box">
        {t("forecasting_desc", lang)}
        </div>
        """, unsafe_allow_html=True)

        forecast_days = st.slider(t("num_days_predict", lang), 7, 60, 30)

        # Prepare session_state for storing forecast results
        if "forecast_df" not in st.session_state:
            st.session_state["forecast_df"] = None
        if "forecast_days_used" not in st.session_state:
            st.session_state["forecast_days_used"] = None
    
        # Click button to run forecast
        run_forecast = st.button(t("run_forecast", lang))
        if run_forecast:
            if df_filtered.empty:
                st.warning(t("no_data_forecast", lang))
            else:
                with st.spinner(t("loading_forecast", lang)):
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
                t("display_last_run", lang, days=st.session_state['forecast_days_used'])
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
                    t("mean_predicted", lang),
                    f"{forecast_df['predicted'].mean():.0f} {t('complaints_per_day', lang)}"
                )

            with col2:
                st.metric(
                    t("max_predicted", lang),
                    f"{forecast_df['predicted'].max():.0f} {t('complaints_per_day', lang)}"
                )

            with col3:
                st.metric(
                    t("min_predicted", lang),
                    f"{forecast_df['predicted'].min():.0f} {t('complaints_per_day', lang)}"
                )

            # Show forecast data
            with st.expander(t("see_forecast_data", lang)):
                forecast_display = forecast_df.copy()
                forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
                forecast_display.columns = [t('date', lang), t('predicted', lang), t('lower_bound', lang), t('upper_bound', lang)]
                st.dataframe(forecast_display, use_container_width=True, height=400)

        else:
            st.info(t("please_run_forecast", lang))

    # Tab 5: Anomaly Detection
    with tab5:
        st.header(t("anomaly_detection", lang))

        st.markdown(f"""
        <div class="info-box">
        {t("anomaly_desc", lang)}
        </div>
        """, unsafe_allow_html=True)

        # Settings for data sampling
        st.markdown(f"##### {t('settings_for_sampling', lang)}")

        col_setting1, col_setting2 = st.columns([2, 3])

        # Use filtered data from main dashboard
        df_for_anomaly = df_filtered.copy()
        total_data = len(df_for_anomaly)

        with col_setting1:
            # Allow sampling if dataset is large
            if total_data > 50000:
                sample_percentage = st.slider(
                    t("sample_percentage", lang),
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=10,
                    format="%d%%",
                    help=t("reduce_data_help", lang),
                )
                sample_size = int(total_data * sample_percentage / 100)
                sample_size = max(5000, sample_size)
                df_for_anomaly = df_for_anomaly.sample(n=sample_size, random_state=42).copy()
            else:
                sample_percentage = 100

        with col_setting2:
            st.info(t("using_real_data", lang, count=len(df_for_anomaly)))

        # Detect anomalies
        st.markdown("---")
        progress_text = t("loading_anomaly", lang)
        progress_bar = st.progress(0, text=progress_text)

        @st.cache_data(ttl=3600, show_spinner=False)
        def detect_anomalies_cached(_ml_int, data_hash, size):
            return _ml_int.detect_anomalies(df_for_anomaly)

        try:
            progress_bar.progress(30, text=t("preparing_features", lang))

            # Create hash based on data
            data_hash = hash(str(len(df_for_anomaly)) + str(df_for_anomaly['timestamp'].min()) + str(df_for_anomaly['timestamp'].max()))

            progress_bar.progress(70, text=t("processing_isolation", lang))
            df_with_anomalies = detect_anomalies_cached(ml_integrator, data_hash, len(df_for_anomaly))

            progress_bar.progress(100, text=t("completed", lang))
            progress_bar.empty()
        except Exception as e:
            progress_bar.empty()
            st.error(t("error_anomaly", lang, error=str(e)))
            st.stop()

        # Anomaly statistics
        total_anomalies = df_with_anomalies['is_anomaly'].sum()
        anomaly_rate = (total_anomalies / len(df_with_anomalies)) * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                t("num_anomalies", lang),
                f"{total_anomalies:,}"
            )

        with col2:
            st.metric(
                t("anomaly_rate", lang),
                f"{anomaly_rate:.2f}%"
            )

        with col3:
            if total_anomalies > 0:
                avg_anomaly_score = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]['anomaly_score'].mean()
                st.metric(
                    t("avg_anomaly_score", lang),
                    f"{avg_anomaly_score:.2f}"
                )
            else:
                st.metric(
                    t("avg_anomaly_score", lang),
                    "N/A"
                )

        # Data source info
        st.info(f"**{t('data_source', lang)}:** {t('actual_data', lang, count=len(df_with_anomalies))}")

        # Anomaly scatter plot
        st.subheader(t("anomaly_timeline", lang))
        st.plotly_chart(plot_anomaly_scatter(df_with_anomalies), use_container_width=True)

        # Anomaly distribution
        st.subheader(t("anomaly_distribution_title", lang))
        st.plotly_chart(plot_anomaly_distribution(df_with_anomalies), use_container_width=True)

        # Anomaly table
        st.subheader(t("detected_anomalies", lang))
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1].copy()

        if len(anomalies) > 0:
            anomalies_display = anomalies[['timestamp', 'district', 'primary_type', 'solve_days', 'anomaly_score']].sort_values(
                'anomaly_score', ascending=False
            ).head(50)

            anomalies_display.columns = [t('date', lang), t('district', lang), t('type', lang), t('resolution_days', lang), t('anomaly_score', lang)]
            st.dataframe(anomalies_display, use_container_width=True, height=400)
        else:
            st.info(t("no_anomalies", lang))

    # Tab 6: Outage Clustering
    with tab6:
        st.header(t("clustering_title", lang))

        st.markdown(f"""
        <div class="info-box">
        {t("clustering_desc", lang)}
        </div>
        """, unsafe_allow_html=True)

        if ml_integrator.outage_model is None:
            st.warning(t("clustering_warning", lang))
            st.info(t("train_model_info", lang))
        else:
            # Load outage data with clusters
            outage_data_path = Path("data/power_outage_with_clusters.csv")

            if not outage_data_path.exists():
                st.error(t("cluster_file_not_found", lang, path=outage_data_path))
                st.info(t("train_first", lang))
            else:
                with st.spinner(t("loading_cluster", lang)):
                    df_outage = pd.read_csv(outage_data_path)

                st.success(t("loaded_successfully", lang, count=len(df_outage)))

                # Show key metrics
                st.markdown(f"### {t('summary_statistics', lang)}")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(t("num_clusters", lang), f"{df_outage['cluster'].nunique()}")

                with col2:
                    avg_duration = df_outage['duration'].mean()
                    st.metric(t("avg_duration", lang), f"{avg_duration:.0f} {t('minutes', lang)}")
                with col3:
                    total_outages = len(df_outage)
                    st.metric(t("total_outages", lang), f"{total_outages:,}")

                with col4:
                    unique_districts = df_outage['district'].nunique()
                    st.metric(t("num_districts", lang), f"{unique_districts}")

                st.markdown("---")

                # Cluster distribution
                st.subheader(t("cluster_distribution", lang))
                st.plotly_chart(plot_cluster_distribution(df_outage), use_container_width=True)

                st.markdown("---")

                # Cluster characteristics
                st.subheader(t("cluster_characteristics", lang))
                st.plotly_chart(plot_cluster_characteristics(df_outage), use_container_width=True)

                st.markdown("---")

                # Time patterns
                st.subheader(t("time_patterns", lang))
                st.plotly_chart(plot_cluster_by_time(df_outage), use_container_width=True)

                st.markdown("---")

                # Geographic and temporal patterns
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(t("geographic_distribution", lang))
                    st.plotly_chart(plot_cluster_by_district(df_outage), use_container_width=True)

                with col2:
                    st.subheader(t("distribution_by_day", lang))
                    st.plotly_chart(plot_cluster_by_day(df_outage), use_container_width=True)

                st.markdown("---")

                # Weather correlation
                st.subheader(t("weather_correlation", lang))
                st.plotly_chart(plot_cluster_weather_correlation(df_outage), use_container_width=True)

                st.markdown("---")

                # Cluster details
                st.subheader(t("detailed_cluster", lang))

                clusters = sorted(df_outage['cluster'].unique())
                selected_cluster = st.selectbox(
                    t("select_cluster", lang),
                    clusters,
                    format_func=lambda x: t("cluster", lang, num=x)
                )

                render_cluster_summary(df_outage, selected_cluster)

                # Show sample data
                with st.expander(t("view_sample_data", lang)):
                    cluster_sample = df_outage[df_outage['cluster'] == selected_cluster].head(20)
                    display_cols = ['date', 'day_of_week', 'district', 'start', 'end',
                                   'duration', 'temp', 'rain', 'wind_gust', 'cluster']
                    st.dataframe(cluster_sample[display_cols], use_container_width=True)

    # Tab 7: Additional Analytics
    with tab7:
        st.header(t("additional_analysis", lang))

        # Time patterns
        st.subheader(t("time_patterns_title", lang))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{t('hourly_pattern', lang)}**")
            st.plotly_chart(plot_hourly_pattern(df_filtered), use_container_width=True)

        with col2:
            st.markdown(f"**{t('weekday_pattern', lang)}**")
            st.plotly_chart(plot_weekday_pattern(df_filtered), use_container_width=True)

        # Time series comparison
        st.subheader(t("compare_trends", lang))
        st.markdown(f"""
        <div class="info-box">
        {t("compare_trends_desc", lang)}
        </div>
        """, unsafe_allow_html=True)

        top_districts_for_comparison = df_filtered['district'].value_counts().head(10).index.tolist()
        selected_districts = st.multiselect(
            t("select_districts_compare", lang),
            top_districts_for_comparison,
            default=top_districts_for_comparison[:5]
        )

        if selected_districts:
            st.plotly_chart(plot_time_series_comparison(df_filtered, selected_districts), use_container_width=True)

        # Summary statistics
        st.subheader(t("summary_stats", lang))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**{t('top_5_districts', lang)}**")
            top_districts = df_filtered['district'].value_counts().head(5)
            for district, count in top_districts.items():
                st.write(f"- {district}: {count:,}")

        with col2:
            st.markdown(f"**{t('top_5_types', lang)}**")
            top_types = df_filtered['primary_type'].value_counts().head(5)
            for ptype, count in top_types.items():
                st.write(f"- {ptype}: {count:,}")

        with col3:
            st.markdown(f"**{t('resolution_stats', lang)}**")
            st.write(f"- {t('average', lang)}: {df_filtered['solve_days'].mean():.1f} {t('days', lang)}")
            st.write(f"- {t('median', lang)}: {df_filtered['solve_days'].median():.1f} {t('days', lang)}")
            st.write(f"- {t('maximum', lang)}: {df_filtered['solve_days'].max():.0f} {t('days', lang)}")
            st.write(f"- {t('minimum', lang)}: {max(0, df_filtered['solve_days'].min()):.0f} {t('days', lang)}")

    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p style='font-size: 1.2rem; font-weight: bold;'>{t("footer_title", lang)}</p>
            <p>{t("footer_team", lang)}</p>
            <p>{t("data_source", lang)}: Bangkok Traffy Fondue | {t("data_rows", lang)}: {len(df):,}</p>
            <p>{t("ml_models", lang)}: RandomForest Forecaster + Isolation Forest Anomaly Detector + K-Means Outage Clustering</p>
            <p>{t("last_updated", lang)}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
