"""
Outage K-Means Clustering Visualization Module
Visualizes power outage clustering patterns
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


def plot_cluster_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plot distribution of outages across clusters

    คำอธิบาย: แสดงจำนวนไฟดับในแต่ละคลัสเตอร์
    """
    cluster_counts = df['cluster'].value_counts().sort_index()

    fig = go.Figure(data=[
        go.Bar(
            x=[f'Cluster {i}' for i in cluster_counts.index],
            y=cluster_counts.values,
            marker=dict(
                color=cluster_counts.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="จำนวน")
            ),
            text=cluster_counts.values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>จำนวน: %{y}<extra></extra>'
        )
    ])

    fig.update_layout(
        title='การกระจายของไฟดับในแต่ละคลัสเตอร์',
        xaxis_title='คลัสเตอร์',
        yaxis_title='จำนวนเหตุการณ์ไฟดับ',
        template='plotly_white',
        height=400
    )

    return fig


def plot_cluster_by_time(df: pd.DataFrame) -> go.Figure:
    """
    Plot clusters by start time

    คำอธิบาย: แสดงรูปแบบเวลาที่ไฟดับในแต่ละคลัสเตอร์
    """
    fig = go.Figure()

    clusters = sorted(df['cluster'].unique())

    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]

        fig.add_trace(go.Scatter(
            x=cluster_data['start_min'],
            y=cluster_data['duration'],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(
                size=8,
                opacity=0.6
            ),
            hovertemplate='<b>Cluster %{fullData.name}</b><br>' +
                         'เวลาเริ่ม: %{x} นาที<br>' +
                         'ระยะเวลา: %{y} นาที<extra></extra>'
        ))

    fig.update_layout(
        title='รูปแบบเวลาของไฟดับแยกตามคลัสเตอร์',
        xaxis_title='เวลาเริ่มไฟดับ (นาทีของวัน)',
        yaxis_title='ระยะเวลาไฟดับ (นาที)',
        template='plotly_white',
        height=500,
        hovermode='closest'
    )

    return fig


def plot_cluster_characteristics(df: pd.DataFrame) -> go.Figure:
    """
    Plot average characteristics of each cluster

    คำอธิบาย: แสดงลักษณะเฉลี่ยของแต่ละคลัสเตอร์
    """
    cluster_stats = df.groupby('cluster')[['temp', 'rain', 'wind_gust', 'start_min', 'duration']].mean()

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'อุณหภูมิเฉลี่ย (°C)',
            'ปริมาณฝนเฉลี่ย (mm)',
            'ความเร็วลมเฉลี่ย (km/h)',
            'เวลาเริ่มเฉลี่ย (นาที)',
            'ระยะเวลาเฉลี่ย (นาที)',
            'จำนวนเหตุการณ์'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )

    clusters = [f'C{i}' for i in cluster_stats.index]

    # Temperature
    fig.add_trace(
        go.Bar(x=clusters, y=cluster_stats['temp'], marker_color='indianred',
               hovertemplate='%{x}<br>อุณหภูมิ: %{y:.1f}°C<extra></extra>'),
        row=1, col=1
    )

    # Rain
    fig.add_trace(
        go.Bar(x=clusters, y=cluster_stats['rain'], marker_color='lightblue',
               hovertemplate='%{x}<br>ฝน: %{y:.1f} mm<extra></extra>'),
        row=1, col=2
    )

    # Wind gust
    fig.add_trace(
        go.Bar(x=clusters, y=cluster_stats['wind_gust'], marker_color='lightgreen',
               hovertemplate='%{x}<br>ลม: %{y:.1f} km/h<extra></extra>'),
        row=1, col=3
    )

    # Start time
    fig.add_trace(
        go.Bar(x=clusters, y=cluster_stats['start_min'], marker_color='orange',
               hovertemplate='%{x}<br>เวลาเริ่ม: %{y:.0f} นาที<extra></extra>'),
        row=2, col=1
    )

    # Duration
    fig.add_trace(
        go.Bar(x=clusters, y=cluster_stats['duration'], marker_color='purple',
               hovertemplate='%{x}<br>ระยะเวลา: %{y:.0f} นาที<extra></extra>'),
        row=2, col=2
    )

    # Count
    cluster_counts = df['cluster'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=clusters, y=cluster_counts.values, marker_color='steelblue',
               hovertemplate='%{x}<br>จำนวน: %{y}<extra></extra>'),
        row=2, col=3
    )

    fig.update_layout(
        title_text='ลักษณะเฉลี่ยของแต่ละคลัสเตอร์',
        showlegend=False,
        template='plotly_white',
        height=600
    )

    return fig


def plot_cluster_by_district(df: pd.DataFrame) -> go.Figure:
    """
    Plot cluster distribution by district

    คำอธิบาย: แสดงการกระจายของคลัสเตอร์ในแต่ละเขต
    """
    cluster_district = pd.crosstab(df['district'], df['cluster'])

    fig = go.Figure()

    for cluster in cluster_district.columns:
        fig.add_trace(go.Bar(
            name=f'Cluster {cluster}',
            x=cluster_district.index,
            y=cluster_district[cluster],
            hovertemplate='<b>%{x}</b><br>Cluster ' + str(cluster) + ': %{y}<extra></extra>'
        ))

    fig.update_layout(
        title='การกระจายของคลัสเตอร์ในแต่ละเขต',
        xaxis_title='เขต',
        yaxis_title='จำนวนเหตุการณ์',
        barmode='stack',
        template='plotly_white',
        height=500,
        xaxis_tickangle=-45
    )

    return fig


def plot_cluster_by_day(df: pd.DataFrame) -> go.Figure:
    """
    Plot cluster distribution by day of week

    คำอธิบาย: แสดงการกระจายของคลัสเตอร์ตามวันในสัปดาห์
    """
    day_names = {0: 'จันทร์', 1: 'อังคาร', 2: 'พุธ', 3: 'พฤหัสบดี',
                 4: 'ศุกร์', 5: 'เสาร์', 6: 'อาทิตย์'}

    df['day_name'] = df['day_of_week'].map(day_names)
    cluster_day = pd.crosstab(df['day_name'], df['cluster'])

    # Reorder by day of week
    day_order = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์']
    cluster_day = cluster_day.reindex([d for d in day_order if d in cluster_day.index])

    fig = go.Figure()

    for cluster in cluster_day.columns:
        fig.add_trace(go.Bar(
            name=f'Cluster {cluster}',
            x=cluster_day.index,
            y=cluster_day[cluster],
            hovertemplate='<b>%{x}</b><br>Cluster ' + str(cluster) + ': %{y}<extra></extra>'
        ))

    fig.update_layout(
        title='การกระจายของคลัสเตอร์ตามวันในสัปดาห์',
        xaxis_title='วัน',
        yaxis_title='จำนวนเหตุการณ์',
        barmode='group',
        template='plotly_white',
        height=400
    )

    return fig


def plot_cluster_weather_correlation(df: pd.DataFrame) -> go.Figure:
    """
    Plot weather conditions for each cluster

    คำอธิบาย: แสดงความสัมพันธ์ระหว่างสภาพอากาศกับคลัสเตอร์
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('อุณหภูมิ vs ระยะเวลา', 'ฝน vs ระยะเวลา', 'ลม vs ระยะเวลา'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
    )

    clusters = sorted(df['cluster'].unique())
    colors = px.colors.qualitative.Plotly[:len(clusters)]

    for i, cluster in enumerate(clusters):
        cluster_data = df[df['cluster'] == cluster]

        # Temperature vs Duration
        fig.add_trace(
            go.Scatter(
                x=cluster_data['temp'],
                y=cluster_data['duration'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(color=colors[i], size=6, opacity=0.6),
                legendgroup=f'cluster{cluster}',
                hovertemplate='Temp: %{x:.1f}°C<br>Duration: %{y:.0f} min<extra></extra>'
            ),
            row=1, col=1
        )

        # Rain vs Duration
        fig.add_trace(
            go.Scatter(
                x=cluster_data['rain'],
                y=cluster_data['duration'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(color=colors[i], size=6, opacity=0.6),
                legendgroup=f'cluster{cluster}',
                showlegend=False,
                hovertemplate='Rain: %{x:.1f} mm<br>Duration: %{y:.0f} min<extra></extra>'
            ),
            row=1, col=2
        )

        # Wind vs Duration
        fig.add_trace(
            go.Scatter(
                x=cluster_data['wind_gust'],
                y=cluster_data['duration'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(color=colors[i], size=6, opacity=0.6),
                legendgroup=f'cluster{cluster}',
                showlegend=False,
                hovertemplate='Wind: %{x:.1f} km/h<br>Duration: %{y:.0f} min<extra></extra>'
            ),
            row=1, col=3
        )

    fig.update_xaxes(title_text="อุณหภูมิ (°C)", row=1, col=1)
    fig.update_xaxes(title_text="ฝน (mm)", row=1, col=2)
    fig.update_xaxes(title_text="ลม (km/h)", row=1, col=3)
    fig.update_yaxes(title_text="ระยะเวลา (นาที)", row=1, col=1)

    fig.update_layout(
        title_text='ความสัมพันธ์ระหว่างสภาพอากาศและระยะเวลาไฟดับ',
        template='plotly_white',
        height=500,
        hovermode='closest'
    )

    return fig


def render_cluster_summary(df: pd.DataFrame, cluster_id: int):
    """
    Render summary statistics for a specific cluster

    คำอธิบาย: แสดงสถิติสรุปสำหรับคลัสเตอร์ที่เลือก
    """
    cluster_data = df[df['cluster'] == cluster_id]

    if len(cluster_data) == 0:
        st.warning(f"ไม่พบข้อมูลในคลัสเตอร์ {cluster_id}")
        return

    st.markdown(f"### สรุปคลัสเตอร์ {cluster_id}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("จำนวนเหตุการณ์", f"{len(cluster_data):,}")

    with col2:
        avg_duration = cluster_data['duration'].mean()
        st.metric("ระยะเวลาเฉลี่ย", f"{avg_duration:.0f} นาที")

    with col3:
        avg_temp = cluster_data['temp'].mean()
        st.metric("อุณหภูมิเฉลี่ย", f"{avg_temp:.1f}°C")

    with col4:
        avg_rain = cluster_data['rain'].mean()
        st.metric("ปริมาณฝนเฉลี่ย", f"{avg_rain:.1f} mm")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**เขตที่พบบ่อย (Top 5)**")
        top_districts = cluster_data['district'].value_counts().head(5)
        for district, count in top_districts.items():
            st.write(f"- {district}: {count} ครั้ง")

    with col2:
        st.markdown("**วันที่พบบ่อย**")
        day_names = {0: 'จันทร์', 1: 'อังคาร', 2: 'พุธ', 3: 'พฤหัสบดี',
                     4: 'ศุกร์', 5: 'เสาร์', 6: 'อาทิตย์'}
        cluster_data_copy = cluster_data.copy()
        cluster_data_copy['day_name'] = cluster_data_copy['day_of_week'].map(day_names)
        top_days = cluster_data_copy['day_name'].value_counts().head(3)
        for day, count in top_days.items():
            st.write(f"- {day}: {count} ครั้ง")

    st.markdown("---")

    st.markdown("**ลักษณะเด่นของคลัสเตอร์นี้:**")

    avg_start_hour = cluster_data['start_min'].mean() / 60
    avg_duration_hours = cluster_data['duration'].mean() / 60

    st.write(f"- เวลาเริ่มไฟดับเฉลี่ย: {avg_start_hour:.1f} ชั่วโมง ({int(avg_start_hour)}:{int((avg_start_hour % 1) * 60):02d})")
    st.write(f"- ระยะเวลาไฟดับเฉลี่ย: {avg_duration_hours:.1f} ชั่วโมง")

    if avg_rain > 10:
        st.write(f"- คลัสเตอร์นี้มักเกิดในช่วงที่มีฝนตก (ฝนเฉลี่ย {avg_rain:.1f} mm)")

    if cluster_data['wind_gust'].mean() > 30:
        st.write(f"- คลัสเตอร์นี้มักเกิดในช่วงที่มีลมแรง (ลมเฉลี่ย {cluster_data['wind_gust'].mean():.1f} km/h)")
