"""
Visualization Modules for Urban Issue Dashboard
Contains reusable visualization functions for district and complaint analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_complaints_by_district(df: pd.DataFrame, selected_complaint: str = 'All') -> go.Figure:
    """
    กราฟแสดงแต่ละเขตมี complaint อะไรบ้าง จำนวนเท่าไหร่

    คำอธิบาย: แสดงจำนวน complaint แยกตามประเภทในแต่ละเขต
    ช่วยให้เห็นว่าแต่ละเขตมีปัญหาประเภทใดบ้าง และปัญหาไหนที่พบมากที่สุด
    """
    df_filtered = df.copy()

    # Filter by selected complaint type if specified
    if selected_complaint != 'All':
        df_filtered = df_filtered[df_filtered['primary_type'] == selected_complaint]

    # Get top districts and complaint types for better visualization
    top_districts = df_filtered['district'].value_counts().head(15).index
    df_filtered = df_filtered[df_filtered['district'].isin(top_districts)]

    # Count complaints by district and type
    district_type_counts = df_filtered.groupby(['district', 'primary_type']).size().reset_index(name='count')

    # Get top 10 complaint types
    top_types = df_filtered['primary_type'].value_counts().head(10).index
    district_type_counts = district_type_counts[district_type_counts['primary_type'].isin(top_types)]

    fig = px.bar(
        district_type_counts,
        x='district',
        y='count',
        color='primary_type',
        title=f'จำนวน Complaint แต่ละประเภทในแต่ละเขต{" - " + selected_complaint if selected_complaint != "All" else ""}',
        labels={'district': 'เขต', 'count': 'จำนวน Complaint', 'primary_type': 'ประเภท Complaint'},
        barmode='stack',
        height=500
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    return fig


def plot_complaint_distribution_across_districts(df: pd.DataFrame, selected_district: str = 'All') -> go.Figure:
    """
    กราฟแสดงแต่ละ complaint มีในเขตไหนบ้าง เขตละเท่าไหร่

    คำอธิบาย: แสดงการกระจายตัวของแต่ละประเภท complaint ในแต่ละเขต
    ช่วยระบุว่าปัญหาแต่ละประเภทเกิดในเขตใดบ้าง
    """
    df_filtered = df.copy()

    # Filter by selected district if specified
    if selected_district != 'All':
        df_filtered = df_filtered[df_filtered['district'] == selected_district]

    # Get top complaint types
    top_types = df_filtered['primary_type'].value_counts().head(10).index
    df_filtered = df_filtered[df_filtered['primary_type'].isin(top_types)]

    # Count by complaint type and district
    type_district_counts = df_filtered.groupby(['primary_type', 'district']).size().reset_index(name='count')

    # Get top districts for each complaint type
    top_districts_per_type = type_district_counts.groupby('primary_type').apply(
        lambda x: x.nlargest(10, 'count')
    ).reset_index(drop=True)

    fig = px.bar(
        top_districts_per_type,
        x='primary_type',
        y='count',
        color='district',
        title=f'การกระจายตัวของแต่ละ Complaint ในแต่ละเขต{" - " + selected_district if selected_district != "All" else ""}',
        labels={'primary_type': 'ประเภท Complaint', 'count': 'จำนวน', 'district': 'เขต'},
        barmode='stack',
        height=500
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    return fig


def plot_top_complaint_districts(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    กราฟเขตไหน complaint เยอะสุด

    คำอธิบาย: จัดอันดับเขตที่มีจำนวน complaint มากที่สุด
    ช่วยระบุพื้นที่ที่ต้องให้ความสนใจเป็นพิเศษ
    """
    district_counts = df['district'].value_counts().head(top_n)

    # Calculate percentage
    total = len(df)
    percentages = (district_counts / total * 100).round(2)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=district_counts.index,
        y=district_counts.values,
        text=[f'{count:,}<br>({pct}%)' for count, pct in zip(district_counts.values, percentages.values)],
        textposition='outside',
        marker=dict(
            color=district_counts.values,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title='จำนวน')
        ),
        hovertemplate='<b>%{x}</b><br>จำนวน: %{y:,}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Top {top_n} เขตที่มี Complaint มากที่สุด',
        xaxis_title='เขต',
        yaxis_title='จำนวน Complaint',
        template='plotly_white',
        height=500,
        xaxis_tickangle=-45
    )

    return fig


def plot_complaint_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap แสดงความเข้มของ complaint แต่ละเขตในแต่ละเดือน

    คำอธิบาย: แสดงแนวโน้มของ complaint ตามเวลาในแต่ละเขต
    ช่วยระบุรูปแบบตามฤดูกาลหรือช่วงเวลาที่มีปัญหามากเป็นพิเศษ
    """
    df_copy = df.copy()
    df_copy['month_year'] = df_copy['timestamp'].dt.to_period('M').astype(str)

    # Create pivot table
    pivot = df_copy.pivot_table(
        index='district',
        columns='month_year',
        aggfunc='size',
        fill_value=0
    )

    # Limit to top districts and recent months
    pivot = pivot.nlargest(20, pivot.columns[-1])
    pivot = pivot[pivot.columns[-12:]]  # Last 12 months

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='YlOrRd',
        hovertemplate='เขต: %{y}<br>เดือน: %{x}<br>จำนวน: %{z}<extra></extra>',
        colorbar=dict(title='จำนวน<br>Complaint')
    ))

    fig.update_layout(
        title='ความเข้มของ Complaint แต่ละเขตในแต่ละเดือน (12 เดือนล่าสุด)',
        xaxis_title='เดือน',
        yaxis_title='เขต',
        template='plotly_white',
        height=600
    )

    return fig


def plot_complaint_types_pie(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Pie chart แสดงสัดส่วนของ complaint แต่ละประเภท

    คำอธิบาย: แสดงสัดส่วนการกระจายของปัญหาแต่ละประเภท
    ช่วยให้เห็นภาพรวมว่าปัญหาประเภทใดเป็นปัญหาหลัก
    """
    complaint_counts = df['primary_type'].value_counts().head(top_n)

    fig = go.Figure(data=[go.Pie(
        labels=complaint_counts.index,
        values=complaint_counts.values,
        hole=0.4,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>จำนวน: %{value:,}<br>สัดส่วน: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=f'สัดส่วนประเภท Complaint (Top {top_n})',
        template='plotly_white',
        height=500
    )

    return fig


def plot_resolution_time_by_district(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Box plot แสดงระยะเวลาในการแก้ปัญหาแต่ละเขต

    คำอธิบาย: แสดงการกระจายของเวลาแก้ปัญหาในแต่ละเขต
    ช่วยระบุเขตที่ใช้เวลานานหรือมีความแปรปรวนสูง
    """
    # Get top districts by complaint count
    top_districts = df['district'].value_counts().head(top_n).index
    df_filtered = df[df['district'].isin(top_districts)].copy()

    # Remove outliers (>99th percentile)
    percentile_99 = df_filtered['solve_days'].quantile(0.99)
    df_filtered = df_filtered[df_filtered['solve_days'] <= percentile_99]

    fig = px.box(
        df_filtered,
        x='district',
        y='solve_days',
        color='district',
        title=f'ระยะเวลาในการแก้ปัญหาแต่ละเขต (Top {top_n})',
        labels={'district': 'เขต', 'solve_days': 'จำนวนวัน'},
        height=500
    )

    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        xaxis_tickangle=-45
    )

    return fig


def plot_time_series_comparison(df: pd.DataFrame, districts: list = None) -> go.Figure:
    """
    เปรียบเทียบแนวโน้ม complaint ของหลายเขตตามเวลา

    คำอธิบาย: แสดงแนวโน้มจำนวน complaint ของเขตต่างๆ
    ช่วยเปรียบเทียบการเปลี่ยนแปลงตามเวลาระหว่างเขต
    """
    if districts is None or len(districts) == 0:
        # Get top 5 districts by default
        districts = df['district'].value_counts().head(5).index.tolist()

    df_filtered = df[df['district'].isin(districts)].copy()
    df_filtered['date'] = df_filtered['timestamp'].dt.date

    # Count by date and district
    daily_counts = df_filtered.groupby(['date', 'district']).size().reset_index(name='count')

    fig = px.line(
        daily_counts,
        x='date',
        y='count',
        color='district',
        title='เปรียบเทียบแนวโน้ม Complaint ของแต่ละเขตตามเวลา',
        labels={'date': 'วันที่', 'count': 'จำนวน Complaint', 'district': 'เขต'},
        height=500
    )

    fig.update_layout(
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def plot_hourly_pattern(df: pd.DataFrame) -> go.Figure:
    """
    แสดงรูปแบบการรายงาน complaint ตามช่วงเวลาในวัน

    คำอธิบาย: แสดงว่าช่วงเวลาไหนของวันมีการรายงานปัญหามากที่สุด
    ช่วยวางแผนการจัดทีมงานให้เหมาะสม
    """
    hourly_counts = df.groupby('hour').size()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hourly_counts.index,
        y=hourly_counts.values,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='ชั่วโมง: %{x}:00<br>จำนวน: %{y:,}<extra></extra>'
    ))

    fig.update_layout(
        title='รูปแบบการรายงาน Complaint ตามช่วงเวลาในวัน',
        xaxis_title='ชั่วโมง (0-23)',
        yaxis_title='จำนวน Complaint',
        template='plotly_white',
        height=400,
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2
        )
    )

    return fig


def plot_weekday_pattern(df: pd.DataFrame) -> go.Figure:
    """
    แสดงรูปแบบการรายงาน complaint ตามวันในสัปดาห์

    คำอธิบาย: แสดงว่าวันไหนในสัปดาห์มีการรายงานปัญหามากที่สุด
    ช่วยระบุรูปแบบการรายงานตามวัน
    """
    weekday_counts = df.groupby('day_of_week').size()
    weekday_names = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=weekday_names,
        y=weekday_counts.values,
        marker=dict(
            color=weekday_counts.values,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='จำนวน')
        ),
        text=[f'{v:,}' for v in weekday_counts.values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>จำนวน: %{y:,}<extra></extra>'
    ))

    fig.update_layout(
        title='รูปแบบการรายงาน Complaint ตามวันในสัปดาห์',
        xaxis_title='วัน',
        yaxis_title='จำนวน Complaint',
        template='plotly_white',
        height=400
    )

    return fig


def plot_state_distribution(df: pd.DataFrame) -> go.Figure:
    """
    แสดงสถานะของ complaint (รอรับเรื่อง, กำลังดำเนินการ, เสร็จสิ้น)

    คำอธิบาย: แสดงสัดส่วนสถานะการดำเนินการของปัญหา
    ช่วยตรวจสอบประสิทธิภาพการแก้ปัญหา
    """
    state_counts = df['state'].value_counts()

    colors = {
        'เสร็จสิ้น': '#2ecc71',
        'กำลังดำเนินการ': '#f39c12',
        'รอรับเรื่อง': '#e74c3c',
        'Unknown': '#95a5a6'
    }

    fig = go.Figure(data=[go.Pie(
        labels=state_counts.index,
        values=state_counts.values,
        marker=dict(colors=[colors.get(state, '#95a5a6') for state in state_counts.index]),
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>จำนวน: %{value:,}<br>สัดส่วน: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title='สัดส่วนสถานะการดำเนินการ',
        template='plotly_white',
        height=400
    )

    return fig
