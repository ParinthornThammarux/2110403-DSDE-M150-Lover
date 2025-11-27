# app.py
import os
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

from complaint_network import build_complaint_network

import folium
from folium.plugins import MarkerCluster


st.set_page_config(
    page_title="Urban Complaint Network",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔗 Urban Complaint Network & Map")
st.markdown(
    """
Visualization แสดง **ความสัมพันธ์ของประเภทปัญหา** (Graph Network) 
พร้อมกับ **แผนที่ตำแหน่งเรื่องร้องเรียน** แยกตามพื้นที่ (เขต/แขวง)
"""
)

# -------------------------------------------------
# Sidebar: โหลดข้อมูล + ตั้งค่าพื้นที่ + parameter
# -------------------------------------------------
st.sidebar.header("⚙️ Settings")

uploaded = st.sidebar.file_uploader("📂 อัปโหลด Traffy CSV", type=["csv"])

if not uploaded:
    st.info("กรุณาอัปโหลดไฟล์ Traffy CSV เพื่อเริ่มต้นใช้งาน")
    st.stop()

# อ่านข้อมูล
df = pd.read_csv(uploaded)
# ตัด row ที่ไม่มี type
df = df.dropna(subset=["data_type"])

st.sidebar.success(f"โหลดข้อมูลสำเร็จ: {len(df):,} records")

# เลือกโหมดพื้นที่
area_mode = st.sidebar.selectbox(
    "โหมดการเลือกพื้นที่",
    ["ทั้งเมือง", "ตามเขต (district)", "ตามแขวง (subdistrict)"],
)

area_col = None
area_value = None

if area_mode == "ตามเขต (district)":
    if "district" in df.columns:
        districts = sorted(df["district"].dropna().unique().tolist())
        area_value = st.sidebar.selectbox("เลือกเขต", ["(ทั้งหมด)"] + districts)
        if area_value != "(ทั้งหมด)":
            area_col = "district"
        else:
            area_value = None
    else:
        st.sidebar.warning("ไม่พบคอลัมน์ 'district' ในไฟล์ CSV")

elif area_mode == "ตามแขวง (subdistrict)":
    if "subdistrict" in df.columns:
        subs = sorted(df["subdistrict"].dropna().unique().tolist())
        area_value = st.sidebar.selectbox("เลือกแขวง", ["(ทั้งหมด)"] + subs)
        if area_value != "(ทั้งหมด)":
            area_col = "subdistrict"
        else:
            area_value = None
    else:
        st.sidebar.warning("ไม่พบคอลัมน์ 'subdistrict' ในไฟล์ CSV")


min_occ = st.sidebar.slider(
    "ขั้นต่ำจำนวนครั้งที่ประเภทต้องเกิดร่วมกัน (ใช้สร้าง edge ใน network)",
    5,
    100,
    20,
)

# -------------------------------------------------
# เตรียม DataFrame ที่ filter ตามพื้นที่
# -------------------------------------------------
df_area = df.copy()
area_label = "ทั้งเมือง"
if area_col and area_value:
    df_area = df_area[df_area[area_col] == area_value]
    area_label = f"{area_col}: {area_value}"

if df_area.empty:
    st.error("ไม่มีข้อมูลหลังจาก filter ตามพื้นที่ — ลองเลือกพื้นที่อื่นหรือลดเงื่อนไข")
    st.stop()

st.markdown(f"**พื้นที่ที่เลือก:** {area_label}  \nจำนวน records: **{len(df_area):,}**")

# -------------------------------------------------
# Tabs: Network / Map
# -------------------------------------------------
tab_net, tab_map = st.tabs(["🔗 Network Graph", "🗺️ Map View"])

# -------------------------------------------------
# TAB 1: Network Graph
# -------------------------------------------------
with tab_net:
    st.subheader("🔗 Complaint Type Relationship Network")

    try:
        net, summary = build_complaint_network(
            df_area,
            min_co_occurrence=min_occ,
            area_col=area_col,
            area_value=area_value,
        )
    except ValueError as e:
        st.error(str(e))
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("จำนวน ticket", f"{summary['num_tickets']:,}")
        col2.metric("จำนวนประเภทปัญหา", f"{summary['num_types']:,}")
        col3.metric("จำนวน edges", f"{summary['num_edges']:,}")
        col4.metric("ขั้นต่ำ co-occurrence", summary["min_co_occurrence"])

        # save html แล้ว embed ใน streamlit
        net_file = "network.html"
        net.save_graph(net_file)

        with open(net_file, "r", encoding="utf-8") as f:
            html(f.read(), height=700)

# -------------------------------------------------
# TAB 2: Map View (Folium)
# -------------------------------------------------
def parse_coords_to_latlon(coords_str: str):
    """
    รับค่า string "lat,lon" → return (lat, lon) แบบ float
    ถ้า parse ไม่ได้ ให้ return None
    """
    try:
        if isinstance(coords_str, str):
            parts = coords_str.split(",")
            if len(parts) != 2:
                return None
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            return lat, lon
    except Exception:
        return None
    return None


with tab_map:
    st.subheader("🗺️ Complaint Map")

    if "coords" not in df_area.columns:
        st.warning("ไม่พบคอลัมน์ 'coords' ในข้อมูล จึงแสดงแผนที่ไม่ได้")
    else:
        # แปลง coords เป็น lat, lon
        coords_parsed = df_area["coords"].apply(parse_coords_to_latlon)
        df_map = df_area.copy()
        df_map["lat"] = coords_parsed.apply(lambda x: x[0] if x else None)
        df_map["lon"] = coords_parsed.apply(lambda x: x[1] if x else None)
        df_map = df_map.dropna(subset=["lat", "lon"])

        if df_map.empty:
            st.warning("ไม่มีข้อมูลตำแหน่ง (coords) ที่ parse ได้")
        else:
            # จุดกลางสำหรับ map
            center_lat = df_map["lat"].mean()
            center_lon = df_map["lon"].mean()

            m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
            marker_cluster = MarkerCluster().add_to(m)

            # limit จำนวน marker ถ้า data ใหญ่มาก
            max_points = 5000
            df_show = df_map.head(max_points)

            for _, row in df_show.iterrows():
                popup_items = []
                if "type" in row:
                    popup_items.append(f"<b>Type:</b> {row['type']}")
                if "district" in row and pd.notna(row["district"]):
                    popup_items.append(f"<b>District:</b> {row['district']}")
                if "subdistrict" in row and pd.notna(row["subdistrict"]):
                    popup_items.append(f"<b>Subdistrict:</b> {row['subdistrict']}")
                if "timestamp" in row and pd.notna(row["timestamp"]):
                    popup_items.append(f"<b>Time:</b> {row['timestamp']}")
                popup_html = "<br>".join(popup_items)

                folium.CircleMarker(
                    location=(row["lat"], row["lon"]),
                    radius=4,
                    popup=folium.Popup(popup_html, max_width=300),
                ).add_to(marker_cluster)

            st.caption(f"แสดงสูงสุด {len(df_show):,} จุด จาก {len(df_map):,} records")

            # แสดง map ใน streamlit
            from streamlit.components.v1 import html as st_html

            st_html(m._repr_html_(), height=700)
