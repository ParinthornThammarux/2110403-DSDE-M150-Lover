# complaint_network.py
import pandas as pd
from pyvis.network import Network
from itertools import combinations


def build_complaint_network(
    df: pd.DataFrame,
    min_co_occurrence: int = 20,
    area_col: str | None = None,
    area_value: str | None = None,
):
    """
    สร้าง network ของประเภทปัญหาที่พบร่วมกันใน ticket เดียวกัน
    + รองรับ filter ตามพื้นที่ (เช่น district / subdistrict)

    Parameters:
        df: DataFrame ที่มีคอลัมน์ 'type' และอาจมี 'district', 'subdistrict'
        min_co_occurrence: จำนวนครั้งขั้นต่ำที่ 2 ประเภทจะถูกเชื่อมโยง
        area_col: ชื่อคอลัมน์พื้นที่ เช่น 'district' หรือ 'subdistrict'
        area_value: ค่าพื้นที่ที่ต้องการ filter

    Returns:
        PyVis Network object, dict summary
    """

    df = df.copy()

    # -----------------------------
    # Filter ตามพื้นที่ (ถ้าระบุ)
    # -----------------------------
    area_label = "ทั้งเมือง"
    if area_col and area_value:
        df = df[df[area_col] == area_value]
        area_label = f"{area_col} = {area_value}"

    if df.empty:
        raise ValueError("ไม่มีข้อมูลหลังจาก filter ตามพื้นที่")

    # -----------------------------
    # เตรียม type list (รองรับหลาย label ใน 1 ticket)
    # -----------------------------
    df["type_list"] = df["data_type"].astype(str).apply(
        lambda x: [t.strip() for t in x.split(",") if t.strip()]
    )

    # -----------------------------
    # นับ co-occurrence
    # -----------------------------
    pair_count: dict[tuple[str, str], int] = {}

    for types in df["type_list"]:
        if len(types) < 2:
            continue
        for p in combinations(sorted(set(types)), 2):
            pair_count[p] = pair_count.get(p, 0) + 1

    # -----------------------------
    # สร้าง PyVis graph
    # -----------------------------
    net = Network(
        height="650px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=False,
    )
    net.barnes_hut()  # physics layout

    # nodes
    unique_types = sorted({t for sublist in df["type_list"] for t in sublist})
    for t in unique_types:
        net.add_node(t, label=t)

    # edges
    edge_used = 0
    for (t1, t2), count in pair_count.items():
        if count >= min_co_occurrence:
            net.add_edge(t1, t2, value=count, title=f"{count} co-occurrences")
            edge_used += 1

    summary = {
        "area_label": area_label,
        "num_tickets": len(df),
        "num_types": len(unique_types),
        "num_edges": edge_used,
        "min_co_occurrence": min_co_occurrence,
    }

    return net, summary
