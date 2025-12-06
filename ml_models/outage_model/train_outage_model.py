#ไอเดียโมเดล
#แบ่งเหตุไฟดับออกเป็น “กลุ่มพฤติกรรมคล้ายกัน เช่น คลัสเตอร์ไฟดับตอนเช้า-วันทำงาน

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


# ================================
# 1) Load data
# ================================
df = pd.read_csv("../data/clean_scraping_data.csv")

# ตรวจดูว่าคอลัมน์หลัก ๆ มีอยู่ไหม
required_cols = [
    "date", "day_of_week", "start", "end",
    "district", "temp", "rain", "wind_gust"
]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")


# ================================
# 2) Feature Engineering
# ================================

# แปลงเวลาเป็นนาทีของวัน
def time_to_minutes(t):
    """
    t: string format HH:MM
    """
    h, m = map(int, t.split(":"))
    return h * 60 + m

df["start_min"] = df["start"].astype(str).apply(time_to_minutes)
df["end_min"] = df["end"].astype(str).apply(time_to_minutes)

# duration ถ้า end < start (ข้ามเที่ยงคืน) ก็ + 24 ชั่วโมง
df["duration"] = df["end_min"] - df["start_min"]
df.loc[df["duration"] < 0, "duration"] += 24 * 60

# เลือกฟีเจอร์สำหรับ clustering
feature_cols = [
    "day_of_week",
    "district",
    "temp",
    "rain",
    "wind_gust",
    "start_min",
    "duration",
]

X = df[feature_cols].copy()

# column types
cat_cols = ["day_of_week", "district"]
num_cols = ["temp", "rain", "wind_gust", "start_min", "duration"]

# Preprocessor: OneHot encoding + numerical passthrough
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# ================================
# 3) Train-test split 80/20
# ================================
X_train, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")


# ================================
# 4) เลือกจำนวนคลัสเตอร์ k ที่เหมาะสม
#    ด้วย Silhouette Score บน train set
# ================================
def find_best_k(X_train, preprocessor, k_min=2, k_max=10):
    best_k = None
    best_score = -1

    for k in range(k_min, k_max + 1):
        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("cluster", KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            ))
        ])

        model.fit(X_train)
        # transform แล้วค่อยคำนวณ silhouette
        X_train_trans = model.named_steps["preprocess"].transform(X_train)
        labels = model.named_steps["cluster"].labels_
        score = silhouette_score(X_train_trans, labels)

        print(f"k={k}  ->  silhouette score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score


best_k, best_score = find_best_k(X_train, preprocessor, k_min=2, k_max=8)

print("\n==============================")
print(f"Best k on TRAIN: {best_k}  (silhouette = {best_score:.4f})")
print("==============================\n")


# ================================
# 5) สร้างโมเดลสุดท้ายด้วย k ที่ดีที่สุด
# ================================
final_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("cluster", KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    ))
])

final_model.fit(X_train)

# ================================
# 6) ประเมินโมเดลบน TRAIN และ TEST
# ================================
# --- Train ---
X_train_trans = final_model.named_steps["preprocess"].transform(X_train)
train_labels = final_model.named_steps["cluster"].labels_
train_silhouette = silhouette_score(X_train_trans, train_labels)

# --- Test ---
X_test_trans = final_model.named_steps["preprocess"].transform(X_test)
test_labels = final_model.named_steps["cluster"].predict(X_test_trans)
test_silhouette = silhouette_score(X_test_trans, test_labels)

print("=== Model Evaluation ===")
print(f"Train Silhouette Score: {train_silhouette:.4f}")
print(f"Test  Silhouette Score: {test_silhouette:.4f}")


# ================================
# 7) วิเคราะห์คลัสเตอร์คร่าวๆ
# ================================
# ใส่ label กลับไปใน df (เฉพาะแถว train-before-split หรือทั้ง df)
all_trans = final_model.named_steps["preprocess"].transform(X)
all_labels = final_model.named_steps["cluster"].predict(all_trans)

df["cluster"] = all_labels

print("\nCluster counts:")
print(df["cluster"].value_counts().sort_index())

print("\nAverage features by cluster:")
print(
    df.groupby("cluster")[["temp", "rain", "wind_gust", "start_min", "duration"]]
    .mean()
    .round(2)
)


# ================================
# 8) Save model & clustered data
# ================================
joblib.dump(final_model, "ml_models/outage_model/model/outage_kmeans_model.pkl")
df.to_csv("ml_models/outage_model/model/power_outage_with_clusters.csv", index=False)

print("\nModel saved as: outage_kmeans_model.pkl")
print("Clustered data saved as: power_outage_with_clusters.csv")
