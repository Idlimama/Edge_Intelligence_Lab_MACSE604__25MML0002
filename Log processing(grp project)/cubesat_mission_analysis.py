"""
CubeSat Telemetry: End-to-End Generation & Anomaly Detection
============================================================
This single script fulfills the entire assignment workflow:
1. Generates physics-aware synthetic telemetry data (15,000 logs).
2. Saves the log data to a JSON file.
3. Performs Advanced Anomaly Detection using 3 techniques:
   - Statistical (IQR / Boxplots)
   - Machine Learning (Isolation Forest)
   - Density-Based (DBSCAN)
4. Generates and saves high-quality visualization plots.

Requires: pandas, matplotlib, scikit-learn, numpy
Install via: pip install pandas matplotlib scikit-learn numpy
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("❌ Missing required machine learning libraries.")
    print("Please run: pip install scikit-learn")
    exit(1)

# ==============================================================================
# 1. DATA GENERATION ENGINE
# ==============================================================================
print("🚀 STEP 1: Generating CubeSat Telemetry Data...")

NUM_RECORDS = 15_000
TIME_INTERVAL_SEC = 60
START_TIME = datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc)
ORBIT_PERIOD_MIN = 92
DAY_FRACTION = 0.55
ANOMALY_RATE = 0.07
PROC_TEMP_CRITICAL = 85.0
BATTERY_TEMP_OFFSET = -8.0

random.seed(42)
np.random.seed(42)

def generate_telemetry():
    records = []
    processor_temps = []
    proc_temp = 50.0 + np.random.normal(0, 2)

    for i in range(NUM_RECORDS):
        timestamp = START_TIME + timedelta(seconds=i * TIME_INTERVAL_SEC)
        
        # Orbit Phase Setup
        pos_in_orbit = (i % ORBIT_PERIOD_MIN) / ORBIT_PERIOD_MIN
        sun_flag = 1 if pos_in_orbit < DAY_FRACTION else 0
        phase = "day" if sun_flag == 1 else "night"

        # AI Subsystem & CPU Generation
        ai_prob = 0.7 if sun_flag == 1 else 0.4
        ai_active = 1 if random.random() < ai_prob else 0
        cpu = np.clip(np.random.normal(0.75, 0.10) if ai_active else np.random.normal(0.15, 0.06), 0.02, 0.99)
        
        status = {
            "ai_model_active": ai_active,
            "camera_active": ai_active,
            "radio_active": 1 if random.random() < 0.20 else 0
        }

        # Physics Thermal Model (Mean-Reverting)
        equil_temp = 42.0 + 16.0 * sun_flag + 14.0 * cpu
        proc_temp += 0.08 * (equil_temp - proc_temp) + np.random.normal(0, 0.5)
        if proc_temp < 35: proc_temp = 35 + abs(np.random.normal(0, 0.5))

        # True Anomaly Injection (Ground Truth)
        is_anomaly = 0
        anomaly_boost = 0.0
        if random.random() < ANOMALY_RATE:
            is_anomaly = 1
            anomaly_type = random.choice(["spike", "rapid_rise"])
            anomaly_boost = np.random.uniform(10, 25) if anomaly_type == "spike" else np.random.uniform(6, 14)

        display_temp = proc_temp + anomaly_boost
        display_temp_rounded = round(display_temp, 2)

        # Build Record
        processor_temps.append(display_temp_rounded)
        temp_roc = round(processor_temps[-1] - processor_temps[-2], 3) if len(processor_temps) >= 2 else 0.0
        
        record = {
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "processor_temp": display_temp_rounded,
            "battery_temp": round(proc_temp + BATTERY_TEMP_OFFSET + np.random.normal(0, 1.5), 2),
            "cpu_usage": round(cpu, 4),
            "power_consumption": round(max(25 + cpu * 110 + status["radio_active"] * 15 + np.random.normal(0, 3), 10), 2),
            "sun_exposure_flag": sun_flag,
            "orbit_phase": phase,
            "temp_rate_of_change": temp_roc,
            "is_anomaly": is_anomaly # This is our ground-truth label!
        }
        records.append(record)
    
    return records

# Generate and convert to DataFrame
data = generate_telemetry()
df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Save to JSON to fulfill log requirement
json_path = "cubesat_telemetry_logs.json"
df.reset_index().to_json(json_path, orient="records", date_format="iso")
print(f"✅ Data generated and saved to '{json_path}'")

# ==============================================================================
# 2. LOG DATA ANALYSIS & ANOMALY DETECTION
# ==============================================================================
print("\n🧠 STEP 2: Running Anomaly Detection Models...")

# Prepare features for machine learning (CPU, Power, Processor Temp, Rate of Change)
features = ["processor_temp", "cpu_usage", "power_consumption", "temp_rate_of_change"]
X = df[features].values

# Standardize the data (crucial for distance-based algorithms like DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Technique A: Statistical (IQR) ---
print("  -> Running IQR (Statistical)...")
Q1, Q3 = df["processor_temp"].quantile(0.25), df["processor_temp"].quantile(0.75)
IQR = Q3 - Q1
df["iqr_anomaly"] = ((df["processor_temp"] < Q1 - 1.5 * IQR) | (df["processor_temp"] > Q3 + 1.5 * IQR)).astype(int)

# --- Technique B: Machine Learning (Isolation Forest) ---
# Isolation Forest isolates observations by randomly selecting a feature and split value.
print("  -> Running Isolation Forest (Machine Learning)...")
iso_forest = IsolationForest(contamination=ANOMALY_RATE, random_state=42, n_jobs=-1)
# IsolationForest returns -1 for outliers and 1 for inliers. We map -1 to 1 (anomaly) and 1 to 0 (normal).
df["iforest_anomaly"] = iso_forest.fit_predict(X_scaled)
df["iforest_anomaly"] = df["iforest_anomaly"].apply(lambda x: 1 if x == -1 else 0)

# --- Technique C: Density-Based (DBSCAN) ---
# Groups together points that are closely packed, marking points in low-density regions as outliers
print("  -> Running DBSCAN (Density-Based)...")
# eps and min_samples tuned for this normalized dataset
dbscan = DBSCAN(eps=1.2, min_samples=10) 
df["dbscan_anomaly"] = dbscan.fit_predict(X_scaled)
# DBSCAN returns -1 for noise/outliers
df["dbscan_anomaly"] = df["dbscan_anomaly"].apply(lambda x: 1 if x == -1 else 0)

# ==============================================================================
# 3. VISUALIZATION
# ==============================================================================
print("\n📊 STEP 3: Generating Visualizations...")

# Matplotlib dark theme configuration
plt.style.use('dark_background')
COLORS = {"normal": "#58a6ff", "anomaly": "#ff7eb6", "iqr": "#f85149", "iforest": "#3fb950", "dbscan": "#d29922"}

fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
fig.suptitle("CubeSat Edge Node: Temperature Log Analysis & Anomaly Detection", fontsize=18, fontweight='bold', y=0.92)

# Subplot 1: Ground Truth 
ax = axes[0]
ax.plot(df.index, df["processor_temp"], color=COLORS["normal"], linewidth=0.5, alpha=0.8, label="Processor Temp")
ground_truth = df[df["is_anomaly"] == 1]
ax.scatter(ground_truth.index, ground_truth["processor_temp"], color=COLORS["anomaly"], s=20, label=f"Ground Truth Anomalies ({len(ground_truth)})", zorder=5)
ax.axhline(PROC_TEMP_CRITICAL, color='red', linestyle='--', linewidth=1, label="Critical Threshold (85°C)")
ax.set_title("1. Raw Telemetry & Ground Truth Anomalies", fontsize=14)
ax.legend(loc="upper right")
ax.grid(alpha=0.2)
ax.set_ylabel("Temp (°C)")

# Subplot 2: IQR Results
ax = axes[1]
ax.plot(df.index, df["processor_temp"], color=COLORS["normal"], linewidth=0.5, alpha=0.5)
iqr_anomalies = df[df["iqr_anomaly"] == 1]
ax.scatter(iqr_anomalies.index, iqr_anomalies["processor_temp"], color=COLORS["iqr"], s=20, label=f"IQR Outliers ({len(iqr_anomalies)})", zorder=5)
ax.set_title("2. Statistical Detection: IQR Outlier Method", fontsize=14)
ax.legend(loc="upper right")
ax.grid(alpha=0.2)
ax.set_ylabel("Temp (°C)")

# Subplot 3: Isolation Forest Results
ax = axes[2]
ax.plot(df.index, df["processor_temp"], color=COLORS["normal"], linewidth=0.5, alpha=0.5)
if_anomalies = df[df["iforest_anomaly"] == 1]
ax.scatter(if_anomalies.index, if_anomalies["processor_temp"], color=COLORS["iforest"], s=20, marker='x', label=f"Isolation Forest Anomalies ({len(if_anomalies)})", zorder=5)
ax.set_title("3. Machine Learning Detection: Isolation Forest (Multi-variate)", fontsize=14)
ax.legend(loc="upper right")
ax.grid(alpha=0.2)
ax.set_ylabel("Temp (°C)")

# Subplot 4: DBSCAN Results
ax = axes[3]
ax.plot(df.index, df["processor_temp"], color=COLORS["normal"], linewidth=0.5, alpha=0.5)
db_anomalies = df[df["dbscan_anomaly"] == 1]
ax.scatter(db_anomalies.index, db_anomalies["processor_temp"], color=COLORS["dbscan"], s=20, marker='^', label=f"DBSCAN Outliers ({len(db_anomalies)})", zorder=5)
ax.set_title("4. Density-Based Detection: DBSCAN (Multi-variate clustering)", fontsize=14)
ax.legend(loc="upper right")
ax.grid(alpha=0.2)
ax.set_xlabel("Timestamp")
ax.set_ylabel("Temp (°C)")

# Save plot
plot_path = "anomaly_detection_results.png"
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(plot_path, dpi=150)
print(f"✅ Detection Results Plot saved to '{plot_path}'")
plt.close()

# Generate a 2D Scatter plot to show exactly HOW Isolation Forest finds multi-dimensional anomalies
plt.figure(figsize=(10, 8))
plt.scatter(df["cpu_usage"], df["processor_temp"], c=COLORS["normal"], s=5, alpha=0.4, label="Normal")
plt.scatter(if_anomalies["cpu_usage"], if_anomalies["processor_temp"], c=COLORS["iforest"], s=30, marker='x', label="Isolation Forest Anomaly")
plt.axhline(PROC_TEMP_CRITICAL, color='red', linestyle='--', linewidth=1, label="Critical Temp Threshold")
plt.title("Isolation Forest Decision Space (CPU Usage vs. Temperature)", fontsize=14, fontweight='bold')
plt.xlabel("CPU Usage")
plt.ylabel("Processor Temp (°C)")
plt.legend()
plt.grid(alpha=0.2)
scatter_path = "isolation_forest_scatter.png"
plt.savefig(scatter_path, dpi=150)
print(f"✅ ML Scatter Plot saved to '{scatter_path}'")
plt.close()

print("\n🎉 Assignment execution complete! You can view the generated images.")
