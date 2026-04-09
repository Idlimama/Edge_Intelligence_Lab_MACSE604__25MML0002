"""
CubeSat Telemetry - Visualization & Outlier Analysis
=====================================================
Comprehensive plots for the 15,000-sample synthetic dataset.
Run: python3 cubesat_visualization.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────
# 0. Load Data
# ────────────────────────────────────────────────────
print("Loading telemetry data...")
with open("cubesat_telemetry_15000.json") as f:
    raw = json.load(f)

df = pd.json_normalize(raw)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Separate normal vs anomaly
normal = df[df["is_anomaly"] == 0]
anomaly = df[df["is_anomaly"] == 1]

print(f"Loaded {len(df)} records  |  Normal: {len(normal)}  |  Anomalies: {len(anomaly)}")

# ────────────────────────────────────────────────────
# Style Configuration
# ────────────────────────────────────────────────────
BG       = "#0d1117"
CARD_BG  = "#161b22"
TEXT     = "#c9d1d9"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"
CYAN     = "#39d2c0"
PINK     = "#ff7eb6"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   TEXT,
    "text.color":        TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "grid.color":        "#21262d",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "font.size":         9,
})

THRESHOLD = 85.0  # Critical temp °C


# ═══════════════════════════════════════════════════════
# PLOT 1 — Processor Temperature Time-Series with Anomalies
# ═══════════════════════════════════════════════════════
print("[1/8] Processor Temperature Time-Series...")
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(df.index, df["processor_temp"], color=ACCENT, linewidth=0.3, alpha=0.7, label="Processor Temp")
ax.scatter(anomaly.index, anomaly["processor_temp"], c=RED, s=12, zorder=5,
           edgecolors="white", linewidths=0.3, label=f"Anomalies ({len(anomaly)})")
ax.axhline(y=THRESHOLD, color=ORANGE, linestyle="--", linewidth=1.2, label=f"Critical Threshold ({THRESHOLD}°C)")

ax.fill_between(df.index, THRESHOLD, df["processor_temp"].max() + 5,
                alpha=0.08, color=RED, label="Danger Zone")

ax.set_title("🛰️  CubeSat Processor Temperature — Full Mission Timeline", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Temperature (°C)")
ax.set_xlabel("Timestamp")
ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot1_temp_timeseries.png", dpi=180, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════
# PLOT 2 — Zoomed Anomaly Window (first 500 records)
# ═══════════════════════════════════════════════════════
print("[2/8] Zoomed Anomaly Window...")
fig, ax = plt.subplots(figsize=(16, 4))

window = df.iloc[:500]
w_normal = window[window["is_anomaly"] == 0]
w_anomaly = window[window["is_anomaly"] == 1]

ax.plot(window.index, window["processor_temp"], color=CYAN, linewidth=0.8, alpha=0.9)
ax.scatter(w_anomaly.index, w_anomaly["processor_temp"], c=RED, s=40, zorder=5,
           marker="^", edgecolors="white", linewidths=0.5, label="Anomaly Spike")
ax.axhline(y=THRESHOLD, color=ORANGE, linestyle="--", linewidth=1)

# Shade day/night
for i in range(len(window) - 1):
    if window.iloc[i]["orbit_phase"] == "night":
        ax.axvspan(window.index[i], window.index[i + 1], alpha=0.06, color="blue")

ax.set_title("🔍  Zoomed View — First 500 Minutes (Day/Night Shading)", fontsize=12, fontweight="bold", pad=10)
ax.set_ylabel("Processor Temp (°C)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_zoomed_anomalies.png", dpi=180, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════
# PLOT 3 — Distribution Panel (4 subplots)
# ═══════════════════════════════════════════════════════
print("[3/8] Distribution Panel...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 3a: Processor Temp Distribution (right-skewed)
ax = axes[0, 0]
ax.hist(normal["processor_temp"], bins=60, color=ACCENT, alpha=0.7, edgecolor="#30363d", label="Normal")
ax.hist(anomaly["processor_temp"], bins=30, color=RED, alpha=0.8, edgecolor="#30363d", label="Anomaly")
ax.axvline(THRESHOLD, color=ORANGE, linestyle="--", linewidth=1.5, label="85°C Threshold")
ax.set_title("Processor Temp Distribution (Right-Skewed)", fontsize=10, fontweight="bold")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=7)

# 3b: CPU Usage Distribution (bimodal)
ax = axes[0, 1]
ax.hist(df["cpu_usage"], bins=50, color=PURPLE, alpha=0.8, edgecolor="#30363d")
ax.axvline(0.4, color=ORANGE, linestyle="--", linewidth=1, label="Idle/Active Split")
ax.set_title("CPU Usage Distribution (Bimodal)", fontsize=10, fontweight="bold")
ax.set_xlabel("CPU Usage")
ax.legend(fontsize=7)

# 3c: Temp Rate of Change (centred near 0)
ax = axes[1, 0]
ax.hist(df["temp_rate_of_change"], bins=80, color=CYAN, alpha=0.8, edgecolor="#30363d")
ax.axvline(0, color=RED, linestyle="-", linewidth=1, alpha=0.5)
ax.set_title("Temp Rate of Change (Centered ~0)", fontsize=10, fontweight="bold")
ax.set_xlabel("°C / minute")

# 3d: Sun Exposure Flag (binary uniform)
ax = axes[1, 1]
counts = df["sun_exposure_flag"].value_counts().sort_index()
bars = ax.bar(["Night (0)", "Day (1)"], counts.values, color=[PURPLE, ORANGE], edgecolor="#30363d", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
            str(val), ha="center", fontsize=10, fontweight="bold", color=TEXT)
ax.set_title("Sun Exposure Flag (Approx. Uniform)", fontsize=10, fontweight="bold")
ax.set_ylabel("Count")

for a in axes.flat:
    a.grid(True, alpha=0.2)

plt.suptitle("📊  Feature Distribution Analysis", fontsize=14, fontweight="bold", y=1.02, color=TEXT)
plt.tight_layout()
plt.savefig("plot3_distributions.png", dpi=180, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════
# PLOT 4 — Box Plot Outlier Detection (IQR Method)
# ═══════════════════════════════════════════════════════
print("[4/8] Box Plot Outlier Detection...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

features_box = [
    ("processor_temp", "Processor Temp (°C)", ACCENT),
    ("battery_temp", "Battery Temp (°C)", GREEN),
    ("power_consumption", "Power (W)", ORANGE),
]

for ax, (col, label, color) in zip(axes, features_box):
    bp = ax.boxplot(df[col], vert=True, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.4, edgecolor=color),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    medianprops=dict(color=RED, linewidth=2),
                    flierprops=dict(marker="o", markerfacecolor=RED, markersize=3, alpha=0.5))
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Count IQR outliers
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    ax.text(0.95, 0.95, f"IQR Outliers: {outliers}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_BG, edgecolor=RED, alpha=0.8))

plt.suptitle("📦  Box Plot — Outlier Detection (IQR Method)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plot4_boxplots.png", dpi=180, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════
# PLOT 5 — Scatter: CPU vs Temp (coloured by anomaly)
# ═══════════════════════════════════════════════════════
print("[5/8] CPU vs Temperature Scatter...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(normal["cpu_usage"], normal["processor_temp"], c=ACCENT, s=4, alpha=0.3, label="Normal")
ax.scatter(anomaly["cpu_usage"], anomaly["processor_temp"], c=RED, s=18, alpha=0.8,
           edgecolors="white", linewidths=0.3, marker="D", label="Anomaly")
ax.axhline(THRESHOLD, color=ORANGE, linestyle="--", linewidth=1, label="85°C Threshold")

ax.set_title("🔬  CPU Usage vs Processor Temperature", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("CPU Usage")
ax.set_ylabel("Processor Temperature (°C)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("plot5_cpu_vs_temp.png", dpi=180, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════
# PLOT 6 — Correlation Heatmap
# ═══════════════════════════════════════════════════════
print("[6/8] Correlation Heatmap...")
corr_cols = ["processor_temp", "battery_temp", "solar_panel_temp", "cpu_usage",
             "power_consumption", "sun_exposure_flag", "thermal_margin",
             "temp_rate_of_change", "cpu_sun_interaction", "is_anomaly"]

corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
short_labels = [c.replace("_", "\n") for c in corr_cols]
ax.set_xticklabels(short_labels, fontsize=7, rotation=45, ha="right")
ax.set_yticklabels(short_labels, fontsize=7)

# Annotate cells
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr.values[i, j]
        color = "white" if abs(val) > 0.5 else TEXT
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

cb = plt.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Pearson Correlation", fontsize=9)
ax.set_title("🧬  Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("plot6_correlation.png", dpi=180, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════
# PLOT 7 — Multi-Sensor Overlay (Temp + Power + CPU)
# ═══════════════════════════════════════════════════════
print("[7/8] Multi-Sensor Overlay...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

# Slice for readability
s = df.iloc[:1000]

# Processor + Battery Temp
ax1.plot(s.index, s["processor_temp"], color=ACCENT, linewidth=0.6, label="Processor")
ax1.plot(s.index, s["battery_temp"], color=GREEN, linewidth=0.6, alpha=0.7, label="Battery")
ax1.axhline(THRESHOLD, color=RED, linestyle="--", linewidth=0.8, alpha=0.7)
s_anom = s[s["is_anomaly"] == 1]
ax1.scatter(s_anom.index, s_anom["processor_temp"], c=RED, s=20, zorder=5, label="Anomaly")
ax1.set_ylabel("Temperature (°C)")
ax1.legend(fontsize=7, loc="upper right")
ax1.set_title("🌡️  Temperature Sensors", fontsize=10, fontweight="bold")
ax1.grid(True, alpha=0.2)

# Power Consumption
ax2.fill_between(s.index, s["power_consumption"], alpha=0.4, color=ORANGE)
ax2.plot(s.index, s["power_consumption"], color=ORANGE, linewidth=0.5)
ax2.set_ylabel("Power (W)")
ax2.set_title("⚡  Power Consumption", fontsize=10, fontweight="bold")
ax2.grid(True, alpha=0.2)

# CPU Usage
ax3.fill_between(s.index, s["cpu_usage"], alpha=0.4, color=PURPLE)
ax3.plot(s.index, s["cpu_usage"], color=PURPLE, linewidth=0.5)
ax3.set_ylabel("CPU Usage")
ax3.set_xlabel("Timestamp")
ax3.set_title("🖥️  CPU Utilization", fontsize=10, fontweight="bold")
ax3.grid(True, alpha=0.2)

plt.suptitle("📡  Multi-Sensor Dashboard — First 1000 Minutes",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plot7_multi_sensor.png", dpi=180, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════
# PLOT 8 — Thermal Margin & Predicted Temp
# ═══════════════════════════════════════════════════════
print("[8/8] Thermal Margin & Prediction...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

s = df.iloc[:1500]
s_anom = s[s["is_anomaly"] == 1]

# Thermal Margin
ax1.plot(s.index, s["thermal_margin"], color=GREEN, linewidth=0.5, alpha=0.8)
ax1.axhline(0, color=RED, linestyle="--", linewidth=1.2, label="Zero Margin (85°C)")
ax1.fill_between(s.index, 0, s["thermal_margin"],
                 where=s["thermal_margin"] < 0, color=RED, alpha=0.3, label="Over Threshold")
ax1.fill_between(s.index, 0, s["thermal_margin"],
                 where=s["thermal_margin"] >= 0, color=GREEN, alpha=0.1)
ax1.scatter(s_anom.index, s_anom["thermal_margin"], c=RED, s=15, zorder=5)
ax1.set_ylabel("Thermal Margin (°C)")
ax1.set_title("🛡️  Thermal Safety Margin (85°C - Current Temp)", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.2)

# Predicted Temp vs Actual
ax2.plot(s.index, s["processor_temp"], color=ACCENT, linewidth=0.5, alpha=0.8, label="Actual Temp")
ax2.plot(s.index, s["predicted_temp_next_3min"], color=PINK, linewidth=0.5, alpha=0.6, label="Predicted +3min")
ax2.axhline(THRESHOLD, color=ORANGE, linestyle="--", linewidth=0.8)
ax2.scatter(s_anom.index, s_anom["predicted_temp_next_3min"], c=RED, s=12, zorder=5, label="Anomaly Prediction")
ax2.set_ylabel("Temperature (°C)")
ax2.set_xlabel("Timestamp")
ax2.set_title("🔮  Actual vs Predicted Temperature (+3 min)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("plot8_thermal_margin.png", dpi=180, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  ✅  ALL 8 PLOTS SAVED SUCCESSFULLY")
print("=" * 55)
plots = [
    ("plot1_temp_timeseries.png",   "Full mission temperature + anomalies"),
    ("plot2_zoomed_anomalies.png",  "Zoomed 500-min window with day/night"),
    ("plot3_distributions.png",     "Feature distributions (4-panel)"),
    ("plot4_boxplots.png",          "Box plots with IQR outlier counts"),
    ("plot5_cpu_vs_temp.png",       "CPU vs Temp scatter (anomaly colored)"),
    ("plot6_correlation.png",       "Feature correlation heatmap"),
    ("plot7_multi_sensor.png",      "Multi-sensor dashboard overlay"),
    ("plot8_thermal_margin.png",    "Thermal margin + prediction analysis"),
]
for fname, desc in plots:
    print(f"  📊 {fname:35s} — {desc}")
print("=" * 55)
