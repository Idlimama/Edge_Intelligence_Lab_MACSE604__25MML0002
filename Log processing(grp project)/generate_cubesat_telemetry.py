"""
CubeSat Thermal Monitoring - Synthetic Telemetry Data Generator
===============================================================
Generates 15,000 realistic, physics-aware telemetry records for a
Low Earth Orbit CubeSat with onboard Edge AI processing.

Features:
  - Realistic orbital day/night cycling (~90 min orbit)
  - Physics-driven heating (solar) and cooling (radiation)
  - Correlated CPU, power, and temperature behaviour
  - Engineered features (rolling avg, variance, predictions)
  - 5-10% injected anomalies
  - Controlled distributions (right-skew temp, bimodal CPU, etc.)
"""

import json
import math
import random
import numpy as np
from datetime import datetime, timedelta, timezone

# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────
NUM_RECORDS = 15_000
TIME_INTERVAL_SEC = 60  # 1 minute
START_TIME = datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc)
ORBIT_PERIOD_MIN = 92  # typical LEO orbit ~92 minutes
DAY_FRACTION = 0.55  # fraction of orbit in sunlight
ANOMALY_RATE = 0.07  # ~7% anomaly injection

# Thermal thresholds
PROC_TEMP_CRITICAL = 85.0
PROC_TEMP_BASE = 50.0  # equilibrium in shadow
BATTERY_TEMP_OFFSET = -8.0  # battery runs cooler than processor

# Rolling window
WINDOW = 5

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────

def orbit_phase_at(minute_index: int):
    """Return (sun_exposure_flag, orbit_phase) based on orbital position."""
    pos_in_orbit = (minute_index % ORBIT_PERIOD_MIN) / ORBIT_PERIOD_MIN
    if pos_in_orbit < DAY_FRACTION:
        return 1, "day"
    return 0, "night"


def bimodal_cpu(sun_flag: int, ai_active: int) -> float:
    """Generate bimodal CPU usage: idle cluster vs active cluster."""
    if ai_active:
        # Active mode: centred around 0.75
        val = np.random.normal(0.75, 0.10)
    else:
        # Idle mode: centred around 0.15
        val = np.random.normal(0.15, 0.06)
    return float(np.clip(val, 0.02, 0.99))


def subsystem_status(sun_flag: int, minute_index: int):
    """Decide which subsystems are on. AI tends to run in bursts."""
    # AI model active ~60% of time, more likely during day
    ai_prob = 0.7 if sun_flag == 1 else 0.4
    ai_active = 1 if random.random() < ai_prob else 0

    # Camera active when AI is active
    camera_active = ai_active

    # Radio active in short bursts (~20% of time)
    radio_active = 1 if random.random() < 0.20 else 0

    return {
        "ai_model_active": ai_active,
        "camera_active": camera_active,
        "radio_active": radio_active,
    }, ai_active


def solar_panel_temp(sun_flag: int) -> float:
    """Solar panel temp: high in sunlight, very cold in shadow."""
    if sun_flag == 1:
        return float(np.random.uniform(70, 120))
    return float(np.random.uniform(-50, 20))


# ──────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────

def generate_dataset():
    records = []
    processor_temps = []  # history for rolling calcs

    # State variable for processor temperature (continuous simulation)
    proc_temp = PROC_TEMP_BASE + np.random.normal(0, 2)

    print(f"Generating {NUM_RECORDS} telemetry records...")

    for i in range(NUM_RECORDS):
        timestamp = START_TIME + timedelta(seconds=i * TIME_INTERVAL_SEC)
        sun_flag, phase = orbit_phase_at(i)

        # ── Subsystem status ──
        status, ai_active = subsystem_status(sun_flag, i)

        # ── CPU usage (bimodal) ──
        cpu = bimodal_cpu(sun_flag, ai_active)

        # ── Physics-driven processor temperature (mean-reverting model) ──
        # Equilibrium temperature depends on sun exposure and CPU load.
        #   Day + high CPU  → equilibrium ~72°C
        #   Day + low CPU   → equilibrium ~58°C
        #   Night + high CPU → equilibrium ~55°C
        #   Night + low CPU  → equilibrium ~42°C
        equil_temp = 42.0 + 16.0 * sun_flag + 14.0 * cpu
        # Mean-reversion rate (how fast temp moves toward equilibrium)
        reversion_rate = 0.08  # per minute step
        # Ornstein-Uhlenbeck style update
        delta_temp = reversion_rate * (equil_temp - proc_temp)
        noise = np.random.normal(0, 0.5)
        proc_temp += delta_temp + noise

        # Hard safety floor (processor never drops below ~35°C in this sim)
        if proc_temp < 35:
            proc_temp = 35 + abs(np.random.normal(0, 0.5))

        # ── Anomaly injection ──
        is_anomaly = 0
        anomaly_boost = 0.0
        if random.random() < ANOMALY_RATE:
            is_anomaly = 1
            anomaly_type = random.choice(["spike", "rapid_rise", "sensor_glitch"])
            if anomaly_type == "spike":
                anomaly_boost = np.random.uniform(10, 25)
            elif anomaly_type == "rapid_rise":
                anomaly_boost = np.random.uniform(6, 14)
            elif anomaly_type == "sensor_glitch":
                anomaly_boost = np.random.choice([-12, 18]) * random.random()

        # Display temperature includes anomaly; underlying state does NOT drift
        display_temp = proc_temp + anomaly_boost

        # display_temp = what the sensor "reports" (includes anomaly)
        # proc_temp   = clean internal physics state (no permanent drift)
        display_temp_rounded = round(display_temp, 2)

        # ── Battery temperature (tracks clean state, not anomaly) ──
        batt_temp = proc_temp + BATTERY_TEMP_OFFSET + np.random.normal(0, 1.5)
        batt_temp = round(batt_temp, 2)

        # ── Solar panel temperature ──
        sp_temp = round(solar_panel_temp(sun_flag), 2)

        # ── Power consumption (correlated with CPU) ──
        base_power = 25  # idle power draw (Watts)
        power = base_power + cpu * 110 + status["radio_active"] * 15
        power += np.random.normal(0, 3)
        power = round(max(power, 10), 2)

        # ── Radiator orientation angle ──
        # Optimal is 0° (facing deep space). System adjusts based on sun.
        if sun_flag == 1:
            rad_angle = round(np.random.uniform(0, 60), 1)
        else:
            rad_angle = round(np.random.uniform(30, 180), 1)

        # ── Store DISPLAYED temp for rolling calculations (what's logged) ──
        processor_temps.append(display_temp_rounded)

        # ── Engineered features ──
        # temp_rate_of_change
        if len(processor_temps) >= 2:
            temp_roc = round(processor_temps[-1] - processor_temps[-2], 3)
        else:
            temp_roc = 0.0

        # rolling_avg_temp (window=5)
        window_data = processor_temps[-WINDOW:]
        rolling_avg = round(float(np.mean(window_data)), 2)

        # temp_variance (window=5)
        temp_var = round(float(np.var(window_data)), 3) if len(window_data) >= 2 else 0.0

        # thermal_margin (based on displayed/reported temp)
        thermal_margin = round(PROC_TEMP_CRITICAL - display_temp_rounded, 2)

        # predicted_temp_next_3min (simple linear extrapolation)
        predicted_temp = round(display_temp_rounded + 3 * temp_roc, 2)

        # time_to_threshold (minutes until 85°C at current rate)
        if temp_roc > 0.01:
            ttt = round(thermal_margin / temp_roc, 1)
            ttt = max(ttt, 0)  # can't be negative
            ttt = min(ttt, 9999)  # cap
        else:
            ttt = 9999  # effectively infinite

        # external_heat_load
        ext_heat = round(sun_flag * np.random.uniform(0.7, 1.0), 3)

        # cpu_sun_interaction
        cpu_sun = round(cpu * sun_flag, 4)

        # ── Build record ──
        record = {
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "processor_temp": display_temp_rounded,
            "battery_temp": batt_temp,
            "solar_panel_temp": sp_temp,
            "cpu_usage": round(cpu, 4),
            "power_consumption": power,
            "sun_exposure_flag": sun_flag,
            "orbit_phase": phase,
            "radiator_orientation_angle": rad_angle,
            "subsystem_status": status,
            "temp_rate_of_change": temp_roc,
            "rolling_avg_temp": rolling_avg,
            "temp_variance": temp_var,
            "thermal_margin": thermal_margin,
            "predicted_temp_next_3min": predicted_temp,
            "time_to_threshold": ttt,
            "external_heat_load": ext_heat,
            "cpu_sun_interaction": cpu_sun,
            "is_anomaly": is_anomaly,
        }
        records.append(record)

    return records


# ──────────────────────────────────────────────────
# Statistics summary
# ──────────────────────────────────────────────────

def print_stats(records):
    temps = [r["processor_temp"] for r in records]
    cpus = [r["cpu_usage"] for r in records]
    anomalies = sum(r["is_anomaly"] for r in records)
    sun_flags = sum(r["sun_exposure_flag"] for r in records)

    print("\n" + "=" * 55)
    print("  DATASET STATISTICS")
    print("=" * 55)
    print(f"  Total records        : {len(records)}")
    print(f"  Time span            : {records[0]['timestamp']} → {records[-1]['timestamp']}")
    print(f"  ─── Processor Temp ───")
    print(f"    Min / Max          : {min(temps):.2f}°C / {max(temps):.2f}°C")
    print(f"    Mean / Std         : {np.mean(temps):.2f}°C / {np.std(temps):.2f}°C")
    print(f"    Skewness           : {float(skewness(temps)):.3f}")
    print(f"  ─── CPU Usage ───")
    print(f"    Min / Max          : {min(cpus):.4f} / {max(cpus):.4f}")
    print(f"    Mean               : {np.mean(cpus):.4f}")
    print(f"  ─── Flags ───")
    print(f"    Sun exposure (day) : {sun_flags} ({100*sun_flags/len(records):.1f}%)")
    print(f"    Anomalies          : {anomalies} ({100*anomalies/len(records):.1f}%)")
    print("=" * 55)


def skewness(data):
    """Compute skewness of a list."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return (np.sum(((np.array(data) - mean) / std) ** 3)) / n


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    records = generate_dataset()
    print_stats(records)

    output_file = "cubesat_telemetry_15000.json"
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n✅ Dataset saved to: {output_file}")
    print(f"   File contains {len(records)} records.")
