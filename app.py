from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.battery_simulator import BatterySimulationConfig, simulate_battery
from src.battery_aging import estimate_cycle_aging
from src.energy_calculator import power_from_voltage_current, remaining_nominal_energy_wh
from src.ml_predictor import (
    build_synthetic_dataset,
    load_model,
    predict_energy,
    predict_soc,
    train_energy_predictor,
    train_soc_predictor,
)
from src.report_generator import generate_report
from src.safety_monitor import SafetyMonitor
from src.safety_forecast import forecast_threshold_crossings
from src.race_ev_simulator import (
    RaceEVConfig,
    analyze_sector_times,
    rank_setup_tradeoffs,
    run_race_setup_sweep,
    simulate_driver_inputs,
    simulate_race_ev,
)

st.set_page_config(page_title="CAEV-BMS", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #07131f 0%, #0d1f2d 100%);
            color: #eef7ff;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, h4 {
            color: #dff9ff;
        }
        .dashboard-title {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.25rem 0 0;
        }
        .dashboard-title h1 {
            margin: 0;
            font-size: clamp(1.55rem, 3vw, 2.35rem);
        }
        .title-battery {
            display: inline-block;
            font-size: 2.2rem;
            line-height: 1;
            filter: drop-shadow(0 0 9px rgba(65, 211, 189, 0.75));
            animation: battery-flash 1.4s ease-in-out infinite;
        }
        @keyframes battery-flash {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.55; transform: scale(1.12); }
        }
        div[data-testid="stMetricValue"], .metric-container {
            background: rgba(17, 40, 52, 0.8);
            border: 1px solid rgba(89, 201, 199, 0.4);
            border-radius: 12px;
            padding: 0.75rem;
        }
        .fault-banner {
            margin: 1rem 0 1.5rem;
            padding: 1rem 1.15rem;
            border: 1px solid rgba(255, 107, 107, 0.55);
            border-radius: 15px;
            background: linear-gradient(110deg, rgba(67, 20, 29, 0.92), rgba(29, 24, 39, 0.94));
            box-shadow: 0 0 24px rgba(255, 75, 75, 0.15);
            animation: fault-arrive 0.45s ease-out both;
        }
        .fault-banner-heading {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
            color: #ffe4e6;
            font-size: 1.05rem;
            font-weight: 750;
        }
        .fault-count {
            border-radius: 999px;
            padding: 0.2rem 0.65rem;
            color: #fff;
            background: rgba(239, 68, 68, 0.3);
            font-size: 0.8rem;
            white-space: nowrap;
        }
        .fault-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.65rem;
        }
        .fault-item {
            border: 1px solid var(--fault-border);
            border-left: 4px solid var(--fault-color);
            border-radius: 10px;
            padding: 0.65rem 0.75rem;
            background: var(--fault-bg);
            color: #f8fafc;
        }
        .fault-severity {
            display: block;
            margin-bottom: 0.18rem;
            color: var(--fault-color);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .fault-detail {
            display: block;
            margin-top: 0.38rem;
            color: #dbeafe;
            font-size: 0.82rem;
        }
        .fault-action {
            display: block;
            margin-top: 0.48rem;
            padding-top: 0.4rem;
            border-top: 1px solid rgba(226, 232, 240, 0.14);
            color: #cbd5e1;
            font-size: 0.78rem;
        }
        @keyframes fault-arrive {
            from { opacity: 0; transform: translateY(-8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stAlert, .stSuccess, .stWarning {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_battery_pack_figure(series: pd.DataFrame, time_index: int) -> go.Figure:
    """Create a stylised 3D battery pack view with cells colored by SoC."""
    row = series.iloc[min(time_index, len(series) - 1)]
    cell_count = 36
    x = np.linspace(-1.5, 1.5, 6)
    y = np.linspace(-1.0, 1.0, 6)
    z = np.linspace(-0.8, 0.8, 3)
    xx, yy, zz = np.meshgrid(x, y, z)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    zz = zz.reshape(-1)

    soc_values = np.full(len(xx), float(row["soc_percent"]))
    cell_health = np.clip((soc_values / 100.0) * 255, 20, 220)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xx,
                y=yy,
                z=zz,
                mode="markers",
                marker={
                    "size": 11,
                    "color": soc_values,
                    "colorscale": [[0, "#1f6feb"], [0.5, "#41d3bd"], [1, "#d7f9ff"]],
                    "cmin": 0,
                    "cmax": 100,
                    "opacity": 0.95,
                    "colorbar": {"title": "SoC (%)"},
                },
                hovertemplate=(
                    "Battery cell<br>"
                    "SoC: %{marker.color:.1f}%<br>"
                    "Voltage: " + f"{row['voltage_v']:.2f}" + " V<br>"
                    "Temp: " + f"{row['temperature_c']:.1f}" + " °C<extra></extra>"
                ),
            )
        ],
        layout={
            "title": "3D battery cell pack view",
            "template": "plotly_dark",
            "scene": {
                "xaxis": {"title": "Pack width"},
                "yaxis": {"title": "Pack depth"},
                "zaxis": {"title": "Pack height"},
                "aspectmode": "cube",
                "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.3}},
            },
            "margin": {"l": 0, "r": 0, "b": 0, "t": 45},
        },
    )
    return fig


def build_race_vehicle_figure(vehicle: pd.DataFrame, frame_index: int, track_length_m: float) -> go.Figure:
    """Render the simulated formula-style EV on a stylised 3D circuit."""
    row = vehicle.iloc[min(frame_index, len(vehicle) - 1)]
    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    semi_major = track_length_m * 0.19
    semi_minor = track_length_m * 0.135
    track_x = semi_major * np.cos(theta)
    track_y = semi_minor * np.sin(theta)

    def vehicle_traces(vehicle_row: pd.Series) -> list[go.BaseTraceType]:
        distance_phase = (float(vehicle_row["distance_m"]) % track_length_m) / track_length_m
        car_theta = distance_phase * 2.0 * np.pi
        car_x = semi_major * np.cos(car_theta)
        car_y = semi_minor * np.sin(car_theta)
        tangent_x = -semi_major * np.sin(car_theta)
        tangent_y = semi_minor * np.cos(car_theta)
        tangent_length = np.hypot(tangent_x, tangent_y) or 1.0
        tangent_x /= tangent_length
        tangent_y /= tangent_length
        lateral_x, lateral_y = -tangent_y, tangent_x

        def to_world(forward: float, lateral: float) -> tuple[float, float]:
            return (
                car_x + forward * tangent_x + lateral * lateral_x,
                car_y + forward * tangent_y + lateral * lateral_y,
            )

        local_body = [(3.2, 0.0), (2.2, 0.68), (-1.7, 0.9), (-2.9, 0.48), (-3.2, 0.0), (-2.9, -0.48), (-1.7, -0.9), (2.2, -0.68)]
        body_points = [to_world(forward, lateral) for forward, lateral in local_body]
        wheel_points = [to_world(forward, lateral) for forward, lateral in ((2.0, 1.3), (2.0, -1.3), (-2.0, 1.3), (-2.0, -1.3))]
        wing_points = [to_world(forward, lateral) for forward, lateral in ((3.0, -1.55), (3.0, 1.55), (-3.0, -1.45), (-3.0, 1.45))]
        return [
            go.Mesh3d(
                x=[point[0] for point in body_points],
                y=[point[1] for point in body_points],
                z=[7.0] * len(body_points),
                i=[0] * 6,
                j=list(range(1, 7)),
                k=list(range(2, 8)),
                color="#ff4b65" if vehicle_row["motor_temp_c"] < 100 else "#ff9f43" if vehicle_row["motor_temp_c"] < 125 else "#ff1744",
                flatshading=True,
                name="Electric race car",
                hovertemplate=(
                    f"Time: {vehicle_row['time_s']:.1f} s<br>"
                    f"Speed: {vehicle_row['speed_kmh']:.1f} km/h<br>"
                    f"Battery SoC: {vehicle_row['soc_percent']:.1f}%<br>"
                    f"Motor temperature: {vehicle_row['motor_temp_c']:.1f} °C<br>"
                    f"Power limit: {vehicle_row['power_derate_pct']:.0f}%<extra></extra>"
                ),
            ),
            go.Scatter3d(
                x=[wing_points[0][0], wing_points[1][0], None, wing_points[2][0], wing_points[3][0]],
                y=[wing_points[0][1], wing_points[1][1], None, wing_points[2][1], wing_points[3][1]],
                z=[8.0, 8.0, None, 7.5, 7.5],
                mode="lines",
                line={"color": "#ffc857", "width": 7},
                name="Aero wings",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=[point[0] for point in wheel_points],
                y=[point[1] for point in wheel_points],
                z=[5.8] * len(wheel_points),
                mode="markers",
                marker={"color": "#101820", "size": 10, "symbol": "circle", "line": {"color": "#61717c", "width": 2}},
                name="Race tires",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=[car_x],
                y=[car_y],
                z=[13.0],
                mode="text",
                text=[f"{vehicle_row['speed_kmh']:.0f} km/h"],
                textfont={"color": "#eaf6ff", "size": 12},
                showlegend=False,
                hoverinfo="skip",
            ),
        ]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=track_x,
            y=track_y,
            z=np.zeros_like(track_x),
            mode="lines",
            line={"color": "#263b4a", "width": 20},
            name="Circuit surface",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=track_x,
            y=track_y,
            z=np.full_like(track_x, 0.2),
            mode="lines",
            line={"color": "#8ba2b2", "width": 2, "dash": "dash"},
            name="Racing line",
            hoverinfo="skip",
        )
    )
    for trace in vehicle_traces(row):
        figure.add_trace(trace)

    frame_indices = np.unique(
        np.append(
            np.linspace(0, len(vehicle) - 1, min(80, len(vehicle)), dtype=int),
            min(frame_index, len(vehicle) - 1),
        )
    )
    animation_frames = [
        go.Frame(
            name=str(index),
            data=vehicle_traces(vehicle.iloc[index]),
            traces=[2, 3, 4, 5],
        )
        for index in frame_indices
    ]
    slider_steps = [
        {
            "label": f"{vehicle.iloc[index]['time_s']:.0f}s",
            "method": "animate",
            "args": [[str(index)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
        }
        for index in frame_indices
    ]
    figure.update_layout(
        title=f"3D electric race car · Lap {int(row['lap']) + 1} · {row['distance_m'] % track_length_m:.0f} m",
        template="plotly_dark",
        height=540,
        margin={"l": 0, "r": 0, "b": 80, "t": 50},
        legend={"orientation": "h", "y": -0.08},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.0,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"fromcurrent": True, "mode": "immediate", "frame": {"duration": 200, "redraw": True}, "transition": {"duration": 170}}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[{"active": int(np.argmin(np.abs(frame_indices - frame_index))), "steps": slider_steps, "x": 0.08, "y": -0.12, "len": 0.88}],
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 0.72, "z": 0.38},
            "camera": {"eye": {"x": 1.35, "y": -1.65, "z": 1.6}},
        },
    )
    figure.frames = animation_frames
    return figure


def build_health_radar_figure(row: pd.Series) -> go.Figure:
    """Build a radar plot summarising battery health indicators."""
    voltage_score = min(max((float(row["voltage_v"]) / 48.0) * 100.0, 0.0), 100.0)
    soc_score = float(row["soc_percent"])
    temperature_score = max(100.0 - abs(float(row["temperature_c"]) - 25.0) * 2.0, 0.0)
    current_score = max(100.0 - abs(float(row["current_a"])) * 1.6, 0.0)
    safety_score = 100.0 if not row.name else 100.0
    values = [voltage_score, soc_score, temperature_score, current_score, safety_score]
    categories = ["Voltage", "SoC", "Temperature", "Current", "Safety"]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                line={"color": "#41d3bd", "width": 2},
                marker={"color": "#d7f9ff", "size": 8},
            )
        ],
        layout={
            "title": "Battery health radar",
            "template": "plotly_dark",
            "polar": {"radialaxis": {"visible": True, "range": [0, 100]}},
            "margin": {"l": 50, "r": 50, "b": 50, "t": 50},
        },
    )
    return fig


st.markdown(
    """
    <div class="dashboard-title">
        <span class="title-battery" aria-label="Battery status indicator">🔋</span>
        <h1>Connected EV Intelligent Energy &amp; Battery Management System (CAEV-BMS)</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Simulated diagnostic dashboard for academic demonstration. All displayed values are simulated data.")

if "timeline_index" not in st.session_state:
    st.session_state.timeline_index = 0
if "timeline_slider" not in st.session_state:
    st.session_state.timeline_slider = st.session_state.timeline_index

with st.sidebar:
    st.header("Battery simulation controls")
    nominal_voltage_v = st.number_input("Battery nominal voltage (V)", min_value=1.0, max_value=400.0, value=48.0, step=0.5)
    capacity_ah = st.number_input("Battery capacity (Ah)", min_value=1.0, max_value=500.0, value=40.0, step=1.0)
    initial_soc = st.slider("Initial SoC (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    discharge_current_a = st.number_input("Discharge current (A)", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
    mode = st.selectbox("Operating mode", ["discharge", "charge"])
    temperature_c = st.number_input("Battery temperature (°C)", min_value=-30.0, max_value=100.0, value=27.0, step=1.0)
    duration_s = st.number_input("Simulation duration (s)", min_value=30.0, max_value=3600.0, value=600.0, step=10.0)
    sampling_interval_s = st.number_input("Sampling interval (s)", min_value=1.0, max_value=300.0, value=30.0, step=1.0)
    forecast_horizon_minutes = st.slider("Predictive watch window (minutes)", min_value=1, max_value=60, value=10, step=1)

    st.subheader("Battery aging assumptions")
    rated_cycle_life = st.number_input("Rated full-cycle life", min_value=100, max_value=20000, value=2000, step=100)
    end_of_life_soh_pct = st.slider("End-of-life SoH (%)", min_value=70, max_value=90, value=80, step=1)
    daily_efc = st.slider("Assumed use (EFC/day)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    st.subheader("Electric race vehicle concept")
    race_mass_kg = st.number_input("Vehicle mass (kg)", min_value=500.0, max_value=2000.0, value=1000.0, step=25.0)
    race_motor_power_kw = st.number_input("Peak motor power (kW)", min_value=50.0, max_value=800.0, value=350.0, step=10.0)
    race_battery_kwh = st.number_input("Vehicle battery capacity (kWh)", min_value=10.0, max_value=150.0, value=54.0, step=1.0)
    race_initial_soc = st.slider("Vehicle starting SoC (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
    race_track_length_m = st.number_input("Concept circuit length (m)", min_value=500.0, max_value=10000.0, value=3000.0, step=100.0)
    race_regen_pct = st.slider("Regenerative braking strength (%)", min_value=0, max_value=100, value=65, step=5)
    race_ambient_temp_c = st.number_input("Powertrain ambient temperature (°C)", min_value=-20.0, max_value=60.0, value=25.0, step=1.0)
    race_cooling_pct = st.slider("Cooling system capacity (%)", min_value=0, max_value=150, value=100, step=5)

    st.subheader("Presentation controls")
    playback_speed = st.slider("Animation speed (x)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
    play_animation = st.checkbox("Play simulation animation", value=False)
    st.caption("Auto-play cycles the battery timeline smoothly for live demonstration.")

    st.subheader("Safety thresholds (demonstration only)")
    undervoltage = st.number_input("Undervoltage limit (V)", value=30.0, min_value=1.0, max_value=100.0, step=0.5)
    overvoltage = st.number_input("Overvoltage limit (V)", value=58.0, min_value=1.0, max_value=100.0, step=0.5)
    overcurrent = st.number_input("Overcurrent limit (A)", value=40.0, min_value=1.0, max_value=200.0, step=1.0)
    high_temperature = st.number_input("High temperature limit (°C)", value=50.0, min_value=1.0, max_value=100.0, step=1.0)
    low_soc = st.number_input("Low SoC limit (%)", value=20.0, min_value=0.0, max_value=100.0, step=1.0)

config = BatterySimulationConfig(
    nominal_voltage_v=nominal_voltage_v,
    capacity_ah=capacity_ah,
    initial_soc=initial_soc,
    discharge_current_a=discharge_current_a,
    mode=mode,
    temperature_c=temperature_c,
    duration_s=duration_s,
    sampling_interval_s=sampling_interval_s,
)

simulation = simulate_battery(config)
df = simulation["summary"]
series = simulation["dataframe"]
race_vehicle = simulate_race_ev(
    RaceEVConfig(
        mass_kg=race_mass_kg,
        motor_peak_power_kw=race_motor_power_kw,
        battery_capacity_kwh=race_battery_kwh,
        initial_soc_pct=race_initial_soc,
        track_length_m=race_track_length_m,
        duration_s=duration_s,
        time_step_s=0.5,
        regen_strength_pct=float(race_regen_pct),
        ambient_temp_c=race_ambient_temp_c,
        cooling_capacity_pct=float(race_cooling_pct),
    )
)
aging_profile = estimate_cycle_aging(
    series,
    capacity_ah=capacity_ah,
    rated_cycle_life=float(rated_cycle_life),
    end_of_life_soh_pct=float(end_of_life_soh_pct),
    daily_efc=daily_efc,
)

last_row = series.iloc[-1]
if len(series) > 1:
    if play_animation:
        st.session_state.timeline_index = (st.session_state.timeline_index + 1) % len(series)
        st.session_state.timeline_slider = st.session_state.timeline_index
    else:
        st.session_state.timeline_slider = min(st.session_state.timeline_slider, len(series) - 1)
    selected_time_index = st.slider(
        "Battery timeline",
        0,
        len(series) - 1,
        key="timeline_slider",
    )
    st.session_state.timeline_index = selected_time_index
else:
    selected_time_index = 0

preview_row = series.iloc[selected_time_index]
race_frame_index = int(np.searchsorted(race_vehicle["time_s"].to_numpy(), float(preview_row["time_s"]), side="right") - 1)
race_frame_index = max(0, min(race_frame_index, len(race_vehicle) - 1))
race_row = race_vehicle.iloc[race_frame_index]

st.subheader("Battery overview")
metric_cols = st.columns(4)
metric_cols[0].metric("Battery pack voltage (V)", f"{preview_row['voltage_v']:.2f}", delta="Simulated data")
metric_cols[1].metric("Battery current (A)", f"{preview_row['current_a']:.2f}", delta="Simulated data")
metric_cols[2].metric("Battery temperature (°C)", f"{preview_row['temperature_c']:.2f}", delta="Simulated data")
metric_cols[3].metric("State of Charge (SoC %)", f"{preview_row['soc_percent']:.1f}", delta="Simulated data")

energy_cols = st.columns(3)
energy_cols[0].metric("Instantaneous power (W)", f"{preview_row['power_w']:.1f}", delta="Simulated data")
energy_cols[1].metric("Accumulated energy consumption (Wh)", f"{preview_row['discharge_energy_wh']:.1f}", delta="Simulated data")
energy_cols[2].metric("Estimated remaining energy (Wh)", f"{preview_row['remaining_energy_wh']:.1f}", delta="Simulated data")

health_cols = st.columns(4)
health_cols[0].metric("Health score", f"{min(100.0, max(0.0, preview_row['soc_percent'])):.0f}/100")
health_cols[1].metric("Thermal health", f"{max(0.0, 100.0 - abs(preview_row['temperature_c'] - 25.0) * 2.0):.0f}/100")
health_cols[2].metric("Capacity retention", f"{max(0.0, min(100.0, (preview_row['remaining_energy_wh'] / max(preview_row['discharge_energy_wh'] + 1, 1)) * 100.0)):.0f}/100")
health_cols[3].metric("Safety readiness", f"{max(0.0, 100.0 - abs(preview_row['current_a']) * 1.5):.0f}/100")

st.subheader("Battery aging & remaining useful life")
st.caption(
    "Cycle-throughput estimate only: assumes linear capacity fade from 100% SoH to the selected end-of-life threshold. "
    "It excludes calendar aging, chemistry, temperature, and rate effects."
)
aging_row = aging_profile.iloc[selected_time_index]
aging_cols = st.columns(4)
aging_cols[0].metric("Estimated cycle SoH", f"{aging_row['soh_pct']:.2f}%")
aging_cols[1].metric("Equivalent full cycles", f"{aging_row['equivalent_full_cycles']:.4f}")
aging_cols[2].metric("Remaining cycle life", f"{aging_row['rul_cycles']:.1f} cycles")
aging_cols[3].metric("Estimated remaining days", f"{aging_row['rul_days']:.0f} days", delta=f"At {daily_efc:.1f} EFC/day")

lifecycle_cycles = np.linspace(0.0, float(rated_cycle_life), 120)
fade_per_cycle_pct = (100.0 - float(end_of_life_soh_pct)) / float(rated_cycle_life)
lifecycle_soh = np.maximum(
    float(end_of_life_soh_pct),
    100.0 - lifecycle_cycles * fade_per_cycle_pct,
)
aging_figure = go.Figure()
aging_figure.add_trace(
    go.Scatter(
        x=lifecycle_cycles,
        y=lifecycle_soh,
        mode="lines",
        name="Assumed cycle-aging curve",
        line={"color": "#41d3bd", "width": 3},
    )
)
aging_figure.add_trace(
    go.Scatter(
        x=[float(aging_row["equivalent_full_cycles"])],
        y=[float(aging_row["soh_pct"])],
        mode="markers",
        name="Current simulated point",
        marker={"color": "#ffbd4a", "size": 12, "line": {"color": "#fff", "width": 1}},
    )
)
aging_figure.add_hline(
    y=float(end_of_life_soh_pct),
    line_dash="dash",
    line_color="#ff6b6b",
    annotation_text="Configured end-of-life threshold",
)
aging_figure.update_layout(
    template="plotly_dark",
    height=360,
    title="Illustrative capacity fade over rated equivalent full cycles",
    xaxis_title="Equivalent full cycles",
    yaxis_title="Estimated SoH (%)",
    yaxis_range=[max(0, float(end_of_life_soh_pct) - 5), 102],
    hovermode="x unified",
)
st.plotly_chart(aging_figure, use_container_width=True)

st.subheader("Formula-racing electric vehicle simulation")
st.caption(
    "Concept vehicle, not an official F1/Formula E specification. Speed profile, circuit and telemetry are simulated; "
    "the vehicle model uses simplified longitudinal dynamics and first-order powertrain thermal estimates. Use Play/Pause and the scrubber inside the 3D viewer; "
    "the initial frame follows the battery timeline above."
)
race_metric_cols = st.columns(4)
race_metric_cols[0].metric("Vehicle speed", f"{race_row['speed_kmh']:.1f} km/h", delta=f"Lap {int(race_row['lap']) + 1}")
race_metric_cols[1].metric("Longitudinal acceleration", f"{race_row['acceleration_mps2']:.2f} m/s²")
power_delta = "Regenerative charging" if race_row["battery_power_kw"] < 0 else "Traction / auxiliary draw"
race_metric_cols[2].metric("Battery power", f"{race_row['battery_power_kw']:.1f} kW", delta=power_delta)
race_metric_cols[3].metric("Vehicle battery SoC", f"{race_row['soc_percent']:.1f}%", delta=f"{race_row['cumulative_regen_wh']:.0f} Wh regen recovered")
thermal_metric_cols = st.columns(4)
thermal_metric_cols[0].metric("Battery pack temperature", f"{race_row['battery_temp_c']:.1f} °C")
thermal_metric_cols[1].metric("Motor temperature", f"{race_row['motor_temp_c']:.1f} °C")
thermal_metric_cols[2].metric("Inverter temperature", f"{race_row['inverter_temp_c']:.1f} °C")
thermal_metric_cols[3].metric("Available power", f"{race_row['power_derate_pct']:.0f}%", delta="Thermal derating active" if race_row["power_derate_pct"] < 99.5 else "No thermal derating")

race_columns = st.columns([1.15, 1.0])
race_columns[0].plotly_chart(
    build_race_vehicle_figure(race_vehicle, race_frame_index, race_track_length_m),
    use_container_width=True,
)
race_telemetry = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Speed & circuit target", "Battery power (+ draw / − regen)", "Vehicle battery SoC", "Powertrain temperatures"),
)
race_telemetry.add_trace(
    go.Scatter(x=race_vehicle["time_s"], y=race_vehicle["speed_kmh"], name="Speed", line={"color": "#41d3bd", "width": 2}),
    row=1,
    col=1,
)
race_telemetry.add_trace(
    go.Scatter(x=race_vehicle["time_s"], y=race_vehicle["target_speed_kmh"], name="Track target", line={"color": "#94a3b8", "dash": "dot"}),
    row=1,
    col=1,
)
race_telemetry.add_trace(
    go.Scatter(x=race_vehicle["time_s"], y=race_vehicle["battery_power_kw"], name="Battery power", line={"color": "#ff9f43", "width": 2}, fill="tozeroy"),
    row=2,
    col=1,
)
race_telemetry.add_trace(
    go.Scatter(x=race_vehicle["time_s"], y=race_vehicle["soc_percent"], name="SoC", line={"color": "#8b7bff", "width": 2}),
    row=3,
    col=1,
)
race_telemetry.add_trace(
    go.Scatter(x=race_vehicle["time_s"], y=race_vehicle["battery_temp_c"], name="Battery", line={"color": "#41d3bd"}),
    row=4,
    col=1,
)
race_telemetry.add_trace(
    go.Scatter(x=race_vehicle["time_s"], y=race_vehicle["motor_temp_c"], name="Motor", line={"color": "#ff9f43"}),
    row=4,
    col=1,
)
race_telemetry.add_trace(
    go.Scatter(x=race_vehicle["time_s"], y=race_vehicle["inverter_temp_c"], name="Inverter", line={"color": "#ff4d6d"}),
    row=4,
    col=1,
)
race_telemetry.add_vline(x=float(race_row["time_s"]), line_dash="dash", line_color="#f8fafc", row="all", col=1)
race_telemetry.update_yaxes(title_text="km/h", row=1, col=1)
race_telemetry.update_yaxes(title_text="kW", row=2, col=1, zeroline=True, zerolinecolor="#94a3b8")
race_telemetry.update_yaxes(title_text="%", row=3, col=1, range=[0, 100])
race_telemetry.update_yaxes(title_text="°C", row=4, col=1)
race_telemetry.update_xaxes(title_text="Simulation time (s)", row=4, col=1)
race_telemetry.update_layout(template="plotly_dark", height=700, margin={"l": 8, "r": 8, "t": 60, "b": 8}, hovermode="x unified", showlegend=False)
race_columns[1].plotly_chart(race_telemetry, use_container_width=True)

st.markdown("#### Race telemetry & sector analysis")
st.caption(
    "Analysis layout follows public motorsport telemetry conventions (distance-aligned channels, driver inputs, speed and sector timing). "
    "All channels here are synthetic; gear and RPM are illustrative proxies, not physical Formula car telemetry."
)
telemetry_figure = make_subplots(
    rows=5,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.055,
    specs=[[{}], [{}], [{"secondary_y": True}], [{}], [{}]],
    subplot_titles=(
        "Speed trace · target vs simulated",
        "Driver inputs · throttle and brake",
        "Virtual powertrain · RPM proxy and gear estimate",
        "Energy deployment · draw and recovered regen",
        "Powertrain thermal channels",
    ),
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["speed_kmh"], name="Speed", line={"color": "#41d3bd", "width": 2}), row=1, col=1
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["target_speed_kmh"], name="Target speed", line={"color": "#94a3b8", "dash": "dot"}), row=1, col=1
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["throttle_pct"], name="Throttle demand", line={"color": "#41d3bd"}, fill="tozeroy"), row=2, col=1
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["brake_pct"], name="Brake demand", line={"color": "#ff4d6d"}, fill="tozeroy"), row=2, col=1
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["virtual_rpm"], name="RPM proxy", line={"color": "#b69cff"}), row=3, col=1, secondary_y=False
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["virtual_gear"], name="Virtual gear", mode="lines", line={"color": "#ffc857", "shape": "hv"}), row=3, col=1, secondary_y=True
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["battery_power_kw"], name="Battery power", line={"color": "#ff9f43"}), row=4, col=1
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["regen_power_kw"], name="Regen recovered", line={"color": "#41d3bd", "dash": "dash"}), row=4, col=1
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["battery_temp_c"], name="Battery °C", line={"color": "#41d3bd"}), row=5, col=1
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["motor_temp_c"], name="Motor °C", line={"color": "#ff9f43"}), row=5, col=1
)
telemetry_figure.add_trace(
    go.Scatter(x=race_vehicle["distance_m"], y=race_vehicle["inverter_temp_c"], name="Inverter °C", line={"color": "#ff4d6d"}), row=5, col=1
)
telemetry_figure.add_vline(x=float(race_row["distance_m"]), line_dash="dash", line_color="#f8fafc", row="all", col=1)
telemetry_figure.update_yaxes(title_text="km/h", row=1, col=1)
telemetry_figure.update_yaxes(title_text="%", range=[0, 100], row=2, col=1)
telemetry_figure.update_yaxes(title_text="RPM proxy", row=3, col=1, secondary_y=False)
telemetry_figure.update_yaxes(title_text="Gear", range=[0.5, 7.5], dtick=1, row=3, col=1, secondary_y=True)
telemetry_figure.update_yaxes(title_text="kW", row=4, col=1, zeroline=True, zerolinecolor="#94a3b8")
telemetry_figure.update_yaxes(title_text="°C", row=5, col=1)
telemetry_figure.update_xaxes(title_text="Distance from start (m)", row=5, col=1)
telemetry_figure.update_layout(template="plotly_dark", height=1050, margin={"l": 8, "r": 8, "t": 70, "b": 8}, hovermode="x unified", showlegend=True, legend={"orientation": "h", "y": 1.02})

telemetry_columns = st.columns([1.5, 1.0])
telemetry_columns[0].plotly_chart(telemetry_figure, use_container_width=True)
sector_progress = analyze_sector_times(
    race_vehicle,
    track_length_m=float(race_track_length_m),
)
with telemetry_columns[1]:
    st.markdown("**Simulated-run sector timing**")
    if sector_progress.empty:
        st.info("The simulated duration did not complete a full sector.")
    else:
        sector_best = sector_progress.groupby("Sector")["Sector time (s)"].min().rename("Best sector (s)")
        latest_completed_lap = int(sector_progress["Lap"].max())
        latest_sectors = sector_progress.loc[sector_progress["Lap"] == latest_completed_lap].copy()
        latest_sectors = latest_sectors.join(sector_best, on="Sector")
        latest_sectors["Delta to best (s)"] = latest_sectors["Sector time (s)"] - latest_sectors["Best sector (s)"]
        latest_sectors["Sector time (s)"] = latest_sectors["Sector time (s)"].map(lambda value: f"{value:.3f}")
        latest_sectors["Best sector (s)"] = latest_sectors["Best sector (s)"].map(lambda value: f"{value:.3f}")
        latest_sectors["Delta to best (s)"] = latest_sectors["Delta to best (s)"].map(lambda value: f"{value:+.3f}")
        st.caption(f"Latest lap with a completed sector: {latest_completed_lap} · benchmarked against this simulated run")
        st.dataframe(
            latest_sectors[["Sector", "Sector time (s)", "Best sector (s)", "Delta to best (s)"]],
            hide_index=True,
            use_container_width=True,
        )
        st.metric("Completed sectors", len(sector_progress))
st.download_button(
    "Download race telemetry CSV",
    race_vehicle.to_csv(index=False),
    file_name="caev_race_vehicle_telemetry.csv",
    mime="text/csv",
)

st.subheader("Interactive driver-input sandbox")
st.caption(
    "Change the pedal inputs to see a short what-if run. This educational model uses simplified forces; "
    "braking overrides throttle, and the outputs are not vehicle control advice."
)
sandbox_controls = st.columns(4)
sandbox_initial_speed = sandbox_controls[0].slider("Starting speed (km/h)", 0, 250, 100, 5, key="sandbox_initial_speed")
sandbox_throttle = sandbox_controls[1].slider("Throttle input (%)", 0, 100, 70, 5, key="sandbox_throttle")
sandbox_brake = sandbox_controls[2].slider("Brake input (%)", 0, 100, 0, 5, key="sandbox_brake")
sandbox_duration = sandbox_controls[3].slider("What-if duration (s)", 2, 20, 10, 1, key="sandbox_duration")
sandbox_result = simulate_driver_inputs(
    mass_kg=race_mass_kg,
    motor_peak_power_kw=race_motor_power_kw,
    battery_capacity_kwh=race_battery_kwh,
    initial_soc_pct=race_initial_soc,
    initial_speed_kmh=float(sandbox_initial_speed),
    throttle_pct=float(sandbox_throttle),
    brake_pct=float(sandbox_brake),
    regen_strength_pct=float(race_regen_pct),
    duration_s=float(sandbox_duration),
)
sandbox_power = sandbox_result["battery_power_kw"].to_numpy()
sandbox_times = sandbox_result["time_s"].to_numpy()
sandbox_net_energy_kwh = float(np.sum(0.5 * (sandbox_power[:-1] + sandbox_power[1:]) * np.diff(sandbox_times)) / 3600.0)
sandbox_metrics = st.columns(4)
sandbox_metrics[0].metric(
    "Final speed",
    f"{sandbox_result['speed_kmh'].iloc[-1]:.1f} km/h",
    delta=f"{sandbox_result['speed_kmh'].iloc[-1] - sandbox_initial_speed:+.1f} km/h",
)
sandbox_metrics[1].metric("Distance travelled", f"{sandbox_result['distance_m'].iloc[-1]:.1f} m")
sandbox_metrics[2].metric("Net battery energy", f"{sandbox_net_energy_kwh:.3f} kWh", delta="positive = consumed")
sandbox_metrics[3].metric("Regen energy recovered", f"{sandbox_result['cumulative_regen_wh'].iloc[-1]:.1f} Wh")

sandbox_figure = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=("Speed response", "Pedal inputs", "Battery power and regen", "Battery SoC"),
)
sandbox_figure.add_trace(
    go.Scatter(x=sandbox_result["time_s"], y=sandbox_result["speed_kmh"], name="Speed", line={"color": "#41d3bd", "width": 3}), row=1, col=1
)
sandbox_figure.add_trace(
    go.Scatter(x=sandbox_result["time_s"], y=sandbox_result["throttle_pct"], name="Effective throttle", line={"color": "#41d3bd"}, fill="tozeroy"), row=2, col=1
)
sandbox_figure.add_trace(
    go.Scatter(x=sandbox_result["time_s"], y=sandbox_result["brake_pct"], name="Brake", line={"color": "#ff4d6d"}, fill="tozeroy"), row=2, col=1
)
sandbox_figure.add_trace(
    go.Scatter(x=sandbox_result["time_s"], y=sandbox_result["battery_power_kw"], name="Battery power", line={"color": "#ff9f43", "width": 2}), row=3, col=1
)
sandbox_figure.add_trace(
    go.Scatter(x=sandbox_result["time_s"], y=sandbox_result["regen_power_kw"], name="Regen", line={"color": "#8b7bff", "dash": "dash"}), row=3, col=1
)
sandbox_figure.add_trace(
    go.Scatter(x=sandbox_result["time_s"], y=sandbox_result["soc_percent"], name="SoC", line={"color": "#e2e8f0", "width": 2}), row=4, col=1
)
sandbox_figure.update_yaxes(title_text="km/h", row=1, col=1)
sandbox_figure.update_yaxes(title_text="%", range=[0, 100], row=2, col=1)
sandbox_figure.update_yaxes(title_text="kW", row=3, col=1, zeroline=True, zerolinecolor="#94a3b8")
sandbox_figure.update_yaxes(title_text="%", range=[0, 100], row=4, col=1)
sandbox_figure.update_xaxes(title_text="Time (s)", row=4, col=1)
sandbox_figure.update_layout(template="plotly_dark", height=760, hovermode="x unified", showlegend=True, legend={"orientation": "h", "y": 1.02})
st.plotly_chart(sandbox_figure, use_container_width=True)
if sandbox_brake > 0 and sandbox_throttle > 0:
    st.info("Brake is applied, so the simulator cuts effective throttle during this what-if run.")

st.subheader("Automated race setup sweep")
st.caption(
    "Compare motor-power and cooling combinations under the same simulated track and driving profile. "
    "The rankings are model-only experiments, not real vehicle setup recommendations."
)
if st.button("Run setup sweep", type="primary"):
    sweep_motor_options = sorted({
        round(max(10.0, race_motor_power_kw * 0.75), 1),
        round(float(race_motor_power_kw), 1),
        round(min(1000.0, race_motor_power_kw * 1.25), 1),
    })
    sweep_cooling_options = sorted({
        max(0.0, float(race_cooling_pct) - 50.0),
        float(race_cooling_pct),
        min(150.0, float(race_cooling_pct) + 50.0),
    })
    sweep_base_config = RaceEVConfig(
        mass_kg=race_mass_kg,
        motor_peak_power_kw=race_motor_power_kw,
        battery_capacity_kwh=race_battery_kwh,
        initial_soc_pct=race_initial_soc,
        track_length_m=race_track_length_m,
        duration_s=duration_s,
        time_step_s=0.5,
        max_speed_kmh=250.0,
        regen_strength_pct=float(race_regen_pct),
        ambient_temp_c=race_ambient_temp_c,
        cooling_capacity_pct=float(race_cooling_pct),
    )
    with st.spinner("Simulating setup combinations on the same track..."):
        st.session_state["race_setup_sweep_df"] = run_race_setup_sweep(
            sweep_base_config,
            motor_power_options_kw=sweep_motor_options,
            cooling_options_pct=sweep_cooling_options,
        )

setup_sweep_df = st.session_state.get("race_setup_sweep_df")
if setup_sweep_df is not None:
    completed_lap_sweeps = setup_sweep_df.dropna(subset=["Best lap (s)"])
    pace_priority_pct = st.slider(
        "Setup recommendation priority · pace (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        help="0 prioritizes lower energy per distance; 100 prioritizes pace.",
        key="setup_pace_priority",
    )
    ranked_sweep = rank_setup_tradeoffs(setup_sweep_df, pace_weight=pace_priority_pct / 100.0)
    recommended_setup = ranked_sweep.iloc[0]
    setup_metrics = st.columns(4)
    if not completed_lap_sweeps.empty:
        fastest_setup = completed_lap_sweeps.loc[completed_lap_sweeps["Best lap (s)"].idxmin()]
        setup_metrics[0].metric(
            "Fastest simulated setup",
            f"{fastest_setup['Best lap (s)']:.2f} s",
            delta=f"{fastest_setup['Motor power (kW)']:.0f} kW · cooling {fastest_setup['Cooling capacity (%)']:.0f}%",
        )
    else:
        fastest_by_mean_speed = setup_sweep_df.loc[setup_sweep_df["Mean speed (km/h)"].idxmax()]
        setup_metrics[0].metric(
            "Highest mean speed",
            f"{fastest_by_mean_speed['Mean speed (km/h)']:.1f} km/h",
            delta="No full lap completed",
        )
    efficient_setup = setup_sweep_df.loc[setup_sweep_df["Net energy (Wh/km)"].idxmin()]
    setup_metrics[1].metric(
        "Lowest net energy / distance",
        f"{efficient_setup['Net energy (Wh/km)']:.1f} Wh/km",
        delta=f"{efficient_setup['Motor power (kW)']:.0f} kW · cooling {efficient_setup['Cooling capacity (%)']:.0f}%",
    )
    recommended_pace = (
        f"{recommended_setup['Best lap (s)']:.2f} s"
        if recommended_setup["Pace objective"] == "Best lap (s)"
        else f"{recommended_setup['Mean speed (km/h)']:.1f} km/h"
    )
    setup_metrics[2].metric(
        "Balanced recommendation",
        recommended_pace,
        delta=f"{recommended_setup['Motor power (kW)']:.0f} kW · cooling {recommended_setup['Cooling capacity (%)']:.0f}%",
        help=f"Weighted score {recommended_setup['Preference score']:.3f} at {pace_priority_pct}% pace priority.",
    )
    setup_metrics[3].metric(
        "Pareto-optimal setups",
        int(ranked_sweep["Pareto optimal"].sum()),
        delta="Not dominated on both pace and energy",
    )

    if not completed_lap_sweeps.empty:
        setup_scatter = px.scatter(
            ranked_sweep,
            x="Net energy (Wh/km)",
            y="Best lap (s)",
            color="Cooling capacity (%)",
            size="Motor power (kW)",
            symbol="Trade-off status",
            symbol_map={"Pareto optimal": "star", "Dominated": "circle"},
            hover_data=["Motor power (kW)", "Cooling capacity (%)", "Preference score", "Peak motor temp (°C)", "Minimum available power (%)"],
            title="Setup trade-off · faster lap versus lower net energy per distance",
            color_continuous_scale=[[0, "#ff6b6b"], [0.5, "#ffc857"], [1, "#41d3bd"]],
        )
        setup_scatter.update_layout(template="plotly_dark", hovermode="closest")
        st.plotly_chart(setup_scatter, use_container_width=True)
    st.dataframe(
        ranked_sweep.round(3),
        hide_index=True,
        use_container_width=True,
    )

battery_html = f"""
<div style='width:100%;height:280px;border-radius:18px;background:linear-gradient(135deg,#0a1f2b,#102a39);display:flex;align-items:center;justify-content:center;overflow:hidden;border:1px solid rgba(89,201,199,0.4);'>
  <div style='position:relative;width:180px;height:230px;transform-style:preserve-3d;animation:spin 8s linear infinite;filter:drop-shadow(0 0 20px rgba(65,211,189,0.5));'>
    <div style='position:absolute;inset:10px;border-radius:24px;border:4px solid rgba(255,255,255,0.8);background:linear-gradient(180deg, rgba(15,45,60,0.95), rgba(8,22,32,0.98));box-shadow: inset 0 0 20px rgba(65,211,189,0.3);'></div>
    <div style='position:absolute;left:50%;top:-12px;transform:translateX(-50%);width:50px;height:18px;border-radius:8px 8px 0 0;background:#9ae8ff;opacity:0.9;'></div>
    <div style='position:absolute;left:18px;right:18px;top:28px;bottom:18px;border-radius:16px;background:linear-gradient(180deg, rgba(18,180,168,0.9) 0%, rgba(14,79,89,0.6) {max(0.0, min(100.0, preview_row['soc_percent'])):.0f}%, rgba(8,31,39,0.2) {max(0.0, min(100.0, preview_row['soc_percent'])):.0f}%, rgba(8,31,39,0.2) 100%);border:2px solid rgba(255,255,255,0.2);'></div>
    <div style='position:absolute;left:35px;top:38px;color:white;font-weight:bold;font-size:26px;'>SoC</div>
    <div style='position:absolute;left:50%;top:74px;transform:translateX(-50%);color:#dff9ff;font-weight:700;font-size:30px;'>{preview_row['soc_percent']:.0f}%</div>
  </div>
</div>
<style>
@keyframes spin {{
  0% {{ transform: rotateX(45deg) rotateY(0deg) rotateZ(0deg); }}
  50% {{ transform: rotateX(25deg) rotateY(180deg) rotateZ(10deg); }}
  100% {{ transform: rotateX(45deg) rotateY(360deg) rotateZ(0deg); }}
}}
</style>
"""
st.components.v1.html(battery_html, height=340)

st.markdown("> All values shown on this page are simulated demonstration data, not measured live battery telemetry.")

monitor = SafetyMonitor(
    undervoltage=undervoltage,
    overvoltage=overvoltage,
    overcurrent=overcurrent,
    high_temperature=high_temperature,
    low_soc=low_soc,
)
alerts = monitor.evaluate(
    {
        "voltage": float(preview_row["voltage_v"]),
        "current": float(preview_row["current_a"]),
        "temperature": float(preview_row["temperature_c"]),
        "soc": float(preview_row["soc_percent"]),
    }
)

if alerts:
    fault_severity = {
        "undervoltage": ("Critical", "#ff4d6d", "rgba(127, 29, 45, 0.32)", "rgba(255, 77, 109, 0.45)"),
        "overvoltage": ("Critical", "#ff4d6d", "rgba(127, 29, 45, 0.32)", "rgba(255, 77, 109, 0.45)"),
        "overcurrent": ("High", "#ff9f43", "rgba(124, 66, 20, 0.3)", "rgba(255, 159, 67, 0.4)"),
        "high_temperature": ("High", "#ff9f43", "rgba(124, 66, 20, 0.3)", "rgba(255, 159, 67, 0.4)"),
        "low_soc": ("Moderate", "#ffd166", "rgba(112, 84, 18, 0.28)", "rgba(255, 209, 102, 0.38)"),
    }
    fault_diagnostics = {
        "undervoltage": (
            f"{preview_row['voltage_v']:.2f} V · demo minimum {undervoltage:.2f} V",
            "Check the simulated pack voltage and review the undervoltage threshold.",
        ),
        "overvoltage": (
            f"{preview_row['voltage_v']:.2f} V · demo maximum {overvoltage:.2f} V",
            "Check the simulated pack voltage and review the overvoltage threshold.",
        ),
        "overcurrent": (
            f"{abs(preview_row['current_a']):.2f} A · demo maximum {overcurrent:.2f} A",
            "Review the simulated load current and the configured current limit.",
        ),
        "high_temperature": (
            f"{preview_row['temperature_c']:.1f} °C · demo maximum {high_temperature:.1f} °C",
            "Review the simulated thermal conditions and the configured temperature limit.",
        ),
        "low_soc": (
            f"{preview_row['soc_percent']:.1f}% · demo minimum {low_soc:.1f}%",
            "Review the simulated charge state and the configured low-SoC threshold.",
        ),
    }
    fault_items = "".join(
        f"""
        <div class="fault-item" style="--fault-color:{fault_severity[alert][1]};--fault-bg:{fault_severity[alert][2]};--fault-border:{fault_severity[alert][3]};">
            <span class="fault-severity">{fault_severity[alert][0]} severity</span>
            {alert.replace('_', ' ').title()}
            <span class="fault-detail">Reading: {fault_diagnostics[alert][0]}</span>
            <span class="fault-action">Suggested check: {fault_diagnostics[alert][1]}</span>
        </div>
        """
        for alert in alerts
    )
    st.markdown(
        f"""
        <section class="fault-banner" role="alert">
            <div class="fault-banner-heading">
                <span>⚠ LIVE EV FAULT PANEL</span>
                <span class="fault-count">{len(alerts)} active fault{'s' if len(alerts) != 1 else ''}</span>
            </div>
            <div class="fault-grid">{fault_items}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Predictive safety watch")
st.caption(
    "Experimental linear trend heuristic using recent simulated samples. "
    "It is not a validated forecast or a battery protection function."
)
forecast_thresholds = {
    "undervoltage": undervoltage,
    "overvoltage": overvoltage,
    "overcurrent": overcurrent,
    "high_temperature": high_temperature,
    "low_soc": low_soc,
}
forecasts = forecast_threshold_crossings(
    series,
    current_index=selected_time_index,
    thresholds=forecast_thresholds,
    horizon_s=forecast_horizon_minutes * 60.0,
)

if len(series.iloc[: selected_time_index + 1]) < 3:
    st.info("Collect at least three timeline samples to calculate a trend-based watch.")
elif not forecasts:
    st.success(f"No threshold crossings estimated within the next {forecast_horizon_minutes} minutes.")
else:
    forecast_cols = st.columns(min(len(forecasts), 3))
    for forecast_index, forecast in enumerate(forecasts):
        eta_seconds = float(forecast["seconds_to_limit"])
        eta_text = f"{eta_seconds / 60:.1f} min" if eta_seconds >= 60 else f"{eta_seconds:.0f} sec"
        metric_labels = {
            "voltage_v": "Voltage",
            "current_a": "Current magnitude",
            "temperature_c": "Temperature",
            "soc_percent": "SoC",
        }
        metric_units = {
            "voltage_v": "V",
            "current_a": "A",
            "temperature_c": "°C",
            "soc_percent": "%",
        }
        metric_name = str(forecast["metric"])
        target = forecast_cols[forecast_index % len(forecast_cols)]
        with target:
            with st.container(border=True):
                st.markdown(f"**Approaching: {str(forecast['fault']).replace('_', ' ').title()}**")
                st.metric("Estimated time to threshold", eta_text)
                st.caption(
                    f"{metric_labels[metric_name]} trend: {float(forecast['current_value']):.2f} "
                    f"{metric_units[metric_name]} → demo limit {float(forecast['threshold']):.2f} "
                    f"{metric_units[metric_name]}"
                )

st.markdown("### Interactive visualisations")
fig1 = px.line(series, x="time_s", y="soc_percent", title="SoC versus time", labels={"time_s": "Time (s)", "soc_percent": "SoC (%)"})
fig2 = px.line(series, x="time_s", y="voltage_v", title="Voltage versus time", labels={"time_s": "Time (s)", "voltage_v": "Voltage (V)"})
fig3 = px.line(series, x="time_s", y="current_a", title="Current versus time", labels={"time_s": "Time (s)", "current_a": "Current (A)"})
fig4 = px.line(series, x="time_s", y="temperature_c", title="Temperature versus time", labels={"time_s": "Time (s)", "temperature_c": "Temperature (°C)"})
fig5 = px.line(series, x="time_s", y="power_w", title="Power versus time", labels={"time_s": "Time (s)", "power_w": "Power (W)"})
fig6 = px.line(series, x="time_s", y="cumulative_energy_wh", title="Cumulative energy versus time", labels={"time_s": "Time (s)", "cumulative_energy_wh": "Energy (Wh)"})

for fig in [fig1, fig2, fig3, fig4, fig5, fig6]:
    fig.update_layout(template="plotly_dark", hovermode="x unified")

plot_cols = st.columns(2)
plot_cols[0].plotly_chart(fig1, use_container_width=True)
plot_cols[1].plotly_chart(fig2, use_container_width=True)
plot_cols[0].plotly_chart(fig3, use_container_width=True)
plot_cols[1].plotly_chart(fig4, use_container_width=True)
plot_cols[0].plotly_chart(fig5, use_container_width=True)
plot_cols[1].plotly_chart(fig6, use_container_width=True)

st.subheader("Advanced battery analytics")
advanced_fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("Voltage vs current", "SoC and remaining energy", "Discharge vs charge energy", "Temperature profile"),
)
advanced_fig.add_trace(
    px.scatter(series, x="current_a", y="voltage_v", color="soc_percent", title="Voltage vs current").data[0],
    row=1,
    col=1,
)
advanced_fig.add_trace(
    px.line(series, x="time_s", y="soc_percent", title="SoC").data[0],
    row=1,
    col=2,
)
advanced_fig.add_trace(
    px.line(series, x="time_s", y="remaining_energy_wh", title="Remaining energy").data[0],
    row=1,
    col=2,
)
advanced_fig.add_trace(
    px.area(series, x="time_s", y="discharge_energy_wh", title="Discharge energy").data[0],
    row=2,
    col=1,
)
advanced_fig.add_trace(
    px.area(series, x="time_s", y="charge_returned_wh", title="Charge returned").data[0],
    row=2,
    col=1,
)
advanced_fig.add_trace(
    px.line(series, x="time_s", y="temperature_c", title="Temperature").data[0],
    row=2,
    col=2,
)
advanced_fig.update_layout(template="plotly_dark", height=800)
st.plotly_chart(advanced_fig, use_container_width=True)

st.subheader("3D battery state visualisation")
three_d_fig = build_battery_pack_figure(series, selected_time_index)
st.plotly_chart(three_d_fig, use_container_width=True)

st.subheader("Battery health radar")
radar_fig = build_health_radar_figure(preview_row)
st.plotly_chart(radar_fig, use_container_width=True)

st.subheader("Machine-learning model comparison")
model_path = Path("models/energy_predictor.joblib")
soc_model_path = Path("models/soc_predictor.joblib")
future_horizon_minutes = st.slider("Prediction horizon (minutes)", min_value=1, max_value=60, value=10)

if st.button("Train/retrain ML models"):
    dataset = build_synthetic_dataset(samples=500, seed=42)
    energy_result = train_energy_predictor(
        dataframe=dataset,
        future_horizon_minutes=future_horizon_minutes,
        model_path=model_path,
    )
    soc_result = train_soc_predictor(
        dataframe=dataset,
        model_path=soc_model_path,
    )
    st.session_state["energy_result"] = energy_result
    st.session_state["soc_result"] = soc_result
    st.success("Energy and SoC models trained and saved locally.")

energy_result = st.session_state.get("energy_result")
if energy_result is None:
    dataset = build_synthetic_dataset(samples=500, seed=42)
    energy_result = train_energy_predictor(
        dataframe=dataset,
        future_horizon_minutes=future_horizon_minutes,
        model_path=model_path,
    )
    st.session_state["energy_result"] = energy_result

soc_result = st.session_state.get("soc_result")
if soc_result is None:
    dataset = build_synthetic_dataset(samples=500, seed=42)
    soc_result = train_soc_predictor(
        dataframe=dataset,
        model_path=soc_model_path,
    )
    st.session_state["soc_result"] = soc_result

metrics = energy_result["metrics"]
energy_model = energy_result["model"]

st.caption("Demonstration models on held-out synthetic data. These results are experimental estimates and not validated real-world forecasts.")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Baseline MAE (Wh)", f"{metrics['baseline_mae_wh']:.3f}")
col_b.metric("Random Forest MAE (Wh)", f"{metrics['rf_mae_wh']:.3f}")
col_c.metric("Random Forest RMSE (Wh)", f"{metrics['rf_rmse_wh']:.3f}")

soc_comparison_df = pd.DataFrame(
    [
        {"Model": name, "MAE (%)": values["mae_pct"], "RMSE (%)": values["rmse_pct"]}
        for name, values in soc_result["comparison"].items()
    ]
)

soc_comparison_chart = px.bar(
    soc_comparison_df,
    x="Model",
    y=["MAE (%)", "RMSE (%)"],
    barmode="group",
    title="SoC model comparison",
)
soc_comparison_chart.update_layout(template="plotly_dark")
st.plotly_chart(soc_comparison_chart, use_container_width=True)

pred_row = {
    "current_a": float(last_row["current_a"]),
    "voltage_v": float(last_row["voltage_v"]),
    "soc_percent": float(last_row["soc_percent"]),
    "temperature_c": float(last_row["temperature_c"]),
    "mode_code": 0 if mode == "discharge" else 1,
}
ml_prediction_wh = predict_energy(energy_model, pred_row)
st.metric("Energy prediction for current state (Wh)", f"{ml_prediction_wh:.2f}", delta=f"{future_horizon_minutes} minutes")

soc_prediction_pct = predict_soc(soc_result["model"], {
    "current_a": float(last_row["current_a"]),
    "voltage_v": float(last_row["voltage_v"]),
    "temperature_c": float(last_row["temperature_c"]),
    "mode_code": 0 if mode == "discharge" else 1,
})
st.metric("SoC model estimate (%)", f"{soc_prediction_pct:.2f}", delta="Experimental")

plot_df = energy_result["evaluation_df"].copy()
actual_vs_pred_fig = px.scatter(
    plot_df,
    x="actual_energy_wh",
    y="random_forest_pred_wh",
    title="Actual versus predicted energy on the held-out test set",
    labels={"actual_energy_wh": "Actual energy (Wh)", "random_forest_pred_wh": "Predicted energy (Wh)"},
)
actual_vs_pred_fig.update_layout(template="plotly_dark")
st.plotly_chart(actual_vs_pred_fig, use_container_width=True)

soc_plot_df = soc_result["evaluation_df"].copy()
soc_pred_fig = px.scatter(
    soc_plot_df,
    x="actual_soc_percent",
    y="random_forest_pred_soc",
    title="Actual versus predicted SoC on the held-out test set",
    labels={"actual_soc_percent": "Actual SoC (%)", "random_forest_pred_soc": "Predicted SoC (%)"},
)
soc_pred_fig.update_layout(template="plotly_dark")
st.plotly_chart(soc_pred_fig, use_container_width=True)

st.subheader("Data export")
export_data = series.to_csv(index=False)
st.download_button("Download simulation CSV", export_data, file_name="caev_bms_simulation.csv", mime="text/csv")

ml_csv = energy_result["evaluation_df"].to_csv(index=False)
st.download_button("Download ML evaluation CSV", ml_csv, file_name="caev_bms_ml_evaluation.csv", mime="text/csv")

model_bytes = model_path.read_bytes() if model_path.exists() else b""
st.download_button("Download trained model", model_bytes, file_name="energy_predictor.joblib", mime="application/octet-stream")

st.subheader("Academic documentation")
st.markdown(
    """
    - Project objective: present a connected EV battery management simulation for demonstration.
    - System architecture: battery simulation, SoC estimator, safety logic, and ML predictor.
    - Data flow: sidebar settings -> battery simulation -> metrics and plots -> safety checks -> model training.
    - Mathematical models: SoC update uses Coulomb counting and power uses P = V × I; energy is integrated over time.
    - SoC method: simplified Coulomb counting with sign-aware charging/discharging logic and clamped output.
    - Energy calculation: Wh is estimated from power integration over the sampling interval.
    - Safety logic: configurable software thresholds only; not a certified BMS.
    - ML workflow: synthetic dataset generation, train/test split, baseline comparison, Random Forest regression, and model persistence.
    - Model evaluation: MAE and RMSE are reported on a held-out dataset for demonstration only.
    - Limitations: synthetic data, simplified voltage model, and no physical sensor or hardware protection integration.
    - Future hardware integration: ESP32, battery voltage sensing, ACS712 current sensing, DS18B20 temperature sensing, and hardware-level safety switching would be required for real deployment.
    """
)

report_text = generate_report(
    {
        "nominal_voltage_v": nominal_voltage_v,
        "capacity_ah": capacity_ah,
        "initial_soc": initial_soc,
        "mode": mode,
        "temperature_c": temperature_c,
        "duration_s": duration_s,
        "sampling_interval_s": sampling_interval_s,
    },
    alerts,
    metrics,
)

st.subheader("Summary report")
st.text(report_text)

report_bytes = report_text.encode("utf-8")
st.download_button("Download summary report", report_bytes, file_name="caev_bms_report.txt", mime="text/plain")

st.caption("This software simulation does not replace independent battery protection hardware or certified BMS safety hardware.")

if play_animation:
    time.sleep(max(0.1, 0.6 / playback_speed))
    st.rerun()
