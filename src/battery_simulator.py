"""Battery simulation and time-series generation for the CAEV-BMS dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.energy_calculator import remaining_nominal_energy_wh
from src.soc_estimator import coulomb_counting_soc


@dataclass
class BatterySimulationConfig:
    nominal_voltage_v: float = 48.0
    capacity_ah: float = 40.0
    initial_soc: float = 80.0
    discharge_current_a: float = 15.0
    mode: str = "discharge"
    temperature_c: float = 27.0
    duration_s: float = 600.0
    sampling_interval_s: float = 30.0


def simulate_battery(config: BatterySimulationConfig) -> dict:
    """Generate a simple battery time series using a configurable voltage model.

    The model is intentionally simplified and explicit: it is a software demo, not
    a validated electrochemical simulation. Positive current indicates discharge,
    negative current indicates charging.
    """
    if config.nominal_voltage_v <= 0:
        raise ValueError("Nominal voltage must be positive.")
    if config.capacity_ah <= 0:
        raise ValueError("Battery capacity must be positive.")
    if config.initial_soc < 0 or config.initial_soc > 100:
        raise ValueError("Initial SoC must be between 0 and 100 percent.")
    if config.duration_s <= 0:
        raise ValueError("Duration must be positive.")
    if config.sampling_interval_s <= 0:
        raise ValueError("Sampling interval must be positive.")

    mode = str(config.mode).lower()
    if mode not in {"discharge", "charge"}:
        raise ValueError("Mode must be 'discharge' or 'charge'.")

    current_a = float(config.discharge_current_a)
    if mode == "charge":
        current_a = -abs(current_a)
    else:
        current_a = abs(current_a)

    n_steps = int(config.duration_s / config.sampling_interval_s)
    if n_steps <= 0:
        raise ValueError("The chosen duration and sample interval produce no simulation points.")

    timestamps = np.arange(0, config.duration_s + config.sampling_interval_s, config.sampling_interval_s)
    timestamps = timestamps[:n_steps]

    soc_values = np.empty(n_steps, dtype=float)
    voltage_values = np.empty(n_steps, dtype=float)
    current_values = np.empty(n_steps, dtype=float)
    temperature_values = np.empty(n_steps, dtype=float)
    power_values = np.empty(n_steps, dtype=float)
    cumulative_energy = np.zeros(n_steps, dtype=float)
    discharge_energy = np.zeros(n_steps, dtype=float)
    charge_returned = np.zeros(n_steps, dtype=float)
    remaining_energy = np.zeros(n_steps, dtype=float)

    current_soc = float(config.initial_soc)
    net_energy_wh = 0.0
    discharge_wh = 0.0
    charge_wh = 0.0

    for idx, ts in enumerate(timestamps):
        dt_hours = config.sampling_interval_s / 3600.0
        temp_offset = math.sin(ts / max(config.duration_s, 1.0) * 2 * math.pi) * 1.5
        temperature = config.temperature_c + temp_offset
        temperature_values[idx] = temperature

        voltage = config.nominal_voltage_v * (0.96 + 0.04 * (current_soc / 100.0))
        voltage -= 0.04 * abs(current_a)
        voltage += 0.02 * (temperature - 25.0)
        voltage = max(voltage, 0.1)
        voltage_values[idx] = voltage

        current_soc = coulomb_counting_soc(current_soc, current_a, dt_hours, config.capacity_ah)
        soc_values[idx] = current_soc

        power_w = voltage * current_a
        power_values[idx] = power_w

        net_energy_wh += power_w * dt_hours
        cumulative_energy[idx] = net_energy_wh

        if power_w >= 0:
            discharge_wh += power_w * dt_hours
        else:
            charge_wh += abs(power_w) * dt_hours
        discharge_energy[idx] = discharge_wh
        charge_returned[idx] = charge_wh
        remaining_energy[idx] = remaining_nominal_energy_wh(config.nominal_voltage_v, config.capacity_ah, current_soc)

        current_values[idx] = current_a

    df = pd.DataFrame(
        {
            "time_s": timestamps,
            "time_h": timestamps / 3600.0,
            "voltage_v": voltage_values,
            "current_a": current_values,
            "temperature_c": temperature_values,
            "soc_percent": soc_values,
            "power_w": power_values,
            "cumulative_energy_wh": cumulative_energy,
            "discharge_energy_wh": discharge_energy,
            "charge_returned_wh": charge_returned,
            "remaining_energy_wh": remaining_energy,
            "mode": mode,
        }
    )

    summary = {
        "final_voltage_v": float(df["voltage_v"].iloc[-1]),
        "final_current_a": float(df["current_a"].iloc[-1]),
        "final_temperature_c": float(df["temperature_c"].iloc[-1]),
        "final_soc_percent": float(df["soc_percent"].iloc[-1]),
        "final_power_w": float(df["power_w"].iloc[-1]),
        "final_cumulative_energy_wh": float(df["cumulative_energy_wh"].iloc[-1]),
        "remaining_energy_wh": float(df["remaining_energy_wh"].iloc[-1]),
    }
    return {"dataframe": df, "summary": summary}
