"""Energy and power calculation helpers for the CAEV-BMS simulation."""

from __future__ import annotations


def power_from_voltage_current(voltage_v: float, current_a: float) -> float:
    """Return instantaneous power in watts.

    Current sign convention: positive values indicate discharge, negative values
    indicate charging. The power is therefore positive during discharge and
    negative during charge.
    """
    if voltage_v <= 0:
        raise ValueError("Voltage must be positive for a valid power calculation.")
    return float(voltage_v * current_a)


def calculate_energy_wh(power_watts, dt_hours) -> float:
    """Integrate power over time to estimate energy in Wh.

    Parameters
    ----------
    power_watts : iterable of float
        Instantaneous power values in watts.
    dt_hours : iterable of float
        Time intervals in hours for each power sample.
    """
    if power_watts is None or dt_hours is None:
        raise ValueError("Power and time inputs must not be None.")
    power_list = list(power_watts)
    time_list = list(dt_hours)
    if not power_list or not time_list:
        raise ValueError("Power and time arrays must not be empty.")
    if len(power_list) != len(time_list):
        raise ValueError("Power and time arrays must have the same length.")

    total_energy_wh = 0.0
    for power_w, delta_hours in zip(power_list, time_list):
        if delta_hours < 0:
            raise ValueError("Time intervals must be non-negative.")
        total_energy_wh += power_w * delta_hours
    return float(total_energy_wh)


def energy_components_wh(power_watts, dt_hours):
    """Separate discharge, charge, and net energy from the power waveform."""
    if power_watts is None or dt_hours is None:
        raise ValueError("Power and time inputs must not be None.")
    power_list = list(power_watts)
    time_list = list(dt_hours)
    if not power_list or not time_list:
        raise ValueError("Power and time arrays must not be empty.")
    if len(power_list) != len(time_list):
        raise ValueError("Power and time arrays must have the same length.")

    discharge_wh = 0.0
    charge_wh = 0.0
    for power_w, delta_hours in zip(power_list, time_list):
        if delta_hours < 0:
            raise ValueError("Time intervals must be non-negative.")
        if power_w >= 0:
            discharge_wh += power_w * delta_hours
        else:
            charge_wh += abs(power_w) * delta_hours
    net_wh = discharge_wh - charge_wh
    return discharge_wh, charge_wh, net_wh


def remaining_nominal_energy_wh(nominal_voltage_v: float, capacity_ah: float, soc_percent: float) -> float:
    """Return nominal remaining energy in Wh based on SoC."""
    if nominal_voltage_v <= 0:
        raise ValueError("Nominal voltage must be positive.")
    if capacity_ah <= 0:
        raise ValueError("Capacity must be positive.")
    if soc_percent < 0 or soc_percent > 100:
        raise ValueError("SoC percent must be in the range 0 to 100.")
    return float(nominal_voltage_v * capacity_ah * (soc_percent / 100.0))
