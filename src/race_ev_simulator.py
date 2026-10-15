"""Simplified longitudinal simulator for an electric formula-racing concept."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class RaceEVConfig:
    mass_kg: float = 1000.0
    motor_peak_power_kw: float = 350.0
    battery_capacity_kwh: float = 54.0
    initial_soc_pct: float = 90.0
    track_length_m: float = 3000.0
    duration_s: float = 600.0
    time_step_s: float = 0.5
    max_speed_kmh: float = 250.0
    regen_strength_pct: float = 65.0
    drive_efficiency: float = 0.92
    regen_efficiency: float = 0.82
    auxiliary_power_kw: float = 1.5
    ambient_temp_c: float = 25.0
    cooling_capacity_pct: float = 100.0


def _target_speed_mps(distance_m: float, track_length_m: float, max_speed_mps: float) -> float:
    """Create an illustrative circuit speed profile with three braking zones."""
    phase = (distance_m % track_length_m) / track_length_m
    target_speed = max_speed_mps
    corner_speed_mps = 90.0 / 3.6
    corner_width = 0.055
    for corner_center in (0.18, 0.49, 0.81):
        phase_delta = abs(phase - corner_center)
        phase_delta = min(phase_delta, 1.0 - phase_delta)
        corner_weight = np.exp(-0.5 * (phase_delta / corner_width) ** 2)
        target_speed -= (max_speed_mps - corner_speed_mps) * corner_weight
    return max(corner_speed_mps, float(target_speed))


def _thermal_derate(temp_c: float, soft_limit_c: float, hard_limit_c: float) -> float:
    """Return a smooth 100%-to-35% power scale over a thermal limit band."""
    if temp_c <= soft_limit_c:
        return 1.0
    progress = min((temp_c - soft_limit_c) / (hard_limit_c - soft_limit_c), 1.0)
    return 1.0 - 0.65 * progress


def simulate_race_ev(config: RaceEVConfig) -> pd.DataFrame:
    """Simulate speed, energy use, regeneration, and SoC around a concept track.

    This is a demonstrator, not a vehicle homologation model: it uses a simple
    longitudinal force balance, synthetic braking zones, and first-order
    powertrain thermal nodes. It is not a vehicle homologation or tire model.
    """
    positive_parameters = {
        "mass_kg": config.mass_kg,
        "motor_peak_power_kw": config.motor_peak_power_kw,
        "battery_capacity_kwh": config.battery_capacity_kwh,
        "track_length_m": config.track_length_m,
        "duration_s": config.duration_s,
        "time_step_s": config.time_step_s,
        "max_speed_kmh": config.max_speed_kmh,
    }
    for name, value in positive_parameters.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
    if not 0.0 <= config.initial_soc_pct <= 100.0:
        raise ValueError("initial_soc_pct must be between 0 and 100.")
    if not 0.0 <= config.regen_strength_pct <= 100.0:
        raise ValueError("regen_strength_pct must be between 0 and 100.")
    if not 0.0 < config.drive_efficiency <= 1.0:
        raise ValueError("drive_efficiency must be in (0, 1].")
    if not 0.0 < config.regen_efficiency <= 1.0:
        raise ValueError("regen_efficiency must be in (0, 1].")
    if not np.isfinite(config.auxiliary_power_kw) or config.auxiliary_power_kw < 0:
        raise ValueError("auxiliary_power_kw must be a non-negative finite number.")
    if not np.isfinite(config.ambient_temp_c) or not -30.0 <= config.ambient_temp_c <= 65.0:
        raise ValueError("ambient_temp_c must be between -30 and 65 degrees C.")
    if not np.isfinite(config.cooling_capacity_pct) or not 0.0 <= config.cooling_capacity_pct <= 150.0:
        raise ValueError("cooling_capacity_pct must be between 0 and 150.")

    time_s = np.arange(0.0, config.duration_s + config.time_step_s, config.time_step_s)
    time_s = time_s[time_s <= config.duration_s + 1e-9]
    sample_count = len(time_s)
    distance_m = np.zeros(sample_count)
    speed_mps = np.zeros(sample_count)
    acceleration_mps2 = np.zeros(sample_count)
    battery_power_kw = np.full(sample_count, config.auxiliary_power_kw)
    regen_power_kw = np.zeros(sample_count)
    soc_pct = np.full(sample_count, config.initial_soc_pct)
    cumulative_regen_wh = np.zeros(sample_count)
    target_speed_kmh = np.zeros(sample_count)
    battery_temp_c = np.full(sample_count, config.ambient_temp_c)
    motor_temp_c = np.full(sample_count, config.ambient_temp_c)
    inverter_temp_c = np.full(sample_count, config.ambient_temp_c)
    power_derate_pct = np.full(sample_count, 100.0)
    throttle_pct = np.zeros(sample_count)
    brake_pct = np.zeros(sample_count)
    virtual_gear = np.ones(sample_count, dtype=int)
    virtual_rpm = np.zeros(sample_count)

    mass = float(config.mass_kg)
    motor_power_w = float(config.motor_peak_power_kw) * 1000.0
    max_speed_mps = float(config.max_speed_kmh) / 3.6
    regen_power_limit_w = motor_power_w * 0.55 * float(config.regen_strength_pct) / 100.0
    battery_capacity_wh = float(config.battery_capacity_kwh) * 1000.0
    air_density = 1.225
    drag_area_m2 = 0.72
    rolling_resistance = 0.014
    regen_energy_wh = 0.0
    cooling_scale = config.cooling_capacity_pct / 100.0

    for index in range(sample_count):
        current_speed = speed_mps[index]
        current_distance = distance_m[index]
        target_speed = _target_speed_mps(current_distance, config.track_length_m, max_speed_mps)
        target_speed_kmh[index] = target_speed * 3.6
        battery_derate = _thermal_derate(battery_temp_c[index], 50.0, 70.0)
        motor_derate = _thermal_derate(motor_temp_c[index], 100.0, 145.0)
        inverter_derate = _thermal_derate(inverter_temp_c[index], 90.0, 130.0)
        drive_derate = min(battery_derate, motor_derate, inverter_derate)
        power_derate_pct[index] = drive_derate * 100.0

        if index == sample_count - 1:
            break

        dt = time_s[index + 1] - time_s[index]
        drag_force = 0.5 * air_density * drag_area_m2 * current_speed**2
        rolling_force = rolling_resistance * mass * 9.81
        resistance_force = drag_force + rolling_force
        current_soc = soc_pct[index]
        drive_force = 0.0
        brake_force = 0.0
        regen_force = 0.0
        mechanical_power_w = 0.0
        mechanical_regen_w = 0.0
        accepted_regen_w = 0.0
        motor_heat_w = 0.0
        inverter_heat_w = 0.0
        next_battery_power_w = config.auxiliary_power_kw * 1000.0

        if target_speed > current_speed + 0.35 and current_soc > 0.0:
            requested_accel = min(4.2, (target_speed - current_speed) * 0.85)
            requested_drive_force = mass * requested_accel + resistance_force
            available_power_force = motor_power_w * drive_derate / max(current_speed, 2.0)
            available_tire_force = 1.6 * mass * 9.81
            drive_force = min(requested_drive_force, available_power_force, available_tire_force)
            actual_accel = (drive_force - resistance_force) / mass
            throttle_pct[index] = float(np.clip(drive_force / (1.6 * mass * 9.81) * 100.0, 0.0, 100.0))
            mechanical_power_w = drive_force * current_speed
            drive_electrical_w = mechanical_power_w / config.drive_efficiency
            next_battery_power_w += drive_electrical_w
            motor_heat_w = mechanical_power_w * (1.0 - config.drive_efficiency)
            inverter_heat_w = drive_electrical_w * (1.0 - 0.97)
        elif target_speed < current_speed - 0.35:
            requested_decel = min(8.0, (current_speed - target_speed) * 1.25)
            brake_force = max(0.0, mass * requested_decel - resistance_force)
            brake_pct[index] = float(np.clip(brake_force / (8.0 * mass) * 100.0, 0.0, 100.0))
            regen_force = min(
                brake_force * config.regen_strength_pct / 100.0,
                regen_power_limit_w * drive_derate / max(current_speed, 2.0),
            )
            mechanical_regen_w = regen_force * current_speed
            accepted_regen_w = mechanical_regen_w * config.regen_efficiency
            remaining_capacity_wh = max(0.0, (100.0 - current_soc) * battery_capacity_wh / 100.0)
            accepted_regen_w = min(accepted_regen_w, remaining_capacity_wh * 3600.0 / dt)
            next_battery_power_w -= accepted_regen_w
            regen_power_kw[index] = accepted_regen_w / 1000.0
            regen_energy_wh += accepted_regen_w * dt / 3600.0
            motor_heat_w = mechanical_regen_w * (1.0 - config.regen_efficiency)
            inverter_heat_w = accepted_regen_w * (1.0 / 0.97 - 1.0)
            actual_accel = -(brake_force + resistance_force) / mass
        else:
            actual_accel = -resistance_force / mass

        next_speed = float(np.clip(current_speed + actual_accel * dt, 0.0, max_speed_mps))
        speed_mps[index + 1] = next_speed
        acceleration_mps2[index + 1] = (next_speed - current_speed) / dt
        distance_m[index + 1] = current_distance + (current_speed + next_speed) * 0.5 * dt
        battery_power_kw[index] = next_battery_power_w / 1000.0
        next_soc = current_soc - next_battery_power_w * dt / 3600.0 / battery_capacity_wh * 100.0
        soc_pct[index + 1] = float(np.clip(next_soc, 0.0, 100.0))
        cumulative_regen_wh[index + 1] = regen_energy_wh

        battery_current_a = abs(next_battery_power_w) / 800.0
        battery_heat_w = battery_current_a**2 * 0.025
        battery_temp_c[index + 1] = battery_temp_c[index] + (
            battery_heat_w - 180.0 * cooling_scale * (battery_temp_c[index] - config.ambient_temp_c)
        ) * dt / 180000.0
        motor_temp_c[index + 1] = motor_temp_c[index] + (
            motor_heat_w - 360.0 * cooling_scale * (motor_temp_c[index] - config.ambient_temp_c)
        ) * dt / 85000.0
        inverter_temp_c[index + 1] = inverter_temp_c[index] + (
            inverter_heat_w - 320.0 * cooling_scale * (inverter_temp_c[index] - config.ambient_temp_c)
        ) * dt / 48000.0

    battery_power_kw[-1] = battery_power_kw[-2] if sample_count > 1 else config.auxiliary_power_kw
    cumulative_regen_wh[-1] = regen_energy_wh
    if sample_count > 1:
        power_derate_pct[-1] = power_derate_pct[-2]
        throttle_pct[-1] = throttle_pct[-2]
        brake_pct[-1] = brake_pct[-2]
    gear_speed_edges = np.array([0.0, 35.0, 65.0, 95.0, 125.0, 155.0, 190.0])
    virtual_gear = np.searchsorted(gear_speed_edges, speed_mps * 3.6, side="right").clip(1, 7)
    virtual_rpm = np.clip((speed_mps * 3.6 / max(float(config.max_speed_kmh), 1.0)) * 12000.0, 0.0, 12000.0)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "distance_m": distance_m,
            "lap": np.floor(distance_m / config.track_length_m).astype(int),
            "speed_kmh": speed_mps * 3.6,
            "target_speed_kmh": target_speed_kmh,
            "acceleration_mps2": acceleration_mps2,
            "battery_power_kw": battery_power_kw,
            "regen_power_kw": regen_power_kw,
            "soc_percent": soc_pct,
            "cumulative_regen_wh": cumulative_regen_wh,
            "battery_temp_c": battery_temp_c,
            "motor_temp_c": motor_temp_c,
            "inverter_temp_c": inverter_temp_c,
            "power_derate_pct": power_derate_pct,
            "throttle_pct": throttle_pct,
            "brake_pct": brake_pct,
            "virtual_gear": virtual_gear,
            "virtual_rpm": virtual_rpm,
        }
    )


def analyze_sector_times(telemetry: pd.DataFrame, track_length_m: float) -> pd.DataFrame:
    """Calculate completed lap-sector times from distance/time telemetry."""
    required = {"time_s", "distance_m"}
    if not isinstance(telemetry, pd.DataFrame) or telemetry.empty:
        raise ValueError("A non-empty telemetry dataframe is required.")
    missing = required.difference(telemetry.columns)
    if missing:
        raise ValueError(f"Missing telemetry columns: {', '.join(sorted(missing))}")
    if not np.isfinite(track_length_m) or track_length_m <= 0:
        raise ValueError("track_length_m must be a positive finite number.")

    times = telemetry["time_s"].to_numpy(dtype=float)
    distances = telemetry["distance_m"].to_numpy(dtype=float)
    if not np.isfinite(times).all() or not np.isfinite(distances).all():
        raise ValueError("Telemetry time and distance values must be finite.")
    if np.any(np.diff(times) < 0) or np.any(np.diff(distances) < 0):
        raise ValueError("Telemetry time and distance must be in ascending order.")

    sector_length = float(track_length_m) / 3.0
    completed_distance = float(distances[-1])
    results: list[dict[str, float | int | str]] = []
    max_lap = int(completed_distance // track_length_m)
    for lap_index in range(max_lap + 1):
        lap_start_distance = lap_index * track_length_m
        for sector_index in range(3):
            sector_start = lap_start_distance + sector_index * sector_length
            sector_end = sector_start + sector_length
            if sector_end > completed_distance:
                continue
            start_time = float(np.interp(sector_start, distances, times))
            end_time = float(np.interp(sector_end, distances, times))
            results.append(
                {
                    "Lap": lap_index + 1,
                    "Sector": f"S{sector_index + 1}",
                    "Sector time (s)": end_time - start_time,
                    "Start distance (m)": sector_start,
                    "End distance (m)": sector_end,
                }
            )
    return pd.DataFrame(results, columns=["Lap", "Sector", "Sector time (s)", "Start distance (m)", "End distance (m)"])


def simulate_driver_inputs(
    *,
    mass_kg: float,
    motor_peak_power_kw: float,
    battery_capacity_kwh: float,
    initial_soc_pct: float,
    initial_speed_kmh: float,
    throttle_pct: float,
    brake_pct: float,
    regen_strength_pct: float,
    duration_s: float = 10.0,
    time_step_s: float = 0.1,
    drive_efficiency: float = 0.92,
    regen_efficiency: float = 0.82,
    auxiliary_power_kw: float = 1.5,
) -> pd.DataFrame:
    """Run a short educational what-if simulation from fixed pedal inputs.

    Braking overrides throttle. The model uses simple drag, rolling resistance,
    motor power limiting, and regenerative braking; it is not a tire or vehicle
    control model and should only be used to illustrate cause and effect.
    """
    parameters = {
        "mass_kg": mass_kg,
        "motor_peak_power_kw": motor_peak_power_kw,
        "battery_capacity_kwh": battery_capacity_kwh,
        "duration_s": duration_s,
        "time_step_s": time_step_s,
    }
    for name, value in parameters.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
    for name, value in {
        "initial_soc_pct": initial_soc_pct,
        "throttle_pct": throttle_pct,
        "brake_pct": brake_pct,
        "regen_strength_pct": regen_strength_pct,
    }.items():
        if not np.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be between 0 and 100.")
    if not np.isfinite(initial_speed_kmh) or initial_speed_kmh < 0:
        raise ValueError("initial_speed_kmh must be a non-negative finite number.")
    if not 0 < drive_efficiency <= 1 or not 0 < regen_efficiency <= 1:
        raise ValueError("Efficiencies must be in (0, 1].")
    if not np.isfinite(auxiliary_power_kw) or auxiliary_power_kw < 0:
        raise ValueError("auxiliary_power_kw must be a non-negative finite number.")

    time_s = np.arange(0.0, duration_s + time_step_s, time_step_s)
    time_s = time_s[time_s <= duration_s + 1e-9]
    count = len(time_s)
    speed_mps = np.zeros(count)
    speed_mps[0] = initial_speed_kmh / 3.6
    distance_m = np.zeros(count)
    acceleration_mps2 = np.zeros(count)
    battery_power_kw = np.full(count, auxiliary_power_kw)
    regen_power_kw = np.zeros(count)
    soc_pct = np.full(count, initial_soc_pct)
    cumulative_regen_wh = np.zeros(count)
    actual_throttle_pct = np.zeros(count)
    actual_brake_pct = np.full(count, brake_pct)

    mass = float(mass_kg)
    motor_power_w = float(motor_peak_power_kw) * 1000.0
    capacity_wh = float(battery_capacity_kwh) * 1000.0
    max_speed_mps = 250.0 / 3.6
    cumulative_regen = 0.0
    for index in range(count - 1):
        dt = time_s[index + 1] - time_s[index]
        speed = speed_mps[index]
        drag_force = 0.5 * 1.225 * 0.72 * speed**2
        rolling_force = 0.014 * mass * 9.81
        resistance = drag_force + rolling_force
        brake_force = float(brake_pct) / 100.0 * mass * 1.2 * 9.81
        drive_force = 0.0
        if brake_pct <= 0.0 and throttle_pct > 0.0 and soc_pct[index] > 0.0:
            requested_force = float(throttle_pct) / 100.0 * 4.2 * mass
            power_limited_force = motor_power_w / max(speed, 1.0)
            drive_force = min(requested_force, power_limited_force)

        regen_force = 0.0
        accepted_regen_w = 0.0
        if brake_force > 0.0 and speed > 0.0 and soc_pct[index] < 100.0:
            regen_power_limit_w = motor_power_w * 0.55 * float(regen_strength_pct) / 100.0
            regen_force = min(
                brake_force * float(regen_strength_pct) / 100.0,
                regen_power_limit_w / max(speed, 1.0),
            )
            remaining_capacity_wh = max(0.0, (100.0 - soc_pct[index]) * capacity_wh / 100.0)
            accepted_regen_w = min(
                regen_force * speed * regen_efficiency,
                remaining_capacity_wh * 3600.0 / dt,
            )

        mechanical_drive_w = drive_force * speed
        drive_electrical_w = mechanical_drive_w / drive_efficiency
        battery_power = drive_electrical_w + auxiliary_power_kw * 1000.0 - accepted_regen_w
        actual_acceleration = (drive_force - brake_force - resistance) / mass
        next_speed = float(np.clip(speed + actual_acceleration * dt, 0.0, max_speed_mps))
        speed_mps[index + 1] = next_speed
        acceleration_mps2[index + 1] = (next_speed - speed) / dt
        distance_m[index + 1] = distance_m[index] + 0.5 * (speed + next_speed) * dt
        battery_power_kw[index] = battery_power / 1000.0
        soc_pct[index + 1] = float(
            np.clip(soc_pct[index] - battery_power * dt / 3600.0 / capacity_wh * 100.0, 0.0, 100.0)
        )
        regen_power_kw[index] = accepted_regen_w / 1000.0
        cumulative_regen += accepted_regen_w * dt / 3600.0
        cumulative_regen_wh[index + 1] = cumulative_regen
        actual_throttle_pct[index] = float(throttle_pct) if brake_pct <= 0 else 0.0

    if count > 1:
        battery_power_kw[-1] = battery_power_kw[-2]
        actual_throttle_pct[-1] = actual_throttle_pct[-2]
    return pd.DataFrame(
        {
            "time_s": time_s,
            "distance_m": distance_m,
            "speed_kmh": speed_mps * 3.6,
            "acceleration_mps2": acceleration_mps2,
            "throttle_pct": actual_throttle_pct,
            "brake_pct": actual_brake_pct,
            "battery_power_kw": battery_power_kw,
            "regen_power_kw": regen_power_kw,
            "soc_percent": soc_pct,
            "cumulative_regen_wh": cumulative_regen_wh,
        }
    )


def run_race_setup_sweep(
    base_config: RaceEVConfig,
    motor_power_options_kw: list[float],
    cooling_options_pct: list[float],
) -> pd.DataFrame:
    """Compare motor-power and cooling setups under identical simulated conditions."""
    if not motor_power_options_kw or not cooling_options_pct:
        raise ValueError("At least one motor-power and cooling option is required.")

    results: list[dict[str, float]] = []
    for motor_power_kw in motor_power_options_kw:
        for cooling_pct in cooling_options_pct:
            setup = replace(
                base_config,
                motor_peak_power_kw=float(motor_power_kw),
                cooling_capacity_pct=float(cooling_pct),
            )
            telemetry = simulate_race_ev(setup)
            times_s = telemetry["time_s"].to_numpy(dtype=float)
            distances_m = telemetry["distance_m"].to_numpy(dtype=float)
            power_kw = telemetry["battery_power_kw"].to_numpy(dtype=float)
            net_energy_wh = float(np.sum(0.5 * (power_kw[:-1] + power_kw[1:]) * np.diff(times_s)) / 3.6)
            distance_km = float(distances_m[-1] / 1000.0)

            completed_laps = int(distances_m[-1] // setup.track_length_m)
            if completed_laps > 0:
                lap_distances = np.arange(1, completed_laps + 1, dtype=float) * setup.track_length_m
                lap_crossing_times = np.interp(lap_distances, distances_m, times_s)
                lap_times = np.diff(np.concatenate(([0.0], lap_crossing_times)))
                best_lap_s = float(np.min(lap_times))
            else:
                best_lap_s = float("nan")

            results.append(
                {
                    "Motor power (kW)": float(motor_power_kw),
                    "Cooling capacity (%)": float(cooling_pct),
                    "Best lap (s)": best_lap_s,
                    "Mean speed (km/h)": float(telemetry["speed_kmh"].mean()),
                    "Distance (km)": distance_km,
                    "Net energy (Wh)": net_energy_wh,
                    "Net energy (Wh/km)": net_energy_wh / max(distance_km, 1e-9),
                    "Regen recovered (Wh)": float(telemetry["cumulative_regen_wh"].iloc[-1]),
                    "Peak battery temp (°C)": float(telemetry["battery_temp_c"].max()),
                    "Peak motor temp (°C)": float(telemetry["motor_temp_c"].max()),
                    "Peak inverter temp (°C)": float(telemetry["inverter_temp_c"].max()),
                    "Minimum available power (%)": float(telemetry["power_derate_pct"].min()),
                    "Final SoC (%)": float(telemetry["soc_percent"].iloc[-1]),
                }
            )
    return pd.DataFrame(results)


def rank_setup_tradeoffs(setup_results: pd.DataFrame, pace_weight: float = 0.6) -> pd.DataFrame:
    """Rank setups by weighted pace/energy preference and flag the Pareto front.

    Lower lap time and lower net energy per distance are preferred. If no
    completed lap exists in the sweep, higher mean speed is used as the pace
    objective instead. The normalized preference score ranges from 0 (best)
    to 1 (worst) across the supplied candidate set.
    """
    if not isinstance(setup_results, pd.DataFrame) or setup_results.empty:
        raise ValueError("A non-empty setup result dataframe is required.")
    if not np.isfinite(pace_weight) or not 0.0 <= pace_weight <= 1.0:
        raise ValueError("pace_weight must be between 0 and 1.")
    if "Net energy (Wh/km)" not in setup_results:
        raise ValueError("Setup results must include 'Net energy (Wh/km)'.")

    ranked = setup_results.copy()
    if "Best lap (s)" in ranked and ranked["Best lap (s)"].notna().any():
        ranked = ranked.dropna(subset=["Best lap (s)"]).copy()
        pace_values = ranked["Best lap (s)"].to_numpy(dtype=float)
        pace_metric = "Best lap (s)"
        pace_cost = pace_values
    elif "Mean speed (km/h)" in ranked:
        pace_values = ranked["Mean speed (km/h)"].to_numpy(dtype=float)
        pace_metric = "Mean speed (km/h)"
        pace_cost = -pace_values
    else:
        raise ValueError("Setup results must include best lap time or mean speed.")

    energy_values = ranked["Net energy (Wh/km)"].to_numpy(dtype=float)
    if not np.isfinite(pace_cost).all() or not np.isfinite(energy_values).all():
        raise ValueError("Pace and energy objectives must be finite.")

    def normalize(values: np.ndarray) -> np.ndarray:
        span = float(np.max(values) - np.min(values))
        if span <= 0.0:
            return np.zeros_like(values, dtype=float)
        return (values - float(np.min(values))) / span

    pace_cost_normalized = normalize(pace_cost)
    energy_cost_normalized = normalize(energy_values)
    objective_values = np.column_stack((pace_cost_normalized, energy_cost_normalized))
    pareto_flags = []
    for candidate in objective_values:
        dominates_candidate = np.all(objective_values <= candidate, axis=1) & np.any(objective_values < candidate, axis=1)
        pareto_flags.append(not bool(np.any(dominates_candidate)))

    ranked["Pace objective"] = pace_metric
    ranked["Normalized pace cost"] = pace_cost_normalized
    ranked["Normalized energy cost"] = energy_cost_normalized
    ranked["Preference score"] = (
        float(pace_weight) * pace_cost_normalized
        + (1.0 - float(pace_weight)) * energy_cost_normalized
    )
    ranked["Pareto optimal"] = pareto_flags
    ranked["Trade-off status"] = np.where(ranked["Pareto optimal"], "Pareto optimal", "Dominated")
    return ranked.sort_values("Preference score", ascending=True).reset_index(drop=True)
