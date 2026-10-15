import pytest
import pandas as pd

from src.race_ev_simulator import (
    RaceEVConfig,
    analyze_sector_times,
    rank_setup_tradeoffs,
    simulate_driver_inputs,
    simulate_race_ev,
    run_race_setup_sweep,
)


def test_race_ev_simulation_respects_speed_soc_and_lap_bounds():
    config = RaceEVConfig(
        duration_s=120.0,
        time_step_s=0.25,
        track_length_m=500.0,
        max_speed_kmh=150.0,
        initial_soc_pct=70.0,
    )

    result = simulate_race_ev(config)

    assert result["time_s"].iloc[0] == 0.0
    assert result["time_s"].iloc[-1] == 120.0
    assert result["speed_kmh"].min() >= 0.0
    assert result["speed_kmh"].max() <= config.max_speed_kmh
    assert result["soc_percent"].between(0.0, 100.0).all()
    assert result["lap"].max() >= 1
    assert result[["battery_temp_c", "motor_temp_c", "inverter_temp_c", "power_derate_pct"]].notna().all().all()
    assert result[["throttle_pct", "brake_pct", "virtual_rpm", "virtual_gear"]].notna().all().all()
    assert result["throttle_pct"].between(0.0, 100.0).all()
    assert result["brake_pct"].between(0.0, 100.0).all()
    assert result["virtual_gear"].between(1, 7).all()
    assert result["throttle_pct"].max() > 0.0
    assert result["brake_pct"].max() > 0.0


def test_race_ev_regenerative_braking_recovers_energy():
    result = simulate_race_ev(
        RaceEVConfig(
            duration_s=90.0,
            time_step_s=0.25,
            track_length_m=500.0,
            max_speed_kmh=150.0,
            initial_soc_pct=70.0,
        )
    )

    assert result["regen_power_kw"].max() > 0.0
    assert result["cumulative_regen_wh"].iloc[-1] > 0.0


def test_race_ev_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="mass_kg"):
        simulate_race_ev(RaceEVConfig(mass_kg=0.0))


def test_hot_ambient_conditions_trigger_thermal_power_derating():
    result = simulate_race_ev(
        RaceEVConfig(
            duration_s=30.0,
            time_step_s=0.25,
            ambient_temp_c=60.0,
            cooling_capacity_pct=0.0,
        )
    )

    assert result["battery_temp_c"].max() >= 60.0
    assert result["power_derate_pct"].min() < 100.0


def test_sector_analysis_interpolates_sector_times_from_distance():
    telemetry = pd.DataFrame(
        {
            "time_s": [0.0, 10.0, 20.0, 30.0],
            "distance_m": [0.0, 300.0, 600.0, 900.0],
        }
    )

    sectors = analyze_sector_times(telemetry, track_length_m=900.0)

    assert sectors["Sector"].tolist() == ["S1", "S2", "S3"]
    assert sectors["Sector time (s)"].tolist() == pytest.approx([10.0, 10.0, 10.0])


def test_sector_analysis_only_returns_completed_sectors():
    telemetry = pd.DataFrame(
        {
            "time_s": [0.0, 10.0, 20.0],
            "distance_m": [0.0, 300.0, 450.0],
        }
    )

    sectors = analyze_sector_times(telemetry, track_length_m=900.0)

    assert sectors["Sector"].tolist() == ["S1"]


def test_driver_sandbox_throttle_increases_speed_over_coast():
    common = {
        "mass_kg": 1000.0,
        "motor_peak_power_kw": 350.0,
        "battery_capacity_kwh": 54.0,
        "initial_soc_pct": 80.0,
        "initial_speed_kmh": 40.0,
        "brake_pct": 0.0,
        "regen_strength_pct": 65.0,
        "duration_s": 5.0,
    }

    throttle_run = simulate_driver_inputs(**common, throttle_pct=100.0)
    coast_run = simulate_driver_inputs(**common, throttle_pct=0.0)

    assert throttle_run["speed_kmh"].iloc[-1] > coast_run["speed_kmh"].iloc[-1]
    assert throttle_run["soc_percent"].iloc[-1] < common["initial_soc_pct"]


def test_driver_sandbox_braking_recovers_energy_and_reduces_speed():
    result = simulate_driver_inputs(
        mass_kg=1000.0,
        motor_peak_power_kw=350.0,
        battery_capacity_kwh=54.0,
        initial_soc_pct=70.0,
        initial_speed_kmh=120.0,
        throttle_pct=70.0,
        brake_pct=50.0,
        regen_strength_pct=65.0,
        duration_s=5.0,
    )

    assert result["speed_kmh"].iloc[-1] < 120.0
    assert result["throttle_pct"].max() == 0.0
    assert result["regen_power_kw"].max() > 0.0
    assert result["cumulative_regen_wh"].iloc[-1] > 0.0
    assert result["soc_percent"].iloc[-1] > 70.0


def test_driver_sandbox_default_throttle_run_does_not_instantly_hit_speed_cap():
    result = simulate_driver_inputs(
        mass_kg=1000.0,
        motor_peak_power_kw=350.0,
        battery_capacity_kwh=54.0,
        initial_soc_pct=90.0,
        initial_speed_kmh=100.0,
        throttle_pct=70.0,
        brake_pct=0.0,
        regen_strength_pct=65.0,
        duration_s=10.0,
    )

    assert 100.0 < result["speed_kmh"].iloc[-1] < 250.0


def test_setup_sweep_returns_metrics_for_every_power_and_cooling_pair():
    summary = run_race_setup_sweep(
        RaceEVConfig(duration_s=90.0, time_step_s=0.5, track_length_m=500.0),
        motor_power_options_kw=[150.0, 350.0],
        cooling_options_pct=[50.0, 100.0],
    )

    assert len(summary) == 4
    assert set(summary["Motor power (kW)"]) == {150.0, 350.0}
    assert set(summary["Cooling capacity (%)"]) == {50.0, 100.0}
    assert summary["Distance (km)"].gt(0.0).all()
    assert summary["Net energy (Wh/km)"].notna().all()
    assert summary["Best lap (s)"].notna().all()


def test_setup_tradeoff_ranking_flags_pareto_front_and_obeys_weight():
    candidates = pd.DataFrame(
        {
            "Motor power (kW)": [300.0, 250.0, 200.0, 350.0],
            "Best lap (s)": [60.0, 62.0, 65.0, 59.0],
            "Mean speed (km/h)": [170.0, 165.0, 155.0, 175.0],
            "Net energy (Wh/km)": [700.0, 600.0, 800.0, 900.0],
        }
    )

    pace_focused = rank_setup_tradeoffs(candidates, pace_weight=1.0)
    energy_focused = rank_setup_tradeoffs(candidates, pace_weight=0.0)

    assert set(pace_focused.loc[pace_focused["Pareto optimal"], "Motor power (kW)"]) == {250.0, 300.0, 350.0}
    assert pace_focused.iloc[0]["Motor power (kW)"] == 350.0
    assert energy_focused.iloc[0]["Motor power (kW)"] == 250.0


def test_setup_tradeoff_ranking_uses_mean_speed_without_full_laps():
    candidates = pd.DataFrame(
        {
            "Best lap (s)": [float("nan"), float("nan")],
            "Mean speed (km/h)": [100.0, 120.0],
            "Net energy (Wh/km)": [500.0, 600.0],
        }
    )

    ranked = rank_setup_tradeoffs(candidates, pace_weight=1.0)

    assert ranked.iloc[0]["Pace objective"] == "Mean speed (km/h)"
    assert ranked.iloc[0]["Mean speed (km/h)"] == 120.0
