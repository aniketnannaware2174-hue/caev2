import pandas as pd
import pytest

from src.battery_aging import estimate_cycle_aging


def test_cycle_aging_tracks_throughput_soh_and_remaining_life():
    telemetry = pd.DataFrame(
        {"time_s": [0.0, 3600.0, 7200.0], "current_a": [10.0, 10.0, 10.0]}
    )

    result = estimate_cycle_aging(
        telemetry,
        capacity_ah=10.0,
        rated_cycle_life=1000.0,
        end_of_life_soh_pct=80.0,
        daily_efc=2.0,
    )

    assert result["equivalent_full_cycles"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert result["soh_pct"].tolist() == pytest.approx([100.0, 99.98, 99.96])
    assert result["rul_cycles"].tolist() == pytest.approx([1000.0, 999.0, 998.0])
    assert result["rul_days"].tolist() == pytest.approx([500.0, 499.5, 499.0])


def test_cycle_aging_counts_charge_and_discharge_throughput_equally():
    times = [0.0, 1800.0, 3600.0]
    discharge = pd.DataFrame({"time_s": times, "current_a": [10.0, 10.0, 10.0]})
    charge = pd.DataFrame({"time_s": times, "current_a": [-10.0, -10.0, -10.0]})

    discharge_result = estimate_cycle_aging(discharge, 10.0, 1000.0)
    charge_result = estimate_cycle_aging(charge, 10.0, 1000.0)

    assert discharge_result["equivalent_full_cycles"].tolist() == pytest.approx(
        charge_result["equivalent_full_cycles"].tolist()
    )


def test_cycle_aging_rejects_non_ascending_timestamps():
    telemetry = pd.DataFrame({"time_s": [0.0, 5.0, 4.0], "current_a": [1.0, 1.0, 1.0]})

    with pytest.raises(ValueError, match="ascending"):
        estimate_cycle_aging(telemetry, capacity_ah=10.0, rated_cycle_life=1000.0)
