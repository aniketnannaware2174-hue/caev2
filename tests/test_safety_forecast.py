import pandas as pd
import pytest

from src.safety_forecast import forecast_threshold_crossings


def make_telemetry(soc_values, temperature_values=None):
    size = len(soc_values)
    return pd.DataFrame(
        {
            "time_s": list(range(size)),
            "voltage_v": [48.0] * size,
            "current_a": [10.0] * size,
            "temperature_c": temperature_values or [25.0] * size,
            "soc_percent": soc_values,
        }
    )


def test_forecasts_low_soc_crossing_inside_horizon():
    telemetry = make_telemetry([90.0, 89.0, 88.0, 87.0])

    forecasts = forecast_threshold_crossings(
        telemetry,
        current_index=3,
        thresholds={"low_soc": 84.0},
        horizon_s=5.0,
    )

    assert len(forecasts) == 1
    assert forecasts[0]["fault"] == "low_soc"
    assert forecasts[0]["seconds_to_limit"] == pytest.approx(3.0)


def test_ignores_thresholds_when_trend_moves_away_or_outside_horizon():
    telemetry = make_telemetry([80.0, 81.0, 82.0, 83.0])

    forecasts = forecast_threshold_crossings(
        telemetry,
        current_index=3,
        thresholds={"low_soc": 84.0},
        horizon_s=5.0,
    )

    assert forecasts == []


def test_requires_three_samples_before_forecasting():
    telemetry = make_telemetry([90.0, 89.0])

    forecasts = forecast_threshold_crossings(
        telemetry,
        current_index=1,
        thresholds={"low_soc": 80.0},
        horizon_s=60.0,
    )

    assert forecasts == []
