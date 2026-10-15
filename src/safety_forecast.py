"""Trend-based safety threshold watch for simulated battery telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {"time_s", "voltage_v", "current_a", "temperature_c", "soc_percent"}


def forecast_threshold_crossings(
    dataframe: pd.DataFrame,
    current_index: int,
    thresholds: dict[str, float],
    horizon_s: float,
    lookback_points: int = 8,
) -> list[dict[str, float | str]]:
    """Estimate near-future threshold crossings from recent linear trends.

    This is a lightweight scenario heuristic for synthetic/demo data, not a
    validated battery forecast or hardware protection function. A metric is
    forecast only when at least three recent samples have a trend toward a
    configured limit and the estimated crossing is inside ``horizon_s``.
    """
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("A non-empty telemetry dataframe is required.")
    missing_columns = _REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing telemetry columns: {', '.join(sorted(missing_columns))}")
    if current_index < 0 or current_index >= len(dataframe):
        raise ValueError("current_index is outside the telemetry dataframe.")
    if not np.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("horizon_s must be a positive finite number.")
    if lookback_points < 3:
        raise ValueError("lookback_points must be at least 3.")

    monitored_limits = {
        "undervoltage": ("voltage_v", "low", "undervoltage"),
        "overvoltage": ("voltage_v", "high", "overvoltage"),
        "overcurrent": ("current_a", "high_abs", "overcurrent"),
        "high_temperature": ("temperature_c", "high", "high_temperature"),
        "low_soc": ("soc_percent", "low", "low_soc"),
    }
    unknown_thresholds = set(thresholds).difference(monitored_limits)
    if unknown_thresholds:
        raise ValueError(f"Unknown threshold names: {', '.join(sorted(unknown_thresholds))}")

    start = max(0, current_index - lookback_points + 1)
    history = dataframe.iloc[start : current_index + 1]
    time_values = history["time_s"].to_numpy(dtype=float)
    if len(history) < 3 or not np.isfinite(time_values).all() or np.ptp(time_values) <= 0:
        return []

    current_time = float(dataframe.iloc[current_index]["time_s"])
    forecasts: list[dict[str, float | str]] = []
    for fault, (column, direction, threshold_name) in monitored_limits.items():
        if threshold_name not in thresholds:
            continue

        values = history[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        if direction == "high_abs":
            values = np.abs(values)
        current_value = float(values[-1])
        threshold = float(thresholds[threshold_name])
        if not np.isfinite(threshold):
            raise ValueError(f"Threshold {threshold_name} must be finite.")

        slope = float(np.polyfit(time_values - time_values[-1], values, 1)[0])
        moving_toward_limit = slope > 0 if direction in {"high", "high_abs"} else slope < 0
        currently_beyond_limit = current_value > threshold if direction in {"high", "high_abs"} else current_value < threshold
        if not moving_toward_limit or currently_beyond_limit:
            continue

        seconds_to_limit = (threshold - current_value) / slope
        if not np.isfinite(seconds_to_limit) or seconds_to_limit <= 0 or seconds_to_limit > horizon_s:
            continue

        forecasts.append(
            {
                "fault": fault,
                "metric": column,
                "current_value": current_value,
                "threshold": threshold,
                "seconds_to_limit": float(seconds_to_limit),
                "predicted_time_s": current_time + float(seconds_to_limit),
            }
        )

    return sorted(forecasts, key=lambda event: float(event["seconds_to_limit"]))
