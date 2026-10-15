"""Educational cycle-aging estimates for simulated battery telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_cycle_aging(
    dataframe: pd.DataFrame,
    capacity_ah: float,
    rated_cycle_life: float,
    end_of_life_soh_pct: float = 80.0,
    daily_efc: float = 1.0,
) -> pd.DataFrame:
    """Estimate cycle-based SoH and remaining life from amp-hour throughput.

    The model assumes a linear capacity fade from 100% SoH to the configured
    end-of-life threshold over the configured rated cycle life. One equivalent
    full cycle (EFC) is cumulative absolute amp-hour throughput divided by
    nominal capacity. Calendar aging, chemistry, temperature, and C-rate effects
    are intentionally not modeled; outputs are educational estimates only.
    """
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("A non-empty telemetry dataframe is required.")
    missing = {"time_s", "current_a"}.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing telemetry columns: {', '.join(sorted(missing))}")
    if not np.isfinite(capacity_ah) or capacity_ah <= 0:
        raise ValueError("capacity_ah must be a positive finite number.")
    if not np.isfinite(rated_cycle_life) or rated_cycle_life <= 0:
        raise ValueError("rated_cycle_life must be a positive finite number.")
    if not np.isfinite(end_of_life_soh_pct) or not 0 < end_of_life_soh_pct < 100:
        raise ValueError("end_of_life_soh_pct must be between 0 and 100.")
    if not np.isfinite(daily_efc) or daily_efc <= 0:
        raise ValueError("daily_efc must be a positive finite number.")

    times_s = dataframe["time_s"].to_numpy(dtype=float)
    currents_a = dataframe["current_a"].to_numpy(dtype=float)
    if not np.isfinite(times_s).all() or not np.isfinite(currents_a).all():
        raise ValueError("Telemetry time and current values must be finite.")
    time_steps_h = np.diff(times_s, prepend=times_s[0]) / 3600.0
    if np.any(time_steps_h < 0):
        raise ValueError("Telemetry timestamps must be in ascending order.")

    result = dataframe.copy()
    result["throughput_ah"] = np.cumsum(np.abs(currents_a) * time_steps_h)
    result["equivalent_full_cycles"] = result["throughput_ah"] / capacity_ah

    fade_per_cycle_pct = (100.0 - end_of_life_soh_pct) / rated_cycle_life
    result["soh_pct"] = np.maximum(
        end_of_life_soh_pct,
        100.0 - result["equivalent_full_cycles"] * fade_per_cycle_pct,
    )
    result["rul_cycles"] = np.maximum(
        rated_cycle_life - result["equivalent_full_cycles"],
        0.0,
    )
    result["rul_days"] = result["rul_cycles"] / daily_efc
    return result
