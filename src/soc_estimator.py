"""State-of-charge estimation using Coulomb counting.

This module implements a simplified SoC estimator for academic demonstration.
It is intentionally not described as a high-accuracy electrochemical model.
"""

from __future__ import annotations


def coulomb_counting_soc(initial_soc: float, current_a: float, dt_hours: float, capacity_ah: float) -> float:
    """Estimate SoC using Coulomb counting.

    Relationship:
        SoC(t) = SoC(initial) - 100 * discharged_charge_ah / usable_capacity_ah

    Assumptions:
    - Positive current means discharge and removes charge from the battery.
    - Negative current means charging and adds charge back to the battery.
    - The usable battery capacity is approximated by the configured nominal capacity.
    - The estimate is a low-fidelity software model for demonstration only.

    Parameters
    ----------
    initial_soc : float
        Battery state of charge in percent.
    current_a : float
        Current in amperes. Positive values are discharge current.
    dt_hours : float
        Elapsed time in hours.
    capacity_ah : float
        Battery nominal usable capacity in ampere-hours.

    Returns
    -------
    float
        Updated SoC percentage constrained to the valid range 0-100.
    """
    if initial_soc < 0 or initial_soc > 100:
        raise ValueError("Initial SoC must be between 0 and 100 percent.")
    if capacity_ah <= 0:
        raise ValueError("Battery capacity must be positive.")
    if dt_hours < 0:
        raise ValueError("Time interval must be non-negative.")
    if dt_hours == 0 or current_a == 0:
        return float(max(0.0, min(100.0, initial_soc)))

    discharged_charge_ah = current_a * dt_hours
    updated_soc = initial_soc - 100.0 * discharged_charge_ah / capacity_ah
    return float(max(0.0, min(100.0, updated_soc)))
