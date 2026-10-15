"""Threshold-based safety monitoring for the CAEV-BMS simulation."""

from __future__ import annotations


class SafetyMonitor:
    """Simple software safety check for a demonstration dashboard.

    These thresholds are intentionally configured as demo values and must not be
    presented as certified battery protection limits for real hardware.
    """

    def __init__(
        self,
        undervoltage: float = 30.0,
        overvoltage: float = 58.0,
        overcurrent: float = 40.0,
        high_temperature: float = 50.0,
        low_soc: float = 20.0,
    ) -> None:
        self.undervoltage = float(undervoltage)
        self.overvoltage = float(overvoltage)
        self.overcurrent = float(overcurrent)
        self.high_temperature = float(high_temperature)
        self.low_soc = float(low_soc)

    def evaluate(self, metrics: dict) -> list[str]:
        """Return a list of active safety-warning codes."""
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be provided as a dictionary.")

        required = ["voltage", "current", "temperature", "soc"]
        for key in required:
            if key not in metrics:
                raise ValueError(f"Missing required metric: {key}")

        voltage = float(metrics["voltage"])
        current = float(metrics["current"])
        temperature = float(metrics["temperature"])
        soc = float(metrics["soc"])

        if voltage < 0 or current < 0 and abs(current) < 0:
            raise ValueError("Metric values must be numeric and physically meaningful.")
        if voltage < 0:
            raise ValueError("Voltage must not be negative.")
        if temperature < -50 or soc < 0 or soc > 100:
            raise ValueError("Temperature and SoC values are out of range.")

        warnings: list[str] = []
        if voltage < self.undervoltage:
            warnings.append("undervoltage")
        if voltage > self.overvoltage:
            warnings.append("overvoltage")
        if abs(current) > self.overcurrent:
            warnings.append("overcurrent")
        if temperature > self.high_temperature:
            warnings.append("high_temperature")
        if soc < self.low_soc:
            warnings.append("low_soc")
        return warnings

    def status(self, alerts: list[str]) -> str:
        return "warning" if alerts else "normal"
