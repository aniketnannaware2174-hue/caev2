"""Utility functions for generating a text summary report."""

from __future__ import annotations


def generate_report(settings: dict, alerts: list[str], model_metrics: dict | None = None) -> str:
    """Create a short text report summarising simulation and demonstration metrics."""
    model_metrics = model_metrics or {}
    lines = [
        "Connected EV Intelligent Energy & Battery Management System (CAEV-BMS)",
        "Simulation report",
        "-" * 40,
        f"Nominal voltage: {settings.get('nominal_voltage_v', 'n/a')} V",
        f"Capacity: {settings.get('capacity_ah', 'n/a')} Ah",
        f"Initial SoC: {settings.get('initial_soc', 'n/a')} %",
        f"Operating mode: {settings.get('mode', 'n/a')}",
        f"Temperature: {settings.get('temperature_c', 'n/a')} °C",
        f"Duration: {settings.get('duration_s', 'n/a')} s",
        f"Sample interval: {settings.get('sampling_interval_s', 'n/a')} s",
        "",
        "Safety alerts:",
    ]

    if alerts:
        for alert in alerts:
            lines.append(f"- {alert}")
    else:
        lines.append("- No active safety alerts")

    lines.extend([
        "",
        "Model evaluation:",
    ])

    if model_metrics:
        for key, value in model_metrics.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No model evaluation available")

    lines.extend([
        "",
        "Note: This report is based on simulated demonstration data only and does not constitute hardware validation.",
    ])
    return "\n".join(lines)
