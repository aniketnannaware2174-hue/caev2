# Connected EV Intelligent Energy & Battery Management System (CAEV-BMS)

This project is a self-contained simulation of a battery management dashboard for academic demonstration. It is designed to run locally with Python, Streamlit, NumPy, Pandas, Plotly, scikit-learn, and Joblib. It does not require hardware sensors, cloud services, or paid APIs.

## Overview

The dashboard simulates a battery pack and a formula-racing electric vehicle, tracks state of charge (SoC) using Coulomb counting, calculates power and energy, monitors simple software safety thresholds, evaluates a machine-learning energy prediction model based on synthetic demonstration data, and provides educational cycle-based state-of-health (SoH) and remaining-useful-life (RUL) estimates.

> Important: all values are simulated and intended for educational demonstration only. They are not manufacturer-approved safety limits or validated measurements.

## Project structure

- `app.py` — Streamlit dashboard entry point
- `src/battery_simulator.py` — battery simulation and time-series generator
- `src/soc_estimator.py` — Coulomb-counting SoC estimation
- `src/energy_calculator.py` — voltage, power, and energy calculations
- `src/safety_monitor.py` — threshold-based monitoring logic
- `src/safety_forecast.py` — trend-based simulated threshold watch
- `src/battery_aging.py` — cycle-throughput SoH and RUL estimate
- `src/race_ev_simulator.py` — formula-racing EV longitudinal and energy simulation
- `src/ml_predictor.py` — synthetic data generation, model training, and evaluation
- `src/report_generator.py` — summary report generation
- `tests/` — automated tests for core logic
- `models/` — trained model output
- `data/` — exported CSV files

## Windows PowerShell setup

```powershell
cd "C:\Users\pravi\Desktop\project caev"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run tests

```powershell
cd "C:\Users\pravi\Desktop\project caev"
.\.venv\Scripts\Activate.ps1
python -m pytest
```

## Launch the dashboard

```powershell
cd "C:\Users\pravi\Desktop\project caev"
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

## Notes

- The dashboard uses a simplified configurable voltage model rather than a full electrochemical battery model.
- Current sign convention: positive current means discharge, negative current means charging.
- Safety thresholds are demonstration values only and must not be treated as certified BMS limits.
- The SoH/RUL dashboard uses a configurable linear capacity-fade assumption based on equivalent full cycles. It excludes calendar aging, chemistry, temperature, and charge-rate effects, so it is an educational projection—not a validated battery-life prediction.
- The predictive safety watch extrapolates recent synthetic telemetry and is not a hardware protection function.
- The formula-racing vehicle is a configurable concept model with a synthetic circuit speed profile, simplified force/energy balance, and first-order battery/motor/inverter thermal nodes with power derating; it is not an official F1/Formula E vehicle model or validated lap simulation.
- Its telemetry analysis panel uses a public motorsport convention of aligned speed, throttle, brake, RPM, gear, distance, and sector channels. Throttle/brake are model demands; RPM and gear are explicitly virtual display proxies. Sector references are calculated from the current simulated run only.
- The driver-input sandbox is a short what-if teaching model. It demonstrates simplified throttle, braking, energy, and regenerative effects; it is not suitable for vehicle control, performance prediction, or safety decisions.
- The setup sweep compares a small configured grid of motor power and cooling assumptions using best completed simulated lap and net energy per distance. A weighted preference ranks pace versus efficiency and marks Pareto-optimal setups; these rankings are only meaningful inside the synthetic model and are not real race setup guidance.
- Actual sensor acquisition, ESP32 integration, and hardware switching are not implemented in this version unless extended later.

## Academic documentation summary

The project demonstrates the workflow of a connected EV battery management concept: data acquisition simulation, SoC estimation, energy accounting, safety monitoring, and ML-based prediction. National Laboratory of the Rockies describes battery-lifetime work spanning diagnosis, prediction, and optimization, and notes the importance of cycling, temperature, operating windows, and other aging factors. This project's simplified cycle-fade curve illustrates that workflow only; real health and RUL models require suitable aging data and validation.

Research background: [National Laboratory of the Rockies — Battery Lifespan](https://www.nlr.gov/transportation/battery-lifespan).

Telemetry-analysis background: [FastF1 documentation — Timing and Telemetry Data](https://docs.fastf1.dev/core.html) describes multi-channel telemetry, including speed, RPM, gear, throttle, brake, position, distance, and lap/sector information. FastF1 also notes telemetry may contain interpolated values, so the dashboard labels its own generated channels as simulated.
