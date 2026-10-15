from src.safety_monitor import SafetyMonitor


def test_safety_warnings_activate_when_thresholds_are_crossed():
    monitor = SafetyMonitor(
        undervoltage=30.0,
        overvoltage=60.0,
        overcurrent=15.0,
        high_temperature=45.0,
        low_soc=20.0,
    )

    alerts = monitor.evaluate({
        "voltage": 25.0,
        "current": 20.0,
        "temperature": 50.0,
        "soc": 10.0,
    })

    assert "undervoltage" in alerts
    assert "overcurrent" in alerts
    assert "high_temperature" in alerts
    assert "low_soc" in alerts


def test_invalid_inputs_are_handled():
    monitor = SafetyMonitor()
    try:
        monitor.evaluate({"voltage": -1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid metric")
