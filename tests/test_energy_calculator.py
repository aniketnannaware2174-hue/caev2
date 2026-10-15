from src.energy_calculator import calculate_energy_wh, power_from_voltage_current


def test_energy_calculations_use_correct_units():
    power = power_from_voltage_current(48.0, 10.0)
    assert power == 480.0

    energy = calculate_energy_wh([480.0, 480.0], [0.5, 0.5])
    assert energy == 480.0


def test_invalid_inputs_are_handled():
    try:
        power_from_voltage_current(0.0, 5.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero voltage")

    try:
        calculate_energy_wh([], [])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty input")
