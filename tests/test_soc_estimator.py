from src.soc_estimator import coulomb_counting_soc


def test_soc_decreases_during_discharge():
    soc = coulomb_counting_soc(80.0, 10.0, 0.5, 20.0)
    assert soc < 80.0


def test_soc_increases_during_charging():
    soc = coulomb_counting_soc(50.0, -10.0, 0.5, 20.0)
    assert soc > 50.0


def test_soc_remains_within_bounds():
    assert 0.0 <= coulomb_counting_soc(5.0, 100.0, 1.0, 20.0) <= 100.0
    assert 0.0 <= coulomb_counting_soc(95.0, -100.0, 1.0, 20.0) <= 100.0


def test_zero_current_produces_no_change():
    soc = coulomb_counting_soc(60.0, 0.0, 1.0, 20.0)
    assert soc == 60.0
