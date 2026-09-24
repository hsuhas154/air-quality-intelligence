import pytest

from air_quality_intelligence.analysis.units import (
    UnitConversionError,
    normalize_concentration,
)


def test_pm25_ug_m3_is_unchanged():
    value, unit = normalize_concentration(
        "pm25", 25.0, "µg/m³"
    )

    assert value == pytest.approx(25.0)
    assert unit == "µg/m³"


def test_no2_ppb_to_ug_m3():
    value, unit = normalize_concentration(
        "no2", 19.6, "ppb"
    )

    assert value == pytest.approx(36.848)
    assert unit == "µg/m³"


def test_so2_ppb_to_ug_m3():
    value, unit = normalize_concentration(
        "so2", 7.4, "ppb"
    )

    assert value == pytest.approx(19.388)
    assert unit == "µg/m³"


def test_o3_ppb_to_ug_m3():
    value, unit = normalize_concentration(
        "o3", 50.0, "ppb"
    )

    assert value == pytest.approx(98.0)
    assert unit == "µg/m³"


def test_o3_ppm_to_ug_m3():
    value, unit = normalize_concentration(
        "o3", 0.05, "ppm"
    )

    assert value == pytest.approx(98.0)
    assert unit == "µg/m³"


def test_co_ppb_to_mg_m3():
    # 1000 ppb = 1 ppm
    # 1 ppm CO = 1.145 mg/m³
    value, unit = normalize_concentration(
        "co", 1000.0, "ppb"
    )

    assert value == pytest.approx(1.145)
    assert unit == "mg/m³"


def test_co_ppm_to_mg_m3():
    value, unit = normalize_concentration(
        "co", 1.0, "ppm"
    )

    assert value == pytest.approx(1.145)
    assert unit == "mg/m³"


def test_co_ug_m3_to_mg_m3():
    value, unit = normalize_concentration(
        "co", 1000.0, "µg/m³"
    )

    assert value == pytest.approx(1.0)
    assert unit == "mg/m³"


def test_ascii_microgram_unit_is_supported():
    value, unit = normalize_concentration(
        "pm10", 50.0, "ug/m3"
    )

    assert value == pytest.approx(50.0)
    assert unit == "µg/m³"


def test_unsupported_unit_is_rejected():
    with pytest.raises(UnitConversionError):
        normalize_concentration(
            "no2", 20.0, "ppm"
        )


def test_unsupported_pollutant_is_rejected():
    with pytest.raises(UnitConversionError):
        normalize_concentration(
            "nh3", 10.0, "µg/m³"
        )


def test_nan_is_rejected():
    with pytest.raises(UnitConversionError):
        normalize_concentration(
            "pm25", float("nan"), "µg/m³"
        )


def test_infinity_is_rejected():
    with pytest.raises(UnitConversionError):
        normalize_concentration(
            "pm25", float("inf"), "µg/m³"
        )
