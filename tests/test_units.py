import pytest

from air_quality_intelligence.transform.units import (
    CANONICAL_UNITS,
    PPB_TO_MG_M3_CO,
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


def test_co_labelled_ppb_is_read_as_ppm():
    """The provider correction, applied inside the conversion.

    The CO feed carries a "ppb" label on values that are ppm magnitudes, so
    the label is corrected before conversion. Taking the label at face value
    is what put every CO reading three orders of magnitude low.
    """
    value, unit = normalize_concentration(
        "co", 1.0, "ppb"
    )

    assert value == pytest.approx(1.145)
    assert unit == "mg/m³"

    corrected, _ = normalize_concentration("co", 1.0, "ppm")

    assert value == pytest.approx(corrected)


def test_co_ppb_arithmetic_is_still_available_for_a_genuine_ppb_feed():
    # 1000 ppb = 1 ppm, and 1 ppm CO = 1.145 mg/m³.
    assert 1000.0 * PPB_TO_MG_M3_CO == pytest.approx(1.145)


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


def test_normalizing_a_canonical_value_is_a_no_op():
    """Normalization must be safe to apply to its own output.

    CO's canonical unit is mg/m3, and that unit used to be rejected outright,
    so a second pass over already-normalized data either raised or, for the
    other pollutants, silently rescaled.
    """
    for pollutant, canonical in CANONICAL_UNITS.items():
        once, unit = normalize_concentration(pollutant, 12.5, canonical)
        twice, _ = normalize_concentration(pollutant, once, unit)

        assert unit == canonical, pollutant
        assert once == pytest.approx(12.5), pollutant
        assert twice == pytest.approx(once), pollutant


def test_mg_m3_is_accepted_for_every_pollutant():
    value, unit = normalize_concentration("pm25", 1.0, "mg/m³")

    assert value == pytest.approx(1000.0)
    assert unit == "µg/m³"

    value, unit = normalize_concentration("co", 1.0, "mg/m3")

    assert value == pytest.approx(1.0)
    assert unit == "mg/m³"


def test_non_finite_readings_are_dropped_before_storage():
    """DOUBLE PRECISION accepts NaN, so NOT NULL does not keep it out.

    Nothing downstream uses a NaN reading, but PostgreSQL propagates it
    through AVG and orders it above every real number, so one NaN reading
    took the top of every query ranked by value.
    """
    import numpy as np
    import pandas as pd

    from air_quality_intelligence.db.measurements import drop_non_finite

    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2026-07-01", periods=5, freq="h", tz="UTC"),
            "pollutant": ["pm25"] * 5,
            "value": [24.4, np.nan, np.inf, -np.inf, 31.0],
            "unit": ["µg/m³"] * 5,
            "source": ["openaq"] * 5,
        }
    )

    kept = drop_non_finite(frame)

    assert list(kept["value"]) == [24.4, 31.0]


def test_dropping_non_finite_leaves_a_clean_frame_untouched():
    import pandas as pd

    from air_quality_intelligence.db.measurements import drop_non_finite

    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2026-07-01", periods=3, freq="h", tz="UTC"),
            "pollutant": ["pm25"] * 3,
            "value": [10.0, 20.0, 30.0],
            "unit": ["µg/m³"] * 3,
            "source": ["openaq"] * 3,
        }
    )

    assert len(drop_non_finite(frame)) == 3
    assert drop_non_finite(pd.DataFrame()).empty
