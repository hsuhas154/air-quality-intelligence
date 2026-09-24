"""Unit normalization for air-quality pollutant concentrations."""

from __future__ import annotations

import math

CANONICAL_UNITS = {
    "pm25": "µg/m³",
    "pm10": "µg/m³",
    "no2": "µg/m³",
    "so2": "µg/m³",
    "o3": "µg/m³",
    "co": "mg/m³",
}


PPB_TO_UG_M3 = {
    "no2": 1.88,
    "so2": 2.62,
    "o3": 1.96,
}


PPM_TO_MG_M3_CO = 1.145
PPB_TO_MG_M3_CO = PPM_TO_MG_M3_CO / 1000.0


class UnitConversionError(ValueError):
    """Raised when a pollutant/unit combination cannot be normalized."""


def _normalize_unit(unit: str) -> str:
    return (
        unit.strip()
        .lower()
        .replace("μ", "µ")
    )


def normalize_concentration(
    pollutant: str,
    value: float,
    unit: str,
) -> tuple[float, str]:

    pollutant = pollutant.strip().lower()

    if pollutant not in CANONICAL_UNITS:
        raise UnitConversionError(
            f"Unsupported pollutant: {pollutant}"
        )

    if value is None:
        raise UnitConversionError(
            "Concentration cannot be None."
        )

    value = float(value)

    if not math.isfinite(value):
        raise UnitConversionError(
            "Concentration must be a finite number."
        )

    unit = _normalize_unit(unit)
    canonical_unit = CANONICAL_UNITS[pollutant]

    mass_units = {
        "µg/m³",
        "ug/m3",
        "µg/m3",
        "ug/m³",
    }

    if unit in mass_units:
        if pollutant == "co":
            return value / 1000.0, canonical_unit

        return value, canonical_unit

    if pollutant in PPB_TO_UG_M3 and unit == "ppb":
        return (
            value * PPB_TO_UG_M3[pollutant],
            canonical_unit,
        )

    if pollutant == "co" and unit == "ppb":
        return (
            value * PPB_TO_MG_M3_CO,
            canonical_unit,
        )

    if pollutant == "co" and unit == "ppm":
        return (
            value * PPM_TO_MG_M3_CO,
            canonical_unit,
        )

    if pollutant == "o3" and unit == "ppm":
        return (
            value * 1000.0 * PPB_TO_UG_M3["o3"],
            canonical_unit,
        )

    raise UnitConversionError(
        f"Unsupported unit '{unit}' "
        f"for pollutant '{pollutant}'."
    )


# Known provider unit-label corrections.
#
# OpenAQ exposes the CPCB CO feed with a "ppb" unit label, but the values it
# carries are ppm magnitudes. Station 17 sensor 12234782 reports CO around
# 0.45 to 0.48 over the sampled window. Interpreted as ppb that is 0.00055
# mg/m3, roughly three orders of magnitude below any ambient CO ever measured.
# Interpreted as ppm it is 0.55 mg/m3, which is an ordinary urban reading.
#
# NO2 and SO2 from the same stations are labelled ppb and their magnitudes are
# consistent with ppb (NO2 13.3 ppb -> 25 ug/m3, SO2 12.9 ppb -> 34 ug/m3),
# so the mislabelling is specific to CO rather than general to the feed.
PROVIDER_UNIT_CORRECTIONS: dict[tuple[str, str], str] = {
    ("co", "ppb"): "ppm",
}


def correct_reported_unit(pollutant: str, unit: str) -> str:
    """Correct a known provider unit-label error before conversion.

    Returns the unit that should be used for conversion, which is the reported
    unit unless the pollutant and unit pair is a documented provider error.
    """

    key = (pollutant.strip().lower(), _normalize_unit(unit))

    return PROVIDER_UNIT_CORRECTIONS.get(key, unit)
