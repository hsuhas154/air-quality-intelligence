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
