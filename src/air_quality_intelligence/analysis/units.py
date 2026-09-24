"""Unit normalization for air-quality pollutant concentrations."""

from __future__ import annotations

import math


# Canonical units expected by the CPCB AQI calculation.
CANONICAL_UNITS = {
    "pm25": "µg/m³",
    "pm10": "µg/m³",
    "no2": "µg/m³",
    "so2": "µg/m³",
    "o3": "µg/m³",
    "co": "mg/m³",
}


# CPCB CAAQM protocol conversion factors.
# Gas concentrations are converted to the canonical mass-concentration units.
PPB_TO_UG_M3 = {
    "no2": 1.88,
    "so2": 2.62,
    "o3": 1.96,
}

# CPCB:
#     1 ppm CO = 1.145 mg/m³
PPM_TO_MG_M3_CO = 1.145

# 1 ppb = 0.001 ppm
PPB_TO_MG_M3_CO = PPM_TO_MG_M3_CO / 1000.0


class UnitConversionError(ValueError):
    """Raised when a pollutant/unit combination cannot be normalized."""


def _normalize_unit(unit: str) -> str:
    """Normalize common Unicode/ASCII spellings of units."""
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
    """Convert one pollutant concentration to its canonical AQI unit.

    Canonical units:

    - PM2.5: µg/m³
    - PM10:  µg/m³
    - NO2:   µg/m³
    - SO2:   µg/m³
    - O3:    µg/m³
    - CO:    mg/m³

    Supported source units include the units observed in the project's
    OpenAQ data: µg/m³, ppb, and ppm.
    """
    pollutant = pollutant.strip().lower()

    if pollutant not in CANONICAL_UNITS:
        raise UnitConversionError(
            f"Unsupported pollutant: {pollutant}"
        )

    if value is None:
        raise UnitConversionError("Concentration cannot be None.")

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

    # PM pollutants are already in the correct unit.
    if unit in mass_units:
        if pollutant == "co":
            # CO's canonical AQI unit is mg/m³.
            return value / 1000.0, canonical_unit

        return value, canonical_unit

    # NO2, SO2 and O3: ppb → µg/m³.
    if pollutant in PPB_TO_UG_M3 and unit == "ppb":
        return (
            value * PPB_TO_UG_M3[pollutant],
            canonical_unit,
        )

    # CO: ppb → mg/m³.
    if pollutant == "co" and unit == "ppb":
        return (
            value * PPB_TO_MG_M3_CO,
            canonical_unit,
        )

    # CO: ppm → mg/m³.
    if pollutant == "co" and unit == "ppm":
        return (
            value * PPM_TO_MG_M3_CO,
            canonical_unit,
        )

    # O3: ppm → ppb → µg/m³.
    if pollutant == "o3" and unit == "ppm":
        return (
            value * 1000.0 * PPB_TO_UG_M3["o3"],
            canonical_unit,
        )

    raise UnitConversionError(
        f"Unsupported unit '{unit}' for pollutant '{pollutant}'."
    )
