from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from air_quality_intelligence.config.settings import settings


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def fetch_sensor_measurements(
    sensor_id: int,
    start: datetime,
    end: datetime,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch raw measurements for one OpenAQ sensor."""

    headers = {"X-API-Key": settings.openaq_api_key}

    params = {
        "datetime_from": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "datetime_to": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "limit": limit,
    }

    url = f"{settings.openaq_base_url}/sensors/{sensor_id}/measurements"

    with httpx.Client(timeout=30, headers=headers) as client:
        response = client.get(url, params=params)
        response.raise_for_status()

    rows = response.json().get("results", [])

    records = []

    for row in rows:
        period = row.get("period") or {}
        datetime_from = period.get("datetimeFrom") or {}

        records.append(
            {
                "sensor_id": sensor_id,
                "ts": datetime_from.get("utc"),
                "pollutant": row.get("parameter", {}).get("name"),
                "value": row.get("value"),
                "unit": row.get("parameter", {}).get("units", ""),
                "source": "openaq",
            }
        )

    return pd.DataFrame.from_records(records)


def fetch_recent_sensor_measurements(
    sensor_id: int,
    hours: int = 24,
) -> pd.DataFrame:
    """Fetch recent measurements for a sensor."""

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)

    return fetch_sensor_measurements(
        sensor_id=sensor_id,
        start=start,
        end=end,
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def fetch_location(location_id: int) -> dict:
    """Fetch metadata for one OpenAQ monitoring location."""

    headers = {"X-API-Key": settings.openaq_api_key}
    url = f"{settings.openaq_base_url}/locations/{location_id}"

    with httpx.Client(timeout=30, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        raise ValueError(f"OpenAQ location {location_id} was not found.")

    return results[0]