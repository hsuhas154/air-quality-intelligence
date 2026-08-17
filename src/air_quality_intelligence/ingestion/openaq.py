from __future__ import annotations

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from air_quality_intelligence.config.settings import settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_measurements(country_code: str = "IN", limit: int = 100) -> pd.DataFrame:
    if not settings.openaq_api_key:
        raise RuntimeError("OPENAQ_API_KEY is required for OpenAQ ingestion.")

    headers = {"X-API-Key": settings.openaq_api_key}
    params = {"countries_id": country_code, "limit": limit, "page": 1}
    url = f"{settings.openaq_base_url}/measurements"
    with httpx.Client(timeout=30, headers=headers) as client:
        response = client.get(url, params=params)
        response.raise_for_status()

    rows = response.json().get("results", [])
    records = []
    for row in rows:
        coords = row.get("coordinates") or {}
        records.append(
            {
                "station_id": row.get("locationId"),
                "station_name": row.get("location"),
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
                "ts": row.get("datetimeTo") or row.get("datetimeFrom"),
                "pollutant": row.get("parameter", {}).get("name"),
                "value": row.get("value"),
                "unit": row.get("unit", ""),
                "source": "openaq",
            }
        )
    return pd.DataFrame.from_records(records)
