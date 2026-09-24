from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from air_quality_intelligence.config.settings import settings


MAX_PAGE_SIZE = 1000


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def _fetch_measurement_page(
    sensor_id: int,
    start: datetime,
    end: datetime,
    page: int,
    limit: int,
) -> list[dict]:
    """Fetch one page of raw measurements for an OpenAQ sensor."""

    headers = {"X-API-Key": settings.openaq_api_key}

    params = {
        "datetime_from": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "datetime_to": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "limit": limit,
        "page": page,
    }

    url = f"{settings.openaq_base_url}/sensors/{sensor_id}/measurements"

    with httpx.Client(timeout=30, headers=headers) as client:
        response = client.get(url, params=params)
        response.raise_for_status()

    return response.json().get("results", [])


def fetch_sensor_measurements(
    sensor_id: int,
    start: datetime,
    end: datetime,
    limit: int = MAX_PAGE_SIZE,
    max_pages: int = 100,
) -> pd.DataFrame:
    """Fetch every raw measurement for one OpenAQ sensor in a time window.

    OpenAQ caps a single response at MAX_PAGE_SIZE results. A request that
    spans more observations than that is silently truncated to one page, so
    this walks the pages until the window is exhausted.

    At quarter-hourly resolution one page covers only about 10 days, which is
    why a multi-week window must be paginated rather than requested in one
    call.
    """

    if limit > MAX_PAGE_SIZE:
        raise ValueError(
            f"OpenAQ allows at most {MAX_PAGE_SIZE} results per page."
        )

    records: list[dict] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        rows = _fetch_measurement_page(
            sensor_id=sensor_id,
            start=start,
            end=end,
            page=page,
            limit=limit,
        )

        for row in rows:
            period = row.get("period") or {}
            datetime_from = period.get("datetimeFrom") or {}
            timestamp = datetime_from.get("utc")

            # The same observation must never be counted twice if the
            # provider returns overlapping pages.
            if timestamp in seen:
                continue

            seen.add(timestamp)

            records.append(
                {
                    "sensor_id": sensor_id,
                    "ts": timestamp,
                    "pollutant": row.get("parameter", {}).get("name"),
                    "value": row.get("value"),
                    "unit": row.get("parameter", {}).get("units", ""),
                    "source": "openaq",
                }
            )

        # A short page means the window is exhausted.
        if len(rows) < limit:
            break

    return pd.DataFrame.from_records(records)


def fetch_recent_sensor_measurements(
    sensor_id: int,
    hours: int = 24,
) -> pd.DataFrame:
    """Fetch recent measurements for a sensor."""

    end = datetime.now(UTC)
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