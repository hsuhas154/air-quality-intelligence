from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from air_quality_intelligence.config.settings import settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_hourly_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,boundary_layer_height",
        "timezone": "UTC",
    }
    with httpx.Client(timeout=30) as client:
        response = client.get(settings.open_meteo_base_url, params=params)
        response.raise_for_status()
    hourly = response.json()["hourly"]
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(hourly["time"], utc=True),
            "temp_c": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "wind_dir": hourly["wind_direction_10m"],
            "blh": hourly["boundary_layer_height"],
        }
    )
