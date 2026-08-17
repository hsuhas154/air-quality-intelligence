# Air Quality Intelligence Platform

End-to-end air-quality analytics and forecasting for Indian cities, combining pollutant observations, weather covariates, geospatial station data, statistical analysis and an interactive dashboard.

## Phase 1 MVP

- **Ingestion:** OpenAQ + Open-Meteo with retry handling
- **Storage:** PostgreSQL + PostGIS
- **Transformation:** Pandas + SQL, CPCB-style AQI sub-index calculation
- **Analysis:** hourly/weekday profiles, correlations and station rankings
- **Forecasting:** naive baseline + SARIMA, evaluated with MAE/RMSE
- **Dashboard:** Streamlit + Plotly/PyDeck-ready architecture
- **Engineering:** typed modular package, pytest, Ruff, Docker Compose, GitHub Actions

## Architecture

```text
OpenAQ ─────┐
            ├──> Python ingestion ──> PostgreSQL/PostGIS ──> transform ──> analysis/forecast ──> Streamlit
Open-Meteo ─┘          │                    │
                       └── retries/logging  └── stations + time series + derived AQI
```

## Cities

Phase 1 target: Delhi, Mumbai, Bengaluru, Kolkata and Chennai. Start with two cities for the first working prototype, then scale the same pipeline to five.

## Quick start

```bash
cp .env.example .env
# Add OPENAQ_API_KEY to .env

docker compose up --build
```

Dashboard: `http://localhost:8501`

Run tests locally:

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

## Planned milestones

1. Connect live OpenAQ measurements and station metadata.
2. Add 60–90 day backfill and scheduled ingestion.
3. Persist hourly weather for all target cities.
4. Compute daily AQI and dominant pollutant.
5. Add EDA, station ranking and AQI-weather analysis.
6. Add 24–48 hour forecast evaluation against the naive baseline.
7. Wire the dashboard to PostGIS and publish the application.

## Phase 2

Sentinel-5P/TROPOMI satellite overlays, anomaly detection and alerts, dbt analytics models, FastAPI service layer, stronger ML forecasting and CI/CD deployment.
