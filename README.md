# Air Quality Intelligence Platform

Hourly CPCB air quality index forecasting for Indian cities, built on OpenAQ
observations and Open-Meteo weather, with the forecasting question treated as
something to test rather than assume.

The central finding so far is a negative one, and it is the point of the
project: at forecast horizons shorter than a day, a model that beats the
persistence baseline on paper is usually only exploiting the fact that the AQI
is a 24-hour rolling mean, so two windows less than 24 hours apart share most
of their observations. Once that overlap is removed, the easy win disappears.
`docs/` and `results/` carry the working.

## What is built

| Area | Status |
| --- | --- |
| OpenAQ ingestion, paginated, with station metadata | working |
| Open-Meteo hourly weather ingestion | working |
| PostgreSQL and PostGIS storage | working |
| Unit normalization with a provider-error correction and a plausibility guard | working |
| CPCB AQI calculator, checked against the published worked examples | working |
| Strict CPCB temporal averaging, 24-hour and 8-hour | working |
| Daily station AQI under the published minimum-data rules | working |
| Hourly feature table with exact time-based lags and weather covariates | working |
| EDA: diurnal and weekday profiles, correlations, station rankings | working |
| Persistence, climatology and Random Forest evaluation with chronological CV | working |
| Embargoed final holdout | working |
| Streamlit dashboard | not started, `app/streamlit_app.py` is a placeholder |
| SARIMA | present in `forecast/baseline.py`, never benchmarked |
| Cities beyond Delhi and Bengaluru | not ingested |

## Data

Two cities, Delhi and Bengaluru. 2.39 million pollutant measurements from 77
reporting stations, plus hourly weather, over a 91-day ingestion window.

| | Delhi | Bengaluru |
| --- | --- | --- |
| Stations publishing a daily AQI | 55 | 13 |
| Days with a published AQI | 85 | 85 |
| Mean daily AQI | 119.9 | 72.6 |
| Range | 29 to 712 | 31 to 607 |

Some numbers here are smaller than the raw counts, and the gaps are the
point:

- **77 stations report, 68 publish.** The other nine measure PM alone, so
  they never reach the three pollutants CPCB requires. A station that cannot
  produce a valid AQI is not a station this project can use.
- **85 days of 91.** Three days at the tail of the window are partial, which
  is expected. Three more, 27 to 29 August, hold measurements but no
  publishable AQI in either city: an ingestion gap deep enough that no
  station reached three pollutants with sixteen hours each. The pipeline is
  handling it correctly by publishing nothing, and the hole is real.
- **Delhi carries 27 to 31 reporting PM2.5 stations** against 55 publishing
  overall, which is why the PM2.5 coverage gate is expressed as a fraction of
  the median rather than a fixed count.

Ninety days is the practical ceiling on the OpenAQ and Open-Meteo endpoints
used here, and it is the main limitation on the modelling: the window holds
one season, so the model cannot learn seasonality and the evaluation cannot
test across seasons.

### Known bad sensors

Peenya, Bengaluru reports NO2 between 10 and 23 times its own city's median
on 06, 07, 08 and 18 August and September, peaking at a 24-hour mean of 648
ug/m3 while every other pollutant at that station sits at ordinary levels. A
genuine NO2 episode of that size raises the other combustion tracers with it.
Nothing else moved, on five separate days. That analyser is not measuring
NO2.

It is reported by `scripts/diagnose_aqi_extremes.py` and **not** excluded from
any result. Dropping a station is a decision with its own bias, and the
report exists so that the decision is visible rather than buried in a filter.

## The AQI calculator

`analysis/aqi.py` is the authoritative implementation, and it is checked
against CPCB's own calculator spreadsheet cell by cell:
`tests/test_aqi.py` carries a transcription of those formulas and compares
the two across a dense grid for every pollutant, including the sample inputs
the spreadsheet ships with. `docs/aqi-references.md` names the source for
every constant and records one retraction, where a government table that
prints closed top bands turned out to contradict the official calculator.

Three properties worth knowing before reading the code:

- The sub-index anchors on the previous category's upper limit, not on the
  band's displayed lower bound. The difference is up to one point on every
  value in the dataset.
- The top band is open and the arithmetic has no ceiling, so a sub-index can
  exceed 500. The 500 cap is a reporting convention applied separately, via
  `SubIndex.reported_value`. `SubIndex.extrapolated` and
  `daily_aqi.extrapolated` mark the readings that sit past the highest
  published breakpoint.
- A concentration of zero counts as missing, not as clean air, when deciding
  whether three pollutants are present. That is CPCB's rule, not ours.
- The AQI the model is trained on requires all six pollutants to be present,
  which is stricter than CPCB requires. That is a deliberate project choice,
  recorded in `analysis/features.py`, and it costs rows.

## Architecture

```text
OpenAQ ─────┐
            ├──> ingestion ──> PostgreSQL/PostGIS ──> features ──> AQI ──> evaluation
Open-Meteo ─┘         │              │                   │
                      │              │                   └── strict 24h/8h averaging
                      │              └── stations, measurements, weather, daily_aqi
                      └── retries, pagination, unit normalization
```

## Quick start

```bash
cp .env.example .env
# Add OPENAQ_API_KEY to .env

docker compose up -d db
psql "$DATABASE_URL" -f sql/schema.sql
```

Then, in order:

```bash
python scripts/ingest_openaq.py --days 90
python scripts/ingest_weather.py --days 90
python scripts/calculate_daily_aqi.py
python scripts/run_eda.py
python scripts/evaluate_horizon_forecast.py
python scripts/evaluate_horizon_holdout.py
```

Tests and linting:

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

## Evaluation design

The things that make the numbers trustworthy, and the reasons each is there:

- **Chronological expanding-window folds**, never a random split, with
  assertions that no fold's training set reaches past its test set.
- **An embargo** at each split boundary equal to the forecast horizon, so a
  training row's target cannot fall inside the test period.
- **A fold coverage guard.** An earlier run reported a 22 percent improvement
  from folds that between them covered 13 percent of the rows and none of the
  final month. The guard makes that failure loud instead of flattering.
- **Training-only imputation**, fitted on the training fold and applied to the
  test fold.
- **A leakage guard on the feature list**, so a target column cannot reach the
  model by being renamed.
- **A final holdout** that was embargoed and left untouched until the model
  design was fixed.

## Known gaps

- SARIMA is implemented but has never been benchmarked against persistence.
- The raw-concentration results at 6 and 12 hours were produced on the earlier
  30-day dataset and have not been regenerated.
- The dashboard does not exist yet.
- CPCB's own calculator has a transcription error in the O3 top band that
  would make the sub-index jump 65 points across one unit of ozone. This
  project uses the consistent form instead; see `docs/aqi-references.md`.

## Licence

See `LICENSE`.
