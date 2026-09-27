# Air Quality Intelligence Platform

Hourly CPCB air quality index forecasting for Indian cities, built on OpenAQ
observations and Open-Meteo weather, with the forecasting question treated as
something to test rather than assume.

Two findings, and the second one is the point of the project.

**A four-parameter SARIMA beats a forty-five-feature Random Forest, and at 18
hours it beats persistence too.** The Random Forest was given weather, lags,
station counts and six pollutants. It lost to a univariate model that sees
nothing but the AQI's own history, by 25 percent at 24 hours.

**Most apparent wins in this project turned out to be artifacts, and finding
that out is the work.** A 22 percent improvement came from folds covering 13
percent of the record. A later run reported a mean absolute error of 2.6e54,
because an unconstrained optimiser had found an explosive AR root. Both were
caught by guardrails added after the fact, and both are documented here
rather than quietly removed from the history.

### Why persistence is hard to beat, and why it is not a straw man

The AQI at any hour is a mean over the previous 24 hours, so the AQI now and
the AQI h hours from now are built from windows sharing `(24-h)/24` of their
observations. Persistence gets that for free, which is why beating it below
24 hours proves nothing.

| Horizon | Window overlap | Autocorrelation |
| --- | --- | --- |
| 1h | 96% | 0.999 |
| 6h | 75% | 0.976 |
| 12h | 50% | 0.924 |
| 18h | 25% | 0.860 |
| 24h | **0%** | **0.793** |

The second column is the arithmetic. The third is not. **At 24 hours the
windows share nothing and the AQI is still correlated at 0.79**, and that
residual is real atmospheric persistence: weather regimes outlast a day.
Persistence at 24 hours is therefore a serious baseline rather than an
artifact, which is what makes a model that beats it worth something and a
model that loses to it worth reporting.

This corrects an earlier claim. `analysis/horizon.py` used to document an
autocorrelation of **-0.085** at 24 hours and conclude that the correlation
was nothing but overlap. Those figures came from the 30-day dataset and from
an AQI that was wrong in several ways at once, and they do not survive the
corrected data. The table above is recomputed from the feature table by
`dashboard.data.window_overlap_table` rather than restated, so it cannot go
stale the same way.

## Results

24-hour horizon, against persistence, with moving-block bootstrap intervals
on the mean gain in MAE:

| Model | Cross-validation, n=1340 | Embargoed holdout, n=451 |
| --- | --- | --- |
| SARIMA | +0.68 [-0.59, +2.08] | +1.31 [-0.14, +3.00] |
| Random Forest | -4.79 [-8.81, -1.37] | -0.53 [-2.27, +3.54] |
| Climatology | -19.54 [-25.65, -13.32] | -4.74 [-10.80, +3.30] |

18-hour horizon, cross-validation only:

| Model | Mean gain in MAE | Win rate |
| --- | --- | --- |
| SARIMA | **+1.43 [+0.34, +2.83]** | 55.8% |
| Random Forest | -2.39 [-4.35, -0.18] | 46.4% |
| Climatology | -23.37 [-29.66, -17.11] | 17.2% |

What that supports, and what it does not:

- **At 18 hours SARIMA beats persistence.** The interval excludes zero.
- **At 24 hours it is not established.** Both point estimates are positive
  and cross-validation and the holdout agree on the direction, but neither
  interval excludes zero. The honest statement is "consistent with a gain,
  not demonstrated".
- **The Random Forest is worse than persistence**, clearly so in
  cross-validation at both horizons.
- **The holdout resolves almost nothing**, and that is a property of its
  size rather than of the models. 451 observations is 19 blocks of 24 hours.
  Climatology's 29 percent deficit does not clear the interval either, which
  is why a non-significant result here is reported as "not resolved" with
  its block count attached, never as "indistinguishable".

Reproduce with `python scripts/evaluate_horizon_forecast.py --sarima` and
`python scripts/evaluate_horizon_holdout.py --sarima`.

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
| Streamlit dashboard | working, reads the committed outputs, no database needed |
| SARIMA, walk-forward, on the same folds as the other models | working, `--sarima` |
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

The dashboard needs none of that. It reads the committed CSVs in `outputs/`,
so a fresh clone can start it straight away:

```bash
streamlit run app/streamlit_app.py
```

That is deliberate. Requiring the database would mean requiring ninety days
of ingestion first, and a dashboard nobody can start is not a dashboard. It
shows the overlap finding, the model comparison with its intervals, the AQI
history, and a section on what the data does not cover. Its numbers go
through `analysis.significance`, the same code the command-line evaluation
uses, so it cannot quietly disagree with the run that produced its inputs.

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

### On benchmarking SARIMA

`python scripts/evaluate_horizon_forecast.py --sarima` scores it beside
persistence, climatology and the Random Forest, on the same folds, the same
target and the same embargo. Three things make that comparison honest rather
than decorative:

- **The series keeps its gaps.** The AQI has holes, including three days at
  the end of August. The previous implementation called `dropna()` before
  fitting, which closes the holes and shifts every later observation earlier,
  so a 24-hour seasonal term was being estimated against a series that was no
  longer hourly.
- **Parameters come from the training window only.** The order is chosen by
  AIC over a small candidate set, refitted per fold. At each test hour the
  filter is advanced with the observations that existed then, and asked for a
  forecast; the coefficients never see the test period.
- **A failed fit is reported, not replaced.** The previous implementation
  fell back to the naive forecast below 48 points, which would have published
  persistence's score under SARIMA's name. Rows SARIMA cannot predict are
  counted, and if it covers fewer rows than the other models, every model is
  scored again on exactly the rows it managed.

It is off by default because it fits every candidate order per fold per city
and then steps a Kalman filter through every hour.

The first run of it reported a mean absolute error of 2.6e54. That was not a
bad forecast, it was a divergent one: `enforce_stationarity` was off, so the
optimiser was free to settle on an AR root inside the unit circle, and an
explosive process compounds like phi to the power of the walk. The AQI makes
that trap easy to fall into, because a 24-hour rolling mean has a nearly flat
differenced series and therefore a nearly flat likelihood surface for an
unconstrained optimiser to wander across. The fit is now constrained, and a
forecast outside `[0, 5x the training maximum]` is discarded and counted
rather than scored. It is never clamped: clamping a divergent forecast to a
plausible number hides the divergence inside a respectable-looking error.

## Is a difference real?

Every model is also reported against persistence as a paired comparison, with
a **moving-block bootstrap** interval on the mean gain.

The block matters. Consecutive hours of a 24-hour rolling mean share almost
all their observations, so a run of good hours is one event rather than
twenty-four independent successes. An interval that resamples single hours
treats them as independent and comes out narrower than the data earns.
Resampling 24-hour blocks keeps the dependence where it belongs.

The win rate is printed alongside, because the two answer different
questions: the interval says whether the average gain is real, the win rate
says how often you would actually prefer the model. A model can win on the
mean while losing most hours, if its wins are larger.

## Known gaps

- The raw-concentration results at 6 and 12 hours were produced on the earlier
  30-day dataset and have not been regenerated.
- **SARIMA's 24-hour gain is unresolved, and the holdout cannot settle it.**
  19 blocks is too few. Settling it needs more data, which on these endpoints
  means waiting rather than re-querying, or a second holdout period once the
  record is long enough to afford one.
- The SARIMA order is chosen from three candidates. A wider search, or a
  seasonal term, might do better; `--seasonal` exists and has not been run.
- The dashboard's figure code could not be executed where it was written
  (no plotly available), so `tests/test_dashboard_charts.py` skips there and
  runs on any machine that has it. Its data layer is fully tested either way.
- CPCB's own calculator has a transcription error in the O3 top band that
  would make the sub-index jump 65 points across one unit of ozone. This
  project uses the consistent form instead; see `docs/aqi-references.md`.

## Licence

See `LICENSE`.
