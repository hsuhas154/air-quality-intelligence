"""Air Quality Intelligence: what the data supports, and what it does not.

Reads the committed run artifacts in outputs/, so a fresh clone can start it
without a database or ninety days of ingestion:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from air_quality_intelligence.dashboard.charts import (
    comparison_figure,
    history_figure,
    window_overlap_figure,
)
from air_quality_intelligence.dashboard.data import (
    comparison_frame,
    comparisons,
    coverage_summary,
    daily_city_aqi,
    load_artifacts,
    window_overlap_table,
)

st.set_page_config(
    page_title="Air Quality Intelligence",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

DARK = str(st.get_option("theme.base") or "light").lower() == "dark"
CHART = {"use_container_width": True, "config": {"displayModeBar": False}}


@st.cache_data(show_spinner=False)
def artifacts():
    return load_artifacts("outputs")


data = artifacts()

st.title("Air Quality Intelligence")
st.caption(
    "24-hour CPCB air quality index forecasting for Delhi and Bengaluru. "
    "Every number here is computed from the committed run outputs by the "
    "same code the command-line evaluation uses."
)

if data.missing:
    st.warning(
        "Missing run outputs: "
        + ", ".join(data.missing)
        + ". Produce them with `python scripts/evaluate_horizon_forecast.py "
        "--sarima` and `python scripts/evaluate_horizon_holdout.py --sarima`."
    )

# ---------------------------------------------------------------------------
# The finding
# ---------------------------------------------------------------------------

st.header("Why a short-horizon forecast proves nothing")

st.markdown(
    "The AQI at any hour is a mean over the previous 24 hours. So the AQI "
    "now and the AQI **h** hours from now are computed from windows that "
    "share `(24-h)/24` of their observations. At one hour ahead, 96 percent "
    "of the answer has already been measured, and any model will look "
    "excellent producing it."
)

if data.features is not None and not data.features.empty:
    overlap = window_overlap_table(data.features, range(1, 49))

    st.plotly_chart(window_overlap_figure(overlap, DARK), **CHART)

    at_24 = overlap.loc[overlap["horizon"] == 24]
    correlation = float(at_24["autocorrelation"].iloc[0]) if len(at_24) else float("nan")

    st.markdown(
        f"**The overlap story is only half of it.** At 24 hours the windows "
        f"share nothing at all, and the AQI is still autocorrelated at "
        f"**{correlation:.2f}**. That residual is real atmospheric "
        "persistence: weather regimes last longer than a day. It is why "
        "persistence is a genuinely strong baseline at 24 hours rather than "
        "a straw man, and why a model that extrapolates the trend can beat "
        "it."
    )

    with st.expander("Table view"):
        st.dataframe(
            overlap.rename(
                columns={
                    "horizon": "Horizon (h)",
                    "window_overlap": "Window overlap",
                    "autocorrelation": "Autocorrelation (pooled)",
                    "n": "Paired observations",
                }
            ).round(3),
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

st.header("Does anything beat persistence?")

st.markdown(
    "Mean gain in MAE against persistence, with a **moving-block bootstrap** "
    "interval. Consecutive hours of a 24-hour rolling mean share most of "
    "their observations, so a run of good hours is one event rather than "
    "twenty-four independent successes; resampling 24-hour blocks keeps that "
    "dependence. An interval crossing the dashed line is **not resolved**, "
    "which is not the same as no difference."
)

panels = []

if data.forecast_predictions is not None and not data.forecast_predictions.empty:
    for horizon in sorted(data.forecast_predictions["horizon"].dropna().unique()):
        rows = data.forecast_predictions[
            data.forecast_predictions["horizon"] == horizon
        ]
        panels.append(
            (
                f"Cross-validation, {int(horizon)}h",
                comparison_frame(comparisons(rows, f"target_aqi_{int(horizon)}h")),
            )
        )

if data.holdout_predictions is not None and not data.holdout_predictions.empty:
    panels.append(
        (
            "Embargoed holdout, 24h",
            comparison_frame(comparisons(data.holdout_predictions, "target_aqi_24h")),
        )
    )

for title, frame in panels:
    if frame.empty:
        continue

    st.subheader(title)
    st.plotly_chart(comparison_figure(frame, DARK), **CHART)
    st.dataframe(
        frame.rename(
            columns={
                "model": "Model",
                "n": "n",
                "mae": "MAE",
                "baseline_mae": "Persistence MAE",
                "gain": "Gain",
                "ci_low": "CI low",
                "ci_high": "CI high",
                "win_rate": "Win rate",
                "verdict": "Verdict",
            }
        ).drop(columns=["resolved"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.3f"),
            "Persistence MAE": st.column_config.NumberColumn(format="%.3f"),
            "Gain": st.column_config.NumberColumn(format="%+.3f"),
            "CI low": st.column_config.NumberColumn(format="%+.3f"),
            "CI high": st.column_config.NumberColumn(format="%+.3f"),
            "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

st.info(
    "A four-parameter SARIMA, which sees nothing but the AQI's own history, "
    "beats a Random Forest given weather, lags, station counts and six "
    "pollutants. The forest is measurably worse than doing nothing at all."
)

# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------

st.header("The data, gaps included")

if data.features is not None and not data.features.empty:
    st.plotly_chart(history_figure(daily_city_aqi(data.features), DARK), **CHART)

    coverage = coverage_summary(data.features)

    st.dataframe(
        coverage.rename(
            columns={
                "city": "City",
                "feature_rows": "Feature rows",
                "hours_with_aqi": "Hours with an AQI",
                "hours_in_span": "Hours in span",
                "coverage": "Coverage",
                "first_hour": "First",
                "last_hour": "Last",
                "days_with_aqi": "Days with an AQI",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Coverage": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

st.markdown(
    """
**Known limitations, stated rather than hidden.**

- An ingestion gap on 27 to 29 August produced no publishable AQI in either
  city. The pipeline handled it correctly by publishing nothing; the hole is
  real and is not filled in.
- Peenya, Bengaluru reports NO2 at 10 to 23 times its own city's median on
  five separate days while every other pollutant there sits at ordinary
  levels. That analyser is not measuring NO2. It is reported by
  `scripts/diagnose_aqi_extremes.py` and **not** excluded from any result.
- The holdout carries 451 observations, which is 19 blocks of 24 hours. At
  that size almost nothing resolves, including climatology's 29 percent
  deficit. That is a limit on the test, not a finding about the models.
- Ninety days is one season. The model cannot learn seasonality and the
  evaluation cannot test across seasons.
"""
)

st.caption(
    "Sources for every AQI constant are in docs/aqi-references.md, traced to "
    "CPCB's own calculator spreadsheet."
)
