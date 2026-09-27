"""Structural checks on the figures.

These skip where plotly is not installed. That is not a formality: the
environment these were written in could not install plotly, so the figure
code was never executed there and `pytest` on a machine that has it is its
first run. The assertions are therefore about the things a reader cannot
check by looking, and about the two mistakes that make a chart lie.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from air_quality_intelligence.dashboard.charts import (  # noqa: E402
    DARK,
    LIGHT,
    comparison_figure,
    history_figure,
    window_overlap_figure,
)
from air_quality_intelligence.dashboard.data import (  # noqa: E402
    comparison_frame,
    comparisons,
    daily_city_aqi,
    window_overlap_table,
)

START = pd.Timestamp("2026-07-01T00:00:00Z")


def features(hours=300):
    rows = []

    for offset, city in enumerate(("Delhi", "Bengaluru")):
        index = pd.date_range(START, periods=hours, freq="h", tz="UTC")

        for i, ts in enumerate(index):
            rows.append(
                {
                    "city": city,
                    "hour": ts,
                    "hourly_aqi": 100 + offset * 30 + 25 * np.sin(i / 30.0),
                }
            )

    return pd.DataFrame(rows)


def predictions(n=300):
    rng = np.random.default_rng(0)
    truth = 100 + rng.normal(0, 20, n)

    return pd.DataFrame(
        {
            "city": "Delhi",
            "hour": pd.date_range(START, periods=n, freq="h", tz="UTC"),
            "target_aqi_24h": truth,
            "persistence": truth + rng.normal(0, 10, n),
            "sarima": truth + rng.normal(0, 4, n),
            "random_forest": truth + rng.normal(0, 16, n),
        }
    )


def all_figures():
    overlap = window_overlap_table(features(), [1, 6, 12, 18, 24, 36, 48])
    frame = comparison_frame(comparisons(predictions(), "target_aqi_24h"))
    daily = daily_city_aqi(features())

    for dark in (False, True):
        yield window_overlap_figure(overlap, dark)
        yield comparison_figure(frame, dark)
        yield history_figure(daily, dark)


# ---------------------------------------------------------------------------
# The two mistakes that make a chart lie
# ---------------------------------------------------------------------------


def test_no_figure_has_a_second_y_axis():
    """Two y-scales let any pair of curves be made to look aligned.

    The overlap figure exists to test whether autocorrelation tracks window
    overlap. On two scales it could not answer that question, only illustrate
    a chosen answer.
    """
    for figure in all_figures():
        for key in figure.layout:
            assert not str(key).startswith("yaxis2"), key


def test_the_overlap_figure_puts_both_series_on_one_fractional_scale():
    figure = window_overlap_figure(
        window_overlap_table(features(), [1, 12, 24, 48])
    )

    assert figure.layout.yaxis.tickformat == ".0%"
    assert len(figure.data) == 2


# ---------------------------------------------------------------------------
# Identity never rests on colour alone
# ---------------------------------------------------------------------------


def test_multi_series_figures_carry_a_legend():
    overlap = window_overlap_figure(window_overlap_table(features(), [1, 12, 24]))
    history = history_figure(daily_city_aqi(features()))

    for figure in (overlap, history):
        named = [trace for trace in figure.data if trace.name]

        assert len(named) >= 2
        assert figure.layout.legend is not None
        assert all(trace.showlegend is not False for trace in named)


def test_the_comparison_figure_names_models_on_the_axis_not_in_a_legend():
    """Colour there carries the verdict, so identity has to come from the
    axis labels."""
    frame = comparison_frame(comparisons(predictions(), "target_aqi_24h"))
    figure = comparison_figure(frame)

    assert all(trace.showlegend is False for trace in figure.data)

    labelled = {value for trace in figure.data for value in trace.y}

    assert set(frame["model"]) <= labelled


def test_the_verdict_is_written_out_not_left_to_colour():
    frame = comparison_frame(comparisons(predictions(), "target_aqi_24h"))
    figure = comparison_figure(frame)

    templates = [t.hovertemplate for t in figure.data if t.hovertemplate]

    assert templates
    assert any("customdata[3]" in template for template in templates)


# ---------------------------------------------------------------------------
# Palette and marks
# ---------------------------------------------------------------------------


def test_every_colour_comes_from_the_validated_palette():
    allowed = set()

    for palette in (LIGHT, DARK):
        allowed.update(palette["series"])
        allowed.update(
            palette[key] for key in ("good", "critical", "neutral", "muted")
        )

    for figure in all_figures():
        for trace in figure.data:
            for colour in (
                getattr(trace.line, "color", None),
                getattr(trace.marker, "color", None),
            ):
                if isinstance(colour, str) and colour.startswith("#"):
                    assert colour in allowed, colour


def test_light_and_dark_use_different_series_steps():
    """Dark is its own set of steps for the dark surface, not a flip."""
    assert LIGHT["series"] != DARK["series"]
    assert LIGHT["surface"] != DARK["surface"]


def test_lines_are_thin_and_markers_are_large_enough_to_hit():
    for figure in all_figures():
        for trace in figure.data:
            width = getattr(trace.line, "width", None)

            if width is not None:
                assert width <= 2, width

            size = getattr(trace.marker, "size", None)

            if size is not None and trace.mode and "markers" in trace.mode:
                assert size >= 8, size


def test_every_data_trace_has_a_hover_layer():
    for figure in all_figures():
        for trace in figure.data:
            if trace.hoverinfo == "skip":
                continue

            assert trace.hovertemplate, trace.name


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_an_empty_comparison_renders_an_empty_figure_rather_than_raising():
    figure = comparison_figure(comparison_frame([]))

    assert len(figure.data) == 0


def test_a_single_city_history_still_renders():
    frame = features()
    frame = frame[frame["city"] == "Delhi"]

    figure = history_figure(daily_city_aqi(frame))

    assert len(figure.data) == 1
