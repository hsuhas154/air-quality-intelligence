"""Figures for the dashboard.

Colours come from a validated categorical palette: the first three slots of
the reference set, which clear the colourblind-separation and
normal-vision floors on all pairs in both light and dark modes. Dark is a
separate set of steps chosen for the dark surface, not an automatic flip of
the light one.

Two rules shape the forms here. Identity never rests on colour alone: every
figure with more than one series carries a legend, and the dashboard prints
the same numbers as a table underneath. And there is never a second y-axis:
where two measures share a figure they share a scale, and where they cannot,
they get separate figures.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e6e5e1",
    "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    "good": "#0ca30c",
    "critical": "#d03b3b",
    "neutral": "#52514e",
}

DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "muted": "#c3c2b7",
    "grid": "#33322f",
    "series": ("#3987e5", "#d95926", "#199e70"),
    "good": "#0ca30c",
    "critical": "#d03b3b",
    "neutral": "#c3c2b7",
}


def theme(dark: bool) -> dict:
    return DARK if dark else LIGHT


def _base(figure: go.Figure, palette: dict, height: int = 380) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": palette["text"], "size": 13},
        hoverlabel={"font_size": 13},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=palette["grid"],
        tickfont={"color": palette["muted"]},
        title_font={"color": palette["muted"], "size": 12},
    )
    figure.update_yaxes(
        gridcolor=palette["grid"],
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont={"color": palette["muted"]},
        title_font={"color": palette["muted"], "size": 12},
    )

    return figure


def window_overlap_figure(table: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Autocorrelation against window overlap, on one shared 0 to 1 scale.

    Both series are fractions, so they belong on the same axis. Giving them
    separate scales would let any pair of curves be made to look aligned,
    which is exactly the claim the figure is supposed to test.
    """

    palette = theme(dark)
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=table["horizon"],
            y=table["window_overlap"],
            name="Window overlap",
            mode="lines",
            line={"width": 2, "color": palette["series"][1], "dash": "dot"},
            hovertemplate="%{x}h ahead<br>windows share %{y:.0%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table["horizon"],
            y=table["autocorrelation"],
            name="AQI autocorrelation",
            mode="lines",
            line={"width": 2, "color": palette["series"][0]},
            hovertemplate="%{x}h ahead<br>correlation %{y:.3f}<extra></extra>",
        )
    )

    figure.add_vline(
        x=24,
        line={"width": 1, "color": palette["muted"], "dash": "dash"},
        annotation_text="24h: windows share nothing",
        annotation_position="top right",
        annotation_font={"color": palette["muted"], "size": 11},
    )

    figure.update_layout(hovermode="x unified")
    figure.update_xaxes(title="Forecast horizon, hours")
    figure.update_yaxes(title="Fraction", range=[-0.05, 1.05], tickformat=".0%")

    return _base(figure, palette, height=400)


def comparison_figure(frame: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Mean gain against persistence, with its bootstrap interval.

    A dot and interval rather than bars, because the quantity is an effect
    size with uncertainty and a bar chart would draw the eye to a length
    that the interval says is not established. Model identity sits on the
    axis, so colour is free to carry the verdict, and the verdict is written
    out in the hover and in the table beneath rather than left to hue.
    """

    palette = theme(dark)
    figure = go.Figure()

    if frame.empty:
        return _base(figure, palette, height=240)

    ordered = frame.iloc[::-1].reset_index(drop=True)

    for row in ordered.itertuples():
        if not row.resolved:
            colour = palette["neutral"]
        elif row.gain > 0:
            colour = palette["good"]
        else:
            colour = palette["critical"]

        figure.add_trace(
            go.Scatter(
                x=[row.ci_low, row.ci_high],
                y=[row.model, row.model],
                mode="lines",
                line={"width": 2, "color": colour},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[row.gain],
                y=[row.model],
                mode="markers",
                marker={
                    "size": 11,
                    "color": colour,
                    "line": {"width": 2, "color": palette["surface"]},
                },
                showlegend=False,
                customdata=[[row.mae, row.baseline_mae, row.win_rate, row.verdict]],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "gain %{x:+.3f} MAE points<br>"
                    "MAE %{customdata[0]:.3f} vs %{customdata[1]:.3f}<br>"
                    "wins %{customdata[2]:.1%} of hours<br>"
                    "%{customdata[3]}<extra></extra>"
                ),
            )
        )

    figure.add_vline(
        x=0,
        line={"width": 1, "color": palette["muted"], "dash": "dash"},
        annotation_text="persistence",
        annotation_position="top",
        annotation_font={"color": palette["muted"], "size": 11},
    )

    figure.update_xaxes(title="Mean gain in MAE against persistence, AQI points")
    figure.update_yaxes(title=None)

    return _base(figure, palette, height=60 + 52 * len(ordered))


def history_figure(daily: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Daily mean AQI per city."""

    palette = theme(dark)
    figure = go.Figure()

    for index, (city, group) in enumerate(daily.groupby("city")):
        figure.add_trace(
            go.Scatter(
                x=group["day"],
                y=group["aqi"],
                name=str(city),
                mode="lines",
                line={
                    "width": 2,
                    "color": palette["series"][index % len(palette["series"])],
                },
                hovertemplate="%{x|%d %b}<br>AQI %{y:.0f}<extra>" + str(city) + "</extra>",
            )
        )

    figure.update_layout(hovermode="x unified")
    figure.update_xaxes(title=None)
    figure.update_yaxes(title="Daily mean AQI")

    return _base(figure, palette, height=360)
