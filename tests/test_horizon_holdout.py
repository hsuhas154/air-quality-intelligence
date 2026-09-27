import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "holdout", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_horizon_holdout.py"
)
holdout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(holdout)


def _evaluable(hours=200, cities=("Delhi", "Bengaluru")):
    rows = []
    for city in cities:
        index = pd.date_range("2026-07-01", periods=hours, freq="h", tz="UTC")
        for i, ts in enumerate(index):
            rows.append(
                {
                    "city": city, "hour": ts,
                    "aqi": 100.0 + i, "target_aqi_24h": 100.0 + i + 24,
                }
            )
    return pd.DataFrame(rows)


def test_holdout_is_the_latest_share_of_each_city():
    data = _evaluable()
    development, hold, cutoffs, _ = holdout.split_with_embargo(data, 24, 0.20)

    for city, cutoff in cutoffs.items():
        assert development[development["city"] == city]["hour"].max() < cutoff
        assert hold[hold["city"] == city]["hour"].min() >= cutoff


def test_no_development_target_reaches_the_holdout():
    data = _evaluable()
    development, _, cutoffs, embargoed = holdout.split_with_embargo(data, 24, 0.20)

    assert embargoed > 0

    for city, cutoff in cutoffs.items():
        latest = development[development["city"] == city]["hour"].max()
        assert latest + pd.Timedelta(hours=24) < cutoff


def test_the_embargo_scales_with_the_horizon():
    data = _evaluable()

    _, _, _, short = holdout.split_with_embargo(data, 1, 0.20)
    _, _, _, long = holdout.split_with_embargo(data, 24, 0.20)

    assert long > short


def test_holdout_size_matches_the_requested_fraction():
    data = _evaluable(hours=200)
    _, hold, _, _ = holdout.split_with_embargo(data, 24, 0.20)

    for _, group in hold.groupby("city"):
        assert group.shape[0] == pytest.approx(200 * 0.20, abs=1)
