from datetime import UTC, datetime, timedelta

import httpx

from air_quality_intelligence.ingestion.openaq import (
    MAX_PAGE_SIZE,
    fetch_sensor_measurements,
)


def _observations(count):
    base = datetime(2026, 7, 1, tzinfo=UTC)
    return [
        {
            "period": {"datetimeFrom": {"utc": (base + timedelta(minutes=15 * i)).isoformat()}},
            "parameter": {"name": "pm25", "units": "µg/m³"},
            "value": float(i),
        }
        for i in range(count)
    ]


def _install(monkeypatch, pages):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params=None):
            return FakeResponse({"results": pages(params or {})})

    monkeypatch.setattr(httpx, "Client", FakeClient)


def test_a_window_longer_than_one_page_is_fully_retrieved(monkeypatch):
    total = MAX_PAGE_SIZE * 2 + 137
    data = _observations(total)

    def pages(params):
        limit, page = params["limit"], params["page"]
        return data[(page - 1) * limit: page * limit]

    _install(monkeypatch, pages)

    df = fetch_sensor_measurements(1, datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC))

    assert len(df) == total
    assert df["ts"].nunique() == total


def test_overlapping_pages_are_not_double_counted(monkeypatch):
    data = _observations(MAX_PAGE_SIZE + 500)

    def pages(params):
        limit, page = params["limit"], params["page"]
        start = max(0, (page - 1) * limit - 50)
        return data[start: start + limit]

    _install(monkeypatch, pages)

    df = fetch_sensor_measurements(1, datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC))

    assert len(df) == df["ts"].nunique()


def test_a_short_page_ends_pagination(monkeypatch):
    data = _observations(17)
    calls = []

    def pages(params):
        calls.append(params["page"])
        return data

    _install(monkeypatch, pages)

    df = fetch_sensor_measurements(1, datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 7, 2, tzinfo=UTC))

    assert len(df) == 17
    assert calls == [1]
