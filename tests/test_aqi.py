from air_quality_intelligence.transform.aqi import calculate_aqi, sub_index


def test_pm25_breakpoint_interpolation():
    assert sub_index("pm2.5", 30) == 50
    assert sub_index("pm2.5", 60) == 100


def test_aqi_returns_dominant_pollutant():
    aqi, dominant = calculate_aqi({"pm2.5": 90, "pm10": 50})
    assert aqi == round(sub_index("pm2.5", 90))
    assert dominant == "pm2.5"


def test_unsupported_pollutant():
    try:
        sub_index("lead", 10)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
