CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS stations (
    station_id BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(Point, 4326) NOT NULL
);

CREATE TABLE IF NOT EXISTS measurements (
    station_id BIGINT NOT NULL REFERENCES stations(station_id),
    ts TIMESTAMPTZ NOT NULL,
    pollutant TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (station_id, ts, pollutant, source)
);

CREATE TABLE IF NOT EXISTS weather (
    city TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    temp_c DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    wind_dir DOUBLE PRECISION,
    blh DOUBLE PRECISION,
    PRIMARY KEY (city, ts)
);

CREATE TABLE IF NOT EXISTS daily_aqi (
    station_id BIGINT NOT NULL REFERENCES stations(station_id),
    date DATE NOT NULL,
    aqi INTEGER NOT NULL,
    dominant_pollutant TEXT NOT NULL,
    PRIMARY KEY (station_id, date)
);

CREATE TABLE IF NOT EXISTS forecasts (
    station_id BIGINT NOT NULL REFERENCES stations(station_id),
    horizon_ts TIMESTAMPTZ NOT NULL,
    predicted_aqi DOUBLE PRECISION NOT NULL,
    model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (station_id, horizon_ts, model)
);

CREATE INDEX IF NOT EXISTS measurements_ts_idx ON measurements(ts);
CREATE INDEX IF NOT EXISTS measurements_pollutant_idx ON measurements(pollutant);
CREATE INDEX IF NOT EXISTS stations_geom_idx ON stations USING GIST(geom);
