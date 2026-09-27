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
    -- How many pollutant sub-indices stood behind this value. CPCB publishes
    -- an AQI only from three or more, so anything below 3 is a defect.
    pollutant_count SMALLINT,
    -- True when any sub-index came from continuing the top band's line past
    -- the highest published breakpoint, so the value can exceed 500.
    extrapolated BOOLEAN,
    PRIMARY KEY (station_id, date)
);

-- Added after the table shipped; kept here so an existing database picks the
-- columns up when the schema is reapplied.
ALTER TABLE daily_aqi
    ADD COLUMN IF NOT EXISTS pollutant_count SMALLINT;

ALTER TABLE daily_aqi
    ADD COLUMN IF NOT EXISTS extrapolated BOOLEAN;

CREATE TABLE IF NOT EXISTS forecasts (
    station_id BIGINT NOT NULL REFERENCES stations(station_id),
    horizon_ts TIMESTAMPTZ NOT NULL,
    predicted_aqi DOUBLE PRECISION NOT NULL,
    model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (station_id, horizon_ts, model)
);

-- measurements.value is DOUBLE PRECISION, which accepts NaN and Infinity, so
-- NOT NULL does not keep them out and the ingestion stored them. Every
-- consumer discards them, so no AQI was ever wrong because of one. SQL is
-- where they bite: AVG over a group holding a single NaN is NaN for the whole
-- group, and PostgreSQL orders NaN above every real number, so NaN rows take
-- the top of any query ranked by value.
--
-- Added NOT VALID so a table that already holds such rows still loads. To
-- clear them out and enforce this on the whole table:
--
--     DELETE FROM measurements
--     WHERE value = 'NaN'::double precision
--        OR value = 'Infinity'::double precision
--        OR value = '-Infinity'::double precision;
--
--     ALTER TABLE measurements VALIDATE CONSTRAINT measurements_value_finite;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'measurements_value_finite'
    ) THEN
        ALTER TABLE measurements
            ADD CONSTRAINT measurements_value_finite
            CHECK (
                value <> 'NaN'::double precision
                AND value < 'Infinity'::double precision
                AND value > '-Infinity'::double precision
            ) NOT VALID;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS measurements_ts_idx ON measurements(ts);
CREATE INDEX IF NOT EXISTS measurements_pollutant_idx ON measurements(pollutant);
CREATE INDEX IF NOT EXISTS stations_geom_idx ON stations USING GIST(geom);
