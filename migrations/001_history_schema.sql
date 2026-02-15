-- AirTrace RU v4.0.0
-- Issue 1.1: Historical storage schema for hourly snapshots and daily aggregates.

BEGIN;

CREATE TABLE IF NOT EXISTS historical_snapshots (
    id BIGSERIAL PRIMARY KEY,
    snapshot_hour_utc TIMESTAMPTZ NOT NULL,
    city_code VARCHAR(64),
    latitude DOUBLE PRECISION NOT NULL CHECK (latitude >= -90 AND latitude <= 90),
    longitude DOUBLE PRECISION NOT NULL CHECK (longitude >= -180 AND longitude <= 180),
    aqi SMALLINT NOT NULL CHECK (aqi >= 0 AND aqi <= 500),
    pm2_5 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    no2 DOUBLE PRECISION,
    so2 DOUBLE PRECISION,
    o3 DOUBLE PRECISION,
    data_source VARCHAR(16) NOT NULL CHECK (data_source IN ('live', 'forecast', 'historical', 'fallback')),
    freshness VARCHAR(16) NOT NULL CHECK (freshness IN ('fresh', 'stale', 'expired')),
    confidence NUMERIC(4,3) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dedupe_key CHAR(64) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_historical_snapshots_dedupe_key
    ON historical_snapshots(dedupe_key);

CREATE INDEX IF NOT EXISTS idx_historical_snapshots_city_hour
    ON historical_snapshots(city_code, snapshot_hour_utc DESC);

CREATE INDEX IF NOT EXISTS idx_historical_snapshots_geo_hour
    ON historical_snapshots(latitude, longitude, snapshot_hour_utc DESC);

CREATE INDEX IF NOT EXISTS idx_historical_snapshots_source_hour
    ON historical_snapshots(data_source, snapshot_hour_utc DESC);

CREATE TABLE IF NOT EXISTS daily_aggregates (
    id BIGSERIAL PRIMARY KEY,
    day_utc DATE NOT NULL,
    city_code VARCHAR(64),
    latitude DOUBLE PRECISION NOT NULL CHECK (latitude >= -90 AND latitude <= 90),
    longitude DOUBLE PRECISION NOT NULL CHECK (longitude >= -180 AND longitude <= 180),
    aqi_min SMALLINT NOT NULL CHECK (aqi_min >= 0 AND aqi_min <= 500),
    aqi_max SMALLINT NOT NULL CHECK (aqi_max >= 0 AND aqi_max <= 500),
    aqi_avg DOUBLE PRECISION NOT NULL CHECK (aqi_avg >= 0 AND aqi_avg <= 500),
    sample_count SMALLINT NOT NULL CHECK (sample_count BETWEEN 1 AND 24),
    pm2_5_avg DOUBLE PRECISION,
    pm10_avg DOUBLE PRECISION,
    no2_avg DOUBLE PRECISION,
    so2_avg DOUBLE PRECISION,
    o3_avg DOUBLE PRECISION,
    dominant_source VARCHAR(16) NOT NULL CHECK (dominant_source IN ('live', 'forecast', 'historical', 'fallback')),
    avg_confidence NUMERIC(4,3) NOT NULL CHECK (avg_confidence >= 0 AND avg_confidence <= 1),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(day_utc, city_code, latitude, longitude)
);

CREATE INDEX IF NOT EXISTS idx_daily_aggregates_city_day
    ON daily_aggregates(city_code, day_utc DESC);

CREATE INDEX IF NOT EXISTS idx_daily_aggregates_geo_day
    ON daily_aggregates(latitude, longitude, day_utc DESC);

COMMIT;
