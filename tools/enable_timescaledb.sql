CREATE EXTENSION IF NOT EXISTS timescaledb;

SELECT create_hypertable(
    'air_quality_snapshots',
    'snapshot_hour_utc',
    if_not_exists => TRUE,
    migrate_data => TRUE
);
