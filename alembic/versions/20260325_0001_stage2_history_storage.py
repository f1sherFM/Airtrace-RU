"""Create Stage 2 history storage tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260325_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "locations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("city_code", sa.String(length=64), nullable=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("coordinate_key", sa.String(length=64), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("latitude >= -90 AND latitude <= 90", name=op.f("ck_locations_latitude_range")),
        sa.CheckConstraint("longitude >= -180 AND longitude <= 180", name=op.f("ck_locations_longitude_range")),
        sa.UniqueConstraint("city_code", name=op.f("uq_locations_city_code")),
        sa.UniqueConstraint("coordinate_key", name=op.f("uq_locations_coordinate_key")),
    )

    op.create_table(
        "air_quality_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("location_id", sa.Integer(), nullable=False),
        sa.Column("snapshot_hour_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_timestamp_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("aqi", sa.SmallInteger(), nullable=False),
        sa.Column("pm2_5", sa.Float(), nullable=True),
        sa.Column("pm10", sa.Float(), nullable=True),
        sa.Column("no2", sa.Float(), nullable=True),
        sa.Column("so2", sa.Float(), nullable=True),
        sa.Column("o3", sa.Float(), nullable=True),
        sa.Column("data_source", sa.String(length=16), nullable=False),
        sa.Column("freshness", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False),
        sa.Column("dedupe_key", sa.String(length=64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("aqi >= 0 AND aqi <= 500", name=op.f("ck_air_quality_snapshots_aqi_range")),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name=op.f("ck_air_quality_snapshots_confidence_range")),
        sa.ForeignKeyConstraint(["location_id"], ["locations.id"], name=op.f("fk_air_quality_snapshots_location_id_locations"), ondelete="CASCADE"),
        sa.UniqueConstraint("dedupe_key", name=op.f("uq_air_quality_snapshots_dedupe_key")),
    )
    op.create_index(op.f("ix_air_quality_snapshots_location_id"), "air_quality_snapshots", ["location_id"], unique=False)
    op.create_index(op.f("ix_air_quality_snapshots_snapshot_hour_utc"), "air_quality_snapshots", ["snapshot_hour_utc"], unique=False)
    op.create_index(op.f("ix_air_quality_snapshots_data_source"), "air_quality_snapshots", ["data_source"], unique=False)
    op.create_index("ix_air_quality_snapshots_location_hour", "air_quality_snapshots", ["location_id", "snapshot_hour_utc"], unique=False)

    op.create_table(
        "data_provenance",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("snapshot_id", sa.Integer(), nullable=False),
        sa.Column("data_source", sa.String(length=16), nullable=False),
        sa.Column("freshness", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False),
        sa.Column("confidence_explanation", sa.Text(), nullable=True),
        sa.Column("fallback_used", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("cache_age_seconds", sa.Integer(), nullable=True),
        sa.Column("source_chain", sa.JSON(), nullable=True),
        sa.Column("raw_metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name=op.f("ck_data_provenance_provenance_confidence_range")),
        sa.ForeignKeyConstraint(["snapshot_id"], ["air_quality_snapshots.id"], name=op.f("fk_data_provenance_snapshot_id_air_quality_snapshots"), ondelete="CASCADE"),
        sa.UniqueConstraint("snapshot_id", name=op.f("uq_data_provenance_snapshot_id")),
    )


def downgrade() -> None:
    op.drop_table("data_provenance")
    op.drop_index("ix_air_quality_snapshots_location_hour", table_name="air_quality_snapshots")
    op.drop_index(op.f("ix_air_quality_snapshots_data_source"), table_name="air_quality_snapshots")
    op.drop_index(op.f("ix_air_quality_snapshots_snapshot_hour_utc"), table_name="air_quality_snapshots")
    op.drop_index(op.f("ix_air_quality_snapshots_location_id"), table_name="air_quality_snapshots")
    op.drop_table("air_quality_snapshots")
    op.drop_table("locations")
