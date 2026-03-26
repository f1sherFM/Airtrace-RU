"""Create Stage 4 alert subscription storage tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260326_0002"
down_revision = "20260325_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alert_subscriptions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("city_code", sa.String(length=64), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("coordinate_key", sa.String(length=64), nullable=True),
        sa.Column("aqi_threshold", sa.SmallInteger(), nullable=True),
        sa.Column("nmu_levels", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("cooldown_minutes", sa.Integer(), nullable=False, server_default=sa.text("60")),
        sa.Column("quiet_hours_start", sa.SmallInteger(), nullable=True),
        sa.Column("quiet_hours_end", sa.SmallInteger(), nullable=True),
        sa.Column("channel", sa.String(length=32), nullable=False, server_default=sa.text("'telegram'")),
        sa.Column("chat_id", sa.String(length=128), nullable=True),
        sa.Column("last_triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_delivery_status", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alert_subscriptions")),
        sa.CheckConstraint(
            "(latitude IS NULL AND longitude IS NULL) OR (latitude IS NOT NULL AND longitude IS NOT NULL)",
            name=op.f("ck_alert_subscriptions_alert_subscription_coordinates_pair"),
        ),
        sa.CheckConstraint(
            "aqi_threshold IS NULL OR (aqi_threshold >= 0 AND aqi_threshold <= 500)",
            name=op.f("ck_alert_subscriptions_alert_subscription_aqi_range"),
        ),
        sa.CheckConstraint(
            "cooldown_minutes >= 1 AND cooldown_minutes <= 1440",
            name=op.f("ck_alert_subscriptions_alert_subscription_cooldown_range"),
        ),
        sa.CheckConstraint(
            "(quiet_hours_start IS NULL AND quiet_hours_end IS NULL) OR (quiet_hours_start IS NOT NULL AND quiet_hours_end IS NOT NULL)",
            name=op.f("ck_alert_subscriptions_alert_subscription_quiet_hours_pair"),
        ),
        sa.CheckConstraint(
            "quiet_hours_start IS NULL OR (quiet_hours_start >= 0 AND quiet_hours_start <= 23)",
            name=op.f("ck_alert_subscriptions_alert_subscription_quiet_hours_start_range"),
        ),
        sa.CheckConstraint(
            "quiet_hours_end IS NULL OR (quiet_hours_end >= 0 AND quiet_hours_end <= 23)",
            name=op.f("ck_alert_subscriptions_alert_subscription_quiet_hours_end_range"),
        ),
    )
    op.create_index(op.f("ix_alert_subscriptions_city_code"), "alert_subscriptions", ["city_code"], unique=False)
    op.create_index(op.f("ix_alert_subscriptions_coordinate_key"), "alert_subscriptions", ["coordinate_key"], unique=False)

    op.create_table(
        "alert_delivery_attempts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("subscription_id", sa.String(length=64), nullable=False),
        sa.Column("event_id", sa.String(length=128), nullable=True),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("provider_response", sa.JSON(), nullable=True),
        sa.Column("dead_lettered", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["subscription_id"],
            ["alert_subscriptions.id"],
            name=op.f("fk_alert_delivery_attempts_subscription_id_alert_subscriptions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alert_delivery_attempts")),
    )
    op.create_index(op.f("ix_alert_delivery_attempts_subscription_id"), "alert_delivery_attempts", ["subscription_id"], unique=False)

    op.create_table(
        "alert_audit_log",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("subscription_id", sa.String(length=64), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["subscription_id"],
            ["alert_subscriptions.id"],
            name=op.f("fk_alert_audit_log_subscription_id_alert_subscriptions"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alert_audit_log")),
    )
    op.create_index(op.f("ix_alert_audit_log_subscription_id"), "alert_audit_log", ["subscription_id"], unique=False)

    op.create_table(
        "alert_idempotency_keys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("scope", sa.String(length=160), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=128), nullable=False),
        sa.Column("response_payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alert_idempotency_keys")),
        sa.UniqueConstraint("scope", "idempotency_key", name=op.f("uq_alert_idempotency_keys_scope")),
    )


def downgrade() -> None:
    op.drop_table("alert_idempotency_keys")
    op.drop_index(op.f("ix_alert_audit_log_subscription_id"), table_name="alert_audit_log")
    op.drop_table("alert_audit_log")
    op.drop_index(op.f("ix_alert_delivery_attempts_subscription_id"), table_name="alert_delivery_attempts")
    op.drop_table("alert_delivery_attempts")
    op.drop_index(op.f("ix_alert_subscriptions_coordinate_key"), table_name="alert_subscriptions")
    op.drop_index(op.f("ix_alert_subscriptions_city_code"), table_name="alert_subscriptions")
    op.drop_table("alert_subscriptions")
