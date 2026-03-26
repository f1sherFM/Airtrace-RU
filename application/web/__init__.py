"""Web-facing application services and page builders for SSR."""

from .exports import create_csv_export, create_json_export, prepare_export_data
from .formatters import (
    format_time,
    get_action_plan,
    get_aqi_class,
    get_nmu_config,
    normalize_api_status,
    translate_api_status,
    translate_freshness,
    translate_source,
    translate_trend,
)
from .pages import (
    build_alerts_page_context,
    build_city_page_context,
    build_compare_page_context,
    build_history_page_context,
    build_index_context,
    build_trends_page_context,
)
from .service import WebAppService

__all__ = [
    "WebAppService",
    "build_alerts_page_context",
    "build_city_page_context",
    "build_compare_page_context",
    "build_history_page_context",
    "build_index_context",
    "build_trends_page_context",
    "create_csv_export",
    "create_json_export",
    "format_time",
    "get_action_plan",
    "get_aqi_class",
    "get_nmu_config",
    "normalize_api_status",
    "prepare_export_data",
    "translate_api_status",
    "translate_freshness",
    "translate_source",
    "translate_trend",
]
