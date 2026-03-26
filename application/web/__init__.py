"""Web-facing application services and page builders for SSR."""

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
]
