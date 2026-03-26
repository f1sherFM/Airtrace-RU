"""Export helpers for SSR download routes."""

from __future__ import annotations

import csv
import io
import json
from typing import Any


def prepare_export_data(time_series_data: list[dict[str, Any]], city_name: str) -> list[dict[str, Any]]:
    export_data: list[dict[str, Any]] = []
    for point in time_series_data:
        export_data.append(
            {
                "timestamp": point["timestamp"],
                "city": city_name,
                "latitude": point["location"]["latitude"],
                "longitude": point["location"]["longitude"],
                "aqi_value": point["aqi"]["value"],
                "aqi_category": point["aqi"]["category"],
                "pm2_5": point["pollutants"]["pm2_5"],
                "pm10": point["pollutants"]["pm10"],
                "no2": point["pollutants"]["no2"],
                "so2": point["pollutants"]["so2"],
                "o3": point["pollutants"]["o3"],
                "nmu_risk": point["nmu_risk"],
            }
        )
    return export_data


def create_csv_export(data: list[dict[str, Any]]) -> str:
    if not data:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()


def create_json_export(data: list[dict[str, Any]]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)

