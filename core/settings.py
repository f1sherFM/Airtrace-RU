"""Typed configuration helpers for Stage 1-owned config."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from core.validation import CoordinateValidator

try:
    import yaml
except ImportError:  # pragma: no cover - optional runtime dependency during migration
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITIES_CONFIG_PATH = REPO_ROOT / "config" / "cities.yaml"


class CitiesConfigError(ValueError):
    """Raised when the centralized city configuration is invalid."""


class CityConfig(BaseModel):
    """Validated city entry loaded from config/cities.yaml."""

    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    lat: float
    lon: float

    @model_validator(mode="after")
    def validate_coordinates(self) -> "CityConfig":
        is_valid, error = CoordinateValidator.validate_russian_territory(self.lat, self.lon)
        if not is_valid:
            raise ValueError(error)
        return self

    def to_legacy_payload(self) -> dict[str, Any]:
        return {"name": self.name, "lat": self.lat, "lon": self.lon}


class CitiesConfig(BaseModel):
    """Centralized validated collection of city entries."""

    cities: dict[str, CityConfig]

    @model_validator(mode="after")
    def ensure_keys_match_entries(self) -> "CitiesConfig":
        normalized: dict[str, CityConfig] = {}
        for key, city in self.cities.items():
            normalized[key] = city.model_copy(update={"key": key})
        self.cities = normalized
        return self

    def as_legacy_mapping(self) -> dict[str, dict[str, Any]]:
        return {key: city.to_legacy_payload() for key, city in self.cities.items()}


def _duplicate_preserving_object(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise CitiesConfigError(f"Duplicate key in city config: {key}")
        mapping[key] = value
    return mapping


def _yaml_duplicate_loader():
    if yaml is None:
        return None

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[str, Any]:
        pairs: list[tuple[Any, Any]] = []
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            value = loader.construct_object(value_node, deep=deep)
            pairs.append((key, value))
        return _duplicate_preserving_object(pairs)

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return UniqueKeyLoader


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    loader = _yaml_duplicate_loader()
    if loader is not None:
        loaded = yaml.load(text, Loader=loader)
    else:
        loaded = json.loads(text, object_pairs_hook=_duplicate_preserving_object)
    if not isinstance(loaded, dict):
        raise CitiesConfigError("City config root must be an object with a 'cities' mapping")
    return loaded


def parse_cities_config(path: Path) -> CitiesConfig:
    if not path.exists():
        raise CitiesConfigError(f"City config file not found: {path}")

    raw = _load_mapping(path)
    cities_raw = raw.get("cities")
    if not isinstance(cities_raw, dict):
        raise CitiesConfigError("City config must contain a 'cities' mapping")

    try:
        payload = {
            "cities": {
                key: {"key": key, **value} if isinstance(value, dict) else value
                for key, value in cities_raw.items()
            }
        }
        return CitiesConfig.model_validate(payload)
    except ValidationError as exc:
        raise CitiesConfigError(f"Invalid city config: {exc}") from exc


@lru_cache(maxsize=1)
def load_cities_config(path: Path | None = None) -> CitiesConfig:
    """Load and cache the centralized city configuration."""

    return parse_cities_config(path or DEFAULT_CITIES_CONFIG_PATH)


def get_cities_mapping(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the legacy dict shape used by current SSR templates and handlers."""

    return load_cities_config(path).as_legacy_mapping()


def clear_cities_config_cache() -> None:
    """Reset cached city configuration for tests."""

    load_cities_config.cache_clear()
