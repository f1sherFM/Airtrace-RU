from pathlib import Path

import pytest

from core.settings import (
    CitiesConfigError,
    clear_cities_config_cache,
    get_cities_mapping,
    load_cities_config,
    parse_cities_config,
)


@pytest.fixture(autouse=True)
def _clear_city_cache():
    clear_cities_config_cache()
    yield
    clear_cities_config_cache()


def test_stage1_city_config_loads_valid_centralized_mapping():
    config = load_cities_config()

    assert "moscow" in config.cities
    assert config.cities["moscow"].name == "Москва"
    assert config.cities["surgut"].lat == pytest.approx(61.254)


def test_stage1_city_config_exposes_legacy_mapping_shape():
    cities = get_cities_mapping()

    assert cities["moscow"]["name"] == "Москва"
    assert cities["moscow"]["lat"] == pytest.approx(55.7558)
    assert cities["moscow"]["lon"] == pytest.approx(37.6176)


def test_stage1_city_config_rejects_invalid_coordinates(tmp_path: Path):
    config_path = tmp_path / "cities.yaml"
    config_path.write_text(
        '{"cities": {"bad_city": {"name": "Bad", "lat": 12.0, "lon": 200.0}}}',
        encoding="utf-8",
    )

    with pytest.raises(CitiesConfigError):
        parse_cities_config(config_path)


def test_stage1_city_config_rejects_missing_required_fields(tmp_path: Path):
    config_path = tmp_path / "cities.yaml"
    config_path.write_text(
        '{"cities": {"bad_city": {"lat": 55.0, "lon": 37.0}}}',
        encoding="utf-8",
    )

    with pytest.raises(CitiesConfigError):
        parse_cities_config(config_path)


def test_stage1_city_config_rejects_duplicate_city_keys(tmp_path: Path):
    config_path = tmp_path / "cities.yaml"
    config_path.write_text(
        '{"cities": {"moscow": {"name": "Москва", "lat": 55.7, "lon": 37.6},'
        ' "moscow": {"name": "Дубль", "lat": 55.8, "lon": 37.7}}}',
        encoding="utf-8",
    )

    with pytest.raises(CitiesConfigError):
        parse_cities_config(config_path)
