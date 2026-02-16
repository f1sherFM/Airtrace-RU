"""
Pydantic модели для валидации данных AirTrace RU Backend

Определяет структуры данных для запросов, ответов и внутренних моделей
с валидацией согласно требованиям API.
"""

from pydantic import BaseModel, Field, ConfigDict, field_serializer
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class CoordinatesRequest(BaseModel):
    """Модель для валидации координат в запросах"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lat": 55.7558,
                "lon": 37.6176
            }
        }
    )
    
    lat: float = Field(
        ..., 
        ge=-90, 
        le=90, 
        description="Широта в десятичных градусах"
    )
    lon: float = Field(
        ..., 
        ge=-180, 
        le=180, 
        description="Долгота в десятичных градусах"
    )


class PollutantData(BaseModel):
    """Данные о загрязнителях воздуха"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pm2_5": 25.4,
                "pm10": 45.2,
                "no2": 35.1,
                "so2": 12.3,
                "o3": 85.7
            }
        }
    )
    
    pm2_5: Optional[float] = Field(
        None, 
        ge=0, 
        description="PM2.5 концентрация в мкг/м³"
    )
    pm10: Optional[float] = Field(
        None, 
        ge=0, 
        description="PM10 концентрация в мкг/м³"
    )
    no2: Optional[float] = Field(
        None, 
        ge=0, 
        description="NO2 концентрация в мкг/м³"
    )
    so2: Optional[float] = Field(
        None, 
        ge=0, 
        description="SO2 концентрация в мкг/м³"
    )
    o3: Optional[float] = Field(
        None, 
        ge=0, 
        description="O3 концентрация в мкг/м³"
    )


class AQIInfo(BaseModel):
    """Информация об индексе качества воздуха"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "value": 85,
                "category": "Умеренное",
                "color": "#FFFF00",
                "description": "Качество воздуха приемлемо для большинства людей"
            }
        }
    )
    
    value: int = Field(
        ..., 
        ge=0, 
        le=500, 
        description="Значение AQI индекса"
    )
    category: str = Field(
        ..., 
        description="Категория качества воздуха на русском языке"
    )
    color: str = Field(
        ..., 
        pattern=r"^#[0-9A-Fa-f]{6}$",
        description="Цветовой код для отображения"
    )
    description: str = Field(
        ..., 
        description="Описание качества воздуха на русском языке"
    )


class LocationInfo(BaseModel):
    """Информация о местоположении"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "latitude": 55.7558,
                "longitude": 37.6176
            }
        }
    )
    
    latitude: float = Field(..., description="Широта")
    longitude: float = Field(..., description="Долгота")


def get_utc_now() -> datetime:
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


# WeatherAPI data models
class TemperatureData(BaseModel):
    """Temperature data from WeatherAPI"""
    celsius: float = Field(..., description="Температура в градусах Цельсия")
    fahrenheit: float = Field(..., description="Температура в градусах Фаренгейта")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Временная метка данных")
    source: str = Field(default="weatherapi", description="Источник данных")
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime) -> str:
        return dt.isoformat()


class WindData(BaseModel):
    """Wind data from WeatherAPI"""
    speed_kmh: float = Field(..., description="Скорость ветра в км/ч")
    speed_mph: float = Field(..., description="Скорость ветра в милях/ч")
    direction_degrees: int = Field(..., description="Направление ветра в градусах")
    direction_compass: str = Field(..., description="Направление ветра по компасу")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Временная метка данных")
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime) -> str:
        return dt.isoformat()


class PressureData(BaseModel):
    """Atmospheric pressure data from WeatherAPI"""
    pressure_mb: float = Field(..., description="Атмосферное давление в миллибарах")
    pressure_in: float = Field(..., description="Атмосферное давление в дюймах ртутного столба")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Временная метка данных")
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime) -> str:
        return dt.isoformat()


class WeatherInfo(BaseModel):
    """Combined weather information from WeatherAPI"""
    temperature: Optional[TemperatureData] = Field(None, description="Данные о температуре")
    wind: Optional[WindData] = Field(None, description="Данные о ветре")
    pressure: Optional[PressureData] = Field(None, description="Данные об атмосферном давлении")
    location_name: Optional[str] = Field(None, description="Название местоположения")


class ResponseMetadata(BaseModel):
    """Единая metadata-модель происхождения и качества данных"""
    data_source: str = Field(default="live", description="Источник данных: live/forecast/historical/fallback")
    freshness: str = Field(default="fresh", description="Свежесть данных: fresh/stale/expired")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Уверенность в данных [0..1]")
    confidence_explanation: Optional[str] = Field(
        default="Calculated from source availability and fallback usage",
        description="Краткое объяснение расчета confidence",
    )
    fallback_used: bool = Field(default=False, description="Использовался ли fallback-путь")
    cache_age_seconds: Optional[int] = Field(
        default=None,
        ge=0,
        description="Возраст данных в кэше в секундах (если применимо)",
    )


class AirQualityData(BaseModel):
    """Основная модель данных о качестве воздуха с интеграцией WeatherAPI"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-15T12:00:00Z",
                "location": {
                    "latitude": 55.7558,
                    "longitude": 37.6176
                },
                "aqi": {
                    "value": 85,
                    "category": "Умеренное",
                    "color": "#FFFF00",
                    "description": "Качество воздуха приемлемо для большинства людей"
                },
                "pollutants": {
                    "pm2_5": 25.4,
                    "pm10": 45.2,
                    "no2": 35.1,
                    "so2": 12.3
                },
                "weather": {
                    "temperature": {
                        "celsius": 15.5,
                        "fahrenheit": 59.9,
                        "timestamp": "2024-01-15T12:00:00Z",
                        "source": "weatherapi"
                    },
                    "wind": {
                        "speed_kmh": 12.5,
                        "speed_mph": 7.8,
                        "direction_degrees": 180,
                        "direction_compass": "S",
                        "timestamp": "2024-01-15T12:00:00Z"
                    },
                    "pressure": {
                        "pressure_mb": 1013.25,
                        "pressure_in": 29.92,
                        "timestamp": "2024-01-15T12:00:00Z"
                    },
                    "location_name": "Moscow"
                },
                "recommendations": "Чувствительные люди должны ограничить длительное пребывание на улице",
                "nmu_risk": "low",
                "health_warnings": []
            }
        }
    )
    
    timestamp: datetime = Field(
        default_factory=get_utc_now,
        description="Временная метка данных в UTC"
    )
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime) -> str:
        return dt.isoformat()
    
    location: LocationInfo = Field(
        ..., 
        description="Координаты местоположения"
    )
    aqi: AQIInfo = Field(
        ..., 
        description="Информация об индексе качества воздуха"
    )
    pollutants: PollutantData = Field(
        ..., 
        description="Данные о концентрации загрязнителей"
    )
    weather: Optional[WeatherInfo] = Field(
        None,
        description="Дополнительные данные о погоде от WeatherAPI"
    )
    recommendations: str = Field(
        ..., 
        description="Рекомендации на русском языке"
    )
    nmu_risk: Optional[str] = Field(
        None, 
        description="Уровень риска НМУ (low/medium/high)"
    )
    health_warnings: List[str] = Field(
        default_factory=list,
        description="Предупреждения о здоровье на русском языке"
    )
    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="Унифицированная provenance metadata для всех ответов API",
    )


class HealthCheckResponse(BaseModel):
    """Ответ health check эндпоинта"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T12:00:00Z",
                "services": {
                    "api": "healthy",
                    "external_api": "healthy",
                    "cache": "healthy"
                }
            }
        }
    )
    
    status: str = Field(
        ..., 
        description="Общий статус сервиса (healthy/degraded/unhealthy)"
    )
    timestamp: datetime = Field(
        default_factory=get_utc_now,
        description="Время проверки"
    )
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime) -> str:
        return dt.isoformat()
    services: Dict[str, Any] = Field(
        ...,
        description="Статус компонентов системы в нормализованном формате"
    )


class ErrorResponse(BaseModel):
    """Стандартная модель для ошибок API"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Некорректные координаты",
                "details": {
                    "latitude": "Значение должно быть между -90 и 90"
                },
                "timestamp": "2024-01-15T12:00:00Z"
            }
        }
    )
    
    code: str = Field(..., description="Код ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Дополнительные детали ошибки"
    )
    timestamp: datetime = Field(
        default_factory=get_utc_now,
        description="Время возникновения ошибки"
    )
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime) -> str:
        return dt.isoformat()


# Дополнительные модели для внутреннего использования

class OpenMeteoResponse(BaseModel):
    """Модель ответа от Open-Meteo API"""
    model_config = ConfigDict(extra="allow")  # Разрешить дополнительные поля
    
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    hourly: Dict[str, List[Any]]
    hourly_units: Dict[str, str]


class CacheEntry(BaseModel):
    """Модель записи в кэше"""
    data: Dict[str, Any]
    timestamp: datetime
    ttl_seconds: int = 900  # 15 минут по умолчанию

    def is_expired(self) -> bool:
        """Проверка истечения срока действия записи"""
        from datetime import timedelta
        return datetime.now(timezone.utc) > self.timestamp + timedelta(seconds=self.ttl_seconds)


class DataSource(str, Enum):
    """Источник данных для исторической платформы"""
    LIVE = "live"
    FORECAST = "forecast"
    HISTORICAL = "historical"
    FALLBACK = "fallback"


class HistoryFreshness(str, Enum):
    """Уровень свежести исторических данных"""
    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"


class HistoricalSnapshotRecord(BaseModel):
    """Каноническая запись часового среза качества воздуха"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "snapshot_hour_utc": "2026-02-15T18:00:00Z",
                "city_code": "moscow",
                "latitude": 55.7558,
                "longitude": 37.6176,
                "aqi": 85,
                "pollutants": {"pm2_5": 25.4, "pm10": 45.2, "no2": 35.1, "so2": 12.3, "o3": 85.7},
                "data_source": "live",
                "freshness": "fresh",
                "confidence": 0.92,
                "ingested_at": "2026-02-15T18:03:21Z"
            }
        }
    )

    snapshot_hour_utc: datetime = Field(..., description="Начало часа в UTC")
    city_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_-]+$",
        description="Канонический код города (для преднастроенных городов)"
    )
    latitude: float = Field(..., ge=-90, le=90, description="Широта")
    longitude: float = Field(..., ge=-180, le=180, description="Долгота")
    aqi: int = Field(..., ge=0, le=500, description="Индекс качества воздуха AQI")
    pollutants: PollutantData = Field(..., description="Концентрации загрязнителей")
    data_source: DataSource = Field(..., description="Источник данных")
    freshness: HistoryFreshness = Field(..., description="Свежесть записи")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность в данных [0..1]")
    ingested_at: datetime = Field(default_factory=get_utc_now, description="Время записи в хранилище")
    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="Унифицированная provenance metadata для history API",
    )
    anomaly_detected: bool = Field(
        default=False,
        description="Флаг аномалии относительно локального baseline",
    )
    anomaly_type: Optional[str] = Field(
        default=None,
        description="Тип аномалии: spike/dropout",
    )
    anomaly_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Нормализованная сила аномалии (чем больше, тем сильнее отклонение)",
    )
    anomaly_baseline_aqi: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Baseline AQI, относительно которого детектирована аномалия",
    )

    @field_serializer("snapshot_hour_utc", "ingested_at")
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()


class DailyAggregateRecord(BaseModel):
    """Агрегированная дневная запись по историческим данным"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "day_utc": "2026-02-15T00:00:00Z",
                "city_code": "moscow",
                "latitude": 55.7558,
                "longitude": 37.6176,
                "aqi_min": 52,
                "aqi_max": 112,
                "aqi_avg": 76.4,
                "sample_count": 24,
                "dominant_source": "live",
                "avg_confidence": 0.88,
                "calculated_at": "2026-02-16T00:05:00Z"
            }
        }
    )

    day_utc: datetime = Field(..., description="Начало суток в UTC")
    city_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_-]+$",
        description="Канонический код города"
    )
    latitude: float = Field(..., ge=-90, le=90, description="Широта")
    longitude: float = Field(..., ge=-180, le=180, description="Долгота")
    aqi_min: int = Field(..., ge=0, le=500, description="Минимальный AQI за сутки")
    aqi_max: int = Field(..., ge=0, le=500, description="Максимальный AQI за сутки")
    aqi_avg: float = Field(..., ge=0, le=500, description="Средний AQI за сутки")
    sample_count: int = Field(..., ge=1, le=24, description="Количество часовых срезов в агрегации")
    dominant_source: DataSource = Field(..., description="Преобладающий источник данных за сутки")
    avg_confidence: float = Field(..., ge=0.0, le=1.0, description="Средняя уверенность [0..1]")
    calculated_at: datetime = Field(default_factory=get_utc_now, description="Время расчета агрегата")

    @field_serializer("day_utc", "calculated_at")
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()


class HistoryRange(str, Enum):
    """Разрешенные пресеты диапазона для history API"""
    LAST_24H = "24h"
    LAST_7D = "7d"
    LAST_30D = "30d"


class HistoryQueryResponse(BaseModel):
    """Ответ history API с пагинацией"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "range": "24h",
                "page": 1,
                "page_size": 50,
                "total": 3,
                "items": [],
            }
        }
    )

    range: HistoryRange = Field(..., description="Примененный диапазон выборки")
    page: int = Field(..., ge=1, description="Номер страницы (с 1)")
    page_size: int = Field(..., ge=1, le=500, description="Размер страницы")
    total: int = Field(..., ge=0, description="Общее число найденных записей")
    items: List[HistoricalSnapshotRecord] = Field(default_factory=list, description="Список исторических записей")


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertRuleCreate(BaseModel):
    """Модель создания правила алерта"""
    name: str = Field(..., min_length=1, max_length=120)
    enabled: bool = Field(default=True)
    aqi_threshold: Optional[int] = Field(default=None, ge=0, le=500)
    nmu_levels: List[str] = Field(default_factory=list, description="Список уровней НМУ для триггера")
    cooldown_minutes: int = Field(default=60, ge=1, le=1440)
    quiet_hours_start: Optional[int] = Field(default=None, ge=0, le=23)
    quiet_hours_end: Optional[int] = Field(default=None, ge=0, le=23)
    channel: str = Field(default="telegram", description="Канал доставки (пока telegram)")
    chat_id: Optional[str] = Field(default=None, min_length=1, max_length=128, description="Подписка канала")


class AlertRuleUpdate(BaseModel):
    """Частичное обновление правила алерта"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    enabled: Optional[bool] = None
    aqi_threshold: Optional[int] = Field(default=None, ge=0, le=500)
    nmu_levels: Optional[List[str]] = None
    cooldown_minutes: Optional[int] = Field(default=None, ge=1, le=1440)
    quiet_hours_start: Optional[int] = Field(default=None, ge=0, le=23)
    quiet_hours_end: Optional[int] = Field(default=None, ge=0, le=23)
    channel: Optional[str] = None
    chat_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class AlertRule(AlertRuleCreate):
    """Полная модель правила алерта"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=get_utc_now)

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime) -> str:
        return dt.isoformat()


class AlertEvent(BaseModel):
    """Событие алерта при срабатывании правила"""
    rule_id: str
    rule_name: str
    triggered_at: datetime = Field(default_factory=get_utc_now)
    severity: AlertSeverity
    reasons: List[str] = Field(default_factory=list)
    suppressed: bool = Field(default=False, description="Был ли алерт подавлен cooldown/quiet-hours")

    @field_serializer("triggered_at")
    def serialize_triggered_at(self, dt: datetime) -> str:
        return dt.isoformat()


class TelegramSendRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4096)


class DeliveryResult(BaseModel):
    channel: str = Field(default="telegram")
    status: str = Field(..., description="sent/failed")
    attempts: int = Field(..., ge=1)
    event_id: Optional[str] = None
    error: Optional[str] = None


class DailyDigestResponse(BaseModel):
    """Daily digest summary for selected location."""
    location_label: str
    period: str = Field(default="24h")
    trend: str = Field(..., description="improving/stable/worsening")
    top_warnings: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    summary_text: str
