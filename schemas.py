"""
Pydantic модели для валидации данных AirTrace RU Backend

Определяет структуры данных для запросов, ответов и внутренних моделей
с валидацией согласно требованиям API.
"""

from pydantic import BaseModel, Field, ConfigDict, field_serializer
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum


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


class AirQualityData(BaseModel):
    """Основная модель данных о качестве воздуха"""
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
        description="Общий статус сервиса (healthy/unhealthy)"
    )
    timestamp: datetime = Field(
        default_factory=get_utc_now,
        description="Время проверки"
    )
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime) -> str:
        return dt.isoformat()
    services: Dict[str, str] = Field(
        ..., 
        description="Статус отдельных компонентов системы"
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