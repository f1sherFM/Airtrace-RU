"""
Сервисы для интеграции с внешними API и бизнес-логика AirTrace RU Backend

Содержит асинхронные сервисы для получения данных о качестве воздуха,
кэширования и обработки данных с акцентом на приватность.
"""

import httpx
import asyncio
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import hashlib
import json

from schemas import (
    AirQualityData, 
    PollutantData, 
    AQIInfo, 
    LocationInfo,
    OpenMeteoResponse,
    CacheEntry,
    WeatherInfo,
    TemperatureData,
    WindData,
    PressureData,
    ResponseMetadata,
)
from utils import AQICalculator, check_nmu_risk, is_blacksky_conditions, get_nmu_recommendations
from cache import MultiLevelCacheManager
from connection_pool import get_connection_pool_manager, ServiceType, APIRequest
from weather_api_manager import weather_api_manager
from config import config
from confidence_scoring import ConfidenceInputs, calculate_confidence

logger = logging.getLogger(__name__)


def mask_coordinates(text: str) -> str:
    """
    ✅ FIX #9: Mask coordinates in log messages for privacy
    
    Replaces coordinate values with *** to prevent logging actual locations.
    """
    # Mask latitude/longitude parameters in URLs
    text = re.sub(r'latitude=[\d\.\-]+', 'latitude=***', text)
    text = re.sub(r'longitude=[\d\.\-]+', 'longitude=***', text)
    text = re.sub(r'lat=[\d\.\-]+', 'lat=***', text)
    text = re.sub(r'lon=[\d\.\-]+', 'lon=***', text)
    
    # Mask coordinate tuples like (55.7558, 37.6176)
    text = re.sub(r'\([\d\.\-]+,\s*[\d\.\-]+\)', '(***, ***)', text)
    
    # Mask coordinate objects like {"latitude": 55.7558, "longitude": 37.6176}
    text = re.sub(r'"latitude":\s*[\d\.\-]+', '"latitude": ***', text)
    text = re.sub(r'"longitude":\s*[\d\.\-]+', '"longitude": ***', text)
    
    return text


def log_with_masked_coordinates(level: str, message: str, **kwargs):
    """
    ✅ FIX #9: Log message with masked coordinates
    
    Args:
        level: Log level (debug, info, warning, error)
        message: Log message
        **kwargs: Additional context to log
    """
    masked_message = mask_coordinates(message)
    
    # Mask coordinates in kwargs
    masked_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            masked_kwargs[key] = mask_coordinates(value)
        else:
            masked_kwargs[key] = value
    
    log_func = getattr(logger, level.lower())
    if masked_kwargs:
        log_func(f"{masked_message} - {masked_kwargs}")
    else:
        log_func(masked_message)



class AirQualityService:
    """
    Сервис для получения данных о качестве воздуха из Open-Meteo API
    с поддержкой кэширования, connection pooling и обработки ошибок.
    """
    
    def __init__(self):
        self.base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        # Use connection pool manager instead of direct httpx client
        self.use_connection_pool = config.performance.connection_pooling_enabled
        
        # Fallback client for when connection pooling is disabled
        if not self.use_connection_pool:
            self.client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                # Ignore system proxy env vars for service-to-service API calls.
                trust_env=False
            )
        else:
            self.client = None
        
        self.cache_manager = MultiLevelCacheManager()
        self.aqi_calculator = AQICalculator()
    
    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход"""
        if self.client:
            await self.client.aclose()
    
    async def get_current_air_quality(self, lat: float, lon: float) -> AirQualityData:
        """
        Получение текущих данных о качестве воздуха.
        
        Сначала проверяет кэш, затем обращается к внешнему API при необходимости.
        Обрабатывает сбои кэша корректно, продолжая работу с прямыми API вызовами.
        """
        # Проверка кэша с обработкой сбоев
        cached_data = None
        try:
            cached_data = await self.cache_manager.get(lat, lon)
            if cached_data:
                return AirQualityData(**cached_data)
        except Exception as e:
            logger.warning(f"Cache get operation failed, continuing with direct API call: {e}")
            # Продолжаем выполнение без кэша
        
        # Получение данных из внешнего API
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone",
                "timezone": "Europe/Moscow"
            }
            
            # Логирование внешнего запроса с координатами (через privacy middleware)
            request_url = f"{self.base_url}?latitude={lat}&longitude={lon}&current=pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone&timezone=Europe/Moscow"
            # ✅ FIX #9: Mask coordinates in logs
            masked_url = mask_coordinates(request_url)
            self._log_external_request(masked_url, has_coordinates=True)
            
            # Use connection pool if enabled, otherwise fallback to direct client
            if self.use_connection_pool:
                api_request = APIRequest(
                    method="GET",
                    url=self.base_url,
                    params=params
                )
                api_response = await get_connection_pool_manager().execute_request(
                    ServiceType.OPEN_METEO, 
                    api_request
                )
                api_data = api_response.data
            else:
                response = await self.client.get(self.base_url, params=params)
                response.raise_for_status()
                api_data = response.json()
            
            processed_data = await self._process_current_data(api_data, lat, lon)
            
            # Кэширование результата с обработкой сбоев
            try:
                await self.cache_manager.set(lat, lon, processed_data.model_dump())
            except Exception as e:
                logger.warning(f"Cache set operation failed, continuing without caching: {e}")
                # Продолжаем выполнение без кэширования
            
            return processed_data
            
        except httpx.RequestError as e:
            logger.error(f"Network error fetching air quality data: {e}")
            raise Exception("Ошибка сети при получении данных о качестве воздуха")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching air quality data: {e}")
            raise Exception("Внешний сервис временно недоступен")
        except Exception as e:
            logger.error(f"Unexpected error processing air quality data: {e}")
            raise Exception("Ошибка обработки данных о качестве воздуха")
    
    async def get_forecast_air_quality(self, lat: float, lon: float) -> List[AirQualityData]:
        """
        Получение прогноза качества воздуха на 24 часа.
        
        Возвращает список данных с почасовым прогнозом.
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone",
                "forecast_days": 1,
                "timezone": "Europe/Moscow"
            }
            
            # Логирование внешнего запроса с координатами
            request_url = f"{self.base_url}?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone&forecast_days=1&timezone=Europe/Moscow"
            # ✅ FIX #9: Mask coordinates in logs
            masked_url = mask_coordinates(request_url)
            self._log_external_request(masked_url, has_coordinates=True)
            
            # Use connection pool if enabled, otherwise fallback to direct client
            if self.use_connection_pool:
                api_request = APIRequest(
                    method="GET",
                    url=self.base_url,
                    params=params
                )
                api_response = await get_connection_pool_manager().execute_request(
                    ServiceType.OPEN_METEO, 
                    api_request
                )
                api_data = api_response.data
            else:
                response = await self.client.get(self.base_url, params=params)
                response.raise_for_status()
                api_data = response.json()
            
            return await self._process_forecast_data(api_data, lat, lon)
            
        except httpx.RequestError as e:
            logger.error(f"Network error fetching forecast data: {e}")
            raise Exception("Ошибка сети при получении прогноза")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching forecast data: {e}")
            raise Exception("Внешний сервис прогноза временно недоступен")
        except Exception as e:
            logger.error(f"Unexpected error processing forecast data: {e}")
            raise Exception("Ошибка обработки прогноза качества воздуха")
    
    async def _process_current_data(self, api_data: Dict[str, Any], lat: float, lon: float) -> AirQualityData:
        """Обработка текущих данных от Open-Meteo API с интеграцией WeatherAPI"""
        # Валидация структуры ответа
        if not isinstance(api_data, dict):
            raise ValueError("Invalid API response format")
        
        # Проверка обязательных полей
        if "latitude" not in api_data or "longitude" not in api_data:
            raise ValueError("Missing required location fields in API response")
        
        # Валидация типов координат
        try:
            api_lat = float(api_data["latitude"])
            api_lon = float(api_data["longitude"])
        except (ValueError, TypeError):
            raise ValueError("Invalid coordinate types in API response")
        
        # Проверка разумности координат
        if not (-90 <= api_lat <= 90) or not (-180 <= api_lon <= 180):
            raise ValueError("Invalid coordinate values in API response")
        
        current_data = api_data.get("current", {})
        if not current_data:
            raise ValueError("Missing current data in API response")
        
        # Извлечение данных о загрязнителях
        pollutants = PollutantData(
            pm2_5=current_data.get("pm2_5"),
            pm10=current_data.get("pm10"),
            no2=current_data.get("nitrogen_dioxide"),
            so2=current_data.get("sulphur_dioxide"),
            o3=current_data.get("ozone")
        )
        
        # Проверка, что есть хотя бы одно значение загрязнителя
        pollutant_values = pollutants.model_dump(exclude_none=True)
        if not pollutant_values:
            # Если нет данных о загрязнителях, возвращаем минимальный AQI
            logger.warning("No pollutant data available in API response, using default values")
            aqi_info = AQIInfo(
                value=0,
                category="Нет данных",
                color="#CCCCCC",
                description="Данные о качестве воздуха временно недоступны"
            )
            recommendations = "Данные о качестве воздуха временно недоступны. Попробуйте позже."
            nmu_risk = "unknown"
            health_warnings = ["Данные о качестве воздуха недоступны"]
            
            # Try to get weather data even if air quality data is unavailable
            weather_info = await self._get_weather_data(lat, lon)
            confidence_score, confidence_reason = calculate_confidence(
                ConfidenceInputs(
                    data_source="fallback",
                    source_available=False,
                    cache_age_seconds=0,
                    fallback_used=True,
                )
            )
            
            return AirQualityData(
                timestamp=datetime.now(timezone.utc),
                location=LocationInfo(latitude=lat, longitude=lon),
                aqi=aqi_info,
                pollutants=pollutants,
                weather=weather_info,
                recommendations=recommendations,
                nmu_risk=nmu_risk,
                health_warnings=health_warnings,
                metadata=ResponseMetadata(
                    data_source="fallback",
                    freshness="stale",
                    confidence=confidence_score,
                    confidence_explanation=confidence_reason,
                    fallback_used=True,
                    cache_age_seconds=0,
                ),
            )
        
        # Валидация значений загрязнителей
        for pollutant, value in pollutant_values.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Invalid pollutant value for {pollutant}: {value}")
        
        # Расчет AQI
        aqi_value, category, color = self.aqi_calculator.calculate_aqi(pollutant_values)
        
        aqi_info = AQIInfo(
            value=aqi_value,
            category=category,
            color=color,
            description=self.aqi_calculator.get_category_description(category)
        )
        
        # Генерация рекомендаций
        recommendations = self.aqi_calculator.get_recommendations(aqi_value, category)
        
        # Определение риска НМУ
        nmu_risk = check_nmu_risk(pollutant_values, {})
        
        # Генерация предупреждений о здоровье
        health_warnings = self._generate_health_warnings(aqi_value, pollutants)
        
        # Получение данных о погоде от WeatherAPI
        weather_info = await self._get_weather_data(lat, lon)
        confidence_score, confidence_reason = calculate_confidence(
            ConfidenceInputs(
                data_source="live",
                source_available=True,
                cache_age_seconds=0,
                fallback_used=False,
            )
        )
        
        return AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=lat, longitude=lon),
            aqi=aqi_info,
            pollutants=pollutants,
            weather=weather_info,
            recommendations=recommendations,
            nmu_risk=nmu_risk,
            health_warnings=health_warnings,
            metadata=ResponseMetadata(
                data_source="live",
                freshness="fresh",
                confidence=confidence_score,
                confidence_explanation=confidence_reason,
                fallback_used=False,
                cache_age_seconds=0,
            ),
        )
    
    async def _process_forecast_data(self, api_data: Dict[str, Any], lat: float, lon: float) -> List[AirQualityData]:
        """Обработка прогнозных данных от Open-Meteo API с интеграцией WeatherAPI"""
        # Валидация структуры ответа
        if not isinstance(api_data, dict):
            raise ValueError("Invalid API response format")
        
        # Проверка обязательных полей
        if "latitude" not in api_data or "longitude" not in api_data:
            raise ValueError("Missing required location fields in API response")
        
        hourly_data = api_data.get("hourly", {})
        if not hourly_data:
            raise ValueError("Missing hourly data in API response")
        
        times = hourly_data.get("time", [])
        if not times:
            raise ValueError("Missing time data in API response")
        
        forecast_list = []
        
        # Get weather data once for the location (current weather applies to forecast)
        weather_info = await self._get_weather_data(lat, lon)
        
        # Ограничиваем до 24 часов
        for i in range(min(24, len(times))):
            pollutants = PollutantData(
                pm2_5=hourly_data.get("pm2_5", [None])[i],
                pm10=hourly_data.get("pm10", [None])[i],
                no2=hourly_data.get("nitrogen_dioxide", [None])[i],
                so2=hourly_data.get("sulphur_dioxide", [None])[i],
                o3=hourly_data.get("ozone", [None])[i]
            )
            
            # Проверка, что есть хотя бы одно значение загрязнителя для этого часа
            pollutant_values = pollutants.model_dump(exclude_none=True)
            if not pollutant_values:
                # Пропускаем часы без данных о загрязнителях
                continue
            
            # Расчет AQI для каждого часа
            aqi_value, category, color = self.aqi_calculator.calculate_aqi(pollutant_values)
            
            aqi_info = AQIInfo(
                value=aqi_value,
                category=category,
                color=color,
                description=self.aqi_calculator.get_category_description(category)
            )
            
            recommendations = self.aqi_calculator.get_recommendations(aqi_value, category)
            nmu_risk = check_nmu_risk(pollutant_values, {})
            health_warnings = self._generate_health_warnings(aqi_value, pollutants)
            confidence_score, confidence_reason = calculate_confidence(
                ConfidenceInputs(
                    data_source="forecast",
                    source_available=True,
                    cache_age_seconds=0,
                    fallback_used=False,
                )
            )
            
            forecast_item = AirQualityData(
                timestamp=datetime.fromisoformat(times[i].replace('Z', '+00:00')),
                location=LocationInfo(latitude=lat, longitude=lon),
                aqi=aqi_info,
                pollutants=pollutants,
                weather=weather_info,  # Include weather data in forecast
                recommendations=recommendations,
                nmu_risk=nmu_risk,
                health_warnings=health_warnings,
                metadata=ResponseMetadata(
                    data_source="forecast",
                    freshness="fresh",
                    confidence=confidence_score,
                    confidence_explanation=confidence_reason,
                    fallback_used=False,
                    cache_age_seconds=0,
                ),
            )
            
            forecast_list.append(forecast_item)
        
        # Проверяем, что у нас есть хотя бы один час с данными
        if not forecast_list:
            raise ValueError("No valid forecast data available")
        
        return forecast_list
    
    async def _get_weather_data(self, lat: float, lon: float) -> Optional[WeatherInfo]:
        """
        Получение данных о погоде от WeatherAPI с обработкой ошибок и fallback.
        
        Возвращает None если WeatherAPI недоступен и fallback отключен.
        """
        if not config.weather_api.enabled:
            logger.debug("WeatherAPI is disabled, skipping weather data")
            return None
        
        try:
            # Get combined weather data from WeatherAPI
            weather_data = await weather_api_manager.get_combined_weather(lat, lon)
            
            # Convert to schema format
            weather_info = WeatherInfo(
                temperature=TemperatureData(
                    celsius=weather_data.temperature.celsius,
                    fahrenheit=weather_data.temperature.fahrenheit,
                    timestamp=weather_data.temperature.timestamp,
                    source=weather_data.temperature.source
                ),
                wind=WindData(
                    speed_kmh=weather_data.wind.speed_kmh,
                    speed_mph=weather_data.wind.speed_mph,
                    direction_degrees=weather_data.wind.direction_degrees,
                    direction_compass=weather_data.wind.direction_compass,
                    timestamp=weather_data.wind.timestamp
                ) if weather_data.wind else None,
                pressure=PressureData(
                    pressure_mb=weather_data.pressure.pressure_mb,
                    pressure_in=weather_data.pressure.pressure_in,
                    timestamp=weather_data.pressure.timestamp
                ) if weather_data.pressure else None,
                location_name=weather_data.location_name
            )
            
            logger.debug(f"Successfully retrieved weather data from WeatherAPI for {lat}, {lon}")
            return weather_info
            
        except Exception as e:
            logger.warning(f"Failed to get weather data from WeatherAPI: {e}")
            
            # Return None if fallback is disabled or if we want to continue without weather data
            if not config.weather_api.fallback_enabled:
                logger.debug("WeatherAPI fallback disabled, continuing without weather data")
                return None
            
            # Return minimal fallback weather data
            logger.debug("Using fallback weather data")
            return WeatherInfo(
                temperature=TemperatureData(
                    celsius=0.0,
                    fahrenheit=32.0,
                    timestamp=datetime.now(timezone.utc),
                    source="fallback"
                ),
                wind=None,  # No fallback wind data
                pressure=None,  # No fallback pressure data
                location_name=f"{lat:.2f},{lon:.2f}"
            )

    def _generate_health_warnings(self, aqi_value: int, pollutants: PollutantData) -> List[str]:
        """Генерация предупреждений о здоровье на основе AQI и загрязнителей"""
        warnings = []
        
        # Проверка условий "Черное небо"
        pollutant_dict = pollutants.model_dump(exclude_none=True)
        blacksky = is_blacksky_conditions(pollutant_dict)
        
        if blacksky:
            warnings.append("РЕЖИМ 'ЧЕРНОЕ НЕБО' - КРИТИЧЕСКАЯ СИТУАЦИЯ!")
            warnings.extend([
                "Немедленно покиньте улицу и зайдите в помещение",
                "Закройте все окна и двери",
                "Используйте респиратор при необходимости выхода"
            ])
            return warnings
        
        # Стандартные предупреждения по AQI
        if aqi_value > 150:
            warnings.append("Избегайте физических нагрузок на открытом воздухе")
        
        if aqi_value > 200:
            warnings.append("Рекомендуется оставаться в помещении")
            warnings.append("Используйте очистители воздуха при наличии")
        
        # Специфические предупреждения для отдельных загрязнителей
        if pollutants.pm2_5 and pollutants.pm2_5 > 75:
            warnings.append("Высокий уровень PM2.5 - особенно опасно для людей с заболеваниями сердца и легких")
        
        if pollutants.no2 and pollutants.no2 > 200:
            warnings.append("Повышенный уровень NO2 - избегайте нахождения вблизи дорог")
        
        # Добавление НМУ рекомендаций
        nmu_risk = check_nmu_risk(pollutant_dict)
        if nmu_risk in ["high", "critical"]:
            nmu_recommendations = get_nmu_recommendations(nmu_risk, blacksky)
            warnings.extend(nmu_recommendations)
        
        return warnings
    
    async def check_external_api_health(self) -> str:
        """
        Проверка доступности внешнего API для health check.
        
        Выполняет простой запрос к Open-Meteo API для проверки связности
        и корректности ответа.
        
        Returns:
            str: Статус API ("healthy", "unhealthy", "degraded")
        """
        try:
            # Simple ping request based on service type
            test_params = {
                "latitude": 55.7558,  # Москва для тестирования
                "longitude": 37.6176,
                "current": "pm10"  # Минимальный набор данных
            }
            
            # Логирование запроса через privacy middleware
            test_url = f"{self.base_url}?latitude=55.7558&longitude=37.6176&current=pm10"
            # ✅ FIX #9: Mask coordinates in logs
            masked_url = mask_coordinates(test_url)
            self._log_external_request(masked_url, has_coordinates=True)
            
            # Use connection pool if enabled, otherwise fallback to direct client
            if self.use_connection_pool:
                api_request = APIRequest(
                    method="GET",
                    url=self.base_url,
                    params=test_params,
                    timeout=10.0
                )
                api_response = await get_connection_pool_manager().execute_request(
                    ServiceType.OPEN_METEO, 
                    api_request
                )
                data = api_response.data
                status_code = api_response.status_code
            else:
                response = await self.client.get(
                    self.base_url,
                    params=test_params,
                    timeout=10.0  # Короткий таймаут для health check
                )
                response.raise_for_status()
                data = response.json()
                status_code = response.status_code
            
            # Проверка структуры ответа
            if not isinstance(data, dict):
                logger.warning("External API returned invalid response format")
                return "degraded"
            
            # Проверка наличия обязательных полей
            required_fields = ["latitude", "longitude"]
            for field in required_fields:
                if field not in data:
                    logger.warning(f"External API response missing required field: {field}")
                    return "degraded"
            
            # Проверка наличия данных - более гибкая проверка
            current_data = data.get("current", {})
            if not current_data:
                logger.warning("External API response missing current data")
                return "degraded"
            
            # Проверяем, что есть хотя бы одно значение загрязнителя
            has_pollutant_data = any(
                current_data.get(field) is not None 
                for field in ["pm10", "pm2_5", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
            )
            
            if not has_pollutant_data:
                logger.warning("External API response missing pollutant data")
                return "degraded"
            
            logger.info("External API health check passed")
            return "healthy"
            
        except httpx.TimeoutException:
            logger.error("External API health check timed out")
            return "unhealthy"
        except httpx.HTTPStatusError as e:
            logger.error(f"External API health check failed with HTTP error: {e.response.status_code}")
            return "unhealthy"
        except httpx.RequestError as e:
            logger.error(f"External API health check failed with network error: {e}")
            return "unhealthy"
        except ValueError as e:
            logger.error(f"External API health check failed - invalid JSON response: {e}")
            return "degraded"
        except Exception as e:
            logger.error(f"External API health check failed with unexpected error: {e}")
            return "unhealthy"
    
    def _log_external_request(self, url: str, has_coordinates: bool = False):
        """
        Логирование внешних запросов через privacy middleware
        
        ✅ FIX #9: URL should already be masked before calling this function
        """
        try:
            from middleware import get_privacy_middleware
            
            privacy_middleware = get_privacy_middleware()
            if privacy_middleware:
                # Валидация разрешенности запроса
                if not privacy_middleware.validate_external_request(url):
                    logger.warning(f"External request blocked by privacy middleware: {url}")
                    return
                
                # Логирование запроса (URL already masked)
                privacy_middleware.log_external_request(url, has_coordinates)
            else:
                # Fallback логирование если middleware недоступен
                logger.info(f"External API request - URL: {url}, Has coordinates: {has_coordinates}")
                
        except Exception as e:
            logger.error(f"Error logging external request: {e}")
    
    async def cleanup(self):
        """Очистка ресурсов и устаревших записей кэша"""
        await self.cache_manager.clear_expired()
        if self.client:
            await self.client.aclose()
