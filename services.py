"""
Сервисы для интеграции с внешними API и бизнес-логика AirTrace RU Backend

Содержит асинхронные сервисы для получения данных о качестве воздуха,
кэширования и обработки данных с акцентом на приватность.
"""

import httpx
import asyncio
import logging
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
    CacheEntry
)
from utils import AQICalculator, check_nmu_risk, is_blacksky_conditions, get_nmu_recommendations

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Менеджер кэширования данных с акцентом на приватность.
    
    Использует хэшированные ключи для предотвращения сохранения
    реальных координат пользователей.
    """
    
    def __init__(self, ttl_minutes: int = 15):
        self._cache: Dict[str, CacheEntry] = {}
        self.ttl_seconds = ttl_minutes * 60
        self.aqi_calculator = AQICalculator()
    
    def _generate_key(self, lat: float, lon: float) -> str:
        """
        Генерация приватного ключа кэша без сохранения координат.
        
        Округляет координаты для группировки близких запросов
        и использует хэширование для обеспечения приватности.
        """
        # Округление до 2 знаков для группировки близких локаций
        rounded_lat = round(lat, 2)
        rounded_lon = round(lon, 2)
        key_string = f"{rounded_lat}:{rounded_lon}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Получение данных из кэша с обработкой сбоев"""
        try:
            key = self._generate_key(lat, lon)
            
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    logger.info("Cache hit for location")
                    return entry.data
                else:
                    # Удаление устаревшей записи
                    del self._cache[key]
                    logger.info("Cache entry expired and removed")
            
            logger.info("Cache miss for location")
            return None
        except Exception as e:
            logger.warning(f"Cache get operation failed: {e}")
            return None
    
    async def set(self, lat: float, lon: float, data: Dict[str, Any]) -> None:
        """Сохранение данных в кэш с обработкой сбоев"""
        try:
            key = self._generate_key(lat, lon)
            entry = CacheEntry(
                data=data,
                timestamp=datetime.now(timezone.utc),
                ttl_seconds=self.ttl_seconds
            )
            self._cache[key] = entry
            logger.info("Data cached for location")
        except Exception as e:
            logger.warning(f"Cache set operation failed: {e}")
            # Не выбрасываем исключение, просто логируем
    
    async def clear_expired(self) -> None:
        """Очистка устаревших записей кэша с обработкой сбоев"""
        try:
            expired_keys = [
                key for key, entry in self._cache.items() 
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired cache entries")
        except Exception as e:
            logger.warning(f"Cache clear_expired operation failed: {e}")
            # Не выбрасываем исключение, просто логируем
    
    def get_status(self) -> str:
        """Получение статуса кэша для health check с обработкой сбоев"""
        try:
            total_entries = len(self._cache)
            return f"healthy ({total_entries} entries)"
        except Exception as e:
            logger.warning(f"Cache status check failed: {e}")
            return "unhealthy (status check failed)"


class AirQualityService:
    """
    Сервис для получения данных о качестве воздуха из Open-Meteo API
    с поддержкой кэширования и обработки ошибок.
    """
    
    def __init__(self):
        self.base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self.cache_manager = CacheManager()
        self.aqi_calculator = AQICalculator()
    
    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход"""
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
            self._log_external_request(request_url, has_coordinates=True)
            
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
            self._log_external_request(request_url, has_coordinates=True)
            
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
        """Обработка текущих данных от Open-Meteo API"""
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
            
            return AirQualityData(
                timestamp=datetime.now(timezone.utc),
                location=LocationInfo(latitude=lat, longitude=lon),
                aqi=aqi_info,
                pollutants=pollutants,
                recommendations=recommendations,
                nmu_risk=nmu_risk,
                health_warnings=health_warnings
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
        
        return AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=lat, longitude=lon),
            aqi=aqi_info,
            pollutants=pollutants,
            recommendations=recommendations,
            nmu_risk=nmu_risk,
            health_warnings=health_warnings
        )
    
    async def _process_forecast_data(self, api_data: Dict[str, Any], lat: float, lon: float) -> List[AirQualityData]:
        """Обработка прогнозных данных от Open-Meteo API"""
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
            
            forecast_item = AirQualityData(
                timestamp=datetime.fromisoformat(times[i].replace('Z', '+00:00')),
                location=LocationInfo(latitude=lat, longitude=lon),
                aqi=aqi_info,
                pollutants=pollutants,
                recommendations=recommendations,
                nmu_risk=nmu_risk,
                health_warnings=health_warnings
            )
            
            forecast_list.append(forecast_item)
        
        # Проверяем, что у нас есть хотя бы один час с данными
        if not forecast_list:
            raise ValueError("No valid forecast data available")
        
        return forecast_list
    
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
            # Простой запрос для проверки доступности с коротким таймаутом
            test_params = {
                "latitude": 55.7558,  # Москва для тестирования
                "longitude": 37.6176,
                "current": "pm10"  # Минимальный набор данных
            }
            
            # Логирование запроса через privacy middleware
            test_url = f"{self.base_url}?latitude=55.7558&longitude=37.6176&current=pm10"
            self._log_external_request(test_url, has_coordinates=True)
            
            response = await self.client.get(
                self.base_url,
                params=test_params,
                timeout=10.0  # Короткий таймаут для health check
            )
            
            response.raise_for_status()
            
            # Проверка структуры ответа
            data = response.json()
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
        """Логирование внешних запросов через privacy middleware"""
        try:
            from middleware import get_privacy_middleware
            
            privacy_middleware = get_privacy_middleware()
            if privacy_middleware:
                # Валидация разрешенности запроса
                if not privacy_middleware.validate_external_request(url):
                    logger.warning(f"External request blocked by privacy middleware: {url}")
                    return
                
                # Логирование запроса
                privacy_middleware.log_external_request(url, has_coordinates)
            else:
                # Fallback логирование если middleware недоступен
                logger.info(f"External API request - URL: [COORDINATES_FILTERED], Has coordinates: {has_coordinates}")
                
        except Exception as e:
            logger.error(f"Error logging external request: {e}")
    
    async def cleanup(self):
        """Очистка ресурсов и устаревших записей кэша"""
        await self.cache_manager.clear_expired()
        await self.client.aclose()