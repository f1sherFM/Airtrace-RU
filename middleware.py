"""
Privacy Middleware для AirTrace RU Backend

Middleware для защиты приватности пользователей, включая фильтрацию
координат из логов, санитизацию выходных данных и безопасное логирование ошибок.
"""

import logging
import re
import json
from typing import Dict, Any, Optional, List
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import traceback
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PrivacyAwareFormatter(logging.Formatter):
    """
    Кастомный форматтер для логов, который удаляет координаты и другие
    чувствительные данные из сообщений логов.
    """
    
    # Паттерны для поиска координат в различных форматах
    COORDINATE_PATTERNS = [
        r'lat(?:itude)?["\s]*[:=]["\s]*(-?\d+\.?\d*)',
        r'lon(?:gitude)?["\s]*[:=]["\s]*(-?\d+\.?\d*)',
        r'latitude["\s]*[:=]["\s]*(-?\d+\.?\d*)',
        r'longitude["\s]*[:=]["\s]*(-?\d+\.?\d*)',
        r'(?:lat|lon)=(-?\d+\.?\d*)',
        r'(-?\d{1,3}\.\d+),\s*(-?\d{1,3}\.\d+)',  # Формат "lat,lon"
    ]
    
    # Другие чувствительные данные
    SENSITIVE_PATTERNS = [
        r'ip["\s]*[:=]["\s]*"?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"?',
        r'user_id["\s]*[:=]["\s]*"?([^"\s,}]+)"?',
        r'session["\s]*[:=]["\s]*"?([^"\s,}]+)"?',
        r'User session:\s*(.+?)(?:\s|$)',  # Handle "User session: <data>" pattern
        r'user_id:\s*(.+?)(?:\s|$)',       # Handle "user_id: <data>" pattern
        r'session=(.+?)(?:\s|&|$)',        # Handle "session=<data>" pattern
    ]
    
    def format(self, record):
        """Форматирование записи лога с удалением чувствительных данных"""
        # Получаем исходное сообщение
        original_message = super().format(record)
        
        # Удаляем координаты
        sanitized_message = self._sanitize_coordinates(original_message)
        
        # Удаляем другие чувствительные данные
        sanitized_message = self._sanitize_sensitive_data(sanitized_message)
        
        return sanitized_message
    
    def _sanitize_coordinates(self, message: str) -> str:
        """Удаление координат из сообщения лога"""
        sanitized = message
        
        for pattern in self.COORDINATE_PATTERNS:
            sanitized = re.sub(pattern, r'[COORDINATE_FILTERED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _sanitize_sensitive_data(self, message: str) -> str:
        """Удаление других чувствительных данных"""
        sanitized = message
        
        for pattern in self.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, r'[SENSITIVE_DATA_FILTERED]', sanitized, flags=re.IGNORECASE)
        
        # Special handling for patterns that might contain sensitive keywords
        if any(keyword in message.lower() for keyword in ['ip', 'user_id', 'session']):
            # More aggressive filtering for messages containing sensitive keywords
            sanitized = re.sub(r':\s*([^"\s,}]+)', r': [SENSITIVE_DATA_FILTERED]', sanitized)
        
        return sanitized


class PrivacyMiddleware(BaseHTTPMiddleware):
    """
    Middleware для защиты приватности пользователей.
    
    Функции:
    1. Фильтрация координат из логов
    2. Санитизация выходных данных
    3. Безопасное логирование ошибок
    4. Контроль передачи данных третьим лицам
    """
    
    def __init__(self, app: ASGIApp, enable_request_logging: bool = True):
        super().__init__(app)
        self.enable_request_logging = enable_request_logging
        
        # Настройка privacy-aware логгера
        self._setup_privacy_logger()
        
        # Список разрешенных внешних доменов для передачи координат
        self.allowed_external_domains = [
            "air-quality-api.open-meteo.com",
            "api.open-meteo.com"
        ]
        
        # Поля, которые должны быть санитизированы в ответах
        self.sensitive_response_fields = [
            "user_id", "session_id", "ip_address", "device_id"
        ]
    
    def _setup_privacy_logger(self):
        """Настройка логгера с privacy-aware форматтером"""
        privacy_logger = logging.getLogger("privacy_middleware")
        
        # Удаляем существующие обработчики
        for handler in privacy_logger.handlers[:]:
            privacy_logger.removeHandler(handler)
        
        # Создаем новый обработчик с privacy форматтером
        handler = logging.StreamHandler()
        handler.setFormatter(PrivacyAwareFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        privacy_logger.addHandler(handler)
        privacy_logger.setLevel(logging.INFO)
        privacy_logger.propagate = False
        
        self.privacy_logger = privacy_logger
    
    async def dispatch(self, request: Request, call_next):
        """Основная логика middleware"""
        start_time = datetime.now(timezone.utc)
        
        # Логирование запроса (без координат)
        if self.enable_request_logging:
            await self._log_request_safely(request)
        
        try:
            # Обработка запроса
            response = await call_next(request)
            
            # Санитизация ответа
            sanitized_response = await self._sanitize_response(response)
            
            # Логирование успешного ответа
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.privacy_logger.info(
                f"Request processed successfully in {processing_time:.3f}s - "
                f"Method: {request.method}, Path: {request.url.path}, "
                f"Status: {sanitized_response.status_code}"
            )
            
            return sanitized_response
            
        except Exception as e:
            # Безопасное логирование ошибок
            await self._log_error_safely(request, e)
            
            # Возврат санитизированного ответа об ошибке
            return await self._create_safe_error_response(e)
    
    async def _log_request_safely(self, request: Request):
        """Безопасное логирование запроса без координат"""
        # Получаем параметры запроса
        query_params = dict(request.query_params)
        
        # Удаляем координаты из параметров для логирования
        safe_params = self._filter_coordinates_from_params(query_params)
        
        # Получаем заголовки (исключая чувствительные)
        safe_headers = self._filter_sensitive_headers(dict(request.headers))
        
        self.privacy_logger.info(
            f"Incoming request - Method: {request.method}, "
            f"Path: {request.url.path}, "
            f"Params: {safe_params}, "
            f"User-Agent: {safe_headers.get('user-agent', 'Unknown')}"
        )
    
    def _filter_coordinates_from_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Удаление координат из параметров запроса для логирования"""
        safe_params = {}
        coordinate_fields = ['lat', 'lon', 'latitude', 'longitude']
        
        for key, value in params.items():
            if key.lower() in coordinate_fields:
                safe_params[key] = "[COORDINATE_FILTERED]"
            else:
                safe_params[key] = value
        
        return safe_params
    
    def _filter_sensitive_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Фильтрация чувствительных заголовков"""
        safe_headers = {}
        sensitive_headers = [
            'authorization', 'cookie', 'x-api-key', 'x-auth-token',
            'x-forwarded-for', 'x-real-ip'
        ]
        
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                safe_headers[key] = "[SENSITIVE_HEADER_FILTERED]"
            else:
                safe_headers[key] = value
        
        return safe_headers
    
    async def _sanitize_response(self, response: Response) -> Response:
        """Санитизация ответа для удаления чувствительных данных"""
        # Если это не JSON ответ, возвращаем как есть
        if not isinstance(response, JSONResponse):
            return response
        
        try:
            # Получаем тело ответа
            response_body = response.body
            if not response_body:
                return response
            
            # Парсим JSON
            response_data = json.loads(response_body.decode())
            
            # Санитизируем данные
            sanitized_data = self._sanitize_json_data(response_data)
            
            # Создаем новый ответ с санитизированными данными
            return JSONResponse(
                content=sanitized_data,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.privacy_logger.warning(f"Failed to sanitize response: {e}")
            return response
    
    def _sanitize_json_data(self, data: Any) -> Any:
        """Рекурсивная санитизация JSON данных"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if key.lower() in self.sensitive_response_fields:
                    sanitized[key] = "[SENSITIVE_DATA_FILTERED]"
                else:
                    sanitized[key] = self._sanitize_json_data(value)
            return sanitized
        
        elif isinstance(data, list):
            return [self._sanitize_json_data(item) for item in data]
        
        else:
            return data
    
    async def _log_error_safely(self, request: Request, error: Exception):
        """Безопасное логирование ошибок без координат"""
        # Получаем информацию о запросе без координат
        safe_params = self._filter_coordinates_from_params(dict(request.query_params))
        
        # Получаем traceback без чувствительных данных
        error_traceback = traceback.format_exc()
        sanitized_traceback = self._sanitize_traceback(error_traceback)
        
        self.privacy_logger.error(
            f"Request failed - Method: {request.method}, "
            f"Path: {request.url.path}, "
            f"Params: {safe_params}, "
            f"Error: {type(error).__name__}: {str(error)}, "
            f"Traceback: {sanitized_traceback}"
        )
    
    def _sanitize_traceback(self, traceback_str: str) -> str:
        """Санитизация traceback от координат и чувствительных данных"""
        sanitized = traceback_str
        
        # Удаляем координаты из traceback
        formatter = PrivacyAwareFormatter()
        sanitized = formatter._sanitize_coordinates(sanitized)
        sanitized = formatter._sanitize_sensitive_data(sanitized)
        
        return sanitized
    
    async def _create_safe_error_response(self, error: Exception) -> JSONResponse:
        """Создание безопасного ответа об ошибке"""
        # Определяем тип ошибки и соответствующий код статуса
        if isinstance(error, ValueError):
            status_code = 400
            error_code = "VALIDATION_ERROR"
            message = "Некорректные данные запроса"
        elif isinstance(error, ConnectionError):
            status_code = 503
            error_code = "SERVICE_UNAVAILABLE"
            message = "Внешний сервис временно недоступен"
        else:
            status_code = 500
            error_code = "INTERNAL_ERROR"
            message = "Внутренняя ошибка сервера"
        
        # Создаем безопасный ответ об ошибке
        error_response = {
            "error": {
                "code": error_code,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
            }
        }
        
        return JSONResponse(
            content=error_response,
            status_code=status_code
        )
    
    def validate_external_request(self, url: str) -> bool:
        """
        Валидация внешних запросов для контроля передачи данных третьим лицам.
        
        Args:
            url: URL внешнего запроса
            
        Returns:
            bool: True если запрос разрешен
        """
        from urllib.parse import urlparse
        
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Проверяем, разрешен ли домен
            for allowed_domain in self.allowed_external_domains:
                if domain == allowed_domain or domain.endswith('.' + allowed_domain):
                    return True
            
            self.privacy_logger.warning(
                f"Blocked external request to unauthorized domain: {domain}"
            )
            return False
            
        except Exception as e:
            self.privacy_logger.error(f"Error validating external request: {e}")
            return False
    
    def log_external_request(self, url: str, has_coordinates: bool = False):
        """
        Логирование внешних запросов с указанием передачи координат.
        
        Args:
            url: URL внешнего запроса
            has_coordinates: Передаются ли координаты в запросе
        """
        from urllib.parse import urlparse
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        if has_coordinates:
            self.privacy_logger.info(
                f"External API request with coordinates - Domain: {domain}, "
                f"Path: {parsed_url.path}, Coordinates: [FILTERED_FOR_PRIVACY]"
            )
        else:
            self.privacy_logger.info(
                f"External API request without coordinates - Domain: {domain}, "
                f"Path: {parsed_url.path}"
            )


def setup_privacy_logging():
    """
    Настройка privacy-aware логирования для всего приложения.
    
    Должна быть вызвана при инициализации приложения.
    """
    # Получаем root logger
    root_logger = logging.getLogger()
    
    # Создаем privacy-aware форматтер
    privacy_formatter = PrivacyAwareFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Обновляем все существующие обработчики
    for handler in root_logger.handlers:
        handler.setFormatter(privacy_formatter)
    
    # Настраиваем логгеры для основных модулей
    module_loggers = [
        'main', 'services', 'utils', 'middleware', 'schemas'
    ]
    
    for module_name in module_loggers:
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(logging.INFO)
        
        # Если у модуля нет обработчиков, добавляем стандартный
        if not module_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(privacy_formatter)
            module_logger.addHandler(handler)
            module_logger.propagate = False


# Глобальный экземпляр middleware для использования в других модулях
privacy_middleware_instance: Optional[PrivacyMiddleware] = None


def get_privacy_middleware() -> Optional[PrivacyMiddleware]:
    """Получение глобального экземпляра privacy middleware"""
    return privacy_middleware_instance


def set_privacy_middleware(middleware: PrivacyMiddleware):
    """Установка глобального экземпляра privacy middleware"""
    global privacy_middleware_instance
    privacy_middleware_instance = middleware