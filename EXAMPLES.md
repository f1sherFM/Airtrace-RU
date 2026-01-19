# 📚 Примеры использования AirTrace RU

Этот файл содержит практические примеры использования AirTrace RU API и веб-интерфейса.

## 🚀 Быстрый старт

### Запуск приложения
```bash
# Клонирование и установка
git clone https://github.com/f1sherFM/Airtrace-RU.git
cd Airtrace-RU
pip install -r requirements.txt

# Запуск (API + Web интерфейс)
python start_app.py
```

После запуска:
- API: http://localhost:8000
- Web UI: http://localhost:3000
- API Docs: http://localhost:8000/docs

## 🌐 Примеры API запросов

### 1. Получение текущих данных

#### Магнитогорск
```bash
curl "http://localhost:8000/weather/current?lat=53.4069&lon=58.9794"
```

**Ответ:**
```json
{
  "timestamp": "2026-01-17T16:35:14.123456+00:00",
  "location": {
    "latitude": 53.4069,
    "longitude": 58.9794
  },
  "aqi": {
    "value": 42,
    "category": "Хорошее",
    "color": "#00E400",
    "description": "Качество воздуха считается удовлетворительным для большинства людей."
  },
  "pollutants": {
    "pm2_5": 12.1,
    "pm10": 12.8,
    "no2": 31.6,
    "so2": 3.1,
    "o3": 49.0
  },
  "recommendations": "Отличное качество воздуха для активного отдыха на улице.",
  "nmu_risk": "low",
  "health_warnings": []
}
```

#### Москва
```bash
curl "http://localhost:8000/weather/current?lat=55.7558&lon=37.6176"
```

#### Санкт-Петербург
```bash
curl "http://localhost:8000/weather/current?lat=59.9311&lon=30.3609"
```

### 2. Прогноз качества воздуха

```bash
curl "http://localhost:8000/weather/forecast?lat=53.4069&lon=58.9794"
```

**Ответ (массив на 24 часа):**
```json
[
  {
    "timestamp": "2026-01-17T17:00:00+00:00",
    "aqi": {
      "value": 45,
      "category": "Хорошее",
      "color": "#00E400"
    },
    "pollutants": {
      "pm2_5": 13.2,
      "pm10": 14.1,
      "no2": 28.4,
      "so2": 2.8,
      "o3": 52.1
    }
  }
]
```

### 3. Проверка здоровья системы

```bash
curl "http://localhost:8000/health"
```

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-17T16:35:14.123456+00:00",
  "services": {
    "api": "healthy",
    "external_api": "healthy",
    "cache": "healthy (3 entries)",
    "aqi_calculator": "healthy",
    "privacy_middleware": "healthy",
    "nmu_detector": "healthy"
  },
  "uptime": "2h 15m 30s",
  "version": "1.0.0"
}
```

## 🌍 Примеры координат городов

### Крупные города России

| Город | Широта | Долгота | API запрос |
|-------|--------|---------|------------|
| **Москва** | 55.7558 | 37.6176 | `?lat=55.7558&lon=37.6176` |
| **Санкт-Петербург** | 59.9311 | 30.3609 | `?lat=59.9311&lon=30.3609` |
| **Новосибирск** | 55.0084 | 82.9357 | `?lat=55.0084&lon=82.9357` |
| **Екатеринбург** | 56.8431 | 60.6454 | `?lat=56.8431&lon=60.6454` |
| **Казань** | 55.8304 | 49.0661 | `?lat=55.8304&lon=49.0661` |
| **Нижний Новгород** | 56.3269 | 44.0075 | `?lat=56.3269&lon=44.0075` |
| **Челябинск** | 55.1644 | 61.4368 | `?lat=55.1644&lon=61.4368` |
| **Самара** | 53.2001 | 50.15 | `?lat=53.2001&lon=50.15` |

### Промышленные города

| Город | Широта | Долгота | Особенности |
|-------|--------|---------|-------------|
| **Магнитогорск** | 53.4069 | 58.9794 | Металлургия |
| **Норильск** | 69.3558 | 88.1893 | Горнодобыча |
| **Череповец** | 59.1374 | 37.9097 | Металлургия |
| **Липецк** | 52.6031 | 39.5708 | Металлургия |

## 📍 Форматы ввода координат

### В веб-интерфейсе

#### 1. Десятичные градусы
```
Широта: 61.25
Долгота: 73.4333
```

#### 2. Градусы-минуты-секунды (DMS)
```
Широта: 61° 15' 00" N
Долгота: 73° 26' 00" E
```

#### 3. Текстовый ввод (поддерживаемые форматы)
```
61°15′00″ с. ш., 73°26′00″ в. д.
61°15'00" N, 73°26'00" E
61 15 00 N 73 26 00 E
61.25, 73.4333
```

## 🧪 Примеры тестирования

### Запуск всех тестов
```bash
pytest tests/ -v
```

### Тестирование конкретного компонента
```bash
# AQI калькулятор
pytest tests/test_property_aqi_calculation.py -v

# Валидация координат
pytest tests/test_property_coordinate_validation.py -v

# Кэширование
pytest tests/test_property_cache_lifecycle.py -v
```

### Property-based тестирование
```bash
# Все property-based тесты
pytest tests/test_property_*.py -v

# С подробным выводом
pytest tests/test_property_aqi_calculation.py -v -s
```

## 🔧 Примеры конфигурации

### Переменные окружения
```bash
# .env файл
CACHE_TTL=300
MAX_CACHE_SIZE=1000
LOG_LEVEL=INFO
API_TIMEOUT=30
```

### Кастомные настройки
```python
# config.py
class Settings:
    cache_ttl: int = 300  # 5 минут
    max_cache_size: int = 1000
    api_timeout: int = 30
    log_level: str = "INFO"
```

## 📊 Интерпретация данных

### Уровни AQI
```python
def interpret_aqi(aqi_value):
    if aqi_value <= 50:
        return "🟢 Хорошее - безопасно для всех"
    elif aqi_value <= 100:
        return "🟡 Умеренное - приемлемо для большинства"
    elif aqi_value <= 150:
        return "🟠 Вредно для чувствительных групп"
    elif aqi_value <= 200:
        return "🔴 Вредно для всех"
    elif aqi_value <= 300:
        return "🟣 Очень вредно"
    else:
        return "⚫ Опасно - критическая ситуация"
```

### НМУ риски
```python
def interpret_nmu_risk(risk_level):
    risks = {
        "low": "🟢 Низкий - обычные условия",
        "medium": "🟡 Умеренный - следите за изменениями", 
        "high": "🟠 Высокий - возможны НМУ",
        "critical": "🔴 Критический - режим 'Черное небо'"
    }
    return risks.get(risk_level, "❓ Неизвестно")
```

## 🚨 Обработка ошибок

### Типичные ошибки API

#### Неверные координаты
```bash
curl "http://localhost:8000/weather/current?lat=999&lon=999"
```
**Ответ:**
```json
{
  "detail": "Широта должна быть от -90 до 90"
}
```

#### Недоступность внешнего API
```json
{
  "detail": "API недоступен: Connection timeout"
}
```

### Обработка в коде
```python
import httpx

async def get_air_quality(lat: float, lon: float):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:8000/weather/current",
                params={"lat": lat, "lon": lon}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP ошибка: {e.response.status_code}")
    except httpx.RequestError as e:
        print(f"Ошибка запроса: {e}")
```

## 📱 Интеграция в приложения

### Python
```python
import asyncio
import httpx

class AirQualityClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def get_current(self, lat: float, lon: float):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/weather/current",
                params={"lat": lat, "lon": lon}
            )
            return response.json()

# Использование
client = AirQualityClient()
data = asyncio.run(client.get_current(53.4069, 58.9794))
print(f"AQI: {data['aqi']['value']} - {data['aqi']['category']}")
```

### JavaScript
```javascript
class AirQualityAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async getCurrent(lat, lon) {
        const response = await fetch(
            `${this.baseUrl}/weather/current?lat=${lat}&lon=${lon}`
        );
        return await response.json();
    }
}

// Использование
const api = new AirQualityAPI();
api.getCurrent(53.4069, 58.9794)
    .then(data => {
        console.log(`AQI: ${data.aqi.value} - ${data.aqi.category}`);
    });
```

---

**💡 Больше примеров в [документации API](http://localhost:8000/docs) и [веб-интерфейсе](http://localhost:3000)!**