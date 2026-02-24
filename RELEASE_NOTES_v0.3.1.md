# AirTrace RU v0.3.1

Дата релиза: 2026-02-24

## Что вошло

### 1) Исправления rate limiting (security + correctness)
- `#36`: устранен глобальный bypass rate limiting:
  - удален `"/"` из `skip_paths` в `main.py`
  - substring matching заменен на явный контракт сопоставления путей
- `#37`: зафиксирован контракт `skip_paths`:
  - exact match или path-segment prefix
  - без случайных совпадений (`/api/docs-backup`, `/docsx`, `/openapi.json.copy`)
- `#34`: `RateLimitManager` теперь привязывается к live middleware instance FastAPI/Starlette
  - `enable()/disable()` и stats работают с реальным middleware, а не с дублирующим экземпляром

### 2) Усиление безопасности определения client IP
- `#35`: обработка `X-Forwarded-For` / `X-Real-IP` сделана `safe-by-default`
  - без явной настройки используется `request.client.host`
  - добавлена поддержка trusted proxies (IP/CIDR allowlist)
  - добавлена конфигурация через env:
    - `PERFORMANCE_RATE_LIMIT_TRUST_FORWARDED_HEADERS`
    - `PERFORMANCE_RATE_LIMIT_TRUSTED_PROXY_IPS`

### 3) Исправление invalidation кэша combined weather data
- `#38`: `UnifiedWeatherService.invalidate_location_cache()` больше не использует несовместимый pattern-key
- Добавлен `MultiLevelCacheManager.invalidate_by_coordinates(lat, lon)`
- Инвалидация теперь использует тот же keyspace (rounding/hash), что и `get/set`

### 4) Тесты и документация
- Добавлены регрессионные тесты для:
  - path matching в `skip_paths`
  - spoofed proxy headers / trusted proxy behavior
  - привязки `RateLimitManager` к live middleware instance
  - cache invalidation по координатам
- Обновлен `README.md`:
  - changelog для `v0.3.1`
  - инструкции по безопасной настройке reverse proxy для rate limiting

## Закрытые issues
- `#33` Code Review: fix rate-limiting bypasses and cache invalidation mismatch
- `#34` Rate limiting: unify RateLimitManager state with live FastAPI middleware instance
- `#35` Security: harden client IP extraction for rate limiting (trusted proxy handling)
- `#36` Rate limiting: fix global bypass caused by skip_paths '/' and substring matching
- `#37` Rate limiting: define and test skip-path matching contract (no substring bypasses)
- `#38` Caching: fix UnifiedWeatherService combined-cache invalidation keyspace mismatch
