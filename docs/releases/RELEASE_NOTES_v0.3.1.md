# AirTrace RU v0.3.1

Р”Р°С‚Р° СЂРµР»РёР·Р°: 2026-02-24

## Р§С‚Рѕ РІРѕС€Р»Рѕ

### 1) РСЃРїСЂР°РІР»РµРЅРёСЏ rate limiting (security + correctness)
- `#36`: СѓСЃС‚СЂР°РЅРµРЅ РіР»РѕР±Р°Р»СЊРЅС‹Р№ bypass rate limiting:
  - СѓРґР°Р»РµРЅ `"/"` РёР· `skip_paths` РІ `main.py`
  - substring matching Р·Р°РјРµРЅРµРЅ РЅР° СЏРІРЅС‹Р№ РєРѕРЅС‚СЂР°РєС‚ СЃРѕРїРѕСЃС‚Р°РІР»РµРЅРёСЏ РїСѓС‚РµР№
- `#37`: Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅ РєРѕРЅС‚СЂР°РєС‚ `skip_paths`:
  - exact match РёР»Рё path-segment prefix
  - Р±РµР· СЃР»СѓС‡Р°Р№РЅС‹С… СЃРѕРІРїР°РґРµРЅРёР№ (`/api/docs-backup`, `/docsx`, `/openapi.json.copy`)
- `#34`: `RateLimitManager` С‚РµРїРµСЂСЊ РїСЂРёРІСЏР·С‹РІР°РµС‚СЃСЏ Рє live middleware instance FastAPI/Starlette
  - `enable()/disable()` Рё stats СЂР°Р±РѕС‚Р°СЋС‚ СЃ СЂРµР°Р»СЊРЅС‹Рј middleware, Р° РЅРµ СЃ РґСѓР±Р»РёСЂСѓСЋС‰РёРј СЌРєР·РµРјРїР»СЏСЂРѕРј

### 2) РЈСЃРёР»РµРЅРёРµ Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё РѕРїСЂРµРґРµР»РµРЅРёСЏ client IP
- `#35`: РѕР±СЂР°Р±РѕС‚РєР° `X-Forwarded-For` / `X-Real-IP` СЃРґРµР»Р°РЅР° `safe-by-default`
  - Р±РµР· СЏРІРЅРѕР№ РЅР°СЃС‚СЂРѕР№РєРё РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ `request.client.host`
  - РґРѕР±Р°РІР»РµРЅР° РїРѕРґРґРµСЂР¶РєР° trusted proxies (IP/CIDR allowlist)
  - РґРѕР±Р°РІР»РµРЅР° РєРѕРЅС„РёРіСѓСЂР°С†РёСЏ С‡РµСЂРµР· env:
    - `PERFORMANCE_RATE_LIMIT_TRUST_FORWARDED_HEADERS`
    - `PERFORMANCE_RATE_LIMIT_TRUSTED_PROXY_IPS`

### 3) РСЃРїСЂР°РІР»РµРЅРёРµ invalidation РєСЌС€Р° combined weather data
- `#38`: `UnifiedWeatherService.invalidate_location_cache()` Р±РѕР»СЊС€Рµ РЅРµ РёСЃРїРѕР»СЊР·СѓРµС‚ РЅРµСЃРѕРІРјРµСЃС‚РёРјС‹Р№ pattern-key
- Р”РѕР±Р°РІР»РµРЅ `MultiLevelCacheManager.invalidate_by_coordinates(lat, lon)`
- РРЅРІР°Р»РёРґР°С†РёСЏ С‚РµРїРµСЂСЊ РёСЃРїРѕР»СЊР·СѓРµС‚ С‚РѕС‚ Р¶Рµ keyspace (rounding/hash), С‡С‚Рѕ Рё `get/set`

### 4) РўРµСЃС‚С‹ Рё РґРѕРєСѓРјРµРЅС‚Р°С†РёСЏ
- Р”РѕР±Р°РІР»РµРЅС‹ СЂРµРіСЂРµСЃСЃРёРѕРЅРЅС‹Рµ С‚РµСЃС‚С‹ РґР»СЏ:
  - path matching РІ `skip_paths`
  - spoofed proxy headers / trusted proxy behavior
  - РїСЂРёРІСЏР·РєРё `RateLimitManager` Рє live middleware instance
  - cache invalidation РїРѕ РєРѕРѕСЂРґРёРЅР°С‚Р°Рј
- РћР±РЅРѕРІР»РµРЅ `README.md`:
  - changelog РґР»СЏ `v0.3.1`
  - РёРЅСЃС‚СЂСѓРєС†РёРё РїРѕ Р±РµР·РѕРїР°СЃРЅРѕР№ РЅР°СЃС‚СЂРѕР№РєРµ reverse proxy РґР»СЏ rate limiting

## Р—Р°РєСЂС‹С‚С‹Рµ issues
- `#33` Code Review: fix rate-limiting bypasses and cache invalidation mismatch
- `#34` Rate limiting: unify RateLimitManager state with live FastAPI middleware instance
- `#35` Security: harden client IP extraction for rate limiting (trusted proxy handling)
- `#36` Rate limiting: fix global bypass caused by skip_paths '/' and substring matching
- `#37` Rate limiting: define and test skip-path matching contract (no substring bypasses)
- `#38` Caching: fix UnifiedWeatherService combined-cache invalidation keyspace mismatch
