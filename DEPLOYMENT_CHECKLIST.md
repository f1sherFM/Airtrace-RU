# 🚀 Чеклист деплоя критических исправлений

## ✅ Предварительная проверка (перед деплоем)

### 1. Проверка кода
- [x] Все исправления внесены
- [x] Код прошел review
- [x] Нет конфликтов в git

```bash
git status
git diff main
```

### 2. Тестирование
- [x] Все unit тесты прошли
- [x] Интеграционные тесты прошли
- [ ] Нагрузочное тестирование выполнено

```bash
# Запустить все тесты
pytest tests/ -v

# Запустить тесты критических исправлений
pytest tests/test_critical_fixes.py -v

# Проверить покрытие
pytest tests/ --cov=. --cov-report=html
```

### 3. Конфигурация
- [ ] API ключи проверены и валидны
- [ ] Переменные окружения настроены
- [ ] Redis доступен и работает

```bash
# Проверить .env файл
cat .env | grep -E "WEATHER_API_KEY|REDIS"

# Проверить Redis
redis-cli ping

# Проверить длину API ключа
echo $WEATHER_API_KEY | wc -c
```

---

## 🔧 Процесс деплоя

### Шаг 1: Backup
```bash
# Создать backup текущей версии
./backup.sh

# Или вручную
cp -r . ../airtrace_backup_$(date +%Y%m%d_%H%M%S)
```

### Шаг 2: Обновление кода
```bash
# Получить последние изменения
git pull origin main

# Или из конкретной ветки
git checkout feature/critical-fixes
git pull
```

### Шаг 3: Обновление зависимостей
```bash
# Обновить Python зависимости (если нужно)
pip install -r requirements.txt --upgrade

# Проверить установленные версии
pip list | grep -E "fastapi|redis|httpx"
```

### Шаг 4: Проверка конфигурации
```bash
# Проверить .env файл
cat .env

# Убедиться что ключи валидны
python -c "
from config import WeatherAPIConfig
try:
    config = WeatherAPIConfig()
    print('✅ Configuration valid')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

### Шаг 5: Запуск тестов в production окружении
```bash
# Запустить быстрые тесты
pytest tests/test_basic.py -v

# Запустить тесты критических исправлений
pytest tests/test_critical_fixes.py -v
```

### Шаг 6: Остановка старой версии
```bash
# Docker
docker-compose down

# Или systemd
sudo systemctl stop airtrace

# Или вручную
pkill -f "python.*start_app.py"
```

### Шаг 7: Запуск новой версии
```bash
# Docker
docker-compose up -d

# Или systemd
sudo systemctl start airtrace

# Или вручную
nohup python start_app.py > logs/app.log 2>&1 &
```

### Шаг 8: Проверка запуска
```bash
# Подождать 10 секунд
sleep 10

# Проверить что процесс запущен
ps aux | grep python

# Проверить логи
tail -f logs/api.log
```

---

## ✅ Проверка после деплоя

### 1. Health Check
```bash
# Базовый health check
curl http://localhost:8000/health

# Ожидаемый результат:
# {
#   "status": "healthy",
#   "services": {
#     "api": "healthy",
#     "external_api": "healthy",
#     "cache": "L1:enabled L2:healthy L3:enabled",
#     ...
#   }
# }
```

### 2. Проверка API
```bash
# Тестовый запрос текущих данных
curl "http://localhost:8000/weather/current?lat=55.7558&lon=37.6176"

# Проверка прогноза
curl "http://localhost:8000/weather/forecast?lat=55.7558&lon=37.6176"
```

### 3. Проверка метрик
```bash
# Общие метрики
curl http://localhost:8000/metrics

# Метрики кэша
curl http://localhost:8000/metrics | grep -E "cache|eviction"

# Метрики Redis
curl http://localhost:8000/metrics | grep redis
```

### 4. Проверка логов
```bash
# Проверить на ошибки
tail -100 logs/api.log | grep -i error

# Проверить на warning
tail -100 logs/api.log | grep -i warning

# Проверить успешные запросы
tail -100 logs/api.log | grep "200 OK"
```

### 5. Мониторинг памяти
```bash
# Использование памяти процессом
ps aux | grep python | awk '{print $6}'

# Системная память
free -h

# Проверить через 5 минут работы
sleep 300
ps aux | grep python | awk '{print $6}'
```

### 6. Проверка Redis
```bash
# Подключение к Redis
redis-cli ping

# Проверить ключи кэша
redis-cli keys "airtrace:v1:*" | wc -l

# Проверить память Redis
redis-cli info memory | grep used_memory_human
```

---

## 🔍 Smoke Tests

### Автоматические smoke tests
```bash
# Создать файл smoke_tests.sh
cat > smoke_tests.sh << 'EOF'
#!/bin/bash

BASE_URL="http://localhost:8000"
FAILED=0

echo "🧪 Running smoke tests..."

# Test 1: Health check
echo "Test 1: Health check"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/health)
if [ $STATUS -eq 200 ]; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed (HTTP $STATUS)"
    FAILED=$((FAILED + 1))
fi

# Test 2: Current weather
echo "Test 2: Current weather API"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/weather/current?lat=55.7558&lon=37.6176")
if [ $STATUS -eq 200 ]; then
    echo "✅ Current weather API passed"
else
    echo "❌ Current weather API failed (HTTP $STATUS)"
    FAILED=$((FAILED + 1))
fi

# Test 3: Forecast
echo "Test 3: Forecast API"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/weather/forecast?lat=55.7558&lon=37.6176")
if [ $STATUS -eq 200 ]; then
    echo "✅ Forecast API passed"
else
    echo "❌ Forecast API failed (HTTP $STATUS)"
    FAILED=$((FAILED + 1))
fi

# Test 4: Metrics
echo "Test 4: Metrics endpoint"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/metrics)
if [ $STATUS -eq 200 ]; then
    echo "✅ Metrics endpoint passed"
else
    echo "❌ Metrics endpoint failed (HTTP $STATUS)"
    FAILED=$((FAILED + 1))
fi

# Summary
echo ""
if [ $FAILED -eq 0 ]; then
    echo "✅ All smoke tests passed!"
    exit 0
else
    echo "❌ $FAILED smoke test(s) failed"
    exit 1
fi
EOF

chmod +x smoke_tests.sh
./smoke_tests.sh
```

---

## 🚨 Rollback процедура

### Если что-то пошло не так:

#### Быстрый rollback (Docker)
```bash
# Остановить новую версию
docker-compose down

# Восстановить из backup
cd ../airtrace_backup_YYYYMMDD_HHMMSS
docker-compose up -d
```

#### Rollback через git
```bash
# Остановить приложение
sudo systemctl stop airtrace

# Откатить изменения
git log --oneline -10  # Найти нужный commit
git checkout <previous-commit-hash>

# Запустить старую версию
sudo systemctl start airtrace
```

#### Ручной rollback
```bash
# Остановить приложение
pkill -f "python.*start_app.py"

# Восстановить файлы из backup
cp -r ../airtrace_backup_YYYYMMDD_HHMMSS/* .

# Запустить
nohup python start_app.py > logs/app.log 2>&1 &
```

---

## 📊 Мониторинг после деплоя

### Первые 15 минут
- [ ] Проверять логи каждые 2 минуты
- [ ] Мониторить использование памяти
- [ ] Проверять response time API
- [ ] Следить за error rate

```bash
# Скрипт мониторинга
watch -n 10 '
echo "=== Memory Usage ==="
ps aux | grep python | awk "{print \$6}"
echo ""
echo "=== Recent Errors ==="
tail -20 logs/api.log | grep -i error
echo ""
echo "=== Request Count ==="
curl -s http://localhost:8000/metrics | grep total_requests
'
```

### Первый час
- [ ] Проверить все основные эндпоинты
- [ ] Убедиться что кэш работает корректно
- [ ] Проверить Redis connections
- [ ] Мониторить eviction rate

### Первые 24 часа
- [ ] Проверять метрики каждые 4 часа
- [ ] Мониторить использование памяти (не должно расти)
- [ ] Проверять логи на warning/errors
- [ ] Собрать feedback от пользователей

---

## 📝 Чеклист завершения

### После успешного деплоя:
- [ ] Все smoke tests прошли
- [ ] Health check возвращает "healthy"
- [ ] API отвечает корректно
- [ ] Логи без критических ошибок
- [ ] Память стабильна
- [ ] Redis работает нормально
- [ ] Документация обновлена
- [ ] Команда уведомлена
- [ ] Мониторинг настроен
- [ ] Backup создан

### Документация:
- [ ] Обновить CHANGELOG.md
- [ ] Обновить версию в pyproject.toml
- [ ] Создать git tag для релиза
- [ ] Обновить README.md (если нужно)

```bash
# Создать git tag
git tag -a v2.1.0 -m "Critical security and performance fixes"
git push origin v2.1.0

# Обновить CHANGELOG
echo "## v2.1.0 - $(date +%Y-%m-%d)" >> CHANGELOG.md
echo "- Fixed race condition in cache" >> CHANGELOG.md
echo "- Added Redis operation timeouts" >> CHANGELOG.md
echo "- Implemented memory growth limitation" >> CHANGELOG.md
echo "- Added API key validation" >> CHANGELOG.md
```

---

## 🎯 Критерии успеха

Деплой считается успешным если:
- ✅ Все тесты прошли (15/15)
- ✅ Health check возвращает "healthy"
- ✅ API response time < 2 секунды
- ✅ Error rate < 1%
- ✅ Память стабильна (не растет)
- ✅ Нет критических ошибок в логах
- ✅ Redis timeout работает корректно
- ✅ Cache eviction работает правильно

---

## 📞 Контакты для экстренных случаев

- **Техническая поддержка:** [ваш email]
- **DevOps:** [devops email]
- **Документация:** CRITICAL_FIXES.md
- **Логи:** logs/api.log

---

**Версия чеклиста:** 1.0  
**Дата:** 2026-02-02  
**Для версии:** v2.1.0
