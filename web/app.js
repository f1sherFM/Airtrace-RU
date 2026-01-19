// AirTrace RU Web Interface
// Интерактивный интерфейс для мониторинга качества воздуха

class AirTraceApp {
    constructor() {
        this.apiBase = 'http://127.0.0.1:8000';
        this.currentCoords = null;
        this.currentCity = null;
        this.refreshInterval = null;
        
        this.cityNames = {
            '55.7558,37.6176': 'Москва',
            '59.9311,30.3609': 'Санкт-Петербург', 
            '53.4069,58.9794': 'Магнитогорск',
            '56.8431,60.6454': 'Екатеринбург',
            '55.0084,82.9357': 'Новосибирск',
            '55.1644,61.4368': 'Челябинск',
            '56.3269,44.0075': 'Нижний Новгород',
            '53.2001,50.15': 'Самара'
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.checkApiHealth();
        
        // Автоматически загружаем данные для Магнитогорска
        setTimeout(() => {
            document.getElementById('city-select').value = '53.4069,58.9794';
            this.onCityChange();
        }, 1000);
    }
    
    setupEventListeners() {
        const citySelect = document.getElementById('city-select');
        const refreshBtn = document.getElementById('refresh-btn');
        
        citySelect.addEventListener('change', () => this.onCityChange());
        refreshBtn.addEventListener('click', () => this.loadAirQualityData());
        
        // Автообновление каждые 5 минут
        this.refreshInterval = setInterval(() => {
            if (this.currentCoords) {
                this.loadAirQualityData();
            }
        }, 5 * 60 * 1000);
    }
    
    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.updateStatus('healthy', 'API активен');
            } else {
                this.updateStatus('warning', 'API частично доступен');
            }
        } catch (error) {
            this.updateStatus('error', 'API недоступен');
            console.error('Health check failed:', error);
        }
    }
    
    updateStatus(status, text) {
        const indicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        
        indicator.className = 'w-2 h-2 rounded-full animate-pulse';
        
        switch (status) {
            case 'healthy':
                indicator.classList.add('bg-green-500');
                break;
            case 'warning':
                indicator.classList.add('bg-yellow-500');
                break;
            case 'error':
                indicator.classList.add('bg-red-500');
                break;
            default:
                indicator.classList.add('bg-gray-500');
        }
        
        statusText.textContent = text;
    }
    
    onCityChange() {
        const citySelect = document.getElementById('city-select');
        const coords = citySelect.value;
        
        if (!coords) {
            this.showLoadingState();
            return;
        }
        
        const [lat, lon] = coords.split(',');
        this.currentCoords = { lat: parseFloat(lat), lon: parseFloat(lon) };
        this.currentCity = this.cityNames[coords];
        
        // Обновляем отображение города
        document.getElementById('location-display').textContent = this.currentCity;
        
        this.loadAirQualityData();
    }
    
    async loadAirQualityData() {
        if (!this.currentCoords) return;
        
        this.updateStatus('loading', 'Загрузка данных...');
        
        try {
            // Загружаем текущие данные
            const currentResponse = await fetch(
                `${this.apiBase}/weather/current?lat=${this.currentCoords.lat}&lon=${this.currentCoords.lon}`
            );
            
            if (!currentResponse.ok) {
                throw new Error(`HTTP ${currentResponse.status}: ${currentResponse.statusText}`);
            }
            
            const currentData = await currentResponse.json();
            
            // Загружаем прогноз
            const forecastResponse = await fetch(
                `${this.apiBase}/weather/forecast?lat=${this.currentCoords.lat}&lon=${this.currentCoords.lon}`
            );
            
            let forecastData = [];
            if (forecastResponse.ok) {
                forecastData = await forecastResponse.json();
            }
            
            this.displayAirQualityData(currentData, forecastData);
            this.updateStatus('healthy', 'Данные обновлены');
            
        } catch (error) {
            console.error('Failed to load air quality data:', error);
            this.showErrorState(error.message);
            this.updateStatus('error', 'Ошибка загрузки');
        }
    }
    
    displayAirQualityData(data, forecast = []) {
        // Скрываем состояния загрузки и ошибки
        this.hideLoadingState();
        this.hideErrorState();
        
        // Показываем основные блоки
        this.showMainContent();
        
        // Обновляем фон в зависимости от AQI
        this.updateBackgroundByAQI(data.aqi.value);
        
        // Основная карточка AQI
        document.getElementById('aqi-value').textContent = data.aqi.value;
        document.getElementById('aqi-category').textContent = data.aqi.category;
        document.getElementById('aqi-description').textContent = data.aqi.description;
        document.getElementById('recommendations').textContent = data.recommendations;
        
        // Цвет категории AQI
        const categoryElement = document.getElementById('aqi-category');
        categoryElement.style.color = data.aqi.color;
        
        // НМУ риск
        this.updateNMURisk(data.nmu_risk);
        
        // Координаты и время
        document.getElementById('latitude').textContent = `${data.location.latitude}°`;
        document.getElementById('longitude').textContent = `${data.location.longitude}°`;
        document.getElementById('last-updated').textContent = this.formatTime(data.timestamp);
        
        // Загрязнители
        this.updatePollutants(data.pollutants);
        
        // Предупреждения о здоровье
        this.updateHealthWarnings(data.health_warnings);
        
        // Прогноз
        if (forecast.length > 0) {
            this.updateForecast(forecast);
        }
    }
    
    updateBackgroundByAQI(aqi) {
        const body = document.getElementById('body');
        body.className = 'min-h-screen text-white p-4 md:p-8';
        
        if (aqi <= 50) {
            body.classList.add('aqi-good');
        } else if (aqi <= 100) {
            body.classList.add('aqi-moderate');
        } else if (aqi <= 150) {
            body.classList.add('aqi-unhealthy-sensitive');
        } else if (aqi <= 200) {
            body.classList.add('aqi-unhealthy');
        } else if (aqi <= 300) {
            body.classList.add('aqi-very-unhealthy');
        } else {
            body.classList.add('aqi-hazardous');
        }
    }
    
    updateNMURisk(risk) {
        const nmuCard = document.getElementById('nmu-card');
        const nmuIcon = document.getElementById('nmu-icon');
        const nmuLevel = document.getElementById('nmu-level');
        const nmuDescription = document.getElementById('nmu-description');
        
        // Удаляем предыдущие классы границы
        nmuCard.className = 'glass rounded-3xl p-6 border-l-4';
        
        switch (risk) {
            case 'low':
                nmuCard.classList.add('border-l-green-400');
                nmuIcon.setAttribute('data-lucide', 'shield-check');
                nmuIcon.className = 'text-green-400 w-8 h-8';
                nmuLevel.textContent = 'Низкий риск';
                nmuDescription.textContent = 'Режим «Черного неба» не ожидается';
                break;
            case 'medium':
                nmuCard.classList.add('border-l-yellow-400');
                nmuIcon.setAttribute('data-lucide', 'shield');
                nmuIcon.className = 'text-yellow-400 w-8 h-8';
                nmuLevel.textContent = 'Умеренный риск';
                nmuDescription.textContent = 'Следите за изменениями качества воздуха';
                break;
            case 'high':
                nmuCard.classList.add('border-l-orange-400');
                nmuIcon.setAttribute('data-lucide', 'shield-alert');
                nmuIcon.className = 'text-orange-400 w-8 h-8';
                nmuLevel.textContent = 'Высокий риск';
                nmuDescription.textContent = 'Возможны неблагоприятные условия';
                break;
            case 'critical':
                nmuCard.classList.add('border-l-red-500');
                nmuIcon.setAttribute('data-lucide', 'shield-x');
                nmuIcon.className = 'text-red-500 w-8 h-8';
                nmuLevel.textContent = 'КРИТИЧЕСКИЙ';
                nmuDescription.textContent = 'Режим «Черного неба» активен!';
                break;
            default:
                nmuCard.classList.add('border-l-gray-400');
                nmuIcon.setAttribute('data-lucide', 'shield-question');
                nmuIcon.className = 'text-gray-400 w-8 h-8';
                nmuLevel.textContent = 'Неизвестно';
                nmuDescription.textContent = 'Данные недоступны';
        }
        
        // Обновляем иконку
        lucide.createIcons();
    }
    
    updatePollutants(pollutants) {
        document.getElementById('pm25-value').textContent = pollutants.pm2_5?.toFixed(1) || '--';
        document.getElementById('pm10-value').textContent = pollutants.pm10?.toFixed(1) || '--';
        document.getElementById('no2-value').textContent = pollutants.no2?.toFixed(1) || '--';
        document.getElementById('so2-value').textContent = pollutants.so2?.toFixed(1) || '--';
        document.getElementById('o3-value').textContent = pollutants.o3?.toFixed(1) || '--';
    }
    
    updateHealthWarnings(warnings) {
        const warningsContainer = document.getElementById('health-warnings');
        const warningsList = document.getElementById('warnings-list');
        
        if (!warnings || warnings.length === 0) {
            warningsContainer.classList.add('hidden');
            return;
        }
        
        warningsContainer.classList.remove('hidden');
        warningsList.innerHTML = '';
        
        warnings.forEach(warning => {
            const warningElement = document.createElement('div');
            warningElement.className = 'flex items-start gap-2 p-2 bg-red-500/20 rounded-lg border border-red-500/30';
            warningElement.innerHTML = `
                <i data-lucide="alert-triangle" class="text-red-400 w-4 h-4 mt-0.5 shrink-0"></i>
                <span class="text-sm">${warning}</span>
            `;
            warningsList.appendChild(warningElement);
        });
        
        lucide.createIcons();
    }
    
    updateForecast(forecast) {
        const forecastSection = document.getElementById('forecast-section');
        const forecastGrid = document.getElementById('forecast-grid');
        
        forecastSection.classList.remove('hidden');
        forecastGrid.innerHTML = '';
        
        // Показываем первые 8 часов прогноза
        const hours = forecast.slice(0, 8);
        
        hours.forEach(hour => {
            const time = new Date(hour.timestamp);
            const card = document.createElement('div');
            card.className = 'forecast-card glass rounded-2xl p-4 text-center';
            
            card.innerHTML = `
                <p class="text-xs opacity-60 mb-2">${time.getHours()}:00</p>
                <div class="text-2xl font-bold mb-1" style="color: ${hour.aqi.color}">${hour.aqi.value}</div>
                <p class="text-xs opacity-80">${hour.aqi.category}</p>
                <div class="mt-2 text-xs opacity-60">
                    <div>PM2.5: ${hour.pollutants.pm2_5?.toFixed(1) || '--'}</div>
                </div>
            `;
            
            forecastGrid.appendChild(card);
        });
    }
    
    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('ru-RU', { 
            hour: '2-digit', 
            minute: '2-digit',
            timeZone: 'Europe/Moscow'
        });
    }
    
    showLoadingState() {
        document.getElementById('loading-state').classList.remove('hidden');
        document.getElementById('error-state').classList.add('hidden');
        this.hideMainContent();
    }
    
    hideLoadingState() {
        document.getElementById('loading-state').classList.add('hidden');
    }
    
    showErrorState(message) {
        document.getElementById('error-state').classList.remove('hidden');
        document.getElementById('error-message').textContent = message;
        document.getElementById('loading-state').classList.add('hidden');
        this.hideMainContent();
    }
    
    hideErrorState() {
        document.getElementById('error-state').classList.add('hidden');
    }
    
    showMainContent() {
        document.getElementById('aqi-card').classList.remove('hidden');
        document.getElementById('side-info').classList.remove('hidden');
        document.getElementById('pollutants-grid').classList.remove('hidden');
    }
    
    hideMainContent() {
        document.getElementById('aqi-card').classList.add('hidden');
        document.getElementById('side-info').classList.add('hidden');
        document.getElementById('pollutants-grid').classList.add('hidden');
        document.getElementById('forecast-section').classList.add('hidden');
    }
}

// Инициализация приложения
document.addEventListener('DOMContentLoaded', () => {
    window.airTraceApp = new AirTraceApp();
});

// Глобальная функция для повторной загрузки данных (используется в HTML)
function loadAirQualityData() {
    if (window.airTraceApp) {
        window.airTraceApp.loadAirQualityData();
    }
}