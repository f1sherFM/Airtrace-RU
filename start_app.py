#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт запуска полного приложения AirTrace RU
Запускает API сервер и веб-интерфейс
"""

import subprocess
import sys
import time
import webbrowser
import signal
import os
from pathlib import Path


# Устанавливаем правильную кодировку для Windows
if sys.platform == 'win32':
    # Устанавливаем кодовую страницу UTF-8 для консоли Windows
    os.system('chcp 65001 > nul')
    
# Устанавливаем переменные окружения для правильной работы с кириллицей
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# Перенастраиваем stdout/stderr на UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

class AirTraceApp:
    def __init__(self):
        self.api_process = None
        self.web_process = None
        
    def print_header(self):
        print("🌬️ AirTrace RU - Полный запуск приложения")
        print("=" * 60)
        print("🐍 Backend API + Python Web Interface")
        print("🔒 Privacy-first архитектура")
        print("🇷🇺 Российские стандарты ПДК")
        print("=" * 60)
    
    def start_api_server(self):
        """Запуск API сервера"""
        print("🔧 Запуск API сервера (порт 8000)...")
        
        try:
            # Создаем окружение с правильной кодировкой
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            self.api_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "main:app",
                "--host", "127.0.0.1",
                "--port", "8000"
            ], env=env)
            
            # Ждем запуска
            print("⏳ Ожидание запуска API сервера...")
            time.sleep(5)
            
            if self.api_process.poll() is None:
                print("✅ API сервер запущен успешно")
                return True
            else:
                print("❌ API сервер завершился с ошибкой")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка запуска API: {e}")
            return False
    
    def start_web_server(self):
        """Запуск веб-сервера"""
        print("🌐 Запуск веб-интерфейса (порт 3000)...")
        
        web_dir = Path("web")
        if not web_dir.exists():
            print("❌ Папка web/ не найдена!")
            return False
        
        try:
            # Создаем окружение с правильной кодировкой
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            self.web_process = subprocess.Popen([
                sys.executable, "web_app.py"
            ], cwd=web_dir, env=env)
            
            # Ждем запуска
            print("⏳ Ожидание запуска веб-интерфейса...")
            time.sleep(5)
            
            if self.web_process.poll() is None:
                print("✅ Веб-интерфейс запущен успешно")
                return True
            else:
                print("❌ Веб-интерфейс завершился с ошибкой")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка запуска веб-интерфейса: {e}")
            return False
    
    def open_browser(self):
        """Открытие браузера"""
        print("🔗 Открываем браузер...")
        try:
            time.sleep(2)
            webbrowser.open("http://localhost:3000")
            print("✅ Браузер открыт")
        except Exception as e:
            print(f"⚠️ Не удалось открыть браузер: {e}")
            print("🔗 Откройте вручную: http://localhost:3000")
    
    def monitor(self):
        """Мониторинг приложения"""
        print("\n📊 Приложение запущено!")
        print("📡 API: http://localhost:8000")
        print("🌐 Web: http://localhost:3000")
        print("📖 Docs: http://localhost:8000/docs")
        print("\n⚡ Нажмите Ctrl+C для остановки")
        print("-" * 60)
        
        try:
            while True:
                # Проверяем процессы
                api_status = "✅ Работает" if self.api_process and self.api_process.poll() is None else "❌ Остановлен"
                web_status = "✅ Работает" if self.web_process and self.web_process.poll() is None else "❌ Остановлен"
                
                print(f"[{time.strftime('%H:%M:%S')}] API: {api_status} | Web: {web_status}")
                time.sleep(10)
                
        except KeyboardInterrupt:
            self.cleanup()
    
    def cleanup(self):
        """Остановка всех процессов"""
        print("\n🛑 Остановка приложения...")
        
        if self.api_process:
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=5)
                print("✅ API сервер остановлен")
            except:
                self.api_process.kill()
                print("🔥 API сервер принудительно остановлен")
        
        if self.web_process:
            try:
                self.web_process.terminate()
                self.web_process.wait(timeout=5)
                print("✅ Веб-интерфейс остановлен")
            except:
                self.web_process.kill()
                print("🔥 Веб-интерфейс принудительно остановлен")
        
        print("🙏 Спасибо за использование AirTrace RU!")
    
    def run(self):
        """Главная функция запуска"""
        self.print_header()
        
        # Проверяем файлы
        if not Path("main.py").exists():
            print("❌ Файл main.py не найден!")
            return
        
        if not Path("web/web_app.py").exists():
            print("❌ Файл web/web_app.py не найден!")
            return
        
        # Запускаем серверы
        if not self.start_api_server():
            return
        
        if not self.start_web_server():
            self.cleanup()
            return
        
        # Открываем браузер
        self.open_browser()
        
        # Мониторинг
        self.monitor()

def main():
    app = AirTraceApp()
    
    # Обработка сигналов
    def signal_handler(signum, frame):
        app.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        app.run()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        app.cleanup()

if __name__ == "__main__":
    main()