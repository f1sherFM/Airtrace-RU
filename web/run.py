#!/usr/bin/env python3
"""
Простой скрипт запуска для AirTrace RU Web App
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("🌬️ AirTrace RU - Python Web Interface")
    print("=" * 50)
    
    # Проверяем, что мы в правильной директории
    if not Path("web_app.py").exists():
        print("❌ Файл web_app.py не найден!")
        print("Убедитесь, что вы запускаете скрипт из папки web/")
        return
    
    print("🚀 Запуск веб-приложения...")
    print("📡 Сервер будет доступен на http://localhost:3000")
    print("⚡ Нажмите Ctrl+C для остановки")
    print("-" * 50)
    
    try:
        # Запускаем веб-приложение
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "web_app:app", 
            "--host", "0.0.0.0", 
            "--port", "3000",
            "--reload"
        ])
        
        # Ждем немного и открываем браузер
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:3000")
            print("🔗 Браузер открыт автоматически")
        except:
            print("🔗 Откройте браузер и перейдите на http://localhost:3000")
        
        # Ждем завершения
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Остановка сервера...")
        process.terminate()
        print("✅ Сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()