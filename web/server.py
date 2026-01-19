#!/usr/bin/env python3
"""
Простой HTTP сервер для веб-интерфейса AirTrace RU
Обслуживает статические файлы с поддержкой CORS
"""

import http.server
import socketserver
import os
from urllib.parse import urlparse

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP обработчик с поддержкой CORS для работы с API"""
    
    def end_headers(self):
        # Добавляем CORS заголовки
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # Обработка preflight запросов
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        # Логирование запросов
        print(f"[{self.address_string()}] {format % args}")

def start_server(port=3000, directory=None):
    """Запуск HTTP сервера"""
    
    if directory:
        os.chdir(directory)
    
    handler = CORSHTTPRequestHandler
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"🌐 AirTrace RU Web Interface")
        print(f"📡 Сервер запущен на http://localhost:{port}")
        print(f"📁 Обслуживает файлы из: {os.getcwd()}")
        print(f"🔗 API Backend: http://localhost:8000")
        print(f"⚡ Нажмите Ctrl+C для остановки")
        print("-" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Сервер остановлен")

if __name__ == "__main__":
    import sys
    
    port = 3000
    directory = "."
    
    # Парсинг аргументов командной строки
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Ошибка: Порт должен быть числом")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        directory = sys.argv[2]
        if not os.path.exists(directory):
            print(f"Ошибка: Директория {directory} не существует")
            sys.exit(1)
    
    start_server(port, directory)