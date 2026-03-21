# -*- coding: utf-8 -*-
"""Удаление BOM из web_app.py"""

# Читаем файл как bytes
with open('web/web_app.py', 'rb') as f:
    content = f.read()

# Удаляем BOM если есть
if content.startswith(b'\xef\xbb\xbf'):
    print("BOM найден, удаляем...")
    content = content[3:]
else:
    print("BOM не найден")

# Сохраняем без BOM
with open('web/web_app.py', 'wb') as f:
    f.write(content)

print("Готово! Файл сохранен без BOM")
