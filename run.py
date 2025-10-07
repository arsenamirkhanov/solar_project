# run.py - Запускайте этот файл для старта приложения
import sys
import os

# Добавляем корень проекта в путь Python, чтобы он мог найти модуль 'src'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app.main import app  # теперь абсолютный импорт будет работать

import uvicorn
import webbrowser
import threading
import time

def open_browser():
    """Открывает браузер после запуска сервера."""
    time.sleep(3)  # Даем серверу время на запуск
    try:
        webbrowser.open("http://localhost:8000")
        print("✅ Браузер открыт по адресу http://localhost:8000")
    except Exception as e:
        print(f"❌ Не удалось открыть браузер: {e}")

if __name__ == "__main__":
    # Запускаем открытие браузера в фоне
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    print("Запуск сервера Solar Forecast...")
    print(">>> СЕРВЕР БУДЕТ ДОСТУПЕН ПО АДРЕСУ: http://localhost:8000 <<<")
    print(">>> БРАУЗЕР ОТКРОЕТСЯ АВТОМАТИЧЕСКИ <<<")
    print("Для остановки нажмите Ctrl+C")

    # Запускаем сервер с автоматической перезагрузкой (удобно для dev)
    uvicorn.run(
        "src.app.main:app",  # строковый путь, чтобы uvicorn видел абсолютный импорт
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # авто-перезапуск при изменениях кода
    )
