import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Определяем правильный путь к папке static
current_dir = os.path.dirname(__file__)
static_dir = os.path.join(current_dir, "..", "static")

app = FastAPI(
    title="Solar Insolation Forecast API",
    description="API для прогнозирования инсоляции",
    version="1.0.0"
)

# Подключаем статические файлы (HTML, CSS)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    """Главная страница с веб-интерфейсом."""
    return FileResponse(os.path.join(static_dir, "index.html"))

# Импортируем и подключаем API роутеры
from src.app.api.endpoints import router as api_router

app.include_router(api_router, prefix="/api/v1")