from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from ..models.schemas import ForecastRequest, ForecastResponse
from ..core.simple_forecaster import SimpleSolarForecaster
from ..core.ml_forecaster import MLSolarForecaster
from .connections import active_connections  # подключаем список активных WS

router = APIRouter()

# Создаем инстансы прогнозеров
simple_forecaster = SimpleSolarForecaster()
ml_forecaster = MLSolarForecaster()


# ------------------- /forecast -------------------
@router.post("/forecast", response_model=ForecastResponse)
async def get_solar_forecast(request: ForecastRequest):
    try:
        use_ml = getattr(request, 'use_ml', True)
        print(f"📍 Запрос: {request.latitude}, {request.longitude}, ML: {use_ml}")

        if use_ml:
            forecast = ml_forecaster.generate_forecast(request.latitude, request.longitude)
        else:
            forecast = simple_forecaster.generate_forecast(request.latitude, request.longitude)

        print("✅ Прогноз успешно сгенерирован")
        return forecast

    except Exception as e:
        error_msg = f"Ошибка при получении прогноза: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


# ------------------- WebSocket /ws/solar -------------------
@router.websocket("/ws/solar")
async def solar_notifications(websocket: WebSocket):
    # Разрешаем подключение с любого origin
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # пока игнорируем входящие сообщения
    except WebSocketDisconnect:
        active_connections.remove(websocket)


# ------------------- /forecast/export -------------------
@router.post("/forecast/export")
async def export_forecast(data: dict):
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    if latitude is None or longitude is None:
        raise HTTPException(status_code=400, detail="Нужны координаты")

    try:
        forecaster = MLSolarForecaster()
        forecast = forecaster.generate_forecast(latitude, longitude)
        file_path = forecaster.export_to_excel(forecast)

        return FileResponse(
            path=file_path,
            filename=f"solar_forecast_{latitude}_{longitude}.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при экспорте Excel: {str(e)}")
