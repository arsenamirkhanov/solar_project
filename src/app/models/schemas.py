from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class HourlyInsolation(BaseModel):
    datetime: str
    ghi: Optional[float] = None
    dni: Optional[float] = None
    dhi: Optional[float] = None
    wind_speed: Optional[float] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    humidity: Optional[float] = None
    power: Optional[float] = None  # Мощность солнечных панелей

class ForecastMetrics(BaseModel):
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

class PlotData(BaseModel):
    traces: List[Dict[str, Any]]
    layout: Dict[str, Any]

class ForecastResponse(BaseModel):
    historical_data: List[HourlyInsolation]
    forecast_data: List[HourlyInsolation]
    plot_data: List[PlotData]
    metrics: Optional[ForecastMetrics] = None
    model_type: str = "ml"

class ForecastRequest(BaseModel):
    latitude: float
    longitude: float
    use_ml: Optional[bool] = True  # добавляем булево поле, по умолчанию True
