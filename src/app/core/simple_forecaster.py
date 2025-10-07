import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import logging
import traceback
from typing import List, Dict, Any

# Исправляем импорты
try:
    from .weather_client import WeatherDataClient
    from ..models.schemas import ForecastResponse, HourlyInsolation, PlotData  # Добавляем PlotData
except ImportError:
    from app.core.weather_client import WeatherDataClient
    from app.models.schemas import ForecastResponse, HourlyInsolation, PlotData  # Добавляем PlotData

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class SimpleSolarForecaster:
    def __init__(self):
        self.weather_client = WeatherDataClient()
        print("✅ SimpleSolarForecaster инициализирован")

    def generate_forecast(self, latitude: float, longitude: float) -> ForecastResponse:
        """Генерация прогноза с возвратом данных для графика"""
        try:
            print(f"🔍 Запрашиваем данные для координат: {latitude}, {longitude}")

            # Загрузка данных
            weather_df = self.weather_client.fetch_weather_data(latitude, longitude)
            print(f"✅ Данные загружены: {len(weather_df)} строк")

            # Разделение данных
            historical_df = weather_df[weather_df['date'] < pd.Timestamp.now(tz='UTC').normalize()]
            forecast_df = weather_df[weather_df['date'] >= pd.Timestamp.now(tz='UTC').normalize()]

            print(f"📊 Исторические данные: {len(historical_df)} строк")
            print(f"🔮 Прогнозные данные: {len(forecast_df)} строк")

            # Генерация прогноза
            forecast_with_pred = self._generate_simple_forecast(historical_df, forecast_df)

            # Подготовка данных
            historical_data = self._convert_dataframe_to_insolation_list(historical_df.tail(24), False)
            forecast_data = self._convert_dataframe_to_insolation_list(forecast_with_pred, True)

            print("🔄 Подготавливаем данные для графика...")
            plot_data = self._prepare_plot_data(historical_df, forecast_with_pred)
            print("✅ Данные для графика подготовлены")

            # СОЗДАЕМ ОБЪЕКТ PlotData
            plot_data_obj = PlotData(**plot_data)

            return ForecastResponse(
                historical_data=historical_data,
                forecast_data=forecast_data,
                plot_data=plot_data_obj,  # Используем объект PlotData
                model_type="simple"
            )

        except Exception as error:
            print(f"❌ Ошибка: {error}")
            traceback.print_exc()
            raise

    def _generate_simple_forecast(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Генерация прогноза"""
        forecast_with_pred = forecast_df.copy()

        # ДОБАВЛЯЕМ ПРОГНОЗ ДЛЯ СКОРОСТИ ВЕТРА
        for index, row in forecast_with_pred.iterrows():
            hour = row['date'].hour
            similar_hours = historical_df[historical_df['date'].dt.hour == hour].tail(3)

            if len(similar_hours) > 0:
                avg_ghi = similar_hours['shortwave_radiation'].mean()
                avg_wind = similar_hours['wind_speed_10m'].mean()

                forecast_with_pred.at[index, 'ghi_pred'] = max(0, avg_ghi)
                forecast_with_pred.at[index, 'dni_pred'] = max(0, avg_ghi * 0.7)
                forecast_with_pred.at[index, 'dhi_pred'] = max(0, avg_ghi * 0.3)
                forecast_with_pred.at[index, 'wind_speed_pred'] = max(0, avg_wind)
            else:
                last_value = historical_df['shortwave_radiation'].iloc[-1] if len(historical_df) > 0 else 0
                last_wind = historical_df['wind_speed_10m'].iloc[-1] if len(historical_df) > 0 else 0

                forecast_with_pred.at[index, 'ghi_pred'] = max(0, last_value)
                forecast_with_pred.at[index, 'dni_pred'] = max(0, last_value * 0.7)
                forecast_with_pred.at[index, 'dhi_pred'] = max(0, last_value * 0.3)
                forecast_with_pred.at[index, 'wind_speed_pred'] = max(0, last_wind)

        return forecast_with_pred

    def _convert_dataframe_to_insolation_list(self, df: pd.DataFrame, is_forecast: bool) -> List[HourlyInsolation]:
        """Конвертация DataFrame в список"""
        if is_forecast:
            return [
                HourlyInsolation(
                    datetime=row['date'].isoformat(),
                    ghi=row.get('ghi_pred', 0),
                    dni=row.get('dni_pred', 0),
                    dhi=row.get('dhi_pred', 0),
                    wind_speed=row.get('wind_speed_pred', 0),
                    temperature=row.get('temperature_2m', None)
                )
                for _, row in df.iterrows()
            ]
        else:
            return [
                HourlyInsolation(
                    datetime=row['date'].isoformat(),
                    ghi=row['shortwave_radiation'],
                    dni=row['direct_radiation'],
                    dhi=row['diffuse_radiation'],
                    wind_speed=row['wind_speed_10m'],
                    temperature=row['temperature_2m']
                )
                for _, row in df.iterrows()
            ]

    def _prepare_plot_data(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict[str, Any]:
        """Подготовка данных для графика с ветром"""
        try:
            print("🎨 Подготавливаем данные для графика с ветром...")

            historical_last_24h = historical_df.tail(24)

            # График GHI
            ghi_trace = {
                "x": historical_last_24h['date'].dt.strftime('%Y-%m-%d %H:%M').tolist() +
                     forecast_df['date'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
                "y": historical_last_24h['shortwave_radiation'].tolist() +
                     forecast_df['ghi_pred'].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "GHI",
                "line": {"color": "#3498db", "width": 3}
            }

            # График скорости ветра
            wind_trace = {
                "x": historical_last_24h['date'].dt.strftime('%Y-%m-%d %H:%M').tolist() +
                     forecast_df['date'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
                "y": historical_last_24h['wind_speed_10m'].tolist() +
                     forecast_df['wind_speed_pred'].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "Скорость ветра",
                "line": {"color": "#27ae60", "width": 3},
                "yaxis": "y2"
            }

            # Layout с двумя осями Y
            layout = {
                "title": "Прогноз: Инсоляция и Скорость Ветра",
                "xaxis": {"title": "Время"},
                "yaxis": {
                    "title": "Инсоляция (Вт/м²)",
                    "titlefont": {"color": "#3498db"},
                    "tickfont": {"color": "#3498db"}
                },
                "yaxis2": {
                    "title": "Скорость ветра (м/с)",
                    "titlefont": {"color": "#27ae60"},
                    "tickfont": {"color": "#27ae60"},
                    "overlaying": "y",
                    "side": "right"
                },
                "height": 500,
                "showlegend": True,
                "plot_bgcolor": "white",
                "paper_bgcolor": "white"
            }

            return {"traces": [ghi_trace, wind_trace], "layout": layout}

        except Exception as error:
            print(f"❌ Ошибка подготовки графика: {error}")
            return {"traces": [], "layout": {}}