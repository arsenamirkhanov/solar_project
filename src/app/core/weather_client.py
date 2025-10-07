import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class WeatherDataClient:
    """
    Клиент для загрузки метеорологических данных через Open-Meteo API.
    Теперь получаем больше параметров для ML.
    """

    def __init__(self):
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

    def fetch_weather_data(self, latitude: float, longitude: float) -> pd.DataFrame:
        """Загружает расширенные метеорологические данные"""
        try:
            print(f"🌤️ Загружаем расширенные данные для {latitude}, {longitude}")

            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": [
                    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
                    "temperature_2m", "relative_humidity_2m", "pressure_msl",
                    "wind_speed_10m", "wind_direction_10m", "cloud_cover"
                ],
                "past_days": 90,
                "forecast_days": 2,  # Увеличиваем для больше данных для ML
                "timezone": "auto"
            }

            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]

            hourly = response.Hourly()
            hourly_data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}

            # Извлекаем все параметры
            parameters = [
                "shortwave_radiation", "direct_radiation", "diffuse_radiation",
                "temperature_2m", "relative_humidity_2m", "pressure_msl",
                "wind_speed_10m", "wind_direction_10m", "cloud_cover"
            ]

            for i, param in enumerate(parameters):
                hourly_data[param] = hourly.Variables(i).ValuesAsNumpy()

            df = pd.DataFrame(data=hourly_data)
            print(f"✅ Расширенные данные загружены: {len(df)} строк, {len(df.columns)} параметров")
            return df

        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            raise