import pandas as pd
import numpy as np
import warnings
import logging
import traceback
from typing import List, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import asyncio
import os

# Импорты проекта
try:
    from .weather_client import WeatherDataClient
    from ..models.schemas import ForecastResponse, HourlyInsolation, ForecastMetrics, PlotData
except ImportError:
    from ..core.weather_client import WeatherDataClient
    from ..models.schemas import ForecastResponse, HourlyInsolation, ForecastMetrics, PlotData

# Активные WebSocket подключения
try:
    from ..api.connections import active_connections
except ImportError:
    active_connections = []

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Настройки панели
PANEL_AREA = 1.6        # м², площадь одной панели
PANEL_EFFICIENCY = 0.18 # КПД панели
NUM_PANELS = 5          # Количество панелей в системе

# Пороги уведомлений
POWER_THRESHOLD_HIGH = 50   # Вт, высокий уровень мощности
POWER_THRESHOLD_LOW = 10    # Вт, низкий уровень мощности
POWER_DELTA_THRESHOLD = 50  # Вт, резкий скачок

# Словарь последних уведомлений
NOTIFIED_EVENTS = {}

class MLSolarForecaster:
    def __init__(self):
        self.weather_client = WeatherDataClient()
        self.models = {}
        self.scalers = {}
        self.features = [
            'hour', 'day_of_year', 'day_of_week', 'month',
            'temperature_2m', 'relative_humidity_2m', 'pressure_msl',
            'wind_speed_10m', 'wind_direction_10m', 'cloud_cover',
            'historical_ghi_1h', 'historical_ghi_2h', 'historical_ghi_3h'
        ]
        self.targets = [
            'ghi', 'dni', 'dhi', 'wind_speed_10m',
            'temperature_2m', 'pressure_msl', 'relative_humidity_2m'
        ]
        print("✅ MLSolarForecaster инициализирован")

    def generate_forecast(self, latitude: float, longitude: float) -> ForecastResponse:
        try:
            print(f"🧠 Запускаем ML-прогноз для координат: {latitude}, {longitude}")

            # Загружаем данные
            weather_df = self.weather_client.fetch_weather_data(latitude, longitude)
            processed_df = self._prepare_features(weather_df)

            # Разделяем исторические и прогнозные данные
            historical_df = processed_df[processed_df['date'] < pd.Timestamp.now(tz='UTC').normalize()]
            forecast_df = processed_df[processed_df['date'] >= pd.Timestamp.now(tz='UTC').normalize()]

            # Обучаем модели
            self._train_models(historical_df)

            # Прогнозируем
            ml_predictions = self._predict_ml(forecast_df)

            # Рассчёт мощности для всей системы
            ml_predictions['power'] = [
                ghi * PANEL_AREA * PANEL_EFFICIENCY * NUM_PANELS for ghi in ml_predictions.get('ghi', [])
            ]

            # Отправка уведомлений через WebSocket
            asyncio.create_task(self._notify_power(ml_predictions['power']))

            # Подготовка данных для фронтенда
            historical_data = self._prepare_historical_data(historical_df)
            forecast_data = self._prepare_forecast_data(forecast_df, ml_predictions)
            metrics = self._calculate_metrics(historical_df, ml_predictions)
            plot_data = self._prepare_ml_plot_data(historical_df, forecast_df, ml_predictions)

            return ForecastResponse(
                historical_data=historical_data,
                forecast_data=forecast_data,
                plot_data=plot_data,
                metrics=metrics,
                model_type="ml"
            )

        except Exception as error:
            print(f"❌ Ошибка ML-прогноза: {error}")
            traceback.print_exc()
            raise

    async def _notify_power(self, power_list: List[float]):
        """Отправка уведомлений через WebSocket при превышении порога, низкой мощности или резком скачке"""
        max_power = max(power_list, default=0)
        min_power = min(power_list, default=0)
        delta_power = max_power - min_power
        print(f"🔋 Максимальная мощность системы: {max_power:.2f} Вт")

        async def notify_event(message: str, event_type: str):
            """Отправка уведомления через WebSocket только один раз за событие"""
            if NOTIFIED_EVENTS.get(event_type) == message:
                return
            NOTIFIED_EVENTS[event_type] = message
            print(f"📡 Отправка уведомления: {message}")
            for connection in active_connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    print(f"❌ Ошибка при отправке уведомления: {e}")
                    active_connections.remove(connection)

        tasks = []

        # Высокая мощность
        if max_power > POWER_THRESHOLD_HIGH:
            tasks.append(notify_event(f"⚡ Внимание! Прогноз мощности системы: {max_power:.2f} Вт", "high_power"))

        # Низкая мощность
        if min_power < POWER_THRESHOLD_LOW:
            tasks.append(notify_event(f"⚠️ Предупреждение! Прогноз низкой мощности: {min_power:.2f} Вт", "low_power"))

        # Резкий скачок мощности
        if delta_power > POWER_DELTA_THRESHOLD:
            tasks.append(notify_event(f"⚡ Резкий скачок мощности: Δ{delta_power:.2f} Вт", "power_spike"))

        await asyncio.gather(*tasks)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hour'] = df['date'].dt.hour
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        for lag in [1, 2, 3]:
            df[f'historical_ghi_{lag}h'] = df['shortwave_radiation'].shift(lag)
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df

    def _train_models(self, historical_df: pd.DataFrame):
        print("🔄 Обучаем ML-модели...")
        X = historical_df[self.features]
        y_dict = {
            'ghi': historical_df['shortwave_radiation'],
            'dni': historical_df['direct_radiation'],
            'dhi': historical_df['diffuse_radiation'],
            'wind_speed_10m': historical_df['wind_speed_10m'],
            'temperature_2m': historical_df['temperature_2m'],
            'pressure_msl': historical_df['pressure_msl'],
            'relative_humidity_2m': historical_df['relative_humidity_2m']
        }

        for target in self.targets:
            print(f"📚 Обучение модели для {target}...")
            X_train, X_val, y_train, y_val = train_test_split(X, y_dict[target], test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            self.models[target] = model
            self.scalers[target] = scaler
            y_pred = model.predict(X_val_scaled)
            mse = mean_squared_error(y_val, y_pred)
            print(f"✅ Модель {target} обучена, MSE: {mse:.2f}")

    def _predict_ml(self, forecast_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions = {}
        X_forecast = forecast_df[self.features]
        for target in self.targets:
            if target in self.models and target in self.scalers:
                X_scaled = self.scalers[target].transform(X_forecast)
                predictions[target] = self.models[target].predict(X_scaled)
        return predictions

    def _prepare_historical_data(self, historical_df: pd.DataFrame) -> List[HourlyInsolation]:
        return [
            HourlyInsolation(
                datetime=row['date'].isoformat(),
                ghi=row['shortwave_radiation'],
                dni=row['direct_radiation'],
                dhi=row['diffuse_radiation'],
                wind_speed=row['wind_speed_10m'],
                temperature=row['temperature_2m'],
                pressure=row['pressure_msl'],
                humidity=row['relative_humidity_2m'],
                power=row['shortwave_radiation'] * PANEL_AREA * PANEL_EFFICIENCY
            )
            for _, row in historical_df.iterrows()
        ]

    def _prepare_forecast_data(self, forecast_df: pd.DataFrame, predictions: Dict) -> List[HourlyInsolation]:
        forecast_data = []
        for i, (_, row) in enumerate(forecast_df.iterrows()):
            forecast_data.append(HourlyInsolation(
                datetime=row['date'].isoformat(),
                ghi=predictions.get('ghi', [0])[i],
                dni=predictions.get('dni', [0])[i],
                dhi=predictions.get('dhi', [0])[i],
                wind_speed=predictions.get('wind_speed_10m', [0])[i],
                temperature=predictions.get('temperature_2m', [0])[i],
                pressure=predictions.get('pressure_msl', [0])[i],
                humidity=predictions.get('relative_humidity_2m', [0])[i],
                power=predictions.get('power', [0])[i]
            ))
        return forecast_data

    def _calculate_metrics(self, historical_df: pd.DataFrame, predictions: Dict) -> ForecastMetrics:
        try:
            validation_data = historical_df.tail(24)
            actual = validation_data['shortwave_radiation'].values
            predicted = list(predictions.get('ghi', []))[:24]
            mse = mean_squared_error(actual, predicted)
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            print(f"📊 Метрики модели - MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
            return ForecastMetrics(mse=mse, mae=mae, r2=r2)
        except:
            return ForecastMetrics()

    def _prepare_ml_plot_data(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame, predictions: Dict) -> List[PlotData]:
        try:
            historical_last_48h = historical_df.tail(48)
            plot_data_list = []

            params = [
                ('GHI', 'shortwave_radiation', 'ghi', '#3498db'),
                ('DNI', 'direct_radiation', 'dni', '#9b59b6'),
                ('DHI', 'diffuse_radiation', 'dhi', '#1abc9c'),
                ('Скорость ветра', 'wind_speed_10m', 'wind_speed_10m', '#27ae60'),
                ('Температура', 'temperature_2m', 'temperature_2m', '#e67e22'),
                ('Давление', 'pressure_msl', 'pressure_msl', '#8e44ad'),
                ('Влажность', 'relative_humidity_2m', 'relative_humidity_2m', '#16a085'),
                ('Мощность', 'shortwave_radiation', 'power', '#f39c12')
            ]

            # Словарь единиц измерения
            units = {
                "GHI": "Вт/м²",
                "DNI": "Вт/м²",
                "DHI": "Вт/м²",
                "Скорость ветра": "м/с",
                "Температура": "°C",
                "Давление": "гПа",
                "Влажность": "%",
                "Мощность": "Вт"
            }

            for title, hist_col, pred_col, color in params:
                trace_hist = {
                    "x": historical_last_48h['date'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
                    "y": historical_last_48h[hist_col].tolist(),
                    "type": "scatter",
                    "mode": "lines",
                    "name": f"Исторические данные ({title})",
                    "line": {"color": color, "width": 3}
                }

                pred_values = predictions.get(pred_col, [])
                if isinstance(pred_values, np.ndarray):
                    pred_values = pred_values.tolist()

                trace_pred = {
                    "x": forecast_df['date'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
                    "y": pred_values,
                    "type": "scatter",
                    "mode": "lines",
                    "name": f"{title}",
                    "line": {"color": color, "width": 3, "dash": "dash"}
                }

                layout = {
                    "title": f"{title}",
                    "xaxis": {"title": "Время"},
                    "yaxis": {"title": f"{title} ({units.get(title, '')})"},
                    "height": 400,
                    "showlegend": True,
                    "plot_bgcolor": "white",
                    "paper_bgcolor": "white"
                }

                plot_data_list.append(PlotData(traces=[trace_hist, trace_pred], layout=layout))

            return plot_data_list
        except Exception as e:
            print(f"❌ Ошибка подготовки ML-графиков: {e}")
            traceback.print_exc()
            return []

    def export_to_excel(self, forecast_response: 'ForecastResponse', filename: str = None) -> str:
        """
        Экспорт исторических и прогнозных данных в Excel.
        """
        try:
            if filename is None:
                filename = f"solar_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                # Исторические данные
                historical_df = pd.DataFrame([{
                    'datetime': h.datetime,
                    'ghi': h.ghi,
                    'dni': h.dni,
                    'dhi': h.dhi,
                    'wind_speed': h.wind_speed,
                    'temperature': h.temperature,
                    'pressure': h.pressure,
                    'humidity': h.humidity,
                    'power': h.power
                } for h in forecast_response.historical_data])
                historical_df.to_excel(writer, sheet_name='Historical', index=False)

                # Прогнозные данные
                forecast_df = pd.DataFrame([{
                    'datetime': f.datetime,
                    'ghi': f.ghi,
                    'dni': f.dni,
                    'dhi': f.dhi,
                    'wind_speed': f.wind_speed,
                    'temperature': f.temperature,
                    'pressure': f.pressure,
                    'humidity': f.humidity,
                    'power': f.power
                } for f in forecast_response.forecast_data])
                forecast_df.to_excel(writer, sheet_name='Forecast', index=False)

                # Метрики модели
                metrics_df = pd.DataFrame([{
                    'MSE': forecast_response.metrics.mse,
                    'MAE': forecast_response.metrics.mae,
                    'R2': forecast_response.metrics.r2
                }])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

            print(f"✅ Данные успешно экспортированы в Excel: {os.path.abspath(filename)}")
            return os.path.abspath(filename)

        except Exception as e:
            print(f"❌ Ошибка при экспорте в Excel: {e}")
            return ""
