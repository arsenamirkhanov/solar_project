from fastapi import WebSocket

# Список всех активных WebSocket-подключений
active_connections: list[WebSocket] = []
