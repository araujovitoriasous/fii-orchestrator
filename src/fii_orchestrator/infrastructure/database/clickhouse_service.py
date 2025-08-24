import clickhouse_connect
from typing import List, Dict, Any
from loguru import logger

class ClickHouseService:
    """Serviço para operações ClickHouse (analytics)."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.client = None

    def get_client(self):
        """Obtém cliente ClickHouse."""
        if not self.client:
            self.client = clickhouse_connect.get_client(host=self.connection_string)
        return self.client

    def close(self):
        """Fecha conexão ClickHouse."""
        if self.client:
            self.client.close()
