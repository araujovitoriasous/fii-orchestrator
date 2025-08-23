import asyncpg
from typing import List, Optional, Dict, Any
from loguru import logger

class PostgresService:
    """Serviço para operações PostgreSQL."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
