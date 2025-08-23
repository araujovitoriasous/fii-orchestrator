from pydantic import BaseModel, Field
from typing import Optional
import os

class APISettings(BaseModel):
    """Configurações da API com suporte a variáveis de ambiente."""
    
    # Configurações do servidor
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    
    # Configurações do Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_enabled: bool = Field(default=True, env="REDIS_ENABLED")
    
    # Configurações do PostgreSQL
    postgres_url: str = Field(default="postgresql://user:pass@localhost/fii_db", env="POSTGRES_URL")
    postgres_enabled: bool = Field(default=True, env="POSTGRES_ENABLED")
    
    # Configurações do ClickHouse
    clickhouse_url: str = Field(default="http://localhost:8123", env="CLICKHOUSE_URL")
    clickhouse_enabled: bool = Field(default=True, env="CLICKHOUSE_ENABLED")
    
    # Configurações de dados
    data_dir: str = Field(default="./data", env="DATA_DIR")
    
    # Configurações de rate limiting
    rate_limit: str = Field(default="100/minute", env="RATE_LIMIT")
    
    # Configurações de cache
    cache_ttl: int = Field(default=300, env="CACHE_TTL")
    
    # Configurações de logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
