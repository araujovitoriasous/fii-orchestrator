from dependency_injector import containers, providers
from fii_orchestrator.serving.duckdb_service import DuckDBService
from fii_orchestrator.serving.cache_service import RedisCacheService
from fii_orchestrator.infrastructure.database.postgres_service import PostgresService
from fii_orchestrator.infrastructure.database.clickhouse_service import ClickHouseService
from fii_orchestrator.presentation.config import APISettings

class Container(containers.DeclarativeContainer):
    """Container de injeção de dependências."""

    # Configuração
    config = providers.Singleton(APISettings)

    # Serviços
    duckdb_service = providers.Singleton(DuckDBService)
    cache_service = providers.Singleton(RedisCacheService)
    
    # Serviços de Banco de Dados
    postgres_service = providers.Singleton(
        PostgresService, 
        connection_string=config.provided.postgres_url
    )
    clickhouse_service = providers.Singleton(
        ClickHouseService, 
        connection_string=config.provided.clickhouse_url
    )
