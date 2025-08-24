import asyncpg
import asyncio
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from loguru import logger
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass(frozen=True)
class PostgresConfig:
    """Configuração do PostgreSQL."""
    host: str
    port: int
    database: str
    user: str
    password: str
    ssl_mode: str = "prefer"
    max_connections: int = 20
    connection_timeout: int = 30
    command_timeout: int = 60
    
    @property
    def connection_string(self) -> str:
        """Gera string de conexão."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )
    
    @property
    def dsn(self) -> str:
        """Gera DSN para asyncpg."""
        return (
            f"host={self.host} port={self.port} "
            f"dbname={self.database} user={self.user} "
            f"password={self.password} sslmode={self.ssl_mode}"
        )

class PostgresService:
    """Serviço PostgreSQL com pool de conexões e health checks."""
    
    def __init__(self, config: PostgresConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[datetime] = None
        self._is_healthy: bool = False
    
    async def initialize(self) -> None:
        """Inicializa o pool de conexões."""
        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.config.connection_string,
                min_size=5,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout,
                server_settings={
                    'application_name': 'fii_orchestrator',
                    'timezone': 'UTC'
                }
            )
            logger.info(f"Pool PostgreSQL criado com sucesso: {self.config.host}:{self.config.port}")
            
            # Iniciar health check
            self._start_health_check()
            
        except Exception as e:
            logger.error(f"Erro ao criar pool PostgreSQL: {e}")
            raise
    
    async def close(self) -> None:
        """Fecha o pool de conexões."""
        if self._health_check_task:
            self._health_check_task.cancel()
        
        if self.pool:
            await self.pool.close()
            logger.info("Pool PostgreSQL fechado")
    
    @asynccontextmanager
    async def get_connection(self):
        """Context manager para obter conexão do pool."""
        if not self.pool:
            raise RuntimeError("Pool não inicializado. Chame initialize() primeiro.")
        
        async with self.pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Erro na conexão PostgreSQL: {e}")
                await connection.rollback()
                raise
    
    async def execute(self, query: str, *args) -> str:
        """Executa query sem retorno."""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Executa query com retorno de múltiplas linhas."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Executa query com retorno de uma linha."""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Executa query com retorno de um valor."""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def health_check(self) -> bool:
        """Verifica saúde da conexão."""
        try:
            if not self.pool:
                return False
            
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                self._is_healthy = result == 1
                self._last_health_check = datetime.utcnow()
                return self._is_healthy
                
        except Exception as e:
            logger.error(f"Health check PostgreSQL falhou: {e}")
            self._is_healthy = False
            return False
    
    def _start_health_check(self) -> None:
        """Inicia task de health check periódico."""
        async def _health_check_loop():
            while True:
                try:
                    await self.health_check()
                    await asyncio.sleep(30)  # Check a cada 30 segundos
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Erro no health check loop: {e}")
                    await asyncio.sleep(5)
        
        self._health_check_task = asyncio.create_task(_health_check_loop())
    
    @property
    def is_healthy(self) -> bool:
        """Status de saúde atual."""
        return self._is_healthy
    
    @property
    def last_health_check(self) -> Optional[datetime]:
        """Última verificação de saúde."""
        return self._last_health_check
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do pool."""
        if not self.pool:
            return {}
        
        return {
            'pool_size': self.pool.get_size(),
            'free_size': self.pool.get_free_size(),
            'is_healthy': self.is_healthy,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None
        }
