"""
Configuração centralizada e factory para injeção de dependências.
Implementa o padrão Factory para criar instâncias das classes.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from fii_orchestrator.infrastructure.repositories import (
    ParquetFundRepository, ParquetPriceRepository, 
    ParquetDividendRepository, ParquetNewsRepository
)
from fii_orchestrator.application.use_cases import (
    CollectFundDataUseCase, CollectNewsUseCase,
    AnalyzeFundPerformanceUseCase, ValidateDataQualityUseCase
)
from fii_orchestrator.utils.validation import DataValidator

load_dotenv()

@dataclass(frozen=True)
class DatabaseConfig:
    """Configuração do banco de dados."""
    data_dir: Path
    bronze_dir: Path
    meta_dir: Path
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
        return cls(
            data_dir=data_dir,
            bronze_dir=data_dir / "bronze",
            meta_dir=data_dir / "meta"
        )

@dataclass(frozen=True)
class APIConfig:
    """Configuração das APIs externas."""
    yahoo_finance_rate_limit: float
    cvm_timeout: int
    max_retries: int
    base_delay: float
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        return cls(
            yahoo_finance_rate_limit=float(os.getenv("YF_RATE_LIMIT", "0.8")),
            cvm_timeout=int(os.getenv("CVM_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            base_delay=float(os.getenv("BASE_DELAY", "2.0"))
        )

@dataclass(frozen=True)
class ProcessingConfig:
    """Configuração do processamento."""
    use_parallel: bool
    max_workers: int
    chunk_size: int
    batch_size: int
    
    @classmethod
    def from_env(cls) -> 'ProcessingConfig':
        return cls(
            use_parallel=os.getenv("USE_PARALLEL", "false").lower() == "true",
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "10")),
            batch_size=int(os.getenv("BATCH_SIZE", "5"))
        )

@dataclass(frozen=True)
class RSSConfig:
    """Configuração das fontes RSS."""
    sources: list[str]
    update_interval_hours: int
    
    @classmethod
    def from_env(cls) -> 'RSSConfig':
        sources_str = os.getenv("NEWS_RSS_SOURCES", "")
        sources = [s.strip() for s in sources_str.split(",") if s.strip()] if sources_str else [
            "https://www.infomoney.com.br/feed/",
            "https://valor.globo.com/rss/",
            "https://www.suno.com.br/feed/"
        ]
        
        return cls(
            sources=sources,
            update_interval_hours=int(os.getenv("RSS_UPDATE_INTERVAL", "6"))
        )

@dataclass(frozen=True)
class AppConfig:
    """Configuração principal da aplicação."""
    database: DatabaseConfig
    api: APIConfig
    processing: ProcessingConfig
    rss: RSSConfig
    environment: str
    debug: bool
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        return cls(
            database=DatabaseConfig.from_env(),
            api=APIConfig.from_env(),
            processing=ProcessingConfig.from_env(),
            rss=RSSConfig.from_env(),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )

class DependencyContainer:
    """Container de injeção de dependências."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._repositories = {}
        self._use_cases = {}
        self._initialized = False
    
    def _initialize_repositories(self):
        """Inicializa os repositórios."""
        if self._initialized:
            return
        
        # Criar repositórios
        self._repositories['fund'] = ParquetFundRepository(self.config.database.bronze_dir)
        self._repositories['price'] = ParquetPriceRepository(self.config.database.bronze_dir)
        self._repositories['dividend'] = ParquetDividendRepository(self.config.database.bronze_dir)
        self._repositories['news'] = ParquetNewsRepository(self.config.database.bronze_dir)
        
        self._initialized = True
    
    def get_repository(self, name: str):
        """Retorna um repositório pelo nome."""
        self._initialize_repositories()
        return self._repositories.get(name)
    
    def get_use_case(self, name: str):
        """Retorna um caso de uso pelo nome."""
        if name not in self._use_cases:
            self._create_use_case(name)
        return self._use_cases[name]
    
    def _create_use_case(self, name: str):
        """Cria um caso de uso específico."""
        if name == 'collect_fund_data':
            self._use_cases[name] = CollectFundDataUseCase(
                fund_repo=self.get_repository('fund'),
                price_repo=self.get_repository('price'),
                dividend_repo=self.get_repository('dividend'),
                data_provider=None  # TODO: Implementar provedor de dados
            )
        elif name == 'collect_news':
            self._use_cases[name] = CollectNewsUseCase(
                news_repo=self.get_repository('news'),
                data_provider=None  # TODO: Implementar provedor de dados
            )
        elif name == 'analyze_performance':
            self._use_cases[name] = AnalyzeFundPerformanceUseCase(
                price_repo=self.get_repository('price'),
                dividend_repo=self.get_repository('dividend')
            )
        elif name == 'validate_quality':
            self._use_cases[name] = ValidateDataQualityUseCase(
                price_repo=self.get_repository('price'),
                dividend_repo=self.get_repository('dividend'),
                news_repo=self.get_repository('news'),
                quality_validator=DataValidator()
            )
        else:
            raise ValueError(f"Caso de uso desconhecido: {name}")

# Instância global do container
_config = AppConfig.from_env()
_container = DependencyContainer(_config)

def get_config() -> AppConfig:
    """Retorna a configuração da aplicação."""
    return _config

def get_container() -> DependencyContainer:
    """Retorna o container de dependências."""
    return _container

def get_repository(name: str):
    """Retorna um repositório pelo nome."""
    return _container.get_repository(name)

def get_use_case(name: str):
    """Retorna um caso de uso pelo nome."""
    return _container.get_use_case(name)
