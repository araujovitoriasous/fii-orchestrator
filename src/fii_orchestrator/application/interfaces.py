"""
Interfaces (Protocols) da camada de aplicação.

Este módulo define os contratos que devem ser implementados
pelas classes concretas da infraestrutura, garantindo
inversão de dependências e testabilidade.
"""

from typing import Protocol, List, Optional, Dict, Any
from datetime import datetime

from ..domain.entities import Fund, PriceQuote, Dividend, NewsItem, FundMetrics

class FundRepository(Protocol):
    """
    Protocolo para repositório de fundos.
    
    Define as operações básicas de persistência para
    entidades Fund, incluindo CRUD e consultas específicas.
    """
    
    def save(self, fund: Fund) -> None:
        """
        Salva um fundo no repositório.
        
        Args:
            fund: Fundo a ser salvo
            
        Raises:
            RepositoryError: Se houver erro na persistência
        """
        ...
    
    def get_by_ticker(self, ticker: str) -> Optional[Fund]:
        """
        Busca um fundo pelo ticker.
        
        Args:
            ticker: Ticker do fundo
            
        Returns:
            Fundo encontrado ou None se não existir
        """
        ...
    
    def get_all(self) -> List[Fund]:
        """
        Retorna todos os fundos cadastrados.
        
        Returns:
            Lista de todos os fundos
        """
        ...
    
    def exists(self, ticker: str) -> bool:
        """
        Verifica se um fundo existe.
        
        Args:
            ticker: Ticker do fundo
            
        Returns:
            True se o fundo existir, False caso contrário
        """
        ...
    
    def delete(self, ticker: str) -> bool:
        """
        Remove um fundo do repositório.
        
        Args:
            ticker: Ticker do fundo a ser removido
            
        Returns:
            True se o fundo foi removido, False se não existia
        """
        ...
    
    def update(self, fund: Fund) -> bool:
        """
        Atualiza um fundo existente.
        
        Args:
            fund: Fundo com dados atualizados
            
        Returns:
            True se o fundo foi atualizado, False se não existia
        """
        ...

class PriceRepository(Protocol):
    """
    Protocolo para repositório de cotações de preços.
    
    Define as operações de persistência para entidades
    PriceQuote, incluindo consultas por período e fundo.
    """
    
    def save(self, price: PriceQuote) -> None:
        """
        Salva uma cotação no repositório.
        
        Args:
            price: Cotação a ser salva
        """
        ...
    
    def save_many(self, prices: List[PriceQuote]) -> None:
        """
        Salva múltiplas cotações de uma vez.
        
        Args:
            prices: Lista de cotações a serem salvas
        """
        ...
    
    def get_by_fund_and_date_range(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[PriceQuote]:
        """
        Busca cotações de um fundo em um período específico.
        
        Args:
            ticker: Ticker do fundo
            start_date: Data de início
            end_date: Data de fim
            
        Returns:
            Lista de cotações no período especificado
        """
        ...
    
    def get_latest_price(self, ticker: str) -> Optional[PriceQuote]:
        """
        Busca a cotação mais recente de um fundo.
        
        Args:
            ticker: Ticker do fundo
            
        Returns:
            Cotação mais recente ou None se não existir
        """
        ...
    
    def get_price_history(
        self, 
        ticker: str, 
        days: int = 30
    ) -> List[PriceQuote]:
        """
        Busca histórico de preços de um fundo.
        
        Args:
            ticker: Ticker do fundo
            days: Número de dias para buscar
            
        Returns:
            Lista de cotações ordenadas por data
        """
        ...

class DividendRepository(Protocol):
    """
    Protocolo para repositório de dividendos.
    
    Define as operações de persistência para entidades
    Dividend, incluindo consultas por período e fundo.
    """
    
    def save(self, dividend: Dividend) -> None:
        """
        Salva um dividendo no repositório.
        
        Args:
            dividend: Dividendo a ser salvo
        """
        ...
    
    def save_many(self, dividends: List[Dividend]) -> None:
        """
        Salva múltiplos dividendos de uma vez.
        
        Args:
            dividends: Lista de dividendos a serem salvos
        """
        ...
    
    def get_by_fund_and_date_range(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dividend]:
        """
        Busca dividendos de um fundo em um período específico.
        
        Args:
            ticker: Ticker do fundo
            start_date: Data de início
            end_date: Data de fim
            
        Returns:
            Lista de dividendos no período especificado
        """
        ...
    
    def get_pending_dividends(self, ticker: str) -> List[Dividend]:
        """
        Busca dividendos pendentes de um fundo.
        
        Args:
            ticker: Ticker do fundo
            
        Returns:
            Lista de dividendos pendentes
        """
        ...
    
    def get_dividend_yield(
        self, 
        ticker: str, 
        period_days: int = 365
    ) -> Optional[float]:
        """
        Calcula o dividend yield de um fundo.
        
        Args:
            ticker: Ticker do fundo
            period_days: Período em dias para cálculo
            
        Returns:
            Dividend yield como percentual ou None se não houver dados
        """
        ...

class NewsRepository(Protocol):
    """
    Protocolo para repositório de notícias.
    
    Define as operações de persistência para entidades
    NewsItem, incluindo consultas por período e fundos relacionados.
    """
    
    def save(self, news: NewsItem) -> None:
        """
        Salva uma notícia no repositório.
        
        Args:
            news: Notícia a ser salva
        """
        ...
    
    def save_many(self, news_items: List[NewsItem]) -> None:
        """
        Salva múltiplas notícias de uma vez.
        
        Args:
            news_items: Lista de notícias a serem salvas
        """
        ...
    
    def get_by_fund(self, ticker: str, limit: int = 50) -> List[NewsItem]:
        """
        Busca notícias relacionadas a um fundo específico.
        
        Args:
            ticker: Ticker do fundo
            limit: Limite de notícias a retornar
            
        Returns:
            Lista de notícias relacionadas ao fundo
        """
        ...
    
    def get_recent_news(self, days: int = 7) -> List[NewsItem]:
        """
        Busca notícias recentes.
        
        Args:
            days: Número de dias para considerar recente
            
        Returns:
            Lista de notícias recentes
        """
        ...
    
    def search_news(
        self, 
        query: str, 
        limit: int = 50
    ) -> List[NewsItem]:
        """
        Busca notícias por texto.
        
        Args:
            query: Texto para busca
            limit: Limite de resultados
            
        Returns:
            Lista de notícias que correspondem à busca
        """
        ...

class FundMetricsRepository(Protocol):
    """
    Protocolo para repositório de métricas de fundos.
    
    Define as operações de persistência para entidades
    FundMetrics, incluindo consultas por período e indicadores.
    """
    
    def save(self, metrics: FundMetrics) -> None:
        """
        Salva métricas de um fundo.
        
        Args:
            metrics: Métricas a serem salvas
        """
        ...
    
    def get_latest_metrics(self, ticker: str) -> Optional[FundMetrics]:
        """
        Busca as métricas mais recentes de um fundo.
        
        Args:
            ticker: Ticker do fundo
            
        Returns:
            Métricas mais recentes ou None se não existirem
        """
        ...
    
    def get_metrics_history(
        self, 
        ticker: str, 
        days: int = 365
    ) -> List[FundMetrics]:
        """
        Busca histórico de métricas de um fundo.
        
        Args:
            ticker: Ticker do fundo
            days: Número de dias para buscar
            
        Returns:
            Lista de métricas ordenadas por data
        """
        ...

class DataProvider(Protocol):
    """
    Protocolo para provedores de dados externos.
    
    Define a interface que deve ser implementada por
    adaptadores de APIs externas como Yahoo Finance,
    CVM, RSS feeds, etc.
    """
    
    def fetch_prices(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[PriceQuote]:
        """
        Busca preços de um fundo em um período específico.
        
        Args:
            ticker: Ticker do fundo
            start_date: Data de início
            end_date: Data de fim
            
        Returns:
            Lista de cotações coletadas
            
        Raises:
            DataProviderError: Se houver erro na coleta
        """
        ...
    
    def fetch_dividends(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dividend]:
        """
        Busca dividendos de um fundo em um período específico.
        
        Args:
            ticker: Ticker do fundo
            start_date: Data de início
            end_date: Data de fim
            
        Returns:
            Lista de dividendos coletados
            
        Raises:
            DataProviderError: Se houver erro na coleta
        """
        ...
    
    def fetch_news(self, sources: List[str]) -> List[NewsItem]:
        """
        Busca notícias de fontes específicas.
        
        Args:
            sources: Lista de fontes de notícias
            
        Returns:
            Lista de notícias coletadas
            
        Raises:
            DataProviderError: Se houver erro na coleta
        """
        ...

class DataQualityValidator(Protocol):
    """
    Protocolo para validadores de qualidade de dados.
    
    Define a interface para validação de dados coletados,
    incluindo verificações de frescor, completude e consistência.
    """
    
    def validate_data_freshness(
        self, 
        data_source: str, 
        max_age_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Valida se os dados estão atualizados.
        
        Args:
            data_source: Fonte de dados a ser validada
            max_age_hours: Idade máxima em horas para considerar válido
            
        Returns:
            Dicionário com resultados da validação
        """
        ...
    
    def validate_data_completeness(
        self, 
        data_source: str
    ) -> Dict[str, Any]:
        """
        Valida se os dados estão completos.
        
        Args:
            data_source: Fonte de dados a ser validada
            
        Returns:
            Dicionário com resultados da validação
        """
        ...
    
    def validate_data_consistency(
        self, 
        data_source: str
    ) -> Dict[str, Any]:
        """
        Valida se os dados são consistentes.
        
        Args:
            data_source: Fonte de dados a ser validada
            
        Returns:
            Dicionário com resultados da validação
        """
        ...
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Gera relatório completo de qualidade.
        
        Returns:
            Dicionário com relatório de qualidade
        """
        ...
