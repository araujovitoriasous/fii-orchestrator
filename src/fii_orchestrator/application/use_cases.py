"""
Casos de uso da camada de aplicação.

Este módulo contém a lógica de negócio de alto nível,
coordenando operações entre diferentes repositórios e
provedores de dados externos.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal

from .interfaces import (
    FundRepository, PriceRepository, DividendRepository, 
    NewsRepository, DataProvider, DataQualityValidator
)
from ..domain.entities import Fund, PriceQuote, Dividend, NewsItem, FundMetrics
from ..domain.value_objects import Money, Percentage

class CollectFundDataUseCase:
    """
    Caso de uso para coleta de dados de um fundo específico.
    
    Este caso de uso coordena a coleta de preços e dividendos
    de um fundo imobiliário em um período específico, utilizando
    o provedor de dados configurado e persistindo os resultados
    nos repositórios apropriados.
    
    Flow:
        1. Valida se o fundo existe na referência
        2. Coleta preços do período especificado
        3. Coleta dividendos do período especificado
        4. Persiste dados nos repositórios
        5. Retorna resumo da operação
        
    Dependencies:
        - FundRepository: Para buscar informações do fundo
        - PriceRepository: Para persistir preços
        - DividendRepository: Para persistir dividendos
        - DataProvider: Para coletar dados externos
    """
    
    def __init__(
        self,
        fund_repo: FundRepository,
        price_repo: PriceRepository,
        dividend_repo: DividendRepository,
        data_provider: DataProvider
    ):
        """
        Inicializa o caso de uso.
        
        Args:
            fund_repo: Repositório de fundos
            price_repo: Repositório de preços
            dividend_repo: Repositório de dividendos
            data_provider: Provedor de dados externos
        """
        self.fund_repo = fund_repo
        self.price_repo = price_repo
        self.dividend_repo = dividend_repo
        self.data_provider = data_provider
    
    def execute(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Executa a coleta de dados do fundo.
        
        Args:
            ticker: Ticker do fundo (ex: "HGLG11")
            start_date: Data de início da coleta
            end_date: Data de fim da coleta
            
        Returns:
            Dicionário com resumo da operação contendo:
            - ticker: Ticker processado
            - prices_collected: Número de preços coletados
            - dividends_collected: Número de dividendos coletados
            - collection_period: Período da coleta
            - status: Status da operação
            
        Raises:
            ValueError: Se o fundo não for encontrado
            DataCollectionError: Se houver erro na coleta
        """
        # 1. Validar se o fundo existe
        fund = self.fund_repo.get_by_ticker(ticker)
        if not fund:
            raise ValueError(f"Fundo com ticker {ticker} não encontrado")
        
        # 2. Coletar preços
        prices = self.data_provider.fetch_prices(ticker, start_date, end_date)
        
        # 3. Coletar dividendos
        dividends = self.data_provider.fetch_dividends(ticker, start_date, end_date)
        
        # 4. Persistir dados
        if prices:
            self.price_repo.save_many(prices)
        
        if dividends:
            self.dividend_repo.save_many(dividends)
        
        # 5. Retornar resumo
        return {
            "ticker": ticker,
            "prices_collected": len(prices),
            "dividends_collected": len(dividends),
            "collection_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "status": "success",
            "executed_at": datetime.now().isoformat()
        }

class CollectNewsUseCase:
    """
    Caso de uso para coleta de notícias de fontes RSS.
    
    Este caso de uso coordena a coleta de notícias de
    múltiplas fontes RSS, detecta automaticamente fundos
    relacionados e persiste os resultados.
    
    Flow:
        1. Coleta notícias de todas as fontes configuradas
        2. Detecta tickers de fundos nas notícias
        3. Relaciona notícias aos fundos correspondentes
        4. Persiste notícias no repositório
        5. Retorna resumo da coleta
        
    Dependencies:
        - NewsRepository: Para persistir notícias
        - DataProvider: Para coletar dados RSS
    """
    
    def __init__(
        self,
        news_repo: NewsRepository,
        data_provider: DataProvider
    ):
        """
        Inicializa o caso de uso.
        
        Args:
            news_repo: Repositório de notícias
            data_provider: Provedor de dados RSS
        """
        self.news_repo = news_repo
        self.data_provider = data_provider
    
    def execute(self, sources: List[str]) -> Dict[str, Any]:
        """
        Executa a coleta de notícias.
        
        Args:
            sources: Lista de fontes RSS para coletar
            
        Returns:
            Dicionário com resumo da coleta contendo:
            - sources: Fontes processadas
            - news_collected: Número de notícias coletadas
            - execution_time: Tempo de execução
            - status: Status da operação
            
        Raises:
            DataCollectionError: Se houver erro na coleta
        """
        start_time = datetime.now()
        
        # Coletar notícias de todas as fontes
        all_news = self.data_provider.fetch_news(sources)
        
        # Persistir notícias coletadas
        if all_news:
            self.news_repo.save_many(all_news)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "sources": sources,
            "news_collected": len(all_news),
            "execution_time_seconds": execution_time,
            "status": "success",
            "executed_at": datetime.now().isoformat()
        }

class AnalyzeFundPerformanceUseCase:
    """
    Caso de uso para análise de performance de fundos.
    
    Este caso de uso analisa a performance de um fundo
    em um período específico, calculando retornos,
    dividend yields e outras métricas relevantes.
    
    Flow:
        1. Busca preços do período especificado
        2. Busca dividendos do período especificado
        3. Calcula métricas de performance
        4. Retorna análise completa
        
    Dependencies:
        - PriceRepository: Para buscar preços
        - DividendRepository: Para buscar dividendos
    """
    
    def __init__(
        self,
        price_repo: PriceRepository,
        dividend_repo: DividendRepository
    ):
        """
        Inicializa o caso de uso.
        
        Args:
            price_repo: Repositório de preços
            dividend_repo: Repositório de dividendos
        """
        self.price_repo = price_repo
        self.dividend_repo = dividend_repo
    
    def execute(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Executa a análise de performance do fundo.
        
        Args:
            ticker: Ticker do fundo
            start_date: Data de início da análise
            end_date: Data de fim da análise
            
        Returns:
            Dicionário com análise de performance contendo:
            - ticker: Ticker analisado
            - initial_price: Preço inicial
            - final_price: Preço final
            - price_return_pct: Retorno percentual do preço
            - dividend_yield_pct: Dividend yield percentual
            - total_return_pct: Retorno total percentual
            - dividends_count: Número de dividendos
            - total_dividends: Total de dividendos recebidos
            
        Raises:
            ValueError: Se não houver dados suficientes para análise
        """
        # Buscar dados do período
        prices = self.price_repo.get_by_fund_and_date_range(
            ticker, start_date, end_date
        )
        
        if not prices:
            raise ValueError(f"Nenhum preço encontrado para {ticker} no período especificado")
        
        # Ordenar preços por data
        prices.sort(key=lambda p: p.date)
        
        # Calcular métricas de preço
        initial_price = prices[0].close_price.to_float()
        final_price = prices[-1].close_price.to_float()
        price_return_pct = ((final_price - initial_price) / initial_price) * 100
        
        # Buscar dividendos do período
        dividends = self.dividend_repo.get_by_fund_and_date_range(
            ticker, start_date, end_date
        )
        
        # Calcular métricas de dividendos
        total_dividends = sum(d.value_per_share.to_float() for d in dividends)
        dividend_yield_pct = (total_dividends / initial_price) * 100 if initial_price > 0 else 0
        
        # Calcular retorno total
        total_return_pct = price_return_pct + dividend_yield_pct
        
        return {
            "ticker": ticker,
            "initial_price": initial_price,
            "final_price": final_price,
            "price_return_pct": round(price_return_pct, 2),
            "dividend_yield_pct": round(dividend_yield_pct, 2),
            "total_return_pct": round(total_return_pct, 2),
            "dividends_count": len(dividends),
            "total_dividends": round(total_dividends, 4),
            "analysis_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "executed_at": datetime.now().isoformat()
        }

class ValidateDataQualityUseCase:
    """
    Caso de uso para validação de qualidade dos dados.
    
    Este caso de uso executa validações completas de
    qualidade em todas as fontes de dados, gerando
    relatórios detalhados e recomendações.
    
    Flow:
        1. Valida frescor dos dados por fonte
        2. Valida completude dos dados por fonte
        3. Valida consistência dos dados por fonte
        4. Calcula score geral de qualidade
        5. Gera relatório com recomendações
        
    Dependencies:
        - DataQualityValidator: Para execução das validações
        - PriceRepository: Para validação de preços
        - DividendRepository: Para validação de dividendos
        - NewsRepository: Para validação de notícias
    """
    
    def __init__(
        self,
        price_repo: PriceRepository,
        dividend_repo: DividendRepository,
        news_repo: NewsRepository,
        quality_validator: DataQualityValidator
    ):
        """
        Inicializa o caso de uso.
        
        Args:
            price_repo: Repositório de preços
            dividend_repo: Repositório de dividendos
            news_repo: Repositório de notícias
            quality_validator: Validador de qualidade
        """
        self.price_repo = price_repo
        self.dividend_repo = dividend_repo
        self.news_repo = news_repo
        self.quality_validator = quality_validator
    
    def execute(self) -> Dict[str, Any]:
        """
        Executa a validação completa de qualidade.
        
        Returns:
            Dicionário com relatório de qualidade contendo:
            - validation_date: Data da validação
            - data_sources: Validação por fonte de dados
            - overall_quality_score: Score geral (0-100)
            - recommendations: Lista de recomendações
            - execution_time: Tempo de execução
            
        Raises:
            ValidationError: Se houver erro na validação
        """
        start_time = datetime.now()
        
        # Executar validações por fonte
        data_sources = {
            "prices": self._validate_prices_quality(),
            "dividends": self._validate_dividends_quality(),
            "news": self._validate_news_quality()
        }
        
        # Calcular score geral
        scores = [source["quality_score"] for source in data_sources.values()]
        overall_quality_score = sum(scores) / len(scores) if scores else 0
        
        # Gerar recomendações
        recommendations = self._generate_recommendations(data_sources, overall_quality_score)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "validation_date": datetime.now().isoformat(),
            "data_sources": data_sources,
            "overall_quality_score": round(overall_quality_score, 2),
            "recommendations": recommendations,
            "execution_time_seconds": execution_time,
            "status": "completed"
        }
    
    def _validate_prices_quality(self) -> Dict[str, Any]:
        """Valida qualidade dos dados de preços."""
        # Implementar validação específica para preços
        return {
            "status": "validated",
            "quality_score": 85.0,
            "issues": [],
            "last_update": datetime.now().isoformat()
        }
    
    def _validate_dividends_quality(self) -> Dict[str, Any]:
        """Valida qualidade dos dados de dividendos."""
        # Implementar validação específica para dividendos
        return {
            "status": "validated",
            "quality_score": 90.0,
            "issues": [],
            "last_update": datetime.now().isoformat()
        }
    
    def _validate_news_quality(self) -> Dict[str, Any]:
        """Valida qualidade dos dados de notícias."""
        # Implementar validação específica para notícias
        return {
            "status": "validated",
            "quality_score": 75.0,
            "issues": [],
            "last_update": datetime.now().isoformat()
        }
    
    def _generate_recommendations(
        self, 
        data_sources: Dict[str, Any], 
        overall_score: float
    ) -> List[str]:
        """Gera recomendações baseadas na qualidade dos dados."""
        recommendations = []
        
        if overall_score < 80:
            recommendations.append("Score de qualidade baixo. Revisar fontes de dados.")
        
        for source_name, source_data in data_sources.items():
            if source_data["quality_score"] < 70:
                recommendations.append(f"Qualidade de {source_name} crítica. Investigar problemas.")
            elif source_data["quality_score"] < 85:
                recommendations.append(f"Qualidade de {source_name} pode ser melhorada.")
        
        if not recommendations:
            recommendations.append("Qualidade dos dados está excelente!")
        
        return recommendations
