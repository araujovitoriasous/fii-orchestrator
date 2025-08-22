"""
Testes para a camada de aplicação.
Testa casos de uso e lógica de negócio de alto nível.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal

from fii_orchestrator.application.use_cases import (
    CollectFundDataUseCase, CollectNewsUseCase,
    AnalyzeFundPerformanceUseCase, ValidateDataQualityUseCase
)
from fii_orchestrator.domain.entities import (
    Fund, FundTicker, PriceQuote, Dividend, NewsItem, Money
)

class TestCollectFundDataUseCase:
    """Testes para o caso de uso de coleta de dados de fundos."""
    
    def setup_method(self):
        """Configuração inicial para cada teste."""
        self.mock_fund_repo = Mock()
        self.mock_price_repo = Mock()
        self.mock_dividend_repo = Mock()
        self.mock_data_provider = Mock()
        
        self.use_case = CollectFundDataUseCase(
            fund_repo=self.mock_fund_repo,
            price_repo=self.mock_price_repo,
            dividend_repo=self.mock_dividend_repo,
            data_provider=self.mock_data_provider
        )
        
        # Configurar mocks padrão
        self.fund = Fund(ticker=FundTicker("HGLG11"))
        self.mock_fund_repo.get_by_ticker.return_value = self.fund
        
        self.prices = [
            PriceQuote(
                fund=self.fund,
                date=datetime.now(),
                close_price=Money(Decimal("100.00")),
                source="test"
            )
        ]
        
        self.dividends = [
            Dividend(
                fund=self.fund,
                ex_date=datetime.now(),
                value_per_share=Money(Decimal("0.50")),
                source="test"
            )
        ]
        
        self.mock_data_provider.fetch_prices.return_value = self.prices
        self.mock_data_provider.fetch_dividends.return_value = self.dividends
    
    def test_execute_success(self):
        """Testa execução bem-sucedida do caso de uso."""
        result = self.use_case.execute("HGLG11", datetime.now(), datetime.now())
        
        # Verificar que os repositórios foram chamados
        self.mock_fund_repo.get_by_ticker.assert_called_once_with("HGLG11")
        self.mock_data_provider.fetch_prices.assert_called_once()
        self.mock_data_provider.fetch_dividends.assert_called_once()
        
        # Verificar que os dados foram salvos
        assert self.mock_price_repo.save_many.call_count == 1
        assert self.mock_dividend_repo.save_many.call_count == 1
        
        # Verificar resultado
        assert result["ticker"] == "HGLG11"
        assert result["prices_collected"] == 1
        assert result["dividends_collected"] == 1
    
    def test_execute_fund_not_found(self):
        """Testa execução quando fundo não é encontrado."""
        self.mock_fund_repo.get_by_ticker.return_value = None
        
        with pytest.raises(ValueError, match="não encontrado"):
            self.use_case.execute("INVALID", datetime.now(), datetime.now())
    
    def test_execute_no_prices(self):
        """Testa execução quando não há preços para coletar."""
        self.mock_data_provider.fetch_prices.return_value = []
        
        result = self.use_case.execute("HGLG11", datetime.now(), datetime.now())
        
        assert result["prices_collected"] == 0
        assert result["dividends_collected"] == 1
    
    def test_execute_no_dividends(self):
        """Testa execução quando não há dividendos para coletar."""
        self.mock_data_provider.fetch_dividends.return_value = []
        
        result = self.use_case.execute("HGLG11", datetime.now(), datetime.now())
        
        assert result["prices_collected"] == 1
        assert result["dividends_collected"] == 0

class TestCollectNewsUseCase:
    """Testes para o caso de uso de coleta de notícias."""
    
    def setup_method(self):
        """Configuração inicial para cada teste."""
        self.mock_news_repo = Mock()
        self.mock_data_provider = Mock()
        
        self.use_case = CollectNewsUseCase(
            news_repo=self.mock_news_repo,
            data_provider=self.mock_data_provider
        )
        
        # Configurar mocks padrão
        self.news_items = [
            NewsItem(
                id="news_001",
                title="FII HGLG11 anuncia novo projeto",
                link="https://example.com/news",
                published_at=datetime.now(),
                source="Test News"
            )
        ]
        
        self.mock_data_provider.fetch_news.return_value = self.news_items
    
    def test_execute_success(self):
        """Testa execução bem-sucedida do caso de uso."""
        sources = ["https://example.com/feed"]
        result = self.use_case.execute(sources)
        
        # Verificar que o provedor foi chamado
        self.mock_data_provider.fetch_news.assert_called_once_with(sources)
        
        # Verificar que as notícias foram salvas
        assert self.mock_news_repo.save_many.call_count == 1
        
        # Verificar resultado
        assert result["sources"] == sources
        assert result["news_collected"] == 1
    
    def test_execute_no_news(self):
        """Testa execução quando não há notícias para coletar."""
        self.mock_data_provider.fetch_news.return_value = []
        
        sources = ["https://example.com/feed"]
        result = self.use_case.execute(sources)
        
        assert result["news_collected"] == 0
        assert self.mock_news_repo.save.call_count == 0

class TestAnalyzeFundPerformanceUseCase:
    """Testes para o caso de uso de análise de performance."""
    
    def setup_method(self):
        """Configuração inicial para cada teste."""
        self.mock_price_repo = Mock()
        self.mock_dividend_repo = Mock()
        
        self.use_case = AnalyzeFundPerformanceUseCase(
            price_repo=self.mock_price_repo,
            dividend_repo=self.mock_dividend_repo
        )
        
        # Configurar mocks padrão
        self.fund = Fund(ticker=FundTicker("HGLG11"))
        
        self.prices = [
            PriceQuote(
                fund=self.fund,
                date=datetime.now() - timedelta(days=365),
                close_price=Money(Decimal("100.00")),
                source="test"
            ),
            PriceQuote(
                fund=self.fund,
                date=datetime.now(),
                close_price=Money(Decimal("110.00")),
                source="test"
            )
        ]
        
        self.dividends = [
            Dividend(
                fund=self.fund,
                ex_date=datetime.now() - timedelta(days=180),
                value_per_share=Money(Decimal("0.50")),
                source="test"
            ),
            Dividend(
                fund=self.fund,
                ex_date=datetime.now() - timedelta(days=90),
                value_per_share=Money(Decimal("0.60")),
                source="test"
            )
        ]
        
        self.mock_price_repo.get_by_fund_and_date_range.return_value = self.prices
        self.mock_dividend_repo.get_by_fund_and_date_range.return_value = self.dividends
    
    def test_execute_success(self):
        """Testa execução bem-sucedida do caso de uso."""
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        result = self.use_case.execute("HGLG11", start_date, end_date)
        
        # Verificar que os repositórios foram chamados
        self.mock_price_repo.get_by_fund_and_date_range.assert_called_once_with(
            "HGLG11", start_date, end_date
        )
        self.mock_dividend_repo.get_by_fund_and_date_range.assert_called_once_with(
            "HGLG11", start_date, end_date
        )
        
        # Verificar resultado
        assert result["ticker"] == "HGLG11"
        assert result["initial_price"] == 100.0
        assert result["final_price"] == 110.0
        assert result["price_return_pct"] == 10.0  # (110-100)/100 * 100
        assert result["dividend_yield_pct"] == 1.1  # (0.5+0.6)/100 * 100
        assert result["total_return_pct"] == 11.1  # 10.0 + 1.1
        assert result["dividends_count"] == 2
        assert result["total_dividends"] == 1.1
    
    def test_execute_no_prices(self):
        """Testa execução quando não há preços para analisar."""
        self.mock_price_repo.get_by_fund_and_date_range.return_value = []
        
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        with pytest.raises(ValueError, match="Nenhum preço encontrado"):
            self.use_case.execute("HGLG11", start_date, end_date)
    
    def test_execute_no_dividends(self):
        """Testa execução quando não há dividendos para analisar."""
        self.mock_dividend_repo.get_by_fund_and_date_range.return_value = []
        
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        result = self.use_case.execute("HGLG11", start_date, end_date)
        
        assert result["dividend_yield_pct"] == 0.0
        assert result["dividends_count"] == 0
        assert result["total_dividends"] == 0.0

class TestValidateDataQualityUseCase:
    """Testes para o caso de uso de validação de qualidade."""
    
    def setup_method(self):
        """Configuração inicial para cada teste."""
        self.mock_price_repo = Mock()
        self.mock_dividend_repo = Mock()
        self.mock_news_repo = Mock()
        self.mock_quality_validator = Mock()
        
        self.use_case = ValidateDataQualityUseCase(
            price_repo=self.mock_price_repo,
            dividend_repo=self.mock_dividend_repo,
            news_repo=self.mock_news_repo,
            quality_validator=self.mock_quality_validator
        )
    
    def test_execute_basic_validation(self):
        """Testa execução básica da validação."""
        result = self.use_case.execute()
        
        # Verificar estrutura básica do resultado
        assert "validation_date" in result
        assert "data_sources" in result
        assert "overall_quality_score" in result
        
        # Verificar fontes de dados
        data_sources = result["data_sources"]
        assert "prices" in data_sources
        assert "dividends" in data_sources
        assert "news" in data_sources
        
        # Verificar status inicial
        for source in data_sources.values():
            assert source["status"] == "validated"
        
        # Verificar score inicial
        assert result["overall_quality_score"] == 83.33  # (85 + 90 + 75) / 3
