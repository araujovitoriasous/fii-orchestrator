"""
Testes para a camada de domínio.
Testa entidades, value objects e regras de negócio.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from fii_orchestrator.domain.entities import (
    FundTicker, CNPJ, Money, Fund, PriceQuote, Dividend, NewsItem, FundMetrics
)
from fii_orchestrator.domain.value_objects import Percentage, PositiveFloat

class TestFundTicker:
    """Testes para o value object FundTicker."""
    
    def test_valid_ticker(self):
        """Testa criação de ticker válido."""
        ticker = FundTicker("HGLG11")
        assert str(ticker) == "HGLG11"
    
    def test_invalid_ticker_format(self):
        """Testa rejeição de ticker com formato inválido."""
        with pytest.raises(ValueError, match="Ticker inválido"):
            FundTicker("INVALID")
    
    def test_invalid_ticker_length(self):
        """Testa rejeição de ticker com comprimento inválido."""
        with pytest.raises(ValueError):
            FundTicker("HGLG1")  # Muito curto
        with pytest.raises(ValueError):
            FundTicker("HGLG111")  # Muito longo
    
    def test_invalid_ticker_pattern(self):
        """Testa rejeição de ticker com padrão inválido."""
        with pytest.raises(ValueError):
            FundTicker("12345")  # Apenas números
        with pytest.raises(ValueError):
            FundTicker("ABCDEF")  # Apenas letras

class TestCNPJ:
    """Testes para o value object CNPJ."""
    
    def test_valid_cnpj(self):
        """Testa criação de CNPJ válido."""
        cnpj = CNPJ("11.222.333/0001-81")
        assert str(cnpj) == "11.222.333/0001-81"
    
    def test_invalid_cnpj_length(self):
        """Testa rejeição de CNPJ com comprimento inválido."""
        with pytest.raises(ValueError):
            CNPJ("123")  # Muito curto
        with pytest.raises(ValueError):
            CNPJ("123456789012345")  # Muito longo
    
    def test_invalid_cnpj_all_same_digits(self):
        """Testa rejeição de CNPJ com todos os dígitos iguais."""
        with pytest.raises(ValueError):
            CNPJ("11.111.111/1111-11")
    
    def test_invalid_cnpj_check_digits(self):
        """Testa rejeição de CNPJ com dígitos verificadores inválidos."""
        with pytest.raises(ValueError):
            CNPJ("08.441.966/0001-00")  # Dígitos verificadores incorretos

class TestMoney:
    """Testes para o value object Money."""
    
    def test_valid_money(self):
        """Testa criação de valor monetário válido."""
        money = Money(Decimal("100.50"))
        assert money.amount == Decimal("100.50")
        assert money.currency == "BRL"
    
    def test_invalid_negative_money(self):
        """Testa rejeição de valor monetário negativo."""
        with pytest.raises(ValueError, match="não pode ser negativo"):
            Money(Decimal("-100.00"))
    
    def test_money_addition(self):
        """Testa adição de valores monetários."""
        money1 = Money(Decimal("100.00"))
        money2 = Money(Decimal("50.00"))
        result = money1 + money2
        assert result.amount == Decimal("150.00")
    
    def test_money_subtraction(self):
        """Testa subtração de valores monetários."""
        money1 = Money(Decimal("100.00"))
        money2 = Money(Decimal("30.00"))
        result = money1 - money2
        assert result.amount == Decimal("70.00")
    
    def test_money_subtraction_negative_result(self):
        """Testa rejeição de subtração que resulta em valor negativo."""
        money1 = Money(Decimal("50.00"))
        money2 = Money(Decimal("100.00"))
        with pytest.raises(ValueError, match="não pode ser negativo"):
            money1 - money2
    
    def test_money_different_currencies(self):
        """Testa rejeição de operações com moedas diferentes."""
        money1 = Money(Decimal("100.00"), "BRL")
        money2 = Money(Decimal("50.00"), "USD")
        with pytest.raises(ValueError, match="Moedas diferentes"):
            money1 + money2

class TestFund:
    """Testes para a entidade Fund."""
    
    def test_valid_fund(self):
        """Testa criação de fundo válido."""
        ticker = FundTicker("HGLG11")
        cnpj = CNPJ("11.222.333/0001-81")
        fund = Fund(
            ticker=ticker,
            cnpj=cnpj,
            razao_social="CSHG Real Estate FII",
            fonte="test"
        )
        assert str(fund.ticker) == "HGLG11"
        assert str(fund.cnpj) == "11.222.333/0001-81"
        assert fund.razao_social == "CSHG Real Estate FII"
    
    def test_fund_without_optional_fields(self):
        """Testa criação de fundo sem campos opcionais."""
        ticker = FundTicker("HGLG11")
        fund = Fund(ticker=ticker)
        assert fund.cnpj is None
        assert fund.razao_social is None
        assert fund.fonte is None
    
    def test_invalid_razao_social(self):
        """Testa rejeição de razão social muito curta."""
        ticker = FundTicker("HGLG11")
        with pytest.raises(ValueError, match="pelo menos 3 caracteres"):
            Fund(ticker=ticker, razao_social="AB")

class TestPriceQuote:
    """Testes para a entidade PriceQuote."""
    
    def test_valid_price_quote(self):
        """Testa criação de cotação válida."""
        fund = Fund(ticker=FundTicker("HGLG11"))
        close_price = Money(Decimal("100.00"))
        date = datetime.now()
        
        quote = PriceQuote(
            fund=fund,
            date=date,
            close_price=close_price,
            volume=1000,
            source="test"
        )
        
        assert quote.fund == fund
        assert quote.close_price == close_price
        assert quote.volume == 1000
        assert quote.source == "test"
    
    def test_invalid_future_date(self):
        """Testa rejeição de data futura."""
        fund = Fund(ticker=FundTicker("HGLG11"))
        close_price = Money(Decimal("100.00"))
        future_date = datetime.now() + timedelta(days=1)
        
        with pytest.raises(ValueError, match="não pode ser futura"):
            PriceQuote(
                fund=fund,
                date=future_date,
                close_price=close_price
            )

class TestDividend:
    """Testes para a entidade Dividend."""
    
    def test_valid_dividend(self):
        """Testa criação de dividendo válido."""
        fund = Fund(ticker=FundTicker("HGLG11"))
        value_per_share = Money(Decimal("0.50"))
        ex_date = datetime.now()
        
        dividend = Dividend(
            fund=fund,
            ex_date=ex_date,
            value_per_share=value_per_share,
            dividend_type="dividend",
            source="test"
        )
        
        assert dividend.fund == fund
        assert dividend.value_per_share == value_per_share
        assert dividend.ex_date == ex_date
        assert dividend.dividend_type == "dividend"
    
    def test_invalid_future_ex_date(self):
        """Testa rejeição de data ex-dividendo futura."""
        fund = Fund(ticker=FundTicker("HGLG11"))
        value_per_share = Money(Decimal("0.50"))
        future_date = datetime.now() + timedelta(days=1)
        
        with pytest.raises(ValueError, match="não pode ser futura"):
            Dividend(
                fund=fund,
                ex_date=future_date,
                value_per_share=value_per_share
            )
    
    def test_invalid_payment_date_before_ex_date(self):
        """Testa rejeição de data de pagamento anterior à ex-dividendo."""
        fund = Fund(ticker=FundTicker("HGLG11"))
        value_per_share = Money(Decimal("0.50"))
        ex_date = datetime.now()
        payment_date = ex_date - timedelta(days=1)
        
        with pytest.raises(ValueError, match="deve ser posterior"):
            Dividend(
                fund=fund,
                ex_date=ex_date,
                value_per_share=value_per_share,
                payment_date=payment_date
            )

class TestNewsItem:
    """Testes para a entidade NewsItem."""
    
    def test_valid_news_item(self):
        """Testa criação de item de notícia válido."""
        title = "FII HGLG11 anuncia novo projeto"
        link = "https://example.com/news"
        published_at = datetime.now()
        source = "Test News"
        
        news = NewsItem(
            id="news_001",
            title=title,
            link=link,
            published_at=published_at,
            source=source
        )
        
        assert news.title == title
        assert news.link == link
        assert news.published_at == published_at
        assert news.source == source
        assert news.related_funds == []
    
    def test_invalid_empty_title(self):
        """Testa rejeição de título vazio."""
        with pytest.raises(ValueError, match="não pode estar vazio"):
            NewsItem(
                id="news_001",
                title="",
                link="https://example.com",
                published_at=datetime.now(),
                source="Test"
            )
    
    def test_invalid_future_published_date(self):
        """Testa rejeição de data de publicação futura."""
        future_date = datetime.now() + timedelta(days=1)
        
        with pytest.raises(ValueError, match="não pode ser futura"):
            NewsItem(
                id="news_001",
                title="Test News",
                link="https://example.com",
                published_at=future_date,
                source="Test"
            )

class TestFundMetrics:
    """Testes para a entidade FundMetrics."""
    
    def test_valid_fund_metrics(self):
        """Testa criação de métricas válidas."""
        fund = Fund(ticker=FundTicker("HGLG11"))
        date = datetime.now()
        
        metrics = FundMetrics(
            fund=fund,
            date=date,
            p_vp=PositiveFloat(1.05),
            dividend_yield=Percentage(8.5),
            vacancy_rate=Percentage(5.0),
            source="test"
        )
        
        assert metrics.fund == fund
        assert metrics.p_vp == PositiveFloat(1.05)
        assert metrics.dividend_yield == Percentage(8.5)
        assert metrics.vacancy_rate == Percentage(5.0)
    
    def test_invalid_negative_p_vp(self):
        """Testa rejeição de P/VP negativo."""
        fund = Fund(ticker=FundTicker("HGLG11"))
        date = datetime.now()
        
        # P/VP negativo
        with pytest.raises(ValueError, match="Valor deve ser positivo"):
            FundMetrics(
                fund=fund,
                date=date,
                p_vp=PositiveFloat(-1.0)  # Deve falhar na validação do PositiveFloat
            )
    
    def test_invalid_negative_dividend_yield(self):
        """Testa rejeição de dividend yield negativo."""
        fund = Fund(ticker=FundTicker("HGLG11"))
        date = datetime.now()
        
        with pytest.raises(ValueError, match="Percentual deve estar entre 0 e 100"):
            FundMetrics(
                fund=fund,
                date=date,
                dividend_yield=Percentage(-5.0)
            )
    
    def test_invalid_vacancy_rate_out_of_range(self):
        """Testa rejeição de taxa de vacância fora do intervalo."""
        fund = Fund(ticker=FundTicker("HGLG11"))
        date = datetime.now()
        
        # Taxa negativa
        with pytest.raises(ValueError, match="Percentual deve estar entre 0 e 100"):
            FundMetrics(
                fund=fund,
                date=date,
                vacancy_rate=Percentage(-5.0)
            )
        
        # Taxa acima de 100%
        with pytest.raises(ValueError, match="Percentual deve estar entre 0 e 100"):
            FundMetrics(
                fund=fund,
                date=date,
                vacancy_rate=Percentage(105.0)
            )
