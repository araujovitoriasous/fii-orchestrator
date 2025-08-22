"""
Entidades do domínio de Fundos Imobiliários.

Este módulo contém as entidades principais que representam
os conceitos de negócio do domínio de FIIs, incluindo
validações e regras de negócio.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Set

from fii_orchestrator.domain.value_objects import FundTicker, CNPJ, Money, Percentage, PositiveFloat

@dataclass
class Fund:
    """
    Entidade representando um Fundo Imobiliário.
    
    Um fundo é a entidade central do domínio, contendo
    informações básicas como ticker, CNPJ e razão social.
    Pode ter múltiplas cotações, dividendos e métricas
    associadas ao longo do tempo.
    
    Examples:
        >>> ticker = FundTicker("HGLG11")
        >>> cnpj = CNPJ("08.441.966/0001-53")
        >>> fund = Fund(
        ...     ticker=ticker,
        ...     cnpj=cnpj,
        ...     razao_social="CSHG Real Estate FII"
        ... )
        
    Attributes:
        ticker: Ticker único do fundo
        cnpj: CNPJ do fundo (opcional)
        razao_social: Razão social do fundo (opcional)
        fonte: Fonte dos dados (opcional)
        created_at: Data de criação do registro
        updated_at: Data da última atualização
        
    Invariants:
        - ticker deve ser único no sistema
        - razao_social deve ter pelo menos 3 caracteres se fornecida
        - created_at deve ser anterior ou igual a updated_at
    """
    ticker: FundTicker
    cnpj: Optional[CNPJ] = None
    razao_social: Optional[str] = None
    fonte: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validação pós-inicialização da entidade."""
        if self.razao_social is not None and len(self.razao_social.strip()) < 3:
            raise ValueError("Razão social deve ter pelo menos 3 caracteres")
        
        if self.created_at > self.updated_at:
            raise ValueError("Data de criação deve ser anterior à data de atualização")
    
    def update_razao_social(self, new_razao: str) -> None:
        """
        Atualiza a razão social do fundo.
        
        Args:
            new_razao: Nova razão social
            
        Raises:
            ValueError: Se a nova razão social for muito curta
        """
        if len(new_razao.strip()) < 3:
            raise ValueError("Razão social deve ter pelo menos 3 caracteres")
        
        self.razao_social = new_razao.strip()
        self.updated_at = datetime.now()
    
    def update_fonte(self, new_fonte: str) -> None:
        """
        Atualiza a fonte dos dados do fundo.
        
        Args:
            new_fonte: Nova fonte de dados
        """
        self.fonte = new_fonte
        self.updated_at = datetime.now()
    
    def __str__(self) -> str:
        return f"Fund({self.ticker})"
    
    def __repr__(self) -> str:
        return f"Fund(ticker={self.ticker}, cnpj={self.cnpj}, razao_social='{self.razao_social}')"

@dataclass
class PriceQuote:
    """
    Entidade representando uma cotação de preço de um fundo.
    
    Uma cotação registra o preço de fechamento e volume
    de negociação de um fundo em uma data específica.
    
    Examples:
        >>> fund = Fund(ticker=FundTicker("HGLG11"))
        >>> close_price = Money(100.50)
        >>> quote = PriceQuote(
        ...     fund=fund,
        ...     date=datetime.now(),
        ...     close_price=close_price,
        ...     volume=1000
        ... )
        
    Attributes:
        fund: Fundo ao qual a cotação pertence
        date: Data da cotação
        close_price: Preço de fechamento
        volume: Volume negociado (opcional)
        source: Fonte dos dados
        
    Invariants:
        - date não pode ser futura
        - close_price deve ser positivo
        - volume deve ser não-negativo se fornecido
    """
    fund: Fund
    date: datetime
    close_price: Money
    volume: Optional[int] = None
    source: str = "unknown"
    
    def __post_init__(self):
        """Validação pós-inicialização da entidade."""
        if self.date > datetime.now():
            raise ValueError("Data da cotação não pode ser futura")
        
        if self.volume is not None and self.volume < 0:
            raise ValueError("Volume não pode ser negativo")
    
    def get_return_from_previous(self, previous_quote: 'PriceQuote') -> Optional[Percentage]:
        """
        Calcula o retorno percentual em relação a uma cotação anterior.
        
        Args:
            previous_quote: Cotação anterior para comparação
            
        Returns:
            Retorno percentual ou None se não for possível calcular
            
        Raises:
            ValueError: Se as cotações não forem do mesmo fundo
        """
        if self.fund.ticker != previous_quote.fund.ticker:
            raise ValueError("Cotações devem ser do mesmo fundo")
        
        if self.date <= previous_quote.date:
            return None
        
        if previous_quote.close_price.is_zero():
            return None
        
        # Calcular retorno percentual
        price_change = self.close_price - previous_quote.close_price
        return_percentage = (price_change.amount / previous_quote.close_price.amount) * 100
        
        return Percentage(return_percentage)
    
    def __str__(self) -> str:
        return f"PriceQuote({self.fund.ticker}, {self.date.date()}, {self.close_price})"
    
    def __repr__(self) -> str:
        return f"PriceQuote(fund={self.fund}, date={self.date}, close_price={self.close_price}, volume={self.volume})"

@dataclass
class Dividend:
    """
    Entidade representando um dividendo distribuído por um fundo.
    
    Um dividendo registra o valor por cota distribuído
    em uma data específica (ex-dividendo).
    
    Examples:
        >>> fund = Fund(ticker=FundTicker("HGLG11"))
        >>> value_per_share = Money(0.50)
        >>> dividend = Dividend(
        ...     fund=fund,
        ...     ex_date=datetime.now(),
        ...     value_per_share=value_per_share
        ... )
        
    Attributes:
        fund: Fundo que distribuiu o dividendo
        ex_date: Data ex-dividendo
        payment_date: Data de pagamento (opcional)
        value_per_share: Valor por cota
        dividend_type: Tipo do dividendo
        source: Fonte dos dados
        
    Invariants:
        - ex_date não pode ser futura
        - payment_date deve ser posterior à ex_date se fornecida
        - value_per_share deve ser positivo
    """
    fund: Fund
    ex_date: datetime
    value_per_share: Money
    payment_date: Optional[datetime] = None
    dividend_type: str = "dividend"
    source: str = "unknown"
    
    def __post_init__(self):
        """Validação pós-inicialização da entidade."""
        if self.ex_date > datetime.now():
            raise ValueError("Data ex-dividendo não pode ser futura")
        
        if self.payment_date is not None and self.payment_date <= self.ex_date:
            raise ValueError("Data de pagamento deve ser posterior à data ex-dividendo")
    
    def is_paid(self) -> bool:
        """
        Verifica se o dividendo já foi pago.
        
        Returns:
            True se o dividendo foi pago, False caso contrário
        """
        return self.payment_date is not None and self.payment_date <= datetime.now()
    
    def get_days_to_payment(self) -> Optional[int]:
        """
        Calcula o número de dias até o pagamento.
        
        Returns:
            Número de dias até o pagamento ou None se já foi pago
        """
        if self.payment_date is None:
            return None
        
        if self.is_paid():
            return 0
        
        delta = self.payment_date - datetime.now()
        return delta.days
    
    def __str__(self) -> str:
        return f"Dividend({self.fund.ticker}, {self.ex_date.date()}, {self.value_per_share})"
    
    def __repr__(self) -> str:
        return f"Dividend(fund={self.fund}, ex_date={self.ex_date}, value_per_share={self.value_per_share})"

@dataclass
class NewsItem:
    """
    Entidade representando uma notícia relacionada a fundos.
    
    Uma notícia pode estar relacionada a um ou mais fundos
    e contém informações como título, link e data de publicação.
    
    Examples:
        >>> news = NewsItem(
        ...     id="news_001",
        ...     title="FII HGLG11 anuncia novo projeto",
        ...     link="https://example.com/news",
        ...     published_at=datetime.now()
        ... )
        
    Attributes:
        id: Identificador único da notícia
        title: Título da notícia
        link: Link para a notícia completa
        published_at: Data de publicação
        source: Fonte da notícia
        summary: Resumo da notícia (opcional)
        related_funds: Lista de fundos relacionados (opcional)
        
    Invariants:
        - title não pode estar vazio
        - published_at não pode ser futura
        - id deve ser único no sistema
    """
    id: str
    title: str
    link: str
    published_at: datetime
    source: str
    summary: Optional[str] = None
    related_funds: List[Fund] = field(default_factory=list)
    
    def __post_init__(self):
        """Validação pós-inicialização da entidade."""
        if not self.title.strip():
            raise ValueError("Título da notícia não pode estar vazio")
        
        if self.published_at > datetime.now():
            raise ValueError("Data de publicação não pode ser futura")
    
    def add_related_fund(self, fund: Fund) -> None:
        """
        Adiciona um fundo relacionado à notícia.
        
        Args:
            fund: Fundo a ser relacionado
        """
        if fund not in self.related_funds:
            self.related_funds.append(fund)
    
    def remove_related_fund(self, fund: Fund) -> None:
        """
        Remove um fundo relacionado à notícia.
        
        Args:
            fund: Fundo a ser removido
        """
        if fund in self.related_funds:
            self.related_funds.remove(fund)
    
    def get_related_tickers(self) -> Set[str]:
        """
        Obtém os tickers dos fundos relacionados.
        
        Returns:
            Conjunto de tickers relacionados
        """
        return {str(fund.ticker) for fund in self.related_funds}
    
    def is_recent(self, days: int = 7) -> bool:
        """
        Verifica se a notícia é recente.
        
        Args:
            days: Número de dias para considerar recente
            
        Returns:
            True se a notícia for recente
        """
        delta = datetime.now() - self.published_at
        return delta.days <= days
    
    def __str__(self) -> str:
        return f"NewsItem({self.id}, {self.title[:50]}...)"
    
    def __repr__(self) -> str:
        return f"NewsItem(id='{self.id}', title='{self.title}', source='{self.source}')"

@dataclass
class FundMetrics:
    """
    Entidade representando métricas financeiras de um fundo.
    
    As métricas incluem indicadores como P/VP, dividend yield
    e taxa de vacância em uma data específica.
    
    Examples:
        >>> fund = Fund(ticker=FundTicker("HGLG11"))
        >>> metrics = FundMetrics(
        ...     fund=fund,
        ...     date=datetime.now(),
        ...     p_vp=Percentage(1.05),
        ...     dividend_yield=Percentage(8.5)
        ... )
        
    Attributes:
        fund: Fundo ao qual as métricas pertencem
        date: Data das métricas
        p_vp: Preço sobre valor patrimonial (opcional)
        dividend_yield: Dividend yield (opcional)
        vacancy_rate: Taxa de vacância (opcional)
        source: Fonte dos dados
        
    Invariants:
        - date não pode ser futura
        - p_vp deve ser positivo se fornecido
        - dividend_yield deve estar entre 0 e 100 se fornecido
        - vacancy_rate deve estar entre 0 e 100 se fornecido
    """
    fund: Fund
    date: datetime
    p_vp: Optional[PositiveFloat] = None
    dividend_yield: Optional[Percentage] = None
    vacancy_rate: Optional[Percentage] = None
    source: str = "unknown"
    
    def __post_init__(self):
        """Validação pós-inicialização da entidade."""
        if self.date > datetime.now():
            raise ValueError("Data das métricas não pode ser futura")
        
        # Validar P/VP se fornecido
        if self.p_vp is not None and self.p_vp.value <= 0:
            raise ValueError("P/VP deve ser positivo")
        
        # Validar dividend yield se fornecido
        if self.dividend_yield is not None and self.dividend_yield.value < 0:
            raise ValueError("Dividend Yield não pode ser negativo")
        
        # Validar taxa de vacância se fornecida
        if self.vacancy_rate is not None and (self.vacancy_rate.value < 0 or self.vacancy_rate.value > 100):
            raise ValueError("Taxa de vacância deve estar entre 0 e 100%")
    
    def has_complete_metrics(self) -> bool:
        """
        Verifica se todas as métricas estão disponíveis.
        
        Returns:
            True se todas as métricas estiverem disponíveis
        """
        return all([
            self.p_vp is not None,
            self.dividend_yield is not None,
            self.vacancy_rate is not None
        ])
    
    def get_quality_score(self) -> float:
        """
        Calcula um score de qualidade das métricas.
        
        Returns:
            Score entre 0.0 e 1.0 baseado na completude
        """
        available_metrics = sum([
            self.p_vp is not None,
            self.dividend_yield is not None,
            self.vacancy_rate is not None
        ])
        
        return available_metrics / 3.0
    
    def __str__(self) -> str:
        return f"FundMetrics({self.fund.ticker}, {self.date.date()})"
    
    def __repr__(self) -> str:
        return f"FundMetrics(fund={self.fund}, date={self.date}, p_vp={self.p_vp}, dividend_yield={self.dividend_yield})"
