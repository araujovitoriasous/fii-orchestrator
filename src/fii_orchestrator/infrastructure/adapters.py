"""
Adaptadores para APIs externas.
Implementa as interfaces de provedores de dados.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import yfinance as yf
import feedparser
import requests
import pandas as pd
from loguru import logger
import re

from fii_orchestrator.domain.entities import (
    Fund, PriceQuote, Dividend, NewsItem, FundTicker, Money, CNPJ
)
from fii_orchestrator.application.use_cases import DataProvider
from fii_orchestrator.utils.retry import retry_with_backoff
from fii_orchestrator.utils.text import detect_tickers

class YahooFinanceAdapter(DataProvider):
    """Adaptador para Yahoo Finance (proxy para dados da B3)."""
    
    def __init__(self, rate_limit: float = 0.8):
        self.rate_limit = rate_limit
        self._last_request = 0
    
    def _rate_limit_wait(self):
        """Implementa rate limiting para respeitar limites da API."""
        now = time.time()
        time_since_last = now - self._last_request
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self._last_request = time.time()
    
    def _yf_ticker(self, ticker: str) -> str:
        """Converte ticker para formato Yahoo Finance."""
        return f"{ticker}.SA"
    
    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def fetch_prices(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[PriceQuote]:
        """Busca preços de um FII."""
        self._rate_limit_wait()
        
        yf_ticker = self._yf_ticker(ticker)
        logger.debug(f"Buscando preços para {ticker} ({yf_ticker})")
        
        # Criar fundo temporário para a cotação
        fund = Fund(ticker=FundTicker(ticker))
        
        try:
            tk = yf.Ticker(yf_ticker)
            hist = tk.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=False
            )
            
            prices = []
            for idx, row in hist.iterrows():
                if "Close" in row and not pd.isna(row["Close"]):
                    close_price = Money(amount=row["Close"])
                    volume = int(row["Volume"]) if "Volume" in row and not pd.isna(row["Volume"]) else None
                    
                    price = PriceQuote(
                        fund=fund,
                        date=datetime(idx.year, idx.month, idx.day),
                        close_price=close_price,
                        volume=volume,
                        source="yahoo_finance"
                    )
                    prices.append(price)
            
            logger.info(f"Coletados {len(prices)} preços para {ticker}")
            return prices
            
        except Exception as e:
            logger.error(f"Erro ao buscar preços para {ticker}: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def fetch_dividends(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dividend]:
        """Busca dividendos de um FII."""
        self._rate_limit_wait()
        
        yf_ticker = self._yf_ticker(ticker)
        logger.debug(f"Buscando dividendos para {ticker} ({yf_ticker})")
        
        # Criar fundo temporário para o dividendo
        fund = Fund(ticker=FundTicker(ticker))
        
        try:
            tk = yf.Ticker(yf_ticker)
            divs_pd = tk.dividends
            
            dividends = []
            if divs_pd is not None and not divs_pd.empty:
                for idx, val in divs_pd.items():
                    # Filtrar por período
                    div_date = datetime(idx.year, idx.month, idx.day)
                    if start_date <= div_date <= end_date:
                        value_per_share = Money(amount=val)
                        
                        dividend = Dividend(
                            fund=fund,
                            ex_date=div_date,
                            value_per_share=value_per_share,
                            dividend_type="dividend",
                            source="yahoo_finance"
                        )
                        dividends.append(dividend)
            
            logger.info(f"Coletados {len(dividends)} dividendos para {ticker}")
            return dividends
            
        except Exception as e:
            logger.error(f"Erro ao buscar dividendos para {ticker}: {e}")
            raise
    
    def fetch_news(self, sources: List[str]) -> List[NewsItem]:
        """Yahoo Finance não fornece notícias via API."""
        return []

class RSSFeedAdapter(DataProvider):
    """Adaptador para feeds RSS de notícias."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def fetch_prices(self, ticker: str, start_date: datetime, end_date: datetime) -> List[PriceQuote]:
        """RSS não fornece dados de preços."""
        return []
    
    def fetch_dividends(self, ticker: str, start_date: datetime, end_date: datetime) -> List[Dividend]:
        """RSS não fornece dados de dividendos."""
        return []
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def fetch_news(self, sources: List[str]) -> List[NewsItem]:
        """Busca notícias de feeds RSS."""
        logger.info(f"Coletando notícias de {len(sources)} fontes RSS")
        
        all_news = []
        
        for source in sources:
            try:
                logger.debug(f"Processando fonte: {source}")
                feed = feedparser.parse(source)
                
                for entry in feed.entries:
                    # Extrair dados da entrada
                    title = getattr(entry, "title", "") or ""
                    link = getattr(entry, "link", "") or ""
                    summary = getattr(entry, "summary", "") or ""
                    
                    # Determinar data de publicação
                    if getattr(entry, "published_parsed", None):
                        pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    else:
                        pub_date = datetime.now(timezone.utc)
                    
                    # Detectar tickers relacionados
                    tickers_in_title = detect_tickers(title)
                    tickers_in_summary = detect_tickers(summary)
                    all_tickers = list(set(tickers_in_title + tickers_in_summary))
                    
                    # Criar entidades de fundo para tickers relacionados
                    related_funds = []
                    for ticker in all_tickers:
                        try:
                            fund = Fund(ticker=FundTicker(ticker))
                            related_funds.append(fund)
                        except ValueError:
                            logger.warning(f"Ticker inválido detectado: {ticker}")
                    
                    # Criar item de notícia
                    news_item = NewsItem(
                        id=getattr(entry, "id", link or title),
                        title=title,
                        link=link,
                        published_at=pub_date,
                        source=feed.feed.get("title", source),
                        summary=summary,
                        related_funds=related_funds
                    )
                    
                    all_news.append(news_item)
                
                logger.info(f"Fonte {source}: {len(feed.entries)} notícias processadas")
                
            except Exception as e:
                logger.error(f"Erro ao processar fonte RSS {source}: {e}")
                continue
        
        logger.info(f"Total de notícias coletadas: {len(all_news)}")
        return all_news

class CVMAPIAdapter(DataProvider):
    """Adaptador para APIs da CVM."""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def fetch_prices(self, ticker: str, start_date: datetime, end_date: datetime) -> List[PriceQuote]:
        """CVM não fornece dados de preços em tempo real."""
        return []
    
    def fetch_dividends(self, ticker: str, start_date: datetime, end_date: datetime) -> List[Dividend]:
        """CVM não fornece dados de dividendos em tempo real."""
        return []
    
    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def fetch_fund_reports(self, ticker: str) -> dict:
        """Busca relatórios de um fundo específico."""
        logger.info(f"Buscando relatórios CVM para {ticker}")
        
        # URLs para buscar relatórios
        search_urls = [
            f"https://www.gov.br/cvm/pt-br/assuntos/fundos-de-investimento/fundos/{ticker}",
            f"https://www.fundinfo.com/fund/{ticker}",
            f"https://www.fundsexplorer.com.br/funds/{ticker}"
        ]
        
        reports_data = {
            "ticker": ticker,
            "collection_date": datetime.now().isoformat(),
            "reports": [],
            "indicators": {}
        }
        
        for url in search_urls:
            try:
                logger.debug(f"Tentando coletar de: {url}")
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Extrair informações do HTML
                html_content = response.text
                extracted_info = self._extract_fund_info(html_content)
                
                if extracted_info:
                    reports_data["reports"].append({
                        "source_url": url,
                        "extracted_info": extracted_info,
                        "status": "success"
                    })
                    
                    # Consolidar indicadores
                    if "indicators" in extracted_info:
                        for key, value in extracted_info["indicators"].items():
                            if key not in reports_data["indicators"]:
                                reports_data["indicators"][key] = value
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Erro ao coletar de {url}: {e}")
                reports_data["reports"].append({
                    "source_url": url,
                    "status": "error",
                    "error": str(e)
                })
                continue
        
        return reports_data
    
    def _extract_fund_info(self, html_content: str) -> dict:
        """Extrai informações de fundo do HTML."""
        info = {}
        
        # Detectar CNPJ
        cnpj_pattern = r'(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})'
        cnpj_match = re.search(cnpj_pattern, html_content)
        if cnpj_match:
            cnpj_str = cnpj_match.group(1)
            try:
                info['cnpj'] = str(CNPJ(cnpj_str))
            except ValueError:
                logger.warning(f"CNPJ inválido encontrado: {cnpj_str}")
        
        # Detectar indicadores financeiros
        indicators = {}
        
        # P/VP
        pvp_patterns = [
            r'P/VP[:\s]*([\d,]+\.?\d*)',
            r'Preço/Valor Patrimonial[:\s]*([\d,]+\.?\d*)'
        ]
        for pattern in pvp_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                try:
                    pvp = float(match.group(1).replace(',', '.'))
                    indicators['p_vp'] = pvp
                    break
                except ValueError:
                    continue
        
        # Dividend Yield
        dy_patterns = [
            r'Dividend Yield[:\s]*([\d,]+\.?\d*)%?',
            r'DY[:\s]*([\d,]+\.?\d*)%?'
        ]
        for pattern in dy_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                try:
                    dy = float(match.group(1).replace(',', '.'))
                    indicators['dividend_yield'] = dy
                    break
                except ValueError:
                    continue
        
        if indicators:
            info['indicators'] = indicators
        
        return info
    
    def fetch_news(self, sources: List[str]) -> List[NewsItem]:
        """CVM não fornece notícias via RSS."""
        return []

class CSVFileAdapter(DataProvider):
    """Adaptador para arquivos CSV locais."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
    
    def fetch_prices(self, ticker: str, start_date: datetime, end_date: datetime) -> List[PriceQuote]:
        """Busca preços de arquivo CSV local."""
        # Implementar leitura de CSV local
        return []
    
    def fetch_dividends(self, ticker: str, start_date: datetime, end_date: datetime) -> List[Dividend]:
        """Busca dividendos de arquivo CSV local."""
        # Implementar leitura de CSV local
        return []
    
    def fetch_news(self, sources: List[str]) -> List[NewsItem]:
        """CSV não fornece notícias."""
        return []
    
    def load_funds_reference(self) -> List[Fund]:
        """Carrega referência de fundos de CSV."""
        csv_path = self.base_path / "funds_reference.csv"
        if not csv_path.exists():
            return []
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            funds = []
            for _, row in df.iterrows():
                try:
                    ticker = FundTicker(row['ticker'])
                    cnpj = CNPJ(row['cnpj']) if pd.notna(row['cnpj']) else None
                    
                    fund = Fund(
                        ticker=ticker,
                        cnpj=cnpj,
                        razao_social=row['razao_social'] if pd.notna(row['razao_social']) else None,
                        fonte="csv_local"
                    )
                    funds.append(fund)
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"Erro ao processar linha do CSV: {e}")
                    continue
            
            logger.info(f"Carregados {len(funds)} fundos do CSV")
            return funds
            
        except Exception as e:
            logger.error(f"Erro ao carregar CSV de fundos: {e}")
            return []

# Factory para criar adaptadores
class DataProviderFactory:
    """Factory para criar provedores de dados."""
    
    @staticmethod
    def create_yahoo_finance(rate_limit: float = 0.8) -> YahooFinanceAdapter:
        """Cria adaptador para Yahoo Finance."""
        return YahooFinanceAdapter(rate_limit=rate_limit)
    
    @staticmethod
    def create_rss_feed(timeout: int = 30) -> RSSFeedAdapter:
        """Cria adaptador para feeds RSS."""
        return RSSFeedAdapter(timeout=timeout)
    
    @staticmethod
    def create_cvm_api(timeout: int = 30, max_retries: int = 3) -> CVMAPIAdapter:
        """Cria adaptador para API da CVM."""
        return CVMAPIAdapter(timeout=timeout, max_retries=max_retries)
    
    @staticmethod
    def create_csv_file(base_path: str) -> CSVFileAdapter:
        """Cria adaptador para arquivos CSV."""
        return CSVFileAdapter(base_path=base_path)
