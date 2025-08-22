"""
Implementações concretas dos repositórios.
Implementa o acesso a dados e persistência.
"""

import polars as pl
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import json

from fii_orchestrator.domain.entities import Fund, PriceQuote, Dividend, NewsItem
from fii_orchestrator.application.use_cases import (
    FundRepository, PriceRepository, DividendRepository, NewsRepository
)

class ParquetFundRepository(FundRepository):
    """Implementação de repositório de fundos usando Parquet."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path / "reference"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.funds_file = self.base_path / "funds.parquet"
    
    def save(self, fund: Fund) -> None:
        """Salva um fundo."""
        # Converter entidade para dict
        fund_data = {
            "ticker": str(fund.ticker),
            "cnpj": str(fund.cnpj) if fund.cnpj else None,
            "razao_social": fund.razao_social,
            "fonte": fund.fonte
        }
        
        # Ler dados existentes ou criar novo
        if self.funds_file.exists():
            df = pl.read_parquet(self.funds_file)
            # Atualizar se já existe
            df = df.filter(pl.col("ticker") != fund_data["ticker"])
            df = pl.concat([df, pl.DataFrame([fund_data])])
        else:
            df = pl.DataFrame([fund_data])
        
        df.write_parquet(self.funds_file)
    
    def get_by_ticker(self, ticker: str) -> Optional[Fund]:
        """Busca fundo por ticker."""
        if not self.funds_file.exists():
            return None
        
        df = pl.read_parquet(self.funds_file)
        fund_row = df.filter(pl.col("ticker") == ticker)
        
        if fund_row.is_empty():
            return None
        
        row = fund_row.row(0)
        return Fund(
            ticker=row[0],
            cnpj=row[1] if row[1] else None,
            razao_social=row[2],
            fonte=row[3]
        )
    
    def get_all(self) -> List[Fund]:
        """Retorna todos os fundos."""
        if not self.funds_file.exists():
            return []
        
        df = pl.read_parquet(self.funds_file)
        funds = []
        
        for row in df.iter_rows():
            fund = Fund(
                ticker=row[0],
                cnpj=row[1] if row[1] else None,
                razao_social=row[2],
                fonte=row[3]
            )
            funds.append(fund)
        
        return funds
    
    def exists(self, ticker: str) -> bool:
        """Verifica se fundo existe."""
        return self.get_by_ticker(ticker) is not None

class ParquetPriceRepository(PriceRepository):
    """Implementação de repositório de preços usando Parquet."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path / "prices"
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, price: PriceQuote) -> None:
        """Salva uma cotação."""
        # Criar estrutura de diretórios particionada
        year = price.date.year
        month = price.date.month
        ticker = str(price.fund.ticker)
        
        price_dir = self.base_path / f"ticker={ticker}" / f"year={year}" / f"month={month:02d}"
        price_dir.mkdir(parents=True, exist_ok=True)
        
        # Converter para dict
        price_data = {
            "date": price.date,
            "ticker": str(price.fund.ticker),
            "close": float(price.close_price.amount),
            "volume": price.volume,
            "source": price.source
        }
        
        # Salvar em arquivo particionado
        filename = f"prices_{year}{month:02d}.parquet"
        filepath = price_dir / filename
        
        if filepath.exists():
            # Ler e concatenar
            df_existing = pl.read_parquet(filepath)
            df_new = pl.DataFrame([price_data])
            df_combined = pl.concat([df_existing, df_new])
        else:
            df_combined = pl.DataFrame([price_data])
        
        df_combined.write_parquet(filepath)
    
    def get_by_fund_and_date_range(
        self, 
        fund_ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[PriceQuote]:
        """Busca cotações por fundo e período."""
        # Implementar busca em arquivos particionados
        # Por enquanto, retorna lista vazia
        return []
    
    def get_latest_by_fund(self, fund_ticker: str) -> Optional[PriceQuote]:
        """Busca cotação mais recente de um fundo."""
        # Implementar busca da cotação mais recente
        # Por enquanto, retorna None
        return None

class ParquetDividendRepository(DividendRepository):
    """Implementação de repositório de dividendos usando Parquet."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path / "dividends"
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, dividend: Dividend) -> None:
        """Salva um dividendo."""
        # Estrutura similar ao repositório de preços
        year = dividend.ex_date.year
        month = dividend.ex_date.month
        ticker = str(dividend.fund.ticker)
        
        dividend_dir = self.base_path / f"ticker={ticker}" / f"year={year}" / f"month={month:02d}"
        dividend_dir.mkdir(parents=True, exist_ok=True)
        
        dividend_data = {
            "ex_date": dividend.ex_date,
            "payment_date": dividend.payment_date,
            "ticker": str(dividend.fund.ticker),
            "value": float(dividend.value_per_share.amount),
            "tipo": dividend.dividend_type,
            "source": dividend.source
        }
        
        filename = f"dividends_{year}{month:02d}.parquet"
        filepath = dividend_dir / filename
        
        if filepath.exists():
            df_existing = pl.read_parquet(filepath)
            df_new = pl.DataFrame([dividend_data])
            df_combined = pl.concat([df_existing, df_new])
        else:
            df_combined = pl.DataFrame([dividend_data])
        
        df_combined.write_parquet(filepath)
    
    def get_by_fund_and_date_range(
        self, 
        fund_ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dividend]:
        """Busca dividendos por fundo e período."""
        # Implementar busca em arquivos particionados
        return []

class ParquetNewsRepository(NewsRepository):
    """Implementação de repositório de notícias usando Parquet."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path / "news"
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, news: NewsItem) -> None:
        """Salva uma notícia."""
        # Estrutura particionada por data
        year = news.published_at.year
        month = news.published_at.month
        day = news.published_at.day
        
        news_dir = self.base_path / f"year={year}" / f"month={month:02d}" / f"day={day:02d}"
        news_dir.mkdir(parents=True, exist_ok=True)
        
        news_data = {
            "id": news.id,
            "title": news.title,
            "link": news.link,
            "published_at": news.published_at,
            "source": news.source,
            "summary": news.summary,
            "related_funds": [str(f.ticker) for f in news.related_funds] if news.related_funds else []
        }
        
        filename = f"news_{year}{month:02d}{day:02d}.parquet"
        filepath = news_dir / filename
        
        if filepath.exists():
            df_existing = pl.read_parquet(filepath)
            df_new = pl.DataFrame([news_data])
            df_combined = pl.concat([df_existing, df_new])
        else:
            df_combined = pl.DataFrame([news_data])
        
        df_combined.write_parquet(filepath)
    
    def get_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[NewsItem]:
        """Busca notícias por período."""
        # Implementar busca em arquivos particionados
        return []
    
    def get_by_fund(self, fund_ticker: str) -> List[NewsItem]:
        """Busca notícias relacionadas a um fundo."""
        # Implementar busca por fundo
        return []
