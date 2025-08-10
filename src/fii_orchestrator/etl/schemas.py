from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal

class NewsItem(BaseModel):
    id: str
    title: str
    link: str
    published_at: datetime
    source: str
    summary: Optional[str] = None
    raw_tickers: list[str] = Field(default_factory=list)

class FundRef(BaseModel):
    ticker: str                   # ex: HGLG11
    cnpj: Optional[str] = None    # ex: 08.441.966/0001-53
    razao_social: Optional[str] = None
    fonte: Optional[str] = None   # ex: 'seed_csv', 'cvm', 'fnet', 'yfinance'

class PriceRecord(BaseModel):
    date: datetime                
    ticker: str
    close: float
    volume: Optional[float] = None
    fonte: Optional[str] = None

class DividendRecord(BaseModel):
    ex_date: datetime
    payment_date: Optional[datetime] = None
    ticker: str
    value: float                  # valor por cota
    tipo: Literal["dividend","income","other"] = "dividend"
    fonte: Optional[str] = None
