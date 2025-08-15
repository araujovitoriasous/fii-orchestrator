from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable

from fii_orchestrator.etl.schemas import PriceRecord, DividendRecord


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_prices(
        self, ticker: str, start: datetime, end: datetime
    ) -> Iterable[PriceRecord]:
        """Return price records for ticker between start and end."""

    @abstractmethod
    def fetch_dividends(
        self, ticker: str, start: datetime, end: datetime
    ) -> Iterable[DividendRecord]:
        """Return dividend records for ticker between start and end."""
