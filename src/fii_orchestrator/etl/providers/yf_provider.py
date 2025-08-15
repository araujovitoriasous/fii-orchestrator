import time
from datetime import datetime
from typing import Iterable

import yfinance as yf
from fii_orchestrator.etl.schemas import PriceRecord, DividendRecord
from .base import MarketDataProvider


class YFinanceProvider(MarketDataProvider):
    def __init__(self, pause: float = 0.8):
        self.pause = pause

    @staticmethod
    def _yf_ticker(t: str) -> str:
        return f"{t}.SA"

    def fetch_prices(
        self, ticker: str, start: datetime, end: datetime
    ) -> Iterable[PriceRecord]:
        yf_t = self._yf_ticker(ticker)
        tk = yf.Ticker(yf_t)
        hist = tk.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
        for idx, row in hist.iterrows():
            close = row.get("Close")
            if close == close:  # not NaN
                volume = row.get("Volume")
                yield PriceRecord(
                    date=datetime(idx.year, idx.month, idx.day),
                    ticker=ticker,
                    close=float(close),
                    volume=float(volume) if volume == volume else None,
                    fonte="yfinance",
                )
        if self.pause:
            time.sleep(self.pause)

    def fetch_dividends(
        self, ticker: str, start: datetime, end: datetime
    ) -> Iterable[DividendRecord]:
        yf_t = self._yf_ticker(ticker)
        tk = yf.Ticker(yf_t)
        divs_pd = tk.dividends
        if divs_pd is not None and not divs_pd.empty:
            for idx, val in divs_pd.items():
                d_dt = datetime(idx.year, idx.month, idx.day)
                if d_dt >= start.replace(tzinfo=None) and d_dt <= end.replace(
                    tzinfo=None
                ):
                    yield DividendRecord(
                        ex_date=d_dt,
                        payment_date=None,
                        ticker=ticker,
                        value=float(val),
                        tipo="dividend",
                        fonte="yfinance",
                    )
        if self.pause:
            time.sleep(self.pause)
