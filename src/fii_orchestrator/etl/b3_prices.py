from __future__ import annotations
import time
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl
from loguru import logger
import yfinance as yf
from datetime import timezone

from fii_orchestrator.config import BRONZE
from fii_orchestrator.etl.schemas import PriceRecord, DividendRecord

REF_FUNDS = BRONZE / "reference" / "funds.parquet"
PRICES_DIR = BRONZE / "prices"
DIVS_DIR = BRONZE / "dividends"
for d in (PRICES_DIR, DIVS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _load_tickers() -> list[str]:
    if not REF_FUNDS.exists():
        raise FileNotFoundError(f"Referencia não encontrada: {REF_FUNDS}. Rode reference_funds.py antes.")
    df = pl.read_parquet(REF_FUNDS)
    return df["ticker"].to_list()

def _yf_ticker(t: str) -> str:
    return f"{t}.SA"  # ajuste simples para B3 no yfinance

def _partitioned_write(df: pl.DataFrame, base: Path, ticker: str, date_col: str, fname_prefix: str):
    if df.is_empty():
        return
    # particiona por year/month
    df = df.with_columns(
        pl.col(date_col).dt.year().cast(pl.Utf8).alias("year"),
        pl.col(date_col).dt.month().cast(pl.Utf8).str.zfill(2).alias("month")
    )
    for (year, month), sub in df.group_by(["year", "month"], maintain_order=True):
        outdir = base / f"ticker={ticker}" / f"year={year}" / f"month={month}"
        outdir.mkdir(parents=True, exist_ok=True)
        sub.drop(["year", "month"]).write_parquet(outdir / f"{fname_prefix}_{year}{month}.parquet")

def fetch_prices_and_dividends(ticker: str, start: str = "2018-01-01", end: str | None = None):
    yf_t = _yf_ticker(ticker)
    end = end or (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"{ticker} -> {yf_t} | {start}..{end}")

    tk = yf.Ticker(yf_t)
    hist = tk.history(start=start, end=end, auto_adjust=False)  # DataFrame pandas

    prices: list[PriceRecord] = []
    if not hist.empty:
        # prefer close e volume
        for idx, row in hist.iterrows():
            if "Close" in row and not pl.Series([row["Close"]]).is_null().any():
                prices.append(
                    PriceRecord(
                        date=datetime(idx.year, idx.month, idx.day),
                        ticker=ticker,
                        close=float(row["Close"]),
                        volume=float(row["Volume"]) if "Volume" in row and row["Volume"] == row["Volume"] else None,
                        fonte="yfinance",
                    )
                )

    # dividends: yfinance traz série por data (valor por cota)
    divs_pd = tk.dividends
    dividends: list[DividendRecord] = []
    if divs_pd is not None and not divs_pd.empty:
        for idx, val in divs_pd.items():
            dividends.append(
                DividendRecord(
                    ex_date=datetime(idx.year, idx.month, idx.day),
                    payment_date=None,
                    ticker=ticker,
                    value=float(val),
                    tipo="dividend",
                    fonte="yfinance",
                )
            )

    return prices, dividends

def run():
    tickers = _load_tickers()
    if not tickers:
        logger.warning("Nenhum ticker na referência.")
        return

    for i, t in enumerate(tickers, 1):
        try:
            prices, dividends = fetch_prices_and_dividends(t)
            # salvar preços
            if prices:
                dfp = pl.DataFrame([p.model_dump() for p in prices])
                _partitioned_write(dfp, PRICES_DIR, t, "date", "prices")
            # salvar proventos
            if dividends:
                dfd = pl.DataFrame([d.model_dump() for d in dividends])
                _partitioned_write(dfd, DIVS_DIR, t, "ex_date", "dividends")
            logger.info(f"{t}: prices={len(prices)} dividends={len(dividends)}")
            time.sleep(0.8)  # educado com a API
        except Exception as e:
            logger.exception(f"Falha em {t}: {e}")

if __name__ == "__main__":
    run()
