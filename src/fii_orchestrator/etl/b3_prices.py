from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import polars as pl
from loguru import logger

from fii_orchestrator.config import BRONZE

REF_FUNDS = BRONZE / "reference" / "funds.parquet"
PRICES_DIR = BRONZE / "prices"
DIVS_DIR = BRONZE / "dividends"
for d in (PRICES_DIR, DIVS_DIR):
    d.mkdir(parents=True, exist_ok=True)

PROVIDER = os.getenv("PROVIDER", "yf").lower()


def _load_tickers() -> list[str]:
    if not REF_FUNDS.exists():
        raise FileNotFoundError(
            f"Referencia não encontrada: {REF_FUNDS}. Rode reference_funds.py antes."
        )
    df = pl.read_parquet(REF_FUNDS)
    return df["ticker"].to_list()


def _partitioned_write(
    df: pl.DataFrame, base: Path, ticker: str, date_col: str, fname_prefix: str
):
    if df.is_empty():
        return
    # particiona por year/month
    df = df.with_columns(
        pl.col(date_col).dt.year().cast(pl.Utf8).alias("year"),
        pl.col(date_col).dt.month().cast(pl.Utf8).str.zfill(2).alias("month"),
    )
    for (year, month), sub in df.group_by(["year", "month"], maintain_order=True):
        outdir = base / f"ticker={ticker}" / f"year={year}" / f"month={month}"
        outdir.mkdir(parents=True, exist_ok=True)
        sub.drop(["year", "month"]).write_parquet(
            outdir / f"{fname_prefix}_{year}{month}.parquet"
        )


def _get_provider():
    if PROVIDER == "b3_vendor":
        from fii_orchestrator.etl.providers.b3_vendor import B3VendorProvider

        return B3VendorProvider()
    elif PROVIDER == "yf":
        from fii_orchestrator.etl.providers.yf_provider import YFinanceProvider

        return YFinanceProvider()
    else:
        raise ValueError(f"PROVIDER desconhecido: {PROVIDER}")


def run():
    tickers = _load_tickers()
    if not tickers:
        logger.warning("Nenhum ticker na referência.")
        return

    provider = _get_provider()
    start = datetime.fromisoformat(os.getenv("PRICES_START", "2018-01-01")).replace(
        tzinfo=timezone.utc
    )
    end_env = os.getenv("PRICES_END")
    end = (
        datetime.fromisoformat(end_env).replace(tzinfo=timezone.utc)
        if end_env
        else (datetime.now(timezone.utc) + timedelta(days=1))
    )

    for t in tickers:
        try:
            prices = list(provider.fetch_prices(t, start, end))
            if prices:
                dfp = pl.DataFrame([r.model_dump() for r in prices])
                _partitioned_write(dfp, PRICES_DIR, t, "date", "prices")

            divs = list(provider.fetch_dividends(t, start, end))
            if divs:
                dfd = pl.DataFrame([r.model_dump() for r in divs])
                _partitioned_write(dfd, DIVS_DIR, t, "ex_date", "dividends")

            logger.info(
                f"{t}: prices={len(prices)} dividends={len(divs)} via {PROVIDER}"
            )
        except Exception as e:
            logger.exception(f"Falha em {t}: {e}")


if __name__ == "__main__":
    run()
