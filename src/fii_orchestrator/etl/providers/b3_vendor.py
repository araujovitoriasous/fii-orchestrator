from __future__ import annotations
import os
import json
from datetime import datetime
from pathlib import Path
import polars as pl
from loguru import logger
from dotenv import load_dotenv
from fii_orchestrator.etl.schemas import DividendRecord, PriceRecord

load_dotenv()

# Diretório com arquivos CSV “oficiais” já baixados (SFTP, cron, etc.)
B3_VENDOR_DIR = Path(os.getenv("B3_VENDOR_DIR", "./vendor/b3")).resolve()
if not B3_VENDOR_DIR.exists():
    logger.warning(f"B3 vendor directory {B3_VENDOR_DIR} not found")

# Mapas de colunas (configuráveis no .env para não re-compilar)
# Ex.: B3_PRICE_COLMAP='{"date":"DATA","ticker":"TICKER","close":"FECHAMENTO","volume":"VOLUME"}'
PRICE_COLMAP = json.loads(os.getenv("B3_PRICE_COLMAP", "{}"))
DIV_COLMAP = json.loads(os.getenv("B3_DIV_COLMAP", "{}"))


def _read_price_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, ignore_errors=True)
    # aplica renomeações do COLMAP (se houver)
    if PRICE_COLMAP:
        df = df.rename(
            {k: v for k, v in PRICE_COLMAP.items() if k in df.columns}
            | {v: k for k, v in PRICE_COLMAP.items() if v in df.columns}
        )
    return df


def _read_div_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, ignore_errors=True)
    if DIV_COLMAP:
        df = df.rename(
            {k: v for k, v in DIV_COLMAP.items() if k in df.columns}
            | {v: k for k, v in DIV_COLMAP.items() if v in df.columns}
        )
    return df


class B3VendorProvider:
    """
    Lê arquivos CSV "oficiais" já sincronizados localmente (SFTP/REST->dump),
    normaliza para PriceRecord/DividendRecord.
    """

    def __init__(self):
        self.price_glob = os.getenv("B3_PRICE_GLOB", "prices_*.csv")
        self.div_glob = os.getenv("B3_DIV_GLOB", "dividends_*.csv")

    def _iter_price_files(self):
        files = sorted(B3_VENDOR_DIR.glob(self.price_glob))
        if not files:
            logger.warning(
                f"No price files matching {self.price_glob} under {B3_VENDOR_DIR}"
            )
        yield from files

    def _iter_div_files(self):
        files = sorted(B3_VENDOR_DIR.glob(self.div_glob))
        if not files:
            logger.warning(
                f"No dividend files matching {self.div_glob} under {B3_VENDOR_DIR}"
            )
        yield from files

    def fetch_prices(self, ticker: str, start: datetime, end: datetime):
        ticker = ticker.upper()
        for csv_path in self._iter_price_files():
            df = _read_price_csv(csv_path)
            # Esperado: colunas padronizadas pós-map: ['date','ticker','close','volume', ...]
            missing = [c for c in ["date", "ticker", "close"] if c not in df.columns]
            if missing:
                logger.warning(f"{csv_path} sem colunas {missing}, pulando.")
                continue

            # filtros de ticker e janela
            out = (
                df.filter((pl.col("ticker").str.to_uppercase() == ticker))
                .with_columns(
                    pl.col("date").str.strptime(pl.Date, strict=False).cast(pl.Datetime)
                )
                .filter(
                    (pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end))
                )
                .select(
                    [
                        "date",
                        "ticker",
                        "close",
                        pl.col("volume").fill_null(0).alias("volume"),
                    ]
                )
            )

            if out.height == 0:
                logger.debug(
                    f"{csv_path} sem dados para {ticker} entre {start.date()} e {end.date()}"
                )
                continue

            for r in out.iter_rows(named=True):
                yield PriceRecord(
                    date=r["date"].to_pydatetime(),
                    ticker=r["ticker"],
                    close=float(r["close"]),
                    volume=float(r["volume"]) if r["volume"] is not None else None,
                    fonte="b3_vendor",
                )

    def fetch_dividends(self, ticker: str, start: datetime, end: datetime):
        ticker = ticker.upper()
        for csv_path in self._iter_div_files():
            df = _read_div_csv(csv_path)
            # Esperado: ['ex_date','payment_date','ticker','value','tipo'?]
            # Se vier como 'event_date' e 'amount', mapeie no .env com B3_DIV_COLMAP
            # Ex.: {"ex_date":"DATA_EX","payment_date":"PAGAMENTO","ticker":"TICKER","value":"VALOR"}
            if (
                "ex_date" not in df.columns
                or "ticker" not in df.columns
                or "value" not in df.columns
            ):
                logger.warning(f"{csv_path} faltam colunas essenciais, pulando.")
                continue

            out = (
                df.filter((pl.col("ticker").str.to_uppercase() == ticker))
                .with_columns(
                    pl.col("ex_date")
                    .str.strptime(pl.Date, strict=False)
                    .cast(pl.Datetime),
                    pl.when(pl.col("payment_date").is_not_null())
                    .then(
                        pl.col("payment_date")
                        .str.strptime(pl.Date, strict=False)
                        .cast(pl.Datetime)
                    )
                    .otherwise(None)
                    .alias("payment_date"),
                )
                .filter(
                    (pl.col("ex_date") >= pl.lit(start))
                    & (pl.col("ex_date") <= pl.lit(end))
                )
                .select(["ex_date", "payment_date", "ticker", "value"])
            )

            if out.height == 0:
                logger.debug(
                    f"{csv_path} sem dados para {ticker} entre {start.date()} e {end.date()}"
                )
                continue

            for r in out.iter_rows(named=True):
                yield DividendRecord(
                    ex_date=r["ex_date"].to_pydatetime(),
                    payment_date=(
                        r["payment_date"].to_pydatetime() if r["payment_date"] else None
                    ),
                    ticker=r["ticker"],
                    value=float(r["value"]),
                    tipo="dividend",
                    fonte="b3_vendor",
                )
