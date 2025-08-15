import os
from pathlib import Path
import polars as pl
from loguru import logger
from dotenv import load_dotenv

from fii_orchestrator.config import BRONZE

load_dotenv()

REF_DIR = BRONZE / "reference"
REF_DIR.mkdir(parents=True, exist_ok=True)
FUNDS_PARQUET = REF_DIR / "funds.parquet"


def _from_env_tickers() -> pl.DataFrame:
    raw = os.getenv("FII_TICKERS", "").strip()
    if not raw:
        return pl.DataFrame({"ticker": []})
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    return pl.DataFrame({"ticker": tickers, "fonte": ["env"] * len(tickers)})


def _from_seed_csv() -> pl.DataFrame:
    csv_path = os.getenv("FII_REFERENCE_CSV", "").strip()
    if not csv_path or not Path(csv_path).exists():
        return pl.DataFrame({"ticker": []})
    df = pl.read_csv(
        csv_path, dtypes={"ticker": pl.Utf8, "cnpj": pl.Utf8, "razao_social": pl.Utf8}
    )
    df = df.with_columns(
        pl.col("ticker").str.to_uppercase(),
        pl.lit("seed_csv").alias("fonte"),
    )
    # garantir colunas
    for col in ["cnpj", "razao_social"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    return df.select(["ticker", "cnpj", "razao_social", "fonte"])


def run():
    logger.info("Construindo referência FII <-> CNPJ")
    env_df = _from_env_tickers()
    seed_df = _from_seed_csv()

    # full outer e priorizar seed (tem mais dados)
    df = env_df.join(seed_df, on="ticker", how="full", coalesce=True)

    if df.is_empty():
        logger.warning("Nenhum FII encontrado em FII_TICKERS ou seed CSV.")
        return

    df = (
        df.with_columns(
            pl.when(pl.col("cnpj").is_not_null() | pl.col("razao_social").is_not_null())
            .then(1)
            .otherwise(0)
            .alias("score")
        )
        .sort(["ticker", "score"], descending=[False, True])
        .unique(subset=["ticker"], keep="first")
        .drop("score")
    )

    df.write_parquet(FUNDS_PARQUET)
    logger.info(f"Referencia salva em {FUNDS_PARQUET} ({df.height} FIIs)")


if __name__ == "__main__":
    run()
